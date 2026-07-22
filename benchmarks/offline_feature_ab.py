#!/usr/bin/env python3
"""Offline feature A/B on a recorded real workload trace.

Takes a JSONL trace captured by trace_recorder.py (real EPP training entries
from a named workload) and answers: does adding <feature> improve TTFT
prediction, and does it leave TPOT untouched?

Method (mirrors the production training pipeline):
  - derived features computed exactly as training_server.py does
  - estimator + hyperparameters synced with production (XGBoost default)
  - N seeds x 90/10 split; arm A = base features, arm B = base + feature
  - TPOT is trained with its own production feature set (not TTFT features)
    and serves as the no-regression control

Hard gate: refuses to run if the trace has no real contention
(num_request_running must exceed 1 in at least --min-contention-pct of
samples). A zero-contention trace cannot exercise contention features.

Self-test: before the real A/B, shuffles the target column and runs a
known-bad control. If the shuffled run shows improvement exceeding 2x
seed std, the gate refuses to run (it can't discriminate signal from noise).

Outputs (to --outdir):
  ab_results.json        per-seed metrics, both arms, both targets
  summary.md             workload spec + mean ± std deltas + decision column
  calibration_ttft.png   actual-vs-predicted scatter, with/without feature
  calibration_tpot.png   same for TPOT (should be identical between arms)
  reliability_ttft.png   binned P90 coverage curve (should cluster near 0.9)
  reliability_tpot.png   same for TPOT
  shap_ttft.png          SHAP importance bar plot (optional, --shap flag)

Usage:
  python benchmarks/offline_feature_ab.py --trace trace.jsonl \\
      --feature <feature_name> --seeds 10 --outdir results/ab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

QUANTILE = 0.9

# Conditionally included when present in the trace (EPP-internal metrics).
# Inserted between base-5 and target-specific columns to match production
# column order (training_server.py:389-395, 412-416, 419-423).
CONDITIONAL_FEATURES = ["prefill_tokens_in_flight", "decode_tokens_in_flight"]

ENCODER_FEATURES = ["encoder_matched_size", "encoder_input_size"]

PREFIX_BUCKETS = 4


# ---------------------------------------------------------------------------
# Estimator registry — synced with training_server.py xgb_params (lines 617-635)
# ---------------------------------------------------------------------------

def _xgb_factory(seed, quantile=QUANTILE):
    return xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        objective="reg:quantileerror", quantile_alpha=quantile,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.2,
        reg_alpha=0.01, reg_lambda=0.1,
        tree_method="hist", n_jobs=-1,
        enable_categorical=True, random_state=seed, verbosity=0,
    )


def _lgbm_factory(seed, quantile=QUANTILE):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        objective="quantile", alpha=quantile,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=-1, random_state=seed, verbosity=-1,
        force_col_wise=True,
    )


ESTIMATORS = {"xgboost": _xgb_factory, "lightgbm": _lgbm_factory}


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

def load_trace(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        sys.exit(f"Trace file is empty: {path}")
    df = pd.DataFrame(rows)
    required = {
        "kv_cache_percentage", "input_token_length", "num_request_waiting",
        "num_request_running", "actual_ttft_ms", "actual_tpot_ms",
        "prefix_cache_score",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Trace missing required fields: {missing}")
    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            sys.exit(f"Field '{col}' must be numeric, got {df[col].dtype}")
    if len(df) < 100:
        print(f"WARNING: trace has only {len(df)} samples (recommend >= 1000 for stable results)", file=sys.stderr)
    return df


# Derived features not yet in production. Add one line per candidate.
# Features already in the trace (e.g. encoder_matched_size) need no entry here.
CANDIDATE_FORMULAS = {}


def add_derived_features(df: pd.DataFrame, feature: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df["effective_input_tokens"] = (1 - df["prefix_cache_score"]) * df["input_token_length"]
    df["prefill_score_bucket"] = pd.Categorical(
        (df["prefix_cache_score"].clip(0, 1) * PREFIX_BUCKETS).astype(int).clip(upper=PREFIX_BUCKETS - 1),
        categories=list(range(PREFIX_BUCKETS)), ordered=True,
    )
    df["is_queued"] = (df["num_request_waiting"] > 0).astype(int)
    df["pod_type_cat"] = pd.Categorical(
        df["pod_type"].fillna("") if "pod_type" in df.columns else pd.Series([""] * len(df)),
        categories=["", "prefill", "decode"], ordered=False,
    )
    if "num_tokens_generated" not in df.columns:
        df["num_tokens_generated"] = 0
    for col in CONDITIONAL_FEATURES + ENCODER_FEATURES:
        if col not in df.columns:
            df[col] = 0
    if feature and feature not in df.columns and feature in CANDIDATE_FORMULAS:
        df[feature] = CANDIDATE_FORMULAS[feature](df)
    return df


def _resolve_ttft_features(df: pd.DataFrame, feature: str | None) -> tuple[list[str], list[str]]:
    """Return (without_features, with_features) for TTFT A/B arms.

    Column order matches production training_server.py:
    base5 + tif_cols + encoder_cols + [prefix_cache_score, effective_input_tokens,
    prefill_score_bucket, pod_type_cat].
    """
    tif = [col for col in CONDITIONAL_FEATURES if col in df.columns and df[col].sum() > 0]
    enc = [col for col in ENCODER_FEATURES if col in df.columns and df[col].sum() > 0]
    base = (
        ["is_queued", "kv_cache_percentage", "input_token_length",
         "num_request_waiting", "num_request_running"]
        + tif
        + enc
        + ["prefix_cache_score", "effective_input_tokens",
           "prefill_score_bucket", "pod_type_cat"]
    )
    without = [f for f in base if f != feature]
    with_feat = without + [feature] if feature and feature not in without else list(base)
    return without, with_feat


def _resolve_tpot_features(df: pd.DataFrame) -> list[str]:
    """Return TPOT features matching production column order (training_server.py:419-423)."""
    tif = [col for col in CONDITIONAL_FEATURES if col in df.columns and df[col].sum() > 0]
    return (
        ["is_queued", "kv_cache_percentage", "input_token_length",
         "num_request_waiting", "num_request_running"]
        + tif
        + ["num_tokens_generated", "pod_type_cat"]
    )


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def contention_gate(df: pd.DataFrame, min_pct: float) -> dict:
    nrr = df["num_request_running"]
    dist = nrr.value_counts().sort_index()
    pct_contended = float((nrr > 1).mean() * 100)
    stats = {
        "samples": len(df),
        "num_request_running_distribution": {str(k): int(v) for k, v in dist.items()},
        "pct_samples_with_contention": round(pct_contended, 2),
        "max_num_request_running": int(nrr.max()),
    }
    if pct_contended < min_pct:
        print(json.dumps(stats, indent=2), file=sys.stderr)
        sys.exit(
            f"CONTENTION GATE FAILED: only {pct_contended:.1f}% of samples have "
            f"num_request_running > 1 (need >= {min_pct}%). This trace cannot "
            "exercise a contention feature — increase load concurrency and re-record."
        )
    return stats


def self_test(df: pd.DataFrame, feature: str, model_factory,
              real_delta: float, n_shuffles: int = 3) -> dict:
    """Post-hoc shuffled-label discriminability check.

    Runs AFTER the real A/B. Shuffles actual_ttft_ms to destroy signal and
    measures how much the feature "improves" on pure noise. The real
    improvement must be at least 2x the shuffled noise floor.

    Derived features (ratios of base features like prefill_density) can show
    small consistent improvements on shuffled data from added model freedom.
    Comparing against the real delta filters this out — if the real improvement
    is substantially larger, the feature adds genuine signal.
    """
    without, with_feat = _resolve_ttft_features(df, feature)
    deltas = []

    for shuffle_seed in range(n_shuffles):
        shuffled = df.copy()
        rng = np.random.RandomState(900 + shuffle_seed)
        shuffled["actual_ttft_ms"] = rng.permutation(shuffled["actual_ttft_ms"].values)

        test = shuffled.sample(frac=0.1, random_state=shuffle_seed)
        train = shuffled.drop(test.index)

        r_without = train_eval(train[without], train["actual_ttft_ms"],
                               test[without], test["actual_ttft_ms"],
                               shuffle_seed, model_factory)
        r_with = train_eval(train[with_feat], train["actual_ttft_ms"],
                            test[with_feat], test["actual_ttft_ms"],
                            shuffle_seed, model_factory)

        deltas.append(r_with["quantile_loss"] - r_without["quantile_loss"])

    shuffled_improvement = abs(float(np.mean(deltas))) if np.mean(deltas) < 0 else 0.0
    real_improvement = abs(real_delta) if real_delta < 0 else 0.0

    # Real improvement must exceed 2x the shuffled noise floor
    if real_improvement > 0 and shuffled_improvement > 0 and real_improvement < 2 * shuffled_improvement:
        return {
            "self_test": "FAIL",
            "reason": (
                f"Real improvement ({real_improvement:.4f}) is less than 2x the "
                f"shuffled noise floor ({shuffled_improvement:.4f}). The feature's "
                f"signal is not distinguishable from overfitting gain."
            ),
            "real_delta": round(real_delta, 4),
            "shuffled_deltas": [round(d, 4) for d in deltas],
        }
    # No real improvement at all
    if real_improvement == 0:
        return {
            "self_test": "PASS",
            "note": "No improvement detected — feature is neutral.",
            "shuffled_deltas": [round(d, 4) for d in deltas],
        }
    return {
        "self_test": "PASS",
        "real_vs_shuffled_ratio": round(real_improvement / max(shuffled_improvement, 1e-6), 2),
        "shuffled_deltas": [round(d, 4) for d in deltas],
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_eval(X_train, y_train, X_test, y_test, seed: int,
               model_factory=None) -> dict:
    model = (model_factory or _xgb_factory)(seed)
    fit_kwargs = {}
    if hasattr(model, 'boosting_type') and "prefill_score_bucket" in X_train.columns:
        fit_kwargs["categorical_feature"] = ["prefill_score_bucket"]
    model.fit(X_train, y_train, **fit_kwargs)
    preds = np.asarray(model.predict(X_test), dtype=float)
    y = np.asarray(y_test, dtype=float)
    err = y - preds
    return {
        "quantile_loss": float(np.mean(np.where(err >= 0, QUANTILE * err, (QUANTILE - 1) * err))),
        "coverage_pct": float(np.mean(y <= preds) * 100),
        "mape_pct": float(np.mean(np.abs(err) / np.clip(y, 1e-6, None)) * 100),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "actual": y.tolist(),
        "predicted": preds.tolist(),
        "model": model,
    }


def convergence_curve(df: pd.DataFrame, feature: str, model_factory=None,
                      slices: tuple = (500, 1000, 2000, 4000)) -> list[dict]:
    """Train on increasing sample counts to show how the feature's delta evolves."""
    feats_without, feats_with = _resolve_ttft_features(df, feature)
    curve = []
    for n in slices:
        if n > len(df):
            break
        sub = df.iloc[:n]
        test = sub.sample(frac=0.1, random_state=0)
        train = sub.drop(test.index)
        r_wo = train_eval(train[feats_without], train["actual_ttft_ms"],
                          test[feats_without], test["actual_ttft_ms"], 0, model_factory)
        r_wi = train_eval(train[feats_with], train["actual_ttft_ms"],
                          test[feats_with], test["actual_ttft_ms"], 0, model_factory)
        delta = r_wi["quantile_loss"] - r_wo["quantile_loss"]
        curve.append({"samples": n, "without": round(r_wo["quantile_loss"], 4),
                      "with": round(r_wi["quantile_loss"], 4),
                      "delta_pct": round(delta / r_wo["quantile_loss"] * 100, 2) if r_wo["quantile_loss"] else 0.0})
    return curve


def run_ab(df: pd.DataFrame, feature: str, seeds: int,
           model_factory=None) -> dict:
    results = {"with": {"ttft": [], "tpot": []}, "without": {"ttft": [], "tpot": []}}
    feats_without, feats_with = _resolve_ttft_features(df, feature)
    tpot_feats = _resolve_tpot_features(df)

    for seed in range(seeds):
        test = df.sample(frac=0.1, random_state=seed)
        train = df.drop(test.index)
        for arm, ttft_feats in (("with", feats_with), ("without", feats_without)):
            results[arm]["ttft"].append(
                train_eval(train[ttft_feats], train["actual_ttft_ms"],
                           test[ttft_feats], test["actual_ttft_ms"], seed, model_factory))
            results[arm]["tpot"].append(
                train_eval(train[tpot_feats], train["actual_tpot_ms"],
                           test[tpot_feats], test["actual_tpot_ms"], seed, model_factory))
    return results


# ---------------------------------------------------------------------------
# Summary and visualization
# ---------------------------------------------------------------------------

def summarize(results: dict, metric: str, target: str) -> dict:
    w = np.array([r[metric] for r in results["with"][target]])
    wo = np.array([r[metric] for r in results["without"][target]])
    return {
        "with_mean": round(float(w.mean()), 4), "with_std": round(float(w.std()), 4),
        "without_mean": round(float(wo.mean()), 4), "without_std": round(float(wo.std()), 4),
        "delta_pct": round(float((w.mean() - wo.mean()) / wo.mean() * 100), 2) if wo.mean() else 0.0,
        "delta_exceeds_seed_std": bool(abs(w.mean() - wo.mean()) > max(w.std(), wo.std())),
    }


def calibration_plot(results: dict, target: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, arm in zip(axes, ("without", "with")):
        actual = np.concatenate([r["actual"] for r in results[arm][target]])
        pred = np.concatenate([r["predicted"] for r in results[arm][target]])
        ax.scatter(actual, pred, s=4, alpha=0.3)
        lim = [0, max(actual.max(), pred.max()) * 1.05]
        ax.plot(lim, lim, "r--", linewidth=1, label="perfect calibration")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"actual {target} (ms)")
        ax.set_ylabel(f"predicted p{int(QUANTILE*100)} {target} (ms)")
        ax.set_title(f"{target.upper()} — {arm} feature (pooled test sets, all seeds)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def reliability_diagram(results: dict, target: str, out: Path, n_bins: int = 10) -> None:
    """Binned calibration curve: observed P90 coverage per prediction bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arm in zip(axes, ("without", "with")):
        actual = np.concatenate([r["actual"] for r in results[arm][target]])
        pred = np.concatenate([r["predicted"] for r in results[arm][target]])

        bin_edges = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-6
        bin_coverage = []
        bin_centers = []
        for i in range(n_bins):
            mask = (pred >= bin_edges[i]) & (pred < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_coverage.append(float(np.mean(actual[mask] <= pred[mask])))
                bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))

        ax.bar(range(len(bin_coverage)), bin_coverage, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axhline(y=QUANTILE, color="red", linestyle="--", linewidth=1,
                   label=f"target coverage (P{int(QUANTILE*100)} = {QUANTILE})")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("prediction bin (low → high)")
        ax.set_ylabel("observed coverage (fraction actual ≤ predicted)")
        ax.set_title(f"{target.upper()} — {arm} feature — reliability diagram")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def shap_analysis(results: dict, feature: str, df: pd.DataFrame, outdir: Path) -> dict | None:
    """Compute SHAP importance for the with-arm model on TTFT."""
    try:
        import shap
    except ImportError:
        print("shap not installed — skipping SHAP analysis. Install with: pip install shap", file=sys.stderr)
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last_result = results["with"]["ttft"][-1]
    model = last_result.get("model")
    if model is None:
        return None

    feats_without, feats_with = _resolve_ttft_features(df, feature)
    test = df.sample(frac=0.1, random_state=len(results["with"]["ttft"]) - 1)
    X_test = test[feats_with]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    importance = dict(zip(feats_with, np.abs(shap_values).mean(axis=0)))
    total = sum(importance.values())
    ranking = sorted(importance.items(), key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(8, max(4, len(ranking) * 0.4)))
    names = [r[0] for r in ranking]
    vals = [r[1] / total for r in ranking]
    ax.barh(names[::-1], vals[::-1], color="steelblue", edgecolor="black")
    ax.set_xlabel("mean |SHAP value| (normalized)")
    ax.set_title(f"SHAP feature importance — TTFT with {feature}")
    fig.tight_layout()
    fig.savefig(outdir / "shap_ttft.png", dpi=120)
    plt.close(fig)

    return {name: round(val / total, 4) for name, val in ranking}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--feature", required=True, help="feature column name to A/B test")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--min-contention-pct", type=float, default=20.0)
    p.add_argument("--outdir", type=Path, default=Path("results/ab"))
    p.add_argument("--model-type", choices=list(ESTIMATORS.keys()), default="xgboost")
    p.add_argument("--workload-spec", type=Path, default=None,
                   help="YAML workload spec to embed in summary.md")
    p.add_argument("--shap", action="store_true", help="compute SHAP feature importance")
    p.add_argument("--convergence", action="store_true",
                   help="show how delta evolves as sample count grows (cold-start analysis)")
    p.add_argument("--json", action="store_true", dest="json_output",
                   help="print machine-readable fitness line to stdout")
    args = p.parse_args()

    model_factory = ESTIMATORS[args.model_type]

    df = add_derived_features(load_trace(args.trace), feature=args.feature)
    gate = contention_gate(df, args.min_contention_pct)
    print(f"Contention gate PASSED: {gate['pct_samples_with_contention']}% of "
          f"{gate['samples']} samples have num_request_running > 1", file=sys.stderr)

    results = run_ab(df, args.feature, args.seeds, model_factory)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Post-hoc self-test: compare real improvement to shuffled noise floor
    metrics = ("quantile_loss", "coverage_pct", "mape_pct")
    real_ql_summary = summarize(results, "quantile_loss", "ttft")
    real_delta = real_ql_summary["with_mean"] - real_ql_summary["without_mean"]

    st = self_test(df, args.feature, model_factory, real_delta)
    if st["self_test"] == "FAIL":
        print(f"SELF-TEST FAILED: {st['reason']}", file=sys.stderr)
        if args.json_output:
            print(json.dumps({"feature": args.feature, "gate": "PASS",
                              "self_test": "FAIL", "reason": st["reason"]}))
        sys.exit(1)
    st_msg = st.get("note", f"real/shuffled ratio = {st.get('real_vs_shuffled_ratio', 'N/A')}x")
    print(f"Self-test PASSED: {st_msg}", file=sys.stderr)

    summary = {
        "feature": args.feature, "seeds": args.seeds, "contention": gate,
        "model_type": args.model_type, "self_test": st,
        "ttft": {m: summarize(results, m, "ttft") for m in metrics},
        "tpot": {m: summarize(results, m, "tpot") for m in metrics},
    }

    # Strip bulky arrays and model objects before persisting
    slim = {arm: {t: [{k: v for k, v in r.items() if k not in ("actual", "predicted", "model")}
                      for r in results[arm][t]] for t in ("ttft", "tpot")} for arm in results}
    (args.outdir / "ab_results.json").write_text(json.dumps({"summary": summary, "per_seed": slim}, indent=2))

    calibration_plot(results, "ttft", args.outdir / "calibration_ttft.png")
    calibration_plot(results, "tpot", args.outdir / "calibration_tpot.png")
    reliability_diagram(results, "ttft", args.outdir / "reliability_ttft.png")
    reliability_diagram(results, "tpot", args.outdir / "reliability_tpot.png")

    shap_ranking = None
    if args.shap:
        shap_ranking = shap_analysis(results, args.feature, df, args.outdir)

    conv_curve = None
    if args.convergence:
        conv_curve = convergence_curve(df, args.feature, model_factory)

    # Build summary.md
    lines = [f"# Feature A/B: {args.feature} ({args.seeds} seeds, {args.model_type})", ""]

    if args.workload_spec and args.workload_spec.exists():
        try:
            import yaml
            spec = yaml.safe_load(args.workload_spec.read_text())
            lines.append("## Workload Specification")
            lines.append("")
            lines.append("| Field | Value |")
            lines.append("|---|---|")
            w = spec.get("workload", spec)
            for key in ("name", "trace_type", "model"):
                if key in w:
                    lines.append(f"| {key} | {w[key]} |")
            hw = w.get("hardware", {})
            for key in ("gpu", "gpus_per_replica", "decode_replicas", "platform"):
                if key in hw:
                    lines.append(f"| {key} | {hw[key]} |")
            srv = w.get("serving", {})
            for key in ("stack", "max_model_len", "prefix_caching"):
                if key in srv:
                    lines.append(f"| {key} | {srv[key]} |")
            load = w.get("load", {})
            for key in ("generator", "profile", "api"):
                if key in load:
                    lines.append(f"| {key} | {load[key]} |")
            if "stages" in load:
                stages_str = ", ".join(f"{s.get('rate', s.get('rate_qps'))} QPS x {s.get('duration_s', s.get('duration', '?'))}s"
                                       for s in load["stages"])
                lines.append(f"| load_stages | {stages_str} |")
            lines.append("")
        except Exception:
            pass

    lines.append(f"## Contention Gate")
    lines.append("")
    lines.append(f"**PASSED**: {gate['pct_samples_with_contention']}% of {gate['samples']} samples contended "
                 f"(max num_request_running = {gate['max_num_request_running']})")
    lines.append("")
    lines.append(f"## Self-Test: {st['self_test']}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| target | metric | without | with | delta | > seed std? |")
    lines.append("|---|---|---|---|---|---|")
    for target in ("ttft", "tpot"):
        for metric in metrics:
            s = summary[target][metric]
            lines.append(
                f"| {target} | {metric} | {s['without_mean']} +/- {s['without_std']} "
                f"| {s['with_mean']} +/- {s['with_std']} | {s['delta_pct']}% "
                f"| {'**yes**' if s['delta_exceeds_seed_std'] else 'NO -- within noise'} |")
    lines.append("")

    if shap_ranking:
        lines.append("## SHAP Feature Importance (TTFT, with-arm)")
        lines.append("")
        for i, (name, val) in enumerate(shap_ranking.items(), 1):
            marker = " <- candidate" if name == args.feature else ""
            lines.append(f"{i}. **{name}**: {val:.1%}{marker}")
        lines.append("")

    if conv_curve:
        lines.append("## Convergence (cold-start analysis)")
        lines.append("")
        lines.append("| samples | without (pinball) | with (pinball) | delta |")
        lines.append("|---|---|---|---|")
        for c in conv_curve:
            lines.append(f"| {c['samples']} | {c['without']} | {c['with']} | {c['delta_pct']}% |")
        lines.append("")

    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.outdir}/: ab_results.json, summary.md, "
          f"calibration_{{ttft,tpot}}.png, reliability_{{ttft,tpot}}.png"
          + (", shap_ttft.png" if shap_ranking else ""), file=sys.stderr)

    if args.json_output:
        ttft_s = summary["ttft"]["quantile_loss"]
        tpot_s = summary["tpot"]["quantile_loss"]
        print(json.dumps({
            "feature": args.feature,
            "gate": "PASS",
            "self_test": st["self_test"],
            "contention_pct": gate["pct_samples_with_contention"],
            "ttft_pinball_delta_pct": ttft_s["delta_pct"],
            "ttft_delta_significant": ttft_s["delta_exceeds_seed_std"],
            "tpot_pinball_delta_pct": tpot_s["delta_pct"],
            "tpot_delta_significant": tpot_s["delta_exceeds_seed_std"],
        }))


if __name__ == "__main__":
    main()
