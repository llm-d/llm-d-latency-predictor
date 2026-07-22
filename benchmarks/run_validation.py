#!/usr/bin/env python3
"""Runner script: single-command validation for latency predictor features.

Chains: load trace → contention gate → A/B → self-test → artifacts → exit code.

Usage:
  python benchmarks/run_validation.py \
      --trace benchmarks/traces/sharegpt-h200.jsonl \
      --feature <feature_name> \
      --seeds 10 \
      --workload-spec benchmarks/traces/sharegpt-h200-spec.yaml \
      --outdir results/<feature_name> \
      --shap --json

Exit codes:
  0 — all gates passed, artifacts generated
  1 — contention gate or self-test failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_feature_ab import (
    ESTIMATORS,
    add_derived_features,
    calibration_plot,
    contention_gate,
    load_trace,
    reliability_diagram,
    run_ab,
    self_test,
    shap_analysis,
    summarize,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Single-command feature validation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--trace", type=Path, required=True,
                   help="JSONL trace from trace_recorder.py")
    p.add_argument("--feature", required=True,
                   help="feature column name to A/B test")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--min-contention-pct", type=float, default=20.0)
    p.add_argument("--model-type", choices=list(ESTIMATORS.keys()), default="xgboost")
    p.add_argument("--workload-spec", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("results/ab"))
    p.add_argument("--shap", action="store_true")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    model_factory = ESTIMATORS[args.model_type]
    df = add_derived_features(load_trace(args.trace))

    gate = contention_gate(df, args.min_contention_pct)
    print(f"Contention gate PASSED: {gate['pct_samples_with_contention']}% of "
          f"{gate['samples']} samples have num_request_running > 1", file=sys.stderr)

    results = run_ab(df, args.feature, args.seeds, model_factory)
    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics = ("quantile_loss", "coverage_pct", "mape_pct")
    real_ql = summarize(results, "quantile_loss", "ttft")
    real_delta = real_ql["with_mean"] - real_ql["without_mean"]

    st = self_test(df, args.feature, model_factory, real_delta)
    if st["self_test"] == "FAIL":
        print(f"SELF-TEST FAILED: {st['reason']}", file=sys.stderr)
        if args.json_output:
            print(json.dumps({"feature": args.feature, "gate": "PASS",
                              "self_test": "FAIL", "reason": st["reason"]}))
        return 1
    st_msg = st.get("note", f"real/shuffled ratio = {st.get('real_vs_shuffled_ratio', 'N/A')}x")
    print(f"Self-test PASSED: {st_msg}", file=sys.stderr)

    summary = {
        "feature": args.feature, "seeds": args.seeds, "contention": gate,
        "model_type": args.model_type, "self_test": st,
        "ttft": {m: summarize(results, m, "ttft") for m in metrics},
        "tpot": {m: summarize(results, m, "tpot") for m in metrics},
    }

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
                stages_str = ", ".join(
                    f"{s.get('rate', s.get('rate_qps'))} QPS x {s.get('duration_s', s.get('duration', '?'))}s"
                    for s in load["stages"])
                lines.append(f"| load_stages | {stages_str} |")
            lines.append("")
        except Exception:
            pass

    lines.append("## Contention Gate")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
