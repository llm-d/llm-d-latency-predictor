#!/usr/bin/env python3
"""
Offline replay-based validation of the drift-retraining thresholds
(llm-d-latency-predictor, PR #79 / issue #40).

Feeds (predicted, actual) latency pairs through the ACTUAL
LatencyPredictor.should_retrain_due_to_drift() logic -- not a
reimplementation of it -- while sweeping DRIFT_NRMSE_MULTIPLIER,
DRIFT_VIOLATION_RATE_ABS_INCREASE, MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK,
and LIVE_ERROR_WINDOW_SIZE one at a time against the shipped defaults.

This is NOT a pytest test and isn't meant to run in CI -- run it
directly and attach/summarize the CSV it produces as evidence in the
PR review thread.

PREREQUISITE: this calls predictor._snapshot_baseline_if_enough_
samples(), which only exists after the train()-dedup refactor
(extracting the inlined ttft/tpot snapshot block into that method).
If you haven't applied that yet, do it first -- you'll get the same
AttributeError the two existing unit tests hit.

Usage:
    python replay_drift_validation.py \
        --calibration data/calibration.csv \
        --steady data/steady.csv \
        --drift data/drift.csv \
        --out sweep_results.csv
"""

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime as real_datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Import the actual shipped predictor + settings so the sweep validates
# real code, not a reimplementation of the trigger logic. Adjust this
# path if your checkout layout differs.
def _find_repo_root(start: Path) -> Path:
    """Walk up from the script's location looking for the repo root, so
    this works no matter which directory the script itself lives in
    (repo root, scripts/, scripts/validation/, ...)."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "training" / "training_server.py").exists():
            return candidate
    raise RuntimeError(
        "Could not find the repo root (looked for pyproject.toml or "
        "training/training_server.py in every parent directory of "
        f"{start}). Run this from inside the llm-d-latency-predictor checkout."
    )


sys.path.insert(0, str(_find_repo_root(Path(__file__).resolve().parent)))
# Same import path your test suite uses (tests/test_drift_retrain.py) --
# importing as a package rather than a bare module also protects against
# any relative imports inside training_server.py breaking.
from training.training_server import LatencyPredictor, settings  # noqa: E402

REQUIRED_SAMPLE_FIELDS = [
    "kv_cache_percentage", "num_tokens_generated", "input_token_length",
    "num_request_waiting", "num_request_running", "prefix_cache_score",
]


# --------------------------------------------------------------------------
# Simulated clock: should_retrain_due_to_drift()'s cooldown check calls
# datetime.now(UTC) directly, which would barely advance during a fast
# offline replay loop and effectively disable the cooldown. Patching
# training_server.datetime lets the cooldown advance with the replay
# data's own timestamps instead, so trigger counts aren't inflated by
# rapid re-firing that wouldn't happen in production.
# --------------------------------------------------------------------------
class _SimNow:
    def __init__(self):
        self.current = real_datetime.now(UTC)

    def now(self, tz=None):
        return self.current

    def set(self, epoch_seconds: float):
        self.current = real_datetime.fromtimestamp(epoch_seconds, tz=UTC)


@contextmanager
def simulated_clock():
    sim = _SimNow()
    with patch("training.training_server.datetime", sim):
        yield sim


# --------------------------------------------------------------------------
# Sweep configuration: default row first, then one-variable-at-a-time
# variants (matches the shape of sample_sweep_results.csv).
# --------------------------------------------------------------------------
DEFAULTS = dict(
    nrmse_multiplier=settings.DRIFT_NRMSE_MULTIPLIER,
    violation_rate_delta=settings.DRIFT_VIOLATION_RATE_ABS_INCREASE,
    min_live_samples=settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK,
    window_size=settings.LIVE_ERROR_WINDOW_SIZE,
)

SWEEP_VALUES = {
    "nrmse_multiplier": [1.3, 1.7],
    "violation_rate_delta": [0.10, 0.20],
    "min_live_samples": [30],
    "window_size": [100, 300],
}


def build_sweep_configs():
    configs = [dict(DEFAULTS)]
    for key, values in SWEEP_VALUES.items():
        for v in values:
            cfg = dict(DEFAULTS)
            cfg[key] = v
            configs.append(cfg)
    return configs


def load_samples(path: str) -> list[dict]:
    df = pd.read_csv(path).sort_values("timestamp_s").reset_index(drop=True)
    required = ["timestamp_s", "actual_ttft_ms", "actual_tpot_ms",
                "predicted_ttft_ms", "predicted_tpot_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    # add_training_sample() requires these to be present and numeric even
    # though the drift path doesn't use them -- default to 0 if your
    # replay data doesn't carry them.
    for col in REQUIRED_SAMPLE_FIELDS:
        if col not in df.columns:
            df[col] = 0.0
    return df.to_dict("records")


def seed_baseline(predictor: LatencyPredictor, calibration_samples: list[dict], clock: _SimNow):
    """Mimic the end of a normal training cycle: accumulate live
    error/violation stats over the calibration window, then snapshot
    them as the baseline -- what train() does after a real retrain,
    without needing to actually fit a model."""
    for s in calibration_samples:
        clock.set(s["timestamp_s"])
        predictor.add_training_sample(s)
    predictor.last_retrain_time = clock.now()
    predictor._snapshot_baseline_if_enough_samples("ttft")
    predictor._snapshot_baseline_if_enough_samples("tpot")
    if predictor.ttft_baseline_nrmse is None or predictor.tpot_baseline_nrmse is None:
        raise RuntimeError(
            "Calibration window didn't produce a baseline for one or both "
            "metrics -- need at least MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK valid "
            "(predicted, actual) pairs of each in the calibration file."
        )


def replay(predictor: LatencyPredictor, samples: list[dict], clock: _SimNow):
    """Feed samples one at a time, checking the trigger after each. On a
    fire, reseed the baseline from what's accumulated so far and clear
    the cooldown clock -- mimicking a real retrain -- so one trigger
    doesn't just repeat on every subsequent sample."""
    events = []
    for s in samples:
        clock.set(s["timestamp_s"])
        predictor.add_training_sample(s)
        fired, reason = predictor.should_retrain_due_to_drift()
        if fired:
            events.append((s["timestamp_s"], reason))
            predictor.last_retrain_time = clock.now()
            predictor._snapshot_baseline_if_enough_samples("ttft")
            predictor._snapshot_baseline_if_enough_samples("tpot")
    return events


def run_one_config(cfg, calibration, steady, drift, injection_ts):
    settings.DRIFT_NRMSE_MULTIPLIER = cfg["nrmse_multiplier"]
    settings.DRIFT_VIOLATION_RATE_ABS_INCREASE = cfg["violation_rate_delta"]
    settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK = cfg["min_live_samples"]
    settings.LIVE_ERROR_WINDOW_SIZE = cfg["window_size"]

    rows = []
    for metric in ("ttft", "tpot"):
        with simulated_clock() as clock:
            p = LatencyPredictor()
            seed_baseline(p, calibration, clock)
            steady_events = [e for e in replay(p, steady, clock) if metric in e[1]]
        duration_hours = (steady[-1]["timestamp_s"] - steady[0]["timestamp_s"]) / 3600
        fp_per_hour = len(steady_events) / duration_hours if duration_hours > 0 else float("nan")

        with simulated_clock() as clock:
            p2 = LatencyPredictor()
            seed_baseline(p2, calibration, clock)
            drift_events = [e for e in replay(p2, drift, clock) if metric in e[1]]
        if drift_events:
            lag = drift_events[0][0] - injection_ts
            lag_str = round(lag, 2)
        else:
            lag_str = "never"  # worth flagging explicitly, not just a big number

        rows.append(dict(
            metric=metric,
            nrmse_multiplier=cfg["nrmse_multiplier"],
            violation_rate_delta=cfg["violation_rate_delta"],
            min_live_samples=cfg["min_live_samples"],
            window_size=cfg["window_size"],
            false_positives_per_hour=round(fp_per_hour, 2),
            detection_lag_seconds=lag_str,
            n_triggers_total=len(steady_events) + len(drift_events),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True, help="CSV used to seed the initial baseline")
    ap.add_argument("--steady", required=True, help="normal traffic, no injected shift -- measures false positives")
    ap.add_argument("--drift", required=True, help="a real/injected latency shift starts partway through -- measures detection lag")
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--drift-injection-ts", type=float, default=None,
                     help="epoch seconds where the injected shift actually starts within "
                          "--drift. Auto-loaded from drift_meta.json next to --drift if "
                          "generate_synthetic_replay_data.py wrote one; falls back to the "
                          "drift file's own first timestamp (with a warning) otherwise, "
                          "which UNDERSTATES the true detection lag if the drift file has "
                          "pre-injection lead-in samples.")
    args = ap.parse_args()

    calibration = load_samples(args.calibration)
    steady = load_samples(args.steady)
    drift = load_samples(args.drift)

    injection_ts = args.drift_injection_ts
    if injection_ts is None:
        meta_path = Path(args.drift).parent / "drift_meta.json"
        if meta_path.exists():
            injection_ts = json.loads(meta_path.read_text())["injection_timestamp_s"]
            print(f"Loaded injection timestamp {injection_ts:.1f}s from {meta_path}")
        else:
            injection_ts = drift[0]["timestamp_s"]
            print(f"WARNING: no drift_meta.json found next to --drift and "
                  f"--drift-injection-ts not given -- measuring lag from the drift "
                  f"file's first row (t={injection_ts:.1f}s). If your drift file has "
                  f"pre-injection lead-in samples, reported detection_lag_seconds will "
                  f"be inflated by however long that lead-in is.")

    all_rows = []
    for cfg in build_sweep_configs():
        all_rows.extend(run_one_config(cfg, calibration, steady, drift, injection_ts))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

