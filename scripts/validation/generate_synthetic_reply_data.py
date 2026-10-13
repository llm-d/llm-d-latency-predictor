#!/usr/bin/env python3
"""
Generates synthetic calibration / steady-state / drift-injected CSVs
for replay_drift_validation.py, since real historical (predicted,
actual) logs aren't available yet.

This follows the same idea as your own test_synthetic_drift_triggers_
retrain() in tests/test_drift_retrain.py -- predicted_* is fixed at
the true p90 of the underlying distribution (matching the model's
quantile objective), so ~10% of actuals naturally exceed it in the
steady-state segment. That's the expected baseline violation rate the
trigger should NOT treat as drift -- exactly the thing you wanted to
confirm before trusting the sweep numbers.

Usage:
    python3  scripts/validation/generate_synthetic_replay_data.py --out-dir data/
Produces: data/calibration.csv, data/steady.csv, data/drift.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
Z_P90 = 1.2816  # z-score for the 90th percentile of a normal distribution


def make_segment(n, start_ts, dt_s, ttft_mean, ttft_std, tpot_mean, tpot_std):
    """n chronological samples of a stable regime."""
    ttft_p90 = ttft_mean + Z_P90 * ttft_std
    tpot_p90 = tpot_mean + Z_P90 * tpot_std
    return pd.DataFrame({
        "timestamp_s": start_ts + np.arange(n) * dt_s,
        "actual_ttft_ms": RNG.normal(ttft_mean, ttft_std, n).clip(min=10),
        "actual_tpot_ms": RNG.normal(tpot_mean, tpot_std, n).clip(min=1),
        # predictions stay fixed at the calibration regime's true p90 --
        # this is what makes it "stale" once drift is injected below.
        "predicted_ttft_ms": ttft_p90,
        "predicted_tpot_ms": tpot_p90,
    })


def inject_drift(df: pd.DataFrame, start_frac: float, ttft_shift_ms: float, tpot_shift_ms: float):
    """Ramp a latency shift in from start_frac of the segment onward --
    e.g. a noisy neighbor or a regression in a new build. Predictions
    are left untouched (stale), only actuals move."""
    df = df.copy()
    n = len(df)
    start = int(n * start_frac)
    ramp = np.linspace(0, 1, n - start)
    idx = df.index[start:]
    df.loc[idx, "actual_ttft_ms"] += ttft_shift_ms * ramp
    df.loc[idx, "actual_tpot_ms"] += tpot_shift_ms * ramp
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--dt-s", type=float, default=0.2, help="seconds between requests")
    ap.add_argument("--calibration-n", type=int, default=500)
    ap.add_argument("--steady-n", type=int, default=20000, help="~1hr at dt=0.2s")
    ap.add_argument("--drift-n", type=int, default=20000)
    ap.add_argument("--drift-start-frac", type=float, default=0.3)
    ap.add_argument("--ttft-shift-ms", type=float, default=150.0)
    ap.add_argument("--tpot-shift-ms", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same underlying regime for calibration and steady-state -- only
    # the drift file gets a shift injected partway through.
    regime = dict(ttft_mean=400, ttft_std=80, tpot_mean=25, tpot_std=6)

    calibration = make_segment(args.calibration_n, start_ts=0, dt_s=args.dt_s, **regime)
    steady = make_segment(
        args.steady_n, start_ts=calibration["timestamp_s"].iloc[-1] + args.dt_s,
        dt_s=args.dt_s, **regime,
    )
    drift_base = make_segment(
        args.drift_n, start_ts=steady["timestamp_s"].iloc[-1] + args.dt_s,
        dt_s=args.dt_s, **regime,
    )
    drift = inject_drift(
        drift_base,
        start_frac=args.drift_start_frac,
        ttft_shift_ms=args.ttft_shift_ms,
        tpot_shift_ms=args.tpot_shift_ms,
    )

    calibration.to_csv(out_dir / "calibration.csv", index=False)
    steady.to_csv(out_dir / "steady.csv", index=False)
    drift.to_csv(out_dir / "drift.csv", index=False)

    injection_row = int(len(drift) * args.drift_start_frac)
    injection_ts = float(drift["timestamp_s"].iloc[injection_row])

    # So replay_drift_validation.py can measure detection lag from the
    # actual start of the injected shift, not from the start of the
    # drift file (which includes deliberate pre-injection normal traffic).
    with open(out_dir / "drift_meta.json", "w") as f:
        json.dump({"injection_timestamp_s": injection_ts, "injection_row": injection_row}, f)

    print(f"Wrote {len(calibration)} calibration, {len(steady)} steady, "
          f"{len(drift)} drift rows to {out_dir}/")
    print(f"Drift injected from row {injection_row} onward (t={injection_ts:.1f}s) "
          f"-- recorded in {out_dir}/drift_meta.json")


if __name__ == "__main__":
    main()

