#!/usr/bin/env python3
"""Verify benchmark feature engineering matches production.

Run this after production changes to catch drift:
    python benchmarks/test_parity.py

Exits 0 if benchmark and production produce the same columns.
Exits 1 if they diverge — update add_derived_features() to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offline_feature_ab import _resolve_tpot_features, _resolve_ttft_features, add_derived_features


def main() -> int:
    try:
        from training.training_server import LatencyPredictor
    except ImportError:
        print("FAIL: training_server not importable. Run from repo root or set PYTHONPATH.", file=sys.stderr)
        return 1

    class _Minimal:
        prefix_buckets = 4

    df = pd.DataFrame(
        {
            "kv_cache_percentage": [0.5],
            "input_token_length": [100],
            "num_request_waiting": [1],
            "num_request_running": [2],
            "prefix_cache_score": [0.3],
            "prefill_tokens_in_flight": [500],
            "decode_tokens_in_flight": [100],
            "num_tokens_generated": [50],
            "actual_ttft_ms": [50.0],
            "actual_tpot_ms": [5.0],
        }
    )

    prod_ttft = list(LatencyPredictor._prepare_features_with_interaction(_Minimal(), df.copy(), "ttft").columns)
    prod_tpot = list(LatencyPredictor._prepare_features_with_interaction(_Minimal(), df.copy(), "tpot").columns)

    bench_df = add_derived_features(df.copy())
    _, bench_ttft = _resolve_ttft_features(bench_df, None)
    bench_tpot = _resolve_tpot_features(bench_df)

    errors = []
    if bench_ttft != prod_ttft:
        errors.append(f"TTFT mismatch:\n  production: {prod_ttft}\n  benchmark:  {bench_ttft}")
    if bench_tpot != prod_tpot:
        errors.append(f"TPOT mismatch:\n  production: {prod_tpot}\n  benchmark:  {bench_tpot}")

    if errors:
        print("PARITY CHECK FAILED — benchmark and production diverged:")
        for e in errors:
            print(f"  {e}")
        print("\nUpdate add_derived_features() and _resolve_*_features() to match production.")
        return 1

    print("PARITY CHECK PASSED — benchmark columns match production exactly")
    print(f"  TTFT: {prod_ttft}")
    print(f"  TPOT: {prod_tpot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
