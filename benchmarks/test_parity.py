#!/usr/bin/env python3
"""Verify benchmark fallback feature engineering matches production.

Validates that the handwritten fallback (used by CI and external users
when production code is not importable) produces the same feature columns
AND values as production. Uses a committed golden file as the reference
when production is unavailable.

Inside the repo (production importable):
  1. Compares fallback columns and values against live production.
  2. Updates golden_columns.json only after parity check passes.

Outside the repo (CI, external users):
  1. Compares fallback columns against golden_columns.json — catches drift.

Run after production changes to catch drift:
    python benchmarks/test_parity.py

Exits 0 if fallback matches the reference (production or golden file).
Exits 1 if they diverge — update the fallback to match production.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Pin feature flags before importing offline_feature_ab so _tif_enabled() and
# _enc_enabled() read deterministic values regardless of the caller's shell.
os.environ.setdefault("LATENCY_ENABLE_TOKEN_IN_FLIGHT_FEATURES", "true")
os.environ.setdefault("LATENCY_ENABLE_ENCODER_FEATURES", "true")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offline_feature_ab import (
    CONDITIONAL_FEATURES,
    ENCODER_FEATURES,
    PREFIX_BUCKETS,
    _add_derived_features_fallback,
    _resolve_tpot_features,
    _resolve_ttft_features,
)

GOLDEN_FILE = Path(__file__).resolve().parent / "golden_columns.json"

SAMPLE_DF = pd.DataFrame(
    {
        "kv_cache_percentage": [0.5, 0.0, 1.0, 0.8, 0.2],
        "input_token_length": [100, 1, 500, 200, 50],
        "num_request_waiting": [1, 0, 5, 0, 3],
        "num_request_running": [2, 1, 10, 4, 1],
        "prefix_cache_score": [0.3, 0.0, 1.0, 0.5, 0.99],
        "prefill_tokens_in_flight": [500, 0, 2000, 100, 50],
        "decode_tokens_in_flight": [100, 0, 800, 50, 10],
        "encoder_matched_size": [64, 0, 256, 32, 128],
        "encoder_input_size": [128, 0, 512, 64, 256],
        "num_tokens_generated": [50, 1, 200, 100, 10],
        "pod_type": ["", "prefill", "decode", "prefill", ""],
        "actual_ttft_ms": [50.0, 10.0, 200.0, 80.0, 30.0],
        "actual_tpot_ms": [5.0, 1.0, 20.0, 8.0, 3.0],
    }
)


def _check_derived_values(df: pd.DataFrame) -> list[str]:
    """Validate fallback-derived values against known formulas.

    Works without production code — catches formula bugs in CI.
    """
    errors = []
    for i in range(len(df)):
        row = df.iloc[i]
        w = row["num_request_waiting"]
        s = row["prefix_cache_score"]
        l = row["input_token_length"]  # noqa: E741

        expected_queued = 1 if w > 0 else 0
        if int(row["is_queued"]) != expected_queued:
            errors.append(f"Row {i}: is_queued expected {expected_queued}, got {int(row['is_queued'])} (waiting={w})")

        expected_eit = (1 - s) * l
        if abs(row["effective_input_tokens"] - expected_eit) > 1e-9:
            errors.append(
                f"Row {i}: effective_input_tokens expected {expected_eit}, "
                f"got {row['effective_input_tokens']} (score={s}, length={l})"
            )

        expected_bucket = min(int(max(0.0, min(1.0, s)) * PREFIX_BUCKETS), PREFIX_BUCKETS - 1)
        if int(row["prefill_score_bucket"]) != expected_bucket:
            errors.append(
                f"Row {i}: prefill_score_bucket expected {expected_bucket}, "
                f"got {int(row['prefill_score_bucket'])} (score={s})"
            )

    cats = set(df["pod_type_cat"].cat.categories)
    expected_cats = {"", "prefill", "decode"}
    if cats != expected_cats:
        errors.append(f"pod_type_cat categories expected {expected_cats}, got {cats}")

    return errors


def _build_fallback_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Force the fallback path and return the full derived DataFrame."""
    df = df.copy()
    for col in CONDITIONAL_FEATURES + ENCODER_FEATURES + ["num_tokens_generated"]:
        if col not in df.columns:
            df[col] = 0
    _add_derived_features_fallback(df)
    return df


def _get_fallback_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Force the fallback path and return its column lists."""
    fallback_df = _build_fallback_frame(df)
    with patch("offline_feature_ab._resolve_production_columns", return_value=None):
        _, fallback_ttft = _resolve_ttft_features(fallback_df, None)
        fallback_tpot = _resolve_tpot_features(fallback_df)
    return fallback_ttft, fallback_tpot


def main() -> int:
    fallback_ttft, fallback_tpot = _get_fallback_columns(SAMPLE_DF)

    prod_available = False
    try:
        from training.training_server import LatencyPredictor

        prod_available = True
    except ImportError:
        pass

    if prod_available:

        class _Minimal:
            prefix_buckets = PREFIX_BUCKETS

        prod_ttft = list(
            LatencyPredictor._prepare_features_with_interaction(_Minimal(), SAMPLE_DF.copy(), "ttft").columns
        )
        prod_tpot = list(
            LatencyPredictor._prepare_features_with_interaction(_Minimal(), SAMPLE_DF.copy(), "tpot").columns
        )
        reference_ttft, reference_tpot = prod_ttft, prod_tpot
        source = "production"
    else:
        if not GOLDEN_FILE.exists():
            print(
                "FAIL: production not importable and golden_columns.json missing.\n"
                "Run test_parity.py from repo root first to generate it.",
                file=sys.stderr,
            )
            return 1
        golden = json.loads(GOLDEN_FILE.read_text())
        reference_ttft, reference_tpot = golden["ttft"], golden["tpot"]
        source = f"golden file ({GOLDEN_FILE.name})"

    errors = []
    if fallback_ttft != reference_ttft:
        if set(fallback_ttft) == set(reference_ttft):
            errors.append(
                f"TTFT column ORDER mismatch (same features, wrong position — "
                f"order is load-bearing for XGBoost):\n"
                f"  {source}: {reference_ttft}\n  fallback:    {fallback_ttft}"
            )
        else:
            errors.append(f"TTFT column mismatch:\n  {source}: {reference_ttft}\n  fallback:    {fallback_ttft}")
    if fallback_tpot != reference_tpot:
        if set(fallback_tpot) == set(reference_tpot):
            errors.append(
                f"TPOT column ORDER mismatch (same features, wrong position — "
                f"order is load-bearing for XGBoost):\n"
                f"  {source}: {reference_tpot}\n  fallback:    {fallback_tpot}"
            )
        else:
            errors.append(f"TPOT column mismatch:\n  {source}: {reference_tpot}\n  fallback:    {fallback_tpot}")

    # Value comparison: verify fallback computes the same derived values as production
    if prod_available and not errors:
        prod_df = SAMPLE_DF.copy()
        for col in CONDITIONAL_FEATURES + ENCODER_FEATURES + ["num_tokens_generated"]:
            if col not in prod_df.columns:
                prod_df[col] = 0
        LatencyPredictor._prepare_features_with_interaction(_Minimal(), prod_df, "ttft")

        fallback_df = _build_fallback_frame(SAMPLE_DF)

        derived_cols = [c for c in prod_df.columns if c not in SAMPLE_DF.columns]
        for col in derived_cols:
            if col not in fallback_df.columns:
                errors.append(f"Derived column '{col}' missing from fallback")
                continue
            try:
                pd.testing.assert_series_equal(
                    prod_df[col].reset_index(drop=True),
                    fallback_df[col].reset_index(drop=True),
                    check_names=False,
                    check_dtype=False,
                )
            except AssertionError as e:
                errors.append(f"Value mismatch in '{col}':\n  {e}")

    # Derived-value validation: catches formula bugs without production code.
    fallback_df = _build_fallback_frame(SAMPLE_DF)
    value_errors = _check_derived_values(fallback_df)
    for ve in value_errors:
        errors.append(f"Derived value error: {ve}")

    if errors:
        print(f"PARITY CHECK FAILED — fallback diverged from {source}:")
        for e in errors:
            print(f"  {e}")
        print("\nUpdate _add_derived_features_fallback() and _resolve_*_features() fallback lists to match production.")
        return 1

    if prod_available:
        golden = {"ttft": prod_ttft, "tpot": prod_tpot}
        GOLDEN_FILE.write_text(json.dumps(golden, indent=2) + "\n")

    value_note = " (columns + values)" if prod_available else " (columns only)"
    print(f"PARITY CHECK PASSED — fallback matches {source}{value_note}")
    print(f"  TTFT: {reference_ttft}")
    print(f"  TPOT: {reference_tpot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
