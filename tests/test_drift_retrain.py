# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the live drift-detection / dynamic retraining trigger (issue #40).

Unlike tests/test_dual_server_client.py, these instantiate LatencyPredictor directly
in-process rather than hitting a live HTTP deployment - the trigger logic is pure
state manipulation over deques, so it doesn't need a running server or a trained
model to exercise meaningfully.
"""

import time

import pytest

from training.training_server import LatencyPredictor, TrainingEntry, settings


def make_predictor() -> LatencyPredictor:
    return LatencyPredictor()


def feed(
    predictor: LatencyPredictor,
    n: int,
    actual_ttft: float,
    predicted_ttft: float,
    actual_tpot: float = 20.0,
    predicted_tpot: float = 20.0,
):
    """Push n samples with a fixed actual/predicted pair through add_training_sample,
    bypassing the bucket train/test split concerns - only the live-drift bookkeeping
    is under test here."""
    for _ in range(n):
        predictor.add_training_sample(
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 100,
                "num_request_waiting": 0,
                "num_request_running": 1,
                "actual_ttft_ms": actual_ttft,
                "actual_tpot_ms": actual_tpot,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.0,
                "predicted_ttft_ms": predicted_ttft,
                "predicted_tpot_ms": predicted_tpot,
            }
        )


def test_training_entry_accepts_missing_predictions():
    """Older routers that don't send predicted_ttft_ms/predicted_tpot_ms must still validate."""
    entry = TrainingEntry(
        kv_cache_percentage=0.5,
        input_token_length=100,
        num_request_waiting=0,
        num_request_running=1,
        actual_ttft_ms=50.0,
        actual_tpot_ms=20.0,
        num_tokens_generated=10,
        prefix_cache_score=0.0,
    )
    assert entry.predicted_ttft_ms is None
    assert entry.predicted_tpot_ms is None


def test_samples_without_predictions_dont_feed_live_window():
    """A sample with no predicted_ttft_ms/predicted_tpot_ms must not contribute to
    the live drift signal (this is what makes the fields backward compatible)."""
    p = make_predictor()
    for _ in range(10):
        p.add_training_sample(
            {
                "kv_cache_percentage": 0.5,
                "input_token_length": 100,
                "num_request_waiting": 0,
                "num_request_running": 1,
                "actual_ttft_ms": 50.0,
                "actual_tpot_ms": 20.0,
                "num_tokens_generated": 10,
                "prefix_cache_score": 0.0,
                # predicted_ttft_ms / predicted_tpot_ms omitted entirely
            }
        )
    metrics = p.get_live_drift_metrics()
    assert metrics["ttft_live_sample_count"] == 0
    assert metrics["tpot_live_sample_count"] == 0


def test_no_trigger_without_baseline():
    """Before any model generation has completed, there's no baseline to drift
    against - the dynamic trigger must not fire, even with wildly inaccurate
    predictions, and should defer to the fixed interval instead."""
    p = make_predictor()
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK + 10, actual_ttft=500.0, predicted_ttft=10.0)
    triggered, reason = p.should_retrain_due_to_drift()
    assert triggered is False
    assert "baseline" not in reason  # falls through to "no drift", not a baseline-specific message
    assert p.get_live_drift_metrics()["ttft_live_nrmse"] is not None


def test_synthetic_drift_triggers_retrain():
    """Core acceptance case: once a baseline exists, injecting a large synthetic
    residual (predictions far below actuals) must trigger the dynamic retrain."""
    p = make_predictor()

    # Simulate a completed model generation with a tight baseline (small, consistent
    # residual) - this is what train() would snapshot after a real retrain.
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK, actual_ttft=100.0, predicted_ttft=98.0)
    baseline_nrmse, baseline_violation = LatencyPredictor._live_nrmse_and_violation(
        p.live_ttft_sq_errors, p.live_ttft_actuals, p.live_ttft_violations
    )
    p.ttft_baseline_nrmse = baseline_nrmse
    p.ttft_baseline_violation_rate = baseline_violation
    p.live_ttft_sq_errors.clear()
    p.live_ttft_actuals.clear()
    p.live_ttft_violations.clear()
    # Backdate last_retrain_time so the cooldown window has already elapsed.
    from datetime import UTC, datetime, timedelta

    p.last_retrain_time = datetime.now(UTC) - timedelta(seconds=settings.MIN_SECONDS_BETWEEN_RETRAINS + 1)

    # Now inject synthetic drift: predictions consistently and severely underestimate actuals.
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK, actual_ttft=100.0, predicted_ttft=20.0)

    triggered, reason = p.should_retrain_due_to_drift()
    assert triggered is True
    assert "ttft" in reason


def test_stable_predictions_do_not_trigger_retrain():
    """Negative case: once a baseline exists, live predictions that stay close to
    it must not trigger a retrain."""
    p = make_predictor()

    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK, actual_ttft=100.0, predicted_ttft=98.0)
    baseline_nrmse, baseline_violation = LatencyPredictor._live_nrmse_and_violation(
        p.live_ttft_sq_errors, p.live_ttft_actuals, p.live_ttft_violations
    )
    p.ttft_baseline_nrmse = baseline_nrmse
    p.ttft_baseline_violation_rate = baseline_violation
    p.live_ttft_sq_errors.clear()
    p.live_ttft_actuals.clear()
    p.live_ttft_violations.clear()
    from datetime import UTC, datetime, timedelta

    p.last_retrain_time = datetime.now(UTC) - timedelta(seconds=settings.MIN_SECONDS_BETWEEN_RETRAINS + 1)

    # New window looks statistically like the baseline generation.
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK, actual_ttft=100.0, predicted_ttft=97.0)

    triggered, _ = p.should_retrain_due_to_drift()
    assert triggered is False


def test_cooldown_suppresses_retrigger_immediately_after_retrain():
    """Even with severe drift present, a retrain that just happened must not
    immediately trigger another one - avoids retrain storms."""
    p = make_predictor()
    p.ttft_baseline_nrmse = 0.02
    p.ttft_baseline_violation_rate = 0.1
    from datetime import UTC, datetime

    p.last_retrain_time = datetime.now(UTC)  # just retrained

    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK + 10, actual_ttft=100.0, predicted_ttft=10.0)

    triggered, reason = p.should_retrain_due_to_drift()
    assert triggered is False
    assert reason == "cooldown"


def test_violation_rate_alone_can_trigger():
    """A predictor that's systematically optimistic (always predicts low) but with
    small-magnitude errors should still trigger via violation rate, even if NRMSE
    alone might stay under the multiplier threshold."""
    p = make_predictor()
    # Baseline: well-calibrated, ~50% violation rate (typical for a well-fit model
    # under a symmetric error distribution).
    for i in range(settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK):
        actual = 100.0 + (1 if i % 2 == 0 else -1)
        feed(p, 1, actual_ttft=actual, predicted_ttft=100.0)
    baseline_nrmse, baseline_violation = LatencyPredictor._live_nrmse_and_violation(
        p.live_ttft_sq_errors, p.live_ttft_actuals, p.live_ttft_violations
    )
    p.ttft_baseline_nrmse = baseline_nrmse
    p.ttft_baseline_violation_rate = baseline_violation
    p.live_ttft_sq_errors.clear()
    p.live_ttft_actuals.clear()
    p.live_ttft_violations.clear()
    from datetime import UTC, datetime, timedelta

    p.last_retrain_time = datetime.now(UTC) - timedelta(seconds=settings.MIN_SECONDS_BETWEEN_RETRAINS + 1)

    # Drift: every sample now underestimates by a small, consistent margin -> violation rate ~100%.
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK, actual_ttft=101.0, predicted_ttft=100.0)

    triggered, reason = p.should_retrain_due_to_drift()
    assert triggered is True
    assert "violation rate" in reason

def test_baseline_not_overwritten_below_sample_floor():
    """A retrain during a quiet period (live window below the floor) must not
    overwrite an existing baseline with a noisy few-sample snapshot - the prior
    baseline carries over, while the window still resets for the new generation."""
    p = make_predictor()
    p.ttft_baseline_nrmse = 0.05
    p.ttft_baseline_violation_rate = 0.10

    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK - 1, actual_ttft=100.0, predicted_ttft=10.0)
    p._snapshot_baseline_if_enough_samples("ttft")

    assert p.ttft_baseline_nrmse == 0.05
    assert p.ttft_baseline_violation_rate == 0.10
    assert len(p.live_ttft_sq_errors) == 0


def test_baseline_overwritten_at_sample_floor():
    """Sanity check the opposite boundary: exactly MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK
    samples is enough to trust a new baseline snapshot."""
    p = make_predictor()
    p.ttft_baseline_nrmse = 0.05

    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK, actual_ttft=100.0, predicted_ttft=98.0)
    p._snapshot_baseline_if_enough_samples("ttft")

    assert p.ttft_baseline_nrmse != 0.05
    assert len(p.live_ttft_sq_errors) == 0

def test_tpot_drift_triggers_independently_of_ttft():
    p = make_predictor()
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK,
         actual_ttft=100.0, predicted_ttft=98.0, actual_tpot=20.0, predicted_tpot=19.5)
    ttft_nrmse, ttft_violation = LatencyPredictor._live_nrmse_and_violation(
        p.live_ttft_sq_errors, p.live_ttft_actuals, p.live_ttft_violations)
    tpot_nrmse, tpot_violation = LatencyPredictor._live_nrmse_and_violation(
        p.live_tpot_sq_errors, p.live_tpot_actuals, p.live_tpot_violations)
    p.ttft_baseline_nrmse, p.ttft_baseline_violation_rate = ttft_nrmse, ttft_violation
    p.tpot_baseline_nrmse, p.tpot_baseline_violation_rate = tpot_nrmse, tpot_violation
    p.live_ttft_sq_errors.clear(); p.live_ttft_actuals.clear(); p.live_ttft_violations.clear()
    p.live_tpot_sq_errors.clear(); p.live_tpot_actuals.clear(); p.live_tpot_violations.clear()

    from datetime import UTC, datetime, timedelta
    p.last_retrain_time = datetime.now(UTC) - timedelta(seconds=settings.MIN_SECONDS_BETWEEN_RETRAINS + 1)

    # ttft stays close to baseline; tpot degrades severely.
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK,
         actual_ttft=100.0, predicted_ttft=97.5, actual_tpot=20.0, predicted_tpot=4.0)

    triggered, reason = p.should_retrain_due_to_drift()
    assert triggered is True
    assert "tpot" in reason

def test_no_trigger_with_baseline_but_insufficient_live_samples():
    """Baseline exists from a prior retrain, but the current window hasn't
    reached MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK yet - must not trigger even with
    severe drift in the few samples collected so far."""
    p = make_predictor()
    p.ttft_baseline_nrmse = 0.02
    p.ttft_baseline_violation_rate = 0.10
    from datetime import UTC, datetime, timedelta
    p.last_retrain_time = datetime.now(UTC) - timedelta(seconds=settings.MIN_SECONDS_BETWEEN_RETRAINS + 1)

    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK - 1, actual_ttft=500.0, predicted_ttft=10.0)

    triggered, _ = p.should_retrain_due_to_drift()
    assert triggered is False

def test_disabled_flag_short_circuits(monkeypatch):
    p = make_predictor()
    p.ttft_baseline_nrmse = 0.01
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK + 10, actual_ttft=1000.0, predicted_ttft=1.0)
    monkeypatch.setattr(settings, "ENABLE_DRIFT_RETRAINING", False)

    triggered, reason = p.should_retrain_due_to_drift()
    assert triggered is False
    assert reason == "drift retraining disabled"

def test_get_live_drift_metrics_is_read_only():
    p = make_predictor()
    feed(p, settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK, actual_ttft=100.0, predicted_ttft=90.0)

    first = p.get_live_drift_metrics()
    second = p.get_live_drift_metrics()

    assert first == second
    assert len(p.live_ttft_sq_errors) == settings.MIN_LIVE_SAMPLES_FOR_DRIFT_CHECK
    assert p.ttft_baseline_nrmse is None

