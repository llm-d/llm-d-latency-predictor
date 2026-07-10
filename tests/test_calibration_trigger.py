"""Unit tests for the calibration-drift trigger logic (calibration.py, #32).

Fixed inputs, no live servers -- proves the detector fires on sustained drift,
ignores one-off/noisy readings, never fires when calibrated, uses the 0-100
coverage scale, and does not queue back-to-back retrains while one is in flight.
"""

import datetime

from llm_d_latency_predictor.calibration import CalibrationTrigger


def new_trigger(target_pct=90.0, threshold=5.0, k=2, ema_alpha=0.3):
    return CalibrationTrigger(target_pct=target_pct, threshold=threshold, k=k, ema_alpha=ema_alpha)


def test_steady_drift_fires():
    t = new_trigger()
    # coverage stuck at 70 (20pp below the 90 target): EMA climbs past threshold and
    # stays there for k consecutive evals.
    assert t.update(70, 70, None) is False  # bad=1
    assert t.update(70, 70, None) is True  # bad=k -> fires


def test_one_off_bad_reading_does_not_fire():
    t = new_trigger()
    t.update(90, 90, None)  # calibrated
    t.update(70, 70, None)  # a single bad reading -> bad=1
    # the next good reading pulls the EMA back below threshold, so the counter resets
    # before reaching k.
    for cov in (90, 90, 90):
        assert t.update(cov, cov, None) is False


def test_noisy_readings_on_small_buffer_do_not_fire():
    t = new_trigger()
    # noisy coverage bouncing around the target; individual deviations spike but the
    # EMA never sustains above threshold.
    for cov in (82, 96, 85, 94, 88, 92, 87, 93):
        assert t.update(cov, cov, None) is False


def test_steady_calibrated_never_fires():
    t = new_trigger()
    for _ in range(50):
        assert t.update(90, 90, None) is False
    assert t.max_dev == 0.0


def test_coverage_uses_0_to_100_scale():
    # Correct scale: coverage 90 vs target 90 -> zero deviation, never fires.
    t = new_trigger()
    for _ in range(10):
        assert t.update(90.0, 90.0, None) is False

    # Scale insurance: if coverage were passed as a 0-1 fraction (0.9) against the
    # 90 target, deviation is ~89 and the trigger would fire nonstop. This asserts
    # the failure mode so a future scale regression is caught here, not in prod.
    misused = new_trigger()
    misused.update(0.9, 0.9, None)
    assert misused.update(0.9, 0.9, None) is True


def test_uses_worse_of_ttft_tpot():
    t = new_trigger()
    # tpot calibrated, ttft badly drifted -> should still fire on the worse signal.
    assert t.update(60, 90, None) is False
    assert t.update(60, 90, None) is True


def test_no_refire_until_retrain_lands_then_resets():
    t = new_trigger()
    assert t.update(70, 70, None) is False
    assert t.update(70, 70, None) is True  # fires; now awaiting the retrain
    assert t.awaiting_retrain

    # Retrain is in flight (last_retrain_time unchanged) and the model is still
    # drifted -> must NOT queue another retrain.
    for _ in range(5):
        assert t.update(70, 70, None) is False
    assert t.awaiting_retrain

    # Retrain lands (last_retrain_time changes): this eval resets the detector.
    landed = datetime.datetime(2026, 1, 1)
    assert t.update(70, 70, landed) is False
    assert not t.awaiting_retrain
    assert t.consecutive_bad == 0 and t.max_dev == 0.0

    # Detection resumes from a clean slate: fresh sustained drift is needed to fire
    # again -- no immediate re-fire off pre-retrain history.
    assert t.update(70, 70, landed) is False  # bad=1 (fresh)
    assert t.update(70, 70, landed) is True  # bad=k -> fires again


def test_skipped_retrain_releases_pending_and_refires():
    # kaushikmitr's deadlock scenario on #32: the trigger fires, the training
    # loop consumes it, but train() concludes WITHOUT landing a model (e.g.
    # /flush emptied the buckets so samples < MIN_SAMPLES_FOR_RETRAIN).
    # last_retrain_time never changes, so without the attempt counter the
    # detector would wait forever and the feature silently dies.
    t = new_trigger()
    assert t.update(70, 70, None, train_attempts=7) is False
    assert t.update(70, 70, None, train_attempts=7) is True  # fires at attempts=7
    assert t.awaiting_retrain

    # The triggered train() concludes (attempts advances) but skips: no model
    # landed, last_retrain_time still None -> pending state is released.
    assert t.update(70, 70, None, train_attempts=8) is False
    assert not t.awaiting_retrain
    # Drift EMA is preserved (drift is still live), only the streak restarts.
    assert t.max_dev > 0.0
    assert t.consecutive_bad == 0

    # Detection resumes: k more consecutive bad evals re-fire, giving periodic
    # retries until a retrain can actually land.
    assert t.update(70, 70, None, train_attempts=8) is False  # bad=1
    assert t.update(70, 70, None, train_attempts=8) is True  # bad=k -> re-fires


def test_landed_retrain_still_resets_with_attempt_counter():
    t = new_trigger()
    t.update(70, 70, None, train_attempts=3)
    assert t.update(70, 70, None, train_attempts=3) is True
    # Retrain lands: attempts advanced AND last_retrain_time changed. The
    # landed branch must win (full reset, clean slate), not the skip branch.
    landed = datetime.datetime(2026, 1, 1)
    assert t.update(70, 70, landed, train_attempts=4) is False
    assert not t.awaiting_retrain
    assert t.max_dev == 0.0 and t.consecutive_bad == 0


def test_pending_without_attempt_signal_keeps_waiting():
    # Callers that don't pass train_attempts keep the original semantics:
    # wait on last_retrain_time alone.
    t = new_trigger()
    t.update(70, 70, None)
    assert t.update(70, 70, None) is True
    for _ in range(10):
        assert t.update(70, 70, None) is False
    assert t.awaiting_retrain
