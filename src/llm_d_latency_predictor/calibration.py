"""Calibration-drift detector for continuous coverage evaluation.

The decision logic is pure (no threading, sleeps, or servers) so it can be
unit-tested with fixed inputs. `continuous_coverage_loop` feeds one coverage
evaluation per interval into `CalibrationTrigger.update`, which returns True on
the step that should request a retrain.

See llm-d/llm-d-latency-predictor#19 / #32.
"""

from __future__ import annotations

# Sentinel distinct from None, since last_retrain_time can legitimately be None
# (model never retrained). _UNSET means "not waiting on a triggered retrain".
_UNSET = object()


class CalibrationTrigger:
    """Detects sustained calibration drift and decides when to request a retrain.

    Coverage is expected on a 0-100 scale (matching `quantile_coverage`), compared
    against `target_pct` (e.g. 90 for p90) -- NOT a 0-1 fraction. An EMA of the
    absolute deviation smooths transient noise; a retrain is requested only after
    the EMA stays above `threshold` for `k` consecutive evaluations.

    After firing, the detector suppresses further firing until the requested
    retrain lands (detected by a change in `last_retrain_time`), then resets its
    EMA and counter so post-retrain history starts clean. This prevents a slow
    retrain from queuing back-to-back retrains while the model is still drifted.

    A triggered retrain can also conclude WITHOUT landing a model: train()
    returns early when samples are below the minimum, when training raises, or
    when the predictor is not ready (e.g. /flush emptied the buckets right after
    a trigger). Waiting on `last_retrain_time` alone would then deadlock the
    detector in the pending state. `update` therefore also accepts the
    predictor's `train_attempts` counter; if an attempt concludes while
    `last_retrain_time` is unchanged, the pending state is released with the
    drift EMA preserved, so detection resumes and can re-fire after `k` more
    consecutive bad evaluations.
    """

    def __init__(self, target_pct: float, threshold: float, k: int, ema_alpha: float):
        self.target_pct = target_pct
        self.threshold = threshold
        self.k = k
        self.ema_alpha = ema_alpha
        self.ttft_ema = 0.0
        self.tpot_ema = 0.0
        self.consecutive_bad = 0
        self._pending_retrain = _UNSET
        self._pending_attempts: int | None = None

    @property
    def max_dev(self) -> float:
        return max(self.ttft_ema, self.tpot_ema)

    @property
    def awaiting_retrain(self) -> bool:
        return self._pending_retrain is not _UNSET

    def update(
        self,
        ttft_cov: float | None,
        tpot_cov: float | None,
        last_retrain_time,
        train_attempts: int | None = None,
    ) -> bool:
        """Fold in one coverage evaluation. Returns True if a retrain should fire now.

        `train_attempts` is the predictor's monotonic count of concluded train()
        calls (landed or not). When provided, it releases the pending state if a
        triggered retrain concluded without landing a model.
        """
        a = self.ema_alpha
        if ttft_cov is not None:
            self.ttft_ema = (1 - a) * self.ttft_ema + a * abs(ttft_cov - self.target_pct)
        if tpot_cov is not None:
            self.tpot_ema = (1 - a) * self.tpot_ema + a * abs(tpot_cov - self.target_pct)

        # A triggered retrain is in flight: keep tracking the EMA (for /metrics) but
        # do not re-fire. Once the retrain lands, reset so history starts fresh.
        if self._pending_retrain is not _UNSET:
            if last_retrain_time != self._pending_retrain:
                self.reset()
            elif (
                train_attempts is not None
                and self._pending_attempts is not None
                and train_attempts > self._pending_attempts
            ):
                # A train() call concluded without landing a model (skipped or
                # failed), so the requested retrain is not coming. Release the
                # pending state but keep the drift EMA: detection resumes and
                # re-fires after k more consecutive bad evaluations.
                self._pending_retrain = _UNSET
                self._pending_attempts = None
                self.consecutive_bad = 0
            return False

        if self.max_dev > self.threshold:
            self.consecutive_bad += 1
            if self.consecutive_bad >= self.k:
                self.consecutive_bad = 0
                self._pending_retrain = last_retrain_time
                self._pending_attempts = train_attempts
                return True
        else:
            self.consecutive_bad = 0
        return False

    def reset(self) -> None:
        """Clear EMA, counter, and pending-retrain state."""
        self.ttft_ema = 0.0
        self.tpot_ema = 0.0
        self.consecutive_bad = 0
        self._pending_retrain = _UNSET
        self._pending_attempts = None
