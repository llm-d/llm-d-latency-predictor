"""Adaptive Conformal Inference (Gibbs & Candes, 2021) for online interval
calibration of the latency predictor.

The base model predicts a fixed quantile qhat(x). Between scheduled retrains,
distribution drift makes that static interval under-cover. ACIState keeps the
interval valid by adapting a one-sided upper offset c_t online, so long-run
coverage holds at (1 - target) even as the input distribution drifts.

See llm-d/llm-d-latency-predictor#19.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ACIState:
    """One-sided upper adaptive-conformal state for a single target (ttft/tpot).

    Maintains alpha_t and a buffer of recent conformity scores s = y - qhat(x).
    The served interval is qhat(x) + c_t, where c_t is the (1 - alpha_t) quantile
    of recent scores. alpha_t adapts toward the target miscoverage using the
    realized miss rate, so coverage self-corrects under drift.
    """

    def __init__(self, gamma: float, buffer_size: int, target: float):
        self.gamma = gamma
        self.target = target  # target miscoverage = 1 - quantile_alpha (e.g. 0.1 for p90)
        self.alpha = target
        self.scores: deque[float] = deque(maxlen=buffer_size)

    def offset(self) -> float:
        """Current additive interval offset c_t. Zero until the buffer has data."""
        if not self.scores:
            return 0.0
        return float(np.quantile(self.scores, 1.0 - self.alpha))

    def update_batch(self, scores) -> None:
        """Step alpha_t once from a batch of (y - qhat) scores, using the realized
        miss rate against the current offset, then fold the scores into the buffer.

        The batched form matches the continuous-coverage loop cadence (one step per
        evaluation interval); the score buffer provides coarse reactivity and gamma
        the fine adjustment.
        """
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return
        miss_rate = float(np.mean(scores > self.offset()))
        self.alpha = min(1.0, max(0.0, self.alpha + self.gamma * (self.target - miss_rate)))
        self.scores.extend(scores.tolist())

    def rebase(self) -> None:
        """Reset after a retrain. The buffered scores were computed against the old
        qhat; a fresh model already absorbs that shift, so keeping them would inflate
        the interval and cause transient over-coverage. Clear them and reset alpha.
        """
        self.scores.clear()
        self.alpha = self.target
