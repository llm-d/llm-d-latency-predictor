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
        """Apply a batch of (y - qhat) scores as sequential per-observation ACI
        steps. For each score: evaluate the miss against the current offset, step
        alpha_t, then fold the score into the buffer.

        Stepping per score (rather than once per batch) keeps the effective learning
        rate independent of how many scores arrive per evaluation interval, so gamma
        carries the same meaning here as in the per-request formulation. A single
        per-batch step would scale the rate by 1/N and make gamma nearly inert at
        the server's interval cadence.
        """
        for s in np.asarray(scores, dtype=float):
            miss = 1.0 if s > self.offset() else 0.0
            self.alpha = min(1.0, max(0.0, self.alpha + self.gamma * (self.target - miss)))
            self.scores.append(float(s))

    def rebase(self) -> None:
        """Reset after a retrain. The buffered scores were computed against the old
        qhat; a fresh model already absorbs that shift, so keeping them would inflate
        the interval and cause transient over-coverage. Clear them and reset alpha.
        """
        self.scores.clear()
        self.alpha = self.target
