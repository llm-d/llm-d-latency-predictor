"""Unit tests for the Adaptive Conformal Inference state machine (aci.py, #19)."""

import numpy as np

from llm_d_latency_predictor.aci import ACIState


def test_offset_empty_buffer_is_zero():
    a = ACIState(gamma=0.02, buffer_size=100, target=0.1)
    assert a.offset() == 0.0


def test_offset_is_one_minus_alpha_quantile_of_buffer():
    a = ACIState(gamma=0.0, buffer_size=200, target=0.1)  # gamma=0 keeps alpha fixed
    scores = np.arange(0, 100, dtype=float)
    a.update_batch(scores)
    assert a.alpha == 0.1  # unchanged with gamma=0
    assert abs(a.offset() - float(np.quantile(scores, 0.9))) < 1e-9


def test_single_miss_lowers_alpha_widening_interval():
    a = ACIState(gamma=0.05, buffer_size=500, target=0.1)
    a.update_batch(np.array([50.0]))  # empty-buffer offset 0, score exceeds it -> miss
    assert abs(a.alpha - (0.1 + 0.05 * (0.1 - 1.0))) < 1e-9  # 0.055


def test_single_cover_raises_alpha_narrowing_interval():
    a = ACIState(gamma=0.05, buffer_size=500, target=0.1)
    a.update_batch(np.array([-50.0]))  # does not exceed offset -> no miss
    assert abs(a.alpha - (0.1 + 0.05 * 0.1)) < 1e-9  # 0.105


def test_sustained_misses_drive_alpha_down():
    a = ACIState(gamma=0.001, buffer_size=200, target=0.1)
    a.update_batch(np.arange(1, 51, dtype=float) * 100)  # strictly increasing -> every step misses
    assert a.alpha < 0.06  # ~0.1 - 50*0.001*0.9 = 0.055


def test_steps_per_score_not_per_batch():
    # The cadence fix: a batch of N misses must move alpha ~N x a single miss,
    # otherwise the effective learning rate scales as 1/N and gamma goes inert.
    single = ACIState(gamma=0.001, buffer_size=200, target=0.1)
    single.update_batch(np.array([1000.0]))
    batch = ACIState(gamma=0.001, buffer_size=200, target=0.1)
    batch.update_batch(np.arange(1, 51, dtype=float) * 100)  # 50 sequential misses
    assert (0.1 - batch.alpha) > 20 * (0.1 - single.alpha)


def test_alpha_clipped_to_unit_interval():
    lo = ACIState(gamma=0.9, buffer_size=10, target=0.1)
    lo.alpha = 0.05
    lo.update_batch(np.array([100.0]))  # would go negative -> clipped to 0
    assert lo.alpha == 0.0

    hi = ACIState(gamma=0.9, buffer_size=10, target=0.1)
    hi.alpha = 0.95
    hi.update_batch(np.array([-100.0]))  # would exceed 1 -> clipped to 1
    assert hi.alpha == 1.0


def test_empty_batch_is_noop():
    a = ACIState(gamma=0.05, buffer_size=100, target=0.1)
    a.update_batch(np.array([]))
    assert a.alpha == 0.1
    assert len(a.scores) == 0


def test_rebase_clears_buffer_and_resets_alpha():
    a = ACIState(gamma=0.05, buffer_size=100, target=0.1)
    a.update_batch(np.arange(50, dtype=float))
    a.alpha = 0.3
    a.rebase()
    assert len(a.scores) == 0
    assert a.alpha == a.target
    assert a.offset() == 0.0


def test_buffer_respects_maxlen():
    a = ACIState(gamma=0.0, buffer_size=10, target=0.1)
    a.update_batch(np.arange(100, dtype=float))
    assert len(a.scores) == 10
