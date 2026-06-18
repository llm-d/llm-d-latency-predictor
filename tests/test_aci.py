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


def test_high_miss_rate_lowers_alpha_widening_interval():
    a = ACIState(gamma=0.05, buffer_size=500, target=0.1)
    start = a.alpha
    a.update_batch(np.full(100, 50.0))  # empty-buffer offset 0, all miss -> miss_rate 1
    assert a.alpha < start  # 0.1 + 0.05*(0.1 - 1.0) = 0.055


def test_low_miss_rate_raises_alpha_narrowing_interval():
    a = ACIState(gamma=0.05, buffer_size=500, target=0.1)
    a.update_batch(np.full(100, -50.0))  # offset 0, none exceed it -> miss_rate 0
    assert a.alpha > a.target  # 0.1 + 0.05*(0.1 - 0.0) = 0.105


def test_alpha_clipped_to_unit_interval():
    lo = ACIState(gamma=0.9, buffer_size=10, target=0.1)
    lo.alpha = 0.05
    lo.update_batch(np.full(5, 100.0))  # would go negative -> clipped to 0
    assert lo.alpha == 0.0

    hi = ACIState(gamma=0.9, buffer_size=10, target=0.1)
    hi.alpha = 0.95
    hi.update_batch(np.full(5, -100.0))  # would exceed 1 -> clipped to 1
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
