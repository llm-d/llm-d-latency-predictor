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
"""
Regression test for https://github.com/llm-d/llm-d-latency-predictor/issues/80

_drop_timestamp was defined at module scope but called as self._drop_timestamp()
inside LatencyPredictor, raising AttributeError on every train() invocation and
leaving predictions permanently stuck at the cold-start default (~10.0).
"""
import os
import random
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

# Lower the sample threshold so tests don't need to generate 1000+ rows
os.environ.setdefault("LATENCY_MIN_SAMPLES_FOR_RETRAIN", "20")
os.environ.setdefault("LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH", "10")
os.environ.setdefault("LATENCY_ENSEMBLE_MODE", "false")  # keep it simple for unit tests

from training.training_server import LatencyPredictor, _drop_timestamp, settings  # noqa: E402


def _make_sample(seed: int) -> dict:
    """Return a realistic training sample matching the required fields in add_training_sample()."""
    rng = random.Random(seed)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "input_token_length": rng.randint(32, 512),
        "num_tokens_generated": rng.randint(16, 256),
        "kv_cache_percentage": rng.uniform(0.1, 0.9),
        "num_request_waiting": rng.randint(0, 5),
        "num_request_running": rng.randint(1, 10),
        "prefix_cache_score": rng.uniform(0.0, 1.0),
        "actual_ttft_ms": rng.uniform(50.0, 500.0),
        "actual_tpot_ms": rng.uniform(5.0, 50.0),
        "pod_type": "decode",
    }


# --- unit tests for the module-level helper ---

def test_drop_timestamp_removes_key():
    rows = [{"timestamp": "2026-01-01", "a": 1}, {"timestamp": "2026-01-02", "b": 2}]
    result = _drop_timestamp(rows)
    assert all("timestamp" not in row for row in result)
    assert result[0]["a"] == 1
    assert result[1]["b"] == 2


def test_drop_timestamp_noop_when_no_timestamp():
    rows = [{"a": 1, "b": 2}]
    result = _drop_timestamp(rows)
    assert result == [{"a": 1, "b": 2}]


def test_drop_timestamp_empty():
    assert _drop_timestamp([]) == []


# --- regression test: train() must not raise AttributeError ---

def test_train_completes_without_attribute_error():
    """
    Before the fix, train() raised:
        AttributeError: 'LatencyPredictor' object has no attribute '_drop_timestamp'
    After the fix it must complete and update last_retrain_time.
    """
    predictor = LatencyPredictor(model_type="xgboost")
    threshold = settings.MIN_SAMPLES_FOR_RETRAIN

    samples = [_make_sample(i) for i in range(threshold + 10)]
    for s in samples:
        predictor.add_training_sample(s)

    # Must not raise AttributeError
    try:
        predictor.train()
    except AttributeError as exc:
        pytest.fail(f"train() raised AttributeError — fix not applied: {exc}")

    assert predictor.last_retrain_time is not None, (
        "last_retrain_time was not updated — train() silently failed"
    )
