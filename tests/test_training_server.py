# Copyright 2026 The llm-d Authors.
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
import os
from datetime import UTC, datetime

import pytest

import training.training_server as ts
from common.types import ObjectiveType
from training.training_server import LatencyPredictor, TrainingEntry


def _make_sample(idx: int, waiting: int | None = None) -> dict:
    """Create a realistic timestamped training sample via TrainingEntry.model_dump()."""
    w = idx % 4 if waiting is None else waiting
    inp_len = 100 + idx * 20
    kv = 0.2 + (idx % 5) * 0.15
    running = 1 + (idx % 3)
    prefix_cache = (idx % 4) * 0.25
    tokens = 10 + idx
    entry = TrainingEntry(
        kv_cache_percentage=min(kv, 0.95),
        input_token_length=inp_len,
        num_request_waiting=w,
        num_request_running=running,
        actual_ttft_ms=50.0 + inp_len * 0.5 + w * 10.0 - prefix_cache * 20.0,
        actual_tpot_ms=8.0 + running * 2.0 + kv * 5.0,
        num_tokens_generated=tokens,
        prefix_cache_score=prefix_cache,
        pod_type="",
        prefill_tokens_in_flight=inp_len,
        decode_tokens_in_flight=tokens,
        encoder_matched_size=0,
        encoder_input_size=0,
        timestamp=datetime.now(UTC),
    )
    return entry.model_dump()


@pytest.fixture()
def configured_settings(tmp_path, monkeypatch):
    """Redirect model artifacts to tmp_path, lower sample thresholds, and pin env-driven settings for unit tests."""
    # Pin settings that Settings reads from LATENCY_* env vars, so a developer's
    # environment can't change which code paths train() exercises.
    monkeypatch.setattr(ts.settings, "MODEL_TYPE", "xgboost")
    monkeypatch.setattr(ts.settings, "OBJECTIVE_TYPE", ObjectiveType.QUANTILE)
    monkeypatch.setattr(ts.settings, "ENSEMBLE_MODE", True)
    monkeypatch.setattr(ts.settings, "TTFT_MODEL_PATH", str(tmp_path / "ttft.joblib"))
    monkeypatch.setattr(ts.settings, "TPOT_MODEL_PATH", str(tmp_path / "tpot.joblib"))
    monkeypatch.setattr(ts.settings, "TTFT_SCALER_PATH", str(tmp_path / "ttft_scaler.joblib"))
    monkeypatch.setattr(ts.settings, "TPOT_SCALER_PATH", str(tmp_path / "tpot_scaler.joblib"))
    monkeypatch.setattr(ts.settings, "TTFT_GATED_MODEL_PATH", str(tmp_path / "ttft_gated.joblib"))
    monkeypatch.setattr(ts.settings, "TPOT_GATED_MODEL_PATH", str(tmp_path / "tpot_gated.joblib"))
    monkeypatch.setattr(ts.settings, "MIN_SAMPLES_FOR_RETRAIN_FRESH", 5)
    monkeypatch.setattr(ts.settings, "MIN_SAMPLES_FOR_RETRAIN", 5)
    monkeypatch.setattr(ts.settings, "MIN_SAMPLES_FOR_ENSEMBLE_SPLIT", 3)
    return tmp_path


def test_drop_timestamp_staticmethod_and_instance():
    """Verify _drop_timestamp is accessible on both LatencyPredictor and its instances."""
    predictor = LatencyPredictor()
    now = datetime.now(UTC)
    rows = [
        {"kv_cache_percentage": 0.5, "actual_ttft_ms": 42.0, "timestamp": now},
        {"kv_cache_percentage": 0.7, "actual_ttft_ms": 55.0},
    ]

    for fn in (LatencyPredictor._drop_timestamp, predictor._drop_timestamp):
        cleaned = fn(rows)
        assert len(cleaned) == 2
        for orig, stripped in zip(rows, cleaned):
            assert "timestamp" not in stripped
            assert stripped["kv_cache_percentage"] == orig["kv_cache_percentage"]
            assert stripped["actual_ttft_ms"] == orig["actual_ttft_ms"]
        # Ensure original dicts were not mutated in place
        assert "timestamp" in rows[0]


def test_train_and_test_metrics_with_timestamped_samples(configured_settings):
    """Regression test for PR #58: train() and _calculate_metrics_on_test must strip timestamps cleanly."""
    predictor = LatencyPredictor()

    # Populate both noqueue (waiting=0) and queued (waiting>0) training buckets with timestamped samples
    for i in range(6):
        noqueue_sample = _make_sample(i, waiting=0)
        queued_sample = _make_sample(i + 6, waiting=(i % 3) + 1)
        for sample in (noqueue_sample, queued_sample):
            assert "timestamp" in sample
            bucket_key = predictor._get_bucket_key(sample)
            predictor.ttft_data_buckets[bucket_key].append(sample)
            predictor.tpot_data_buckets[bucket_key].append(sample)

    # Populate test deques with timestamped samples
    for i in range(4):
        test_sample = _make_sample(i + 20)
        assert "timestamp" in test_sample
        predictor.ttft_test_data.append(test_sample)
        predictor.tpot_test_data.append(test_sample)

    predictor.train()

    assert predictor.ttft_model is not None, "Expected ttft_model to be trained"
    assert predictor.tpot_model is not None, "Expected tpot_model to be trained"
    assert predictor.last_retrain_time is not None
    assert os.path.exists(ts.settings.TTFT_MODEL_PATH)
    assert os.path.exists(ts.settings.TPOT_MODEL_PATH)

    # Ensemble sub-models also call self._drop_timestamp(samples)
    assert predictor.ensemble_active is True
    assert predictor.ttft_gated is not None
    assert predictor.tpot_gated is not None
    assert os.path.exists(ts.settings.TTFT_GATED_MODEL_PATH)
    assert os.path.exists(ts.settings.TPOT_GATED_MODEL_PATH)

    # Verify _calculate_metrics_on_test executed cleanly during train() and when called directly
    assert len(predictor.ttft_quantile_loss_scores) == 1
    assert len(predictor.tpot_quantile_loss_scores) == 1

    ql, coverage, violation = predictor._calculate_metrics_on_test(
        predictor.ttft_model,
        predictor.ttft_scaler,
        list(predictor.ttft_test_data),
        "ttft",
        "actual_ttft_ms",
    )
    assert ql is not None
    assert coverage is not None
    assert violation is not None
