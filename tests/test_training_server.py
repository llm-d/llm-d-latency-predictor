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
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import training.training_server as ts
from common.protocol import latency_pb2 as pb
from common.protocol.json_codec import from_dict, marshal
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


@pytest.mark.parametrize("case", ["training_legacy", "training_ttft", "training_tpot"])
def test_protocol_preserves_training_inputs(case):
    wire = json.loads((Path(__file__).parent / "testdata/latency-wire.json").read_text())
    payload = dict(wire[case])
    payload.setdefault("timestamp", "2026-11-09T16:24:02Z")
    before = TrainingEntry.model_validate(payload).model_dump()
    message = from_dict(payload, pb.TrainingEntry())
    after = TrainingEntry.model_validate_json(marshal(message)).model_dump()
    assert before == after
    assert "predicted_ttft_ms" not in after
    assert "predicted_tpot_ms" not in after

    predictor = LatencyPredictor()
    for target in ("ttft", "tpot"):
        expected = predictor._prepare_features_with_interaction(
            ts.pd.DataFrame(predictor._drop_timestamp([before])), target
        )
        actual = predictor._prepare_features_with_interaction(
            ts.pd.DataFrame(predictor._drop_timestamp([after])), target
        )
        ts.pd.testing.assert_frame_equal(expected, actual)


def test_protocol_preserves_training_request_validation():
    payload = {"actual_ttft_ms": 10, "actual_tpot_ms": 0}
    message = from_dict(payload, pb.TrainingEntry())
    with pytest.raises(ValidationError):
        TrainingEntry.model_validate_json(marshal(message))


def test_request_fields_come_from_schema():
    from training.training_server import PredictionRequest

    assert set(PredictionRequest.model_fields) == set(pb.Features.DESCRIPTOR.fields_by_name) - {
        "prefill_tokens_in_flight",
        "decode_tokens_in_flight",
    }
    assert set(TrainingEntry.model_fields) == set(pb.Features.DESCRIPTOR.fields_by_name) | {
        "actual_ttft_ms",
        "actual_tpot_ms",
        "timestamp",
    }


def test_training_prediction_preserves_legacy_field_filter():
    from common.protocol.models import request_model
    from training.training_server import PredictionRequest

    ServingRequest = request_model(pb.PredictionRequest, "ServingRequest", __name__)

    row = dict(
        kv_cache_percentage=0.4,
        input_token_length=100,
        num_request_waiting=0,
        num_request_running=1,
        num_tokens_generated=1,
        prefix_cache_score=0.5,
        prefill_tokens_in_flight=17,
        decode_tokens_in_flight=23,
    )
    training_prediction = PredictionRequest(**row).model_dump()
    assert "prefill_tokens_in_flight" not in training_prediction
    assert "decode_tokens_in_flight" not in training_prediction
    serving_prediction = ServingRequest(**row)
    assert serving_prediction.prefill_tokens_in_flight == 17
    assert serving_prediction.decode_tokens_in_flight == 23


@pytest.mark.parametrize("encoder", [False, True])
@pytest.mark.parametrize("tokens", [False, True])
def test_feature_plan_preserves_model_column_order(encoder, tokens):
    from types import SimpleNamespace

    from common.feature_plan import feature_columns

    flags = SimpleNamespace(ENABLE_ENCODER_FEATURES=encoder, ENABLE_TOKEN_IN_FLIGHT_FEATURES=tokens)
    common = ["is_queued", "kv_cache_percentage", "input_token_length", "num_request_waiting", "num_request_running"]
    if tokens:
        common += ["prefill_tokens_in_flight", "decode_tokens_in_flight"]
    expected_ttft = (
        common
        + (["encoder_matched_size", "encoder_input_size"] if encoder else [])
        + ["prefix_cache_score", "effective_input_tokens", "prefill_score_bucket", "pod_type_cat"]
    )
    assert feature_columns("ttft", flags) == expected_ttft
    assert feature_columns("tpot", flags) == common + ["num_tokens_generated", "pod_type_cat"]
    assert feature_columns("ttft", flags, noqueue=True) == [
        name for name in expected_ttft if name not in {"is_queued", "num_request_waiting"}
    ]


def test_mixed_historical_rows_are_filled_before_dropping_missing_data():
    from common.feature_plan import training_frame

    old = _make_sample(1)
    current = _make_sample(2)
    for name in ("encoder_input_size", "encoder_matched_size", "prefill_tokens_in_flight", "decode_tokens_in_flight"):
        old.pop(name)
    current["predicted_ttft_ms"] = None
    frame = training_frame([old, current], "ttft", ts.settings)
    assert len(frame) == 2
    assert frame["encoder_input_size"].tolist() == [0, 0]
    assert "predicted_ttft_ms" not in frame
    assert "encoder_input_size" not in old


@pytest.mark.parametrize("target", ["ttft", "tpot"])
def test_named_legacy_artifacts_accept_current_feature_plan(tmp_path, target):
    from types import SimpleNamespace

    import joblib
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler

    from common.feature_plan import align_features, prepare_features

    rows = ts.pd.DataFrame([_make_sample(index) for index in range(8)])
    old_flags = SimpleNamespace(ENABLE_ENCODER_FEATURES=False, ENABLE_TOKEN_IN_FLIGHT_FEATURES=False)
    old = prepare_features(rows.copy(), target, old_flags).drop(columns=["prefill_score_bucket"], errors="ignore")
    old = ts.pd.get_dummies(old, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)
    scaler = StandardScaler().fit(old)
    model = BayesianRidge().fit(scaler.transform(old), rows[f"actual_{target}_ms"])
    path = tmp_path / "legacy.joblib"
    joblib.dump((model, scaler), path)
    model, scaler = joblib.load(path)
    expected = model.predict(scaler.transform(old))
    current = prepare_features(rows.copy(), target, ts.settings).drop(columns=["prefill_score_bucket"], errors="ignore")
    current = ts.pd.get_dummies(current, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)
    actual = model.predict(scaler.transform(align_features(current, model, scaler)))
    ts.np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)


@pytest.fixture()
def configured_settings(tmp_path, monkeypatch):
    """Redirect model artifacts to tmp_path and lower sample thresholds for unit tests."""
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
    if ts.settings.ENSEMBLE_MODE:
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
