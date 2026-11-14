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

import io
import json
import tarfile
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

import training.training_server as ts
from common.feature_plan import align_features, prepare_features, raw_frames, validate_model_features
from common.protocol import latency_pb2 as pb
from common.protocol.models import request_model
from common.types import ModelType, QueueGatedModel


@pytest.fixture()
def ps(tmp_path, monkeypatch):
    for name in ("TTFT_MODEL", "TPOT_MODEL", "TTFT_SCALER", "TPOT_SCALER", "TTFT_GATED_MODEL", "TPOT_GATED_MODEL"):
        monkeypatch.setenv(f"LOCAL_{name}_PATH", str(tmp_path / f"{name}.joblib"))
    from prediction import prediction_server

    for name in ("TTFT_MODEL", "TPOT_MODEL", "TTFT_SCALER", "TPOT_SCALER", "TTFT_GATED_MODEL", "TPOT_GATED_MODEL"):
        monkeypatch.setattr(prediction_server.settings, f"LOCAL_{name}_PATH", str(tmp_path / f"{name}.joblib"))
    return prediction_server


def sample(waiting=0):
    return dict(
        kv_cache_percentage=0.4,
        input_token_length=100,
        num_request_waiting=waiting,
        num_request_running=3,
        num_tokens_generated=1,
        prefix_cache_score=0.5,
        prefill_tokens_in_flight=123,
        decode_tokens_in_flight=456,
        encoder_input_size=9,
        encoder_matched_size=4,
        pod_type="prefill",
    )


class FeatureRegressor:
    def __init__(self):
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        return frame["prefill_tokens_in_flight"].to_numpy() + frame["decode_tokens_in_flight"].to_numpy()


@pytest.mark.parametrize("ensemble", [False, True])
def test_plain_and_fast_paths_preserve_their_inflight_feature_behavior(monkeypatch, ensemble, ps):
    monkeypatch.setattr(ps.settings, "ENABLE_ENCODER_FEATURES", True)
    monkeypatch.setattr(ps.settings, "ENABLE_TOKEN_IN_FLIGHT_FEATURES", True)
    predictor = ps.LightweightPredictor()
    predictor.model_type = ModelType.XGBOOST
    predictor.ttft_model = FeatureRegressor()
    predictor.tpot_model = FeatureRegressor()
    predictor.ensemble_active = ensemble
    predictor.ttft_gated = QueueGatedModel(FeatureRegressor(), FeatureRegressor())
    predictor.tpot_gated = QueueGatedModel(FeatureRegressor(), FeatureRegressor())
    rows = [sample(0), sample(2)]
    single = np.asarray([predictor.predict(row) for row in rows]).T
    plain = predictor.predict_batch(rows)
    fast = predictor.predict_batch_fast([ps.PredictionRequest(**row) for row in rows])
    np.testing.assert_allclose(single, plain)
    np.testing.assert_allclose(single, [[0, 0], [0, 0]])
    np.testing.assert_allclose(fast, [[579, 579], [579, 579]])
    models = (
        [predictor.ttft_model]
        if not ensemble
        else [predictor.ttft_gated.noqueue_model, predictor.ttft_gated.queued_model]
    )
    for model in models:
        for frame in model.frames:
            assert (frame["encoder_input_size"] == 9).all()
            assert (frame["encoder_matched_size"] == 4).all()
            assert (frame["pod_type_cat"] == "prefill").all()


@pytest.mark.parametrize("target", ["ttft", "tpot"])
@pytest.mark.parametrize("engine", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("noqueue", [False, True])
def test_named_tree_artifacts_keep_predictions_after_optional_columns_are_added(tmp_path, target, engine, noqueue, ps):
    flags = SimpleNamespace(ENABLE_ENCODER_FEATURES=False, ENABLE_TOKEN_IN_FLIGHT_FEATURES=False)
    rows = [dict(sample(index % 3), input_token_length=100 + index) for index in range(12)]
    frame = prepare_features(pd.DataFrame(rows), target, flags)
    if noqueue:
        frame = frame.drop(columns=["is_queued", "num_request_waiting"])
    if engine == "xgboost":
        model = ps.xgb.XGBRegressor(n_estimators=2, max_depth=2, enable_categorical=True, n_jobs=1)
    else:
        model = ps.lgb.LGBMRegressor(n_estimators=2, min_child_samples=1, verbosity=-1, n_jobs=1)
    model.fit(frame, np.arange(len(rows), dtype=float))
    path = tmp_path / "legacy.joblib"
    joblib.dump(model, path)
    model = joblib.load(path)
    expected = model.predict(frame)
    current = prepare_features(pd.DataFrame(rows), target, ps.settings)
    np.testing.assert_allclose(model.predict(align_features(current, model)), expected)
    validate_model_features(model, target, ps.settings, noqueue=noqueue)


@pytest.mark.parametrize("old_features", [False, True])
@pytest.mark.parametrize("ensemble", [False, True])
def test_ridge_artifacts_survive_feature_flag_changes(monkeypatch, old_features, ensemble, ps):
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler

    from common.types import ObjectiveType

    old_settings = SimpleNamespace(ENABLE_ENCODER_FEATURES=old_features, ENABLE_TOKEN_IN_FLIGHT_FEATURES=old_features)
    rows = [
        dict(
            sample(index % 2),
            input_token_length=100 + index * 7,
            prefill_tokens_in_flight=index * 13,
            decode_tokens_in_flight=index * 5,
            encoder_input_size=index * 3,
            encoder_matched_size=index,
        )
        for index in range(16)
    ]

    def frame_for(target, frame_rows, noqueue=False):
        frame = prepare_features(pd.DataFrame(frame_rows), target, old_settings)
        if noqueue:
            frame = frame.drop(columns=["is_queued", "num_request_waiting"])
        frame = frame.drop(columns=["prefill_score_bucket"], errors="ignore")
        return pd.get_dummies(frame, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)

    models = {}
    for target in ("ttft", "tpot"):
        for regime in ("single", "noqueue", "queued"):
            selected = (
                rows
                if regime == "single"
                else [row for row in rows if (row["num_request_waiting"] == 0) == (regime == "noqueue")]
            )
            frame = frame_for(target, selected, noqueue=regime == "noqueue")
            scaler = StandardScaler().fit(frame)
            values = np.asarray([row["input_token_length"] + row["prefill_tokens_in_flight"] for row in selected])
            model = BayesianRidge().fit(scaler.transform(frame), values)
            models[target, regime] = model, scaler
        model, scaler = models[target, "single"]
        joblib.dump(model, getattr(ps.settings, f"LOCAL_{target.upper()}_MODEL_PATH"))
        joblib.dump(scaler, getattr(ps.settings, f"LOCAL_{target.upper()}_SCALER_PATH"))
        noqueue_model, noqueue_scaler = models[target, "noqueue"]
        queued_model, queued_scaler = models[target, "queued"]
        joblib.dump(
            QueueGatedModel(noqueue_model, queued_model, noqueue_scaler, queued_scaler),
            getattr(ps.settings, f"LOCAL_{target.upper()}_GATED_MODEL_PATH"),
        )

    monkeypatch.setattr(ps.settings, "MODEL_TYPE", ModelType.BAYESIAN_RIDGE)
    monkeypatch.setattr(ps.settings, "OBJECTIVE_TYPE", ObjectiveType.MEAN)
    monkeypatch.setattr(ps.settings, "ENSEMBLE_MODE", ensemble)
    monkeypatch.setattr(ps.settings, "ENABLE_ENCODER_FEATURES", not old_features)
    monkeypatch.setattr(ps.settings, "ENABLE_TOKEN_IN_FLIGHT_FEATURES", not old_features)
    predictor = ps.LightweightPredictor()
    assert predictor.load_models()
    assert predictor.ensemble_active == ensemble

    requests = [rows[2], rows[3]]
    expected_rows = [dict(row) for row in requests]
    if old_features:
        for row in expected_rows:
            row.update(
                prefill_tokens_in_flight=0, decode_tokens_in_flight=0, encoder_input_size=0, encoder_matched_size=0
            )
    expected = []
    for row in expected_rows:
        predictions = []
        for target in ("ttft", "tpot"):
            regime = ("noqueue" if row["num_request_waiting"] == 0 else "queued") if ensemble else "single"
            model, scaler = models[target, regime]
            frame = frame_for(target, [row], noqueue=regime == "noqueue")
            predictions.append(model.predict(scaler.transform(frame))[0])
        expected.append(predictions)
    expected = np.asarray(expected).T
    np.testing.assert_allclose(np.asarray([predictor.predict(row) for row in requests]).T, expected)
    np.testing.assert_allclose(predictor.predict_batch(requests), expected)
    np.testing.assert_allclose(
        predictor.predict_batch_fast([ps.PredictionRequest(**row) for row in requests]), expected
    )


def test_reload_keeps_usable_models_when_new_artifact_requires_unknown_features(tmp_path, monkeypatch, ps):
    for name in ("TTFT_MODEL", "TPOT_MODEL", "TTFT_SCALER", "TPOT_SCALER"):
        monkeypatch.setattr(ps.settings, f"LOCAL_{name}_PATH", str(tmp_path / f"{name}.joblib"))
    monkeypatch.setattr(ps.settings, "ENSEMBLE_MODE", False)
    predictor = ps.LightweightPredictor()
    predictor.model_type = ModelType.XGBOOST
    old = FeatureRegressor()
    predictor.ttft_model = predictor.tpot_model = old
    for path in (ps.settings.LOCAL_TTFT_MODEL_PATH, ps.settings.LOCAL_TPOT_MODEL_PATH):
        incompatible = FeatureRegressor()
        incompatible.feature_names_in_ = np.asarray(["future_unsupported_feature"])
        joblib.dump(incompatible, path)
    assert predictor.load_models() is False
    assert predictor.ttft_model is old
    assert predictor.tpot_model is old
    assert predictor._loaded_checksums == {}


def ridge_pair(target, flags, noqueue=False):
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler

    rows = [dict(sample(index % 2), input_token_length=100 + index * 7) for index in range(6)]
    frame = prepare_features(pd.DataFrame(rows), target, flags)
    if noqueue:
        frame = frame.drop(columns=["is_queued", "num_request_waiting"])
    frame = frame.drop(columns=["prefill_score_bucket"], errors="ignore")
    frame = pd.get_dummies(frame, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)
    scaler = StandardScaler().fit(frame)
    model = BayesianRidge().fit(scaler.transform(frame), np.arange(len(rows)))
    return model, scaler


@pytest.mark.parametrize("target", ["ttft", "tpot"])
@pytest.mark.parametrize("noqueue", [False, True])
@pytest.mark.parametrize("mismatch", ["estimator", "coefficients"])
def test_ridge_pair_dimensions_are_checked_before_activation(target, noqueue, mismatch):
    flags = SimpleNamespace(ENABLE_ENCODER_FEATURES=False, ENABLE_TOKEN_IN_FLIGHT_FEATURES=False)
    model, scaler = ridge_pair(target, flags, noqueue)
    if mismatch == "estimator":
        from sklearn.linear_model import BayesianRidge

        model = BayesianRidge().fit(np.arange(12).reshape(6, 2), np.arange(6))
    else:
        model.coef_ = model.coef_[:-1]
    with pytest.raises(ValueError, match="dimension"):
        validate_model_features(model, target, flags, scaler, noqueue=noqueue)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("target", ["ttft", "tpot"])
def test_ridge_dimension_failure_retains_serving_state(tmp_path, monkeypatch, ps, warm, target):
    from sklearn.linear_model import BayesianRidge

    monkeypatch.setattr(ps.settings, "ENSEMBLE_MODE", False)
    monkeypatch.setattr(ps.settings, "ENABLE_ENCODER_FEATURES", False)
    monkeypatch.setattr(ps.settings, "ENABLE_TOKEN_IN_FLIGHT_FEATURES", False)
    predictor = ps.LightweightPredictor()
    predictor.model_type = ModelType.BAYESIAN_RIDGE
    for name in ("ttft", "tpot"):
        model, scaler = ridge_pair(name, ps.settings)
        joblib.dump(model, getattr(ps.settings, f"LOCAL_{name.upper()}_MODEL_PATH"))
        joblib.dump(scaler, getattr(ps.settings, f"LOCAL_{name.upper()}_SCALER_PATH"))
    if warm:
        assert predictor.load_models()
    old_refs = (predictor.ttft_model, predictor.tpot_model, predictor.ttft_scaler, predictor.tpot_scaler)
    old_checksums = predictor._loaded_checksums.copy()
    old_time = predictor.last_load
    expected = predictor.predict(sample()) if warm else None
    incompatible = BayesianRidge().fit(np.arange(12).reshape(6, 2), np.arange(6))
    joblib.dump(incompatible, getattr(ps.settings, f"LOCAL_{target.upper()}_MODEL_PATH"))
    assert predictor.load_models() is False
    assert all(
        a is b
        for a, b in zip(
            old_refs, (predictor.ttft_model, predictor.tpot_model, predictor.ttft_scaler, predictor.tpot_scaler)
        )
    )
    assert predictor._loaded_checksums == old_checksums
    assert predictor.last_load == old_time
    assert predictor.is_ready == warm
    if warm:
        np.testing.assert_array_equal(predictor.predict(sample()), expected)
    else:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as error:
            predictor.predict(sample())
        assert error.value.status_code == 503


@pytest.mark.parametrize("target", ["ttft", "tpot"])
@pytest.mark.parametrize("regime", ["noqueue", "queued"])
def test_bad_ridge_gated_pair_falls_back_to_valid_single_models(tmp_path, monkeypatch, ps, target, regime):
    from sklearn.linear_model import BayesianRidge

    monkeypatch.setattr(ps.settings, "ENSEMBLE_MODE", False)
    predictor = ps.LightweightPredictor()
    predictor.model_type = ModelType.BAYESIAN_RIDGE
    for name in ("ttft", "tpot"):
        model, scaler = ridge_pair(name, ps.settings)
        joblib.dump(model, getattr(ps.settings, f"LOCAL_{name.upper()}_MODEL_PATH"))
        joblib.dump(scaler, getattr(ps.settings, f"LOCAL_{name.upper()}_SCALER_PATH"))
        noqueue_model, noqueue_scaler = ridge_pair(name, ps.settings, noqueue=True)
        joblib.dump(
            QueueGatedModel(noqueue_model, model, noqueue_scaler, scaler),
            getattr(ps.settings, f"LOCAL_{name.upper()}_GATED_MODEL_PATH"),
        )
    assert predictor.load_models()
    expected = predictor.predict(sample())
    path = getattr(ps.settings, f"LOCAL_{target.upper()}_GATED_MODEL_PATH")
    gated = joblib.load(path)
    incompatible = BayesianRidge().fit(np.arange(12).reshape(6, 2), np.arange(6))
    setattr(gated, f"{regime}_model", incompatible)
    joblib.dump(gated, path)
    monkeypatch.setattr(ps.settings, "ENSEMBLE_MODE", True)
    assert predictor.load_models()
    assert predictor.is_ready
    assert not predictor.ensemble_active
    np.testing.assert_array_equal(predictor.predict(sample()), expected)


def test_schema_only_optional_feature_enters_validation_training_and_prediction(monkeypatch):
    # Extend a descriptor in memory; application code has no knowledge of this field.
    file = descriptor_pb2.FileDescriptorProto(
        name="test_feature_extension.proto", package="test.latency", syntax="proto3"
    )
    file.dependency.append(pb.DESCRIPTOR.name)
    features = file.message_type.add(name="Features")
    pb.Features.DESCRIPTOR.CopyToProto(features)
    field = features.field.add(name="test_extra_size", number=99, type=descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    field.options.Extensions[pb.field_rules].CopyFrom(
        pb.FieldRules(minimum=0, default_json="7", ttft_order=85, tpot_order=75)
    )
    descriptor = descriptor_pool.Default().AddSerializedFile(file.SerializeToString())
    extended = message_factory.GetMessageClass(descriptor.message_types_by_name["Features"])
    model = request_model(extended, "ExtendedFeatures", __name__)
    assert model(**sample()).test_extra_size == 7
    with pytest.raises(ValueError):
        model(**sample(), test_extra_size=-1)
    from common import feature_plan

    monkeypatch.setattr(pb, "Features", extended)
    feature_plan._fields.cache_clear()
    try:
        frames = raw_frames([model(**sample(), test_extra_size=17).model_dump()], ts.settings)
        for target, frame in zip(("ttft", "tpot"), frames):
            assert prepare_features(frame, target, ts.settings)["test_extra_size"].iloc[0] == 17
        historical = feature_plan.training_frame([dict(sample(), actual_ttft_ms=5)], "ttft", ts.settings)
        assert historical["test_extra_size"].iloc[0] == 7
    finally:
        feature_plan._fields.cache_clear()


def test_descaled_coefficients_and_metrics_use_fitted_column_names():
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler

    rows = pd.DataFrame([dict(sample(index % 2), input_token_length=100 + index) for index in range(8)])
    frame = prepare_features(rows.copy(), "tpot", ts.settings)
    frame = pd.get_dummies(frame, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)
    scaler = StandardScaler().fit(frame)
    model = BayesianRidge().fit(scaler.transform(frame), np.arange(8, dtype=float))
    predictor = ts.LatencyPredictor(ModelType.BAYESIAN_RIDGE)
    coefficients = predictor._store_descaled_coefficients(model, scaler, list(scaler.feature_names_in_), "TPOT")
    actual = sum(frame[name] * coefficients[name] for name in frame) + coefficients["intercept"]
    np.testing.assert_allclose(actual, model.predict(scaler.transform(frame)), atol=1e-10)
    predictor.tpot_model = model
    predictor.tpot_coefficients = coefficients
    metrics = predictor.get_metrics()
    assert 'tpot_coef{feature="prefill_tokens_in_flight"}' in metrics
    assert 'tpot_coef{feature="pod_type_prefill"}' in metrics
    assert "tpot_intercept{}" in metrics


def test_export_metadata_uses_shared_protocol(tmp_path, monkeypatch):
    predictor = ts.LatencyPredictor(ModelType.XGBOOST)
    predictor.ttft_model = predictor.tpot_model = FeatureRegressor()
    monkeypatch.setattr(ts, "predictor", predictor)
    monkeypatch.setattr(ts, "_get_model_paths", lambda: {})
    response = ts.export_models()

    async def read():
        return b"".join([chunk async for chunk in response.body_iterator])

    import asyncio

    with tarfile.open(fileobj=io.BytesIO(asyncio.run(read())), mode="r:gz") as archive:
        metadata = json.load(archive.extractfile("metadata.json"))
    assert metadata["model_type"] == "xgboost"
    assert metadata["ttft_samples"] == 0
    assert metadata["tpot_samples"] == 0
    from common.protocol.json_codec import from_dict, to_dict

    assert to_dict(from_dict(metadata, pb.ModelExportMetadata())) == metadata
