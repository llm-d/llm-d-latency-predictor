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

import asyncio
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import training.training_server as ts
from common.types import ModelType


def test_lightgbm_export_serves_learned_feature_importances(tmp_path, monkeypatch):
    frame = pd.DataFrame({"new_signal": np.arange(32)})
    model = lgb.LGBMRegressor(n_estimators=2, min_child_samples=1, n_jobs=1, verbosity=-1).fit(
        frame, frame["new_signal"] * 2
    )
    predictor = ts.LatencyPredictor(ModelType.LIGHTGBM)
    predictor.ttft_model = predictor.tpot_model = model
    monkeypatch.setattr(ts, "predictor", predictor)
    for target in ("TTFT", "TPOT"):
        monkeypatch.setattr(ts.settings, f"{target}_MODEL_PATH", str(tmp_path / f"{target}.joblib"))
    predictor._save_models_unlocked()
    for endpoint in (ts.ttft_lgb_importances, ts.tpot_lgb_importances):
        response = asyncio.run(endpoint())
        importance = json.loads(response.body)["new_signal"]
        assert type(importance) is int
        assert importance > 0


def test_ridge_training_exports_fitted_column_names(tmp_path, monkeypatch):
    for target in ("TTFT", "TPOT"):
        for artifact in ("MODEL", "SCALER"):
            monkeypatch.setattr(ts.settings, f"{target}_{artifact}_PATH", str(tmp_path / f"{target}_{artifact}.joblib"))
    for name, value in {
        "MIN_SAMPLES_FOR_RETRAIN": 2,
        "TEST_TRAIN_RATIO": 0,
        "ENSEMBLE_MODE": False,
        "ENABLE_ENCODER_FEATURES": True,
        "ENABLE_TOKEN_IN_FLIGHT_FEATURES": True,
    }.items():
        monkeypatch.setattr(ts.settings, name, value)
    rows = [
        dict(
            kv_cache_percentage=0.4,
            input_token_length=100 + index,
            num_request_waiting=index % 2,
            num_request_running=3,
            num_tokens_generated=1,
            prefix_cache_score=0.5,
            prefill_tokens_in_flight=123,
            decode_tokens_in_flight=456,
            encoder_input_size=9,
            encoder_matched_size=4,
            pod_type="prefill" if index % 2 else "decode",
            actual_ttft_ms=20 + index,
            actual_tpot_ms=5 + index,
        )
        for index in range(8)
    ]
    predictor = ts.LatencyPredictor(ModelType.BAYESIAN_RIDGE)
    predictor.add_training_samples(rows)
    predictor.train()
    assert predictor.last_retrain_time is not None
    metrics = predictor.get_metrics()
    for target in ("ttft", "tpot"):
        coefficients = getattr(predictor, f"{target}_coefficients")
        scaler = getattr(predictor, f"{target}_scaler")
        model = getattr(predictor, f"{target}_model")
        names = list(scaler.feature_names_in_)
        assert set(coefficients) == set(names) | {"intercept"}
        frame = predictor._prepare_features_with_interaction(pd.DataFrame(rows), target)
        frame = frame.drop(columns=["prefill_score_bucket"], errors="ignore")
        frame = pd.get_dummies(frame, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)[names]
        exported = coefficients["intercept"] + sum(frame[name] * coefficients[name] for name in names)
        np.testing.assert_allclose(exported, model.predict(scaler.transform(frame)), atol=1e-10)
        for name in names:
            assert f'{target}_coef{{feature="{name}"}} ' in metrics
