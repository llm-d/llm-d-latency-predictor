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
import tempfile

import numpy as np
import pytest

# Importing the module builds a ModelSyncer that creates the model directories,
# so point them at a temp dir before the import.
_MODEL_DIR = tempfile.mkdtemp(prefix="latency-predictor-test-")
for _var, _name in (
    ("LOCAL_TTFT_MODEL_PATH", "ttft.joblib"),
    ("LOCAL_TPOT_MODEL_PATH", "tpot.joblib"),
    ("LOCAL_TTFT_SCALER_PATH", "ttft_scaler.joblib"),
    ("LOCAL_TPOT_SCALER_PATH", "tpot_scaler.joblib"),
    ("LOCAL_TTFT_GATED_MODEL_PATH", "ttft_gated.joblib"),
    ("LOCAL_TPOT_GATED_MODEL_PATH", "tpot_gated.joblib"),
):
    os.environ.setdefault(_var, os.path.join(_MODEL_DIR, _name))

import prediction.prediction_server as ps  # noqa: E402
from common.types import ModelType, QueueGatedModel  # noqa: E402


class _InFlightEchoModel:
    """Stub tree model whose prediction is the sum of the token-in-flight columns."""

    def predict(self, df):
        return (df["prefill_tokens_in_flight"] + df["decode_tokens_in_flight"]).to_numpy(dtype=float)


REQUESTS = [
    {
        "kv_cache_percentage": 0.5,
        "input_token_length": 200,
        "num_request_waiting": 4,
        "num_request_running": 1,
        "num_tokens_generated": 4,
        "prefix_cache_score": 0.7,
        "prefill_tokens_in_flight": 5000,
        "decode_tokens_in_flight": 1200,
    },
    {
        "kv_cache_percentage": 0.3,
        "input_token_length": 150,
        "num_request_waiting": 0,
        "num_request_running": 2,
        "num_tokens_generated": 5,
        "prefix_cache_score": 0.5,
        "prefill_tokens_in_flight": 300,
        "decode_tokens_in_flight": 40,
    },
]
EXPECTED = [6200.0, 340.0]


@pytest.fixture(params=[False, True], ids=["single-model", "ensemble"])
def stub_predictor(request, monkeypatch):
    monkeypatch.setattr(ps.settings, "ENABLE_TOKEN_IN_FLIGHT_FEATURES", True)
    predictor = ps.LightweightPredictor()
    predictor.model_type = ModelType.XGBOOST
    predictor.ttft_model = _InFlightEchoModel()
    predictor.tpot_model = _InFlightEchoModel()
    if request.param:
        predictor.ttft_gated = QueueGatedModel(_InFlightEchoModel(), _InFlightEchoModel())
        predictor.tpot_gated = QueueGatedModel(_InFlightEchoModel(), _InFlightEchoModel())
        predictor.ensemble_active = True
    return predictor


def test_predict_uses_token_in_flight_features(stub_predictor):
    for req, expected in zip(REQUESTS, EXPECTED):
        ttft, tpot = stub_predictor.predict(dict(req))
        assert ttft == pytest.approx(expected)
        assert tpot == pytest.approx(expected)


def test_predict_batch_uses_token_in_flight_features(stub_predictor):
    ttft, tpot = stub_predictor.predict_batch([dict(r) for r in REQUESTS])
    np.testing.assert_allclose(ttft, EXPECTED)
    np.testing.assert_allclose(tpot, EXPECTED)


def test_all_predict_paths_agree(stub_predictor):
    reqs = [ps.PredictionRequest(**r) for r in REQUESTS]
    fast_ttft, fast_tpot = stub_predictor.predict_batch_fast(reqs)
    batch_ttft, batch_tpot = stub_predictor.predict_batch([r.model_dump() for r in reqs])
    single = [stub_predictor.predict(r.model_dump()) for r in reqs]
    np.testing.assert_allclose(batch_ttft, fast_ttft)
    np.testing.assert_allclose(batch_tpot, fast_tpot)
    np.testing.assert_allclose([s[0] for s in single], fast_ttft)
    np.testing.assert_allclose([s[1] for s in single], fast_tpot)
