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
import math
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from common.types import ModelType, ObjectiveType


@pytest.mark.parametrize("configured_type", list(ModelType))
def test_bayesian_sync_downloads_required_scalers(configured_type, monkeypatch, tmp_path):
    destination = tmp_path / "prediction"
    paths = {
        "ttft": "LOCAL_TTFT_MODEL_PATH",
        "tpot": "LOCAL_TPOT_MODEL_PATH",
        "ttft_scaler": "LOCAL_TTFT_SCALER_PATH",
        "tpot_scaler": "LOCAL_TPOT_SCALER_PATH",
        "ttft_gated": "LOCAL_TTFT_GATED_MODEL_PATH",
        "tpot_gated": "LOCAL_TPOT_GATED_MODEL_PATH",
    }
    for name, setting in paths.items():
        monkeypatch.setenv(setting, str(destination / f"{name}.joblib"))

    from prediction import prediction_server as server

    for name, setting in paths.items():
        monkeypatch.setattr(server.settings, setting, str(destination / f"{name}.joblib"))
    monkeypatch.setattr(server.settings, "MODEL_TYPE", configured_type)
    monkeypatch.setattr(server.settings, "OBJECTIVE_TYPE", ObjectiveType.MEAN)
    monkeypatch.setattr(server.settings, "ENSEMBLE_MODE", False)
    monkeypatch.setattr(server, "XGBOOST_AVAILABLE", False)
    monkeypatch.setattr(server, "LIGHTGBM_AVAILABLE", False)
    predictor = server.LightweightPredictor()
    monkeypatch.setattr(server, "predictor", predictor)
    assert predictor.model_type == ModelType.BAYESIAN_RIDGE

    features = {
        "kv_cache_percentage": 0.2,
        "input_token_length": 20,
        "num_request_waiting": 0,
        "num_request_running": 1,
        "num_tokens_generated": 4,
        "prefix_cache_score": 0.5,
    }
    rows = pd.DataFrame([{**features, "input_token_length": length} for length in range(10, 30)])
    source = tmp_path / "training"
    source.mkdir()
    for name in ("ttft", "tpot"):
        engineered = predictor._prepare_features_with_interaction(rows.copy(), name)
        engineered = engineered.drop(columns=["prefill_score_bucket"], errors="ignore")
        engineered = pd.get_dummies(engineered, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)
        scaler = StandardScaler().fit(engineered)
        model = BayesianRidge().fit(scaler.transform(engineered), np.arange(len(rows)) + 10)
        joblib.dump(model, source / f"{name}.joblib")
        joblib.dump(scaler, source / f"{name}_scaler.joblib")

    class ModelHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            _, _, name, action = self.path.split("/")
            path = source / f"{name}.joblib"
            if not path.exists():
                self.send_error(404)
                return
            if action == "info":
                body = json.dumps(
                    {"last_modified": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()}
                ).encode()
            else:
                body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=http_server.serve_forever)
    thread.start()
    monkeypatch.setattr(server.settings, "TRAINING_SERVER_URL", f"http://127.0.0.1:{http_server.server_port}")
    syncer = server.ModelSyncer()
    syncer._http.trust_env = False
    client = TestClient(server.app)
    try:
        assert client.get("/readyz").status_code == 503
        assert syncer.sync_models()
        assert predictor.load_models()
        assert client.get("/readyz").status_code == 200
        response = client.post("/predict", json=features)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["model_type"] == ModelType.BAYESIAN_RIDGE
        assert math.isfinite(result["ttft_ms"])
        assert math.isfinite(result["tpot_ms"])
        assert server.settings.MODEL_TYPE == configured_type
    finally:
        client.close()
        syncer.shutdown()
        http_server.shutdown()
        http_server.server_close()
        thread.join(timeout=5)
