#!/usr/bin/env python3
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
"""Train or serve one source tree for the prediction comparison driver."""

import hashlib
import json
import os
import random
import sys
from pathlib import Path


def main():
    source, mode, artifact, payload, result = sys.argv[1:]
    sys.path.insert(0, source)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    artifact = Path(artifact)
    artifact.mkdir(parents=True, exist_ok=True)
    if mode == "train":
        # Each retraining comparison starts from priors, never a previous test's fitted files.
        if any(artifact.iterdir()):
            raise ValueError("Training requires an empty artifact directory")
    data = json.loads(Path(payload).read_text())
    for prefix in ("LATENCY", "LOCAL"):
        for name in ("TTFT_MODEL", "TPOT_MODEL", "TTFT_SCALER", "TPOT_SCALER", "TTFT_GATED_MODEL", "TPOT_GATED_MODEL"):
            os.environ[f"{prefix}_{name}_PATH"] = str(artifact / (name + ".joblib"))

    import numpy as np
    import pandas as pd
    from fastapi.testclient import TestClient

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest()

    output = {}
    if mode == "train":
        import training.training_server as ts

        random.seed(913)
        np.random.seed(913)
        ts.predictor.load_models()
        before = {
            name: hashlib.sha256((artifact / (name + ".joblib")).read_bytes()).hexdigest()
            for name in ("TTFT_MODEL", "TPOT_MODEL")
        }
        client = TestClient(ts.app)
        response = client.post("/add_training_data_bulk", json={"entries": data["rows"]})
        assert response.status_code == 202, response.text
        retained = {}
        for target in ("ttft", "tpot"):
            buckets = getattr(ts.predictor, target + "_data_buckets")
            retained[target] = [(str(key), list(values)) for key, values in sorted(buckets.items())]
        output["retained_digest"] = digest(retained)
        output["counts"] = {target: sum(len(values) for _, values in retained[target]) for target in ("ttft", "tpot")}
        frames = {}
        for target in ("ttft", "tpot"):
            raw = pd.DataFrame(data["queries"])
            for regime in ("single", "noqueue", "queued"):
                frame = (
                    ts.predictor._prepare_features_with_interaction(raw.copy(), target)
                    if regime == "single"
                    else ts.predictor._prepare_features_for_ensemble(raw.copy(), target, regime)
                )
                frames[target + "/" + regime] = {
                    "columns": list(frame.columns),
                    "dtypes": list(map(str, frame.dtypes)),
                    "categories": {
                        name: {"values": list(frame[name].cat.categories), "ordered": frame[name].cat.ordered}
                        for name in frame.select_dtypes(include="category").columns
                    },
                    "values": json.loads(frame.to_json(orient="split", double_precision=15)),
                }
        output["frames"] = frames
        ts.predictor.train()
        assert ts.predictor.last_retrain_time is not None, "Training did not complete"
        output["ensemble_active"] = ts.predictor.ensemble_active
        if os.environ["LATENCY_ENSEMBLE_MODE"] == "true":
            assert output["ensemble_active"]
        output["artifacts_changed"] = {
            name: hashlib.sha256((artifact / (name + ".joblib")).read_bytes()).hexdigest() != old
            for name, old in before.items()
        }
        assert all(output["artifacts_changed"].values()), output
    else:
        import prediction.prediction_server as ps

        assert ps.predictor.load_models(), "Candidate artifact admission failed"
        client = TestClient(ps.app)
        rows = data["queries"]
        output["single"] = []
        for row in rows:
            response = client.post("/predict", json=row)
            assert response.status_code == 200, response.text
            value = response.json()
            output["single"].append([value["ttft_ms"], value["tpot_ms"]])
        for route in ("/predict/bulk", "/predict/bulk/strict"):
            response = client.post(route, json={"requests": rows})
            assert response.status_code == 200, response.text
            value = response.json()
            assert value["failed_predictions"] == 0, value
            output[route] = [[p["ttft_ms"], p["tpot_ms"]] for p in value["predictions"]]
        output["validation"] = {}
        row = rows[0]
        cases = {
            "missing": {k: v for k, v in row.items() if k != "input_token_length"},
            "numeric_null": dict(row, encoder_input_size=None),
            "fractional_int": dict(row, input_token_length=1.5),
            "nullable_pod": dict(row, pod_type=None),
            "unknown": dict(row, future_feature=17),
            "nested_conflict": dict(row, features={"input_token_length": 99999}),
        }
        for name, body in cases.items():
            response = client.post("/predict", json=body)
            value = response.json()
            output["validation"][name] = {
                "status": response.status_code,
                "body": value
                if response.status_code != 200
                else {k: value[k] for k in ("ttft_ms", "tpot_ms", "model_type", "objective_type", "quantile")},
            }
        output["ensemble_active"] = ps.predictor.ensemble_active
    Path(result).write_text(json.dumps(output, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
