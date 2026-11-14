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

"""Train and reload a schema-only feature through the real Go/Python HTTP paths."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--go", required=True)
    parser.add_argument("--engine", choices=("bayesian_ridge", "xgboost", "lightgbm"), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with tempfile.TemporaryDirectory(prefix="latency-extension-models-") as directory:
        os.environ.update(
            LATENCY_MODEL_TYPE=args.engine,
            LATENCY_OBJECTIVE_TYPE="mean",
            LATENCY_TEST_TRAIN_RATIO="0",
            LATENCY_MIN_SAMPLES_FOR_RETRAIN="8",
            LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH="8",
            LATENCY_ENSEMBLE_MODE="false",
            LATENCY_ENABLE_ENCODER_FEATURES="false",
            LATENCY_ENABLE_TOKEN_IN_FLIGHT_FEATURES="false",
        )
        names = ("TTFT_MODEL", "TPOT_MODEL", "TTFT_SCALER", "TPOT_SCALER", "TTFT_GATED_MODEL", "TPOT_GATED_MODEL")
        for name in names:
            for prefix in ("LATENCY", "LOCAL"):
                os.environ[f"{prefix}_{name}_PATH"] = str(Path(directory) / f"{name}.joblib")

        import uvicorn
        from fastapi import FastAPI

        import prediction.prediction_server as ps
        import training.training_server as ts
        from common.protocol import latency_pb2 as pb

        feature = "extension_test_size"
        assert feature in pb.Features.DESCRIPTOR.fields_by_name, "run through test-feature-extension.py"
        ts.predictor.load_models()
        assert ps.predictor.load_models()
        paths = [Path(ts.settings.TTFT_MODEL_PATH), Path(ts.settings.TPOT_MODEL_PATH)]
        initial = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
        app = FastAPI()
        app.mount("/training", ts.app)
        app.mount("/prediction", ps.app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        env = dict(os.environ, TRAINING_SERVER_URL=url + "/training", PREDICTION_SERVER_URL=url + "/prediction")
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning"))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        worker.start()
        try:
            for _ in range(200):
                if server.started:
                    break
                assert worker.is_alive(), "HTTP server exited before startup"
                time.sleep(0.05)
            else:
                raise RuntimeError("HTTP server startup timeout")

            def go_test(name):
                subprocess.run(
                    [
                        args.go,
                        "test",
                        "-p",
                        "1",
                        "-race",
                        "-count=1",
                        "-v",
                        "-run",
                        f"^{name}$",
                        "./pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient",
                    ],
                    cwd=args.router_dir,
                    env=env,
                    check=True,
                    timeout=90,
                )

            go_test("TestFeatureExtensionTrainingIngest")
            for buckets in (ts.predictor.ttft_data_buckets, ts.predictor.tpot_data_buckets):
                assert sum(map(len, buckets.values())) == 128, "Go training rows were lost"
            ts.predictor.train()
            assert ts.predictor.last_retrain_time is not None
            for path, prior in zip(paths, initial):
                assert hashlib.sha256(path.read_bytes()).digest() != prior, f"{path.name} was not retrained"
                if args.engine == "lightgbm":
                    exported = json.loads(path.with_name(path.stem + "_importances.json").read_text())
                    assert exported[feature] > 0
            for target in ("ttft", "tpot"):
                model = getattr(ts.predictor, f"{target}_model")
                owner = getattr(ts.predictor, f"{target}_scaler") if args.engine == "bayesian_ridge" else model
                assert feature in owner.feature_names_in_
                kind = "coef" if args.engine == "bayesian_ridge" else "importance"
                prefix = f'{target}_{kind}{{feature="{feature}"}} '
                line = next(line for line in ts.predictor.get_metrics().splitlines() if line.startswith(prefix))
                assert float(line.split()[-1]) > 0, "new feature has no learned contribution"
            assert ps.predictor.load_models(), "persisted artifacts did not reload"
            go_test("TestFeatureExtensionModelRoundTrip")
            print(f"{args.engine}: Go ingestion, training, artifact export/reload and prediction passed", flush=True)
        finally:
            server.should_exit = True
            worker.join(timeout=5)
            sock.close()
            assert not worker.is_alive(), "HTTP test server did not stop"


if __name__ == "__main__":
    main()
