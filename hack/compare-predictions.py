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
"""Compare two Predictor checkouts using fixed artifacts and controlled CPU retraining."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def source_record(root):
    paths = sorted(path for package in ("common", "training", "prediction") for path in (root / package).rglob("*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def payload():
    import numpy as np

    rng = np.random.default_rng(913)
    rows = []
    for index in range(512):
        tokens = int(rng.integers(10, 2000))
        queue = 0 if index % 2 == 0 else 3
        prefill = int(rng.integers(0, 10001))
        decode = int(rng.integers(0, 3001))
        rows.append(
            dict(
                kv_cache_percentage=0.1 if index % 3 == 0 else 0.5,
                input_token_length=tokens,
                num_request_waiting=queue,
                num_request_running=index % 5,
                num_tokens_generated=1 + index % 7,
                prefix_cache_score=(index % 3) / 2,
                encoder_input_size=index % 17,
                encoder_matched_size=index % 17 // 2,
                pod_type=["", "prefill", "decode"][index % 3],
                prefill_tokens_in_flight=prefill,
                decode_tokens_in_flight=decode,
                actual_ttft_ms=float(100 + tokens * 0.3 + queue * 4 + prefill * 0.04 + rng.normal(0, 5)),
                actual_tpot_ms=float(20 + decode * 0.02 + index % 5 + rng.normal(0, 2)),
                timestamp=f"2025-01-01T00:{index // 60:02d}:{index % 60:02d}Z",
            )
        )
    queries = [
        {key: value for key, value in row.items() if key not in ("actual_ttft_ms", "actual_tpot_ms", "timestamp")}
        for row in rows[::17]
    ]
    return dict(rows=rows, queries=queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New directory for artifacts and comparison evidence"
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("xgboost", "bayesian_ridge", "lightgbm"),
        default=["xgboost", "bayesian_ridge", "lightgbm"],
    )
    parser.add_argument("--objectives", nargs="+", choices=("mean", "quantile"), default=["mean", "quantile"])
    parser.add_argument("--ensemble", choices=("single", "gated", "both"), default="both")
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    roots = {"baseline": args.baseline_dir.resolve(), "candidate": args.candidate_dir.resolve()}
    for root in roots.values():
        if not (root / "training/training_server.py").is_file():
            parser.error(f"Not a Predictor checkout: {root}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "payload.json"
    data.write_text(json.dumps(payload()))
    provenance = {
        "sources": {name: source_record(root) for name, root in roots.items()},
        "python": sys.version,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "protobuf",
                "pydantic",
                "fastapi",
                "joblib",
            )
        },
        "configuration": {
            "seed": 913,
            "rows": 512,
            "queries": 31,
            "train_test_ratio": 0.2,
            "bucket_capacity": 500,
            "test_capacity": 1000,
            "quantile_alpha": 0.9,
            "encoder_features": True,
            "tokens_in_flight_features": True,
        },
        "scope": "CPU fixed-artifact and retraining comparison; no LLM inference or A-C load test",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    worker = Path(__file__).with_name("testdata") / "prediction-comparison-worker.py"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONHASHSEED="0",
        PYTHONOPTIMIZE="0",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LATENCY_RETRAINING_INTERVAL_SEC="3600",
        MODEL_SYNC_INTERVAL_SEC="3600",
        LATENCY_MIN_SAMPLES_FOR_RETRAIN="5",
        LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH="5",
        LATENCY_MIN_SAMPLES_FOR_ENSEMBLE_SPLIT="3",
        LATENCY_TEST_TRAIN_RATIO="0.2",
        LATENCY_MAX_TRAINING_DATA_SIZE_PER_BUCKET="500",
        LATENCY_MAX_TEST_DATA_SIZE="1000",
        LATENCY_QUANTILE_ALPHA="0.9",
        LATENCY_ENABLE_ENCODER_FEATURES="true",
        LATENCY_ENABLE_TOKEN_IN_FLIGHT_FEATURES="true",
    )
    regimes = [False, True] if args.ensemble == "both" else [args.ensemble == "gated"]
    results = []
    for engine in args.engines:
        for objective in args.objectives:
            for gated in regimes:
                case = f"{engine}-{objective}-ensemble-{str(gated).lower()}"
                directory = output / case
                directory.mkdir()
                settings = dict(
                    env,
                    LATENCY_MODEL_TYPE=engine,
                    LATENCY_OBJECTIVE_TYPE=objective,
                    LATENCY_ENSEMBLE_MODE=str(gated).lower(),
                )
                with (directory / "run.log").open("w") as log:

                    def run(label, root, mode, artifacts):
                        report = directory / f"{label}-{mode}.json"
                        subprocess.run(
                            [sys.executable, str(worker), str(root), mode, str(artifacts), str(data), str(report)],
                            env=settings,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=180,
                        )
                        return json.loads(report.read_text())

                    try:
                        trained, predictions = {}, {}
                        for name, root in roots.items():
                            artifacts = directory / f"{name}-models"
                            trained[name] = run(name, root, "train", artifacts)
                            predictions[name] = run(name, root, "serve", artifacts)
                        fixed = run("fixed", roots["candidate"], "serve", directory / "baseline-models")
                        for key in ("retained_digest", "frames"):
                            if trained["baseline"][key] != trained["candidate"][key]:
                                raise ValueError(f"Training {key} differs")
                        if fixed != predictions["baseline"]:
                            raise ValueError("Fixed-artifact predictions or validation differ")
                        if predictions["candidate"] != predictions["baseline"]:
                            raise ValueError("Controlled retraining predictions or validation differ")
                        result = dict(case=case, status="passed", max_prediction_delta_ms=0)
                    except (ValueError, subprocess.SubprocessError) as error:
                        result = dict(case=case, status="failed", error=str(error))
                results.append(result)
                (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)
    return int(any(result["status"] != "passed" for result in results))


if __name__ == "__main__":
    sys.exit(main())
