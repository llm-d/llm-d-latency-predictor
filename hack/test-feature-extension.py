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

"""Verify a schema + provider extension in disposable copies of both repositories."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--protoc-gen-go", type=Path, required=True)
    parser.add_argument("--go", default="go")
    parser.add_argument("--cpu-limit", type=int, default=2, help="Maximum CPUs for this test and its children on Linux")
    args = parser.parse_args()
    if args.cpu_limit < 1:
        parser.error("--cpu-limit must be positive")
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))[: args.cpu_limit]
        os.sched_setaffinity(0, cpus)
        print(f"Feature-extension test CPU affinity: {cpus}", flush=True)
    root = Path(__file__).resolve().parents[1]
    package = "pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency"
    ignore = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache")
    with tempfile.TemporaryDirectory(prefix="latency-feature-extension-") as temporary:
        predictor = Path(temporary) / "predictor"
        router = Path(temporary) / "router"
        shutil.copytree(root, predictor, ignore=ignore)
        shutil.copytree(args.router_dir, router, ignore=ignore)
        schema = router / "proto/common/protocol/latency.proto"
        schema.write_text(
            schema.read_text().replace(
                "\n}\n\n// The HTTP codec",
                "\n  optional int64 extension_test_size = 99 [(field_rules) = {\n"
                '    minimum: 5 default_json: "7" go_validate: true ttft_order: 85 tpot_order: 75\n'
                "    ridge_ttft_order: 8 ridge_tpot_order: 6\n"
                "  }];\n}\n\n// The HTTP codec",
                1,
            )
        )
        providers = router / package / "features.go"
        providers.write_text(
            providers.read_text().replace(
                "\n}\n\nfunc init()",
                '\n\t{name: "extension_test_size", scope: requestFeature, uses: allFeatureUses, '
                "read: func(featureInput) protoreflect.Value { return protoreflect.ValueOfInt64(17) }},"
                "\n}\n\nfunc init()",
                1,
            )
        )
        run("git", "init", "-q", cwd=router)
        run("git", "add", "proto", cwd=router)
        tree = subprocess.check_output(["git", "write-tree"], cwd=router, text=True).strip()
        env = dict(
            os.environ,
            GIT_AUTHOR_NAME="Feature extension POC",
            GIT_AUTHOR_EMAIL="test@localhost",
            GIT_COMMITTER_NAME="Feature extension POC",
            GIT_COMMITTER_EMAIL="test@localhost",
        )
        commit = subprocess.check_output(
            ["git", "commit-tree", tree, "-m", "Disposable schema extension"], cwd=router, env=env, text=True
        ).strip()
        lock = {"repository": router.as_uri(), "commit": commit}
        for key, relative in (
            ("schema", "proto/common/protocol/latency.proto"),
            ("fixture", "proto/testdata/wire.json"),
        ):
            lock[f"{key}_path"] = relative
            lock[f"{key}_sha256"] = hashlib.sha256((router / relative).read_bytes()).hexdigest()
        (predictor / "common/protocol/schema-source.json").write_text(json.dumps(lock, indent=2) + "\n")
        run(
            sys.executable,
            "hack/generate-protocol.py",
            "--proto-dir",
            str(router / "proto"),
            "--allow-local-source",
            "--protoc",
            args.protoc,
            cwd=predictor,
        )
        run(
            sys.executable,
            "hack/generate-latency-proto.py",
            "--proto-dir",
            str(router / "proto"),
            "--protoc",
            args.protoc,
            "--protoc-gen-go",
            str(args.protoc_gen_go.resolve()),
            cwd=router,
        )
        (router / package / "extension_test.go").write_text("""package predictedlatency
import (
    "encoding/json"
    "testing"
    fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
    client "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
)
func TestFeatureExtension(t *testing.T) {
    state := newPredictedLatencyContext(createTestInferenceRequest("extension", 100, 20))
    for _, use := range []featureUse{predictionFeatures, ttftFeatures, tpotFeatures} {
        features := composeFeatures(state, &fwkdl.Metrics{}, nil, use)
        if features.GetExtensionTestSize() != 17 { t.Fatal(features) }
        for _, row := range []any{client.PredictionFromFeatures(features), client.TrainingFromFeatures(features, client.TrainingEntry{})} {
            data, err := json.Marshal(row)
            if err != nil { t.Fatal(err) }
            var fields map[string]any
            if err := json.Unmarshal(data, &fields); err != nil { t.Fatal(err) }
            if fields["extension_test_size"] != float64(17) { t.Fatal(string(data)) }
        }
    }
    if _, err := client.NewPredictionRequest(0, 1, 0, 0, 1, 0); err != nil { t.Fatal("schema default was not applied before validation:", err) }
    request, err := client.NewPredictionRequest(0, 1, 0, 0, 1, 0, client.WithFeature("extension_test_size", 23))
    if err != nil || request.Features.GetExtensionTestSize() != 23 { t.Fatal(request, err) }
    if _, err := client.NewPredictionRequest(0, 1, 0, 0, 1, 0, client.WithFeature("extension_test_size", -1)); err == nil { t.Fatal("negative feature accepted") }
}
""")
        (router / package / "latencypredictorclient/extension_test.go").write_text(
            """package latencypredictorclient
import (
    "testing"
    "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
)
func TestFeatureExtensionRidge(t *testing.T) {
    // Missing optional fields use the schema default, including local ridge evaluation.
    for _, fields := range [][]protoreflect.FieldDescriptor{ridgeTTFTFields, ridgeTPOTFields} {
        value, err := ridgePrediction(1, map[string]float64{"extension_test_size": 2}, &protocol.Features{}, fields)
        if err != nil || value != 15 { t.Fatal(value, err) }
    }
}
""".replace('"testing"', '"testing"\n"google.golang.org/protobuf/reflect/protoreflect"')
        )
        run(
            args.go,
            "test",
            "-p",
            "1",
            "-race",
            "-run",
            "^TestFeatureExtension(Ridge)?$",
            f"./{package}/...",
            cwd=router,
        )
        run(
            sys.executable,
            "-c",
            """
from types import SimpleNamespace
from common.protocol import latency_pb2 as pb
from common.protocol.models import request_model
from common.feature_plan import raw_frames, prepare_features, training_frame
flags = SimpleNamespace(ENABLE_ENCODER_FEATURES=True, ENABLE_TOKEN_IN_FLIGHT_FEATURES=True)
model = request_model(pb.PredictionRequest, "ExtendedRequest", "__main__")
row = dict(kv_cache_percentage=0, input_token_length=1, num_request_waiting=0,
           num_request_running=0, num_tokens_generated=1, prefix_cache_score=0)
assert model(**row).extension_test_size == 7
request = model(**row, extension_test_size=17)
assert request.to_proto().features.extension_test_size == 17
for target, frame in zip(("ttft", "tpot"), raw_frames([request.model_dump()], flags)):
    assert prepare_features(frame, target, flags)["extension_test_size"].iloc[0] == 17
    historical = training_frame([dict(row, **{f"actual_{target}_ms": 5})], target, flags)
    assert historical["extension_test_size"].iloc[0] == 7
""",
            cwd=predictor,
        )
        shutil.copyfile(
            predictor / "hack/testdata/feature-extension_test.go",
            router / package / "latencypredictorclient/extension_roundtrip_test.go",
        )
        for engine in ("bayesian_ridge", "xgboost", "lightgbm"):
            run(
                sys.executable,
                "hack/feature-extension-roundtrip.py",
                "--router-dir",
                str(router),
                "--go",
                args.go,
                "--engine",
                engine,
                cwd=predictor,
            )
    print("Schema + provider extension passed for Go prediction, training, ridge and Python feature paths.")


if __name__ == "__main__":
    main()
