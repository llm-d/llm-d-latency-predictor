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

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def generation(tmp_path):
    protoc = os.environ.get("PROTOC", shutil.which("protoc"))
    if not protoc:
        pytest.skip("generation checks require protoc 25.3")
    source = tmp_path / "router"
    schema = source / "proto/common/protocol/latency.proto"
    fixture = source / "proto/testdata/wire.json"
    schema.parent.mkdir(parents=True)
    fixture.parent.mkdir(parents=True)
    schema.write_text('syntax = "proto3"; package test.latency; message Features { optional int64 count = 1; }\n')
    fixture.write_text('{"count": 7}\n')
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "add", "proto"], cwd=source, check=True)
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Generation test",
        GIT_AUTHOR_EMAIL="test@localhost",
        GIT_COMMITTER_NAME="Generation test",
        GIT_COMMITTER_EMAIL="test@localhost",
    )
    tree = subprocess.check_output(["git", "write-tree"], cwd=source, text=True).strip()
    commit = subprocess.check_output(
        ["git", "commit-tree", tree, "-m", "Test schema"], cwd=source, env=env, text=True
    ).strip()
    work = tmp_path / "predictor"
    (work / "hack").mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "hack/generate-protocol.py", work / "hack/generate-protocol.py")
    lock = dict(
        repository=source.as_uri(),
        commit=commit,
        schema_path="proto/common/protocol/latency.proto",
        fixture_path="proto/testdata/wire.json",
        schema_sha256=hashlib.sha256(schema.read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
    )
    lock_path = work / "common/protocol/schema-source.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(lock))
    command = [
        sys.executable,
        str(work / "hack/generate-protocol.py"),
        "--proto-dir",
        str(source / "proto"),
        "--protoc",
        protoc,
    ]
    return source, work, lock, lock_path, command


def test_pinned_generation_and_read_only_verification(generation):
    source, work, _, _, command = generation
    subprocess.run([*command, "--allow-local-source"], check=True)
    output = work / "common/protocol/latency_pb2.py"
    copied = work / "tests/testdata/latency-wire.json"
    assert copied.read_bytes() == (source / "proto/testdata/wire.json").read_bytes()
    before = (output.read_bytes(), copied.read_bytes())
    subprocess.run([*command, "--allow-local-source", "--check"], check=True)
    assert (output.read_bytes(), copied.read_bytes()) == before
    output.write_bytes(b"outdated generated output\n")
    result = subprocess.run([*command, "--allow-local-source", "--check"], capture_output=True)
    assert result.returncode == 1
    assert output.read_bytes() == b"outdated generated output\n"
    assert copied.read_bytes() == before[1]


@pytest.mark.parametrize("change", ["schema", "fixture", "missing", "commit", "uncommitted"])
def test_source_pin_rejects_drift_without_replacing_outputs(generation, change):
    source, work, lock, lock_path, command = generation
    subprocess.run([*command, "--allow-local-source"], check=True)
    output = work / "common/protocol/latency_pb2.py"
    copied = work / "tests/testdata/latency-wire.json"
    before = (output.read_bytes(), copied.read_bytes())
    if change in ("schema", "uncommitted"):
        path = source / lock["schema_path"]
        path.write_text(path.read_text() + "// uncommitted edit\n")
        if change == "uncommitted":
            lock["schema_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == "fixture":
        (source / lock["fixture_path"]).write_text("{}\n")
    elif change == "missing":
        (source / lock["schema_path"]).unlink()
    else:
        lock["commit"] = "0" * 40
    lock_path.write_text(json.dumps(lock))
    result = subprocess.run([*command, "--allow-local-source"], capture_output=True)
    assert result.returncode != 0
    assert (output.read_bytes(), copied.read_bytes()) == before


def test_local_schema_provenance_needs_explicit_opt_in(generation):
    _, work, _, _, command = generation
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 2
    assert b"--allow-local-source" in result.stderr
    assert not (work / "common/protocol/latency_pb2.py").exists()
