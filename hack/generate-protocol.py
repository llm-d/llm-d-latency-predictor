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
"""Generate Python messages without importing the model servers."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def pinned_source(source: Path, lock_path: Path, allow_local: bool):
    lock = json.loads(lock_path.read_text())
    repository = lock["repository"]
    if repository not in ("https://github.com/llm-d/llm-d-router", "https://github.com/llm-d/llm-d-router.git"):
        if not allow_local or not repository.startswith("file:///"):
            raise ValueError("schema source must be upstream Router; local POC sources require --allow-local-source")
    commit = lock["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("schema source commit must be a full 40-character SHA")
    repository_root = Path(
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True).strip()
    )
    if source != repository_root / "proto":
        raise ValueError("--proto-dir must point to the pinned Router checkout's proto directory")
    files = {}
    for key, expected in (("schema", "proto/common/protocol/latency.proto"), ("fixture", "proto/testdata/wire.json")):
        if lock[f"{key}_path"] != expected:
            raise ValueError(f"Unexpected {key} path")
        path = repository_root / expected
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != lock[f"{key}_sha256"]:
            raise ValueError(f"Pinned {key} hash does not match source")
        committed = subprocess.check_output(["git", "-C", str(repository_root), "show", f"{commit}:{expected}"])
        if data != committed:
            raise ValueError(f"Pinned {key} differs from the recorded commit")
        files[key] = data
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--proto-dir", required=True, type=Path)
    parser.add_argument("--source-lock", default=root / "common/protocol/schema-source.json", type=Path)
    parser.add_argument(
        "--allow-local-source",
        action="store_true",
        help="Allow an explicitly recorded file:// source for local POCs only",
    )
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.proto_dir.resolve()
    try:
        files = pinned_source(source, args.source_lock, args.allow_local_source)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    version = subprocess.check_output([args.protoc, "--version"], text=True).strip()
    if version != "libprotoc 25.3":
        parser.error(f"protoc 25.3 is required, found {version}")
    relative = Path("common/protocol/latency_pb2.py")
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                args.protoc,
                f"--proto_path={source}",
                f"--python_out={directory}",
                "common/protocol/latency.proto",
            ],
            check=True,
        )
        generated = (Path(directory) / relative).read_bytes()
    for path, data in {relative: generated, Path("tests/testdata/latency-wire.json"): files["fixture"]}.items():
        destination = root / path
        if args.check:
            if not destination.is_file() or destination.read_bytes() != data:
                parser.exit(1, f"{path} differs from pinned Router source; run make generate-proto\n")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


if __name__ == "__main__":
    main()
