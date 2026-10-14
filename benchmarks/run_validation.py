#!/usr/bin/env python3
"""Runner script: single-command validation for latency predictor features.

Thin wrapper around offline_feature_ab.main() with exit-code handling.

Usage:
  python benchmarks/run_validation.py \
      --trace benchmarks/traces/sharegpt-h200.jsonl \
      --feature <feature_name> \
      --seeds 10 \
      --workload-spec benchmarks/traces/sharegpt-h200-spec.yaml \
      --outdir results/<feature_name> \
      --shap --convergence --json

Exit codes:
  0 — all gates passed, artifacts generated
  1 — contention gate, self-test, or regression gate failed
"""

from __future__ import annotations

import sys
from pathlib import Path

_benchmarks_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_benchmarks_dir))
sys.path.insert(0, str(_benchmarks_dir.parent))

from offline_feature_ab import main as ab_main  # noqa: E402


def main() -> int:
    try:
        ab_main()
        return 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
