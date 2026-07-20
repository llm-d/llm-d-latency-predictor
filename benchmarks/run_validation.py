#!/usr/bin/env python3
"""Runner script: single-command validation for latency predictor features.

Chains: load trace → self-test → A/B → artifacts → exit code.
Wraps offline_feature_ab.py functions — no subprocess, direct import.

Usage:
  python benchmarks/run_validation.py \\
      --trace trace.jsonl \\
      --feature prefill_density \\
      --seeds 10 \\
      --workload-spec benchmarks/workload-spec.yaml \\
      --outdir results/prefill_density \\
      --shap --json

Exit codes:
  0 — all gates passed, artifacts generated
  1 — contention gate or self-test failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_feature_ab import (
    ESTIMATORS,
    add_derived_features,
    calibration_plot,
    contention_gate,
    load_trace,
    main as ab_main,
    reliability_diagram,
    run_ab,
    self_test,
    shap_analysis,
    summarize,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Single-command feature validation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--trace", type=Path, required=True,
                   help="JSONL trace from trace_recorder.py")
    p.add_argument("--feature", default="prefill_density",
                   help="feature to A/B test (default: prefill_density)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--min-contention-pct", type=float, default=20.0)
    p.add_argument("--model-type", choices=list(ESTIMATORS.keys()), default="xgboost")
    p.add_argument("--workload-spec", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("results/ab"))
    p.add_argument("--shap", action="store_true")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    # Delegate to offline_feature_ab.main() with the same argv
    # This keeps run_validation.py as a thin entry point
    sys.argv = ["offline_feature_ab.py"]
    sys.argv += ["--trace", str(args.trace)]
    sys.argv += ["--feature", args.feature]
    sys.argv += ["--seeds", str(args.seeds)]
    sys.argv += ["--min-contention-pct", str(args.min_contention_pct)]
    sys.argv += ["--model-type", args.model_type]
    sys.argv += ["--outdir", str(args.outdir)]
    if args.workload_spec:
        sys.argv += ["--workload-spec", str(args.workload_spec)]
    if args.shap:
        sys.argv += ["--shap"]
    if args.json_output:
        sys.argv += ["--json"]

    try:
        ab_main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
