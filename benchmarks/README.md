# Feature Validation Benchmark Template

A repeatable evidence standard for latency-predictor feature changes.
Every engineered feature PR adding a new feature should ship the same
before/after evidence on a named workload, so real signal is distinguishable
from noise and regressions on the untouched target are caught.

> [!NOTE]
> Every engineered feature PR adding a new feature should ship the same
> actual-vs-predicted evidence on a named workload, so real signal is
> distinguishable from noise and regressions on the untouched target are caught.

## The evidence standard

A feature change is validated when the PR includes:

1. **Named workload spec** — model, hardware, input/output length distribution,
   QPS/concurrency, duration, and whether the trace is real or synthetic.
2. **Actual-vs-predicted calibration plots** for TTFT *and* TPOT, with and
   without the feature (pooled test sets across all seeds).
3. **Reliability diagrams** — binned P90 coverage curves showing whether the
   predicted quantile is well-calibrated across the prediction range.
4. **Error bars across ≥10 seeds** — a delta only counts if it exceeds the
   per-arm seed standard deviation (`> seed std?` column in `summary.md`).
5. **Contention evidence** — the trace passes the contention gate
   (`num_request_running > 1` in ≥20% of samples by default). A
   zero-contention trace cannot exercise contention features.
6. **Self-test** — shuffled-label discriminability check passes (the gate
   can tell real signal from noise).

## Quick start

```bash
# Install dependencies
pip install -r benchmarks/requirements.txt

# Run with an included reference trace (3 available — see benchmarks/traces/)
python benchmarks/run_validation.py \
    --trace benchmarks/traces/sharegpt-h200.jsonl \
    --feature <feature_name> \
    --seeds 10 \
    --workload-spec benchmarks/traces/sharegpt-h200-spec.yaml \
    --outdir results/<feature_name> \
    --shap --convergence --json

# Or try long-context / extreme-contention traces
python benchmarks/run_validation.py \
    --trace benchmarks/traces/chatbot-synthetic-h200.jsonl \
    --feature <feature_name> --seeds 10 --json

# Or combine traces for broader coverage
cat benchmarks/traces/*.jsonl > /tmp/combined.jsonl
python benchmarks/run_validation.py \
    --trace /tmp/combined.jsonl \
    --feature <feature_name> --seeds 10 --json

# Or bring your own trace
python benchmarks/run_validation.py \
    --trace /path/to/your/trace.jsonl \
    --feature <feature_name> \
    --seeds 10 \
    --shap --json
```

## Workflow

```text
[1] deploy stack          EPP + latency predictor + vLLM on GPU
        |
[2] insert recorder       trace_recorder.py between EPP and training server
        |                 (TRAINING_SERVER_URL -> recorder -> training server)
[3] generate load         any load generator with sustained concurrency
        |                 (verify num_request_running > 1 in the trace!)
[4] collect trace         JSONL of real training entries (features + actuals)
[5] offline A/B           run_validation.py --trace ... --feature <name>
        |                 self-test → contention gate → N-seed A/B → artifacts
[6] attach artifacts      summary.md + calibration PNGs + reliability PNGs
```

### Step 1-2: deploy with the recorder

Deploy the llm-d stack (EPP + latency predictor + vLLM),
then run the recorder next to the training server and point the EPP at it:

```bash
# recorder (as a sidecar/deployment reachable by the EPP)
FORWARD_URL=http://localhost:8000 TRACE_FILE=/data/trace.jsonl PORT=8002 \
    python benchmarks/trace_recorder.py

# EPP env
TRAINING_SERVER_URL=http://<recorder>:8002
```

The recorder forwards **all** traffic unchanged (predict, model download,
health) and appends every `/add_training_data_bulk` entry to the JSONL trace.

### Step 3: load with real contention

Any load generator works; the requirement is *sustained concurrency* so
multiple requests are in flight per pod. Verify before analyzing:

```bash
python -c "
import json, collections
c = collections.Counter(json.loads(l)['num_request_running'] for l in open('trace.jsonl'))
print(sorted(c.items()))"
```

### Step 5: offline A/B (via runner script)

```bash
python benchmarks/run_validation.py \
    --trace trace.jsonl --feature <feature_name> --seeds 10 \
    --workload-spec benchmarks/traces/sharegpt-h200-spec.yaml \
    --outdir results/ab --shap --json
```

The runner chains: self-test → contention gate → N-seed A/B → all artifacts.
Exit code 0 means all gates passed; exit code 1 means a gate failed.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--trace` | (required) | Path to JSONL trace from trace_recorder.py |
| `--feature` | (required) | Feature column name to A/B test |
| `--seeds` | `10` | Number of random train/test splits |
| `--min-contention-pct` | `20.0` | Minimum % samples with `num_request_running > 1` |
| `--model-type` | `xgboost` | Estimator: `xgboost` or `lightgbm` |
| `--workload-spec` | (none) | YAML file to embed in summary.md |
| `--shap` | off | Compute SHAP feature importance |
| `--json` | off | Print machine-readable fitness line to stdout |

### Interpreting `summary.md`

| Column | Meaning |
|---|---|
| `without` / `with` | metric mean ± std across seeds |
| `delta` | relative change, with-arm vs without-arm |
| `> seed std?` | **the decision column** — `NO — within noise` means the delta is not distinguishable from seed variance and must not be claimed as improvement |

## Metric rationale

**Primary: pinball loss (quantile_loss)** — The correct loss function for P90
quantile regression. Penalizes underpredictions 9× more than overpredictions,
directly measuring what the model optimizes. MAPE is fundamentally wrong for
quantile models: it measures mean prediction quality, is biased toward low
predictions, and has division-by-zero issues (arXiv:1605.02541).

**Secondary: calibration coverage (coverage_pct)** — For a well-calibrated P90
model, ~90% of actual values should fall below the predicted quantile. Coverage
below 90% means the model is underprotective; above 90% means overprotective.

**Tertiary: MAPE** — Kept for familiarity, but should not be used as the
primary evaluation metric for quantile regression models.

## Reliability diagrams

Unlike scatter plots (which show individual predictions), reliability diagrams
bin predictions and show the *observed* P90 coverage in each bin. A
well-calibrated model produces bars near the 0.9 reference line across all
bins. Systematic deviation indicates regions where the quantile is miscalibrated.

## SHAP feature importance

When `--shap` is passed, the template computes SHAP TreeExplainer values on
the with-arm model. Unlike XGBoost's built-in gain/cover (which can be
contradictory and unreliable), SHAP values have mathematical consistency
guarantees from Shapley game theory (Lundberg & Lee, NeurIPS 2017).

## CI integration (proposal)

See `ci/feature-ab.yaml` for a draft GitHub Actions workflow. The key insight:
**GPU is needed only for trace capture, not for offline A/B analysis.** Once a
baseline trace is captured and committed, the CI workflow runs the A/B on CPU
in seconds, making it suitable for CI gating.

## Files

| File | Purpose |
|---|---|
| `offline_feature_ab.py` | contention gate + self-test + N-seed A/B + calibration + reliability + SHAP |
| `run_validation.py` | single-command runner wrapping offline_feature_ab |
| `trace_recorder.py` | recording proxy: captures real EPP training entries to JSONL |
| `workload-spec.yaml` | blank template for documenting new trace captures |
| `traces/*-spec.yaml` | filled workload specs per trace |
| `requirements.txt` | analysis dependencies |
| `ci/feature-ab.yaml` | draft GitHub Actions workflow (proposal for #15) |
