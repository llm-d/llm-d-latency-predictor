# Calibration drift harness

Manual harness for exercising continuous coverage evaluation and the
calibration-triggered retrain end-to-end against a live `training_server`.
Committed here so the automated CI/e2e work (#15) can reuse it rather than
re-derive it. The unit tests for the trigger *logic* live in
`tests/test_calibration_trigger.py` and need no server.

## Scripts

- `synth_workload.py` — drives `/add_training_data_bulk` with a synthetic
  linear-noise latency model, switching regimes at `--drift-at` to inject drift.
  Drift modes: `slope` (multiply slope coefficients — dispersion-dominant on the
  heavy-tailed features), `location` (shift intercepts), `noise` (scale sigma).
- `scrape_metrics.py` — polls `/metrics` and records the coverage (and
  quantile-loss) series to CSV over time.

## Example

```bash
# terminal 1: run the server with the coverage loop enabled
LATENCY_COVERAGE_EVAL_INTERVAL_SEC=5 uvicorn llm_d_latency_predictor.training_server:app --port 8000

# terminal 2: baseline for 90s, then inject drift; watch the trigger fire
python tests/drift_harness/synth_workload.py --duration 300 --drift-at 90 --rate 150
python tests/drift_harness/scrape_metrics.py --duration 300 --output coverage.csv
```
