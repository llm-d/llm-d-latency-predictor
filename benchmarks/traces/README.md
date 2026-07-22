# Benchmark Traces

Recorded traces for offline feature A/B validation. Each trace is a JSONL
file captured by `trace_recorder.py` from a live deployment.

## Using a trace

```bash
python benchmarks/run_validation.py \
    --trace benchmarks/traces/sharegpt-h200.jsonl \
    --feature <feature_name> \
    --workload-spec benchmarks/traces/sharegpt-h200-spec.yaml \
    --seeds 10 --shap --json
```

## Adding a new trace

1. Deploy the llm-d stack with `trace_recorder.py` (see `benchmarks/README.md`)
2. Run load with sustained concurrency (verify `num_request_running > 1`)
3. Copy the JSONL to this directory: `<workload>-<gpu>.jsonl`
4. Add a companion spec: `<workload>-<gpu>-spec.yaml` (copy and fill the template)

## Required fields

A trace must contain at minimum:

| Field | Type | Description |
|-------|------|-------------|
| `kv_cache_percentage` | float | GPU KV cache utilization (0-1) |
| `input_token_length` | int | Input sequence length |
| `num_request_waiting` | int | Queued requests on this pod |
| `num_request_running` | int | Concurrent requests on this pod |
| `actual_ttft_ms` | float | Observed time to first token (ms) |
| `actual_tpot_ms` | float | Observed time per output token (ms) |
| `prefix_cache_score` | float | Prefix cache hit ratio (0-1) |

Optional fields (zero-filled when absent):

| Field | Type | Description |
|-------|------|-------------|
| `prefill_tokens_in_flight` | int | Total prefill tokens across concurrent requests |
| `decode_tokens_in_flight` | int | Total decode tokens across concurrent requests |
| `encoder_matched_size` | int | Encoder cache matched size (multimodal) |
| `encoder_input_size` | int | Encoder input size (multimodal) |
| `num_tokens_generated` | int | Output tokens generated so far |
| `pod_type` | str | `"prefill"`, `"decode"`, or `""` (monolithic) |

## Naming convention

`<workload>-<gpu>.jsonl` with a companion `<workload>-<gpu>-spec.yaml`.

The workload name describes the traffic pattern (not the hardware). The GPU
suffix documents the capture environment for reproducibility.
