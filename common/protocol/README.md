# Shared latency protocol

The canonical schema and wire fixtures are maintained in
[llm-d-router](https://github.com/llm-d/llm-d-router/tree/87f9dca7fc08c81cb50c0b89e209b841150805bd/proto).
`schema-source.json` records the full source commit and schema/fixture hashes.
Runtime services use the generated Python module without a Router checkout.
The technical design is [Router RFC 4218](https://github.com/llm-d/llm-d-router/issues/4218).

## Generate and verify

Use protoc 25.3 and a Router checkout containing the pinned commit:

```bash
make generate-proto ROUTER_PROTO_DIR=/path/to/llm-d-router/proto
make verify-proto ROUTER_PROTO_DIR=/path/to/llm-d-router/proto
```

Verification checks the committed source bytes, both hashes, generated binding
and copied fixture. It writes no outputs. A schema update changes the source
pin and regenerates the binding and fixture together. Committed source pins
name the upstream Router repository; local file sources are for disposable
experiments and require explicit opt-in.

## HTTP compatibility

FastAPI models are generated from field descriptors while the existing routes
retain flat snake_case JSON, validation and response behavior. The optional
`to_proto()` conversion does not replace HTTP validation. Codec conversion
retains optional zero presence and exact integer JSON numbers, discards unknown
fields and replaces a destination only after successful parsing. The wire
fixtures cover prediction/training rows and model artifacts.

## CPU behavior comparison

With both checkouts available in the same installed Python runtime, compare
fixed artifacts, ordered training frames and controlled retraining:

```bash
python hack/compare-predictions.py --baseline-dir /path/to/baseline --output-dir /path/to/new-results
```

The command records source hashes, runtime versions, inputs and per-case
results for all three engines, both objectives and single/gated models. It uses
disposable artifacts and no external services. This comparison does not run LLM
inference or establish the issue's A-C performance gates.
