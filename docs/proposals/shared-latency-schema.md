# Shared schema for latency predictor features

## Summary

Define a protobuf schema for the latency predictor's feature and model-artifact
contracts, and generate Go and Python types from it. Keep the existing JSON
over HTTP interface. The work addresses
[llm-d-router issue 913](https://github.com/llm-d/llm-d-router/issues/913).

## Motivation

Adding an input currently requires coordinated changes in the router, the
training service and the prediction service. A missed change can silently omit
the input from training, prediction or exported model data.

### Goals

- Define shared raw feature fields once for training and prediction.
- Preserve existing clients, historical training rows and model artifacts during
  a rolling upgrade.
- Make a new optional numeric feature require a schema field and a router
  provider, with the Python feature pipeline discovering it from the schema.
- Preserve the current prediction and training behavior through the refactor.

### Non-goals

- Changing HTTP routes or replacing HTTP with gRPC.
- Introducing `image_size` or another production feature in this work.
- Replacing model-library artifacts such as joblib files or LightGBM text.
- Changing model algorithms, scheduling policies or the router's existing
  Bayesian Ridge formula.

## Proposal

The predictor repository owns a `.proto` defining a reusable `Features`
message, training and prediction requests, and model-artifact representations.
Generated Python types live in this repository; the router generates Go types
from the same schema. Field numbers and optional presence provide the basis for
compatible schema evolution.

In a second phase, a router feature registry declares each field's source,
capture scope and provider. The router builds prediction and training data
from those providers and uses constructor options for additional features. The
Python training and serving paths discover raw fields and model-column order
from schema metadata. The existing local Ridge calculation iterates over the
schema-declared terms while retaining its current behavior.

## Design details

### Wire and rollout compatibility

Keep the current flat, snake_case JSON fields. A compatibility adapter flattens
the composed feature message and emits 64-bit integers as JSON numbers, as
existing consumers expect. Optional presence distinguishes a missing field from
an explicitly supplied zero. Unknown fields are ignored at the HTTP boundary.

New optional fields must have an application default for historical rows and
must not reuse removed field numbers or names. Existing servers can ignore a
new field during rollout; they cannot use it until the updated service and
model are deployed. Named model and scaler artifacts retain their recorded
column order, and incompatible unknown required columns must not replace a
working model. Existing response shapes and partial-batch errors stay intact.

The schema covers the coefficient names, tree JSON and model metadata exchanged
between components. Prometheus metrics remain text, and persisted model-library
files keep their current formats. The router records the source schema revision
or digest with its generated files. The matching predictor schema must be
published before a standalone router change depends on it.

### Feature collection

The schema records the field's validation, default and model-column order. A
router provider reads the signal from request data, endpoint state, metrics or
request context. Endpoint values needed by later hooks are captured at the
existing lifecycle point; providers declare their dependencies so scheduling
order remains explicit. The Python feature plan retains the existing engineered
columns and model-specific preprocessing.

The existing Go Ridge formula omits some inputs used by Python, including tokens
in flight. Preserving the formula means local Go and Python predictions can
differ for those inputs. Correcting that pre-existing difference changes
behavior and should be evaluated separately.

### Validation and review sequence

1. Review and approve this proposal, then submit the shared schema and
   compatibility adapters as a small, independently testable change.
2. Submit the registry and schema-derived feature plan after the shared contract
   is available. Keep independent artifact fixes in separate reviewable changes.
3. Verify old/new clients and services, missing historical fields, model reload,
   single and batch paths, and a disposable schema/provider extension across
   XGBoost, Bayesian Ridge and LightGBM.
4. For both phases, run the A-C before/after performance comparisons and
   publish the measurements required by issue 913. Those measurements include
   Kubernetes and load-plus-prefix baselines, NTPOT, output throughput and
   saturation behavior. They are required before claiming the performance gate.

## Alternatives

- A map of features is flexible but loses the typed field contract and presence
  information.
- Duplicating flat fields in each protobuf message keeps the JSON shape simple
  but retains multiple feature definitions.
- Using ProtoJSON directly would change the existing nesting and the JSON
  representation of 64-bit integers.
- gRPC would introduce a transport and deployment change that is not required
  to share the schema.
