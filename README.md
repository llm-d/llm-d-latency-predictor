# llm-d-latency-predictor

[![CI](https://github.com/llm-d/llm-d-latency-predictor/actions/workflows/ci-pr-checks.yaml/badge.svg)](https://github.com/llm-d/llm-d-latency-predictor/actions/workflows/ci-pr-checks.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Latency prediction service for the llm-d ecosystem.**

## Overview

<!-- TODO: Describe what this project does, why it exists, and how it fits into the llm-d ecosystem -->

## Prerequisites

- Python 3.11+
- Docker (for container builds)
- [pre-commit](https://pre-commit.com/) (for local development)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/llm-d/llm-d-latency-predictor.git
cd llm-d-latency-predictor

# Install pre-commit hooks
pre-commit install

# Install Python dependencies
make install

# Run tests
make test

# Run linters
make lint
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, coding standards, and how to submit changes.

### Common Commands

```bash
make help           # Show all available targets
make install        # Install Python dependencies
make test           # Run Python tests
make lint           # Run Python linter (ruff)
make fmt            # Format Python code
make image-build    # Build container images (prediction, training, test)
make pre-commit     # Run pre-commit hooks
```

### Layout

```
src/llm_d_latency_predictor/
  prediction_server.py    # FastAPI prediction server (serves latency predictions)
  training_server.py      # FastAPI training server (trains models from request traces)
tests/
  test_dual_server_client.py  # integration / load-test client exercising both servers
deploy/                   # Kubernetes manifests and kustomization
Dockerfile-prediction     # Image for the prediction server
Dockerfile-training       # Image for the training server
Dockerfile-test           # Image that runs the test client as a Job
build-deploy.sh           # Helper script for building images and deploying to GKE
```

The code is packaged as `llm_d_latency_predictor` — after `make install` (editable) or
`pip install .`, the servers can be run as Python modules:

```bash
uvicorn llm_d_latency_predictor.prediction_server:app --port 8001
uvicorn llm_d_latency_predictor.training_server:app --port 8000
```

## Architecture

<!-- TODO: Add architecture overview, diagrams, or links to design docs -->

## Configuration

<!-- TODO: Document configuration options, environment variables, CLI flags -->

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

All commits must be signed off (DCO). See [PR_SIGNOFF.md](PR_SIGNOFF.md) for instructions.

## Security

To report a security vulnerability, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.
