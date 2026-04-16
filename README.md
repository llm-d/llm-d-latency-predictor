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

### Components

- `prediction_server.py` — FastAPI prediction server (serves latency predictions)
- `training_server.py` — FastAPI training server (trains models from request traces)
- `test_dual_server_client.py` — integration / load-test client exercising both servers
- `deploy/` — Kubernetes manifests and kustomization
- `build-deploy.sh` — helper script for building images and deploying to GKE

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
