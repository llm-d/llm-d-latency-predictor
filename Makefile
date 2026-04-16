# Project configuration
PROJECT_NAME ?= llm-d-latency-predictor
REGISTRY ?= ghcr.io/llm-d
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
PLATFORMS ?= linux/amd64,linux/arm64

# Container image names (one image per service)
PREDICTION_IMAGE ?= $(REGISTRY)/$(PROJECT_NAME)-prediction-server
TRAINING_IMAGE ?= $(REGISTRY)/$(PROJECT_NAME)-training-server
TEST_IMAGE ?= $(REGISTRY)/$(PROJECT_NAME)-test

.DEFAULT_GOAL := help

##@ General

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: install
install: ## Install the package (editable) and dev tools
	pip install -e .
	pip install ruff pytest pytest-asyncio

.PHONY: test
test: ## Run Python tests
	pytest tests/

.PHONY: lint
lint: ## Run Python linter (ruff)
	ruff check .
	ruff format --check .

.PHONY: fmt
fmt: ## Format Python code
	ruff format .

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

##@ Container

.PHONY: image-build
image-build: image-build-prediction image-build-training image-build-test ## Build all service images

.PHONY: image-build-prediction
image-build-prediction: ## Build prediction-server image
	docker buildx build \
		--platform $(PLATFORMS) \
		-f Dockerfile-prediction \
		--tag $(PREDICTION_IMAGE):$(VERSION) \
		--tag $(PREDICTION_IMAGE):latest \
		.

.PHONY: image-build-training
image-build-training: ## Build training-server image
	docker buildx build \
		--platform $(PLATFORMS) \
		-f Dockerfile-training \
		--tag $(TRAINING_IMAGE):$(VERSION) \
		--tag $(TRAINING_IMAGE):latest \
		.

.PHONY: image-build-test
image-build-test: ## Build test image
	docker buildx build \
		--platform $(PLATFORMS) \
		-f Dockerfile-test \
		--tag $(TEST_IMAGE):$(VERSION) \
		--tag $(TEST_IMAGE):latest \
		.

.PHONY: image-push
image-push: ## Build and push all service images
	docker buildx build --platform $(PLATFORMS) --push \
		-f Dockerfile-prediction \
		--annotation "index:org.opencontainers.image.source=https://github.com/llm-d/$(PROJECT_NAME)" \
		--annotation "index:org.opencontainers.image.licenses=Apache-2.0" \
		--tag $(PREDICTION_IMAGE):$(VERSION) --tag $(PREDICTION_IMAGE):latest .
	docker buildx build --platform $(PLATFORMS) --push \
		-f Dockerfile-training \
		--annotation "index:org.opencontainers.image.source=https://github.com/llm-d/$(PROJECT_NAME)" \
		--annotation "index:org.opencontainers.image.licenses=Apache-2.0" \
		--tag $(TRAINING_IMAGE):$(VERSION) --tag $(TRAINING_IMAGE):latest .
	docker buildx build --platform $(PLATFORMS) --push \
		-f Dockerfile-test \
		--annotation "index:org.opencontainers.image.source=https://github.com/llm-d/$(PROJECT_NAME)" \
		--annotation "index:org.opencontainers.image.licenses=Apache-2.0" \
		--tag $(TEST_IMAGE):$(VERSION) --tag $(TEST_IMAGE):latest .

.PHONY: clean
clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
