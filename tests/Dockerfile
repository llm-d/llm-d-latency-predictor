FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install test dependencies from pyproject.toml
COPY pyproject.toml README.md ./
COPY common ./common
COPY prediction ./prediction
COPY training ./training
RUN pip install --no-cache-dir '.[test]'

# Copy test sources
COPY tests ./tests

RUN mkdir -p /test-results

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["pytest", "-v", "-s", "tests/test_dual_server_client.py"]
