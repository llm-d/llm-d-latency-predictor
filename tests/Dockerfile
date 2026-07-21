# 3.11: xgboost/numpy wheels lag on newer Python; EOL Oct 2027
FROM python:3.11-slim

ARG TARGETARCH

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    jq &&\
    if [ "${TARGETARCH}" = "s390x" ]; then \
        apt-get install -y \
            build-essential \
            gcc \
            g++ \
            gfortran \
            pkg-config \
            cmake \
            ninja-build \
            libopenblas-dev \
            liblapack-dev; \
    fi && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install test dependencies (includes main + test extras)
COPY tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy test sources
COPY VERSION ./
COPY tests ./tests

RUN mkdir -p /test-results

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["pytest", "-v", "-s", "tests/test_dual_server_client.py"]
