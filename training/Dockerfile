# 3.11: xgboost/numpy wheels lag on newer Python; EOL Oct 2027
FROM python:3.11-slim

WORKDIR /app

ARG TARGETARCH


RUN apt-get update && apt-get install -y \
    libgomp1 && \
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


# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY common ./common
COPY training ./training

# Pre-create the model directory so the process can write to it when running
# as an arbitrary non-root UID (e.g. OpenShift restricted-v2 SCC).
RUN mkdir -p /models && chmod 777 /models

EXPOSE 8000

# Optional TLS — set SSL_KEYFILE and SSL_CERTFILE (and optionally SSL_CA_CERTS) to enable HTTPS.
# Mount certificate files as a Kubernetes volume and point these env vars at the mount path.
ENV SSL_KEYFILE=""
ENV SSL_CERTFILE=""
ENV SSL_CA_CERTS=""

CMD ["uvicorn", "training.training_server:app", "--host", "0.0.0.0", "--port", "8000"]
