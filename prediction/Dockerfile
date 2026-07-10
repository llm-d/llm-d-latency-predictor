# 3.11: xgboost/numpy wheels lag on newer Python; EOL Oct 2027
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY VERSION ./
COPY common ./common
COPY prediction ./prediction

# Pre-create the default model cache directory so the process can write to it
# when running as an arbitrary non-root UID (e.g. OpenShift restricted-v2 SCC).
RUN mkdir -p /local_models && chmod 777 /local_models


EXPOSE 8001

# Number of worker processes — configurable via env var
ENV UVICORN_WORKERS=8

CMD ["sh", "-c", "uvicorn prediction.prediction_server:app --host 0.0.0.0 --port 8001 --workers $UVICORN_WORKERS"]
