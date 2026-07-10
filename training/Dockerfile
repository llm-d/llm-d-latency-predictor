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
COPY training ./training

# Pre-create the model directory so the process can write to it when running
# as an arbitrary non-root UID (e.g. OpenShift restricted-v2 SCC).
RUN mkdir -p /models && chmod 777 /models

EXPOSE 8000

CMD ["uvicorn", "training.training_server:app", "--host", "0.0.0.0", "--port", "8000"]
