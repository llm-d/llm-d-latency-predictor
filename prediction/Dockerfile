FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install prediction dependencies
COPY prediction/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY common ./common
COPY prediction ./prediction

EXPOSE 8001

# Number of worker processes — configurable via env var
ENV UVICORN_WORKERS=8

CMD ["sh", "-c", "uvicorn prediction.prediction_server:app --host 0.0.0.0 --port 8001 --workers $UVICORN_WORKERS"]
