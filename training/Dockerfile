FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the package with training extras
COPY pyproject.toml README.md ./
COPY common ./common
COPY training ./training
RUN pip install --no-cache-dir '.[training]'

EXPOSE 8000

CMD ["uvicorn", "training.training_server:app", "--host", "0.0.0.0", "--port", "8000"]
