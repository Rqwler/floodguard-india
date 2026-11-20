# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# libgomp1 is required by the CPU PyTorch wheels; curl backs the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so pip does not pull the multi-GB CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<3.0"

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY models ./models
COPY web ./web

RUN mkdir -p data/history data/live data/outbox

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/health" || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
