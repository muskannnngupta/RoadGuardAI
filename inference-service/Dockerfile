# ==============================================================================
# RoadGuard AI — YOLOv8 Inference Microservice Dockerfile
# ==============================================================================
FROM python:3.10-slim

# Prevent generation of .pyc bytecode and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="/app/models/best.pt" \
    CONF_LOWER=0.25 \
    CONF_UPPER=0.50 \
    MODEL_VERSION="yolov8n-rdd2022-v1" \
    MAX_FILE_SIZE_MB=15 \
    HOST=0.0.0.0 \
    PORT=8000

# Install runtime system packages for OpenCV and image decoders
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching layers
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app
COPY tests/ ./tests

# Create models directory (weights mounted at runtime or baked in)
RUN mkdir -p /app/models

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
