# ==============================================================================
# Multi-stage Production Dockerfile for Real-Estate Price Prediction Service
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/models /app/reports /app/config /app/logs && \
    chown -R appuser:appuser /app

COPY --from=builder /root/.local /home/appuser/.local

# Copy application source code and active model artifacts
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser models/active_model.joblib /app/models/active_model.joblib
COPY --chown=appuser:appuser models/preprocessor.joblib /app/models/preprocessor.joblib
COPY --chown=appuser:appuser models/active_model_metadata.json /app/models/active_model_metadata.json

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
