# ============================================================================
# Trader AI -- Binary Options Automated Trading System
# Multi-stage Docker build for Python 3.12
# ============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system dependencies required by native pip packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Create non-root application user
# ---------------------------------------------------------------------------
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies (cached layer)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "git+https://github.com/cleitonleonel/pyquotex.git" || \
    echo "WARNING: pyquotex install failed — Quotex result reading disabled"

# ---------------------------------------------------------------------------
# Copy application source
# ---------------------------------------------------------------------------
COPY src/ ./src/

# Model persistence and fallback storage directories
RUN mkdir -p /app/models /tmp/fallback && chown -R appuser:appuser /app /tmp/fallback

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
USER appuser

EXPOSE 8080

ENV PYTHONPATH=/app/src

# Health check: hits the local result-receiver /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "src/main.py"]
