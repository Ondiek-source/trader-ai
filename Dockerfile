# ============================================================================
# Trader AI — Binary Options Automated Trading System
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
# Install Python dependencies — separate layers for retry resilience
# If one layer fails, Docker caches the others
# ---------------------------------------------------------------------------

RUN pip install --no-cache-dir --upgrade pip

# Layer 1: CPU-only PyTorch (~200MB, most likely to timeout)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1.0"

# Layer 2: Scientific stack
RUN pip install --no-cache-dir \
    pyarrow>=15.0.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    scipy

# Layer 3: ML frameworks
RUN pip install --no-cache-dir \
    lightgbm>=4.1.0 \
    xgboost>=2.0.0 \
    scikit-learn>=1.4.0

# Layer 4: Everything else (small, fast)
RUN pip install --no-cache-dir \
    websockets>=12.0 \
    azure-storage-blob>=12.19.0 \
    psutil>=5.9.8 \
    requests>=2.31.0 \
    aiohttp>=3.13.2 \
    aiodns>=3.2.0 \
    brotli>=1.1.0 \
    python-telegram-bot>=20.7 \
    httpx>=0.27.0 \
    joblib>=1.3.0 \
    python-dotenv>=1.0.0

# Layer 5: Quotex (from git — failure is non-fatal)
RUN pip install --no-cache-dir \
    "git+https://github.com/cleitonleonel/pyquotex.git" \
    || echo "WARNING: pyquotex install failed — Quotex streaming disabled"

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
