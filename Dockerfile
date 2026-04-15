# ============================================================================
# Trader AI — Binary Options Automated Trading System
# Multi-stage Docker build optimized for Python 3.13 & Dictator Pattern
# ============================================================================

FROM python:3.13-slim AS base

# 1. Environment Sanitization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src"

# 2. System Dependencies (Required for building native extensions & git installs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app

# 4. Python Dependency Layers
RUN pip install --no-cache-dir --upgrade pip

# Layer 1: Heavy ML & Data Processing
# Note: pyarrow is required for your Snappy-compressed Parquet strategy
# --extra-index-url https://download.pytorch.org/whl/cu121 GPU wheels if you enable GPU support in the future
RUN pip install --no-cache-dir --retries 10 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1.0" \
    "pandas>=2.2.0" \
    "pyarrow>=15.0.0" \
    "numpy>=1.26.0" \
    "scikit-learn>=1.4.0" \
    "lightgbm>=4.3.0" \
    "xgboost>=2.0.0" \
    "catboost>=1.2.0" \
    "stable-baselines3>=2.3.0" \
    "sb3-contrib>=2.3.0" \
    "gymnasium>=0.29.0"

# Layer 2: API, Networking & Utilities
RUN pip install --no-cache-dir --retries 10 \
    "websockets>=12.0" \
    "azure-storage-blob>=12.19.0" \
    "psutil>=5.9.8" \
    "requests>=2.31.0" \
    "aiohttp>=3.13.2" \
    "python-dotenv>=1.0.0" \
    "python-telegram-bot>=20.7" \
    "joblib>=1.3.0"

# Layer 3: Broker Integration (Git-based)
RUN pip install --no-cache-dir \
    "git+https://github.com/cleitonleonel/pyquotex.git"

# 5. Infrastructure Provisioning
# We create the folders before copying source to ensure 'appuser' ownership 
# and pass the Storage.py write-check.
RUN mkdir -p /app/data/raw /app/data/processed /app/models /tmp/fallback && \
    chown -R appuser:appuser /app /tmp/fallback

# 6. Source Code Deployment
# Using --chown here is faster and more secure than a separate RUN chown
COPY --chown=appuser:appuser . .

# 7. Runtime Configuration
USER appuser

# Healthcheck hits the dashboard /status endpoint (curl is installed above).
# pgrep is not available in python:3.13-slim without procps.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8080/status > /dev/null || exit 1

CMD ["python", "src/main.py"]