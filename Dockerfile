# ============================================================================
# Trader AI — Binary Options Automated Trading System
# ============================================================================
FROM python:3.13-slim

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src"

# ── System deps + non-root user ──────────────────────────────────────────────
# build-essential and git removed — no packages require compilation and
# nothing clones from git at build time.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Heavy ML stack first — changes least often, maximises layer cache reuse.
# GPU wheels: swap cpu→cu121 in the extra-index-url when GPU_ENABLED=true.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 10 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "catboost>=1.2.0" \
    "gymnasium>=0.29.0" \
    "lightgbm>=4.3.0" \
    "numpy>=1.26.0" \
    "pandas>=2.2.0" \
    "pyarrow>=15.0.0" \
    "sb3-contrib>=2.3.0" \
    "scikit-learn>=1.4.0" \
    "stable-baselines3>=2.3.0" \
    "torch>=2.1.0" \
    "xgboost>=2.0.0" 


# Networking / infra stack — changes occasionally.
RUN pip install --no-cache-dir --retries 10 \
    "aiohttp>=3.13.2" \
    "azure-storage-blob>=12.19.0" \
    "beautifulsoup4>=4.12.3" \
    "colorama==0.4.6" \
    "fake-useragent==2.2.0" \
    "joblib>=1.3.0" \
    "orjson>=3.9.0" \
    "psutil>=5.9.8" \
    "pyfiglet>=1.0.2" \
    "python-dotenv>=1.0.0" \
    "python-telegram-bot>=20.7" \
    "requests>=2.31.0" \
    "websocket-client>=1.8.0" \
    "websockets>=12.0" 


# ── pyquotex (patched local fork) ────────────────────────────────────────────
# Copied before full source so this layer survives unrelated source changes.
# Installed as a regular package (not -e) to avoid .pth / PYTHONPATH conflicts.
COPY --chown=appuser:appuser src/pyquotex /app/src/pyquotex
RUN pip install --no-cache-dir poetry-core && \
    pip install --no-cache-dir --no-deps /app/src/pyquotex && \
    mkdir -p /app/data/raw /app/data/processed /app/models /tmp/fallback && \
    chown -R appuser:appuser /app /tmp/fallback

# ── Source code ───────────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY --chown=appuser:appuser deploy/entrypoint.sh /entrypoint.sh

# ── Lock source permissions ───────────────────────────────────────────────────
RUN chmod +x /entrypoint.sh && \
    chmod -R a-w /app/src /app/deploy /app/requirements.txt /app/Dockerfile 2>/dev/null || true

# ── Runtime ───────────────────────────────────────────────────────────────────
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf "http://localhost:${DASHBOARD_PORT:-8080}/status" > /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]