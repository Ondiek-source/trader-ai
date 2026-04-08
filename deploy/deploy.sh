#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build, push, and run the Trader AI container on Azure
#
# Run this every time you want to deploy a new version.
# Requires provision.sh to have been run first.
#
# Usage:
#   chmod +x deploy/deploy.sh
#   ./deploy/deploy.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"
APP_ENV="$PROJECT_ROOT/.env"

if [[ ! -f "$AZURE_ENV" ]]; then
  echo "ERROR: deploy/azure.env not found. Run provision.sh first."
  exit 1
fi

if [[ ! -f "$APP_ENV" ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill it in."
  exit 1
fi

# Load azure config
source "$AZURE_ENV"

IMAGE_TAG="${ACR_LOGIN_SERVER}/trader-ai:latest"

echo ""
echo "=========================================="
echo "  Trader AI — Deploy to Azure"
echo "=========================================="
echo "Registry : $ACR_LOGIN_SERVER"
echo "Container: $ACI_NAME"
echo ""

# ── 1. Build Docker image ─────────────────────────────────────────────────────
echo "[1/4] Building Docker image..."
docker build \
  --platform linux/amd64 \
  -t "trader-ai:latest" \
  -f "$PROJECT_ROOT/Dockerfile" \
  "$PROJECT_ROOT"
echo "      Done."

# ── 2. Push to Azure Container Registry ──────────────────────────────────────
echo "[2/4] Pushing image to ACR..."
docker tag "trader-ai:latest" "$IMAGE_TAG"
echo "$ACR_PASSWORD" | docker login "$ACR_LOGIN_SERVER" \
  --username "$ACR_USERNAME" \
  --password-stdin
docker push "$IMAGE_TAG"
echo "      Done."

# ── 3. Load .env into ACI environment variables ───────────────────────────────
echo "[3/4] Reading .env for container environment..."

# Parse .env into --environment-variables string (skip comments and blanks)
ENV_ARGS=""
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ ]] && continue
  [[ -z "$line" ]] && continue
  KEY="${line%%=*}"
  VAL="${line#*=}"
  # Skip empty values
  [[ -z "$VAL" ]] && continue
  ENV_ARGS="$ENV_ARGS $KEY=$VAL"
done < "$APP_ENV"

# ── 4. Deploy / update Azure Container Instance ───────────────────────────────
echo "[4/4] Deploying container instance..."

# Delete existing instance if present (ACI doesn't support in-place image update)
if az container show \
     --name "$ACI_NAME" \
     --resource-group "$RESOURCE_GROUP" \
     --output none 2>/dev/null; then
  echo "      Removing existing container instance..."
  az container delete \
    --name "$ACI_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --yes \
    --output none
fi

# shellcheck disable=SC2086
az container create \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$IMAGE_TAG" \
  --registry-login-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --cpu "$ACI_CPU" \
  --memory "$ACI_MEMORY" \
  --os-type Linux \
  --restart-policy Always \
  --environment-variables $ENV_ARGS \
  --output table

echo ""
echo "=========================================="
echo "  Deployment complete!"
echo ""
echo "  View logs:"
echo "  az container logs --name $ACI_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "  Check status:"
echo "  az container show --name $ACI_NAME --resource-group $RESOURCE_GROUP --query instanceView.state"
echo "=========================================="
