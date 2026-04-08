#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build and deploy Trader AI entirely in Azure (no Docker needed)
#
# Uses ACR Build Tasks to build the image in the cloud, then deploys to
# Azure Container Instance. Only requires Azure CLI — no Docker Desktop.
#
# Run this every time you want to deploy a new version.
# Requires provision.sh to have been run first.
#
# Usage:
#   bash deploy/deploy.sh
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
  echo "ERROR: .env not found. Copy .env.example and fill it in."
  exit 1
fi

source "$AZURE_ENV"

IMAGE_TAG="${ACR_LOGIN_SERVER}/trader-ai:latest"

echo ""
echo "=========================================="
echo "  Trader AI — Deploy to Azure"
echo "  (no Docker Desktop required)"
echo "=========================================="
echo "Registry : $ACR_LOGIN_SERVER"
echo "Container: $ACI_NAME"
echo ""

# ── 1. Build image in Azure (ACR Build Task) ──────────────────────────────────
echo "[1/3] Building image in Azure Container Registry..."
echo "      (uploads source, builds in cloud — takes 5-10 minutes)"
az acr build \
  --registry "$ACR_NAME" \
  --image "trader-ai:latest" \
  --file "$PROJECT_ROOT/Dockerfile" \
  "$PROJECT_ROOT"
echo "      Done. Image: $IMAGE_TAG"

# ── 2. Parse .env into ACI environment variables ──────────────────────────────
echo "[2/3] Reading .env..."
ENV_ARGS=""
while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line// }" ]] && continue
  KEY="${line%%=*}"
  VAL="${line#*=}"
  [[ -z "$VAL" ]] && continue
  # Escape values with spaces by quoting
  ENV_ARGS="$ENV_ARGS $KEY=$VAL"
done < "$APP_ENV"

# ── 3. Deploy to Azure Container Instance ─────────────────────────────────────
echo "[3/3] Deploying container instance..."

if az container show \
     --name "$ACI_NAME" \
     --resource-group "$RESOURCE_GROUP" \
     --output none 2>/dev/null; then
  echo "      Removing existing container..."
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
echo "  View live logs:"
echo "  az container logs --name $ACI_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "  Check status:"
echo "  az container show --name $ACI_NAME --resource-group $RESOURCE_GROUP --query instanceView.state"
echo "=========================================="
