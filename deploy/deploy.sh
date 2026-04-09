#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy the latest image from ACR to Azure Container Instance
#
# Run this locally after GitHub Actions has finished building the image.
# Requires: Azure CLI + az login
#
# Check CI is done first:
#   gh run list --repo Ondiek-source/trader-ai --limit 3
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
  echo "ERROR: .env not found."
  exit 1
fi

source "$AZURE_ENV"

IMAGE_TAG="${ACR_LOGIN_SERVER}/trader-ai:latest"

# ── Sync GitHub secrets with latest ACR credentials ───────────────────────────
echo "Syncing GitHub secrets from azure.env..."
gh secret set ACR_LOGIN_SERVER --repo Ondiek-source/trader-ai --body "$ACR_LOGIN_SERVER"
gh secret set ACR_USERNAME     --repo Ondiek-source/trader-ai --body "$ACR_USERNAME"
gh secret set ACR_PASSWORD     --repo Ondiek-source/trader-ai --body "$ACR_PASSWORD"
echo "      Done."

# ── Ensure required providers are registered ──────────────────────────────────
for provider in Microsoft.ContainerInstance Microsoft.ContainerRegistry; do
  STATE=$(az provider show --namespace "$provider" --query registrationState --output tsv 2>/dev/null || echo "NotRegistered")
  if [[ "$STATE" != "Registered" ]]; then
    echo "Registering $provider (one-time, ~1 minute)..."
    az provider register --namespace "$provider" --wait
    echo "      Done."
  fi
done

echo ""
echo "=========================================="
echo "  Trader AI — Deploy to Azure Container Instance"
echo "=========================================="
echo "Image    : $IMAGE_TAG"
echo "Container: $ACI_NAME"
echo ""

# ── Parse .env into environment variables for ACI ─────────────────────────────
echo "[1/2] Reading .env..."
ENV_ARGS=""
while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line// }" ]] && continue
  KEY="${line%%=*}"
  VAL="${line#*=}"
  [[ -z "$VAL" ]] && continue
  ENV_ARGS="$ENV_ARGS $KEY=$VAL"
done < "$APP_ENV"

# ── Deploy to ACI ─────────────────────────────────────────────────────────────
echo "[2/2] Deploying container instance..."

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
