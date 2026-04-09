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

# ── Sync .env with current storage connection string ──────────────────────────
echo "Syncing .env with storage account $STORAGE_ACCOUNT..."
STORAGE_CONN=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString --output tsv 2>/dev/null || true)

if [[ -n "$STORAGE_CONN" ]]; then
  _upsert_env() {
    local key="$1" val="$2" file="$3"
    local escaped_val
    escaped_val=$(printf '%s\n' "$val" | sed 's/[&/\]/\\&/g')
    if grep -q "^${key}=" "$file" 2>/dev/null; then
      sed -i "s|^${key}=.*|${key}=${escaped_val}|" "$file"
    else
      echo "${key}=${val}" >> "$file"
    fi
  }
  _upsert_env "AZURE_STORAGE_CONN" "$STORAGE_CONN" "$APP_ENV"
  _upsert_env "CONTAINER_NAME" "$CONTAINER_NAME" "$APP_ENV"
  echo "      .env updated."
else
  echo "WARNING: Could not retrieve storage connection string — .env unchanged."
fi

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
  --ports 8080 \
  --ip-address Public \
  --environment-variables $ENV_ARGS \
  --output table

DASHBOARD_IP=$(az container show \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query ipAddress.ip --output tsv 2>/dev/null || echo "pending")

echo ""
echo "=========================================="
echo "  Deployment complete!"
echo ""
echo "  Dashboard: http://${DASHBOARD_IP}:8080"
echo ""
echo "  View live logs:"
echo "  az container logs --name $ACI_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "  Check status:"
echo "  az container show --name $ACI_NAME --resource-group $RESOURCE_GROUP --query instanceView.state"
echo "=========================================="
