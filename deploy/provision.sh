#!/usr/bin/env bash
# =============================================================================
# provision.sh — One-time Azure infrastructure setup for Trader AI
#
# Idempotent: safe to re-run — skips resources that already exist.
#
# Creates:
#   - Resource group (uksouth)
#   - Storage account + blob container (tick data / model files)
#   - Azure Container Registry (Docker image store)
#   - Windows VM 8GB RAM (trading bot host — 24/7, no screen lock)
#
# Usage:
#   bash deploy/provision.sh
# =============================================================================

set -euo pipefail

# ── Configuration — ALL resources in uksouth ──────────────────────────────────
RESOURCE_GROUP="rg-trader-ai"
LOCATION="southafricanorth"

CONTAINER_NAME="traderai"
ACI_NAME="trader-ai-engine"
ACI_CPU=2
ACI_MEMORY=8

VM_NAME="trader-bot-vm"
VM_LOCATION="southafricanorth"  # same region — Quotex accessible from here
VM_SIZE="Standard_D2s_v3"    # 2 vCPU, 8 GB RAM
VM_IMAGE="MicrosoftWindowsServer:WindowsServer:2022-Datacenter:latest"
VM_ADMIN_USER="traderadmin"
VM_ADMIN_PASS="TraderAI-2024xAz"

echo ""
echo "=========================================="
echo "  Trader AI — Azure Provisioning"
echo "  Region: $LOCATION (all resources)"
echo "=========================================="
echo ""

# ── 1. Resource Group ─────────────────────────────────────────────────────────
echo "[1/6] Resource group $RESOURCE_GROUP..."
if az group show --name "$RESOURCE_GROUP" --output none 2>/dev/null; then
  echo "      Already exists — skipping."
else
  az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
  echo "      Created."
fi

# ── 2. Storage Account ────────────────────────────────────────────────────────
echo "[2/6] Storage account..."
EXISTING_STORAGE=$(az storage account list \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].name" --output tsv 2>/dev/null || true)

if [[ -n "$EXISTING_STORAGE" ]]; then
  STORAGE_ACCOUNT="$EXISTING_STORAGE"
  echo "      Using existing: $STORAGE_ACCOUNT"
else
  RAND=$((RANDOM % 9000 + 1000))
  STORAGE_ACCOUNT="traderai${RAND}"
  az storage account create \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none
  echo "      Created: $STORAGE_ACCOUNT"
fi

STORAGE_CONN=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString \
  --output tsv)

az storage container create \
  --name "$CONTAINER_NAME" \
  --connection-string "$STORAGE_CONN" \
  --output none 2>/dev/null || true
echo "      Blob container '$CONTAINER_NAME' ready."

# ── 3. Container Registry ─────────────────────────────────────────────────────
echo "[3/6] Container registry..."
EXISTING_ACR=$(az acr list \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].name" --output tsv 2>/dev/null || true)

if [[ -n "$EXISTING_ACR" ]]; then
  ACR_NAME="$EXISTING_ACR"
  echo "      Using existing: $ACR_NAME"
else
  RAND=$((RANDOM % 9000 + 1000))
  ACR_NAME="traderaireg${RAND}"
  az acr create \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Basic \
    --admin-enabled true \
    --output none
  echo "      Created: $ACR_NAME — waiting 30s for propagation..."
  sleep 30
fi

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" \
  --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" \
  --query "passwords[0].value" --output tsv)
echo "      Registry: $ACR_LOGIN_SERVER"

# ── 4. Windows VM ─────────────────────────────────────────────────────────────
echo "[4/6] Windows VM $VM_NAME ($VM_SIZE)..."
EXISTING_VM=$(az vm show --name "$VM_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query name --output tsv 2>/dev/null || true)

if [[ -n "$EXISTING_VM" ]]; then
  echo "      Already exists — skipping."
  VM_PUBLIC_IP=$(az vm show --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --show-details --query publicIps --output tsv)
else
  echo "      Creating (3-5 minutes)..."
  az vm create \
    --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$VM_LOCATION" \
    --image "$VM_IMAGE" \
    --size "$VM_SIZE" \
    --admin-username "$VM_ADMIN_USER" \
    --admin-password "$VM_ADMIN_PASS" \
    --public-ip-sku Standard \
    --output table

  az vm open-port \
    --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --port 3389 \
    --output none

  VM_PUBLIC_IP=$(az vm show --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --show-details --query publicIps --output tsv)
  echo "      Created. IP: $VM_PUBLIC_IP"
fi

# ── 5. Disable VM auto-shutdown ───────────────────────────────────────────────
echo "[5/6] Disabling VM auto-shutdown..."
az vm auto-shutdown \
  --name "$VM_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --off \
  --output none 2>/dev/null || true
echo "      Done."

# ── 6. Write azure.env ────────────────────────────────────────────────────────
echo "[6/6] Writing deploy/azure.env..."
SCRIPT_DIR_ABS="$(cd "$(dirname "$0")" && pwd)"
cat > "$SCRIPT_DIR_ABS/azure.env" <<EOF
# Auto-generated by provision.sh — DO NOT COMMIT
RESOURCE_GROUP=$RESOURCE_GROUP
LOCATION=$LOCATION
STORAGE_ACCOUNT=$STORAGE_ACCOUNT
CONTAINER_NAME=$CONTAINER_NAME
ACR_NAME=$ACR_NAME
ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER
ACR_USERNAME=$ACR_USERNAME
ACR_PASSWORD=$ACR_PASSWORD
ACI_NAME=$ACI_NAME
ACI_CPU=$ACI_CPU
ACI_MEMORY=$ACI_MEMORY
VM_NAME=$VM_NAME
VM_PUBLIC_IP=${VM_PUBLIC_IP:-unknown}
VM_ADMIN_USER=$VM_ADMIN_USER
VM_ADMIN_PASS=$VM_ADMIN_PASS
EOF
echo "      Done."

# ── 6b. Auto-sync .env with storage connection string ─────────────────────────
APP_ENV="$SCRIPT_DIR_ABS/../.env"
echo "[6b] Updating .env with storage connection string..."

if [[ ! -f "$APP_ENV" ]]; then
  touch "$APP_ENV"
fi

_upsert_env() {
  local key="$1" val="$2" file="$3"
  # Escape for sed: replace / with \/ in value
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
echo "      .env updated (AZURE_STORAGE_CONN, CONTAINER_NAME)."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Windows VM — RDP credentials:"
echo "  IP   : ${VM_PUBLIC_IP:-see Azure portal}"
echo "  User : $VM_ADMIN_USER"
echo "  Pass : $VM_ADMIN_PASS"
echo "  SAVE THESE — they won't be shown again."
echo "=========================================="
echo ""
echo "  Next: run bash deploy/deploy.sh"
echo "=========================================="
