#!/usr/bin/env bash
# =============================================================================
# provision.sh — One-time Azure infrastructure setup for Trader AI
#
# Creates:
#   - Resource group
#   - Storage account + blob container (tick data / model files)
#   - Azure Container Registry (Docker image store)
#   - Windows VM (trading bot host — stays on 24/7, no screen lock)
#
# Run this ONCE. After it completes, copy the printed values into your .env.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#
# Usage:
#   bash deploy/provision.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
RESOURCE_GROUP="rg-trader-ai"
LOCATION="eastus"

# Storage (name must be globally unique, lowercase, 3-24 chars)
RAND=$((RANDOM % 9000 + 1000))
STORAGE_ACCOUNT="traderai${RAND}"
CONTAINER_NAME="traderai"

# Container Registry
ACR_NAME="traderaireg${RAND}"

# Azure Container Instance (AI engine)
ACI_NAME="trader-ai-engine"
ACI_CPU=2
ACI_MEMORY=4

# Windows VM (trading bot)
VM_NAME="trader-bot-vm"
VM_SIZE="Standard_B2ms"          # 2 vCPU, 8 GB RAM — comfortably above 6 GB min
VM_IMAGE="Win2022Datacenter"
VM_ADMIN_USER="traderadmin"
# Strong random password
VM_ADMIN_PASS="TraderAI2024xAz"

echo ""
echo "=========================================="
echo "  Trader AI — Azure Provisioning"
echo "=========================================="
echo "Resource Group : $RESOURCE_GROUP"
echo "Location       : $LOCATION"
echo "Storage        : $STORAGE_ACCOUNT"
echo "Registry       : $ACR_NAME"
echo "AI Container   : $ACI_NAME  (Linux, ${ACI_CPU} vCPU / ${ACI_MEMORY}GB)"
echo "Bot VM         : $VM_NAME  ($VM_SIZE — Windows Server 2022, 8GB RAM)"
echo ""

# ── 1. Resource Group ─────────────────────────────────────────────────────────
echo "[1/6] Creating resource group $RESOURCE_GROUP..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none 2>/dev/null || echo "      (already exists, continuing)"
echo "      Done."

# ── 2. Storage Account ────────────────────────────────────────────────────────
# Re-use existing storage account if already provisioned
EXISTING_STORAGE=$(az storage account list \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].name" --output tsv 2>/dev/null || true)
if [[ -n "$EXISTING_STORAGE" ]]; then
  STORAGE_ACCOUNT="$EXISTING_STORAGE"
  echo "[2/6] Using existing storage account $STORAGE_ACCOUNT..."
else
  echo "[2/6] Creating storage account $STORAGE_ACCOUNT..."
  az storage account create \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --access-tier Hot \
    --output none
fi

STORAGE_CONN=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString \
  --output tsv)

az storage container create \
  --name "$CONTAINER_NAME" \
  --connection-string "$STORAGE_CONN" \
  --output none
echo "      Done."

# ── 3. Azure Container Registry ───────────────────────────────────────────────
echo "[3/6] Creating container registry $ACR_NAME..."
az acr create \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --sku Basic \
  --admin-enabled true \
  --output none

ACR_LOGIN_SERVER=$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query loginServer \
  --output tsv)

ACR_USERNAME=$(az acr credential show \
  --name "$ACR_NAME" \
  --query username \
  --output tsv)

ACR_PASSWORD=$(az acr credential show \
  --name "$ACR_NAME" \
  --query "passwords[0].value" \
  --output tsv)
echo "      Done. Registry: $ACR_LOGIN_SERVER"

# ── 4. Windows VM for trading bot ─────────────────────────────────────────────
EXISTING_VM=$(az vm show --name "$VM_NAME" --resource-group "$RESOURCE_GROUP" \
  --query name --output tsv 2>/dev/null || true)

if [[ -n "$EXISTING_VM" ]]; then
  echo "[4/6] Using existing VM $VM_NAME..."
  VM_PUBLIC_IP=$(az vm show \
    --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --show-details \
    --query publicIps \
    --output tsv)
else
  echo "[4/6] Creating Windows VM $VM_NAME (this takes ~3 minutes)..."
  az vm create \
    --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --image "$VM_IMAGE" \
    --size "$VM_SIZE" \
    --admin-username "$VM_ADMIN_USER" \
    --admin-password "$VM_ADMIN_PASS" \
    --public-ip-sku Standard

  az vm open-port \
    --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --port 3389 \
    --output none

  VM_PUBLIC_IP=$(az vm show \
    --name "$VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --show-details \
    --query publicIps \
    --output tsv)
fi
echo "      Done. VM IP: $VM_PUBLIC_IP"

# ── 5. Disable VM screen lock via auto-shutdown disable + idle settings ───────
echo "[5/6] Configuring VM to never sleep/lock..."
# Set VM to not auto-shutdown
az vm auto-shutdown \
  --name "$VM_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --off \
  --output none 2>/dev/null || true
echo "      Done."

# ── 6. Save config to deploy/azure.env ────────────────────────────────────────
echo "[6/6] Writing deploy/azure.env..."
cat > "$(dirname "$0")/azure.env" <<EOF
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
VM_PUBLIC_IP=$VM_PUBLIC_IP
VM_ADMIN_USER=$VM_ADMIN_USER
VM_ADMIN_PASS=$VM_ADMIN_PASS
EOF
echo "      Done."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Add these to your .env file:"
echo "=========================================="
echo ""
echo "AZURE_STORAGE_CONN=$STORAGE_CONN"
echo "CONTAINER_NAME=$CONTAINER_NAME"
echo ""
echo "=========================================="
echo "  Windows VM (trading bot):"
echo "  RDP to: $VM_PUBLIC_IP"
echo "  User  : $VM_ADMIN_USER"
echo "  Pass  : $VM_ADMIN_PASS"
echo ""
echo "  SAVE THESE CREDENTIALS — they won't be shown again."
echo "=========================================="
echo ""
echo "  Next step: run bash deploy/deploy.sh"
echo "=========================================="
