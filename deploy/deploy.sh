#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy the latest image from ACR to Azure Container Instance
#
# Run this locally after GitHub Actions has finished building the image.
# Requires: Azure CLI (logged in), GitHub CLI (authenticated)
#
# Check CI is done first:
#   gh run list --repo "$GITHUB_REPO" --limit 3
#
# Usage:
#   bash deploy/deploy.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"
APP_ENV="$PROJECT_ROOT/.env"
GITHUB_REPO="${GITHUB_REPO:-Ondiek-source/trader-ai}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"
DNS_LABEL="${DNS_LABEL:-trader-ai-bot}"

# ── Prerequisites ─────────────────────────────────────────────────────────────

for cmd in az gh; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not installed."
    exit 1
  fi
done

if ! az account show --output none 2>/dev/null; then
  echo "ERROR: Not logged into Azure. Run 'az login' first."
  exit 1
fi

if [[ ! -f "$AZURE_ENV" ]]; then
  echo "ERROR: $AZURE_ENV not found. Run provision.sh first."
  exit 1
fi

if [[ ! -f "$APP_ENV" ]]; then
  echo "ERROR: $APP_ENV not found."
  exit 1
fi

source "$AZURE_ENV"

_REQUIRED_VARS=(
  ACR_NAME ACR_LOGIN_SERVER ACR_USERNAME ACR_PASSWORD
  STORAGE_ACCOUNT CONTAINER_NAME RESOURCE_GROUP LOCATION
  ACI_NAME ACI_CPU ACI_MEMORY
)
_MISSING=()
for var in "${_REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    _MISSING+=("$var")
  fi
done
if (( ${#_MISSING[@]} )); then
  echo "ERROR: Missing required variables in azure.env: ${_MISSING[*]}"
  exit 1
fi

IMAGE_TAG="${ACR_LOGIN_SERVER}/trader-ai:latest"

# ── Pre-flight: verify Azure subscription ──────────────────────────
REQUIRED_SUB="9086726b-ad07-4df0-90de-2ef7b6f6389b"

CURRENT_SUB=$(az account show --query "id" -o tsv 2>/dev/null || true)

if [ -z "$CURRENT_SUB" ]; then
  echo "Not logged in — running az login..."
  az login
  CURRENT_SUB=$(az account show --query "id" -o tsv)
fi

if [ "$CURRENT_SUB" != "$REQUIRED_SUB" ]; then
  echo "Switching subscription: $CURRENT_SUB → $REQUIRED_SUB"
  az account set --subscription "$REQUIRED_SUB"
  echo "✓ Switched."
fi

# Verify resource group exists
if ! az group show -n "$RESOURCE_GROUP" &>/dev/null; then
  echo "❌ Resource group '$RESOURCE_GROUP' not found"
  exit 1
fi

echo "✓ Subscription: $REQUIRED_SUB"
echo "✓ Resource group: $RESOURCE_GROUP"


# ── Helper: upsert a key=value line into a .env file ──────────────────────────

_upsert_env() {
  local key="$1" val="$2" file="$3"
  local tmp
  tmp=$(mktemp)
  grep -v "^${key}=" "$file" > "$tmp" 2>/dev/null || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" "$file"
}

# ── Cleanup: old ACR images ───────────────────────────────────────────────────

cleanup_acr() {
  echo "Cleaning up old ACR images (keeping latest 2 tags + 'latest')..."

  local all_tags
  all_tags=$(az acr repository show-tags \
    --name "$ACR_NAME" \
    --repository trader-ai \
    --orderby time_desc \
    --query "[]" -o tsv 2>/dev/null)

  if [[ -z "$all_tags" ]]; then
    echo "  No tags found — nothing to delete."
    return
  fi

  local tag_count
  tag_count=$(echo "$all_tags" | wc -l | tr -d ' ')

  if (( tag_count > 2 )); then
    echo "$all_tags" | tail -n +3 | while IFS= read -r tag; do
      [[ -z "$tag" || "$tag" == "latest" ]] && continue
      echo "  Deleting: trader-ai:$tag"
      if ! az acr repository delete \
          --name "$ACR_NAME" \
          --image "trader-ai:$tag" \
          --yes \
          --output none; then
        echo "  ⚠ Failed to delete trader-ai:$tag"
      fi
    done
  fi

  echo "  Purging untagged manifests..."
  az acr run --registry "$ACR_NAME" \
    --cmd 'acr purge --filter "trader-ai:.*" --untagged --ago 7d' \
    /dev/null || echo "  ⚠ Purge command failed"

  echo "  ACR cleanup complete."
}


# ── Step 0: Sync .env with current storage connection string ──────────────────

echo "Syncing .env with storage account $STORAGE_ACCOUNT..."

STORAGE_CONN=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString --output tsv 2>/dev/null || true)

if [[ -n "$STORAGE_CONN" ]]; then
  _upsert_env "AZURE_STORAGE_CONN" "$STORAGE_CONN" "$APP_ENV"
  _upsert_env "CONTAINER_NAME"     "$CONTAINER_NAME" "$APP_ENV"
  echo "      .env updated."
else
  echo "WARNING: Could not retrieve storage connection string — .env unchanged."
fi

# ── Step 1: Sync GitHub secrets with latest ACR credentials ───────────────────

echo "Syncing GitHub secrets..."
gh secret set ACR_LOGIN_SERVER --repo "$GITHUB_REPO" --body "$ACR_LOGIN_SERVER"
gh secret set ACR_USERNAME     --repo "$GITHUB_REPO" --body "$ACR_USERNAME"
gh secret set ACR_PASSWORD     --repo "$GITHUB_REPO" --body "$ACR_PASSWORD"
echo "      Done."

# ── Step 2: Ensure required providers are registered ──────────────────────────

for provider in Microsoft.ContainerInstance Microsoft.ContainerRegistry; do
  state=$(az provider show --namespace "$provider" \
            --query registrationState --output tsv 2>/dev/null || echo "NotRegistered")
  if [[ "$state" != "Registered" ]]; then
    echo "Registering $provider (one-time, ~1 minute)..."
    az provider register --namespace "$provider" --wait
    echo "      Done."
  fi
done

# ── Step 3 (optional): Clean up old ACR images and logs ──────────────────────
# Uncomment to enable:
# cleanup_acr

# ── Step 4: Parse .env into plain and secure environment variables ─────────────

echo "[1/2] Reading .env..."

# Keys whose values should be hidden in portal / az container show output.
_SECURE_KEYS=(
  AZURE_STORAGE_CONN
  QUOTEX_PASSWORD
  TELEGRAM_TOKEN
  TWELVEDATA_API_KEY
  DISCORD_WEBHOOK_URL
  WEBHOOK_URL
  QUOTEX_EMAIL
  TELEGRAM_CHAT_ID
)

_is_secure_key() {
  local check="$1"
  for sk in "${_SECURE_KEYS[@]}"; do
    [[ "$sk" == "$check" ]] && return 0
  done
  return 1
}

_ENV_KV_PAIRS=()
_SECURE_KV_PAIRS=()

while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  line="${line#"${line%%[![:space:]]*}"}"   # lstrip
  line="${line%"${line##*[![:space:]]}"}"   # rstrip
  [[ -z "$line" ]] && continue
  [[ "$line" != *=* ]] && continue

  key="${line%%=*}"
  val="${line#*=}"

  # Strip optional surrounding double-quotes
  if [[ "$val" =~ ^\".*\"$ ]]; then
    val="${val:1:${#val}-2}"
  fi

  [[ -z "$val" ]] && continue

  # Validate key: must start with letter or underscore, rest alphanumeric or underscore
  if ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "  SKIPPING invalid key name: '$key'"
    continue
  fi

  if _is_secure_key "$key"; then
    _SECURE_KV_PAIRS+=("${key}=${val}")
  else
    _ENV_KV_PAIRS+=("${key}=${val}")
  fi
done < "$APP_ENV"

if (( ${#_ENV_KV_PAIRS[@]} == 0 && ${#_SECURE_KV_PAIRS[@]} == 0 )); then
  echo "WARNING: No environment variables parsed from $APP_ENV"
fi

echo "  Plain vars : ${#_ENV_KV_PAIRS[@]}"
echo "  Secure vars: ${#_SECURE_KV_PAIRS[@]}"

# ── Step 5: Deploy to ACI ─────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  Trader AI — Deploy to Azure Container Instance"
echo "=========================================="
echo "Image    : $IMAGE_TAG"
echo "Container: $ACI_NAME"
echo "Port     : $CONTAINER_PORT"
echo ""

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

# ── Step 5.5: Setup Log Analytics for container logs ──────────────────────────

# echo "Setting up Log Analytics for container logging..."

# # Check if Log Analytics workspace exists, delete if found
# if az monitor log-analytics workspace show \
#     --resource-group "$RESOURCE_GROUP" \
#     --workspace-name "trader-ai-logs" &>/dev/null; then
#   echo "  Deleting existing Log Analytics workspace..."
#   az monitor log-analytics workspace delete \
#     --resource-group "$RESOURCE_GROUP" \
#     --workspace-name "trader-ai-logs" \
#     --force \
#     --yes \
#     --output none
#   echo "  ✓ Workspace deleted."
# fi

# # Create fresh workspace
# echo "  Creating new Log Analytics workspace..."
# az monitor log-analytics workspace create \
#   --resource-group "$RESOURCE_GROUP" \
#   --workspace-name "trader-ai-logs" \
#   --location "$LOCATION" \
#   --output none

# # Wait a few seconds for workspace to be ready
# sleep 5
# echo "  ✓ Fresh workspace created."

# # Get workspace ID and key
# WORKSPACE_ID=$(az monitor log-analytics workspace show \
#   --resource-group "$RESOURCE_GROUP" \
#   --workspace-name "trader-ai-logs" \
#   --query customerId -o tsv)

# WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
#   --resource-group "$RESOURCE_GROUP" \
#   --workspace-name "trader-ai-logs" \
#   --query primarySharedKey -o tsv)

# if [[ -n "$WORKSPACE_ID" && -n "$WORKSPACE_KEY" ]]; then
#   echo "  ✓ Log Analytics configured with fresh workspace."
# else
#   echo "  WARNING: Could not get Log Analytics keys - logs will not be persisted."
# fi

# ── Step 6: Deploy to ACI ─────────────────────────────────────────────────────

_CREATE_CMD=(
  az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACI_NAME" \
    --image "$IMAGE_TAG" \
    --registry-login-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --cpu "$ACI_CPU" \
    --memory "$ACI_MEMORY" \
    --os-type Linux \
    --ports "$CONTAINER_PORT" \
    --ip-address Public \
    --dns-name-label "$DNS_LABEL" \
    --environment-variables "${_ENV_KV_PAIRS[@]}" \
    --secure-environment-variables "${_SECURE_KV_PAIRS[@]}" \
    --location "$LOCATION" \
    --restart-policy Never \
    --output table
  )

    echo "=== DIAGNOSTIC: .env file location ==="
    echo "APP_ENV resolved to: $APP_ENV"
    echo "File exists: $(test -f "$APP_ENV" && echo YES || echo NO)"
    echo ""
    echo "=== DIAGNOSTIC: _ENV_KV_PAIRS (${#_ENV_KV_PAIRS[@]} items) ==="
    echo "=== DIAGNOSTIC: _SECURE_KV_PAIRS (${#_SECURE_KV_PAIRS[@]} items) ==="
    echo "=========================================="
    # Add Log Analytics if configured
    # if [[ -n "$WORKSPACE_ID" && -n "$WORKSPACE_KEY" ]]; then
    #   _CREATE_CMD+=(--log-analytics-workspace "$WORKSPACE_ID")
    #   _CREATE_CMD+=(--log-analytics-workspace-key "$WORKSPACE_KEY")
    # fi


"${_CREATE_CMD[@]}"

# ── Summary ────────────────────────────────────────────────────────────────────

DASHBOARD_IP=$(az container show \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query ipAddress.ip --output tsv 2>/dev/null || echo "pending")

DNS_FQDN=$(az container show \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query ipAddress.fqdn --output tsv 2>/dev/null || echo "")

echo ""
echo "=========================================="
echo "  Deployment complete!"
echo ""
echo "  Dashboard : http://${DASHBOARD_IP}:${CONTAINER_PORT}"
if [[ -n "$DNS_FQDN" ]]; then
echo "  DNS       : http://${DNS_FQDN}:${CONTAINER_PORT}"
fi
echo ""
echo "  View live logs:"
echo "    az container logs --name $ACI_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "  Check status:"
echo "    az container show --name $ACI_NAME --resource-group $RESOURCE_GROUP --query instanceView.state"
echo ""
echo "  Check for models:"
echo "    az storage blob list --account-name $STORAGE_ACCOUNT --container-name $CONTAINER_NAME --prefix models/ --output table"
echo "=========================================="
