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

MODEL_WAIT_TIMEOUT="${MODEL_WAIT_TIMEOUT:-1200}"
MODEL_POLL_INTERVAL="${MODEL_POLL_INTERVAL:-30}"

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

# ── Helper: upsert a key=value line into a .env file ──────────────────────────

_upsert_env() {
  local key="$1" val="$2" file="$3"
  local tmp
  tmp=$(mktemp)
  grep -v "^${key}=" "$file" > "$tmp" 2>/dev/null || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" "$file"
}

# ── Cleanup: old blob logs ────────────────────────────────────────────────────

cleanup_logs() {
  echo "Deleting blob logs older than 7 days..."

  local cutoff_date
  cutoff_date=$(date -d "7 days ago" +%Y-%m-%d 2>/dev/null \
            || date -v-7d +%Y-%m-%d)

  az storage blob list \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER_NAME" \
    --prefix "logs/" \
    --query "[].name" \
    --output tsv 2>/dev/null | while IFS= read -r blob; do
      local blob_date
      blob_date=$(echo "$blob" | cut -d'/' -f2)
      if [[ -n "$blob_date" && "$blob_date" < "$cutoff_date" ]]; then
        echo "  Deleting: $blob"
        az storage blob delete \
          --account-name "$STORAGE_ACCOUNT" \
          --container-name "$CONTAINER_NAME" \
          --name "$blob" \
          --output none 2>/dev/null || true
      fi
  done

  echo "  Log cleanup complete."
}

# ── Cleanup: old ACR images ───────────────────────────────────────────────────

cleanup_acr() {
  echo "Cleaning up old ACR images (keeping latest 2 tags + 'latest')..."

  local all_tags
  all_tags=$(az acr repository show-tags \
    --name "$ACR_NAME" \
    --repository trader-ai \
    --orderby time_desc \
    --query "[?name != 'latest'].name" \
    --output tsv 2>/dev/null || true)

  if [[ -z "$all_tags" ]]; then
    echo "  No non-latest tags found — nothing to delete."
    return
  fi

  local tag_count
  tag_count=$(echo "$all_tags" | wc -l | tr -d ' ')

  if (( tag_count > 2 )); then
    echo "$all_tags" | tail -n +3 | while IFS= read -r tag; do
      [[ -z "$tag" ]] && continue
      echo "  Deleting: trader-ai:$tag"
      az acr repository delete \
        --name "$ACR_NAME" \
        --image "trader-ai:$tag" \
        --yes \
        --output none 2>/dev/null || true
    done
  fi

  echo "  Purging untagged manifests older than 7 days..."
  az acr run --registry "$ACR_NAME" \
    --cmd 'acr purge --filter "trader-ai:.*" --untagged --ago 7d' \
    /dev/null 2>/dev/null || true

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
# cleanup_logs

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

_CREATE_CMD=(
  az container create
    --name "$ACI_NAME"
    --resource-group "$RESOURCE_GROUP"
    --image "$IMAGE_TAG"
    --registry-login-server "$ACR_LOGIN_SERVER"
    --registry-username "$ACR_USERNAME"
    --registry-password "$ACR_PASSWORD"
    --cpu "$ACI_CPU"
    --memory "$ACI_MEMORY"
    --location "$LOCATION"
    --os-type Linux
    --restart-policy Always
    --ports "$CONTAINER_PORT"
    --ip-address Public
    --dns-name-label "$DNS_LABEL"
    --output table
)

if (( ${#_ENV_KV_PAIRS[@]} > 0 )); then
  _CREATE_CMD+=(--environment-variables "${_ENV_KV_PAIRS[@]}")
fi

if (( ${#_SECURE_KV_PAIRS[@]} > 0 )); then
  _CREATE_CMD+=(--secure-environment-variables "${_SECURE_KV_PAIRS[@]}")
fi

"${_CREATE_CMD[@]}"

# ── Step 6: Wait for models to train and save ─────────────────────────────────

echo ""
echo "Waiting up to ${MODEL_WAIT_TIMEOUT}s for models to appear in blob storage..."

elapsed=0
model_count=0

while (( elapsed < MODEL_WAIT_TIMEOUT )); do
  sleep "$MODEL_POLL_INTERVAL"
  elapsed=$(( elapsed + MODEL_POLL_INTERVAL ))

  model_count=$(az storage blob list \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER_NAME" \
    --prefix "models/" \
    --query "length([])" \
    --output tsv 2>/dev/null || echo "0")

  if (( model_count > 0 )); then
    echo "  ✅ $model_count model file(s) saved to blob storage (${elapsed}s elapsed)."
    break
  fi

  echo "  ... ${elapsed}s elapsed, no models yet (checking every ${MODEL_POLL_INTERVAL}s)"
done

if (( model_count == 0 )); then
  echo ""
  echo "❌ FAILED: No models found after ${MODEL_WAIT_TIMEOUT}s."
  echo ""
  echo "  Check container logs:"
  echo "    az container logs --name $ACI_NAME --resource-group $RESOURCE_GROUP --follow"
  echo ""
  echo "  Check container status:"
  echo "    az container show --name $ACI_NAME --resource-group $RESOURCE_GROUP --query instanceView.state"
  echo ""

  echo "  Last 30 lines of container logs:"
  az container logs \
    --name "$ACI_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --tail 30 2>/dev/null || echo "  (could not retrieve logs)"

  exit 1
fi

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
echo "=========================================="
