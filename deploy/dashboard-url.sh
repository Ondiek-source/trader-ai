#!/usr/bin/env bash
# dashboard-url.sh — Print the current Trader AI dashboard URL.
# Usage: bash deploy/dashboard-url.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"

# Load azure.env (skip comments and blank lines)
if [[ -f "$AZURE_ENV" ]]; then
  while IFS='=' read -r key value; do
    # Skip comments, blank lines, and malformed lines
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
    export "${key}=${value}"
  done < "$AZURE_ENV"
fi

IP=$(az container show \
  --name "$CONTAINER_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query ipAddress.ip \
  --output tsv 2>/dev/null || true)

if [[ -z "$IP" ]]; then
  echo "ERROR: Could not retrieve IP. Is the container running?"
  exit 1
fi

echo "http://${IP}:8080"
echo "http://${DNS_FQDN}:${CONTAINER_PORT}"
