#!/usr/bin/env bash
# dashboard-url.sh — Print the current Trader AI dashboard URL.
# Usage: bash deploy/dashboard-url.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"
APP_ENV="$PROJECT_ROOT/.env"

# Load environment variables
if [[ -f "$AZURE_ENV" ]]; then
  export $(cat "$AZURE_ENV" | xargs)
fi

if [[ -f "$APP_ENV" ]]; then
  export $(cat "$APP_ENV" | xargs)
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
