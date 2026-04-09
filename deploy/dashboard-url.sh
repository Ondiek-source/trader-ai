#!/usr/bin/env bash
# dashboard-url.sh — Print the current Trader AI dashboard URL.
# Usage: bash deploy/dashboard-url.sh

set -euo pipefail

CONTAINER_NAME="trader-ai-engine"
RESOURCE_GROUP="rg-trader-ai"

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
