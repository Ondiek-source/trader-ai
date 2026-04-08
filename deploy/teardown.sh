#!/usr/bin/env bash
# =============================================================================
# teardown.sh — Delete all Azure resources for Trader AI
#
# WARNING: This deletes ALL resources including storage data.
# Use only when you want to fully remove the deployment.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"

if [[ ! -f "$AZURE_ENV" ]]; then
  echo "ERROR: deploy/azure.env not found."
  exit 1
fi

source "$AZURE_ENV"

echo ""
echo "WARNING: This will permanently delete resource group '$RESOURCE_GROUP'"
echo "and ALL resources inside it (storage, registry, containers)."
echo ""
read -r -p "Type 'yes' to confirm: " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

echo "Deleting resource group $RESOURCE_GROUP..."
az group delete \
  --name "$RESOURCE_GROUP" \
  --yes \
  --no-wait

echo "Deletion queued. Resources will be removed within a few minutes."
