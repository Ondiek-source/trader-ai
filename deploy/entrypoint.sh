#!/bin/bash
set -e

echo "Syncing data from Azure Blob..."

# Create directories
mkdir -p /app/data/processed /app/data/raw /app/models

# Download ALL processed bar files (supports multiple pairs)
echo "Downloading processed bar files..."
az storage blob list \
    --container-name "$CONTAINER_NAME" \
    --prefix "processed/" \
    --connection-string "$AZURE_STORAGE_CONN" \
    --query "[].name" -o tsv 2>/dev/null | while read -r blob; do
        local_file="/app/data/${blob}"
        mkdir -p "$(dirname "$local_file")"
        az storage blob download \
            --container-name "$CONTAINER_NAME" \
            --name "$blob" \
            --file "$local_file" \
            --connection-string "$AZURE_STORAGE_CONN" \
            --overwrite 2>/dev/null && echo "  ✓ $blob"
    done || echo "  No processed files found"

# Download ALL model files
echo "Downloading models..."
az storage blob list \
    --container-name "$CONTAINER_NAME" \
    --prefix "models/" \
    --connection-string "$AZURE_STORAGE_CONN" \
    --query "[].name" -o tsv 2>/dev/null | while read -r blob; do
        local_file="/app/${blob}"
        mkdir -p "$(dirname "$local_file")"
        az storage blob download \
            --container-name "$CONTAINER_NAME" \
            --name "$blob" \
            --file "$local_file" \
            --connection-string "$AZURE_STORAGE_CONN" \
            --overwrite 2>/dev/null && echo "  ✓ $(basename "$blob")"
    done || echo "  No model files found"

echo "Sync complete"
exec python -m src.main