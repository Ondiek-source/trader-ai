#!/usr/bin/env bash
# get-logs.sh — List and download persistent logs from blob storage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"

if [[ ! -f "$AZURE_ENV" ]]; then
    echo "ERROR: deploy/azure.env not found."
    exit 1
fi

source "$AZURE_ENV"

# Get date from argument or use today
DATE="${1:-$(date +%Y-%m-%d)}"

echo "=========================================="
echo "  Available log files for $DATE"
echo "=========================================="

# List available log files
az storage blob list \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER_NAME" \
    --prefix "logs/$DATE/" \
    --query "[].name" \
    --output table 2>/dev/null

echo ""
echo "=========================================="
echo "  Download latest log? (Ctrl+C to cancel)"
echo "=========================================="
read -p "Press Enter to download the most recent log..."

# Get the most recent log file
LATEST_LOG=$(az storage blob list \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER_NAME" \
    --prefix "logs/$DATE/" \
    --query "sort_by([], &lastModified)[-1].name" \
    --output tsv 2>/dev/null)

if [[ -n "$LATEST_LOG" ]]; then
    OUTPUT_FILE="$SCRIPT_DIR/logs/$(basename "$LATEST_LOG")"
    mkdir -p "$SCRIPT_DIR/logs"
    
    az storage blob download \
        --account-name "$STORAGE_ACCOUNT" \
        --container-name "$CONTAINER_NAME" \
        --name "$LATEST_LOG" \
        --file "$OUTPUT_FILE" \
        --output none 2>/dev/null
    
    echo "✓ Downloaded: $OUTPUT_FILE"
    echo "  Opening in notepad..."
    notepad "$OUTPUT_FILE"
else
    echo "No logs found for $DATE"
fi