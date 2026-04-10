#!/usr/bin/env bash
# get-logs.sh — Download persistent logs from blob storage and open in editor
# Usage: bash get-logs.sh [date] [editor]

# Get today's logs and open in notepad (Windows)
# bash get-logs.sh

# Get today's logs and open in VS Code
# bash get-logs.sh 2026-04-10 "code"

# Get specific date logs
# bash get-logs.sh 2026-04-09

# Open in notepad++
# bash get-logs.sh 2026-04-10 "notepad++"

#!/usr/bin/env bash
# get-logs.sh — Download persistent logs from blob storage and open in editor
# Usage: bash get-logs.sh [date] [editor]

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

# Get editor from argument or use default
EDITOR="${2:-notepad}"

BLOB_NAME="logs/$DATE/trader-ai-engine.log"
OUTPUT_FILE="$SCRIPT_DIR/logs/trader-ai-logs-$DATE.txt"

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

echo "=========================================="
echo "  Downloading logs from blob storage"
echo "  Date: $DATE"
echo "  Blob: $BLOB_NAME"
echo "=========================================="

# Download the log file
if az storage blob download \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER_NAME" \
    --name "$BLOB_NAME" \
    --file "$OUTPUT_FILE" \
    --output none 2>/dev/null; then
    
    echo "✓ Logs downloaded to: $OUTPUT_FILE"
    echo "  File size: $(wc -l < "$OUTPUT_FILE") lines"
    
    # Open in editor
    echo "  Opening in $EDITOR..."
    "$EDITOR" "$OUTPUT_FILE"
else
    echo "ERROR: No log file found for date $DATE"
    echo ""
    echo "Available log files:"
    az storage blob list \
        --account-name "$STORAGE_ACCOUNT" \
        --container-name "$CONTAINER_NAME" \
        --prefix "logs/" \
        --query "[].name" \
        --output table 2>/dev/null || echo "  No logs found"
    exit 1
fi