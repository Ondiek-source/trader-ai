#!/usr/bin/env bash
# =============================================================================
# get-logs.sh — Fetch Trader AI logs from Log Analytics
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"

if [[ ! -f "$AZURE_ENV" ]]; then
    echo "ERROR: deploy/azure.env not found"
    exit 1
fi

source "$AZURE_ENV"

# ── Defaults ────────────────────────────────────────────────────────────────
LINES=500
HOURS=1
LEVEL=""
LIVE_MODE=false
SINCE=""
WORKSPACE_NAME="trader-ai-logs"

# Log storage path (convert Windows path to Unix-style for Git Bash/WSL)
LOG_DIR="C:/Users/Martin.Owaga.IMPAXAFRICA.000/OneDrive - IMPAX BUSINESS SOLUTIONS LTD/code/Projects/Trader AI/deploy/logs"
# Alternative: Use relative path from script
# LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate timestamp for filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JSON_LOG_FILE="$LOG_DIR/trader_ai_logs_${TIMESTAMP}.json"
JSONL_LOG_FILE="$LOG_DIR/trader_ai_logs_$(date +"%Y%m%d").jsonl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lines) LINES="$2"; shift 2 ;;
        --hours) HOURS="$2"; shift 2 ;;
        --level) LEVEL="${2^^}"; shift 2 ;;
        --live) LIVE_MODE=true; shift ;;
        --since) SINCE="$2"; shift 2 ;;
        --format) OUTPUT_FORMAT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Get workspace ID from container (do not hardcode)
WORKSPACE_ID=$(az container show -n trader-ai-engine -g "$RESOURCE_GROUP" --query "diagnostics.logAnalytics.workspaceId" -o tsv 2>/dev/null)
if [[ -z "$WORKSPACE_ID" ]]; then
    echo "ERROR: Cannot find Log Analytics workspace for container instance. Make sure the container is running and has diagnostics enabled."
    exit 1
fi

echo "=========================================="
echo " Trader AI — Logs"
echo " Workspace: $WORKSPACE_NAME"
echo " Hours: ${SINCE:-$HOURS} Lines: $LINES Level: ${LEVEL:-ALL}"
echo " Live mode: $LIVE_MODE"
echo " Workspace ID: $WORKSPACE_ID"
echo " Resource group: $RESOURCE_GROUP"
echo " JSON Log file: $JSON_LOG_FILE"
echo "=========================================="
echo ""

# Build query - exact format that works
QUERY="ContainerInstanceLog_CL | where ContainerGroup_s == 'trader-ai-engine'"
if [[ -n "$SINCE" ]]; then
    QUERY+=" | where TimeGenerated >= datetime($SINCE)"
else
    QUERY+=" | where TimeGenerated > ago(${HOURS}h)"
fi
if [[ -n "$LEVEL" ]]; then
    QUERY+=" | where Message contains '\"level\": \"$LEVEL\"'"
fi
QUERY+=" | project TimeGenerated, Message | order by TimeGenerated asc"

# Write to file to avoid Windows arg-length bug
echo "$QUERY" > /tmp/query.kql

if $LIVE_MODE; then
    echo "LIVE mode not implemented in this version"
    exit 0
fi

# Run query - get JSON for pretty printing
RESULT=$(az monitor log-analytics query \
  --workspace "$WORKSPACE_ID" \
  --analytics-query @/tmp/query.kql \
  --output json)

# Initialize JSON arrays for storage
PARSED_LOGS=()
RAW_LOGS=()

# Pretty print with jq and save to JSON files
if command -v jq &>/dev/null; then
    ROWS=$(echo "$RESULT" | jq '.tables[0].rows | length')
    if [[ "$ROWS" -eq 0 ]]; then
        echo "No logs found in last ${HOURS}h"
        exit 0
    fi

    # Create JSON array file
    echo '[' > "$JSON_LOG_FILE"
    FIRST_ENTRY=true
    
    # Counter for processed logs
    COUNT=0
    
    echo "$RESULT" | jq -r '.tables[0].rows[] | @tsv' | head -n $LINES | while IFS=$'\t' read -r ts msg; do
        # Try to parse Message as JSON
        if echo "$msg" | jq -e . >/dev/null 2>&1; then
            # Extract fields from JSON message
            timestamp=$(echo "$msg" | jq -r '.timestamp // ""')
            level=$(echo "$msg" | jq -r '.level // "INFO"')
            component=$(echo "$msg" | jq -r '.component // "-"')
            message=$(echo "$msg" | jq -r '.message // ""')
            
            # Clean up message (remove extra whitespace and newlines for display)
            clean_message=$(echo "$message" | tr '\n' ' ' | sed 's/  */ /g')
            
            # Create parsed JSON object with metadata
            parsed_entry=$(jq -n \
                --arg ts "$timestamp" \
                --arg log_ts "${ts:0:19}" \
                --arg lvl "$level" \
                --arg comp "$component" \
                --arg msg "$clean_message" \
                --arg raw "$msg" \
                '{azure_timestamp: $log_ts, timestamp: $ts, level: $lvl, component: $comp, message: $msg, raw_message: $raw}')
            
            # Output to console
            printf "%s | %-8s | %-12s | %s\n" "${ts:0:19}" "$level" "$component" "$clean_message"
            
            # Append to JSON file (as array elements)
            if [[ "$FIRST_ENTRY" == "true" ]]; then
                echo "$parsed_entry" >> "$JSON_LOG_FILE"
                FIRST_ENTRY=false
            else
                echo ",$parsed_entry" >> "$JSON_LOG_FILE"
            fi
            
            # Also append to JSONL file (one JSON object per line)
            echo "$parsed_entry" >> "$JSONL_LOG_FILE"
            
        else
            # Non-JSON log entry (raw)
            printf "%s | RAW | %s\n" "${ts:0:19}" "$msg"
            
            # Create raw entry JSON
            raw_entry=$(jq -n \
                --arg ts "${ts:0:19}" \
                --arg msg "$msg" \
                '{azure_timestamp: $ts, raw: $msg, parsed: false}')
            
            if [[ "$FIRST_ENTRY" == "true" ]]; then
                echo "$raw_entry" >> "$JSON_LOG_FILE"
                FIRST_ENTRY=false
            else
                echo ",$raw_entry" >> "$JSON_LOG_FILE"
            fi
            
            echo "$raw_entry" >> "$JSONL_LOG_FILE"
        fi
        
        COUNT=$((COUNT + 1))
    done
    
    # Close JSON array
    echo ']' >> "$JSON_LOG_FILE"
    
    echo ""
    echo "=========================================="
    echo " Logs saved to:"
    echo "   JSON array: $JSON_LOG_FILE"
    echo "   JSONL format: $JSONL_LOG_FILE"
    echo " Total entries: $COUNT"
    echo "=========================================="
    
else
    # Fallback to table output if no jq
    echo "WARNING: jq not found. Saving raw output only."
    az monitor log-analytics query \
      --workspace "$WORKSPACE_ID" \
      --analytics-query @/tmp/query.kql \
      --output table | tee -a "$LOG_DIR/raw_logs_${TIMESTAMP}.txt"
fi