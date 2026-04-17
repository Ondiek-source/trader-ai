#!/usr/bin/env bash
trap 'rm -f /tmp/log_rows_*.txt' EXIT
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
HOURS=2
LEVEL=""
LIVE_MODE=false
SINCE=""

# Log storage path
LOG_DIR="C:/Users/Martin.Owaga.IMPAXAFRICA.000/OneDrive - IMPAX BUSINESS SOLUTIONS LTD/code/Projects/Trader AI/deploy/logs"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SNAPSHOT_FILE="$LOG_DIR/trader_ai_logs_${TIMESTAMP}.json"
ROLLING_FILE="$LOG_DIR/trader_ai_logs_$(date +"%Y%m%d").jsonl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lines) LINES="$2"; shift 2 ;;
        --hours) HOURS="$2"; shift 2 ;;
        --level) LEVEL="${2^^}"; shift 2 ;;
        --live) LIVE_MODE=true; shift ;;
        --since) SINCE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

WORKSPACE_ID=$(az container show -n trader-ai-engine -g "$RESOURCE_GROUP" --query "diagnostics.logAnalytics.workspaceId" -o tsv 2>/dev/null)
if [[ -z "$WORKSPACE_ID" ]]; then
    echo "ERROR: Cannot find Log Analytics workspace"
    exit 1
fi

# Build query
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

echo "$QUERY" > /tmp/query.kql

if $LIVE_MODE; then
    echo "LIVE mode not implemented"
    exit 0
fi

echo "Fetching logs..."
echo "  Querying last ${HOURS} hours of logs..."

# Run query - Azure returns a direct array, no wrapper
RESULT=$(az monitor log-analytics query \
  --workspace "$WORKSPACE_ID" \
  --analytics-query @/tmp/query.kql \
  --output json)
if [[ -z "$RESULT" ]] || [[ "$RESULT" == "null" ]]; then
    echo "⚠ No logs returned from query"
    exit 0
fi
if command -v jq &>/dev/null; then
    # Count rows - response is a direct array
    ROWS=$(echo "$RESULT" | jq 'length')
    
    if [[ "$ROWS" -eq 0 ]]; then
        echo "✓ No logs found in last ${HOURS}h"
        exit 0
    fi
    echo "  📅 Time range: Last ${HOURS} hours"
    if [[ -n "$SINCE" ]]; then
        echo "  📅 Since: $SINCE"
    fi
    # Create snapshot file (JSON array)
    echo '[' > "$SNAPSHOT_FILE"
    FIRST_ENTRY=true
    COUNT=0
    TEMP_LOG="/tmp/log_rows_$$.txt"
    # Write rows to temp file to avoid pipe issues
    echo "$RESULT" | jq -r '.[] | [.TimeGenerated, .Message] | @tsv' > "$TEMP_LOG" 2>/dev/null
    
    while IFS=$'\t' read -r ts msg; do
        if echo "$msg" | jq -e . >/dev/null 2>&1; then
            timestamp=$(echo "$msg" | jq -r '.timestamp // ""')
            level=$(echo "$msg" | jq -r '.level // "INFO"')
            component=$(echo "$msg" | jq -r '.component // "-"')
            message=$(echo "$msg" | jq -r '.message // ""')
            clean_message=$(echo "$message" | tr '\n' ' ' | sed 's/  */ /g')
            
            parsed_entry=$(jq -n \
                --arg ts "$timestamp" \
                --arg log_ts "${ts:0:19}" \
                --arg lvl "$level" \
                --arg comp "$component" \
                --arg msg "$clean_message" \
                --arg raw "$msg" \
                '{azure_timestamp: $log_ts, timestamp: $ts, level: $lvl, component: $comp, message: $msg, raw_message: $raw}')
            
            if [[ "$FIRST_ENTRY" == "true" ]]; then
                echo "$parsed_entry" >> "$SNAPSHOT_FILE"
                FIRST_ENTRY=false
            else
                echo ",$parsed_entry" >> "$SNAPSHOT_FILE"
            fi
            
            echo "$parsed_entry" >> "$ROLLING_FILE"
        else
            raw_entry=$(jq -n \
                --arg ts "${ts:0:19}" \
                --arg msg "$msg" \
                '{azure_timestamp: $ts, raw: $msg, parsed: false}')
            
            if [[ "$FIRST_ENTRY" == "true" ]]; then
                echo "$raw_entry" >> "$SNAPSHOT_FILE"
                FIRST_ENTRY=false
            else
                echo ",$raw_entry" >> "$SNAPSHOT_FILE"
            fi
            
            echo "$raw_entry" >> "$ROLLING_FILE"
        fi
        
        COUNT=$((COUNT + 1))
    done < <(head -n "$LINES" "$TEMP_LOG")
    
    rm -f "$TEMP_LOG"
    
    # Close snapshot JSON array
    echo ']' >> "$SNAPSHOT_FILE"
    if command -v jq &>/dev/null; then
        jq '.' "$SNAPSHOT_FILE" > "${SNAPSHOT_FILE}.tmp" && mv "${SNAPSHOT_FILE}.tmp" "$SNAPSHOT_FILE"
    fi
    echo ""
    echo "✓ Logs fetched successfully"
    echo "  📁 Location: $LOG_DIR"
    echo "  📋 Rolling file: $(basename "$ROLLING_FILE") (appended)"
    echo "  📸 Snapshot: $(basename "$SNAPSHOT_FILE")"
    echo "  📊 Total entries: $COUNT"
    echo "  📅 Time range: Last ${HOURS} hours"
    echo ""
    
else
    echo "ERROR: jq not found. Please install jq"
    exit 1
fi