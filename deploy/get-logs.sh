#!/usr/bin/env bash
# =============================================================================
# get-logs.sh — Fetch Trader AI logs from Log Analytics workspace
#
# Source: Log Analytics workspace (persisted container logs)
#
# Usage:
#   bash deploy/get-logs.sh                      # last 1 hour, 500 lines
#   bash deploy/get-logs.sh --hours 24           # last 24 hours
#   bash deploy/get-logs.sh --lines 2000         # last 2000 lines
#   bash deploy/get-logs.sh --hours 24 --lines 1000
#   bash deploy/get-logs.sh --level ERROR        # filter by log level
#   bash deploy/get-logs.sh --event signal       # filter by event keyword
#   bash deploy/get-logs.sh --live               # continuously poll (every 5s)
#   bash deploy/get-logs.sh --since "2026-04-16T09:00:00Z"  # specific time
#   bash deploy/get-logs.sh --limit 1000         # max rows to return
#
#
# Requirements: Azure CLI (az), jq (optional but recommended for filtering)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"

if [[ ! -f "$AZURE_ENV" ]]; then
    echo "ERROR: deploy/azure.env not found. Run provision.sh first."
    exit 1
fi

source "$AZURE_ENV"

# ── Argument parsing ──────────────────────────────────────────────────────────

LINES=500
HOURS=1
LEVEL=""
EVENT_FILTER=""
LIVE_MODE=false
SINCE=""
LIMIT=""
WORKSPACE_NAME="trader-ai-logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lines)   LINES="$2";        shift 2 ;;
        --hours)   HOURS="$2";        shift 2 ;;
        --level)   LEVEL="${2^^}";    shift 2 ;;
        --event)   EVENT_FILTER="$2"; shift 2 ;;
        --live)    LIVE_MODE=true;    shift   ;;
        --since)   SINCE="$2";        shift 2 ;;
        --limit)   LIMIT="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

HAS_JQ=false
if command -v jq &>/dev/null; then
    HAS_JQ=true
fi

_build_query() {
    local query="ContainerInstanceLog_CL"
    
    # Add filters
    local filters=()
    
    # Time filter
    if [[ -n "$SINCE" ]]; then
        filters+=("TimeGenerated >= datetime($SINCE)")
    else
        filters+=("TimeGenerated > ago(${HOURS}h)")
    fi
    
    # Level filter (log level appears in LogEntry JSON)
    if [[ -n "$LEVEL" ]]; then
        filters+=("LogEntry contains '\"level\": \"$LEVEL\"'")
    fi
    
    # Event filter
    if [[ -n "$EVENT_FILTER" ]]; then
        filters+=("(LogEntry contains '\"event\":' and LogEntry contains '$EVENT_FILTER')")
    fi
    
    # Combine filters
    if [[ ${#filters[@]} -gt 0 ]]; then
        query+=" | where $(IFS=' and '; echo "${filters[*]}")"
    fi
    
    # Order and limit
    query+=" | order by TimeGenerated desc"
    
    if [[ -n "$LIMIT" ]]; then
        query+=" | take $LIMIT"
    elif [[ $LINES -gt 0 ]]; then
        query+=" | take $LINES"
    fi
    
    echo "$query"
}

_filter_json_logs() {
    # Extract and pretty-print JSON from LogEntry field
    if $HAS_JQ; then
        jq -r '.LogEntry' 2>/dev/null | while IFS= read -r line; do
            # Pretty-print each JSON line
            echo "$line" | jq '.' 2>/dev/null || echo "$line"
        done
    else
        # No jq - just extract LogEntry
        jq -r '.LogEntry' 2>/dev/null || cat
    fi
}

_describe_filters() {
    local desc=""
    [[ -n "$SINCE" ]] && desc+=" since=$SINCE"
    [[ -z "$SINCE" ]] && desc+=" last=${HOURS}h"
    [[ -n "$LEVEL" ]] && desc+=" level=$LEVEL"
    [[ -n "$EVENT_FILTER" ]] && desc+=" event~=$EVENT_FILTER"
    [[ -n "$LIMIT" ]] && desc+=" limit=$LIMIT"
    [[ -z "$LIMIT" && $LINES -gt 0 ]] && desc+=" lines=$LINES"
    echo "${desc:-(no filters)}"
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "  Trader AI — Log Analytics Query"
echo "  Workspace : $WORKSPACE_NAME"
echo "  Resource  : $RESOURCE_GROUP"
echo "  Filters   : $(_describe_filters)"
echo "=========================================="

# Verify workspace exists
if ! az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" &>/dev/null; then
    echo "ERROR: Log Analytics workspace '$WORKSPACE_NAME' not found."
    echo "  Run deploy.sh first to create the workspace."
    exit 1
fi

# Get workspace ID
WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query customerId -o tsv)

if [[ -z "$WORKSPACE_ID" ]]; then
    echo "ERROR: Could not get workspace ID."
    exit 1
fi

if $LIVE_MODE; then
    # ── Live mode — poll every 5 seconds ─────────────────────────────────────
    echo "  Mode      : LIVE (polling every 5s, Ctrl+C to stop)"
    echo "=========================================="
    echo ""
    
    LAST_TIMESTAMP=""
    while true; do
        QUERY=$(cat <<EOF
ContainerInstanceLog_CL
| where TimeGenerated > ago(5m)
| order by TimeGenerated desc
| take 100
EOF
)
        RESULT=$(az monitor log-analytics query \
            --workspace "$WORKSPACE_ID" \
            --analytics-query "$QUERY" \
            --output json 2>/dev/null)
        
        if [[ -n "$RESULT" ]] && echo "$RESULT" | jq -e '.tables[0].rows' >/dev/null 2>&1; then
            # Process new logs
            echo "$RESULT" | jq -r '.tables[0].rows[] | @json' 2>/dev/null | while read -r row; do
                TIMESTAMP=$(echo "$row" | jq -r '.[0]' 2>/dev/null)
                LOG_ENTRY=$(echo "$row" | jq -r '.[1]' 2>/dev/null)
                
                if [[ "$TIMESTAMP" != "$LAST_TIMESTAMP" ]]; then
                    echo "[$TIMESTAMP] $LOG_ENTRY" | jq '.' 2>/dev/null || echo "[$TIMESTAMP] $LOG_ENTRY"
                    LAST_TIMESTAMP="$TIMESTAMP"
                fi
            done
        fi
        
        sleep 5
    done
    exit 0
fi

# ── Snapshot query ────────────────────────────────────────────────────────────

echo "  Mode      : SNAPSHOT"
echo "=========================================="

mkdir -p "$SCRIPT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_FILE="$SCRIPT_DIR/logs/trader-ai-${TIMESTAMP}.log"

echo ""
echo "Querying Log Analytics..."

QUERY=$(_build_query)

# Execute query
RESULT=$(az monitor log-analytics query \
    --workspace "$WORKSPACE_ID" \
    --analytics-query "$QUERY" \
    --output json 2>/dev/null)

if [[ -z "$RESULT" ]] || ! echo "$RESULT" | jq -e '.tables[0].rows' >/dev/null 2>&1; then
    echo "WARNING: No logs found or query failed."
    echo "  Container may not have produced logs yet."
    echo "  Wait a few minutes and try again."
    exit 1
fi

# Extract and save logs
echo "$RESULT" | jq -r '.tables[0].rows[] | .[1]' 2>/dev/null | _filter_json_logs > "$OUT_FILE"

LINE_COUNT=$(wc -l < "$OUT_FILE" | tr -d ' ')
echo ""
echo "  Saved  : $OUT_FILE"
echo "  Lines  : $LINE_COUNT"

# ── Summary stats (requires jq) ───────────────────────────────────────────────

if $HAS_JQ && [[ -s "$OUT_FILE" ]]; then
    echo ""
    echo "── Log Level Breakdown ──"
    jq -r '.level // "UNKNOWN"' "$OUT_FILE" 2>/dev/null \
        | sort | uniq -c | sort -rn \
        | while read -r count level; do
              printf "  %-10s %s\n" "$level" "$count"
          done

    echo ""
    echo "── Recent Events ────────"
    jq -r 'select(.event != null) | "\(.timestamp[11:19] // "?") [\(.level)] \(.event)"' \
        "$OUT_FILE" 2>/dev/null | head -20

    ERROR_COUNT=$(jq -r 'select(.level == "ERROR" or .level == "CRITICAL")' \
        "$OUT_FILE" 2>/dev/null | wc -l | tr -d ' ')
    if (( ERROR_COUNT > 0 )); then
        echo ""
        echo "── Errors / Criticals ($ERROR_COUNT) ────────────────────────────"
        jq -r 'select(.level == "ERROR" or .level == "CRITICAL") |
            "\(.timestamp[11:19] // "?") [\(.level)] \(.message // .error // "no message")"' \
            "$OUT_FILE" 2>/dev/null | head -30
    fi
fi

echo ""
echo "=========================================="
echo "  Done. Full log: $OUT_FILE"
echo ""
echo "  Useful follow-up commands:"
echo "    Stream live:    bash deploy/get-logs.sh --live"
echo "    Errors only:    bash deploy/get-logs.sh --level ERROR"
echo "    Last 24 hours:  bash deploy/get-logs.sh --hours 24"
echo "    Specific time:  bash deploy/get-logs.sh --since \"2026-04-16T09:00:00Z\""
echo "=========================================="