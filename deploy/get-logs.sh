#!/usr/bin/env bash
# =============================================================================
# get-logs.sh — Fetch and dump Trader AI logs for analysis
#
# Source: ACI container stdout (JSON structured logs via Docker json-file driver)
# Logs are NOT stored in blob storage — they come directly from the container.
#
# Usage:
#   bash deploy/get-logs.sh                 # tail 500 lines, save to file
#   bash deploy/get-logs.sh --lines 2000    # tail 2000 lines
#   bash deploy/get-logs.sh --level ERROR   # filter by log level
#   bash deploy/get-logs.sh --event signal  # filter by event keyword
#   bash deploy/get-logs.sh --live          # stream live (follow mode, no save)
#   bash deploy/get-logs.sh --since 1h      # last 1 hour (requires --timestamps)
#
# Output: deploy/logs/trader-ai-<timestamp>.log (pretty-printed JSON)
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
LEVEL=""
EVENT_FILTER=""
LIVE_MODE=false
SINCE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lines)   LINES="$2";        shift 2 ;;
        --level)   LEVEL="${2^^}";    shift 2 ;;   # uppercase: INFO, ERROR, etc.
        --event)   EVENT_FILTER="$2"; shift 2 ;;
        --live)    LIVE_MODE=true;    shift   ;;
        --since)   SINCE="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

HAS_JQ=false
if command -v jq &>/dev/null; then
    HAS_JQ=true
fi

_filter_logs() {
    # stdin: raw log output (one JSON object per line, or plain text)
    # Apply level and event filters if jq is available.
    if $HAS_JQ; then
        local jq_filter='.'
        if [[ -n "$LEVEL" ]]; then
            jq_filter+=" | select(.level == \"$LEVEL\")"
        fi
        if [[ -n "$EVENT_FILTER" ]]; then
            # Match event field OR message field (case-insensitive substring)
            jq_filter+=" | select(
                (.event // \"\" | ascii_downcase | contains(\"${EVENT_FILTER,,}\")) or
                (.message // \"\" | ascii_downcase | contains(\"${EVENT_FILTER,,}\"))
            )"
        fi
        # Pretty-print each JSON line; fall back to raw if it's not JSON
        while IFS= read -r line; do
            echo "$line" | jq -r "$jq_filter // empty" 2>/dev/null || echo "$line"
        done
    else
        # No jq — apply grep-based filters
        local raw
        raw=$(cat)
        if [[ -n "$LEVEL" ]]; then
            raw=$(echo "$raw" | grep -i "\"level\":\"${LEVEL}\"" || true)
        fi
        if [[ -n "$EVENT_FILTER" ]]; then
            raw=$(echo "$raw" | grep -i "$EVENT_FILTER" || true)
        fi
        echo "$raw"
    fi
}

_describe_filters() {
    local desc=""
    [[ -n "$LEVEL" ]]        && desc+=" level=$LEVEL"
    [[ -n "$EVENT_FILTER" ]] && desc+=" event~=$EVENT_FILTER"
    echo "${desc:-(no filters)}"
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "  Trader AI — Log Fetch"
echo "  Container : $ACI_NAME"
echo "  Group     : $RESOURCE_GROUP"
echo "  Filters   : $(_describe_filters)"
echo "=========================================="

if $LIVE_MODE; then
    # ── Live streaming — no file save ─────────────────────────────────────────
    echo "  Mode      : LIVE (Ctrl+C to stop)"
    echo "=========================================="
    echo ""
    az container logs \
        --name "$ACI_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --follow \
        2>/dev/null | _filter_logs
    exit 0
fi

# ── Snapshot dump ─────────────────────────────────────────────────────────────

echo "  Mode      : SNAPSHOT (last $LINES lines)"
echo "=========================================="

mkdir -p "$SCRIPT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_FILE="$SCRIPT_DIR/logs/trader-ai-${TIMESTAMP}.log"

echo ""
echo "Fetching logs from ACI..."

# Fetch raw logs. ACI doesn't support --tail natively so we pipe through tail.
RAW=$(az container logs \
    --name "$ACI_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    2>/dev/null | tail -n "$LINES")

if [[ -z "$RAW" ]]; then
    echo "WARNING: No logs returned. Container may not be running."
    echo "  Check: az container show --name $ACI_NAME --resource-group $RESOURCE_GROUP --query instanceView.state"
    exit 1
fi

# Apply filters and write to file
echo "$RAW" | _filter_logs > "$OUT_FILE"

LINE_COUNT=$(wc -l < "$OUT_FILE" | tr -d ' ')
echo ""
echo "  Saved  : $OUT_FILE"
echo "  Lines  : $LINE_COUNT"

# ── Summary stats (requires jq) ───────────────────────────────────────────────

if $HAS_JQ; then
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
        "$OUT_FILE" 2>/dev/null | tail -20

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
echo "    Stream live:  bash deploy/get-logs.sh --live"
echo "    Errors only:  bash deploy/get-logs.sh --level ERROR"
echo "    Signals only: bash deploy/get-logs.sh --event signal"
echo "    More lines:   bash deploy/get-logs.sh --lines 5000"
echo "=========================================="
