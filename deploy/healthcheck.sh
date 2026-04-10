#!/usr/bin/env bash
# =============================================================================
# healthcheck.sh — Pre-flight checklist for Trader AI
#
# Checks that every component is ready before live trading begins.
# Run this after deploy.sh and after the container has been up for 30+ minutes.
#
# Usage:
#   bash deploy/healthcheck.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"

if [[ ! -f "$AZURE_ENV" ]]; then
  echo "ERROR: deploy/azure.env not found. Run provision.sh first."
  exit 1
fi

source "$AZURE_ENV"

PASS="[PASS]"
FAIL="[FAIL]"
WARN="[WARN]"

ALL_PASSED=true

echo ""
echo "=========================================="
echo "  Trader AI — Pre-Flight Checklist"
echo "=========================================="
echo ""

# ── Helper ─────────────────────────────────────────────────────────────────────
check() {
  local label="$1"
  local result="$2"
  local detail="${3:-}"
  if [[ "$result" == "pass" ]]; then
    echo "  $PASS  $label"
    [[ -n "$detail" ]] && echo "         $detail"
  elif [[ "$result" == "warn" ]]; then
    echo "  $WARN  $label"
    [[ -n "$detail" ]] && echo "         $detail"
  else
    echo "  $FAIL  $label"
    [[ -n "$detail" ]] && echo "         $detail"
    ALL_PASSED=false
  fi
}

# ── 1. Container running ───────────────────────────────────────────────────────
echo "[ 1/8 ] Container status..."
CONTAINER_STATE=$(az container show \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "instanceView.state" \
  --output tsv 2>/dev/null || echo "NotFound")

if [[ "$CONTAINER_STATE" == "Running" ]]; then
  check "Container is running" "pass" "State: $CONTAINER_STATE"
else
  check "Container is running" "fail" "State: $CONTAINER_STATE — run bash deploy/deploy.sh"
fi

# ── 2. Fetch logs ──────────────────────────────────────────────────────────────
echo "[ 2/8 ] Fetching container logs..."
LOGS=$(az container logs \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" 2>/dev/null || echo "")

if [[ -z "$LOGS" ]]; then
  check "Container logs accessible" "fail" "No logs returned — container may still be starting"
else
  check "Container logs accessible" "pass"
fi

# ── 3. Historical data downloaded ─────────────────────────────────────────────
echo "[ 3/8 ] Historical data download..."
if echo "$LOGS" | grep -q "backfill_complete\|backfill_skipped" 2>/dev/null; then
  check "Historical data downloaded (5 years)" "pass"
elif echo "$LOGS" | grep -q "backfill_starting"; then
  check "Historical data downloading" "warn" "Still in progress — wait 20-30 minutes and re-run this script"
else
  check "Historical data downloaded" "fail" "No backfill event found in logs"
fi

# ── 4. Models trained ─────────────────────────────────────────────────────────
echo "[ 4/8 ] Model training..."
TRAINED_COUNT=$(echo "$LOGS" | grep -c "_trained" 2>/dev/null || echo "0")
if [ "${TRAINED_COUNT:-0}" -ge 3 ] 2>/dev/null; then
  check "Models trained ($TRAINED_COUNT models)" "pass"
elif [[ "$TRAINED_COUNT" -gt 0 ]]; then
  check "Models partially trained ($TRAINED_COUNT models)" "warn" "Training still in progress"
else
  check "Models trained" "fail" "No training events found — check logs for errors"
fi

# ── 5. Live price stream active ────────────────────────────────────────────────
echo "[ 5/8 ] Live price stream..."
if echo "$LOGS" | grep -q "stream_connected\|ticks_flushed\|tick_queue"; then
  LAST_TICK=$(echo "$LOGS" | grep "ticks_flushed" | tail -1)
  check "Live price stream active (Twelve Data)" "pass" "$LAST_TICK"
elif echo "$LOGS" | grep -q "stream_stub_mode"; then
  check "Live price stream" "warn" "Running in stub/synthetic mode — check TWELVEDATA_API_KEY"
else
  check "Live price stream active" "fail" "No stream events found — check TWELVEDATA_API_KEY in .env"
fi

# ── 6. Signals being evaluated ────────────────────────────────────────────────
echo "[ 6/8 ] Signal evaluation..."
if echo "$LOGS" | grep -q "signal_fired\|signal_suppressed\|below_threshold"; then
  FIRED=$(echo "$LOGS" | grep -c "signal_fired" 2>/dev/null || echo "0")
  SUPPRESSED=$(echo "$LOGS" | grep -c "signal_suppressed\|below_threshold" 2>/dev/null || echo "0")
  check "Signals being evaluated" "pass" "Fired: $FIRED | Suppressed (below threshold): $SUPPRESSED"
else
  check "Signals being evaluated" "warn" "No signal events yet — models may still be training"
fi

# ── 7. Quotex connection ───────────────────────────────────────────────────────
echo "[ 7/8 ] Quotex account connection..."
if echo "$LOGS" | grep -q "quotex_connected"; then
  BALANCE=$(echo "$LOGS" | grep "quotex_connected" | tail -1 | grep -o '"balance":[0-9.]*' | head -1)
  MODE=$(echo "$LOGS" | grep "quotex_connected" | tail -1 | grep -o '"mode":"[^"]*"' | head -1)
  check "Quotex account connected" "pass" "$MODE | $BALANCE"
elif echo "$LOGS" | grep -q "quotex_connect_failed"; then
  REASON=$(echo "$LOGS" | grep "quotex_connect_failed" | tail -1 | grep -o '"reason":"[^"]*"' | head -1)
  check "Quotex account connected" "fail" "Connection failed: $REASON — check QUOTEX_EMAIL and QUOTEX_PASSWORD in .env"
elif echo "$LOGS" | grep -q "quotex_no_credentials"; then
  check "Quotex account connected" "fail" "No credentials — set QUOTEX_EMAIL and QUOTEX_PASSWORD in .env"
else
  check "Quotex account connected" "warn" "No Quotex events yet — may still be initialising"
fi

# ── 8. Webhook reachable ───────────────────────────────────────────────────────
echo "[ 8/8 ] Webhook endpoint..."
if echo "$LOGS" | grep -q "signal_sent"; then
  LAST=$(echo "$LOGS" | grep "signal_sent" | tail -1 | grep -o '"http_status":[0-9]*' | head -1)
  check "Webhook endpoint reachable" "pass" "Last signal HTTP status: $LAST"
elif echo "$LOGS" | grep -q "webhook_failed"; then
  REASON=$(echo "$LOGS" | grep "webhook_failed" | tail -1)
  check "Webhook endpoint reachable" "fail" "Webhook failed — check WEBHOOK_URL in .env"
else
  check "Webhook endpoint reachable" "warn" "No signals fired yet — system may still be training"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
if [[ "$ALL_PASSED" == "true" ]]; then
  echo "  ALL CHECKS PASSED — System is ready."
else
  echo "  SOME CHECKS FAILED — Review items above."
  echo "  View full logs:"
  echo "  az container logs --name $ACI_NAME --resource-group $RESOURCE_GROUP --follow"
fi
echo "=========================================="
echo ""
