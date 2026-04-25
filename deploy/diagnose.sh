#!/usr/bin/env bash
# =============================================================================
# diagnose.sh — Interpreted diagnostic for Trader AI
# Usage: bash deploy/diagnose.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"
source "$AZURE_ENV"

_EVT_TICK="$_EVT_TICK"
_EVT_PRED="$_EVT_PRED"
_EVT_GATE_REJECTED="$_EVT_GATE_REJECTED"

LOGS=$(az container logs \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" 2>/dev/null || echo "")

# ── Also check dashboard /status for live state ───────────────────────────────
DASHBOARD_IP=$(az container show \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "ipAddress.ip" -o tsv 2>/dev/null || echo "")
STATUS_JSON=""
if [[ -n "$DASHBOARD_IP" ]]; then
  STATUS_JSON=$(curl -s "http://${DASHBOARD_IP}:8080/status" 2>/dev/null || echo "")
fi

echo ""
echo "=========================================="
echo "  Trader AI — Diagnostic Report"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "=========================================="

# ── Container ──────────────────────────────────────────────────────────────────
echo ""
echo "── Container ──────────────────────────────"
STATE=$(az container show --name "$ACI_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "instanceView.state" -o tsv 2>/dev/null || echo "Unknown")

if [[ "$STATE" == "Running" ]]; then
  echo "  OK: Container is running"
else
  echo "  PROBLEM: Container state is $STATE — run deploy/deploy.sh"
fi

# ── Data Source ────────────────────────────────────────────────────────────────
echo ""
echo "── Data Source ────────────────────────────"
if echo "$LOGS" | grep "$_EVT_TICK" > /dev/null 2>&1; then
  PROVIDER=$(echo "$LOGS" | grep "$_EVT_TICK" | tail -1 \
    | grep -o "'provider': '[^']*'" | head -1 | cut -d"'" -f4 || echo "")
  echo "  OK: Streaming live ticks"
  if [[ -n "$PROVIDER" ]]; then
    echo "  OK: Provider is $PROVIDER"
  fi
else
  echo "  PROBLEM: No ticks received yet"
  echo "  FIX: Check logs for connection errors"
fi

# ── Ticks ──────────────────────────────────────────────────────────────────────
echo ""
echo "── Ticks ──────────────────────────────────"
TICK=$(echo "$LOGS" | grep "$_EVT_TICK" | tail -1)
if [[ -n "$TICK" ]]; then
  TICKS=$(echo "$TICK" | grep -o "'ticks': [0-9]*" | head -1 | grep -o "[0-9]*")
  TICKPAIR=$(echo "$TICK" | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
  echo "  OK: $TICKS ticks received (latest on $TICKPAIR)"
else
  echo "  PROBLEM: No ticks yet"
fi

# ── Training ───────────────────────────────────────────────────────────────────
echo ""
echo "── Training ───────────────────────────────"
if echo "$LOGS" | grep "$_EVT_PRED" > /dev/null 2>&1; then
  CONF=$(echo "$LOGS" | grep "$_EVT_PRED" | tail -1 \
    | grep -o "'confidence': [0-9.]*" | head -1 | grep -o "[0-9.]*")
  MODEL=$(echo "$LOGS" | grep "$_EVT_PRED" | tail -1 \
    | grep -o "'model': '[^']*'" | head -1 | cut -d"'" -f4)
  echo "  OK: Models generating predictions (model=$MODEL, conf=$CONF)"
elif echo "$LOGS" | grep "parquet_written" > /dev/null 2>&1; then
  echo "  OK: Data pipeline active (parquet being written)"
else
  echo "  PROBLEM: No model activity found"
  echo "  FIX: Check logs for errors"
fi

# ── Signals ────────────────────────────────────────────────────────────────────
echo ""
echo "── Signals ────────────────────────────────"
FIRED=$(echo "$LOGS" | grep -c "signal_fired" || true)
REJECTED=$(echo "$LOGS" | grep -c "$_EVT_GATE_REJECTED" || true)

if [[ "$FIRED" -gt 0 ]]; then
  echo "  OK: $FIRED signal(s) fired ($REJECTED rejected)"
elif [[ "$REJECTED" -gt 0 ]]; then
  echo "  INFO: 0 fired, $REJECTED rejected — gate is blocking"
  echo "$LOGS" | grep "$_EVT_GATE_REJECTED" | tail -5 | while IFS= read -r line; do
    PAIR=$(echo "$line" | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
    CONF=$(echo "$line" | grep -o "'confidence': [0-9.]*" | head -1 | grep -o "[0-9.]*")
    echo "    $PAIR: confidence $CONF"
  done
  echo "  FIX: Lower CONFIDENCE_THRESHOLD or wait for model retrain"
else
  echo "  WAITING: No signal activity yet"
fi

# ── Quotex Account ─────────────────────────────────────────────────────────────
echo ""
echo "── Quotex Account ─────────────────────────"
if [[ -n "$STATUS_JSON" ]]; then
  QX_CONNECTED=$(echo "$STATUS_JSON" | grep -o '"connected": true' | head -1 || echo "")
  QX_BALANCE=$(echo "$STATUS_JSON" | grep -o '"balance": [0-9.]*' | head -1 | grep -o "[0-9.]*" || echo "")
  if [[ -n "$QX_CONNECTED" ]]; then
    echo "  OK: Connected — Balance: \$${QX_BALANCE:-0}"
  else
    echo "  WARN: Quotex not connected"
  fi
else
  echo "  WARN: Dashboard unreachable — cannot check Quotex status"
fi

# ── Errors ─────────────────────────────────────────────────────────────────────
ERROR_COUNT=$(echo "$LOGS" | grep -c '"level": "ERROR"' || true)
echo ""
echo "── Errors ─────────────────────────────────"
if [[ "$ERROR_COUNT" -eq 0 ]]; then
  echo "  OK: No errors"
else
  echo "  $ERROR_COUNT error(s) — showing most recent:" >&2
  echo "$LOGS" | grep '"level": "ERROR"' | tail -5 | while IFS= read -r line; do
    TS=$(echo "$line" | grep -o '"timestamp": "[^"]*"' | head -1 | cut -d'"' -f4 | cut -c1-19)
    echo "    $TS"
  done
fi

# ── Health ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Health ─────────────────────────────────"
HEALTH=$(echo "$LOGS" | grep "health" | grep "uptime_seconds" | tail -1)
if [[ -n "$HEALTH" ]]; then
  UPTIME=$(echo "$HEALTH" | grep -o "'uptime_seconds': [0-9]*" | head -1 | grep -o "[0-9]*")
  TICKS=$(echo "$HEALTH" | grep -o "'ticks_received': [0-9]*" | head -1 | grep -o "[0-9]*")
  MEM=$(echo "$HEALTH" | grep -o "'memory_usage_gb': [0-9.]*" | head -1 | grep -o "[0-9.]*")
  CPU=$(echo "$HEALTH" | grep -o "'cpu_percent': [0-9.]*" | head -1 | grep -o "[0-9.]*")
  MINS=$((UPTIME / 60))
  SECS=$((UPTIME % 60))
  echo "  Uptime: ${MINS}m ${SECS}s | Ticks: $TICKS | Memory: ${MEM}GB | CPU: ${CPU}%"
else
  echo "  No health events yet"
fi

echo ""
echo "=========================================="
echo "  az container logs --name $ACI_NAME"
echo "  --resource-group $RESOURCE_GROUP --follow"
echo "=========================================="
echo ""
