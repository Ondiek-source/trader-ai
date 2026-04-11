#!/usr/bin/env bash
# =============================================================================
# healthcheck.sh — Pre-flight checklist for Trader AI
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

# ── Fetch logs (ALL, not just tail) ────────────────────────────────────────────
LOGS=$(az container logs \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" 2>/dev/null || echo "")

# ── Also fetch dashboard status ───────────────────────────────────────────────
DASHBOARD_IP=$(az container show \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "ipAddress.ip" -o tsv 2>/dev/null || echo "")
STATUS_JSON=""
if [[ -n "$DASHBOARD_IP" ]]; then
  STATUS_JSON=$(curl -s "http://${DASHBOARD_IP}:8080/status" 2>/dev/null || echo "")
fi

# ── 1. Container running ───────────────────────────────────────────────────────
echo "[ 1/11] Container status..."
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

# ── 2. Container logs ──────────────────────────────────────────────────────────
echo "[ 2/11] Container logs..."
if [[ -z "$LOGS" ]]; then
  check "Container logs accessible" "fail" "No logs returned"
else
  check "Container logs accessible" "pass"
fi

# ── 3. Data source (use dashboard /status) ─────────────────────────────────────
echo "[ 3/11] Data source..."
if [[ -n "$STATUS_JSON" ]]; then
  STREAM_CONNECTED=$(echo "$STATUS_JSON" | grep -o '"connected": true' | head -1 || echo "")
  if [[ -n "$STREAM_CONNECTED" ]]; then
    TICKS=$(echo "$STATUS_JSON" | grep -o '"ticks_received": [0-9]*' | head -1 | grep -o "[0-9]*" || echo "0")
    check "Data source connected" "pass" "$TICKS ticks received"
  else
    check "Data source connected" "warn" "Stream not connected yet"
  fi
elif echo "$LOGS" | grep -q "tick_milestone"; then
  TICKS=$(echo "$LOGS" | grep "tick_milestone" | tail -1 \
    | grep -o "'ticks': [0-9]*" | head -1 | grep -o "[0-9]*")
  check "Data source connected" "pass" "$TICKS ticks received"
else
  check "Data source connected" "fail" "No ticks and dashboard unreachable"
fi

# ── 4. Historical data ─────────────────────────────────────────────────────────
echo "[ 4/11] Historical data download..."
if [[ -n "${STORAGE_ACCOUNT:-}" ]]; then
  ACCOUNT_KEY=$(az storage account keys list \
    --resource-group "$RESOURCE_GROUP" \
    --account-name "$STORAGE_ACCOUNT" \
    --query "[0].value" \
    --output tsv 2>/dev/null || echo "")

  if [[ -n "$ACCOUNT_KEY" ]]; then
    BLOB_COUNT=$(az storage blob list \
      --container-name "$CONTAINER_NAME" \
      --account-name "$STORAGE_ACCOUNT" \
      --account-key "$ACCOUNT_KEY" \
      --query "length([?contains(name, '.parquet')])" \
      --output tsv 2>/dev/null || echo "0")

    if [[ "$BLOB_COUNT" -ge 12 ]]; then
      check "Historical data downloaded" "pass" "$BLOB_COUNT parquet files found"
    elif [[ "$BLOB_COUNT" -gt 0 ]]; then
      check "Historical data downloading" "warn" "$BLOB_COUNT parquet files — still in progress"
    else
      check "Historical data downloaded" "fail" "No parquet files found"
    fi
  else
    check "Historical data downloaded" "warn" "Cannot check blob storage"
  fi
else
  check "Historical data downloaded" "fail" "STORAGE_ACCOUNT not set"
fi

# ── 5. Models trained ──────────────────────────────────────────────────────────
echo "[ 5/11] Model training..."
if echo "$LOGS" | grep "prediction_generated" > /dev/null 2>&1; then
  CONF=$(echo "$LOGS" | grep "prediction_generated" | tail -1 \
    | grep -o "'confidence': [0-9.]*" | head -1 | grep -o "[0-9.]*")
  MODEL=$(echo "$LOGS" | grep "prediction_generated" | tail -1 \
    | grep -o "'model': '[^']*'" | head -1 | cut -d"'" -f4)
  check "Models trained" "pass" "Generating predictions (model=$MODEL, conf=$CONF)"
else
  check "Models trained" "fail" "No predictions found"
fi


# ── 6. Live price stream ───────────────────────────────────────────────────────
echo "[ 6/11] Live price stream..."
if echo "$LOGS" | grep "tick_milestone" > /dev/null 2>&1; then
  TOTAL_TICKS=$(echo "$LOGS" | grep "tick_milestone" | tail -1 \
    | grep -o "'ticks': [0-9]*" | head -1 | grep -o "[0-9]*")
  LATEST_PAIR=$(echo "$LOGS" | grep "tick_milestone" | tail -1 \
    | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
  check "Live price stream active" "pass" "Latest: $TOTAL_TICKS ticks | Pair: $LATEST_PAIR"
else
  check "Live price stream active" "warn" "No ticks yet"
fi

# ── 7. Signals being evaluated ─────────────────────────────────────────────────
echo "[ 7/11] Signal evaluation..."
ATTEMPTS=$(echo "$LOGS" | grep -c "signal_debug\|signal_attempt" 2>/dev/null || echo "0")
[[ ! "$ATTEMPTS" =~ ^[0-9]+$ ]] && ATTEMPTS=0

FIRED=$(echo "$LOGS" | grep -c "signal_fired" 2>/dev/null || echo "0")
[[ ! "$FIRED" =~ ^[0-9]+$ ]] && FIRED=0

REJECTED=$(echo "$LOGS" | grep -c "gate_rejected" 2>/dev/null || echo "0")
[[ ! "$REJECTED" =~ ^[0-9]+$ ]] && REJECTED=0

if [[ "$ATTEMPTS" -gt 0 || "$REJECTED" -gt 0 ]]; then
  check "Signals being evaluated" "pass" "Attempts: $ATTEMPTS | Fired: $FIRED | Rejected: $REJECTED"
else
  check "Signals being evaluated" "warn" "No signal events yet"
fi

# ── 8. Quotex account (use dashboard /status) ─────────────────────────────────
echo "[ 8/11] Quotex account..."
if [[ -n "$STATUS_JSON" ]]; then
  QX_CONNECTED=$(echo "$STATUS_JSON" | grep -o '"connected": true' | head -1 || echo "")
  QX_BALANCE=$(echo "$STATUS_JSON" | grep -o '"balance": [0-9.]*' | head -1 | grep -o "[0-9.]*" || echo "")
  if [[ -n "$QX_CONNECTED" ]]; then
    check "Quotex account connected" "pass" "Balance: \$${QX_BALANCE:-0}"
  else
    check "Quotex account connected" "warn" "Not connected"
  fi
else
  check "Quotex account connected" "warn" "Dashboard unreachable — cannot verify"
fi

# ── 9. Webhook ─────────────────────────────────────────────────────────────────
echo "[ 9/11] Webhook endpoint..."
if echo "$LOGS" | grep -q "signal_fired\|signal_sent"; then
  check "Webhook endpoint reachable" "pass" "Signals have been sent"
elif echo "$LOGS" | grep -q "webhook_failed"; then
  check "Webhook endpoint reachable" "fail" "Webhook failed — check WEBHOOK_URL"
else
  check "Webhook endpoint reachable" "warn" "No signals fired yet — waiting for confidence threshold"
fi

# ── 10. Discord webhook ───────────────────────────────────────────────────────
echo "[10/11] Discord webhook..."
DISCORD_INSIDE=$(az container exec \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --exec-command "printenv DISCORD_WEBHOOK_URL" 2>/dev/null || echo "")
if [[ -n "$DISCORD_INSIDE" ]]; then
  check "Discord webhook configured" "pass"
else
  check "Discord webhook configured" "warn" "Not set inside container"
fi

# ── 11. Telegram bot ──────────────────────────────────────────────────────────
echo "[11/11] Telegram bot..."
TG_TOKEN_INSIDE=$(az container exec \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --exec-command "printenv TELEGRAM_TOKEN" 2>/dev/null || echo "")
TG_CHAT_INSIDE=$(az container exec \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --exec-command "printenv TELEGRAM_CHAT_ID" 2>/dev/null || echo "")
if [[ -n "$TG_TOKEN_INSIDE" && -n "$TG_CHAT_INSIDE" ]]; then
  check "Telegram bot configured" "pass"
else
  check "Telegram bot configured" "warn" "Missing inside container"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
if [[ "$ALL_PASSED" == "true" ]]; then
  echo "  ALL CHECKS PASSED — System is ready."
else
  echo "  SOME CHECKS FAILED — Review items above."
  echo ""
  echo "  View live logs:"
  echo "    az container logs --name $ACI_NAME --resource-group $RESOURCE_GROUP --follow"
fi
echo "=========================================="
echo ""
