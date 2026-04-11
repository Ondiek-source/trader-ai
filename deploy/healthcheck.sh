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

# ── Fetch logs once ────────────────────────────────────────────────────────────
LOGS=$(az container logs \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" 2>/dev/null || echo "")

FRESH_LOGS=$(echo "$LOGS" | tail -200)

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
  check "Container logs accessible" "fail" "No logs returned — container may still be starting"
else
  check "Container logs accessible" "pass"
fi

# ── 3. Data source ─────────────────────────────────────────────────────────────
echo "[ 3/11] Data source..."
if echo "$LOGS" | grep -q "data_source_selected"; then
  PROVIDER=$(echo "$LOGS" | grep "data_source_selected" | tail -1 \
    | grep -o "'provider': '[^']*'" | head -1 | cut -d"'" -f4)
  check "Data source selected" "pass" "Provider: $PROVIDER"

  if [[ "$PROVIDER" == "Quotex OTC" ]]; then
    if echo "$LOGS" | grep -q "quotex_connected"; then
      check "Quotex streaming connected" "pass"
    elif echo "$LOGS" | grep -q "quotex_connect_retry"; then
      check "Quotex streaming connected" "warn" "Retrying connection..."
    elif echo "$LOGS" | grep -q "quotex_disconnected"; then
      check "Quotex streaming connected" "fail" "Disconnected — check credentials"
    else
      check "Quotex streaming connected" "warn" "No connection events yet"
    fi
  fi

  if [[ "$PROVIDER" == "Twelve Data" ]]; then
    if echo "$LOGS" | grep -q "twelveticks_stream.*stream_connected"; then
      check "Twelve Data streaming connected" "pass"
    else
      check "Twelve Data streaming connected" "warn" "No connection confirmation yet"
    fi
  fi
else
  check "Data source selected" "fail" "No data_source_selected event — old code running?"
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
      check "Historical data downloaded" "fail" "No parquet files found in blob storage"
    fi
  else
    check "Historical data downloaded" "warn" "Cannot check blob storage — no access"
  fi
else
  check "Historical data downloaded" "fail" "STORAGE_ACCOUNT not set"
fi

# ── 5. Models trained ──────────────────────────────────────────────────────────
echo "[ 5/11] Model training..."
TRAINED_COUNT=$(echo "$LOGS" | grep -c "model_already_trained\|initial_training_complete" 2>/dev/null || echo "0")
TRAINED_COUNT=$(echo "$TRAINED_COUNT" | tr -d '\n\r' | xargs)
[[ ! "$TRAINED_COUNT" =~ ^[0-9]+$ ]] && TRAINED_COUNT=0

WAITING_COVERAGE=$(echo "$LOGS" | grep -c "training_waiting_for_coverage" 2>/dev/null || echo "0")
WAITING_COVERAGE=$(echo "$WAITING_COVERAGE" | tr -d '\n\r' | xargs)
[[ ! "$WAITING_COVERAGE" =~ ^[0-9]+$ ]] && WAITING_COVERAGE=0

BACKFILLING=$(echo "$LOGS" | grep -q "backfill_chunk" && echo "yes" || echo "no")

if echo "$LOGS" | grep -q "all_models_trained"; then
  check "All models trained" "pass"
elif [[ "$TRAINED_COUNT" -gt 0 ]]; then
  check "Models trained" "warn" "$TRAINED_COUNT pair(s) trained, $WAITING_COVERAGE waiting for coverage"
elif [[ "$BACKFILLING" == "yes" ]]; then
  check "Models trained" "warn" "Waiting for backfill to complete before training"
else
  check "Models trained" "fail" "No training events found — check logs for errors"
fi

# ── 6. Live price stream ───────────────────────────────────────────────────────
echo "[ 6/11] Live price stream..."
TICK_COUNT=$(echo "$FRESH_LOGS" | grep -c "tick_milestone" 2>/dev/null || echo "0")
TICK_COUNT=$(echo "$TICK_COUNT" | tr -d '\n\r' | xargs)
[[ ! "$TICK_COUNT" =~ ^[0-9]+$ ]] && TICK_COUNT=0

if [[ "$TICK_COUNT" -gt 0 ]]; then
  TOTAL_TICKS=$(echo "$FRESH_LOGS" | grep "tick_milestone" | tail -1 \
    | grep -o "'ticks': [0-9]*" | head -1 | grep -o "[0-9]*")
  LATEST_PAIR=$(echo "$FRESH_LOGS" | grep "tick_milestone" | tail -1 \
    | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
  check "Live price stream active" "pass" "Latest: $TOTAL_TICKS ticks | Pair: $LATEST_PAIR"
elif echo "$FRESH_LOGS" | grep -q "stream_stub_mode"; then
  check "Live price stream" "warn" "Running in stub/synthetic mode"
else
  check "Live price stream active" "warn" "No ticks yet — stream may still be connecting"
fi

# ── 7. Signals being evaluated ─────────────────────────────────────────────────
echo "[ 7/11] Signal evaluation..."
ATTEMPTS=$(echo "$LOGS" | grep -c "signal_attempt" 2>/dev/null || echo "0")
ATTEMPTS=$(echo "$ATTEMPTS" | tr -d '\n\r' | xargs)
[[ ! "$ATTEMPTS" =~ ^[0-9]+$ ]] && ATTEMPTS=0

FIRED=$(echo "$LOGS" | grep -c "signal_fired" 2>/dev/null || echo "0")
FIRED=$(echo "$FIRED" | tr -d '\n\r' | xargs)
[[ ! "$FIRED" =~ ^[0-9]+$ ]] && FIRED=0

SUPPRESSED=$(echo "$LOGS" | grep -c "signal_suppressed" 2>/dev/null || echo "0")
SUPPRESSED=$(echo "$SUPPRESSED" | tr -d '\n\r' | xargs)
[[ ! "$SUPPRESSED" =~ ^[0-9]+$ ]] && SUPPRESSED=0

if [[ "$ATTEMPTS" -gt 0 ]]; then
  check "Signals being evaluated" "pass" "Attempts: $ATTEMPTS | Fired: $FIRED | Suppressed: $SUPPRESSED"
elif echo "$LOGS" | grep -q "training_waiting"; then
  check "Signals being evaluated" "warn" "No signals yet — models still training"
else
  check "Signals being evaluated" "warn" "No signal events yet"
fi

# ── 8. Quotex account ──────────────────────────────────────────────────────────
echo "[ 8/11] Quotex account..."
if echo "$FRESH_LOGS" | grep -q "quotex_connected"; then
  BALANCE=$(echo "$FRESH_LOGS" | grep "quotex_connected" | tail -1 \
    | grep -o "'balance': [0-9.]*" | head -1 | grep -o "[0-9.]*")
  MODE=$(echo "$FRESH_LOGS" | grep "quotex_connected" | tail -1 \
    | grep -o "'mode': '[^']*'" | head -1 | cut -d"'" -f4)
  check "Quotex account connected" "pass" "$MODE | Balance: \$${BALANCE:-0}"
elif echo "$FRESH_LOGS" | grep -q "quotex_connect_failed\|Websocket connection rejected"; then
  REASON=$(echo "$FRESH_LOGS" | grep "quotex_connect_failed\|Websocket connection rejected" | tail -1 \
    | grep -o "'reason': '[^']*'" | head -1 | cut -d"'" -f4)
  check "Quotex account connected" "fail" "Failed: ${REASON:-connection rejected}"
elif echo "$FRESH_LOGS" | grep -q "Quotex credentials not set"; then
  check "Quotex account connected" "warn" "No credentials — set QUOTEX_EMAIL and QUOTEX_PASSWORD"
else
  check "Quotex account connected" "warn" "No Quotex events yet"
fi

# ── 9. Webhook ─────────────────────────────────────────────────────────────────
echo "[ 9/11] Webhook endpoint..."
if echo "$LOGS" | grep -q "signal_sent"; then
  LAST_STATUS=$(echo "$LOGS" | grep "signal_sent" | tail -1 \
    | grep -o "'http_status': [0-9]*" | head -1 | grep -o "[0-9]*")
  check "Webhook endpoint reachable" "pass" "Last HTTP status: $LAST_STATUS"
elif echo "$LOGS" | grep -q "webhook_failed"; then
  check "Webhook endpoint reachable" "fail" "Webhook failed — check WEBHOOK_URL"
else
  check "Webhook endpoint reachable" "warn" "No signals fired yet"
fi

# ── 10. Discord webhook ───────────────────────────────────────────────────────
echo "[10/11] Discord webhook..."
if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
  DISCORD_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"Healthcheck OK — $(date -u '+%Y-%m-%d %H:%M UTC')\"}" \
    "$DISCORD_WEBHOOK_URL" 2>/dev/null || echo "000")

  if [[ "$DISCORD_RESPONSE" == "200" || "$DISCORD_RESPONSE" == "204" ]]; then
    check "Discord webhook reachable" "pass" "HTTP $DISCORD_RESPONSE — test message sent"
  elif [[ "$DISCORD_RESPONSE" == "429" ]]; then
    check "Discord webhook reachable" "warn" "Rate limited (HTTP 429)"
  else
    check "Discord webhook reachable" "fail" "HTTP $DISCORD_RESPONSE — check DISCORD_WEBHOOK_URL"
  fi
else
  check "Discord webhook reachable" "warn" "DISCORD_WEBHOOK_URL not set"
fi

# ── 11. Telegram bot ──────────────────────────────────────────────────────────
echo "[11/11] Telegram bot..."
if [[ -n "${TELEGRAM_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
  TG_RESPONSE=$(curl -s \
    "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "text=Healthcheck OK — $(date -u '+%Y-%m-%d %H:%M UTC')" \
    2>/dev/null || echo "")

  TG_OK=$(echo "$TG_RESPONSE" | grep -c '"ok":true' 2>/dev/null || echo "0")
  TG_ERROR=$(echo "$TG_RESPONSE" | grep -o '"description":"[^"]*"' | head -1 | cut -d'"' -f4)

  if [[ "$TG_OK" -ge 1 ]]; then
    check "Telegram bot reachable" "pass" "Test message sent"
  elif [[ -n "$TG_ERROR" ]]; then
    check "Telegram bot reachable" "fail" "$TG_ERROR"
  else
    check "Telegram bot reachable" "fail" "No response — check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID"
  fi
else
  if [[ -z "${TELEGRAM_TOKEN:-}" ]]; then
    check "Telegram bot reachable" "warn" "TELEGRAM_TOKEN not set"
  else
    check "Telegram bot reachable" "warn" "TELEGRAM_CHAT_ID not set"
  fi
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
