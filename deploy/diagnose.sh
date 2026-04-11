#!/usr/bin/env bash
# =============================================================================
# diagnose.sh — Interpreted diagnostic for Trader AI
# Usage: bash deploy/diagnose.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZURE_ENV="$SCRIPT_DIR/azure.env"
source "$AZURE_ENV"

LOGS=$(az container logs \
  --name "$ACI_NAME" \
  --resource-group "$RESOURCE_GROUP" 2>/dev/null || echo "")

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
IMAGE=$(az container show --name "$ACI_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "containers[0].image" -o tsv 2>/dev/null || echo "Unknown")

if [[ "$STATE" == "Running" ]]; then
  echo "  OK: Container is running"
else
  echo "  PROBLEM: Container state is $STATE — run deploy/deploy.sh"
fi

# ── Data Source ────────────────────────────────────────────────────────────────
echo ""
echo "── Data Source ────────────────────────────"
DS=$(echo "$LOGS" | grep "data_source_selected" | tail -1 | grep -o "'provider': '[^']*'" | head -1 | cut -d"'" -f4)

if [[ -z "$DS" ]]; then
  echo "  PROBLEM: No data source selected — old code may be running"
  echo "  FIX: Run deploy/deploy.sh to push latest image"
elif [[ "$DS" == "Quotex OTC" ]]; then
  echo "  OK: Streaming from Quotex OTC"

  if echo "$LOGS" | grep -q "quotex_stream.*quotex_connected"; then
    echo "  OK: Quotex WebSocket connected"
  elif echo "$LOGS" | grep -q "quotex_stream.*quotex_connect_retry"; then
    echo "  PROBLEM: Quotex WebSocket retrying — Cloudflare may be blocking"
    echo "  NOTE: Connection usually succeeds after 2-3 retries"
  fi
elif [[ "$DS" == "Twelve Data" ]]; then
  echo "  OK: Streaming from Twelve Data"

  if echo "$LOGS" | grep -q "twelveticks_stream.*stream_connected"; then
    echo "  OK: Twelve Data WebSocket connected"
  fi
fi

# ── Ticks ──────────────────────────────────────────────────────────────────────
echo ""
echo "── Ticks ──────────────────────────────────"
TICK=$(echo "$LOGS" | grep "tick_milestone" | tail -1)
if [[ -n "$TICK" ]]; then
  TICKS=$(echo "$TICK" | grep -o "'ticks': [0-9]*" | head -1 | grep -o "[0-9]*")
  TICKPAIR=$(echo "$TICK" | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
  echo "  OK: $TICKS ticks received (latest on $TICKPAIR)"
else
  if [[ "$DS" == "Quotex OTC" ]]; then
    echo "  PROBLEM: No ticks from Quotex — price methods failing"
    echo ""
    echo "  Symbol probe results:"
    SYMPROBE=$(echo "$LOGS" | grep "symbol_format_found" | tail -3)
    if [[ -n "$SYMPROBE" ]]; then
      echo "$SYMPROBE" | while IFS= read -r line; do
        SYM=$(echo "$line" | grep -o "'symbol': '[^']*'" | head -1 | cut -d"'" -f4)
        PRICE=$(echo "$line" | grep -o "'price': [0-9.]*" | head -1 | grep -o "[0-9.]*")
        FMT=$(echo "$line" | grep -o "'format': '[^']*'" | head -1 | cut -d"'" -f4)
        echo "    $SYM ($FMT) → $PRICE"
      done
    else
      PFAIL=$(echo "$LOGS" | grep "all_price_methods_failed" | tail -3)
      if [[ -n "$PFAIL" ]]; then
        echo "    All formats tried, none returned a price"
        echo "    FIX: Check if symbol format matches Quotex dashboard"
      else
        echo "    No probe results yet — stream may still be connecting"
      fi
    fi
  else
    echo "  PROBLEM: No ticks — check TWELVEDATA_API_KEY"
  fi
fi

# ── Training ───────────────────────────────────────────────────────────────────
echo ""
echo "── Training ───────────────────────────────"
TRAINED=$(echo "$LOGS" | grep -c "model_already_trained" || true)
ALL_TRAINED=$(echo "$LOGS" | grep -c "all_models_trained" || true)
BACKFILLING=$(echo "$LOGS" | grep -q "backfill_chunk" && echo "yes" || echo "no")

if [[ "$ALL_TRAINED" -gt 0 ]]; then
  echo "  OK: All models trained and ready"
elif [[ "$TRAINED" -gt 0 ]]; then
  echo "  IN PROGRESS: $TRAINED pair(s) trained"

  echo "$LOGS" | grep "training_waiting_for" | tail -5 | while IFS= read -r line; do
    PAIR=$(echo "$line" | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
    if echo "$line" | grep -q "coverage"; then
      SPAN=$(echo "$line" | grep -o "'span_days': [0-9]*" | head -1 | grep -o "[0-9]*")
      TARGET=$(echo "$line" | grep -o "'target_days': [0-9]*" | head -1 | grep -o "[0-9]*")
      PCT=$((SPAN * 100 / TARGET))
      echo "    $PAIR: $SPAN/$TARGET days ($PCT%) — backfilling"
    elif echo "$line" | grep -q "data"; then
      echo "    $PAIR: waiting for data — backfill hasn't reached this pair yet"
    fi
  done
elif [[ "$BACKFILLING" == "yes" ]]; then
  echo "  WAITING: Backfill still running — training starts after backfill completes"

  echo "$LOGS" | grep "backfill_chunk" | tail -1 | while IFS= read -r line; do
    PAIR=$(echo "$line" | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
    BARS=$(echo "$line" | grep -o "'total_bars': [0-9]*" | head -1 | grep -o "[0-9]*")
    echo "    Currently downloading: $PAIR ($BARS bars)"
  done
else
  echo "  PROBLEM: No training or backfill activity"
  echo "  FIX: Check logs for errors — az container logs ... --follow"
fi

# ── Signals ────────────────────────────────────────────────────────────────────
echo ""
echo "── Signals ────────────────────────────────"
FIRED=$(echo "$LOGS" | grep -c "signal_fired" || true)
REJECTED=$(echo "$LOGS" | grep -c "gate_rejected" || true)
WEBHOOK_FAIL=$(echo "$LOGS" | grep -c "webhook_failed" || true)

if [[ "$FIRED" -gt 0 ]]; then
  echo "  OK: $FIRED signal(s) fired ($REJECTED rejected, $WEBHOOK_FAIL webhook failures)"

  echo "$LOGS" | grep "signal_fired" | tail -3 | while IFS= read -r line; do
    SYM=$(echo "$line" | grep -o "'symbol': '[^']*'" | head -1 | cut -d"'" -f4)
    SIDE=$(echo "$line" | grep -o "'side': '[^']*'" | head -1 | cut -d"'" -f4)
    CONF=$(echo "$line" | grep -o "'confidence': [0-9.]*" | head -1 | grep -o "[0-9.]*")
    echo "    Last: $SYM $SIDE (conf=$CONF)"
  done
elif [[ "$REJECTED" -gt 0 ]]; then
  echo "  INFO: 0 fired, $REJECTED rejected — gate is blocking everything"

  echo "$LOGS" | grep "gate_rejected" | tail -5 | while IFS= read -r line; do
    PAIR=$(echo "$line" | grep -o "'pair': '[^']*'" | head -1 | cut -d"'" -f4)
    CONF=$(echo "$line" | grep -o "'confidence': [0-9.]*" | head -1 | grep -o "[0-9.]*")
    echo "    $PAIR: confidence $CONF"
  done
  echo "  FIX: Lower CONFIDENCE_THRESHOLD or wait for model retrain"
else
  echo "  WAITING: No signal activity — models still training or stream not connected"
fi


# ── Quotex Account ─────────────────────────────────────────────────────────────
echo ""
echo "── Quotex Account ─────────────────────────"
QXLINE=$(echo "$LOGS" | grep "quotex_connected" | grep "balance" | tail -1)
if [[ -n "$QXLINE" ]]; then
  BALANCE=$(echo "$QXLINE" | grep -o "'balance': [0-9.]*" | head -1 | grep -o "[0-9.]*")
  MODE=$(echo "$QXLINE" | grep -o "'mode': '[^']*'" | head -1 | cut -d"'" -f4)
  echo "  OK: Connected — $MODE — \$$BALANCE"
else
  if echo "$LOGS" | grep -q "quotex_connect_failed\|Websocket connection rejected"; then
    echo "  PROBLEM: Quotex connection failed — Cloudflare blocking WebSocket"
    echo "  NOTE: This usually resolves on retry. If persistent, consider a proxy."
  else
    echo "  WAITING: No Quotex connection events yet"
  fi
fi

# ── Errors ─────────────────────────────────────────────────────────────────────
ERROR_COUNT=$(echo "$LOGS" | grep -c '"level": "ERROR"' || true)
echo ""
echo "── Errors ─────────────────────────────────"
if [[ "$ERROR_COUNT" -eq 0 ]]; then
  echo "  OK: No errors"
else
  echo "  $ERROR_COUNT error(s) — showing most recent:"
  echo "$LOGS" | grep '"level": "ERROR"' | tail -5 | while IFS= read -r line; do
    TS=$(echo "$line" | grep -o '"timestamp": "[^"]*"' | head -1 | cut -d'"' -f4 | cut -c1-19)
    COMP=$(echo "$line" | grep -o '"component": "[^"]*"' | head -1 | cut -d'"' -f4)
    MSG=$(echo "$line" | grep -o "'event': '[^']*'" | head -1 | cut -d"'" -f4)
    [[ -z "$MSG" ]] && MSG=$(echo "$line" | grep -o '"message": "[^"]*"' | head -1 | cut -d'"' -f4 | cut -c1-60)
    echo "    $TS  [$COMP]  $MSG"
  done
fi

# ── Health ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Health ─────────────────────────────────"
HEALTH=$(echo "$LOGS" | grep '"event": "health"' | tail -1)
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
