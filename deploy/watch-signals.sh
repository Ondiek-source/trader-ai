#!/usr/bin/env bash
# watch-signals.sh — Live signal pipeline monitor
# Usage: bash deploy/watch-signals.sh

az container logs \
  --name trader-ai-engine \
  --resource-group rg-trader-ai \
  --follow 2>&1 | grep -E "health|quotex_reader|balance|feedback" | tail -20
  
az container logs \
  --name trader-ai-engine \
  --resource-group rg-trader-ai \
  --follow 2>&1 | grep -E "signal_debug|signal_task_started|prediction_result|gate_rejected|tick_milestone|signal_sent|feature_extraction_failed|webhook_failed"


