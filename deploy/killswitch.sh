#!/usr/bin/env bash
set -euo pipefail

az container stop \
  --name trader-ai-engine \
  --resource-group rg-trader-ai \
  --subscription 9086726b-ad07-4df0-90de-2ef7b6f6389b

echo "Container stopped."
