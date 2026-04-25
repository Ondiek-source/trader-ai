# Trader AI — Pre-Go-Live Checklist

**Living document.** Before every deployment, verify all items. Update status and notes when fixes are made.

Last full audit: 2026-04-16
Current status: **READY** (all blockers resolved)

---

## How to use

1. Read through every item before running `deploy.sh`.
2. Any item marked `[ ]` (not checked) or `[!]` (needs attention) must be resolved first.
3. After fixing an issue, update the checkbox and add a note with the date.
4. For a full re-audit, ask Claude to "verify the pre-go-live checklist."

---

## 1. Configuration

- [x] **All `.env` keys consumed** — Every key in `.env` is read by `config.py:load_config()`. No orphaned keys.
  - `MODEL_LOAD_ON_STARTUP` was orphaned; removed 2026-04-16.
- [x] **Every `Config` field has a `_parse_*` call** — `config.py:730-803` covers all 40+ fields.
- [x] **Validation ranges are sensible** — Martingale ceiling check, port range (1-65535), interval minimums (60s), expiry whitelist {60, 300, 900}.
- [x] **`.env.example` is in sync** — Contains all keys consumed by `config.py`, with blanked secrets. Updated 2026-04-16.
- [ ] **`WEBHOOK_SECRET`** — Parsed by `config.py` but not present in `.env` or `.env.example`. Clarify: is this required for your webhook endpoint? If yes, add to `.env`.

---

## 2. Entrypoint & Boot Sequence

- [x] **`main.py` calls `Pipeline.run()`** — `src/main.py:71-76`.
- [x] **Top-level crash handler** — `src/main.py:79-96` catches all exceptions, logs CRITICAL, calls `sys.exit(1)`.
- [x] **JSON structured logging configured at boot** — `src/main.py:24-44`, before any component initialization.
- [x] **Boot notifications sent** — `pipeline.py:780` calls `_send_boot_notification()` before `asyncio.gather()`. Sends HTML to Telegram and plain text to Discord. Gracefully skipped if credentials are missing.

---

## 3. Dashboard

- [x] **`/status` returns JSON** — `dashboard.py:503-510`, `Content-Type: application/json`.
- [x] **JS polls every 10 seconds** — `dashboard.py:376`, `setInterval(refresh, 10000)`.
- [x] **All StatusStore fields are pushed**:
  - `stopped` — `pipeline.py:Pipeline.stop()` (pushed on shutdown)
  - `session.is_active` — set `False` by `Pipeline.stop()` and `live.py:_on_kill_switch_activated()`
  - `kill_switch_active` — `live.py:_on_kill_switch_activated()`
  - `martingale_streak`, `confidence_threshold` — `live.py:_push_threshold_to_dashboard()`
  - `stream.connected`, `stream.ticks_received` — `live.py:_push_stream_status()` (uses real stream state, not hardcoded)
  - `session.signals_fired`, `session.elapsed_minutes` — `live.py:_push_stream_status()` (works for all stream types)
  - `quotex.balance`, `quotex.connected` — `pipeline.py:_quotex_status_loop()` (runs every 30s when `USE_QUOTEX_STREAMING=True`)
  - Seed values (zeros, practice_mode, thresholds) — set at boot in `_run_task_group()`
- [x] **`run_dashboard()` is started as an asyncio task** — `pipeline.py:715-719`.

---

## 4. Shutdown & Cleanup

- [x] **`Pipeline.stop()` stops all engines** — `pipeline.py:226-227`.
- [x] **All asyncio tasks cancelled** — `pipeline.py:229-231`.
- [x] **Pushes `stopped=True` and `session.is_active=False`** — `pipeline.py:237-243`.
- [x] **Awaits `reporter.close()`** — `pipeline.py:248-252`. Cancels Telegram poll_loop task and closes aiohttp sessions.
- [x] **Signal handlers registered** — SIGINT/SIGTERM trigger `stop()` gracefully.

---

## 5. Reporter & Notifications

- [x] **`Reporter.close()` exists** — `reporter.py:719`. Sub-reporters (Discord, Telegram) also have `close()`.
- [x] **Daily reports at 23:50 UTC** — `pipeline.py:_daily_report_loop()`. Fires at `86400 - 600 = 85800s` past midnight, sleeps in 60s chunks for responsive shutdown.
- [x] **Telegram + Discord both supported** — Initialized conditionally if credentials provided.
- [x] **Boot notification on deploy** — HTML to Telegram, plain text to Discord.

---

## 6. Stream & Kill Switch

- [x] **`stream.connected` derived from real state** — `live.py:_push_stream_status()`:
  - QuotexDataStream: reads `stream._connected` (bool)
  - TwelveDataStream: checks `stream._thread.is_alive()`
  - Fallback: assumes connected (tick just received)
- [x] **Kill switch pushes correct state** — `live.py:_on_kill_switch_activated()` sets `kill_switch_active=True` and `session.is_active=False`.
- [x] **Kill switch gate in `_process_tick`** — Halted engine skips execution but continues inference/journaling.

---

## 7. Docker & Deployment

- [x] **`.dockerignore` excludes `.env`** — Line 6. Secrets never bake into the image.
- [x] **HEALTHCHECK uses available command** — `curl -sf http://localhost:8080/status`. `curl` is installed in the Dockerfile. `pgrep` (not available in `python:3.13-slim`) was removed 2026-04-16.
- [x] **Port consistency** — `DASHBOARD_PORT=8080` in `.env`, `CONTAINER_PORT=8080` (default) in `deploy.sh`, HEALTHCHECK targets `localhost:8080`. All match.
  - Was `8501` in `.env` — fixed 2026-04-16.
- [x] **Data directories created in Dockerfile** — `/app/data/raw`, `/app/data/processed`, `/app/models`, `/tmp/fallback`. Owned by `appuser`.
- [x] **Non-root user** — `appuser` (uid 1000) runs the container.
- [x] **`--restart-policy Always`** — ACI restarts container on crash.

---

## 8. CI / CD

- [x] **`.github/workflows/deploy.yml` exists** — Builds and pushes image to ACR on push to `main` or manual dispatch. Tags with `latest`, git SHA, run number.
- [x] **No hardcoded credentials in workflows** — Secrets injected via GitHub Secrets (`ACR_LOGIN_SERVER`, `ACR_USERNAME`, `ACR_PASSWORD`).
- [x] **`deploy/azure.env` in `.gitignore`** — Line 3.
- [x] **`.env` in `.gitignore`** — Line 2.

---

## 9. Secrets & Security

- [x] **`.env` gitignored** — Never committed to git.
- [x] **`.env` dockerignored** — Never baked into image.
- [x] **Secrets injected at ACI runtime** — `deploy.sh` reads `.env`, classifies secure keys, passes them via `--secure-environment-variables`.
- [!] **Rotate secrets before first live deployment** — The current `.env` was used in development. Before going live, rotate:
  - `TWELVEDATA_API_KEY`
  - `AZURE_STORAGE_CONN` (AccountKey)
  - `QUOTEX_EMAIL` / `QUOTEX_PASSWORD` (if using a shared/dev account)
  - `TELEGRAM_TOKEN`
  - `DISCORD_WEBHOOK_URL`

---

## 10. Tests & Dependencies

- [x] **`tests/` directory with 47 test files** — Covers config validation, storage, historian, data, engine, ML engine, trading.
- [x] **`pytest.ini` configured** — `pythonpath=src` matches `Dockerfile`'s `PYTHONPATH=/app/src`.
- [x] **`requirements.txt` exists** — Used for local dev/CI. Note: Dockerfile hard-codes all deps in layers for better caching — `requirements.txt` is a subset for local tooling only. This is intentional.

---

## 11. Pre-Deploy Run Commands

Run these before hitting `deploy.sh`:

```bash
# 1. Confirm CI build succeeded
gh run list --repo Ondiek-source/trader-ai --limit 5

# 2. Confirm ACI credentials are set
source deploy/azure.env && echo "ACR: $ACR_LOGIN_SERVER | RG: $RESOURCE_GROUP"

# 3. Check .env has no missing required keys
grep -E "^(TWELVEDATA_API_KEY|QUOTEX_EMAIL|TELEGRAM_TOKEN|DISCORD_WEBHOOK_URL|AZURE_STORAGE_CONN)=" .env

# 4. Confirm dashboard port is 8080
grep DASHBOARD_PORT .env

# 5. Deploy
bash deploy/deploy.sh
```

---

## 12. Post-Deploy Verification

After `deploy.sh` completes:

```bash
# Check container is running
az container show --name $ACI_NAME --resource-group $RESOURCE_GROUP --query instanceView.state

# Stream live logs
bash deploy/get-logs.sh --live

# Check for boot errors in first 2 minutes
bash deploy/get-logs.sh --level ERROR

# Confirm boot notification received on Telegram/Discord
# (manual check)

# Check dashboard is reachable
curl -s http://<ACI_IP>:8080/status | python -m json.tool
```

---

## Known Limitations (not blockers)

| Item                                      | Detail                                                                                                                                                                                                                     |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WEBHOOK_SECRET`                          | Parsed by config but not in `.env`. Investigate if required.                                                                                                                                                               |
| Quotex streaming required for wins/losses | `quotex_status_loop` only updates balance/trade history when `USE_QUOTEX_STREAMING=True`. TwelveData users get live `elapsed_minutes` and `signals_fired` but not win/loss counts (Quotex API required for trade results). |
| Daily report for TwelveData-only          | `send_session_report()` includes wins/losses from StatusStore; if Quotex is off, these stay at zero.                                                                                                                       |
| `requirements.txt` scope                  | Not used in Dockerfile. Intentionally kept for local dev/test tooling.                                                                                                                                                     |
