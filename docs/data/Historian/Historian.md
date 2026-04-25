# TRADER-AI DATA: HISTORIAN.PY DESIGN DOCUMENT

VERSION: 1.1.0 | STATUS: FROZEN / PRODUCTION-READY

ROLE: DATA ARCHAEOLOGIST | VISION: ZERO-GAP HISTORY | RATE-SAFE | CRASH-RESILIENT

---

## 1. ARCHITECTURAL PHILOSOPHY

The `historian.py` module acts as the "Archaeologist" of the system. While
`storage.py` owns the vault and `model.py` guards the doorstep, the Historian
is responsible for excavating and delivering the raw material — two years of
validated M1 OHLCV bars — that the ML training pipeline depends on.

It operates under three core principles:

**The Gap Principle**: The Historian never fetches data that already exists.
On every run, it asks Storage: "What is the last bar you have?" and begins
exactly one minute later. A first run fetches everything from `BACKFILL_YEARS`
ago. Every subsequent run fetches only the gap since the last session. This
makes the system re-entrant and crash-safe at chunk boundaries.

**The Rate Principle**: The Twelve Data free tier allows 8 requests per minute
(800/day). The Historian enforces an 8-second inter-request interval before
every API call, regardless of which symbol or chunk is being fetched. This is
not configurable — violating the rate limit results in account suspension.

**The Dictator Pattern** (consistent with `config.py`, `model.py`, `storage.py`):
The Historian either constructs fully or fails immediately. It never enters a
partially-initialised state. All failures are reported with high-visibility
diagnostic blocks using the Core Diagnostic Language.

---

## 2. THE DIAGNOSTIC LANGUAGE

Consistent with the Core Diagnostic Registry, the Historian uses specific
symbols to identify the nature of each log event:

| Symbol | Category | Context / Usage                                                     |
|--------|----------|---------------------------------------------------------------------|
| `[+]`  | SUCCESS  | Backfill complete, chunk committed, data up to date.                |
| `[!]`  | FATAL    | Invalid API key, all retries exhausted, HistorianError raised.      |
| `[%]`  | LOGIC    | Malformed bar skipped, OHLC violation, volume floor applied.        |
| `[^]`  | NETWORK  | HTTP 429 rate-limit response, aiohttp.ClientError, request timeout. |

---

## 3. CORE COMPONENTS

### A. The Custom Exception (`HistorianError`)

A typed exception distinct from `StorageError` (persistence) and
`aiohttp.ClientError` (network). Carries a `symbol` attribute so the pipeline
can identify which pair failed without parsing the error message string.

Raised only when all `_MAX_RETRIES` fetch attempts have been exhausted for a
chunk. The `backfill_all()` method catches `HistorianError` per-symbol and
continues to the next pair rather than halting the session.

### B. The Gap Detector (`_determine_start`)

Calls `Storage.get_bars(symbol, "M1", max_rows=1)` to read the single most
recent bar for a symbol. Two outcomes:

- **No data** (first run): Returns `now - BACKFILL_YEARS`, normalised to
  midnight UTC so the first chunk begins at a clean day boundary.
- **Data exists** (resume): Returns `last_bar.timestamp + 1 minute`, starting
  the fetch immediately after the already-stored data.

The method never touches the raw tick store (`data/raw/`). It queries only the
processed bar store (`data/processed/`), because the Historian's output is
Bars — it is the upstream supplier of processed data, not ticks.

### C. The Chunk Walker (`_fetch_and_save`)

Walks forward from `start_dt` to `end_dt` in `_CHUNK_DAYS` (7-day) windows,
calling `_fetch_chunk` for each window. Each chunk:

1. Enforces the inter-request rate limit.
2. Issues one HTTP GET to the Twelve Data `/time_series` endpoint.
3. Parses the response into validated `Bar` objects.
4. Persists the bars via `Storage.save_bar_batch()` — one Parquet operation
   per chunk, not one per bar.

The loop advances `chunk_start` after every chunk regardless of whether bars
were returned. Empty chunks (weekends, holidays, illiquid windows) are logged
at DEBUG level and skipped silently.

### D. The Rate Limiter (`_enforce_rate_limit`)

Calculates `elapsed = monotonic() - _last_request_time` and calls
`asyncio.sleep(remaining)` if the full `_REQUEST_INTERVAL_S` (8 seconds) has
not elapsed. The monotonic clock is used instead of wall time to prevent
drift from system clock adjustments.

The `_last_request_time` attribute is updated immediately after each HTTP
response is received (not before the request), so the sleep accurately
accounts for actual round-trip time. Slow responses naturally reduce or
eliminate the inter-request sleep.

### E. The Bar Factory (`_parse_bars`)

Converts raw Twelve Data API dictionaries into validated `Bar` objects.

Key behaviours:

- **Reversal**: The API returns bars newest-first. The method reverses the
  list before constructing bars so the output is chronological, matching
  the expected Storage write order.
- **Volume floor**: `max(raw_volume, _MIN_VOLUME)` prevents zero-volume bars.
  Forex and OTC/synthetic pairs on Twelve Data sometimes report `volume=0`.
  A floor of `1.0` ensures bars are never silently accepted with invalid volume
  or rejected by future Bar validation tightening.
- **Skip-and-continue**: Malformed bars (missing fields, non-numeric prices,
  OHLC violations caught by `Bar.__post_init__`) are skipped with a `[%]`
  warning. A single bad bar does not abort the entire chunk.

### F. The Batch Writer (`_save_bars`)

Delegates to `Storage.save_bar_batch()`, which performs a single atomic
`_atomic_upsert` for the entire chunk. This is dramatically more efficient
than calling `Storage.save_bar()` in a loop:

| Strategy              | Parquet operations for 2yr backfill (~250k bars) |
|-----------------------|--------------------------------------------------|
| `save_bar()` per bar  | ~250,000 read + write cycles                     |
| `save_bar_batch()`    | ~100 read + write cycles (one per chunk)         |

A `StorageError` from the batch write is caught and logged as a `[%]` warning.
The chunk is marked as "0 bars saved" and the walk continues to the next chunk.

### G. The Consecutive Failure Guard (`_fetch_and_save`)

Each chunk where bars were fetched but `_save_bars` returned `0` increments a
`consecutive_failures` counter. When this counter reaches
`_MAX_CONSECUTIVE_STORAGE_FAILURES` (3), the backfill is aborted immediately
by raising `HistorianError`.

**Why 3?**: Three consecutive failures indicates a systemic storage condition
(disk full, mount lost, permissions revoked) rather than a transient blip.
Continuing to fetch from the Twelve Data API at this point would burn the
800 req/day quota for data that cannot land on disk. Aborting preserves the
quota; gap detection will resume from the last successfully stored bar on the
next run.

**Empty chunks are exempt**: Chunks that return zero bars (weekends, public
holidays, illiquid windows) do **not** advance the counter. Only a non-empty
chunk where storage silently fails counts.

**Counter resets on success**: A single successful `_save_bars` call resets
the counter to `0`, so two failures followed by a success followed by two more
failures never triggers the abort threshold.

### H. The Singleton (`get_historian`)

`get_historian()` is a module-level factory that returns the same `Historian`
instance on every call — mirroring `get_settings()` from `config.py`.

**Why a singleton matters for rate limiting**: `_last_request_time` is an
instance attribute on `Historian`. If two independent `Historian()` calls
produced two separate instances, each would have its own `_last_request_time`
clock. Both could fire requests within the same 8-second window — halving the
effective rate guard and risking account suspension on the free tier.

`get_historian()` guarantees that all callers share one `_last_request_time`,
making the rate limit hold regardless of how many modules import it.

---

## 4. GUARDRAIL REGISTRY

### I. Rate Limit Enforcement

- `_enforce_rate_limit()` is called **before every HTTP attempt**, including
  retries. This ensures the rate limit is respected even when retrying after
  a 429 or network error.
- The Twelve Data free tier is 8 req/min. The 8-second interval provides a
  7.5 req/min effective rate — safely within limits with headroom for jitter.
- `_REQUEST_INTERVAL_S = 8.0` is a module-level constant, not a config field.
  It is not user-configurable to prevent accidental rate-limit violations.

### II. Retry Logic

- Each chunk is retried up to `_MAX_RETRIES` (3) times on transient failures.
- Backoff starts at `_RETRY_BACKOFF_S` (15 seconds) and doubles on each retry:
  15s → 30s → 60s. Maximum total wait: 105 seconds per chunk.
- HTTP 429 is handled identically to other error codes, but the backoff is
  applied in addition to the normal rate-limit sleep.
- After all retries are exhausted, `HistorianError` is raised with the
  `[!]` diagnostic block.

### III. Batch Integrity

- `_save_bars` enforces single-symbol, single-timeframe batches via the
  `Storage.save_bar_batch()` contract.
- If a StorageError occurs, the chunk is discarded and the walk continues.
  The gap is recoverable on the next run (gap detection will restart from
  the last successfully stored bar).

### VI. Consecutive Storage Failure Abort

- `_MAX_CONSECUTIVE_STORAGE_FAILURES = 3` is the threshold after which
  `_fetch_and_save` raises `HistorianError` to abort the backfill.
- Empty chunks (zero bars returned) do not increment the counter.
- A successful save resets the counter to zero.
- See Section 3G for full rationale.

### IV. Volume Safety

- `_MIN_VOLUME = 1.0` is applied as a floor in `_parse_bars()`.
- This protects against two failure modes simultaneously:
  1. Current `Bar.__post_init__`: accepts zero volume (only rejects negative).
  2. Future `Bar.__post_init__`: may reject zero volume per the test harness spec.
  Either way, the Historian's output is always safe.

### V. Timezone Consistency

- `_determine_start` attaches `timezone.utc` to any naive `datetime` returned
  by Parquet/pandas before arithmetic operations.
- `_parse_bars` passes naive datetimes to `Bar` — `Bar.__post_init__` strips
  tzinfo automatically, producing consistent naive-UTC storage.
- API `start_date`/`end_date` params are always formatted from UTC datetimes.

---

## 5. DATA FLOW

```text
HISTORIAN DATA FLOW

  Storage.get_bars(max_rows=1)
          │
          ▼
  _determine_start(symbol, now_utc)
          │ Returns: start_dt (UTC)
          ▼
  _fetch_and_save(symbol, start_dt, end_dt)
          │
          └── for each 7-day chunk:
                    │
                    ▼
              _enforce_rate_limit()   ← 8-second inter-request gate
                    │
                    ▼
              _fetch_chunk(session, ...) ──── HTTP GET /time_series ────► TwelveData API
                    │                                                            │
                    │ <─────────────────── JSON { "values": [...] } ────────────┘
                    │
                    ▼
              _parse_bars(symbol, values)
                    │ Returns: list[Bar] (chronological, validated)
                    │
                    ▼
              _save_bars(symbol, bars)
                    │
                    ▼
              Storage.save_bar_batch(bars)   ← single Parquet upsert
                    │
                    ▼
          data/processed/{symbol}_M1.parquet  (updated)
```

---

## 6. MODULE CONSTANTS

| Constant                              | Value  | Rationale                                                   |
|---------------------------------------|--------|-------------------------------------------------------------|
| `_API_BASE`                           | https  | Twelve Data REST root URL.                                  |
| `_BARS_PER_REQUEST`                   | 5000   | Hard cap per Twelve Data API documentation.                 |
| `_REQUEST_INTERVAL_S`                 | 8.0 s  | 7.5 req/min — safely under the 8 req/min free tier.         |
| `_HTTP_TIMEOUT_S`                     | 30.0 s | Generous timeout; Twelve Data is typically fast.            |
| `_MAX_RETRIES`                        | 3      | 3 attempts: initial + 2 retries.                            |
| `_RETRY_BACKOFF_S`                    | 15.0 s | Doubles per retry: 15s → 30s → 60s.                         |
| `_CHUNK_DAYS`                         | 7      | 7 days × ~16 market hrs × 60 min ≈ 6720 M1 bars.            |
| `_MIN_VOLUME`                         | 1.0    | Volume floor: protects against zero-volume OTC bars.        |
| `_MAX_CONSECUTIVE_STORAGE_FAILURES`   | 3      | Consecutive failed saves before aborting to preserve quota. |

---

## 7. PUBLIC API REFERENCE

| Method                      | Returns          | Description                                      |
|-----------------------------|------------------|--------------------------------------------------|
| `backfill(symbol)`          | `int`            | Gap-backfill one symbol. Returns bars committed. |
| `backfill_all()`            | `dict[str, int]` | Gap-backfill all BACKFILL_PAIRS sequentially.    |

### Module-Level Factory

| Function          | Returns     | Description                                                      |
|-------------------|-------------|------------------------------------------------------------------|
| `get_historian()` | `Historian` | Returns the shared singleton instance. Creates it on first call. |

### Private Methods (Internal Use Only)

| Method                                         | Returns      | Description                                          |
|------------------------------------------------|--------------|------------------------------------------------------|
| `_determine_start(symbol, now_utc)`            | `datetime`   | Find UTC start of backfill window via Storage.       |
| `_fetch_and_save(symbol, start_dt, end_dt)`    | `int`        | Walk forward in chunks, fetch and persist each.      |
| `_fetch_chunk(session, api_symbol, ...)`       | `list[Bar]`  | Fetch one 7-day window from Twelve Data API.         |
| `_enforce_rate_limit()`                        | `None`       | Sleep until 8-second inter-request interval clears.  |
| `_parse_bars(symbol, values)`                  | `list[Bar]`  | Parse API dicts into validated, chronological Bars.  |
| `_save_bars(symbol, bars)`                     | `int`        | Persist a chunk batch via Storage.save_bar_batch().  |

---

## 8. FAIL-FAST SCENARIOS

| Trigger                                    | Symbol | Exception / Action          | Context                               |
|--------------------------------------------|--------|-----------------------------|---------------------------------------|
| All retries exhausted on a chunk           | `[!]`  | `HistorianError`            | Network down, invalid key, quota hit  |
| 3 consecutive storage failures             | `[!]`  | `HistorianError` (abort)    | Disk full / mount lost — quota guard  |
| API returns no "values" key                | `[!]`  | Returns `[]` (no crash)     | API-level error; logged as warning    |
| Malformed bar field (KeyError)             | `[%]`  | Bar skipped, walk continues | Missing "open"/"high" etc in response |
| OHLC violation in Bar construction         | `[%]`  | Bar skipped, walk continues | `Bar.__post_init__` raises ValueError |
| StorageError on batch write                | `[%]`  | Chunk discarded, walk cont. | Recoverable on next run               |
| `get_bars` returns None (first run)        | `[+]`  | Full backfill from cutoff   | Normal first-run path                 |
| Data already up to date                    | `[+]`  | Returns 0, no API call      | Normal resume path                    |

Note: Unlike `config.py` and `storage.py`, the Historian does **not** call
`sys.exit()` on failure. A backfill failure is recoverable — the system can
trade with whatever data it has and retry the gap on the next session. The
consecutive-failure abort raises `HistorianError` (caught by `backfill_all`)
rather than exiting, so other pairs in the session are unaffected.

---

## 9. V3 ARCHITECTURE INTEGRATION

In the target V3 "Hybrid Data-as-Code" architecture (see `docs/Issues.bash`),
the Historian runs **locally** as part of `deploy/sync-history.sh`:

```text
[LOCAL] sync-history.sh
    └── asyncio.run(historian.backfill_all())
            │
            ▼
    data/processed/EUR_USD_M1.parquet  ← committed to git
    data/processed/GBP_USD_M1.parquet  ← committed to git
            │
            ▼
    Docker build: COPY data/ /app/data/
            │
            ▼
    [AZURE] Container starts with 2yr history already on disk
    No API calls at startup — training begins immediately.
```

The Historian module itself does not change between local and Azure execution.
Only the caller changes — `sync-history.sh` (local) vs `src/core/pipeline.py`
(Azure Stage 1). The API is identical in both contexts.

---

## 10. ENVIRONMENT CONFIGURATION

The Historian reads two settings from config:

```env
TWELVEDATA_API_KEY=your_32_char_key   # Required. System exits if missing.
BACKFILL_YEARS=2                      # Default: 2. How many years to fetch.
BACKFILL_PAIRS=EUR_USD,GBP_USD        # Default: EUR_USD. Pairs for backfill_all().
```

No other settings are consumed by the Historian directly. Storage and Config
are injected at construction via `get_settings()` and `Storage()`.

---

## 11. VERIFICATION & TEST COVERAGE

STATUS: VERIFIED | TEST RUNNER: pytest-asyncio | TOTAL TESTS: 55 | COVERAGE: 100%

All logic in `historian.py` has been verified against a complete async test
harness. Tests are organised into 10 groups, each targeting a specific layer
of the module. The full harness is documented in `Historian_Test_Harness.md`.

### Test Group Summary

| Group | File                         | Scope                                              | Tests |
|-------|------------------------------|----------------------------------------------------|-------|
| 1     | test_historian_error.py      | `HistorianError` construction and attributes       | 2     |
| 2     | test_init.py                 | `Historian.__init__`, `get_historian()` singleton  | 6     |
| 3     | test_determine_start.py      | Gap detection: first run vs resume                 | 6     |
| 4     | test_parse_bars.py           | Bar parsing, reversal, volume floor, skipping      | 11    |
| 5     | test_save_bars.py            | Batch persistence delegation and error path        | 4     |
| 6     | test_rate_limit.py           | Inter-request sleep enforcement                    | 3     |
| 7     | test_backfill.py             | Public `backfill()` orchestration                  | 5     |
| 8     | test_backfill_all.py         | Multi-pair `backfill_all()` orchestration          | 5     |
| 9     | test_fetch_chunk.py          | HTTP fetch, retry, backoff doubling, exhaustion    | 8     |
| 10    | test_fetch_and_save.py       | Chunk walker, failure counter, singleton guard     | 10    |

### Key Behaviours Verified

- `HistorianError` carries `symbol` attribute and inherits from `Exception`.
- `_determine_start` returns midnight-normalised datetime on first run.
- `_determine_start` returns `last_ts + 1 minute` on resume.
- `_determine_start` handles `pd.Timestamp` (calls `to_pydatetime()`) and empty DataFrames.
- `_parse_bars` reverses API order to produce chronological output.
- `_parse_bars` applies `_MIN_VOLUME` floor when `volume=0`, missing, or negative.
- `_parse_bars` skips malformed bars without aborting the chunk.
- `_parse_bars` logs a WARNING with a skip count when bars are discarded.
- `_save_bars` delegates to `Storage.save_bar_batch()`, not individual `save_bar()`.
- `_save_bars` returns 0 and logs on `StorageError` — does not re-raise.
- `_enforce_rate_limit` sleeps only the remaining portion of the interval.
- `_enforce_rate_limit` does not sleep if the interval has already elapsed.
- `backfill()` returns 0 immediately if `start_dt >= now_utc`.
- `backfill_all()` catches `HistorianError` per-symbol and continues.
- `backfill_all()` returns `{}` immediately for an empty `backfill_pairs` list.
- `_fetch_chunk` handles HTTP 429, 500, 503 with backoff and retry.
- `_fetch_chunk` backoff doubles: 15s → 30s on repeated failures.
- `_fetch_chunk` retries on `aiohttp.ServerTimeoutError` and `ClientConnectionError`.
- `_fetch_chunk` raises `HistorianError` after all retries are exhausted.
- `_fetch_chunk` returns `[]` for API-level errors (no "values" in body).
- `_fetch_and_save` aborts after `_MAX_CONSECUTIVE_STORAGE_FAILURES` consecutive failed saves.
- `_fetch_and_save` resets the failure counter after a successful save.
- `_fetch_and_save` does not count empty chunks toward the failure threshold.
- `get_historian()` returns the same instance on repeated calls (singleton).
- `get_historian()` shares `_last_request_time` across all callers.

### Running the Suite

```bash
pip install pytest pytest-asyncio pytest-cov aioresponses
pytest tests/data/historian/ -v --cov=src.data.historian --cov-report=term-missing
```

---

## "IF IT IS NOT IN THE ARCHIVE, IT CANNOT BE TRADED."
