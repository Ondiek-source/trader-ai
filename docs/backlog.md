# TRADER-AI: ENGINEERING BACKLOG

Last Updated: 2026-04-14

---

## PRIORITY TIERS

- **P1 — Before Production**: Must be resolved before live trading with real money.
- **P2 — First Month**: Important for reliability and operational visibility.
- **P3 — Ongoing**: Performance, security, and long-term maintainability.

---

## P1 — BEFORE PRODUCTION

---

### BL-01: Chunked Data Conversion (backfill / historian)

**Problem:** Processing 2 years of bars as a single in-memory object risks OOM crashes
and triggers the 20,000 tick safety cap.

**Actions:**

- [ ] Modify `backfill_pair` to iterate through 2-year history in 30-day windows.
- [ ] Implement flush-to-disk logic: save each month as a separate Parquet file
      (e.g., `EUR_USD_2024_01.parquet`) to local/Azure storage.
- [ ] Clear the local Python list/DataFrame after each month is saved to reset RAM.
- [ ] Remove `MAX_SEQUENCES` as a global conversion cap; use it only as a per-batch
      safety limit.

**Success Criteria:**

- A 2-year backfill creates ~24 monthly files.
- Memory usage (RSS) remains stable and does not climb linearly with time.

---

### BL-02: Parquet Schema Versioning

**Problem:** If a `Bar` or `Tick` field is added or renamed, `pd.concat` of old and new
DataFrames will fail or silently produce NaN columns. There is no migration path.

**Actions:**

- [ ] Add a `_SCHEMA_VERSION` constant to `storage.py`.
- [ ] Write the version as Parquet metadata on every upsert.
- [ ] On read, detect version mismatch and either migrate columns or log a clear error
      instructing the operator to re-run backfill.
- [ ] Document the migration procedure in a runbook.

**Success Criteria:**

- Deploying a schema change does not silently corrupt existing Parquet files.
- Mismatch produces a clear, actionable error rather than a pandas concat exception.

---

### BL-03: Timezone Validation Hardening

**Problem:** All stored timestamps are naive UTC by design, but no code path enforces
that only UTC-aware or naive datetimes enter the system. A non-UTC aware datetime
(e.g., `Europe/London`) would be stripped of tzinfo and stored as if it were UTC —
a silent, hard-to-detect bug.

**Actions:**

- [ ] In `Tick.__post_init__` and `Bar.__post_init__`, before stripping tzinfo, check:
      if `tzinfo is not None` and `utcoffset() != timedelta(0)`, raise `ValueError`.
- [ ] Add test cases for aware non-UTC datetimes to both `test_tick.py` and `test_bar.py`.

**Success Criteria:**

- Passing a `Europe/London` aware datetime raises `ValueError` immediately.
- Passing a UTC-aware datetime is accepted and stored as naive UTC (current behaviour).

---

### BL-04: API Request/Response Logging

**Problem:** When a backfill fails or returns unexpected data, there is no way to
inspect the exact HTTP request that was sent or the raw JSON response that was received.
Debugging requires guesswork.

**Actions:**

- [ ] Log the full request URL (with params, API key redacted) at DEBUG level before
      each `_fetch_chunk` call.
- [ ] Log the raw response status and first 500 chars of body at DEBUG level on any
      non-200 or error-body response.
- [ ] Ensure `TWELVEDATA_API_KEY` is never written to any log file.

**Success Criteria:**

- Running with `LOG_LEVEL=DEBUG` produces a complete audit trail of every HTTP
  request and response without exposing credentials.

---

## P2 — FIRST MONTH

---

### BL-05: Incremental Model Training

**Problem:** `ModelManager.train()` retrains from scratch on a single large array.
This will fail once data exceeds available RAM (~4 GB+).

**Actions:**

- [ ] Refactor `train()` to accept data in chunks or generators.
- [ ] LightGBM/XGBoost: use `init_model` parameter to resume from previous weights.
- [ ] PyTorch (LSTM/Transformer): implement a `Dataset`/`DataLoader` that streams
      Parquet files from disk instead of loading the full 2-year tensor.
- [ ] RandomForest: implement a sliding window (most recent N months) or switch to
      `SGDClassifier` with `partial_fit`.

**Success Criteria:**

- Training logs show `"Processing Chunk 1/24..."`, `"Processing Chunk 2/24..."`.
- Final model reflects 2 years of patterns without requiring 8 GB+ of RAM.

---

### BL-06: Resume & Warm-Start Logic

**Problem:** Every container restart re-processes 2 years of data from scratch.

**Actions:**

- [ ] Implement a check in `Storage` for existing monthly Parquet files; skip
      conversion for months that already exist.
- [ ] Implement warm-start: load the latest saved `.pkl` model and train only on
      data generated since the model's `last_updated` timestamp.

**Success Criteria:**

- Container restart takes < 5 minutes to reach "Trade Ready" state even with
  2 years of history in the background.

---

### BL-07: PyArrow Direct Reads for Columnar Queries

**Problem:** `Storage.get_last_timestamp()` and `get_tick_count()` use
`pd.read_parquet(columns=[...])` which relies on pandas to push the column
predicate down to PyArrow. For very large files (10M+ rows) this may load more
data than necessary.

**Actions:**

- [ ] Replace `pd.read_parquet(file_path, columns=[_TS_COLUMN])` with:
      ```python
      import pyarrow.parquet as pq
      table = pq.read_table(file_path, columns=[_TS_COLUMN])
      last_ts = table.column(_TS_COLUMN).to_pandas().max()
      ```
- [ ] Benchmark both approaches on a 5M-row file and document the result.
- [ ] Only adopt if the benchmark shows >20% improvement; otherwise close as
      "not needed at current scale".

**Success Criteria:**

- `get_last_timestamp` on a 5M-row file completes in < 100ms.

---

### BL-08: Symbol Name Format Validation

**Problem:** `EUR_USD` and `EURUSD` are both valid-looking strings but would create
two separate Parquet files, silently splitting the dataset.

**Actions:**

- [ ] Add a module-level regex in `storage.py`: `_SYMBOL_PATTERN = re.compile(r'^[A-Z]{3}_[A-Z]{3}$')`.
- [ ] Validate at entry points: `save_tick_batch`, `save_bar`, `save_bar_batch`,
      `get_last_timestamp`, `get_bars`, `get_tick_count`.
- [ ] Raise `ValueError` with a clear message if the pattern does not match.
- [ ] Add tests to `test_storage_init.py` or a new `test_symbol_validation.py`.

**Success Criteria:**

- Passing `"EURUSD"` raises `ValueError: Invalid symbol format`.
- Passing `"EUR_USD"` succeeds as before.

---

### BL-09: Performance Telemetry & Heartbeat Logging

**Problem:** Long-running backfill and training sessions produce no progress indication.
A stalled or memory-leaking job is indistinguishable from a slow one.

**Actions:**

- [ ] Add heartbeat logs to backfill: `{"event": "progress", "symbol": "EUR_USD", "percent": 45}`.
- [ ] Log RAM usage snapshots every 50,000 ticks during training.
- [ ] Log elapsed time and estimated time remaining per chunk in `_fetch_and_save`.

**Success Criteria:**

- Logs clearly indicate if a training session is stalled or leaking memory.
- A 2-year backfill produces one progress log per chunk (~100 entries).

---

### BL-10: Storage Failure Recovery Runbook

**Problem:** When `_fetch_and_save` aborts due to consecutive storage failures, the
operator receives a log entry with the failed window but no documented procedure
for recovery.

**Actions:**

- [ ] Write `docs/runbooks/storage_failure_recovery.md` covering:
      - How to identify the last successfully stored bar.
      - How to re-trigger backfill for a specific symbol and date range.
      - How to switch from CLOUD to LOCAL mode for offline recovery.
      - How to verify data integrity after recovery.
- [ ] Add a `--dry-run` flag to the backfill CLI entry point so operators can
      preview what would be fetched without consuming API quota.

**Success Criteria:**

- An on-call operator can recover from a storage failure without reading source code.

---

## P3 — ONGOING

---

### BL-11: Configuration Bounds Validation

**Problem:** Config fields like `backfill_years` and `tick_flush_size` have no max
validation. A misconfigured `.env` (e.g., `BACKFILL_YEARS=100`) fails silently or
causes a very long runtime.

**Actions:**

- [ ] Add range validation in `load_config()` for:
      - `backfill_years`: 1–10
      - `tick_flush_size`: 1–10,000
      - `martingale_max_streak`: 1–20
      - `confidence_threshold`: 0.5–1.0
- [ ] Raise `ValueError` with clear message on out-of-range values.

**Success Criteria:**

- `BACKFILL_YEARS=0` exits with a clear validation error at boot.

---

### BL-12: Azure Managed Identity (Replace Connection Strings)

**Problem:** `AZURE_STORAGE_CONN` is a full connection string — a powerful credential
that grants broad access. If leaked, it cannot be scoped or easily rotated.

**Actions:**

- [ ] Implement Azure Managed Identity authentication as the preferred CLOUD auth path.
- [ ] Fall back to connection string only when running outside Azure (local dev).
- [ ] Document the Managed Identity setup in `docs/deployment.md`.

**Success Criteria:**

- The production Azure container authenticates without any secrets in the environment.

---

### BL-13: Parquet Encryption at Rest

**Problem:** Market data and model artifacts are stored as unencrypted Parquet files.
On a shared VPS or a misconfigured Azure container, this data is readable by anyone
with filesystem access.

**Actions:**

- [ ] Evaluate PyArrow's built-in Parquet encryption (`encryption_properties`).
- [ ] Implement encryption for `data/processed/` files (model training data).
- [ ] Key management: use Azure Key Vault in CLOUD mode; local keyfile in LOCAL mode.

**Success Criteria:**

- Parquet files cannot be read without the encryption key.

---

### BL-14: Twelve Data API Order Assumption Verification

**Problem:** `_parse_bars` reverses the API response assuming newest-first order.
This is consistent with observed behaviour and the API docs, but it has not been
explicitly verified with a controlled test against the live API.

**Actions:**

- [ ] Write a one-off integration script that fetches a known date range and asserts
      `response["values"][0]["datetime"] > response["values"][-1]["datetime"]`.
- [ ] Run it against the live API once and commit the result to `docs/` as evidence.
- [ ] If the assumption is ever wrong, add a sort step in `_parse_bars` rather than
      relying on reversal.

**Success Criteria:**

- The order assumption is documented as verified, not assumed.

---

### BL-15: Data Quality Monitoring

**Problem:** There is no ongoing visibility into data quality: what percentage of bars
are being skipped, which symbols have the most API errors, or how fresh the stored
data is.

**Actions:**

- [ ] Add structured log entries for skip rates per chunk in `_parse_bars`.
- [ ] Add a `data_quality_report()` method to `Storage` that returns per-symbol
      statistics: row count, min/max timestamp, last write time.
- [ ] Expose the report via the dashboard endpoint or a CLI command.

**Success Criteria:**

- The operator can see at a glance which symbols have gaps or high skip rates.
