# TRADER-AI CORE: STORAGE.PY DESIGN DOCUMENT

VERSION: 1.2.0 | STATUS: FROZEN / PRODUCTION-READY

ROLE: DATA CUSTODIAN | VISION: IMMUTABLE HISTORY | ATOMIC I/O | ZERO-REDUNDANCY

---

## 1. ARCHITECTURAL PHILOSOPHY

The `storage.py` module is the only bridge between volatile RAM and the
persistent Disk/Cloud layer. It treats the file system as a "Write-Once,
Read-Many" vault. Its primary directive is to ensure that the ML Brain always
has a linearly consistent, gap-free, and uncorrupted dataset to train on.

It operates under two core principles:

**The Isolation Principle**: No module outside Storage is permitted to know
the physical paths, file formats, or write strategy of the data store. The
engine, pipeline, and ML trainer all operate on abstract method calls
(`save_tick_batch`, `get_bars`). If the storage backend changes from local
Parquet to Azure Delta Lake, only this file changes.

**The Dictator Pattern**: Consistent with `config.py` and `model.py`,
Storage either initialises fully or crashes immediately. There is no
partially-initialised state. If the data directory cannot be provisioned
at boot, the system exits before any data can be lost or corrupted.

---

## 2. THE DIAGNOSTIC LANGUAGE

Storage uses specific symbols to identify failures in the persistence layer,
consistent with the Core Diagnostic Registry:

| Symbol | Category  | Context / Usage                                              |
|--------|-----------|--------------------------------------------------------------|
| [!]    | FATAL     | File system permission errors, disk full, init failures.     |
| [%]    | LOGIC     | Parquet schema mismatches, corrupted headers, empty files.   |
| [+]    | SUCCESS   | Successful commit confirmations in INFO-level logs.          |

---

## 3. CORE RESPONSIBILITIES

### A. The Parquet Standard

All market data (Ticks and Bars) is stored in Apache Parquet format with
Snappy compression. This choice is deliberate:

* **Speed**: Snappy decompression is the fastest available, critical for
  training loops that load millions of rows per session.
* **Columnar Efficiency**: Parquet enables reading a single column (e.g.,
  `timestamp`) without deserialising the entire file. The `get_last_timestamp`
  and `get_tick_count` methods exploit this to achieve O(1)-equivalent
  performance regardless of file size.
* **Schema Enforcement**: Parquet stores column types in the file header.
  Type mismatches between new data and existing files are caught immediately
  at write time, not silently coerced.
* **Portability**: Parquet files are read identically by Pandas, PyArrow,
  PySpark, and Azure Synapse. The transition from local VPS to Azure Blob
  Storage requires no format conversion.

The engine is pinned to PyArrow via the module-level constant `_ENGINE = "pyarrow"`,
which is passed explicitly to every `pd.read_parquet()` and `DataFrame.to_parquet()`
call. This eliminates any ambiguity from pandas' `engine='auto'` fallback and
guarantees Snappy support in the Docker container regardless of which backends
are installed.

### B. The Atomic Upsert (`_atomic_upsert`)

All writes go through a single private method that encapsulates the full
read-deduplicate-write cycle:

```bash
1. Acquire threading.Lock
2. If file exists:
   a. Read existing Parquet into DataFrame
   b. Concatenate with new_data
   c. Drop duplicates on timestamp (last-in-wins)
   d. Sort by timestamp (chronological order)
   e. Write back to Parquet
3. If file does not exist:
   a. Sort new_data by timestamp
   b. Write directly to Parquet
4. Release threading.Lock
```

This is the "Duplicate Firewall". It ensures that backfilled data which
overlaps with live data is always resolved consistently. The "last-in-wins"
policy means the most recently written record for any given timestamp wins,
which correctly handles the case where a live tick and a backfilled tick
disagree on the same moment in time.

### C. State Recovery (The Gap Finder)

Before each session, the system must determine how much data is missing since
the last shutdown. Storage provides `get_last_timestamp(symbol)` which returns
`max(timestamp)` for any given symbol file via a columnar-only read.

This timestamp is handed to the Historian (backfill engine), which calculates
the gap in minutes and fetches exactly the missing data -- no more, no less.

```bash
Session Start:
  last_ts = storage.get_last_timestamp("EUR_USD")
  if last_ts is None:
      historian.backfill_full("EUR_USD")    # First run
  else:
      historian.backfill_gap("EUR_USD", since=last_ts)  # Resume
```

### D. Training Data Feed (`get_bars`)

The ML training loop calls `get_bars(symbol, timeframe, max_rows)` to receive
a clean, sorted DataFrame of OHLCV bars. The `max_rows` parameter maps
directly to `MAX_RF_ROWS` from config, enforcing the RAM cap at the data
access layer rather than in the model.

### E. Symbol Discovery (`list_symbols`)

`list_symbols()` scans the raw directory for all existing tick files and
returns their symbol names. This allows the pipeline orchestrator to discover
available data without hardcoding the symbol list.

### F. Azure Blob Sync (`sync_to_azure`, `pull_from_azure`)

Two public methods support the training/inference workflow when
`DATA_MODE=CLOUD`:

* **`sync_to_azure(local_path, blob_name=None) -> bool`**: Uploads a local
  file (Parquet dataset or model artifact) to Azure Blob Storage. Used after
  training to push the updated dataset and model weights to the cloud container.
  `blob_name` defaults to `local_path.name` if not supplied.
* **`pull_from_azure(blob_name, local_path) -> bool`**: Downloads a blob to a
  local path. Used at inference-container startup to pull the latest trained
  model before trading begins.

Both methods are **non-fatal**: a failure is logged at ERROR level and `False`
is returned. They are no-ops (`False`, warning logged) when
`_container_client is None` (i.e., `DATA_MODE=LOCAL`), making the same
codebase work offline without conditional guards in callers.

#### Azure Initialisation (`_init_azure_client`)

Called once from `__init__`. Applies the Dictator pattern:

* **LOCAL mode**: returns `None` immediately without touching the Azure SDK.
* **CLOUD mode**: creates a `BlobServiceClient`, gets a `ContainerClient`,
  and calls `get_container_properties()` to verify the container is reachable
  **at boot time**. If the call fails for any reason (wrong connection string,
  container missing, network unreachable) the process exits with code 1.
  This surfaces misconfiguration immediately rather than at the first upload,
  which could be hours into a training run.

---

## 4. THE STORAGEERROR EXCEPTION

Storage raises `StorageError` (a custom exception) rather than the generic
`Exception` for all I/O failures. This allows the engine to distinguish
between:

* `StorageError` — a persistence failure that may be retried or logged
* `ValueError` — a data shape problem from model.py
* `RuntimeError` — a logic failure from DataBuffer

The engine can catch each type with specific handling:

```python
try:
    storage.save_tick_batch(batch)
except StorageError as e:
    logger.error(f"Storage failure: {e}. Retrying next cycle.")
    # Do not crash -- attempt recovery
except ValueError as e:
    logger.critical(f"Data integrity failure: {e}. Halting.")
    sys.exit(1)
```

---

## 5. GUARDRAIL REGISTRY

### I. Thread Safety (The Concurrency Lock)

A single `threading.Lock` serialises ALL write operations across all symbols.
This is intentionally conservative -- using per-symbol locks would require a
lock registry, adding complexity for marginal throughput gain. The lock cost
(microseconds) is negligible compared to Parquet I/O (milliseconds).

**Protected Operations**: `_atomic_upsert` (called by `save_tick_batch` and
`save_bar`). Read operations (`get_last_timestamp`, `get_bars`) do not acquire
the lock -- Parquet reads are safe concurrently with completed writes.

### II. Path Dictatorship

* **Absolute Resolution**: All paths use `Path(__file__).resolve().parents[2]`
  to anchor at the project root. This ensures identical behaviour on Linux
  (VPS Docker container) and Windows (local development).
* **Single Source of Truth**: `_raw_path(symbol)` and `_processed_path(symbol,
  timeframe)` are the only places where file names are constructed. A naming
  convention change touches exactly two lines.
* **Auto-Provisioning**: `_provision_directories()` runs at construction.
  The system crashes with `StorageError` if it cannot acquire write access.

### III. Memory Awareness

* **Batch-Only Writes**: `save_tick_batch` requires a list of ticks, never a
  single tick. This enforces the DataBuffer contract and prevents tick-by-tick
  I/O which would destroy disk performance and SSD longevity on a VPS.
* **Row Cap on Reads**: `get_bars(max_rows=N)` applies the `MAX_RF_ROWS` cap
  at read time, returning only the most recent N bars. This prevents the
  Random Forest trainer from receiving more data than it can fit in RAM.

### IV. Timestamp Consistency

* **Sort on Write**: Every upsert sorts by timestamp before writing. This
  ensures Parquet files are always in chronological order, which is required
  by time-series feature engineering (rolling windows, lag features).
* **Deduplication Key**: The `_TS_COLUMN` constant (`"timestamp"`) is the
  deduplication key across all files. It is defined once at module level to
  prevent silent breakage from a typo.

---

## 6. DIRECTORY HIERARCHY

Storage enforces the following structure outside the `src/` package:

```bash
project_root/
└── data/
    ├── raw/
    │   ├── EUR_USD_ticks.parquet   <- Live + backfilled ticks
    │   ├── GBP_USD_ticks.parquet
    │   └── ...
    └── processed/
        ├── EUR_USD_M1.parquet      <- Aggregated OHLCV bars
        ├── EUR_USD_M5.parquet
        ├── GBP_USD_M1.parquet
        └── ...
```

The `data/` directory is excluded from git (see `.gitignore`). It is
provisioned at runtime on the VPS by the deploy scripts and populated
by the backfill engine on first run.

---

## 7. PUBLIC API REFERENCE

| Method                                          | Returns              | Description                                        |
|-------------------------------------------------|----------------------|----------------------------------------------------|
| `save_tick_batch(ticks)`                        | `bool`               | Persist a tick batch to raw storage.               |
| `save_bar(bar)`                                 | `bool`               | Persist a single bar to processed store.           |
| `save_bar_batch(bars)`                          | `bool`               | Persist a validated batch of bars (one per chunk). |
| `get_last_timestamp(symbol)`                    | `pd.Timestamp / None`| Last known tick time for gap detection.            |
| `get_bars(symbol, timeframe, max_rows)`         | `pd.DataFrame / None`| Load bar data for ML training.                     |
| `get_tick_count(symbol)`                        | `int`                | Total ticks stored for health checks.              |
| `list_symbols()`                                | `list[str]`          | All symbols with tick data on disk.                |
| `sync_to_azure(local_path, blob_name=None)`     | `bool`               | Upload a local file to Azure Blob Storage.         |
| `pull_from_azure(blob_name, local_path)`        | `bool`               | Download a blob from Azure to local disk.          |

---

## 8. FAIL-FAST SCENARIOS (CRITICAL CRASH)

| Trigger                          | Diagnostic Symbol | Exception / Action   | Context                             |
|----------------------------------|-------------------|----------------------|-------------------------------------|
| Directory creation fails         | [!] INIT FAILURE  | `sys.exit(1)`        | Permissions, disk full at boot      |
| Write-check canary fails         | [!] INIT FAILURE  | `sys.exit(1)`        | Directory exists but not writable   |
| Azure container unreachable      | [!] AZURE FAILURE | `sys.exit(1)`        | Wrong conn string, container missing|
| Parquet write fails              | [%] WRITE FAILURE | `StorageError`       | Schema mismatch, disk full          |
| Parquet read fails               | [%] READ FAILURE  | `None` (logged)      | Corrupted header, missing column    |
| Empty tick/bar batch passed      | None (debug log)  | `False` returned     | No-op, caller notified              |
| Mixed symbols in bar batch       | [!] BATCH FAILURE | `ValueError`         | Each batch must be single-symbol    |
| Mixed timeframes in bar batch    | [!] BATCH FAILURE | `ValueError`         | Each batch must be single-timeframe |
| Symbol file missing on query     | None (info log)   | `None` returned      | Returns None, triggers backfill     |
| Azure upload fails               | [!] SYNC FAILURE  | `False` returned     | Non-fatal, logged at ERROR          |
| Azure download fails             | [!] PULL FAILURE  | `False` returned     | Non-fatal, logged at ERROR          |

Note: Read failures and Azure sync failures return `None`/`False` rather than
raising, allowing the engine to recover. Only init failures and Parquet write
failures are fatal — a failed write represents actual unrecoverable data loss.

---

## 9. CLOUD WORKFLOW

Azure Blob Storage is fully integrated. When `DATA_MODE=CLOUD`:

```text
Training Workflow (local machine)
  1. historian.backfill_all()            ← fetch 2yr of bars from Twelve Data
  2. storage.sync_to_azure(parquet_path) ← push dataset to Azure Blob
  3. trainer.run()                        ← train model on local data
  4. storage.sync_to_azure(model_path)   ← push model artifact to Azure Blob

Inference Workflow (Azure container startup)
  1. storage.pull_from_azure("model.pkl", local_model_path)  ← pull latest model
  2. engine.start()                                           ← begin trading
```

The local Parquet files and model artifacts are never removed — Azure Blob is
a mirror, not a replacement. The public API of Storage is identical in both
modes; only `_init_azure_client` distinguishes between LOCAL (returns `None`)
and CLOUD (connects and verifies the container at boot).

---

## "IF IT IS NOT IN STORAGE, IT NEVER HAPPENED."
