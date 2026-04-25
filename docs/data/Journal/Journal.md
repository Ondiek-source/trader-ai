# TRADER-AI DATA: JOURNAL.PY DESIGN DOCUMENT

VERSION: 1.0.0 | STATUS: FROZEN / PRODUCTION-READY

ROLE: THE AUDITOR | VISION: COMPLETE DECISION TRAIL | CRASH-SAFE | APPEND-ONLY

---

## 1. ARCHITECTURAL PHILOSOPHY

The `journal.py` module acts as the "Auditor" of the system. Where
`storage.py` owns the market-price vault (what the market did), the Journal
owns the decision ledger (what the bot decided, and why). This separation
is fundamental to the Isolation Principle: market data and trading decisions
are two different concerns and must never share a file.

The Journal operates under three core principles:

**The Append-Only Principle**: The Journal never updates or deletes existing
rows. Every trade execution and model signal is committed as a new row in its
respective table. The ledger grows monotonically. This makes it suitable as an
audit trail — any question about past decisions has a permanent, tamper-evident
answer.

**The Crash-Safety Principle**: Every write uses a tmp-then-rename strategy.
The new data is written to a `.tmp` shadow file first. Only after the write
completes successfully is the `.tmp` atomically renamed to the live `.parquet`.
A crash between these two steps leaves the live file intact; only the
`.tmp` file is orphaned (and cleaned up on the next write cycle).

**The Dictator Pattern** (consistent with `config.py`, `storage.py`,
`historian.py`): The Journal either constructs fully or exits immediately.
A write-permission check runs during `__init__` so that a misconfigured
filesystem is discovered at boot, not silently at the first trade — when
it is too late to recover the record.

---

## 2. THE DIAGNOSTIC LANGUAGE

Consistent with the Core Diagnostic Registry, the Journal uses specific
symbols to identify the nature of each log event:

| Symbol | Category | Context / Usage                                                  |
|--------|----------|------------------------------------------------------------------|
| `[+]`  | SUCCESS  | Record committed to ledger.                                      |
| `[!]`  | FATAL    | Schema validation failure, unknown direction/side.               |
| `[%]`  | LOGIC    | Confidence out of range, invalid price, write failure.           |
| `[^]`  | WARNING  | Stale `.tmp` file found and removed before write.                |

---

## 3. CORE COMPONENTS

### A. The Custom Exception (`JournalError`)

A typed exception distinct from `StorageError` (market-price vault) and
`HistorianError` (backfill engine). Carries `table` and `path` attributes
so the pipeline can identify which journal file failed without parsing the
error message string.

Raised in two situations:

1. **Schema version mismatch** — an existing file's `_SCHEMA_VERSION_KEY`
   metadata does not match the current `_SCHEMA_VERSION` constant.
2. **Write or read failure** — any `OSError`, `ArrowIOError`, or similar
   filesystem failure during `_append_to_table` or `_read_table`.

`ValueError` from `TradeEntry.__post_init__` or `SignalEntry.__post_init__`
is NOT wrapped in `JournalError` — it propagates unchanged to the caller
because it represents a code defect in the layer that constructed the entry,
not a Journal I/O failure.

### B. The Record Schemas (`TradeEntry`, `SignalEntry`)

Both are frozen dataclasses. The `frozen=True` argument prevents mutation
after construction — any correction to a committed record requires creating a
new entry with distinct field values, preserving the append-only guarantee.

**`TradeEntry`** captures finalised binary-options trade results:

| Field              | Type     | Constraint                        |
|--------------------|----------|-----------------------------------|
| `timestamp`        | datetime | Naive UTC (non-UTC aware rejected) |
| `symbol`           | str      | Any currency pair string           |
| `side`             | str      | Must be `"CALL"` or `"PUT"`        |
| `entry_price`      | float    | Must be > 0                        |
| `exit_price`       | float    | Must be > 0                        |
| `result`           | float    | Any value (positive = profit)      |
| `duration_seconds` | int      | Must be > 0                        |
| `signal_id`        | str      | Must be non-empty                  |

**`SignalEntry`** captures model-generated predictions:

| Field           | Type     | Constraint                          |
|-----------------|----------|-------------------------------------|
| `timestamp`     | datetime | Naive UTC (non-UTC aware rejected)  |
| `symbol`        | str      | Any currency pair string            |
| `confidence`    | float    | Must be in `[0.0, 1.0]`            |
| `direction`     | str      | Must be `"CALL"` or `"PUT"`         |
| `model_version` | str      | Must be non-empty                   |
| `metadata`      | str      | Any string (JSON payload, may be empty) |

Both schemas implement `to_dict()` and `__repr__` consistent with `Tick`
and `Bar` in `model.py`.

### C. The Atomic Appender (`_append_to_table`)

The private write primitive behind both `record_trade()` and
`record_signal()`. Called inside the threading lock.

```text
WRITE CYCLE (crash-safe):
  1. _check_schema_version(file_path)        ← raises JournalError on mismatch
  2. pd.read_parquet(file_path)              ← load existing rows (if any)
  3. pd.concat([existing_df, new_df])        ← append new row(s)
  4. _write_versioned_parquet(tmp_path, df)  ← write with metadata to .tmp
  5. tmp_path.replace(file_path)             ← atomic rename to live file
```

A crash at any point before step 5 leaves the live `.parquet` unchanged.
A crash at step 5 is handled by the OS atomic rename guarantee on all
POSIX-compliant filesystems and NTFS.

A stale `.tmp` file (from a previous crashed write) is detected and removed
at the start of every write cycle before step 1.

### D. The Schema Version Guard (`_check_schema_version`, `_write_versioned_parquet`)

Mirrors the implementation in `storage.py._atomic_upsert`. Every write via
`_write_versioned_parquet` embeds a key-value pair in the Parquet file
metadata:

```
key   : b"trader_ai_journal_schema_version"
value : b"1"   (encoded _SCHEMA_VERSION as UTF-8 bytes)
```

Every read via `_check_schema_version` reads only the Parquet footer
(fast — no column data is loaded) and compares the stored version to
`_SCHEMA_VERSION`. A mismatch raises `JournalError` immediately.

**Migration path**: Increment `_SCHEMA_VERSION`, delete the old journal
files, and re-run the system. The Journal's append-only design means there
is no partial migration — either all rows conform to the current schema, or
none do.

### E. The Table Readers (`get_trade_history`, `get_signal_history`)

Both methods delegate to the private `_read_table()` helper, which acquires
the lock before any filesystem operation. This prevents a concurrent write
from being observed mid-rename by a reader.

The `limit` parameter returns `df.tail(limit)` — the most recent rows by
file order (which is insertion order, since the Journal only appends).
Passing `limit=0` returns the full file.

Both methods return an empty `pd.DataFrame()` (not `None`) when no file
exists, so callers can always do `if df.empty:` without a `None` guard.

### F. The Singleton (`get_journal`)

`get_journal()` is a module-level factory that returns the same `Journal`
instance on every call — mirroring `get_settings()` and `get_historian()`.

**Why a singleton matters for thread safety**: The threading lock is an
instance attribute. Two independent `Journal()` instances would each have
their own lock, making concurrent writes to the same `.parquet` file
entirely unprotected — one write's `.tmp` could be renamed over the other's
mid-cycle. The singleton guarantees one lock governs all writers.

---

## 4. GUARDRAIL REGISTRY

### I. Dataclass Validation

- `TradeEntry.__post_init__` and `SignalEntry.__post_init__` run at
  construction time, before any I/O.
- All violations are collected into a list and reported in a single
  `%`-bordered CRITICAL log block, matching the pattern in `model.py`.
- Invalid entries never reach `_append_to_table`.

### II. Non-UTC Timezone Rejection (BL-03)

- Both `TradeEntry` and `SignalEntry` check `timestamp.utcoffset()` before
  stripping `tzinfo`. A non-zero offset raises `ValueError` immediately.
- A UTC-aware `datetime` is accepted and stored as naive UTC (tzinfo stripped).
- A naive `datetime` is accepted as-is (assumed to be UTC by caller contract).

### III. Schema Versioning (BL-02)

- `_SCHEMA_VERSION = 1` is the current schema generation.
- Every write embeds the version via `_write_versioned_parquet`.
- Every read checks the version via `_check_schema_version`.
- A mismatch raises `JournalError` with a clear instruction: delete the file
  and re-run.

### IV. Write Failure Propagation

- The previous implementation silently swallowed write failures (logged at
  `ERROR` but did not raise).
- The current implementation raises `JournalError` on any write failure.
- The caller (engine, pipeline) is responsible for deciding whether to
  retry, log and continue, or halt. Silent data loss is never acceptable
  for audit records.

### V. Write-Permission Boot Check

- `Journal.__init__` creates a `.write_probe` file and immediately deletes
  it after construction succeeds.
- A `PermissionError` at boot is surfaced immediately with a structured
  `!`-bordered CRITICAL block and `sys.exit(1)`.
- This prevents the common failure mode where a read-only mount produces
  a fully-constructed `Journal` object that silently discards every write.

---

## 5. DATA FLOW

```text
JOURNAL DATA FLOW

  Engine / Signal Layer
          │
          │  TradeEntry(timestamp, symbol, side, entry_price,
          │             exit_price, result, duration_seconds, signal_id)
          │
          ▼
  TradeEntry.__post_init__()   ←── validates all fields, rejects non-UTC tz
          │
          │  raises ValueError immediately on violation
          │
          ▼
  journal.record_trade(entry)
          │
          ▼
  _append_to_table("trades", [entry.to_dict()])   ← lock acquired
          │
          ├── _check_schema_version(trades.parquet)
          │
          ├── pd.read_parquet(trades.parquet)          ← existing rows
          │
          ├── pd.concat([existing, new])
          │
          ├── _write_versioned_parquet(trades.tmp)     ← PA table with metadata
          │
          └── trades.tmp.replace(trades.parquet)       ← atomic rename
                    │
                    ▼
          data/processed/journal/trades.parquet  (updated)


  ─── PARALLEL: SIGNAL FLOW ───────────────────────────────────────────────

  Signal Layer
          │
          │  SignalEntry(timestamp, symbol, confidence, direction,
          │              model_version, metadata)
          │
          ▼
  SignalEntry.__post_init__()
          │
          ▼
  journal.record_signal(entry)
          │
          ▼
  _append_to_table("signals", [entry.to_dict()])
          │
          └── ... same crash-safe cycle ...
                    │
                    ▼
          data/processed/journal/signals.parquet  (updated)
```

---

## 6. MODULE CONSTANTS

| Constant              | Value                              | Rationale                                                        |
|-----------------------|------------------------------------|------------------------------------------------------------------|
| `_ENGINE`             | `"pyarrow"`                        | Pinned engine; eliminates pandas auto-detection ambiguity.       |
| `_COMPRESSION`        | `"snappy"`                         | Fastest decompression; consistent with `storage.py`.            |
| `_SCHEMA_VERSION`     | `1`                                | Current schema generation. Increment on any schema change.       |
| `_SCHEMA_VERSION_KEY` | `b"trader_ai_journal_schema_version"` | Distinct from storage key to allow independent versioning.    |
| `_VALID_SIDES`        | `frozenset({"CALL", "PUT"})`       | Binary-options trade directions accepted by Quotex.             |
| `_VALID_DIRECTIONS`   | `frozenset({"CALL", "PUT"})`       | Signal directions, mirroring `_VALID_SIDES`.                    |

---

## 7. FILE LAYOUT

```text
data/
  processed/
    journal/
      trades.parquet    ← TradeEntry rows, append-only
      signals.parquet   ← SignalEntry rows, append-only
      trades.tmp        ← transient only; present only during active write
      signals.tmp       ← transient only; present only during active write
```

The `.tmp` files are internal implementation details. They should never
appear in backups or be committed to version control. If one is present
after a crash, it is cleaned up automatically on the next write.

---

## 8. COMPARISON WITH STORAGE.PY

| Concern              | `storage.py` (Vault)                  | `journal.py` (Auditor)                    |
|----------------------|---------------------------------------|-------------------------------------------|
| **Data stored**      | Market prices (Tick, Bar)             | Decisions (TradeEntry, SignalEntry)        |
| **Write strategy**   | Read-dedup-write (in-place upsert)    | Read-append-write (tmp-then-rename)        |
| **Deduplication**    | Yes — timestamp last-in-wins          | No — every row is unique by design         |
| **Read lock**        | Yes                                   | Yes                                        |
| **Write lock**       | Yes                                   | Yes                                        |
| **Schema version**   | Yes — `_SCHEMA_VERSION = 1`           | Yes — `_SCHEMA_VERSION = 1`                |
| **Failure handling** | Raises `StorageError`                 | Raises `JournalError`                      |
| **Path derivation**  | `Path(__file__).resolve()`            | `Path(__file__).resolve()`                 |
| **Boot check**       | Directory creation + write probe      | Directory creation + write probe           |
| **Singleton**        | No (Storage is stateless per lock)    | Yes (`get_journal()` — shared lock)        |
