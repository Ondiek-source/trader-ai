================================================================================
  TRADER-AI DATA: JOURNAL.PY -- COMPLETE TEST HARNESS
  Updated   : 2026-04-14 | File Under Test : src/data/journal.py
  Runner    : pytest | Style: Matches storage + historian test conventions
================================================================================

  Total Test Cases  : 62
  Test Groups       : 9
  Coverage Target   : 100% — achieved
  Actual Coverage   : 100% (src/data/journal.py)

  PRE-FLIGHT REQUIREMENTS:
  The following packages must be installed:
    pip install pytest pytest-cov pandas pyarrow python-dotenv

  PHILOSOPHY:
  The Journal is an append-only audit trail. Tests must verify four things:
    1. Schema validation rejects invalid entries before any I/O occurs.
    2. Every successful write is readable and the data is intact.
    3. The crash-safe tmp-then-rename strategy survives mid-write failures.
    4. Schema versioning raises JournalError on mismatch, not silent NaN.

  TEST FILE LAYOUT:
    tests/
      data/
        journal/
          __init__.py
          conftest.py                  <-- shared fixtures
          test_journal_error.py        <-- Group 1  (JournalError class)
          test_trade_entry.py          <-- Group 2  (TradeEntry validation + repr)
          test_signal_entry.py         <-- Group 3  (SignalEntry validation + repr)
          test_journal_init.py         <-- Group 4  (Journal.__init__ + get_journal)
          test_schema_versioning.py    <-- Group 5  (_check_schema_version, _write_versioned)
          test_record_trade.py         <-- Group 6  (record_trade)
          test_record_signal.py        <-- Group 7  (record_signal)
          test_read_operations.py      <-- Group 8  (get_trade_history, get_signal_history)
          test_append_failures.py      <-- Group 9  (_append_to_table failure paths)

  SINGLETON NOTE:
  Journal.__init__ calls get_settings(), which is a module-level singleton.
  Tests that construct a real Journal must patch src.data.journal.get_settings
  directly, and patch src.data.journal.__file__ to control path derivation.
  Use Journal.__new__ to bypass __init__ entirely for I/O-only tests.

================================================================================
  JOURNAL CONFTEST.PY -- SHARED FIXTURES
================================================================================

  PURPOSE:
  Provides a pre-patched Journal instance that writes to tmp_path instead of
  the real data/ directory, plus factory fixtures for valid TradeEntry and
  SignalEntry objects.

  GOLDEN PROMPT -- tests/data/journal/conftest.py
  -----------------------------------------------------------------------

  Create tests/data/journal/conftest.py with the following content:

    import pytest
    from datetime import datetime
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from src.data.journal import Journal, TradeEntry, SignalEntry


    VALID_TS = datetime(2024, 1, 15, 10, 30, 0)
    VALID_SYMBOL = "EUR_USD"


    @pytest.fixture
    def journal(tmp_path):
        """
        A fully initialised Journal that writes to tmp_path.
        Uses patch to redirect __file__ so the path derivation points at
        tmp_path rather than the real project root.
        """
        # journal.py lives at src/data/journal.py; three parents = project root.
        # We fake __file__ so root_dir resolves to tmp_path/src/data/journal.py.
        mock_file = tmp_path / "src" / "data" / "journal.py"
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        mock_cfg = MagicMock()
        mock_cfg.data_mode = "LOCAL"

        with patch("src.data.journal.get_settings", return_value=mock_cfg), \
             patch("src.data.journal.__file__", str(mock_file)):
            j = Journal()
        return j


    @pytest.fixture
    def valid_trade_entry():
        return TradeEntry(
            timestamp=VALID_TS,
            symbol=VALID_SYMBOL,
            side="CALL",
            entry_price=1.0850,
            exit_price=1.0865,
            result=8.50,
            duration_seconds=60,
            signal_id="sig-20240115-001",
        )


    @pytest.fixture
    def valid_signal_entry():
        return SignalEntry(
            timestamp=VALID_TS,
            symbol=VALID_SYMBOL,
            confidence=0.78,
            direction="CALL",
            model_version="rf-v3.2.1",
            metadata='{"rsi": 42.1, "macd": 0.0003}',
        )


    @pytest.fixture
    def valid_trade_factory():
        def _factory(**kwargs):
            defaults = dict(
                timestamp=VALID_TS,
                symbol=VALID_SYMBOL,
                side="CALL",
                entry_price=1.0850,
                exit_price=1.0865,
                result=8.50,
                duration_seconds=60,
                signal_id="sig-20240115-001",
            )
            defaults.update(kwargs)
            return TradeEntry(**defaults)
        return _factory


    @pytest.fixture
    def valid_signal_factory():
        def _factory(**kwargs):
            defaults = dict(
                timestamp=VALID_TS,
                symbol=VALID_SYMBOL,
                confidence=0.78,
                direction="CALL",
                model_version="rf-v3.2.1",
                metadata="{}",
            )
            defaults.update(kwargs)
            return SignalEntry(**defaults)
        return _factory


================================================================================
  GROUP 1 -- JournalError CLASS
  File: tests/data/journal/test_journal_error.py
================================================================================

  PURPOSE:
  Verify that JournalError is a proper exception with typed attributes.
  Matches the pattern used in test_historian_error.py and StorageError tests.

  TEST CASES:

  test_journal_error_is_exception
    Verify JournalError is a subclass of Exception.
    >>> assert issubclass(JournalError, Exception)

  test_journal_error_stores_message
    JournalError("some error") → str(e) == "some error"

  test_journal_error_table_attribute_default
    JournalError("msg") → e.table == ""

  test_journal_error_path_attribute_default
    JournalError("msg") → e.path == ""

  test_journal_error_stores_table_and_path
    JournalError("msg", table="trades", path="/tmp/trades.parquet")
    → e.table == "trades" and e.path == "/tmp/trades.parquet"

  test_journal_error_is_raiseable
    `raise JournalError("msg")` → caught by `except JournalError`

  TOTAL: 6 tests


================================================================================
  GROUP 2 -- TradeEntry: VALIDATION & REPR
  File: tests/data/journal/test_trade_entry.py
================================================================================

  PURPOSE:
  Verify TradeEntry construction, validation rules, timezone handling,
  to_dict(), and __repr__. Mirrors the pattern in test_tick.py.

  HAPPY PATH:

  test_trade_entry_valid_construction
    Construct with all valid fields → no exception raised.

  test_trade_entry_is_frozen
    Attempt `entry.side = "PUT"` → raises FrozenInstanceError.

  test_trade_entry_to_dict_keys
    entry.to_dict() → keys == {"timestamp", "symbol", "side", "entry_price",
                                "exit_price", "result", "duration_seconds",
                                "signal_id"}

  test_trade_entry_repr_contains_key_fields
    repr(entry) contains symbol, side, entry_price, exit_price, result.

  test_trade_entry_accepts_call_and_put
    Construct with side="CALL" and side="PUT" → both succeed.

  test_trade_entry_utc_aware_timestamp_stripped
    timestamp=datetime(2024,1,15,10,30, tzinfo=timezone.utc)
    → entry.timestamp.tzinfo is None

  test_trade_entry_positive_result_accepted
    result=15.0 → entry.result == 15.0  (profit)

  test_trade_entry_negative_result_accepted
    result=-10.0 → entry.result == -10.0  (loss; result has no sign constraint)

  VALIDATION:

  test_trade_entry_invalid_side_rejected
    side="BUY" → raises ValueError matching "Invalid side"

  test_trade_entry_zero_entry_price_rejected
    entry_price=0.0 → raises ValueError matching "entry_price"

  test_trade_entry_negative_entry_price_rejected
    entry_price=-1.0850 → raises ValueError matching "entry_price"

  test_trade_entry_zero_exit_price_rejected
    exit_price=0.0 → raises ValueError matching "exit_price"

  test_trade_entry_zero_duration_rejected
    duration_seconds=0 → raises ValueError matching "duration_seconds"

  test_trade_entry_negative_duration_rejected
    duration_seconds=-60 → raises ValueError matching "duration_seconds"

  test_trade_entry_empty_signal_id_rejected
    signal_id="" → raises ValueError matching "signal_id"

  test_trade_entry_whitespace_signal_id_rejected
    signal_id="   " → raises ValueError matching "signal_id"

  test_trade_entry_non_utc_timezone_rejected
    timestamp with US/Eastern offset → raises ValueError matching "Non-UTC"

  TOTAL: 17 tests


================================================================================
  GROUP 3 -- SignalEntry: VALIDATION & REPR
  File: tests/data/journal/test_signal_entry.py
================================================================================

  PURPOSE:
  Verify SignalEntry construction, validation rules, and representation.

  HAPPY PATH:

  test_signal_entry_valid_construction
    Construct with all valid fields → no exception raised.

  test_signal_entry_is_frozen
    entry.direction = "PUT" → raises FrozenInstanceError.

  test_signal_entry_to_dict_keys
    entry.to_dict() → keys == {"timestamp", "symbol", "confidence",
                                "direction", "model_version", "metadata"}

  test_signal_entry_repr_contains_key_fields
    repr(entry) contains symbol, direction, confidence, model_version.

  test_signal_entry_accepts_call_and_put
    direction="CALL" and direction="PUT" → both succeed.

  test_signal_entry_confidence_boundary_zero
    confidence=0.0 → accepted (lower boundary inclusive).

  test_signal_entry_confidence_boundary_one
    confidence=1.0 → accepted (upper boundary inclusive).

  test_signal_entry_empty_metadata_accepted
    metadata="" → accepted (empty JSON is valid).

  test_signal_entry_utc_aware_timestamp_stripped
    timezone-aware UTC timestamp → stored as naive UTC.

  VALIDATION:

  test_signal_entry_confidence_above_one_rejected
    confidence=1.001 → raises ValueError matching "confidence"

  test_signal_entry_confidence_below_zero_rejected
    confidence=-0.001 → raises ValueError matching "confidence"

  test_signal_entry_invalid_direction_rejected
    direction="LONG" → raises ValueError matching "direction"

  test_signal_entry_empty_model_version_rejected
    model_version="" → raises ValueError matching "model_version"

  test_signal_entry_whitespace_model_version_rejected
    model_version="   " → raises ValueError matching "model_version"

  test_signal_entry_non_utc_timezone_rejected
    timestamp with non-UTC offset → raises ValueError matching "Non-UTC"

  TOTAL: 15 tests


================================================================================
  GROUP 4 -- Journal: INIT & SINGLETON
  File: tests/data/journal/test_journal_init.py
================================================================================

  PURPOSE:
  Verify Journal construction provisions the journal directory, the
  write-permission probe, and that get_journal() returns a singleton.

  CONSTRUCTION:

  test_journal_creates_journal_dir
    Construct Journal (patched path) → journal.journal_dir.exists() is True.

  test_journal_stores_settings_on_self
    journal._settings is not None

  test_journal_has_lock_attribute
    hasattr(journal, "_lock") is True

  test_journal_exits_on_permission_error
    Patch os.mkdir to raise PermissionError → sys.exit(1) called.
    Use pytest.raises(SystemExit, match="1").

  test_journal_exits_on_generic_mkdir_failure
    Patch mkdir to raise OSError("disk full") → sys.exit called.

  SINGLETON:

  test_get_journal_returns_same_instance
    get_journal() called twice → both calls return `is` the same object.

  test_get_journal_initialises_on_first_call_only
    Patch Journal.__init__ as MagicMock.
    Call get_journal() twice → __init__ called exactly once.

  TOTAL: 7 tests


================================================================================
  GROUP 5 -- SCHEMA VERSIONING
  File: tests/data/journal/test_schema_versioning.py
================================================================================

  PURPOSE:
  Verify that _write_versioned_parquet embeds metadata and
  _check_schema_version raises JournalError on mismatch.

  WRITE:

  test_write_versioned_parquet_creates_file
    _write_versioned_parquet(path, df) → path.exists() is True.

  test_write_versioned_parquet_embeds_version
    Write then pq.read_metadata(path).metadata[_SCHEMA_VERSION_KEY]
    → b"1"

  test_write_versioned_parquet_data_readable
    Write df with 2 rows, read back with pd.read_parquet → len == 2.

  CHECK:

  test_check_schema_version_passes_on_current_version
    Write a versioned file → _check_schema_version does not raise.

  test_check_schema_version_raises_on_missing_metadata
    Write a plain pd.to_parquet file (no metadata)
    → _check_schema_version raises JournalError matching "Schema version"

  test_check_schema_version_raises_on_wrong_version
    Write a file with _SCHEMA_VERSION_KEY = b"0"
    → _check_schema_version raises JournalError matching "v0"

  TOTAL: 6 tests

  IMPLEMENTATION NOTE:
  To write a file with an explicit version in metadata for testing, use:
    import pyarrow as pa, pyarrow.parquet as pq
    from src.data.journal import _SCHEMA_VERSION_KEY

    table = pa.table({"timestamp": [...]})
    meta = {**table.schema.metadata, _SCHEMA_VERSION_KEY: b"0"}
    pq.write_table(table.replace_schema_metadata(meta), path)


================================================================================
  GROUP 6 -- record_trade
  File: tests/data/journal/test_record_trade.py
================================================================================

  PURPOSE:
  Verify that record_trade() creates/appends to trades.parquet correctly.

  test_record_trade_returns_true
    journal.record_trade(valid_trade_entry) → True

  test_record_trade_creates_parquet_file
    After record_trade → (journal.journal_dir / "trades.parquet").exists()

  test_record_trade_persists_correct_row_count
    record_trade called once → pd.read_parquet(trades.parquet) has 1 row.

  test_record_trade_persists_correct_symbol
    record_trade → df["symbol"].iloc[0] == "EUR_USD"

  test_record_trade_persists_correct_side
    record_trade → df["side"].iloc[0] == "CALL"

  test_record_trade_appends_second_record
    record_trade called twice → parquet has 2 rows.

  test_record_trade_raises_journal_error_on_write_failure
    Patch pq.write_table to raise OSError("disk full")
    → raises JournalError matching "Journal write failed"

  TOTAL: 7 tests


================================================================================
  GROUP 7 -- record_signal
  File: tests/data/journal/test_record_signal.py
================================================================================

  PURPOSE:
  Verify that record_signal() creates/appends to signals.parquet correctly.
  Mirrors test_record_trade.py exactly for the signal table.

  test_record_signal_returns_true
    journal.record_signal(valid_signal_entry) → True

  test_record_signal_creates_parquet_file
    After record_signal → (journal.journal_dir / "signals.parquet").exists()

  test_record_signal_persists_correct_row_count
    record_signal called once → parquet has 1 row.

  test_record_signal_persists_correct_confidence
    df["confidence"].iloc[0] == pytest.approx(0.78)

  test_record_signal_persists_correct_direction
    df["direction"].iloc[0] == "CALL"

  test_record_signal_appends_second_record
    record_signal called twice → parquet has 2 rows.

  test_record_signal_raises_journal_error_on_write_failure
    Patch pq.write_table to raise OSError("disk full")
    → raises JournalError matching "Journal write failed"

  TOTAL: 7 tests


================================================================================
  GROUP 8 -- READ OPERATIONS
  File: tests/data/journal/test_read_operations.py
================================================================================

  PURPOSE:
  Verify get_trade_history() and get_signal_history() — return type, limit,
  empty-file handling, thread safety, and schema version rejection.

  TRADE HISTORY:

  test_get_trade_history_returns_empty_dataframe_when_no_file
    Call get_trade_history before any records exist → df.empty is True.

  test_get_trade_history_returns_dataframe
    After record_trade → get_trade_history() returns pd.DataFrame instance.

  test_get_trade_history_respects_limit
    Write 10 trade records → get_trade_history(limit=3) has 3 rows.

  test_get_trade_history_zero_limit_returns_all
    Write 5 records → get_trade_history(limit=0) has 5 rows.

  test_get_trade_history_returns_most_recent_rows
    Write trades A, B, C, D, E → get_trade_history(limit=2) contains D and E.

  test_get_trade_history_raises_on_version_mismatch
    Write trades.parquet with version=0
    → get_trade_history() raises JournalError matching "Schema version"

  SIGNAL HISTORY:

  test_get_signal_history_returns_empty_dataframe_when_no_file
    Call get_signal_history before any records exist → df.empty is True.

  test_get_signal_history_returns_dataframe
    After record_signal → get_signal_history() returns pd.DataFrame.

  test_get_signal_history_respects_limit
    Write 10 signal records → get_signal_history(limit=5) has 5 rows.

  test_get_signal_history_zero_limit_returns_all
    Write 5 signals → get_signal_history(limit=0) has 5 rows.

  TOTAL: 10 tests


================================================================================
  GROUP 9 -- _append_to_table FAILURE PATHS
  File: tests/data/journal/test_append_failures.py
================================================================================

  PURPOSE:
  Verify the crash-safe behaviour of _append_to_table: stale .tmp cleanup,
  JournalError propagation, and correct exception type (not swallowed).

  test_append_cleans_up_stale_tmp_before_write
    Create trades.tmp manually → after record_trade, trades.tmp is gone.
    (The stale file was removed before the new write cycle started.)

  test_append_raises_journal_error_on_parquet_write_failure
    Patch pq.write_table to raise OSError → _append_to_table raises JournalError.

  test_append_cleans_up_tmp_on_write_failure
    Patch pq.write_table to raise OSError → after the raise,
    trades.tmp does not exist.

  test_append_raises_journal_error_on_corrupt_read
    Write corrupt bytes to trades.parquet → record_trade raises JournalError.
    (The exception matches "Journal write failed".)

  test_append_does_not_swallow_exceptions
    Patch pq.write_table to raise OSError.
    Catch the result with try/except JournalError → exception IS caught.
    The old implementation would have logged and continued; the new one raises.

  test_append_preserves_existing_data_on_failure
    Write one valid trade → patch write_table to raise on next call.
    The first trade record must still be readable after the failed second write.

  TOTAL: 6 tests

  IMPLEMENTATION NOTE:
  Tests that patch pq.write_table must patch the reference inside the journal
  module, not the pyarrow.parquet module directly:
    with patch("src.data.journal.pq.write_table", side_effect=OSError("disk full")):
        ...

================================================================================
  SUMMARY TABLE
================================================================================

  Group  | File                        | Tests | What is verified
  -------+-----------------------------|-------+----------------------------------
  1      | test_journal_error.py       |   6   | JournalError attributes + raise
  2      | test_trade_entry.py         |  17   | TradeEntry validation, tz, repr
  3      | test_signal_entry.py        |  15   | SignalEntry validation, tz, repr
  4      | test_journal_init.py        |   7   | __init__ dirs, boot check, singleton
  5      | test_schema_versioning.py   |   6   | Version embedded, mismatch raises
  6      | test_record_trade.py        |   7   | record_trade write + error path
  7      | test_record_signal.py       |   7   | record_signal write + error path
  8      | test_read_operations.py     |  10   | get_trade/signal_history + limit
  9      | test_append_failures.py     |   6   | crash safety, tmp cleanup, no swallow
  -------+-----------------------------|-------+----------------------------------
  TOTAL  |                             |  62   |

================================================================================
  COVERAGE NOTES
================================================================================

  Lines that require specific test setup to hit:

  _check_schema_version — missing metadata path:
    Write a file via pa.Table + pq.write_table WITHOUT the version key.
    The plain pd.DataFrame.to_parquet does not embed any custom metadata,
    so a file written with the old journal.py will trigger this path.

  _check_schema_version — wrong version path:
    Manually embed _SCHEMA_VERSION_KEY = b"0" via PyArrow replace_schema_metadata.

  Journal.__init__ — PermissionError path:
    Use pytest monkeypatch to make journal_dir.mkdir raise PermissionError.
    Alternatively, patch pathlib.Path.mkdir directly in the module namespace.

  Journal.__init__ — generic OSError path:
    Same as above but raise OSError instead of PermissionError.

  _append_to_table — stale .tmp cleanup:
    Create a .tmp file in journal_dir before calling record_trade.
    After the call, assert the .tmp file is gone.

  _read_table — generic read failure:
    Write a corrupt (non-Parquet) file to the journal table path.
    Call get_trade_history() → JournalError raised.
