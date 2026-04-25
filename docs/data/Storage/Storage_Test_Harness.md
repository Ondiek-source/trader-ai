# TRADER-AI CORE: STORAGE.PY -- COMPLETE TEST HARNESS

---

Updated : 2026-04-14 | File Under Test : src/data/storage.py
Runner : pytest | Style: Matches config + model test harness conventions

---

Total Test Cases : 78
Test Groups : 10
Coverage Target : 100% — achieved
Actual Coverage : 100% (src/data/storage.py)

PRE-FLIGHT REQUIREMENTS:
The following packages must be installed:
pip install pytest pytest-cov pandas pyarrow python-dotenv azure-storage-blob aioresponses

PyArrow is pinned as the Parquet engine via the module-level constant
\_ENGINE = "pyarrow". Without it, all Parquet read/write calls fail with:
ImportError: Unable to find a usable engine for 'parquet'.

PHILOSOPHY:
Storage tests must be hermetic -- no test may touch the real data/
directory. Every test that exercises file I/O must use pytest's tmp_path
fixture to get a unique temporary directory that is automatically cleaned
up after each test. This guarantees tests are: 1. Isolated -- no test pollutes another's file state 2. Repeatable -- running the suite 100 times gives the same result 3. Safe -- the real data/ vault is never touched

Azure tests mock the BlobServiceClient and ContainerClient via
unittest.mock.MagicMock. No real Azure credentials are required to run
the suite. The \_cloud_storage() helper in test_azure.py constructs a
Storage instance via **new** with a mocked CLOUD config, bypassing
\_init_azure_client() so each test controls \_container_client directly.

SINGLETON NOTE:
Storage.**init** calls get_settings(), which is a module-level singleton.
Tests that exercise the real **init** (test_storage_real_initialization)
must patch src.data.storage.get_settings directly to avoid inheriting a
stale cached config from earlier tests (e.g., DATA_MODE=CLOUD left by a
config validation test).

TEST FILE LAYOUT:
tests/
conftest.py <-- root fixtures (minimal_valid_env)
data/
storage/
**init**.py
conftest.py <-- storage-specific fixtures
test_storage_init.py <-- Group 1 (construction + dirs)
test_save_tick_batch.py <-- Group 2 (tick write operations)
test_save_bar.py <-- Group 3 (bar write operations)
test_save_bar_batch.py <-- Group 4 (bar batch write + guards)
test_read_operations.py <-- Group 5 (queries + bar reads)
test_diagnostics.py <-- Group 6 (tick count + symbols)
test_atomic_upsert.py <-- Group 7 (dedup + sort + schema)
test_concurrency.py <-- Group 8 (thread safety)
test_azure.py <-- Group 9 (\_init_azure_client)
Group 10 (sync_to_azure + pull_from_azure)

---

## STORAGE CONFTEST.PY -- SHARED FIXTURES

PURPOSE:
Provides pre-built Tick and Bar instances, and a patched Storage instance
that writes to tmp_path instead of the real data/ directory. The storage
fixture is the most important fixture in this file -- it must be used in
every test that performs I/O.

GOLDEN PROMPT -- tests/core/storage/conftest.py

---

Create tests/core/storage/conftest.py with the following content:

    import pytest
    from datetime import datetime
    from unittest.mock import patch
    from src.ml_engine.model import Tick, Bar, Timeframe


    VALID_TS = datetime(2024, 1, 15, 10, 30, 0)
    VALID_TS_2 = datetime(2024, 1, 15, 10, 31, 0)
    VALID_TS_3 = datetime(2024, 1, 15, 10, 32, 0)


    @pytest.fixture
    def valid_tick():
        return Tick(
            timestamp=VALID_TS,
            symbol="EUR_USD",
            bid=1.08500,
            ask=1.08520,
            source="TWELVE",
        )


    @pytest.fixture
    def valid_tick_2():
        return Tick(
            timestamp=VALID_TS_2,
            symbol="EUR_USD",
            bid=1.08510,
            ask=1.08530,
            source="TWELVE",
        )


    @pytest.fixture
    def valid_tick_gbp():
        """A tick for a different symbol -- used in mixed-symbol tests."""
        return Tick(
            timestamp=VALID_TS,
            symbol="GBP_USD",
            bid=1.26500,
            ask=1.26520,
            source="QUOTEX",
        )


    @pytest.fixture
    def valid_bar():
        return Bar(
            timestamp=VALID_TS,
            symbol="EUR_USD",
            open=1.0850,
            high=1.0865,
            low=1.0848,
            close=1.0860,
            volume=342,
        )


    @pytest.fixture
    def valid_bar_2():
        return Bar(
            timestamp=VALID_TS_2,
            symbol="EUR_USD",
            open=1.0860,
            high=1.0875,
            low=1.0858,
            close=1.0870,
            volume=280,
        )


    @pytest.fixture
    def incomplete_bar():
        return Bar(
            timestamp=VALID_TS,
            symbol="EUR_USD",
            open=1.0850,
            high=1.0865,
            low=1.0848,
            close=1.0860,
            volume=342,
            is_complete=False,
        )


    @pytest.fixture
    def storage(tmp_path, minimal_valid_env):
        """
        A Storage instance wired to tmp_path instead of the real data/ dir.

        Patches Path(__file__).resolve().parents[2] so that root_dir points
        to tmp_path, keeping all test I/O hermetic.
        Uses minimal_valid_env from root conftest.py for config.
        """
        from unittest.mock import patch, MagicMock
        import os

        with patch.dict(os.environ, minimal_valid_env, clear=True):
            with patch("src.core.storage.Path") as mock_path_class:
                # Wire the path resolution to tmp_path
                mock_file = MagicMock()
                mock_file.resolve.return_value.parents.__getitem__.return_value = tmp_path
                mock_path_class.return_value = mock_file

                # Import fresh to pick up the patched Path
                from src.core.storage import Storage
                store = Storage.__new__(Storage)
                store._settings = None
                store._lock = __import__("threading").Lock()
                store.root_dir = tmp_path / "data"
                store.raw_dir = tmp_path / "data" / "raw"
                store.processed_dir = tmp_path / "data" / "processed"
                store._provision_directories()
                return store

---

NOTE ON THE STORAGE FIXTURE:
The fixture above bypasses **init** and directly wires the path attributes.
This is the cleanest approach because Storage.**init** calls get_settings()
and Path resolution simultaneously. An alternative approach using monkeypatch
on the class attributes is shown in Group 1 tests where **init** itself is
under test.

---

GROUP 1 -- Storage: Initialisation & Directory Provisioning (7 tests)
File: tests/core/storage/test_storage_init.py

---

---

## TEST 1.1 -- Storage provisions raw and processed directories on init

WHAT IT CHECKS:
When Storage is constructed, it must create data/raw/ and
data/processed/ if they do not exist. This is the auto-provisioning
guarantee from the design doc.

GOLDEN PROMPT

---

    def test_storage_provisions_directories_on_init(tmp_path, minimal_valid_env):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, minimal_valid_env, clear=True):
            with patch("src.core.storage.Path") as MockPath:
                instance = MockPath.return_value
                instance.resolve.return_value.parents.__getitem__.return_value = tmp_path
                from src.core.storage import Storage
                s = Storage.__new__(Storage)
                s._lock = __import__("threading").Lock()
                s._settings = None
                s.root_dir = tmp_path / "data"
                s.raw_dir = tmp_path / "data" / "raw"
                s.processed_dir = tmp_path / "data" / "processed"
                s._provision_directories()
        assert (tmp_path / "data" / "raw").exists()
        assert (tmp_path / "data" / "processed").exists()

---

---

## TEST 1.2 -- Storage is idempotent on repeated init calls

WHAT IT CHECKS:
Calling \_provision_directories() twice must not raise. The
exist_ok=True parameter must be in effect.

GOLDEN PROMPT

---

    def test_storage_provision_is_idempotent(storage):
        # Calling provision again on already-existing dirs must not raise
        storage._provision_directories()
        assert storage.raw_dir.exists()
        assert storage.processed_dir.exists()

---

---

## TEST 1.3 -- Storage raises StorageError when directory cannot be created

WHAT IT CHECKS:
If the OS denies directory creation (e.g., permissions), Storage
must raise StorageError immediately -- not silently continue.

GOLDEN PROMPT

---

    def test_storage_raises_on_directory_creation_failure(storage):
        from unittest.mock import patch
        from src.core.storage import StorageError
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(StorageError, match="Cannot create directory"):
                storage._provision_directories()

---

---

## TEST 1.4 -- \_raw_path returns correct filename for a symbol

GOLDEN PROMPT

---

    def test_raw_path_naming_convention(storage):
        path = storage._raw_path("EUR_USD")
        assert path.name == "EUR_USD_ticks.parquet"
        assert path.parent == storage.raw_dir

---

---

## TEST 1.5 -- \_processed_path returns correct filename for symbol + timeframe

GOLDEN PROMPT

---

    def test_processed_path_naming_convention(storage):
        path = storage._processed_path("EUR_USD", "M1")
        assert path.name == "EUR_USD_M1.parquet"
        assert path.parent == storage.processed_dir

---

---

## TEST 1.6 -- \_processed_path handles all timeframe values

GOLDEN PROMPT

---

    @pytest.mark.parametrize("tf", ["M1", "M5", "M15"])
    def test_processed_path_all_timeframes(storage, tf):
        path = storage._processed_path("GBP_USD", tf)
        assert path.name == f"GBP_USD_{tf}.parquet"

---

---

## TEST 1.7 -- StorageError carries symbol and path attributes

WHAT IT CHECKS:
The custom exception must expose .symbol and .path so the engine
can log structured context without parsing the error message string.

GOLDEN PROMPT

---

    def test_storage_error_has_attributes():
        from src.core.storage import StorageError
        err = StorageError("test failure", symbol="EUR_USD", path="/data/raw/x.parquet")
        assert err.symbol == "EUR_USD"
        assert err.path == "/data/raw/x.parquet"
        assert "test failure" in str(err)

---

---

GROUP 2 -- save_tick_batch: Write Operations (9 tests)
File: tests/core/storage/test_save_tick_batch.py

---

---

## TEST 2.1 -- save_tick_batch returns True on successful write

GOLDEN PROMPT

---

    def test_save_tick_batch_returns_true(storage, valid_tick, valid_tick_2):
        result = storage.save_tick_batch([valid_tick, valid_tick_2])
        assert result is True

---

---

## TEST 2.2 -- save_tick_batch creates Parquet file on disk

GOLDEN PROMPT

---

    def test_save_tick_batch_creates_file(storage, valid_tick):
        storage.save_tick_batch([valid_tick])
        expected_path = storage.raw_dir / "EUR_USD_ticks.parquet"
        assert expected_path.exists()

---

---

## TEST 2.3 -- save_tick_batch returns False on empty list

WHAT IT CHECKS:
An empty batch is a no-op and must return False without creating
any file or raising any exception.

GOLDEN PROMPT

---

    def test_save_tick_batch_empty_returns_false(storage):
        result = storage.save_tick_batch([])
        assert result is False
        assert not (storage.raw_dir / "EUR_USD_ticks.parquet").exists()

---

---

## TEST 2.4 -- save_tick_batch persists correct tick count

GOLDEN PROMPT

---

    def test_save_tick_batch_persists_correct_count(storage, valid_tick, valid_tick_2):
        import pandas as pd
        storage.save_tick_batch([valid_tick, valid_tick_2])
        df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
        assert len(df) == 2

---

---

## TEST 2.5 -- save_tick_batch deduplicates on repeated write

WHAT IT CHECKS:
Writing the same tick twice must not result in duplicate rows.
The last-in-wins deduplication must fire on the second write.

GOLDEN PROMPT

---

    def test_save_tick_batch_deduplicates(storage, valid_tick):
        import pandas as pd
        storage.save_tick_batch([valid_tick])
        storage.save_tick_batch([valid_tick])  # same tick again
        df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
        assert len(df) == 1

---

---

## TEST 2.6 -- save_tick_batch appends new ticks to existing file

GOLDEN PROMPT

---

    def test_save_tick_batch_appends_new_ticks(storage, valid_tick, valid_tick_2):
        import pandas as pd
        storage.save_tick_batch([valid_tick])
        storage.save_tick_batch([valid_tick_2])
        df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
        assert len(df) == 2

---

---

## TEST 2.7 -- save_tick_batch output file is sorted by timestamp

WHAT IT CHECKS:
After writing, the Parquet file must be in chronological order.
Writing an out-of-order batch (later tick first, then earlier tick)
must still produce a sorted file.

GOLDEN PROMPT

---

    def test_save_tick_batch_output_is_sorted(storage, valid_tick, valid_tick_2):
        import pandas as pd
        # Write in reverse order
        storage.save_tick_batch([valid_tick_2, valid_tick])
        df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

---

---

## TEST 2.8 -- save_tick_batch raises ValueError on mixed-symbol batch

WHAT IT CHECKS:
A batch containing ticks from different symbols (EUR_USD + GBP_USD)
must be rejected before any file is written. The mixed-symbol guard.

GOLDEN PROMPT

---

    def test_save_tick_batch_rejects_mixed_symbols(storage, valid_tick, valid_tick_gbp):
        with pytest.raises(ValueError, match="mixed symbols"):
            storage.save_tick_batch([valid_tick, valid_tick_gbp])

---

---

## TEST 2.9 -- save_tick_batch raises StorageError on write failure

WHAT IT CHECKS:
If the underlying Parquet write fails (disk full, permissions),
save_tick_batch must raise StorageError, not return False silently.

GOLDEN PROMPT

---

    def test_save_tick_batch_raises_on_write_failure(storage, valid_tick):
        from unittest.mock import patch
        from src.core.storage import StorageError
        with patch.object(storage, "_atomic_upsert", side_effect=StorageError("disk full")):
            with pytest.raises(StorageError):
                storage.save_tick_batch([valid_tick])

---

---

GROUP 3 -- save_bar: Write Operations (7 tests)
File: tests/core/storage/test_save_bar.py

---

---

## TEST 3.1 -- save_bar returns True on successful write

GOLDEN PROMPT

---

    def test_save_bar_returns_true(storage, valid_bar):
        result = storage.save_bar(valid_bar)
        assert result is True

---

---

## TEST 3.2 -- save_bar creates Parquet file with correct name

WHAT IT CHECKS:
The file must be named {symbol}\_{timeframe}.parquet using
timeframe.value (e.g., "M1"), not the Enum repr ("Timeframe.M1").

GOLDEN PROMPT

---

    def test_save_bar_creates_correctly_named_file(storage, valid_bar):
        storage.save_bar(valid_bar)
        expected_path = storage.processed_dir / "EUR_USD_M1.parquet"
        assert expected_path.exists()

---

---

## TEST 3.3 -- save_bar deduplicates on repeated write

GOLDEN PROMPT

---

    def test_save_bar_deduplicates(storage, valid_bar):
        import pandas as pd
        storage.save_bar(valid_bar)
        storage.save_bar(valid_bar)
        df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
        assert len(df) == 1

---

---

## TEST 3.4 -- save_bar appends new bars to existing file

GOLDEN PROMPT

---

    def test_save_bar_appends(storage, valid_bar, valid_bar_2):
        import pandas as pd
        storage.save_bar(valid_bar)
        storage.save_bar(valid_bar_2)
        df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
        assert len(df) == 2

---

---

## TEST 3.5 -- save_bar output file is sorted by timestamp

GOLDEN PROMPT

---

    def test_save_bar_output_is_sorted(storage, valid_bar, valid_bar_2):
        import pandas as pd
        storage.save_bar(valid_bar_2)  # write later bar first
        storage.save_bar(valid_bar)   # write earlier bar second
        df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

---

---

## TEST 3.6 -- save_bar returns False and does not write for incomplete bar

WHAT IT CHECKS:
The is_complete guard. An incomplete bar must be rejected by Storage
itself -- the caller should not have to remember to filter.

GOLDEN PROMPT

---

    def test_save_bar_rejects_incomplete_bar(storage, incomplete_bar):
        result = storage.save_bar(incomplete_bar)
        assert result is False
        assert not (storage.processed_dir / "EUR_USD_M1.parquet").exists()

---

---

## TEST 3.7 -- save_bar uses timeframe value in path not enum repr

WHAT IT CHECKS:
Regression guard for the bar.timeframe.value fix. Ensures "M1"
appears in the filename, not "Timeframe.M1".

GOLDEN PROMPT

---

    def test_save_bar_uses_timeframe_value_not_repr(storage, valid_bar):
        storage.save_bar(valid_bar)
        files = list(storage.processed_dir.glob("*.parquet"))
        assert len(files) == 1
        assert "Timeframe" not in files[0].name
        assert "M1" in files[0].name

---

---

GROUP 4 -- get_last_timestamp: Query Operations (6 tests)
File: tests/core/storage/test_read_operations.py

---

---

## TEST 4.1 -- get_last_timestamp returns None when no file exists

WHAT IT CHECKS:
First run scenario. No file = return None so the engine knows to
trigger a full backfill.

GOLDEN PROMPT

---

    def test_get_last_timestamp_returns_none_when_no_file(storage):
        result = storage.get_last_timestamp("EUR_USD")
        assert result is None

---

---

## TEST 4.2 -- get_last_timestamp returns correct max timestamp

GOLDEN PROMPT

---

    def test_get_last_timestamp_returns_max(storage, valid_tick, valid_tick_2):
        import pandas as pd
        storage.save_tick_batch([valid_tick, valid_tick_2])
        result = storage.get_last_timestamp("EUR_USD")
        assert result == pd.Timestamp("2024-01-15 10:31:00")

---

---

## TEST 4.3 -- get_last_timestamp raises StorageError on corrupted file

WHAT IT CHECKS:
The corruption vs first-run distinction. A corrupted file must
raise StorageError so the engine does not silently re-backfill
over recoverable data.

GOLDEN PROMPT

---

    def test_get_last_timestamp_raises_on_corrupted_file(storage):
        from src.core.storage import StorageError
        file_path = storage.raw_dir / "EUR_USD_ticks.parquet"
        file_path.write_bytes(b"this is not a valid parquet file")
        with pytest.raises(StorageError, match="corrupted"):
            storage.get_last_timestamp("EUR_USD")

---

---

## TEST 4.4 -- get_last_timestamp returns None for empty file

WHAT IT CHECKS:
An empty but valid Parquet file (zero rows) should return None,
treating it as a first-run scenario since there is no data to resume from.

GOLDEN PROMPT

---

    def test_get_last_timestamp_returns_none_for_empty_file(storage):
        import pandas as pd
        # Write a valid but empty Parquet file
        empty_df = pd.DataFrame(columns=["timestamp"])
        file_path = storage.raw_dir / "EUR_USD_ticks.parquet"
        empty_df.to_parquet(file_path, index=False)
        result = storage.get_last_timestamp("EUR_USD")
        assert result is None

---

---

## TEST 4.5 -- get_last_timestamp is symbol-specific

WHAT IT CHECKS:
Querying EUR_USD must not be affected by GBP_USD having data.
Symbol isolation must be respected.

GOLDEN PROMPT

---

    def test_get_last_timestamp_is_symbol_specific(storage, valid_tick_gbp):
        storage.save_tick_batch([valid_tick_gbp])
        result = storage.get_last_timestamp("EUR_USD")
        assert result is None  # EUR_USD has no data

---

---

## TEST 4.6 -- get_last_timestamp performs columnar read (only timestamp col)

WHAT IT CHECKS:
The columnar-read optimisation. The method must request only the
timestamp column from Parquet, not the full file. Verified by
asserting that read_parquet is called with columns=["timestamp"].

GOLDEN PROMPT

---

    def test_get_last_timestamp_reads_only_timestamp_column(storage, valid_tick):
        from unittest.mock import patch
        import pandas as pd
        storage.save_tick_batch([valid_tick])
        with patch("pandas.read_parquet", wraps=pd.read_parquet) as mock_read:
            storage.get_last_timestamp("EUR_USD")
            call_kwargs = mock_read.call_args
            assert call_kwargs.kwargs.get("columns") == ["timestamp"] or \
                   (call_kwargs.args and "timestamp" in str(call_kwargs))

---

---

GROUP 5 -- get_bars: Read Operations (7 tests)
File: tests/core/storage/test_read_operations.py

---

---

## TEST 5.1 -- get_bars returns None when no file exists

GOLDEN PROMPT

---

    def test_get_bars_returns_none_when_no_file(storage):
        result = storage.get_bars("EUR_USD", timeframe="M1")
        assert result is None

---

---

## TEST 5.2 -- get_bars returns DataFrame with correct columns

GOLDEN PROMPT

---

    def test_get_bars_returns_dataframe(storage, valid_bar):
        import pandas as pd
        storage.save_bar(valid_bar)
        df = storage.get_bars("EUR_USD", timeframe="M1")
        assert isinstance(df, pd.DataFrame)
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

---

---

## TEST 5.3 -- get_bars applies max_rows cap correctly

WHAT IT CHECKS:
When max_rows=1 and there are 2 bars, only the most recent bar
must be returned (tail, not head).

GOLDEN PROMPT

---

    def test_get_bars_applies_max_rows_cap(storage, valid_bar, valid_bar_2):
        storage.save_bar(valid_bar)
        storage.save_bar(valid_bar_2)
        df = storage.get_bars("EUR_USD", timeframe="M1", max_rows=1)
        assert len(df) == 1
        # Must be the most recent bar (valid_bar_2 has later timestamp)
        import pandas as pd
        assert df.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15 10:31:00")

---

---

## TEST 5.4 -- get_bars returns all rows when max_rows is None

GOLDEN PROMPT

---

    def test_get_bars_returns_all_rows_when_no_cap(storage, valid_bar, valid_bar_2):
        storage.save_bar(valid_bar)
        storage.save_bar(valid_bar_2)
        df = storage.get_bars("EUR_USD", timeframe="M1", max_rows=None)
        assert len(df) == 2

---

---

## TEST 5.5 -- get_bars raises StorageError on corrupted file

WHAT IT CHECKS:
Corruption vs missing distinction for bar data. A corrupted bar
file must raise StorageError so the trainer does not silently
proceed with no data.

GOLDEN PROMPT

---

    def test_get_bars_raises_on_corrupted_file(storage):
        from src.core.storage import StorageError
        file_path = storage.processed_dir / "EUR_USD_M1.parquet"
        file_path.write_bytes(b"corrupted parquet content")
        with pytest.raises(StorageError, match="corrupted"):
            storage.get_bars("EUR_USD", timeframe="M1")

---

---

## TEST 5.6 -- get_bars output is sorted by timestamp

GOLDEN PROMPT

---

    def test_get_bars_output_is_sorted(storage, valid_bar, valid_bar_2):
        storage.save_bar(valid_bar_2)  # write out of order
        storage.save_bar(valid_bar)
        df = storage.get_bars("EUR_USD", timeframe="M1")
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

---

---

## TEST 5.7 -- get_bars defaults to M1 timeframe

GOLDEN PROMPT

---

    def test_get_bars_defaults_to_m1(storage, valid_bar):
        storage.save_bar(valid_bar)
        df = storage.get_bars("EUR_USD")  # no timeframe argument
        assert df is not None
        assert len(df) == 1

---

---

GROUP 6 -- Diagnostics: get_tick_count + list_symbols (6 tests)
File: tests/core/storage/test_diagnostics.py

---

---

## TEST 6.1 -- get_tick_count returns 0 when no file exists

GOLDEN PROMPT

---

    def test_get_tick_count_zero_when_no_file(storage):
        assert storage.get_tick_count("EUR_USD") == 0

---

---

## TEST 6.2 -- get_tick_count returns correct count after writes

GOLDEN PROMPT

---

    def test_get_tick_count_correct_after_writes(storage, valid_tick, valid_tick_2):
        storage.save_tick_batch([valid_tick, valid_tick_2])
        assert storage.get_tick_count("EUR_USD") == 2

---

---

## TEST 6.3 -- get_tick_count returns 0 on read error (not raise)

WHAT IT CHECKS:
get_tick_count is a diagnostic/health-check method. It must not
crash the system on error -- it returns 0 and logs a warning.

GOLDEN PROMPT

---

    def test_get_tick_count_returns_zero_on_error(storage):
        file_path = storage.raw_dir / "EUR_USD_ticks.parquet"
        file_path.write_bytes(b"corrupted")
        result = storage.get_tick_count("EUR_USD")
        assert result == 0  # soft failure -- no raise

---

---

## TEST 6.4 -- list_symbols returns empty list when no files exist

GOLDEN PROMPT

---

    def test_list_symbols_empty_when_no_files(storage):
        result = storage.list_symbols()
        assert result == []

---

---

## TEST 6.5 -- list_symbols returns all symbol names

GOLDEN PROMPT

---

    def test_list_symbols_returns_all_symbols(storage, valid_tick, valid_tick_gbp):
        storage.save_tick_batch([valid_tick])
        storage.save_tick_batch([valid_tick_gbp])
        result = storage.list_symbols()
        assert "EUR_USD" in result
        assert "GBP_USD" in result

---

---

## TEST 6.6 -- list_symbols returns sorted list

GOLDEN PROMPT

---

    def test_list_symbols_is_sorted(storage, valid_tick, valid_tick_gbp):
        storage.save_tick_batch([valid_tick_gbp])
        storage.save_tick_batch([valid_tick])
        result = storage.list_symbols()
        assert result == sorted(result)

---

---

GROUP 7 -- \_atomic_upsert: Edge Cases (8 tests)
File: tests/core/storage/test_atomic_upsert.py

---

---

## TEST 7.1 -- \_atomic_upsert raises StorageError when timestamp column missing

WHAT IT CHECKS:
The schema guard. A DataFrame without a 'timestamp' column must be
rejected before any file operation is attempted.

GOLDEN PROMPT

---

    def test_atomic_upsert_raises_on_missing_timestamp_column(storage, tmp_path):
        import pandas as pd
        from src.core.storage import StorageError
        bad_df = pd.DataFrame({"price": [1.0, 2.0]})  # no timestamp column
        with pytest.raises(StorageError, match="missing required column"):
            storage._atomic_upsert(tmp_path / "test.parquet", bad_df)

---

---

## TEST 7.2 -- \_atomic_upsert creates new file when none exists

GOLDEN PROMPT

---

    def test_atomic_upsert_creates_new_file(storage, tmp_path):
        import pandas as pd
        from datetime import datetime
        df = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
        path = tmp_path / "test.parquet"
        storage._atomic_upsert(path, df)
        assert path.exists()

---

---

## TEST 7.3 -- \_atomic_upsert deduplicates on second write

GOLDEN PROMPT

---

    def test_atomic_upsert_deduplicates(storage, tmp_path):
        import pandas as pd
        from datetime import datetime
        ts = datetime(2024, 1, 1)
        df = pd.DataFrame({"timestamp": [ts], "value": [1.0]})
        path = tmp_path / "test.parquet"
        storage._atomic_upsert(path, df)
        storage._atomic_upsert(path, df)  # same data again
        result = pd.read_parquet(path)
        assert len(result) == 1

---

---

## TEST 7.4 -- \_atomic_upsert last-in-wins on conflicting timestamp

WHAT IT CHECKS:
When the same timestamp is written twice with different values,
the second write must win. This is the backfill overlap resolution.

GOLDEN PROMPT

---

    def test_atomic_upsert_last_in_wins(storage, tmp_path):
        import pandas as pd
        from datetime import datetime
        ts = datetime(2024, 1, 1)
        df_first = pd.DataFrame({"timestamp": [ts], "value": [1.0]})
        df_second = pd.DataFrame({"timestamp": [ts], "value": [99.0]})
        path = tmp_path / "test.parquet"
        storage._atomic_upsert(path, df_first)
        storage._atomic_upsert(path, df_second)
        result = pd.read_parquet(path)
        assert len(result) == 1
        assert result.iloc[0]["value"] == 99.0

---

---

## TEST 7.5 -- \_atomic_upsert output is sorted by timestamp

GOLDEN PROMPT

---

    def test_atomic_upsert_output_sorted(storage, tmp_path):
        import pandas as pd
        from datetime import datetime
        df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 3), datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "value": [3.0, 1.0, 2.0]
        })
        path = tmp_path / "test.parquet"
        storage._atomic_upsert(path, df)
        result = pd.read_parquet(path)
        timestamps = result["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

---

---

## TEST 7.6 -- \_atomic_upsert raises StorageError on read failure

GOLDEN PROMPT

---

    def test_atomic_upsert_raises_storage_error_on_read_failure(storage, tmp_path):
        import pandas as pd
        from src.core.storage import StorageError
        from datetime import datetime
        path = tmp_path / "test.parquet"
        path.write_bytes(b"corrupted parquet")  # bad file exists
        df = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
        with pytest.raises(StorageError, match="Parquet upsert failed"):
            storage._atomic_upsert(path, df)

---

---

## TEST 7.7 -- \_atomic_upsert raises StorageError on write failure

GOLDEN PROMPT

---

    def test_atomic_upsert_raises_storage_error_on_write_failure(storage, tmp_path):
        import pandas as pd
        from unittest.mock import patch
        from src.core.storage import StorageError
        from datetime import datetime
        df = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
        path = tmp_path / "test.parquet"
        with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk full")):
            with pytest.raises(StorageError, match="Parquet upsert failed"):
                storage._atomic_upsert(path, df)

---

---

## TEST 7.8 -- \_atomic_upsert preserves existing data when appending

GOLDEN PROMPT

---

    def test_atomic_upsert_preserves_existing_data(storage, tmp_path):
        import pandas as pd
        from datetime import datetime
        df1 = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
        df2 = pd.DataFrame({"timestamp": [datetime(2024, 1, 2)], "value": [2.0]})
        path = tmp_path / "test.parquet"
        storage._atomic_upsert(path, df1)
        storage._atomic_upsert(path, df2)
        result = pd.read_parquet(path)
        assert len(result) == 2
        assert set(result["value"].tolist()) == {1.0, 2.0}

---

---

GROUP 8 -- Concurrency: Thread Safety (8 tests)
File: tests/core/storage/test_concurrency.py

---

---

## TEST 8.1 -- Concurrent tick batch writes produce correct total count

WHAT IT CHECKS:
10 threads each writing a 10-tick batch must produce exactly 100
unique ticks in the final file with no duplicates or lost writes.

GOLDEN PROMPT

---

    def test_concurrent_tick_writes_correct_total(storage, valid_tick):
        import threading, pandas as pd
        from datetime import datetime

        num_threads = 10
        ticks_per_thread = 10
        barrier = threading.Barrier(num_threads)
        errors = []

        def write_batch(thread_id):
            from src.ml_engine.model import Tick
            batch = [
                Tick(
                    timestamp=datetime(2024, 1, 15, 10, thread_id, i),
                    symbol="EUR_USD",
                    bid=1.0850,
                    ask=1.0852,
                    source="TWELVE",
                )
                for i in range(ticks_per_thread)
            ]
            barrier.wait()
            try:
                storage.save_tick_batch(batch)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_batch, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Write errors: {errors}"
        df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
        assert len(df) == num_threads * ticks_per_thread

---

---

## TEST 8.2 -- Concurrent bar writes produce correct total count

GOLDEN PROMPT

---

    def test_concurrent_bar_writes_correct_total(storage):
        import threading, pandas as pd
        from datetime import datetime
        from src.ml_engine.model import Bar

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        errors = []

        def write_bar(thread_id):
            bar = Bar(
                timestamp=datetime(2024, 1, 15, 10, thread_id, 0),
                symbol="EUR_USD",
                open=1.0850,
                high=1.0865,
                low=1.0848,
                close=1.0860,
                volume=100 + thread_id,
            )
            barrier.wait()
            try:
                storage.save_bar(bar)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_bar, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
        assert len(df) == num_threads

---

---

## TEST 8.3 -- Concurrent read and write do not cause data corruption

WHAT IT CHECKS:
A reader (get_last_timestamp) running simultaneously with a writer
(save_tick_batch) must not cause corrupted data or exceptions.

GOLDEN PROMPT

---

    def test_concurrent_read_write_no_corruption(storage):
        import threading
        from datetime import datetime
        from src.ml_engine.model import Tick

        errors = []

        def write_ticks():
            for i in range(20):
                tick = Tick(
                    timestamp=datetime(2024, 1, 15, 10, 0, i),
                    symbol="EUR_USD",
                    bid=1.0850,
                    ask=1.0852,
                    source="TWELVE",
                )
                try:
                    storage.save_tick_batch([tick])
                except Exception as e:
                    errors.append(e)

        def read_timestamps():
            for _ in range(20):
                try:
                    storage.get_last_timestamp("EUR_USD")
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=write_ticks)
        t2 = threading.Thread(target=read_timestamps)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []

---

---

## TEST 8.4 -- Concurrent writes to different symbols do not interfere

WHAT IT CHECKS:
Writing EUR_USD and GBP_USD simultaneously must produce two
separate, correct files with no cross-contamination.

GOLDEN PROMPT

---

    def test_concurrent_writes_different_symbols_no_interference(storage):
        import threading, pandas as pd
        from datetime import datetime
        from src.ml_engine.model import Tick

        errors = []

        def write_eur():
            for i in range(10):
                tick = Tick(
                    timestamp=datetime(2024, 1, 15, 10, 0, i),
                    symbol="EUR_USD",
                    bid=1.0850, ask=1.0852, source="TWELVE",
                )
                storage.save_tick_batch([tick])

        def write_gbp():
            for i in range(10):
                tick = Tick(
                    timestamp=datetime(2024, 1, 15, 10, 0, i),
                    symbol="GBP_USD",
                    bid=1.2650, ask=1.2652, source="QUOTEX",
                )
                storage.save_tick_batch([tick])

        t1 = threading.Thread(target=write_eur)
        t2 = threading.Thread(target=write_gbp)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        eur_df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
        gbp_df = pd.read_parquet(storage.raw_dir / "GBP_USD_ticks.parquet")

        assert len(eur_df) == 10
        assert len(gbp_df) == 10
        assert all(eur_df["symbol"] == "EUR_USD")
        assert all(gbp_df["symbol"] == "GBP_USD")

---

---

## TEST 8.5 -- No data loss on simultaneous flush + write

WHAT IT CHECKS:
A graceful shutdown scenario: one thread is writing a final batch
while another is calling get_tick_count for the dashboard.
Neither must crash or produce incorrect results.

GOLDEN PROMPT

---

    def test_no_data_loss_on_simultaneous_flush_and_write(storage):
        import threading
        from datetime import datetime
        from src.ml_engine.model import Tick

        errors = []

        def write():
            for i in range(30):
                tick = Tick(
                    timestamp=datetime(2024, 1, 15, 10, 0, i),
                    symbol="EUR_USD",
                    bid=1.0850, ask=1.0852, source="TWELVE",
                )
                try:
                    storage.save_tick_batch([tick])
                except Exception as e:
                    errors.append(e)

        def read_count():
            for _ in range(10):
                try:
                    storage.get_tick_count("EUR_USD")
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=write)
        t2 = threading.Thread(target=read_count)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []

---

---

## TEST 8.6 -- list_symbols is safe to call during concurrent writes

GOLDEN PROMPT

---

    def test_list_symbols_safe_during_concurrent_writes(storage):
        import threading
        from datetime import datetime
        from src.ml_engine.model import Tick

        errors = []

        def write_symbols():
            for i in range(5):
                tick = Tick(
                    timestamp=datetime(2024, 1, 15, 10, 0, i),
                    symbol="EUR_USD",
                    bid=1.0850, ask=1.0852, source="TWELVE",
                )
                storage.save_tick_batch([tick])

        def list_repeatedly():
            for _ in range(10):
                try:
                    storage.list_symbols()
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=write_symbols)
        t2 = threading.Thread(target=list_repeatedly)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []

---

---

## TEST 8.7 -- Two threads writing the same timestamp produce one row

WHAT IT CHECKS:
Race condition on the deduplication path. If two threads both
write the same timestamp simultaneously, the final file must have
exactly one row for that timestamp.

GOLDEN PROMPT

---

    def test_two_threads_same_timestamp_one_row(storage):
        import threading, pandas as pd
        from datetime import datetime
        from src.ml_engine.model import Tick

        ts = datetime(2024, 1, 15, 10, 0, 0)
        tick = Tick(timestamp=ts, symbol="EUR_USD",
                    bid=1.0850, ask=1.0852, source="TWELVE")

        barrier = threading.Barrier(2)

        def write():
            barrier.wait()
            storage.save_tick_batch([tick])

        t1 = threading.Thread(target=write)
        t2 = threading.Thread(target=write)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
        assert len(df) == 1

---

---

## TEST 8.8 -- Lock is released after StorageError

WHAT IT CHECKS:
If \_atomic_upsert raises StorageError, the lock must be released.
If the lock is not released (leaked lock), the next write will
hang forever. This test verifies the lock is not leaked on failure.

GOLDEN PROMPT

---

    def test_lock_released_after_storage_error(storage, valid_tick):
        from unittest.mock import patch
        from src.core.storage import StorageError
        import threading

        with patch.object(
            storage, "_atomic_upsert",
            side_effect=StorageError("forced failure")
        ):
            with pytest.raises(StorageError):
                storage.save_tick_batch([valid_tick])

        # If lock was leaked, this acquire would block forever (timeout = failure)
        acquired = storage._lock.acquire(timeout=1)
        assert acquired, "Lock was not released after StorageError"
        storage._lock.release()

---

---

GROUP 9 -- save_bar_batch: Batch Write Operations + Guards (9 tests)
File: tests/data/storage/test_save_bar_batch.py

---

WHAT IT COVERS:
save_bar_batch() is the primary write path used by the Historian after
every chunk fetch. It enforces single-symbol and single-timeframe
contracts before committing to disk. All guards and the happy path
are verified here.

---

## TEST 9.1 -- Empty batch returns False without writing

GOLDEN PROMPT

---

    def test_save_bar_batch_empty_batch_returns_false(storage):
        result = storage.save_bar_batch([])
        assert result is False

    def test_save_bar_batch_empty_batch_creates_no_file(storage):
        storage.save_bar_batch([])
        assert len(list(storage.processed_dir.glob("*.parquet"))) == 0

---

---

## TEST 9.2 -- Mixed symbols raises ValueError

GOLDEN PROMPT

---

    def test_save_bar_batch_mixed_symbols_raises(storage):
        bars = [_bar("EUR_USD"), _bar("GBP_USD")]
        with pytest.raises(ValueError, match="mixed symbols"):
            storage.save_bar_batch(bars)

---

---

## TEST 9.3 -- Mixed timeframes raises ValueError

GOLDEN PROMPT

---

    def test_save_bar_batch_mixed_timeframes_raises(storage):
        bars = [_bar(timeframe=Timeframe.M1), _bar(timeframe=Timeframe.M5)]
        with pytest.raises(ValueError, match="mixed timeframes"):
            storage.save_bar_batch(bars)

---

---

## TEST 9.4 -- Happy path: returns True and creates Parquet file

GOLDEN PROMPT

---

    def test_save_bar_batch_returns_true_on_success(storage):
        assert storage.save_bar_batch([_bar()]) is True

    def test_save_bar_batch_creates_parquet_file(storage):
        storage.save_bar_batch([_bar()])
        assert (storage.processed_dir / "EUR_USD_M1.parquet").exists()

---

---

## TEST 9.5 -- Persists correct row count

GOLDEN PROMPT

---

    def test_save_bar_batch_persists_correct_count(storage):
        import pandas as pd
        bars = [_bar(ts=datetime(2026, 4, 12, 10, i, 0)) for i in range(3)]
        storage.save_bar_batch(bars)
        df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
        assert len(df) == 3

---

---

## TEST 9.6 -- Deduplicates on repeated call

GOLDEN PROMPT

---

    def test_save_bar_batch_deduplicates_on_repeated_call(storage):
        import pandas as pd
        bars = [_bar(ts=datetime(2026, 4, 12, 10, 0, 0))]
        storage.save_bar_batch(bars)
        storage.save_bar_batch(bars)
        df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
        assert len(df) == 1

---

---

## TEST 9.7 -- Appends new bars on successive calls

GOLDEN PROMPT

---

    def test_save_bar_batch_appends_new_bars(storage):
        import pandas as pd
        storage.save_bar_batch([_bar(ts=datetime(2026, 4, 12, 10, 0, 0))])
        storage.save_bar_batch([_bar(ts=datetime(2026, 4, 12, 10, 1, 0))])
        df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
        assert len(df) == 2

---

---

GROUP 10 -- Azure Blob Integration (11 tests)
File: tests/data/storage/test_azure.py

---

WHAT IT COVERS:
\_init_azure_client(), sync_to_azure(), and pull_from_azure(). All Azure
SDK calls are mocked via MagicMock — no real credentials required. A
\_cloud_storage(tmp_path) helper constructs a Storage instance via
**new** with a mocked CLOUD config, bypassing **init** so each test
controls \_container_client directly.

---

## TEST 10.1 -- \_init_azure_client returns ContainerClient in CLOUD mode

GOLDEN PROMPT

---

    def test_init_azure_client_returns_container_client_on_success(tmp_path):
        store = _cloud_storage(tmp_path)
        mock_client = MagicMock()
        with patch("src.data.storage.BlobServiceClient") as mock_svc:
            mock_svc.from_connection_string.return_value \
                .get_container_client.return_value = mock_client
            mock_client.get_container_properties.return_value = {}
            result = store._init_azure_client()
        assert result is mock_client

---

---

## TEST 10.2 -- \_init_azure_client exits with code 1 on connection failure

GOLDEN PROMPT

---

    def test_init_azure_client_exits_on_connection_failure(tmp_path):
        store = _cloud_storage(tmp_path)
        with patch("src.data.storage.BlobServiceClient") as mock_svc:
            mock_svc.from_connection_string.return_value \
                .get_container_client.return_value \
                .get_container_properties.side_effect = Exception("unreachable")
            with pytest.raises(SystemExit) as exc_info:
                store._init_azure_client()
        assert exc_info.value.code == 1

---

---

## TEST 10.3 -- \_init_azure_client returns None in LOCAL mode

GOLDEN PROMPT

---

    def test_init_azure_client_returns_none_in_local_mode(tmp_path):
        store = _cloud_storage(tmp_path)
        store._settings.data_mode = "LOCAL"
        with patch("src.data.storage.BlobServiceClient") as mock_svc:
            result = store._init_azure_client()
        mock_svc.assert_not_called()
        assert result is None

---

---

## TEST 10.4 -- sync_to_azure skips and returns False in LOCAL mode

GOLDEN PROMPT

---

    def test_sync_to_azure_skips_when_local_mode(tmp_path):
        store = _cloud_storage(tmp_path)
        store._container_client = None
        assert store.sync_to_azure(tmp_path / "file.parquet") is False

---

---

## TEST 10.5 -- sync_to_azure returns False when file does not exist

GOLDEN PROMPT

---

    def test_sync_to_azure_returns_false_when_file_missing(tmp_path):
        store = _cloud_storage(tmp_path)
        store._container_client = MagicMock()
        assert store.sync_to_azure(tmp_path / "nonexistent.parquet") is False

---

---

## TEST 10.6 -- sync_to_azure returns True and calls upload_blob on success

GOLDEN PROMPT

---

    def test_sync_to_azure_returns_true_on_success(tmp_path):
        store = _cloud_storage(tmp_path)
        mock_container = MagicMock()
        store._container_client = mock_container
        local_file = tmp_path / "model.pkl"
        local_file.write_bytes(b"data")
        with patch("builtins.open", mock_open(read_data=b"data")):
            result = store.sync_to_azure(local_file)
        assert result is True
        mock_container.get_blob_client.assert_called_once()

---

---

## TEST 10.7 -- sync_to_azure uses custom blob_name when supplied

GOLDEN PROMPT

---

    def test_sync_to_azure_uses_custom_blob_name(tmp_path):
        store = _cloud_storage(tmp_path)
        mock_container = MagicMock()
        store._container_client = mock_container
        local_file = tmp_path / "model.pkl"
        local_file.write_bytes(b"data")
        with patch("builtins.open", mock_open(read_data=b"data")):
            store.sync_to_azure(local_file, blob_name="models/prod/model.pkl")
        mock_container.get_blob_client.assert_called_once_with("models/prod/model.pkl")

---

---

## TEST 10.8 -- sync_to_azure returns False on upload error (non-fatal)

GOLDEN PROMPT

---

    def test_sync_to_azure_returns_false_on_upload_error(tmp_path):
        store = _cloud_storage(tmp_path)
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value \
            .upload_blob.side_effect = Exception("network error")
        store._container_client = mock_container
        local_file = tmp_path / "model.pkl"
        local_file.write_bytes(b"data")
        with patch("builtins.open", mock_open(read_data=b"data")):
            assert store.sync_to_azure(local_file) is False

---

---

## TEST 10.9 -- pull_from_azure skips and returns False in LOCAL mode

GOLDEN PROMPT

---

    def test_pull_from_azure_skips_when_local_mode(tmp_path):
        store = _cloud_storage(tmp_path)
        store._container_client = None
        assert store.pull_from_azure("models/model.pkl", tmp_path / "model.pkl") is False

---

---

## TEST 10.10 -- pull_from_azure returns True and writes file on success

GOLDEN PROMPT

---

    def test_pull_from_azure_returns_true_on_success(tmp_path):
        store = _cloud_storage(tmp_path)
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value \
            .download_blob.return_value.readall.return_value = b"model bytes"
        store._container_client = mock_container
        with patch("builtins.open", mock_open()) as mocked_file:
            result = store.pull_from_azure("models/model.pkl", tmp_path / "model.pkl")
        assert result is True
        mocked_file().write.assert_called_once_with(b"model bytes")

---

---

## TEST 10.11 -- pull_from_azure returns False on download error (non-fatal)

GOLDEN PROMPT

---

    def test_pull_from_azure_returns_false_on_download_error(tmp_path):
        store = _cloud_storage(tmp_path)
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value \
            .download_blob.side_effect = Exception("blob not found")
        store._container_client = mock_container
        assert store.pull_from_azure("models/missing.pkl", tmp_path / "missing.pkl") is False

---

---

## RUNNING THE FULL SUITE

INSTALL DEPENDENCIES:
pip install pytest pytest-cov pandas pyarrow python-dotenv azure-storage-blob aioresponses

RUN ALL STORAGE TESTS:
pytest tests/data/storage/ -v

RUN WITH COVERAGE:
pytest tests/data/storage/ -v \
 --cov=src.data.storage \
 --cov-report=term-missing

RUN ONLY AZURE TESTS:
pytest tests/data/storage/test_azure.py -v

RUN ONLY CONCURRENCY TESTS:
pytest tests/data/storage/test_concurrency.py -v

RUN ONLY I/O TESTS (excluding concurrency):
pytest tests/data/storage/ -v --ignore=tests/data/storage/test_concurrency.py

EXPECTED COUNT:
78 tests collected across 10 test files.
src/data/storage.py coverage: 100%

IMPORTANT -- HERMETIC I/O:
Every test that performs file I/O uses the `storage` fixture which
wires Storage to tmp_path. The real data/ directory is NEVER touched
by any test in this suite. Confirm this before running:
grep -r "data/raw" tests/data/storage/ # should return nothing

---

## END OF TEST HARNESS
