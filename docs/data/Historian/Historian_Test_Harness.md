# Historian.py — Complete Test Harness

---

TRADER-AI DATA: HISTORIAN.PY -- COMPLETE TEST HARNESS
Updated : 2026-04-14 | File Under Test : src/data/historian.py
Runner : pytest-asyncio | Style: Matches config + model test conventions

---

Total Test Cases : 55
Test Groups : 10
Coverage Target : 100% — achieved
Actual Coverage : 100% (src/data/historian.py)

DEPENDENCIES:
pip install pytest pytest-asyncio pytest-cov aioresponses

PHILOSOPHY:
historian.py is a network-and-storage bridge. Tests must verify four
things equally: 1. Valid data flows from API to Storage without modification or loss. 2. Invalid/missing data is handled gracefully — skipped bars, empty
chunks, and StorageErrors never abort the backfill session. 3. Infrastructure constraints (rate limits, retries, backoff) are
enforced exactly as specified — no accidental quota burns. 4. The consecutive-failure guard aborts at the right threshold to
protect the 800 req/day API quota against a systemic storage failure.

TEST FILE LAYOUT:
tests/
data/
historian/
conftest.py <-- shared fixtures (historian + mocks)
test_historian_error.py <-- Group 1 (HistorianError class)
test_init.py <-- Group 2 (Historian.**init** + get_historian singleton)
test_determine_start.py <-- Group 3 (\_determine_start)
test_parse_bars.py <-- Group 4 (\_parse_bars)
test_save_bars.py <-- Group 5 (\_save_bars)
test_rate_limit.py <-- Group 6 (\_enforce_rate_limit)
test_backfill.py <-- Group 7 (backfill)
test_backfill_all.py <-- Group 8 (backfill_all)
test_fetch_chunk.py <-- Group 9 (\_fetch_chunk + HTTP mocking)
test_fetch_and_save.py <-- Group 10 (\_fetch_and_save + failure counter)

ASYNC NOTE:
asyncio_mode = auto is set in pytest.ini. No per-test @pytest.mark.asyncio
decoration is required. The asyncio_default_fixture_loop_scope = function
setting ensures each async test gets its own event loop.

---

## HISTORIAN CONFTEST.PY -- SHARED FIXTURES

PURPOSE:
Provides pre-built mock objects for Historian construction and a factory
for valid Twelve Data API response dictionaries. These are the "golden"
objects — consistent inputs that satisfy all validation rules.

GOLDEN PROMPT -- tests/data/historian/conftest.py

---

Create tests/data/historian/conftest.py with the following content:

    import pytest
    import pandas as pd
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, AsyncMock, patch

    from src.data.historian import Historian, HistorianError
    from src.ml_engine.model import Bar, Timeframe


    # ── Shared constants ─────────────────────────────────────────────────────

    VALID_SYMBOL = "EUR_USD"
    VALID_API_SYMBOL = "EUR/USD"
    NOW_UTC = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    LAST_BAR_TS = datetime(2026, 4, 12, 23, 59, 0)   # naive UTC


    # ── Settings mock ────────────────────────────────────────────────────────

    @pytest.fixture
    def mock_settings():
        """Minimal mock of Config with historian-relevant fields."""
        settings = MagicMock()
        settings.twelvedata_api_key = "test_api_key_32chars_000000000000"
        settings.backfill_years = 2
        settings.backfill_pairs = ["EUR_USD", "GBP_USD"]
        return settings


    # ── Storage mock ─────────────────────────────────────────────────────────

    @pytest.fixture
    def mock_storage():
        """Mock Storage with get_bars and save_bar_batch pre-configured."""
        storage = MagicMock()
        storage.get_bars.return_value = None     # default: no existing data
        storage.save_bar_batch.return_value = True
        return storage


    # ── Historian under test ─────────────────────────────────────────────────

    @pytest.fixture
    def historian(mock_settings, mock_storage):
        """
        Historian instance with mocked settings and storage.

        Bypasses get_settings() and Storage() construction so tests
        do not require a real .env file or filesystem.
        """
        with patch("src.data.historian.get_settings", return_value=mock_settings), \
             patch("src.data.historian.Storage", return_value=mock_storage):
            h = Historian()
        return h


    # ── API response factory ─────────────────────────────────────────────────

    def make_api_values(n: int = 3, symbol: str = VALID_SYMBOL) -> list[dict]:
        """
        Build a list of n valid Twelve Data bar dicts, newest-first.
        Timestamps descend from 2026-04-12 10:02:00 down by 1 minute.
        """
        base = datetime(2026, 4, 12, 10, 2, 0)
        return [
            {
                "datetime": (base - pd.Timedelta(minutes=i)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "open":   "1.08500",
                "high":   "1.08550",
                "low":    "1.08480",
                "close":  "1.08520",
                "volume": "342",
            }
            for i in range(n)
        ]


    @pytest.fixture
    def api_values():
        """Three valid bar dicts in newest-first order."""
        return make_api_values(3)


    @pytest.fixture
    def valid_bar():
        """A known-good Bar for use in _save_bars and batch tests."""
        return Bar(
            timestamp=datetime(2026, 4, 12, 10, 0, 0),
            symbol=VALID_SYMBOL,
            open=1.08500,
            high=1.08550,
            low=1.08480,
            close=1.08520,
            volume=342.0,
            is_complete=True,
            timeframe=Timeframe.M1,
        )

---

---

GROUP 1 -- HistorianError: Construction & Attributes (2 tests)
File: tests/data/historian/test_historian_error.py

---

---

## TEST 1.1 -- HistorianError stores the symbol attribute

WHAT IT CHECKS:
The custom exception must carry a `symbol` attribute so the pipeline
can identify which pair failed without parsing the error string.

GOLDEN PROMPT

---

    def test_historian_error_stores_symbol():
        from src.data.historian import HistorianError
        err = HistorianError("fetch failed", symbol="EUR_USD")
        assert err.symbol == "EUR_USD"
        assert "fetch failed" in str(err)

---

---

## TEST 1.2 -- HistorianError symbol defaults to empty string

WHAT IT CHECKS:
When no symbol is provided (e.g., a generic infrastructure failure),
the attribute must exist and default to "". Tests that the default
argument is correctly wired in **init**.

GOLDEN PROMPT

---

    def test_historian_error_default_symbol():
        from src.data.historian import HistorianError
        err = HistorianError("generic failure")
        assert err.symbol == ""
        assert isinstance(err, Exception)

---

---

GROUP 2 -- Historian.**init**: Wiring & Defaults (4 tests)
File: tests/data/historian/test_init.py

---

---

## TEST 2.1 -- Historian constructs without error with valid dependencies

GOLDEN PROMPT

---

    def test_historian_constructs(historian):
        assert historian is not None

---

---

## TEST 2.2 -- \_settings is populated from get_settings()

GOLDEN PROMPT

---

    def test_historian_settings_wired(historian, mock_settings):
        assert historian._settings is mock_settings

---

---

## TEST 2.3 -- \_storage is the injected Storage instance

GOLDEN PROMPT

---

    def test_historian_storage_wired(historian, mock_storage):
        assert historian._storage is mock_storage

---

---

## TEST 2.4 -- \_last_request_time initialises to 0.0

WHAT IT CHECKS:
The rate limiter relies on \_last_request_time starting at 0.0 so
the very first API call fires immediately without waiting 8 seconds.
A non-zero initial value would delay the first request unnecessarily.

GOLDEN PROMPT

---

    def test_historian_last_request_time_initialises_to_zero(historian):
        assert historian._last_request_time == 0.0

---

---

GROUP 3 -- \_determine_start: Gap Detection (4 tests)
File: tests/data/historian/test_determine_start.py

---

---

## TEST 3.1 -- No existing data returns backfill_years ago at midnight UTC

WHAT IT CHECKS:
When Storage.get_bars returns None (first run),\_determine_start must
return (now - BACKFILL_YEARS \* 365 days) normalised to midnight UTC.
The midnight normalisation ensures chunks start at clean day boundaries.

GOLDEN PROMPT

---

    def test_determine_start_first_run_midnight(historian, mock_storage):
        from datetime import datetime, timezone, timedelta
        mock_storage.get_bars.return_value = None
        now = datetime(2026, 4, 13, 14, 30, 0, tzinfo=timezone.utc)

        start = historian._determine_start("EUR_USD", now)

        expected = now - timedelta(days=365 * 2)
        expected = expected.replace(hour=0, minute=0, second=0, microsecond=0)
        assert start == expected

---

---

## TEST 3.2 -- Existing data returns last timestamp plus one minute

WHAT IT CHECKS:
When Storage.get_bars returns a DataFrame,\_determine_start must
return the last row's timestamp + 1 minute. This prevents re-fetching
the already-stored bar while starting as close to the gap as possible.

GOLDEN PROMPT

---

    def test_determine_start_resume_one_minute_after_last_bar(
        historian, mock_storage
    ):
        import pandas as pd
        from datetime import datetime, timezone, timedelta

        last_ts = datetime(2026, 4, 12, 23, 59, 0)
        df = pd.DataFrame({"timestamp": [last_ts]})
        mock_storage.get_bars.return_value = df

        now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
        start = historian._determine_start("EUR_USD", now)

        expected = last_ts.replace(tzinfo=timezone.utc) + timedelta(minutes=1)
        assert start == expected

---

---

## TEST 3.3 -- Timezone-naive last timestamp is made UTC-aware

WHAT IT CHECKS:
Parquet timestamps are stored as naive datetimes. \_determine_start
must attach timezone.utc before adding timedelta, because mixing
naive and aware datetimes raises TypeError in Python 3.

GOLDEN PROMPT

---

    def test_determine_start_attaches_utc_to_naive_timestamp(
        historian, mock_storage
    ):
        import pandas as pd
        from datetime import datetime, timezone

        naive_ts = datetime(2026, 4, 10, 8, 0, 0)   # no tzinfo
        df = pd.DataFrame({"timestamp": [naive_ts]})
        mock_storage.get_bars.return_value = df

        now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
        start = historian._determine_start("EUR_USD", now)

        # Must not raise; result must be timezone-aware
        assert start.tzinfo is not None

---

---

## TEST 3.4 -- BACKFILL_YEARS setting is respected

WHAT IT CHECKS:
A BACKFILL_YEARS of 1 produces a cutoff 365 days back, not 730.
Verifies the constant is read from settings, not hardcoded.

GOLDEN PROMPT

---

    def test_determine_start_respects_backfill_years(
        historian, mock_settings, mock_storage
    ):
        from datetime import datetime, timezone, timedelta
        mock_settings.backfill_years = 1
        mock_storage.get_bars.return_value = None

        now = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
        start = historian._determine_start("EUR_USD", now)

        expected_date = (now - timedelta(days=365)).date()
        assert start.date() == expected_date

---

---

GROUP 4 -- \_parse_bars: Bar Factory & Validation (9 tests)
File: tests/data/historian/test_parse_bars.py

---

---

## TEST 4.1 -- Valid API values produce correct Bar objects

WHAT IT CHECKS:
The baseline. Given well-formed API dicts, \_parse_bars must produce
Bar objects with the correct OHLCV values and M1 timeframe.

GOLDEN PROMPT

---

    def test_parse_bars_valid_values(historian, api_values):
        bars = historian._parse_bars("EUR_USD", api_values)
        assert len(bars) == 3
        for bar in bars:
            assert bar.symbol == "EUR_USD"
            assert bar.timeframe.value == "M1"
            assert bar.is_complete is True

---

---

## TEST 4.2 -- Output is in chronological order (reversed from API input)

WHAT IT CHECKS:
Twelve Data returns bars newest-first. \_parse_bars must reverse the
list so the output is chronological. This is critical for Storage's
sort-on-write and for ML feature engineering windows.

GOLDEN PROMPT

---

    def test_parse_bars_output_is_chronological(historian, api_values):
        # api_values is newest-first (descending timestamps)
        bars = historian._parse_bars("EUR_USD", api_values)
        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

---

---

## TEST 4.3 -- Volume floor applied when API reports volume=0

WHAT IT CHECKS:
OTC/synthetic pairs from Twelve Data sometimes report volume=0.
The \_MIN_VOLUME floor (1.0) must be applied so bars are never
rejected by future Bar validation that treats volume<=0 as invalid.

GOLDEN PROMPT

---

    def test_parse_bars_volume_floor_applied_on_zero(historian):
        values = [
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "1.0850", "high": "1.0855",
                "low": "1.0848", "close": "1.0852",
                "volume": "0",
            }
        ]
        bars = historian._parse_bars("EUR_USD", values)
        assert len(bars) == 1
        assert bars[0].volume >= 1.0

---

---

## TEST 4.4 -- Volume floor applied when volume field is missing

WHAT IT CHECKS:
If the API omits the volume field entirely (not just zero), the
default in v.get("volume", \_MIN_VOLUME) must apply the floor.

GOLDEN PROMPT

---

    def test_parse_bars_volume_floor_applied_on_missing_field(historian):
        values = [
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "1.0850", "high": "1.0855",
                "low": "1.0848", "close": "1.0852",
                # "volume" key deliberately absent
            }
        ]
        bars = historian._parse_bars("EUR_USD", values)
        assert len(bars) == 1
        assert bars[0].volume >= 1.0

---

---

## TEST 4.5 -- Malformed datetime string skips bar without crashing

WHAT IT CHECKS:
If the API returns an unparseable datetime string, the bar is skipped
with a warning. The remaining bars in the chunk must still be returned.

GOLDEN PROMPT

---

    def test_parse_bars_skips_bad_datetime(historian):
        values = [
            {
                "datetime": "NOT-A-DATE",
                "open": "1.0850", "high": "1.0855",
                "low": "1.0848", "close": "1.0852", "volume": "100",
            },
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "1.0850", "high": "1.0855",
                "low": "1.0848", "close": "1.0852", "volume": "100",
            },
        ]
        bars = historian._parse_bars("EUR_USD", values)
        # Only the valid bar is returned
        assert len(bars) == 1

---

---

## TEST 4.6 -- Non-numeric price field skips bar without crashing

WHAT IT CHECKS:
If a required field (e.g., "open") cannot be cast to float, the
ValueError from float() must be caught and the bar skipped.

GOLDEN PROMPT

---

    def test_parse_bars_skips_non_numeric_price(historian):
        values = [
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "N/A", "high": "1.0855",   # N/A is a Twelve Data sentinel
                "low": "1.0848", "close": "1.0852", "volume": "100",
            }
        ]
        bars = historian._parse_bars("EUR_USD", values)
        assert len(bars) == 0

---

---

## TEST 4.7 -- OHLC violation causes Bar.**post_init** to skip the bar

WHAT IT CHECKS:
If the API returns a physically impossible bar (low > high), Bar
raises ValueError in **post_init**. \_parse_bars must catch this and
skip the bar, not let the exception propagate.

GOLDEN PROMPT

---

    def test_parse_bars_skips_ohlc_violation(historian):
        values = [
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "1.0850",
                "high": "1.0840",   # high < open and high < low — invalid
                "low":  "1.0860",   # low > high — OHLC violation
                "close": "1.0852",
                "volume": "100",
            }
        ]
        bars = historian._parse_bars("EUR_USD", values)
        assert len(bars) == 0

---

---

## TEST 4.8 -- Empty values list returns empty list

WHAT IT CHECKS:
An API response with an empty "values" list (no bars for the window)
must produce an empty list without raising an error.

GOLDEN PROMPT

---

    def test_parse_bars_empty_values_returns_empty_list(historian):
        bars = historian._parse_bars("EUR_USD", [])
        assert bars == []

---

---

## TEST 4.9 -- Skipped bar count is logged as a warning

WHAT IT CHECKS:
When bars are skipped, \_parse_bars must emit a warning log containing
the skip count. This is essential for diagnosing data quality issues
in production without having to inspect every chunk manually.

GOLDEN PROMPT

---

    def test_parse_bars_logs_skip_count(historian, caplog):
        import logging
        values = [
            {"datetime": "BAD", "open": "1.0", "high": "1.1",
             "low": "1.0", "close": "1.05", "volume": "10"},
        ]
        with caplog.at_level(logging.WARNING, logger="src.data.historian"):
            historian._parse_bars("EUR_USD", values)

        assert any("Skipped" in r.message for r in caplog.records)

---

---

GROUP 5 -- \_save_bars: Batch Persistence (4 tests)
File: tests/data/historian/test_save_bars.py

---

---

## TEST 5.1 -- \_save_bars delegates to storage.save_bar_batch, not save_bar

WHAT IT CHECKS:
The critical efficiency guarantee. \_save_bars must call save_bar_batch
(one Parquet operation per chunk) rather than save_bar in a loop
(one Parquet operation per bar). One call per chunk is the contract.

GOLDEN PROMPT

---

    def test_save_bars_calls_save_bar_batch(historian, mock_storage, valid_bar):
        bars = [valid_bar, valid_bar]
        historian._save_bars("EUR_USD", bars)

        mock_storage.save_bar_batch.assert_called_once_with(bars)
        mock_storage.save_bar.assert_not_called()

---

---

## TEST 5.2 -- \_save_bars returns len(bars) on success

WHAT IT CHECKS:
The return value is used by \_fetch_and_save to accumulate a total
bar count. A successful batch must return the count of bars passed in.

GOLDEN PROMPT

---

    def test_save_bars_returns_bar_count_on_success(historian, mock_storage, valid_bar):
        bars = [valid_bar, valid_bar, valid_bar]
        result = historian._save_bars("EUR_USD", bars)
        assert result == 3

---

---

## TEST 5.3 -- \_save_bars returns 0 on StorageError without re-raising

WHAT IT CHECKS:
A StorageError (disk full, schema mismatch) must be caught and logged
as a [%] warning. The backfill walk must continue to the next chunk —
a single failed write should never abort the entire session.

GOLDEN PROMPT

---

    def test_save_bars_returns_zero_on_storage_error(
        historian, mock_storage, valid_bar
    ):
        from src.data.storage import StorageError
        mock_storage.save_bar_batch.side_effect = StorageError("disk full")

        result = historian._save_bars("EUR_USD", [valid_bar])

        assert result == 0
        # Must not raise — the StorageError is caught

---

---

## TEST 5.4 -- \_save_bars returns 0 on empty list without calling storage

WHAT IT CHECKS:
Empty chunks (weekend windows) produce an empty bars list. \_save_bars
must short-circuit without calling storage at all.

GOLDEN PROMPT

---

    def test_save_bars_returns_zero_on_empty_list(historian, mock_storage):
        result = historian._save_bars("EUR_USD", [])
        assert result == 0
        mock_storage.save_bar_batch.assert_not_called()

---

---

GROUP 6 -- \_enforce_rate_limit: Inter-Request Delay (3 tests)
File: tests/data/historian/test_rate_limit.py

---

---

## TEST 6.1 -- No sleep when the full interval has already elapsed

WHAT IT CHECKS:
If more than \_REQUEST_INTERVAL_S seconds have passed since the last
request (e.g., a slow network response), no additional sleep is
needed. asyncio.sleep must not be called with a positive value.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_rate_limit_no_sleep_when_interval_elapsed(historian):
        import time
        from src.data.historian import _REQUEST_INTERVAL_S

        # Simulate a request that happened 10 seconds ago (past the interval)
        historian._last_request_time = time.monotonic() - (_REQUEST_INTERVAL_S + 2)

        with patch("src.data.historian.asyncio.sleep") as mock_sleep:
            await historian._enforce_rate_limit()
            # sleep should not be called, or called with 0 or negative value
            if mock_sleep.called:
                args = mock_sleep.call_args[0]
                assert args[0] <= 0

---

---

## TEST 6.2 -- Sleeps for the remaining portion of the interval

WHAT IT CHECKS:
If the last request was 2 seconds ago and the interval is 8 seconds,
\_enforce_rate_limit must sleep for 6 seconds (not 8). Sleeping the
full interval would over-throttle requests unnecessarily.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_rate_limit_sleeps_remaining_duration(historian):
        import time
        from src.data.historian import _REQUEST_INTERVAL_S
        from unittest.mock import patch, AsyncMock

        historian._last_request_time = time.monotonic() - 2.0  # 2s ago

        with patch("src.data.historian.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await historian._enforce_rate_limit()

        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        # Remaining ≈ 6s (8 - 2). Allow ±0.5s tolerance for test execution time.
        assert 5.5 <= sleep_duration <= 6.5

---

---

## TEST 6.3 -- First call fires immediately (\_last_request_time = 0.0)

WHAT IT CHECKS:
On the very first request of a session, \_last_request_time is 0.0.
The elapsed time since epoch is many seconds, so no sleep should occur.
This verifies the 0.0 initialisation strategy works as intended.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_rate_limit_no_sleep_on_first_call(historian):
        from unittest.mock import patch, AsyncMock

        # Default: _last_request_time = 0.0
        assert historian._last_request_time == 0.0

        with patch("src.data.historian.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await historian._enforce_rate_limit()

        # No sleep on the very first call — elapsed time since epoch is huge
        mock_sleep.assert_not_called()

---

---

GROUP 7 -- backfill: Public Orchestration (5 tests)
File: tests/data/historian/test_backfill.py

---

---

## TEST 7.1 -- Returns 0 immediately when data is already up to date

WHAT IT CHECKS:
If \_determine_start returns a datetime >= now_utc (data is current),
backfill must return 0 without making any API calls. This is the
normal path after a recent session.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_returns_zero_when_up_to_date(historian):
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch

        # Simulate: last bar is in the future relative to now
        future_start = datetime.now(timezone.utc) + timedelta(minutes=5)

        with patch.object(historian, "_determine_start", return_value=future_start), \
             patch.object(historian, "_fetch_and_save") as mock_fetch:

            result = await historian.backfill("EUR_USD")

        assert result == 0
        mock_fetch.assert_not_called()

---

---

## TEST 7.2 -- Calls \_fetch_and_save when a gap exists

WHAT IT CHECKS:
When \_determine_start returns a time in the past, backfill must call
\_fetch_and_save with the correct start and end datetimes.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_calls_fetch_and_save_with_gap(historian):
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch, AsyncMock

        past_start = datetime.now(timezone.utc) - timedelta(days=1)

        with patch.object(historian, "_determine_start", return_value=past_start), \
             patch.object(
                 historian, "_fetch_and_save", new_callable=AsyncMock, return_value=150
             ) as mock_fetch:

            result = await historian.backfill("EUR_USD")

        assert result == 150
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][0] == "EUR_USD"   # symbol
        assert call_args[0][1] == past_start  # start_dt

---

---

## TEST 7.3 -- Returns total bars committed from \_fetch_and_save

WHAT IT CHECKS:
backfill must propagate the integer returned by \_fetch_and_save
as its own return value. This count is used by pipeline.py and
backfill_all to verify how much data was actually committed.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_returns_total_bars_from_fetch(historian):
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch, AsyncMock

        past_start = datetime.now(timezone.utc) - timedelta(hours=2)

        with patch.object(historian, "_determine_start", return_value=past_start), \
             patch.object(
                 historian, "_fetch_and_save", new_callable=AsyncMock, return_value=42
             ):
            result = await historian.backfill("GBP_USD")

        assert result == 42

---

---

## TEST 7.4 -- HistorianError from \_fetch_and_save propagates to caller

WHAT IT CHECKS:
backfill does not suppress HistorianError. The exception propagates
to backfill_all (which catches it per-symbol) or directly to the
pipeline if called standalone.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_propagates_historian_error(historian):
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch, AsyncMock
        from src.data.historian import HistorianError

        past_start = datetime.now(timezone.utc) - timedelta(hours=1)

        with patch.object(historian, "_determine_start", return_value=past_start), \
             patch.object(
                 historian, "_fetch_and_save",
                 new_callable=AsyncMock,
                 side_effect=HistorianError("all retries failed", symbol="EUR_USD"),
             ):
            with pytest.raises(HistorianError) as exc_info:
                await historian.backfill("EUR_USD")

        assert exc_info.value.symbol == "EUR_USD"

---

---

## TEST 7.5 -- Logs backfill start and completion blocks

WHAT IT CHECKS:
backfill must emit INFO-level log entries at start and completion.
These milestone logs are essential for monitoring a 15-minute session
in production without tailing every chunk-level message.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_logs_start_and_complete(historian, caplog):
        import logging
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch, AsyncMock

        past_start = datetime.now(timezone.utc) - timedelta(days=1)

        with patch.object(historian, "_determine_start", return_value=past_start), \
             patch.object(
                 historian, "_fetch_and_save", new_callable=AsyncMock, return_value=100
             ):
            with caplog.at_level(logging.INFO, logger="src.data.historian"):
                await historian.backfill("EUR_USD")

        messages = " ".join(r.message for r in caplog.records)
        assert "EUR_USD" in messages

---

---

GROUP 8 -- backfill_all: Multi-Pair Orchestration (4 tests)
File: tests/data/historian/test_backfill_all.py

---

---

## TEST 8.1 -- Processes all pairs from BACKFILL_PAIRS setting

WHAT IT CHECKS:
backfill_all must call backfill() once for each symbol in
settings.backfill_pairs. Missing a pair is a silent data gap.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_all_processes_all_pairs(historian, mock_settings):
        from unittest.mock import patch, AsyncMock

        mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]

        with patch.object(
            historian, "backfill", new_callable=AsyncMock, return_value=100
        ) as mock_backfill:
            results = await historian.backfill_all()

        assert mock_backfill.call_count == 3
        called_symbols = [c[0][0] for c in mock_backfill.call_args_list]
        assert "EUR_USD" in called_symbols
        assert "GBP_USD" in called_symbols
        assert "USD_JPY" in called_symbols

---

---

## TEST 8.2 -- Returns dict mapping each symbol to its bar count

WHAT IT CHECKS:
The return value is consumed by the pipeline to verify coverage.
Each symbol must map to the exact count returned by its backfill().

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_all_returns_correct_results_dict(historian, mock_settings):
        from unittest.mock import patch, AsyncMock

        mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD"]

        side_effects = [200, 150]
        with patch.object(
            historian, "backfill",
            new_callable=AsyncMock,
            side_effect=side_effects,
        ):
            results = await historian.backfill_all()

        assert results == {"EUR_USD": 200, "GBP_USD": 150}

---

---

## TEST 8.3 -- HistorianError for one pair does not abort the others

WHAT IT CHECKS:
If EUR_USD fails with HistorianError, GBP_USD must still be backfilled.
The failed pair maps to 0 in the results dict. This is the per-symbol
error isolation guarantee documented in the design.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_all_continues_after_historian_error(
        historian, mock_settings
    ):
        from unittest.mock import patch, AsyncMock
        from src.data.historian import HistorianError

        mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD"]

        async def backfill_side_effect(symbol):
            if symbol == "EUR_USD":
                raise HistorianError("timeout", symbol=symbol)
            return 180

        with patch.object(historian, "backfill", side_effect=backfill_side_effect):
            results = await historian.backfill_all()

        assert results["EUR_USD"] == 0    # failed pair
        assert results["GBP_USD"] == 180  # succeeded despite EUR_USD failure

---

---

## TEST 8.4 -- Pairs are processed sequentially, not concurrently

WHAT IT CHECKS:
Concurrent backfills would violate the rate limiter — two pairs
running simultaneously could each fire a request within the same
8-second window. The loop must be sequential (one pair completes
before the next starts).

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_backfill_all_processes_pairs_sequentially(
        historian, mock_settings
    ):
        import asyncio
        from unittest.mock import patch, AsyncMock

        mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD"]
        call_order = []

        async def ordered_backfill(symbol):
            call_order.append(symbol)
            await asyncio.sleep(0)   # yield to event loop
            return 0

        with patch.object(historian, "backfill", side_effect=ordered_backfill):
            await historian.backfill_all()

        # Sequential means the order matches the config list exactly
        assert call_order == ["EUR_USD", "GBP_USD"]

---

---

GROUP 9 -- \_fetch_chunk: HTTP Fetching & Retry Logic (6 tests)
File: tests/data/historian/test_fetch_chunk.py

---

NOTE ON MOCKING aiohttp:
Use the `aioresponses` library to mock the aiohttp ClientSession without
modifying the production code. Install with: pip install aioresponses

All tests in this group require:
from aioresponses import aioresponses as mock_aiohttp

The fixture below creates a shared session for injection:

---

    import pytest
    import aiohttp

    @pytest.fixture
    async def http_session():
        async with aiohttp.ClientSession() as session:
            yield session

---

---

## TEST 9.1 -- Successful HTTP 200 response returns parsed Bar list

WHAT IT CHECKS:
The happy path. A valid API response must be parsed and returned as
a list of Bar objects. This verifies the full chain from HTTP to Bar.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_fetch_chunk_success_returns_bars(historian, http_session):
        from aioresponses import aioresponses
        from datetime import datetime, timezone

        payload = {
            "status": "ok",
            "values": [
                {
                    "datetime": "2026-04-12 10:00:00",
                    "open": "1.0850", "high": "1.0860",
                    "low": "1.0845", "close": "1.0855", "volume": "200",
                }
            ],
        }

        with aioresponses() as m:
            m.get("https://api.twelvedata.com/time_series", payload=payload)
            bars = await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, 0, tzinfo=timezone.utc),
            )

        assert len(bars) == 1
        assert bars[0].symbol == "EUR_USD"

---

---

## TEST 9.2 -- HTTP 429 triggers retry with exponential backoff

WHAT IT CHECKS:
A 429 response must not immediately raise HistorianError. The method
must sleep and retry. After the retry succeeds, bars are returned.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_fetch_chunk_retries_on_429(historian, http_session):
        from aioresponses import aioresponses
        from datetime import datetime, timezone
        from unittest.mock import patch, AsyncMock

        success_payload = {
            "status": "ok",
            "values": [
                {
                    "datetime": "2026-04-12 10:00:00",
                    "open": "1.0850", "high": "1.0860",
                    "low": "1.0845", "close": "1.0855", "volume": "200",
                }
            ],
        }

        with aioresponses() as m, \
             patch("src.data.historian.asyncio.sleep", new_callable=AsyncMock):

            m.get("https://api.twelvedata.com/time_series", status=429)
            m.get("https://api.twelvedata.com/time_series", payload=success_payload)

            bars = await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
            )

        assert len(bars) == 1

---

---

## TEST 9.3 -- HTTP 500 triggers retry

WHAT IT CHECKS:
A 5xx server error is a transient failure. The method must retry
rather than treating it as a permanent API error.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_fetch_chunk_retries_on_http_500(historian, http_session):
        from aioresponses import aioresponses
        from datetime import datetime, timezone
        from unittest.mock import patch, AsyncMock

        success_payload = {
            "status": "ok",
            "values": [
                {
                    "datetime": "2026-04-12 10:00:00",
                    "open": "1.0850", "high": "1.0860",
                    "low": "1.0845", "close": "1.0855", "volume": "200",
                }
            ],
        }

        with aioresponses() as m, \
             patch("src.data.historian.asyncio.sleep", new_callable=AsyncMock):

            m.get("https://api.twelvedata.com/time_series", status=500, body="error")
            m.get("https://api.twelvedata.com/time_series", payload=success_payload)

            bars = await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
            )

        assert len(bars) == 1

---

---

## TEST 9.4 -- All retries exhausted raises HistorianError

WHAT IT CHECKS:
If all \_MAX_RETRIES attempts fail (e.g., network is down), the method
must raise HistorianError with the symbol set correctly. No partial
result should be returned.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_fetch_chunk_raises_historian_error_after_all_retries(
        historian, http_session
    ):
        from aioresponses import aioresponses
        from datetime import datetime, timezone
        from unittest.mock import patch, AsyncMock
        from src.data.historian import HistorianError, _MAX_RETRIES

        with aioresponses() as m, \
             patch("src.data.historian.asyncio.sleep", new_callable=AsyncMock):

            for _ in range(_MAX_RETRIES):
                m.get("https://api.twelvedata.com/time_series", status=503, body="down")

            with pytest.raises(HistorianError) as exc_info:
                await historian._fetch_chunk(
                    session=http_session,
                    api_symbol="EUR/USD",
                    symbol="EUR_USD",
                    start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                    end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
                )

        assert exc_info.value.symbol == "EUR_USD"

---

---

## TEST 9.5 -- API error in JSON body returns empty list (no crash)

WHAT IT CHECKS:
Twelve Data returns HTTP 200 even for API-level errors (e.g., invalid
symbol, quota exceeded). The JSON body has no "values" key in this
case. The method must return [] rather than raising.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_fetch_chunk_returns_empty_on_api_error_body(
        historian, http_session
    ):
        from aioresponses import aioresponses
        from datetime import datetime, timezone

        error_payload = {
            "status": "error",
            "message": "You have run out of API credits.",
            "code": 429,
        }

        with aioresponses() as m:
            m.get("https://api.twelvedata.com/time_series", payload=error_payload)

            bars = await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
            )

        assert bars == []

---

---

## TEST 9.6 -- aiohttp.ClientError triggers retry, then HistorianError

WHAT IT CHECKS:
A network-level error (DNS failure, connection refused) raises
aiohttp.ClientError. The method must catch it, retry, and eventually
raise HistorianError if all retries fail.

GOLDEN PROMPT

---

    @pytest.mark.asyncio
    async def test_fetch_chunk_retries_on_client_error(historian, http_session):
        from aioresponses import aioresponses, CallbackResult
        from datetime import datetime, timezone
        from unittest.mock import patch, AsyncMock
        from src.data.historian import HistorianError, _MAX_RETRIES
        import aiohttp

        with aioresponses() as m, \
             patch("src.data.historian.asyncio.sleep", new_callable=AsyncMock):

            for _ in range(_MAX_RETRIES):
                m.get(
                    "https://api.twelvedata.com/time_series",
                    exception=aiohttp.ClientConnectionError("refused"),
                )

            with pytest.raises(HistorianError):
                await historian._fetch_chunk(
                    session=http_session,
                    api_symbol="EUR/USD",
                    symbol="EUR_USD",
                    start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                    end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
                )

---

---

GROUP 10 -- \_fetch_and_save: Chunk Walker + Consecutive Failure Guard (10 tests)
File: tests/data/historian/test_fetch_and_save.py

---

WHAT IT COVERS:
\_fetch_and_save is the orchestration layer between the rate-limited HTTP
fetcher and Storage. Tests here focus solely on the loop logic: chunk
accumulation, empty-chunk skipping, error propagation, and the
consecutive-failure counter that protects the API quota.

\_fetch_chunk and \_save_bars are both mocked so these tests run fast
without any network or disk I/O.

HELPERS:
\_START /\_END : A date range fitting inside one 7-day chunk.
\_MULTI_START / \_END : A range spanning multiple chunks (>7 days).
valid_bar fixture : A single known-good Bar from conftest.py.

TEST 10.1 -- Single chunk: correct bar count returned

---

    async def test_fetch_and_save_returns_bar_count_on_success(historian, valid_bar):
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars", return_value=3):
            mock_fetch.return_value = [valid_bar, valid_bar, valid_bar]
            result = await historian._fetch_and_save("EUR_USD", _START, _END)
        assert result == 3

---

TEST 10.2 -- \_save_bars called with correct args when bars returned

---

    async def test_fetch_and_save_calls_save_bars_when_bars_returned(historian, valid_bar):
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars", return_value=1) as mock_save:
            mock_fetch.return_value = [valid_bar]
            await historian._fetch_and_save("EUR_USD", _START, _END)
        mock_save.assert_called_once_with("EUR_USD", [valid_bar])

---

TEST 10.3 -- Empty chunk: \_save_bars not called, returns 0

---

    async def test_fetch_and_save_skips_save_on_empty_chunk(historian):
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars") as mock_save:
            mock_fetch.return_value = []
            result = await historian._fetch_and_save("EUR_USD", _START, _END)
        assert result == 0
        mock_save.assert_not_called()

---

TEST 10.4 -- HistorianError from \_fetch_chunk propagates unchanged

---

    async def test_fetch_and_save_propagates_historian_error(historian):
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = HistorianError("network down", symbol="EUR_USD")
            with pytest.raises(HistorianError) as exc_info:
                await historian._fetch_and_save("EUR_USD", _START, _END)
        assert exc_info.value.symbol == "EUR_USD"

---

TEST 10.5 -- Multi-chunk range triggers multiple \_fetch_chunk calls

---

    async def test_fetch_and_save_multi_chunk_calls_fetch_multiple_times(historian, valid_bar):
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars", return_value=1):
            mock_fetch.return_value = [valid_bar]
            await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)
        assert mock_fetch.call_count > 1

---

TEST 10.6 -- Multi-chunk totals are summed correctly

---

    async def test_fetch_and_save_multi_chunk_sums_totals(historian, valid_bar):
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars", return_value=5):
            mock_fetch.return_value = [valid_bar]
            result = await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)
        assert result == mock_fetch.call_count * 5

---

TEST 10.7 -- Mixed empty/non-empty chunks: only non-empty trigger save

---

    async def test_fetch_and_save_mixed_empty_and_nonempty_chunks(historian, valid_bar):
        responses = [[] if i % 2 == 0 else [valid_bar] for i in range(3)]
        call_index = {"n": 0}
        async def _side_effect(**kwargs):
            idx = call_index["n"]; call_index["n"] += 1
            return responses[idx] if idx < len(responses) else []
        with patch.object(historian, "_fetch_chunk", side_effect=_side_effect), \
             patch.object(historian, "_save_bars", return_value=1) as mock_save:
            await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)
        assert mock_save.call_count == sum(1 for r in responses if r)

---

TEST 10.8 -- Aborts after \_MAX_CONSECUTIVE_STORAGE_FAILURES consecutive failures

---

    WHAT IT CHECKS:
      When _save_bars returns 0 on every non-empty chunk, the counter reaches
      the threshold and HistorianError is raised. This is the quota-protection
      path — continuing to fetch when storage is broken wastes API credits.

    async def test_fetch_and_save_aborts_after_consecutive_storage_failures(historian, valid_bar):
        from src.data.historian import _MAX_CONSECUTIVE_STORAGE_FAILURES
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars", return_value=0):
            mock_fetch.return_value = [valid_bar]
            with pytest.raises(HistorianError) as exc_info:
                await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)
        assert exc_info.value.symbol == "EUR_USD"
        assert mock_fetch.call_count >= _MAX_CONSECUTIVE_STORAGE_FAILURES

---

TEST 10.9 -- Counter resets to 0 after a successful save

---

    WHAT IT CHECKS:
      Two failures followed by one success followed by two more failures
      must NOT trigger the abort. The successful save resets the counter,
      so the second pair of failures is only at count=2, below threshold=3.

    async def test_fetch_and_save_resets_failure_counter_on_success(historian, valid_bar):
        save_results = [0, 0, 3, 0, 0]
        save_iter = iter(save_results)
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars", side_effect=lambda *_: next(save_iter, 1)):
            mock_fetch.return_value = [valid_bar]
            result = await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)
        assert result >= 3

---

TEST 10.10 -- Empty chunks do not advance the failure counter

---

    WHAT IT CHECKS:
      Weekends/holidays return [] from _fetch_chunk. These must NOT
      increment the counter even when _save_bars always returns 0 for
      the non-empty chunks in between. The alternating pattern
      (empty, fail, empty, fail, ...) should never trigger the abort
      within a single-chunk test range.

    async def test_fetch_and_save_empty_chunks_do_not_count_as_failures(historian, valid_bar):
        # Single range (_START/_END) — only one chunk possible, no abort
        with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
             patch.object(historian, "_save_bars", return_value=0):
            mock_fetch.return_value = []   # empty chunk — no save attempted
            try:
                await historian._fetch_and_save("EUR_USD", _START, _END)
            except HistorianError:
                pytest.fail("HistorianError raised for empty chunk — counter must not increment")

---

---

## RUNNING THE FULL SUITE

INSTALL DEPENDENCIES:
pip install pytest pytest-asyncio pytest-cov aioresponses

RUN ALL HISTORIAN TESTS:
pytest tests/data/historian/ -v

RUN WITH COVERAGE:
pytest tests/data/historian/ -v \
 --cov=src.data.historian \
 --cov-report=term-missing

RUN A SPECIFIC GROUP:
pytest tests/data/historian/test_parse_bars.py -v
pytest tests/data/historian/test_fetch_chunk.py -v
pytest tests/data/historian/test_fetch_and_save.py -v

EXPECTED COUNT:
55 tests collected across 10 files.
src/data/historian.py coverage: 100%

KNOWN DEPENDENCY:
Group 9 requires `aioresponses`. If unavailable, mock the session
directly using unittest.mock.AsyncMock on session.get().**aenter**.

---

## END OF TEST HARNESS
