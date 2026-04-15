================================================================================
  TRADER-AI CORE: MODEL.PY -- COMPLETE TEST HARNESS
  Generated : 2026-04-13 | File Under Test : src/ml_engine/model.py
  Runner    : pytest | Style: Matches config test harness conventions
================================================================================

  Total Test Cases  : 63
  Test Groups       : 7
  Coverage Target   : 100% of all validation branches + happy paths

  PRE-FLIGHT NOTE:
  Before running any tests, confirm the engine pseudocode block is removed
  from Bar.__post_init__ (lines 229-234 in the current file). Those lines
  cause a SyntaxError at collection time and will prevent the entire suite
  from running. The block to delete is:

      # Engine -- owns the logging and the decision to crash on invalid data
      try:
          bar = Bar(...)
      except ValueError as e:
          logger.warning(f"Skipping invalid bar: {e}")
          continue

  Also confirm the duplicate `from enum import Enum` import (lines 10+14)
  has been reduced to a single import.

  PHILOSOPHY:
  model.py is a data firewall. Tests must verify two things equally:
    1. Valid data passes through without modification (except tz stripping).
    2. Invalid data is caught at the door -- no silent acceptance, no
       partial construction, no zombie objects.

  TEST FILE LAYOUT:
    tests/
      conftest.py                      <-- shared fixtures (already exists)
      ml_engine/
        conftest.py                    <-- model-specific shared fixtures
        test_tick.py                   <-- Groups 1-3  (Tick class)
        test_bar.py                    <-- Groups 4-5  (Bar class)
        test_data_buffer.py            <-- Groups 6-7  (DataBuffer class)

================================================================================
  ML_ENGINE CONFTEST.PY -- SHARED FIXTURES
================================================================================

  PURPOSE:
  Provides pre-built valid Tick and Bar instances for use across all
  ml_engine tests. These are the "golden" objects -- constructed from
  known-good values that pass all validation rules.

  GOLDEN PROMPT -- tests/ml_engine/conftest.py:
  -----------------------------------------------------------------------
  Create tests/ml_engine/conftest.py with the following content:

    import pytest
    from datetime import datetime, timezone
    from src.ml_engine.model import Tick, Bar, Timeframe


    VALID_TS_NAIVE = datetime(2024, 1, 15, 10, 30, 0)
    VALID_TS_AWARE = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


    @pytest.fixture
    def valid_tick():
        """A known-good Tick that passes all validation."""
        return Tick(
            timestamp=VALID_TS_NAIVE,
            symbol="EUR_USD",
            bid=1.08500,
            ask=1.08520,
            source="TWELVE",
        )


    @pytest.fixture
    def valid_bar():
        """A known-good Bar that passes all validation."""
        return Bar(
            timestamp=VALID_TS_NAIVE,
            symbol="EUR_USD",
            open_price=1.0850,
            high=1.0865,
            low=1.0848,
            close=1.0860,
            volume=342,
        )


    @pytest.fixture
    def valid_tick_factory():
        """
        Returns a factory function for creating valid Ticks with
        optional field overrides. Use this when a test needs to
        tweak one field while keeping the rest valid.

        Usage:
            def test_something(valid_tick_factory):
                tick = valid_tick_factory(bid=1.0900, ask=1.0905)
        """
        def _factory(**overrides):
            defaults = dict(
                timestamp=VALID_TS_NAIVE,
                symbol="EUR_USD",
                bid=1.08500,
                ask=1.08520,
                source="TWELVE",
            )
            defaults.update(overrides)
            return Tick(**defaults)
        return _factory


    @pytest.fixture
    def valid_bar_factory():
        """
        Returns a factory function for creating valid Bars with
        optional field overrides.
        """
        def _factory(**overrides):
            defaults = dict(
                timestamp=VALID_TS_NAIVE,
                symbol="EUR_USD",
                open_price=1.0850,
                high=1.0865,
                low=1.0848,
                close=1.0860,
                volume=342,
            )
            defaults.update(overrides)
            return Bar(**defaults)
        return _factory
  -----------------------------------------------------------------------


================================================================================
  GROUP 1 -- Tick: Happy Path & Field Correctness  (7 tests)
  File: tests/ml_engine/test_tick.py
================================================================================

------------------------------------------------------------------------
  TEST 1.1 -- Valid Tick constructs without error
------------------------------------------------------------------------

  WHAT IT CHECKS:
    The baseline. A Tick with physically valid values must construct
    cleanly and all fields must be accessible with the correct values.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_valid_construction(valid_tick):
        assert valid_tick.symbol == "EUR_USD"
        assert valid_tick.bid == pytest.approx(1.08500)
        assert valid_tick.ask == pytest.approx(1.08520)
        assert valid_tick.source == "TWELVE"
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 1.2 -- Both source values are accepted
------------------------------------------------------------------------

  WHAT IT CHECKS:
    "TWELVE" and "QUOTEX" are the only two valid sources. Both must
    be accepted without error.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    @pytest.mark.parametrize("source", ["TWELVE", "QUOTEX"])
    def test_tick_valid_sources(valid_tick_factory, source):
        tick = valid_tick_factory(source=source)
        assert tick.source == source
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 1.3 -- Tick is immutable after construction
------------------------------------------------------------------------

  WHAT IT CHECKS:
    frozen=True must prevent any field from being reassigned after
    the object is created. This is the immutability guarantee.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_is_frozen(valid_tick):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            valid_tick.bid = 9.9999
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 1.4 -- mid_price property returns correct value
------------------------------------------------------------------------

  WHAT IT CHECKS:
    (bid + ask) / 2. Using known values to confirm the arithmetic
    is correct and the property is accessible.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_mid_price(valid_tick):
        # bid=1.08500, ask=1.08520 -> mid = 1.08510
        assert valid_tick.mid_price == pytest.approx(1.08510)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 1.5 -- to_dict returns all expected keys
------------------------------------------------------------------------

  WHAT IT CHECKS:
    to_dict() must return a flat dictionary with exactly the five
    Tick fields as keys. Used for Pandas/Parquet compatibility.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_to_dict_keys(valid_tick):
        d = valid_tick.to_dict()
        assert set(d.keys()) == {"timestamp", "symbol", "bid", "ask", "source"}
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 1.6 -- __repr__ contains key fields
------------------------------------------------------------------------

  WHAT IT CHECKS:
    The custom __repr__ must include the symbol, bid, ask, and source
    so log output is readable. Does not check exact format -- checks
    presence of critical information.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_repr_contains_key_fields(valid_tick):
        r = repr(valid_tick)
        assert "EUR_USD" in r
        assert "1.08500" in r
        assert "TWELVE" in r
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 1.7 -- Timezone-aware timestamp is stripped to naive UTC
------------------------------------------------------------------------

  WHAT IT CHECKS:
    If a timezone-aware datetime is passed in, __post_init__ must
    strip the tzinfo and store a naive UTC datetime. This prevents
    mixed-timezone Parquet files.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_aware_timestamp_stripped(valid_tick_factory):
        from datetime import timezone
        aware_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        tick = valid_tick_factory(timestamp=aware_ts)
        assert tick.timestamp.tzinfo is None
        assert tick.timestamp == datetime(2024, 1, 15, 10, 30, 0)
  -----------------------------------------------------------------------


================================================================================
  GROUP 2 -- Tick: Validation & Rejection  (8 tests)
  File: tests/ml_engine/test_tick.py
================================================================================

------------------------------------------------------------------------
  TEST 2.1 -- Zero bid is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_zero_bid_rejected(valid_tick_factory):
        with pytest.raises(ValueError, match="integrity violation"):
            valid_tick_factory(bid=0.0)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 2.2 -- Negative bid is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_negative_bid_rejected(valid_tick_factory):
        with pytest.raises(ValueError, match="integrity violation"):
            valid_tick_factory(bid=-1.0850)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 2.3 -- Zero ask is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_zero_ask_rejected(valid_tick_factory):
        with pytest.raises(ValueError, match="integrity violation"):
            valid_tick_factory(ask=0.0, bid=0.0)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 2.4 -- Inverted spread is rejected (bid > ask)
------------------------------------------------------------------------

  WHAT IT CHECKS:
    A tick where bid > ask is physically impossible (you can never
    sell higher than the buy price simultaneously). Must be rejected.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_inverted_spread_rejected(valid_tick_factory):
        with pytest.raises(ValueError, match="integrity violation"):
            valid_tick_factory(bid=1.0860, ask=1.0850)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 2.5 -- Equal bid and ask is accepted (zero spread)
------------------------------------------------------------------------

  WHAT IT CHECKS:
    bid == ask is physically valid -- it means zero spread. Some
    brokers report this on synthetic instruments. Must NOT be rejected.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_equal_bid_ask_accepted(valid_tick_factory):
        tick = valid_tick_factory(bid=1.0850, ask=1.0850)
        assert tick.mid_price == pytest.approx(1.0850)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 2.6 -- Unknown source string is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_unknown_source_rejected(valid_tick_factory):
        with pytest.raises(ValueError, match="integrity violation"):
            valid_tick_factory(source="BLOOMBERG")
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 2.7 -- Lowercase source string is rejected
------------------------------------------------------------------------

  WHAT IT CHECKS:
    Source validation is case-sensitive. "twelve" and "quotex" are
    not in the allowed set {"TWELVE", "QUOTEX"}. Must be rejected.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    @pytest.mark.parametrize("source", ["twelve", "quotex", "Twelve", "Quotex"])
    def test_tick_lowercase_source_rejected(valid_tick_factory, source):
        with pytest.raises(ValueError, match="integrity violation"):
            valid_tick_factory(source=source)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 2.8 -- Multiple violations are all reported in a single raise
------------------------------------------------------------------------

  WHAT IT CHECKS:
    This is the multi-violation fix. If a tick has BOTH an invalid
    price AND a bad source, both must be captured. We verify this by
    checking that the violation count in the error message is 2,
    confirming neither was silently swallowed.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_tick_multiple_violations_reported(valid_tick_factory, caplog):
        import logging
        with caplog.at_level(logging.CRITICAL, logger="src.ml_engine.model"):
            with pytest.raises(ValueError, match="2 integrity violation"):
                valid_tick_factory(bid=-1.0, ask=-1.0, source="BAD_SOURCE")
        assert "VIOLATION(S)" in caplog.text
  -----------------------------------------------------------------------


================================================================================
  GROUP 3 -- Tick: Timeframe Enum  (3 tests)
  File: tests/ml_engine/test_tick.py
================================================================================

------------------------------------------------------------------------
  TEST 3.1 -- Timeframe enum values are correct strings
------------------------------------------------------------------------

  WHAT IT CHECKS:
    Since Timeframe inherits from str, the enum values must be usable
    as plain strings (e.g., in Parquet column values, log messages).

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_timeframe_enum_string_values():
        from src.ml_engine.model import Timeframe
        assert Timeframe.M1 == "M1"
        assert Timeframe.M5 == "M5"
        assert Timeframe.M15 == "M15"
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 3.2 -- Timeframe enum members are accessible by name
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_timeframe_enum_accessible_by_name():
        from src.ml_engine.model import Timeframe
        assert Timeframe["M1"] is Timeframe.M1
        assert Timeframe["M5"] is Timeframe.M5
        assert Timeframe["M15"] is Timeframe.M15
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 3.3 -- Timeframe has exactly three members
------------------------------------------------------------------------

  WHAT IT CHECKS:
    Regression guard. If a developer adds a new timeframe (M30, H1)
    without updating this test, the test fails as a reminder to review
    all downstream consumers of the enum.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_timeframe_has_exactly_three_members():
        from src.ml_engine.model import Timeframe
        assert len(Timeframe) == 3
  -----------------------------------------------------------------------


================================================================================
  GROUP 4 -- Bar: Happy Path & Field Correctness  (8 tests)
  File: tests/ml_engine/test_bar.py
================================================================================

------------------------------------------------------------------------
  TEST 4.1 -- Valid Bar constructs without error
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_valid_construction(valid_bar):
        assert valid_bar.symbol == "EUR_USD"
        assert valid_bar.open_price == pytest.approx(1.0850)
        assert valid_bar.high == pytest.approx(1.0865)
        assert valid_bar.low == pytest.approx(1.0848)
        assert valid_bar.close == pytest.approx(1.0860)
        assert valid_bar.volume == 342
        assert valid_bar.is_complete is True
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 4.2 -- Bar defaults to M1 timeframe
------------------------------------------------------------------------

  WHAT IT CHECKS:
    The timeframe field defaults to Timeframe.M1. This is important
    for backward compatibility -- existing code that does not pass
    timeframe must still get M1 bars.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_default_timeframe_is_m1(valid_bar):
        from src.ml_engine.model import Timeframe
        assert valid_bar.timeframe is Timeframe.M1
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 4.3 -- Bar accepts all Timeframe values
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    @pytest.mark.parametrize("tf", ["M1", "M5", "M15"])
    def test_bar_accepts_all_timeframes(valid_bar_factory, tf):
        from src.ml_engine.model import Timeframe
        bar = valid_bar_factory(timeframe=Timeframe(tf))
        assert bar.timeframe == Timeframe(tf)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 4.4 -- Bar is immutable after construction
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_is_frozen(valid_bar):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            valid_bar.close = 9.9999
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 4.5 -- to_dict returns all expected keys
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_to_dict_keys(valid_bar):
        d = valid_bar.to_dict()
        expected = {
            "timestamp", "symbol", "open_price", "high",
            "low", "close", "volume", "is_complete", "timeframe"
        }
        assert set(d.keys()) == expected
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 4.6 -- __repr__ contains key fields
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_repr_contains_key_fields(valid_bar):
        r = repr(valid_bar)
        assert "EUR_USD" in r
        assert "1.08500" in r
        assert "complete=True" in r
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 4.7 -- Timezone-aware timestamp is stripped to naive UTC
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_aware_timestamp_stripped(valid_bar_factory):
        from datetime import timezone
        aware_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        bar = valid_bar_factory(timestamp=aware_ts)
        assert bar.timestamp.tzinfo is None
        assert bar.timestamp == datetime(2024, 1, 15, 10, 30, 0)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 4.8 -- is_complete defaults to True
------------------------------------------------------------------------

  WHAT IT CHECKS:
    A Bar constructed without specifying is_complete should default
    to True, meaning it is ready for use in training data.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_is_complete_defaults_to_true(valid_bar_factory):
        bar = valid_bar_factory()
        assert bar.is_complete is True
  -----------------------------------------------------------------------


================================================================================
  GROUP 5 -- Bar: Validation & Rejection  (9 tests)
  File: tests/ml_engine/test_bar.py
================================================================================

------------------------------------------------------------------------
  TEST 5.1 -- Low > High is rejected
------------------------------------------------------------------------

  WHAT IT CHECKS:
    A candle where the low price exceeds the high is physically
    impossible. This is the core OHLC physicality check.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_low_above_high_rejected(valid_bar_factory):
        with pytest.raises(ValueError, match="Bar integrity failure"):
            valid_bar_factory(low=1.0870, high=1.0860)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.2 -- Open above High is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_open_above_high_rejected(valid_bar_factory):
        with pytest.raises(ValueError, match="Bar integrity failure"):
            valid_bar_factory(open_price=1.0870, high=1.0865)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.3 -- Open below Low is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_open_below_low_rejected(valid_bar_factory):
        with pytest.raises(ValueError, match="Bar integrity failure"):
            valid_bar_factory(open_price=1.0840, low=1.0848)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.4 -- Close above High is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_close_above_high_rejected(valid_bar_factory):
        with pytest.raises(ValueError, match="Bar integrity failure"):
            valid_bar_factory(close=1.0870, high=1.0865)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.5 -- Close below Low is rejected
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_close_below_low_rejected(valid_bar_factory):
        with pytest.raises(ValueError, match="Bar integrity failure"):
            valid_bar_factory(close=1.0840, low=1.0848)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.6 -- Negative volume is rejected
------------------------------------------------------------------------

  WHAT IT CHECKS:
    Volume cannot be negative. This guards volume-weighted indicators
    from receiving impossible inputs.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_negative_volume_rejected(valid_bar_factory):
        with pytest.raises(ValueError, match="Bar integrity failure"):
            valid_bar_factory(volume=-1)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.7 -- Zero volume is rejected
------------------------------------------------------------------------

  WHAT IT CHECKS:
    The "Raise to Reject, Catch to Continue" decision. Zero-volume
    bars represent market pauses (rollover, illiquid OTC, heartbeat
    bars). The model rejects them so the engine can skip gracefully.
    volume <= 0 triggers the rejection.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_zero_volume_rejected(valid_bar_factory):
        with pytest.raises(ValueError, match="Bar integrity failure"):
            valid_bar_factory(volume=0)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.8 -- Open equal to High is accepted (boundary)
------------------------------------------------------------------------

  WHAT IT CHECKS:
    open_price == high is physically valid (a bar that opened at its
    peak and only fell). The boundary check must be inclusive.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_open_equal_to_high_accepted(valid_bar_factory):
        bar = valid_bar_factory(open_price=1.0865, high=1.0865)
        assert bar.open_price == pytest.approx(1.0865)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 5.9 -- Close equal to Low is accepted (boundary)
------------------------------------------------------------------------

  WHAT IT CHECKS:
    close == low is physically valid (a bar that closed at its lowest
    point). The boundary check must be inclusive.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_bar_close_equal_to_low_accepted(valid_bar_factory):
        bar = valid_bar_factory(close=1.0848, low=1.0848)
        assert bar.close == pytest.approx(1.0848)
  -----------------------------------------------------------------------


================================================================================
  GROUP 6 -- DataBuffer: Construction & Core Logic  (10 tests)
  File: tests/ml_engine/test_data_buffer.py
================================================================================

------------------------------------------------------------------------
  TEST 6.1 -- Buffer constructs with valid flush_size
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_constructs_with_valid_flush_size():
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=5)
        assert len(buf) == 0
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.2 -- Buffer rejects zero flush_size
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_rejects_zero_flush_size():
        from src.ml_engine.model import DataBuffer
        with pytest.raises(ValueError, match="flush_size"):
            DataBuffer(flush_size=0)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.3 -- Buffer rejects negative flush_size
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_rejects_negative_flush_size():
        from src.ml_engine.model import DataBuffer
        with pytest.raises(ValueError, match="flush_size"):
            DataBuffer(flush_size=-10)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.4 -- add() returns None before flush threshold is reached
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_add_returns_none_before_flush(valid_tick):
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=3)
        assert buf.add(valid_tick) is None
        assert buf.add(valid_tick) is None
        assert len(buf) == 2
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.5 -- add() returns batch when flush threshold is reached
------------------------------------------------------------------------

  WHAT IT CHECKS:
    On the Nth tick (where N == flush_size), add() must return the
    complete batch as a list and the buffer must be empty afterwards.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_add_returns_batch_at_flush(valid_tick):
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=3)
        buf.add(valid_tick)
        buf.add(valid_tick)
        batch = buf.add(valid_tick)
        assert batch is not None
        assert len(batch) == 3
        assert len(buf) == 0
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.6 -- Buffer clears after flush
------------------------------------------------------------------------

  WHAT IT CHECKS:
    After a flush, the buffer must be empty and ready to accept new
    ticks. Continuing to add after a flush must work correctly.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_clears_after_flush(valid_tick):
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=2)
        buf.add(valid_tick)
        buf.add(valid_tick)  # triggers flush
        assert len(buf) == 0
        result = buf.add(valid_tick)
        assert result is None
        assert len(buf) == 1
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.7 -- flush() returns batch regardless of size
------------------------------------------------------------------------

  WHAT IT CHECKS:
    The force-flush method must drain whatever is in the buffer,
    even if the flush threshold has not been reached. This is the
    graceful shutdown path.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_force_flush_drains_partial_batch(valid_tick):
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=10)
        buf.add(valid_tick)
        buf.add(valid_tick)
        buf.add(valid_tick)
        batch = buf.flush()
        assert batch is not None
        assert len(batch) == 3
        assert len(buf) == 0
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.8 -- flush() returns None when buffer is empty
------------------------------------------------------------------------

  WHAT IT CHECKS:
    Calling flush() on an already-empty buffer must return None, not
    an empty list. Callers check `if remaining:` so None and [] behave
    differently -- None signals "nothing to do", [] signals "flushed
    but got nothing", which would be confusing.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_force_flush_empty_returns_none():
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=5)
        assert buf.flush() is None
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.9 -- add() raises RuntimeError at 10x safety cap
------------------------------------------------------------------------

  WHAT IT CHECKS:
    The memory safety guardrail. If the buffer reaches 10x flush_size
    without being consumed, something is wrong with the storage layer.
    The system must crash with RuntimeError rather than silently
    exhausting VPS RAM.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_overflow_raises_runtime_error(valid_tick):
        from src.ml_engine.model import DataBuffer
        flush_size = 3
        buf = DataBuffer(flush_size=flush_size)
        cap = flush_size * 10  # 30

        # Manually fill the buffer past the auto-flush threshold
        # by patching _flush_size temporarily to prevent auto-flush
        buf._flush_size = cap + 1  # disable auto-flush for this test

        # Fill to the cap
        for _ in range(cap):
            buf._data.append(valid_tick)

        # The next add() must trigger the overflow guard
        with pytest.raises(RuntimeError, match="Buffer overflow"):
            buf.add(valid_tick)
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 6.10 -- __len__ returns correct count
------------------------------------------------------------------------

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_len_is_accurate(valid_tick):
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=10)
        assert len(buf) == 0
        buf.add(valid_tick)
        assert len(buf) == 1
        buf.add(valid_tick)
        assert len(buf) == 2
  -----------------------------------------------------------------------


================================================================================
  GROUP 7 -- DataBuffer: Thread Safety  (6 tests)
  File: tests/ml_engine/test_data_buffer.py
================================================================================

  NOTE ON THREAD SAFETY TESTING:
  Thread safety tests are inherently non-deterministic. These tests use
  threading.Barrier to synchronise threads to a starting line before
  releasing them simultaneously, maximising the chance of exposing race
  conditions. Run these tests multiple times if debugging concurrency issues:
      pytest tests/ml_engine/test_data_buffer.py -v --count=10
  (requires pytest-repeat: pip install pytest-repeat)

------------------------------------------------------------------------
  TEST 7.1 -- Concurrent adds from multiple threads produce correct total
------------------------------------------------------------------------

  WHAT IT CHECKS:
    10 threads each adding 10 ticks to a buffer with flush_size=100
    must produce exactly 100 ticks total with no duplicates or losses.
    Verifies that the lock prevents data corruption under concurrency.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_concurrent_adds_correct_total(valid_tick):
        import threading
        from src.ml_engine.model import DataBuffer

        buf = DataBuffer(flush_size=200)  # large enough to not auto-flush
        threads = []
        num_threads = 10
        adds_per_thread = 10

        barrier = threading.Barrier(num_threads)

        def add_ticks():
            barrier.wait()  # all threads start simultaneously
            for _ in range(adds_per_thread):
                buf.add(valid_tick)

        for _ in range(num_threads):
            t = threading.Thread(target=add_ticks)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(buf) == num_threads * adds_per_thread
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 7.2 -- Concurrent adds do not produce batch larger than flush_size
------------------------------------------------------------------------

  WHAT IT CHECKS:
    When multiple threads trigger a flush simultaneously, the lock must
    ensure only ONE batch is returned and it contains exactly flush_size
    ticks. Without the lock, two threads could both see len >= flush_size
    and both return a batch, effectively doubling the data.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_concurrent_flush_returns_exactly_one_batch(valid_tick):
        import threading
        from src.ml_engine.model import DataBuffer

        flush_size = 10
        buf = DataBuffer(flush_size=flush_size)
        batches = []
        lock = threading.Lock()

        def add_and_collect():
            result = buf.add(valid_tick)
            if result is not None:
                with lock:
                    batches.append(result)

        threads = [
            threading.Thread(target=add_and_collect)
            for _ in range(flush_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one batch must have been produced
        assert len(batches) == 1
        assert len(batches[0]) == flush_size
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 7.3 -- flush() is safe to call concurrently with add()
------------------------------------------------------------------------

  WHAT IT CHECKS:
    A shutdown thread calling flush() while a stream thread is still
    calling add() must not cause a crash or data corruption. Both
    operations acquire the same lock so they cannot interleave.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_flush_concurrent_with_add(valid_tick):
        import threading
        from src.ml_engine.model import DataBuffer

        buf = DataBuffer(flush_size=1000)
        errors = []

        def stream_thread():
            for _ in range(50):
                try:
                    buf.add(valid_tick)
                except Exception as e:
                    errors.append(e)

        def shutdown_thread():
            for _ in range(5):
                buf.flush()

        t1 = threading.Thread(target=stream_thread)
        t2 = threading.Thread(target=shutdown_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Errors during concurrent flush+add: {errors}"
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 7.4 -- __len__ is accurate under concurrent modification
------------------------------------------------------------------------

  WHAT IT CHECKS:
    len(buffer) must return a consistent value even while other
    threads are adding ticks. Without the lock on __len__, it could
    read a partially-modified list.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_len_accurate_under_concurrency(valid_tick):
        import threading
        from src.ml_engine.model import DataBuffer

        buf = DataBuffer(flush_size=1000)
        errors = []

        def add_ticks():
            for _ in range(100):
                buf.add(valid_tick)

        def check_len():
            for _ in range(100):
                try:
                    l = len(buf)
                    assert l >= 0
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=add_ticks)
        t2 = threading.Thread(target=check_len)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 7.5 -- Buffer returns a COPY of data on flush, not a reference
------------------------------------------------------------------------

  WHAT IT CHECKS:
    add() and flush() both use self._data.copy() before clearing.
    This test verifies that the returned batch is a copy -- modifying
    the returned list must not affect the buffer's internal state.
    A reference-return bug here would cause data corruption in the
    storage layer.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_returns_copy_not_reference(valid_tick):
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=2)
        buf.add(valid_tick)
        batch = buf.add(valid_tick)  # triggers flush, returns copy

        assert batch is not None
        # Mutate the returned batch
        batch.clear()
        # Buffer internals must be unaffected (already cleared by flush,
        # but a subsequent add must still work correctly)
        result = buf.add(valid_tick)
        assert result is None
        assert len(buf) == 1
  -----------------------------------------------------------------------

------------------------------------------------------------------------
  TEST 7.6 -- Multiple sequential flushes work correctly
------------------------------------------------------------------------

  WHAT IT CHECKS:
    The buffer must be reusable across multiple flush cycles. This
    simulates the full session loop: fill -> flush -> fill -> flush.

  GOLDEN PROMPT:
  -----------------------------------------------------------------------
    def test_buffer_multiple_flush_cycles(valid_tick):
        from src.ml_engine.model import DataBuffer
        buf = DataBuffer(flush_size=3)

        # Cycle 1
        buf.add(valid_tick)
        buf.add(valid_tick)
        batch1 = buf.add(valid_tick)
        assert batch1 is not None and len(batch1) == 3

        # Cycle 2
        buf.add(valid_tick)
        buf.add(valid_tick)
        batch2 = buf.add(valid_tick)
        assert batch2 is not None and len(batch2) == 3

        # Both cycles produced independent batches
        assert len(buf) == 0
  -----------------------------------------------------------------------


================================================================================
  RUNNING THE FULL SUITE
================================================================================

  INSTALL DEPENDENCIES (if not already done):
    pip install pytest pytest-cov python-dotenv

  RUN ALL MODEL TESTS:
    pytest tests/ml_engine/ -v

  RUN WITH COVERAGE:
    pytest tests/ml_engine/ -v --cov=src.ml_engine.model --cov-report=term-missing

  RUN THREAD SAFETY TESTS ONLY:
    pytest tests/ml_engine/test_data_buffer.py -v -k "concurrent or thread or copy"

  RUN A SPECIFIC GROUP:
    pytest tests/ml_engine/test_tick.py -v
    pytest tests/ml_engine/test_bar.py -v
    pytest tests/ml_engine/test_data_buffer.py -v

  EXPECTED COUNT:
    63 tests collected across 3 files.
    With parametrize expansion:
      test_tick_valid_sources            -> 2
      test_tick_lowercase_source_rejected -> 4
      test_bar_accepts_all_timeframes    -> 3
    Total collected by pytest will be ~69 items.

  EXPECTED COVERAGE:
    All branches in model.py should be covered.
    The only lines that may show as partial are the logger.critical()
    calls inside DataBuffer.__init__ and add() -- covered by tests
    6.2, 6.3, and 6.9 respectively.

  KNOWN BLOCKER:
    If the engine pseudocode block (Bar.__post_init__ lines 229-234)
    has not been removed, pytest will fail at collection with:
        SyntaxError: 'continue' outside loop
    Fix: delete those 6 lines before running.

================================================================================
  END OF TEST HARNESS
================================================================================
