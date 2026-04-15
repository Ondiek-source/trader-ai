"""
test_tick.py -- Tests for the Tick dataclass and Timeframe enum.

Covers:
    Group 1 : Tick happy path and field correctness
    Group 2 : Tick validation and rejection
    Group 3 : Timeframe enum correctness
"""

import pytest
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import FrozenInstanceError
from src.ml_engine.model import Tick, Timeframe


# ==============================================================================
# GROUP 1 -- Tick: Happy Path & Field Correctness
# ==============================================================================

def test_tick_valid_construction(valid_tick):
    assert valid_tick.symbol == "EUR_USD"
    assert valid_tick.bid == pytest.approx(1.08500)
    assert valid_tick.ask == pytest.approx(1.08520)
    assert valid_tick.source == "TWELVE"


@pytest.mark.parametrize("source", ["TWELVE", "QUOTEX"])
def test_tick_valid_sources(valid_tick_factory, source):
    tick = valid_tick_factory(source=source)
    assert tick.source == source


def test_tick_is_frozen(valid_tick):
    with pytest.raises(FrozenInstanceError):
        valid_tick.bid = 9.9999


def test_tick_mid_price(valid_tick):
    # bid=1.08500, ask=1.08520 -> mid = 1.08510
    assert valid_tick.mid_price == pytest.approx(1.08510)


def test_tick_to_dict_keys(valid_tick):
    d = valid_tick.to_dict()
    assert set(d.keys()) == {"timestamp", "symbol", "bid", "ask", "source"}


def test_tick_repr_contains_key_fields(valid_tick):
    r = repr(valid_tick)
    assert "EUR_USD" in r
    assert "TWELVE" in r


def test_tick_aware_timestamp_stripped(valid_tick_factory):
    aware_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    tick = valid_tick_factory(timestamp=aware_ts)
    assert tick.timestamp.tzinfo is None
    assert tick.timestamp == datetime(2024, 1, 15, 10, 30, 0)


def test_tick_non_utc_aware_timestamp_rejected(valid_tick_factory):
    # US/Eastern offset (-5h) — silently stripping this would corrupt the timestamp.
    eastern = timezone(timedelta(hours=-5))
    non_utc_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=eastern)
    with pytest.raises(ValueError, match="Non-UTC timezone rejected"):
        valid_tick_factory(timestamp=non_utc_ts)


def test_tick_utc_aware_timestamp_accepted(valid_tick_factory):
    # UTC-aware datetimes are accepted and stored as naive UTC (existing behaviour).
    utc_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    tick = valid_tick_factory(timestamp=utc_ts)
    assert tick.timestamp.tzinfo is None
    assert tick.timestamp == datetime(2024, 1, 15, 10, 30, 0)


# ==============================================================================
# GROUP 2 -- Tick: Validation & Rejection
# ==============================================================================

def test_tick_zero_bid_rejected(valid_tick_factory):
    with pytest.raises(ValueError, match="integrity violation"):
        valid_tick_factory(bid=0.0)


def test_tick_negative_bid_rejected(valid_tick_factory):
    with pytest.raises(ValueError, match="integrity violation"):
        valid_tick_factory(bid=-1.0850)


def test_tick_zero_ask_rejected(valid_tick_factory):
    with pytest.raises(ValueError, match="integrity violation"):
        valid_tick_factory(ask=0.0, bid=0.0)


def test_tick_inverted_spread_rejected(valid_tick_factory):
    with pytest.raises(ValueError, match="integrity violation"):
        valid_tick_factory(bid=1.0860, ask=1.0850)


def test_tick_equal_bid_ask_accepted(valid_tick_factory):
    tick = valid_tick_factory(bid=1.0850, ask=1.0850)
    assert tick.mid_price == pytest.approx(1.0850)


def test_tick_unknown_source_rejected(valid_tick_factory):
    with pytest.raises(ValueError, match="integrity violation"):
        valid_tick_factory(source="BLOOMBERG")


@pytest.mark.parametrize("source", ["twelve", "quotex", "Twelve", "Quotex"])
def test_tick_lowercase_source_rejected(valid_tick_factory, source):
    with pytest.raises(ValueError, match="integrity violation"):
        valid_tick_factory(source=source)


def test_tick_multiple_violations_reported(valid_tick_factory, caplog):
    with caplog.at_level(logging.CRITICAL, logger="src.ml_engine.model"):
        with pytest.raises(ValueError, match="2 integrity violation"):
            valid_tick_factory(bid=-1.0, ask=-1.0, source="BAD_SOURCE")
    assert "VIOLATION(S)" in caplog.text


# ==============================================================================
# GROUP 3 -- Timeframe Enum
# ==============================================================================

def test_timeframe_enum_string_values():
    assert Timeframe.M1 == "M1"
    assert Timeframe.M5 == "M5"
    assert Timeframe.M15 == "M15"


def test_timeframe_enum_accessible_by_name():
    assert Timeframe["M1"] is Timeframe.M1
    assert Timeframe["M5"] is Timeframe.M5
    assert Timeframe["M15"] is Timeframe.M15


def test_timeframe_has_exactly_three_members():
    assert len(Timeframe) == 3
