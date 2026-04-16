"""
test_bar.py -- Tests for the Bar dataclass.

Covers:
    Group 4 : Bar happy path and field correctness
    Group 5 : Bar validation and rejection
"""

import pytest
from datetime import datetime, timezone, timedelta
from dataclasses import FrozenInstanceError
from src.ml_engine.model import Bar, Timeframe


# ==============================================================================
# GROUP 4 -- Bar: Happy Path & Field Correctness
# ==============================================================================


def test_bar_valid_construction(valid_bar):
    assert valid_bar.symbol == "EUR_USD"
    assert valid_bar.open == pytest.approx(1.0850)
    assert valid_bar.high == pytest.approx(1.0865)
    assert valid_bar.low == pytest.approx(1.0848)
    assert valid_bar.close == pytest.approx(1.0860)
    assert valid_bar.volume == 342
    assert valid_bar.is_complete is True


def test_bar_default_timeframe_is_m1(valid_bar):
    assert valid_bar.timeframe is Timeframe.M1


@pytest.mark.parametrize("tf", ["M1", "M5", "M15"])
def test_bar_accepts_all_timeframes(valid_bar_factory, tf):
    bar = valid_bar_factory(timeframe=Timeframe(tf))
    assert bar.timeframe == Timeframe(tf)


def test_bar_is_frozen(valid_bar):
    with pytest.raises(FrozenInstanceError):
        valid_bar.close = 9.9999


def test_bar_to_dict_keys(valid_bar):
    d = valid_bar.to_dict()
    expected = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_complete",
        "timeframe",
    }
    assert set(d.keys()) == expected


def test_bar_repr_contains_key_fields(valid_bar):
    r = repr(valid_bar)
    assert "EUR_USD" in r
    assert "complete=True" in r


def test_bar_aware_timestamp_stripped(valid_bar_factory):
    aware_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    bar = valid_bar_factory(timestamp=aware_ts)
    assert bar.timestamp.tzinfo is None
    assert bar.timestamp == datetime(2024, 1, 15, 10, 30, 0)


def test_bar_non_utc_aware_timestamp_rejected(valid_bar_factory):
    # US/Eastern offset (-5h) — silently stripping this would corrupt the timestamp.
    eastern = timezone(timedelta(hours=-5))
    non_utc_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=eastern)
    with pytest.raises(ValueError, match="Non-UTC timezone rejected"):
        valid_bar_factory(timestamp=non_utc_ts)


def test_bar_utc_aware_timestamp_accepted(valid_bar_factory):
    # UTC-aware datetimes are accepted and stored as naive UTC (existing behaviour).
    utc_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    bar = valid_bar_factory(timestamp=utc_ts)
    assert bar.timestamp.tzinfo is None
    assert bar.timestamp == datetime(2024, 1, 15, 10, 30, 0)


def test_bar_is_complete_defaults_to_true(valid_bar_factory):
    bar = valid_bar_factory()
    assert bar.is_complete is True


# ==============================================================================
# GROUP 5 -- Bar: Validation & Rejection
# ==============================================================================


def test_bar_low_above_high_rejected(valid_bar_factory):
    with pytest.raises(ValueError, match="Bar integrity failure"):
        valid_bar_factory(low=1.0870, high=1.0860)


def test_bar_open_above_high_rejected(valid_bar_factory):
    with pytest.raises(ValueError, match="Bar integrity failure"):
        valid_bar_factory(open=1.0870, high=1.0865)


def test_bar_open_below_low_rejected(valid_bar_factory):
    with pytest.raises(ValueError, match="Bar integrity failure"):
        valid_bar_factory(open=1.0840, low=1.0848)


def test_bar_close_above_high_rejected(valid_bar_factory):
    with pytest.raises(ValueError, match="Bar integrity failure"):
        valid_bar_factory(close=1.0870, high=1.0865)


def test_bar_close_below_low_rejected(valid_bar_factory):
    with pytest.raises(ValueError, match="Bar integrity failure"):
        valid_bar_factory(close=1.0840, low=1.0848)


def test_bar_negative_volume_rejected(valid_bar_factory):
    with pytest.raises(ValueError, match="Bar integrity failure"):
        valid_bar_factory(volume=-1)


def test_bar_zero_volume_accepted(valid_bar_factory):
    bar = valid_bar_factory(volume=0)
    assert bar.volume == 0


def test_bar_open_equal_to_high_accepted(valid_bar_factory):
    bar = valid_bar_factory(open=1.0865, high=1.0865)
    assert bar.open == pytest.approx(1.0865)


def test_bar_close_equal_to_low_accepted(valid_bar_factory):
    bar = valid_bar_factory(close=1.0848, low=1.0848)
    assert bar.close == pytest.approx(1.0848)
