"""
test_context.py — Tests for _compute_context().

Group 8: Session flags, cyclical encoding bounds, placeholder value,
         zero-fill fallback when called with a non-DatetimeIndex.

NOTE: _compute_context is tested by calling it directly for the
non-DatetimeIndex branch, because transform() always coerces the index
to DatetimeIndex before dispatching to group methods.
"""

import numpy as np
import pytest
import pandas as pd

from src.ml_engine.features import (
    FEATURE_SET_BINARY_OPTIONS_AI,
    FeatureEngineer,
    _MINUTES_TO_NEWS_PLACEHOLDER,
)
from .conftest import make_ohlcv


# ── T54 ───────────────────────────────────────────────────────────────────────


def test_session_london_active_at_10_utc(engineer):
    """Hour 10 UTC is inside the London session window (07:00–16:00)."""
    bars = make_ohlcv(n=50, start="2024-01-02 10:00")
    result = engineer.transform(bars)
    assert (result["SESSION_LONDON"] == 1).all()


# ── T55 ───────────────────────────────────────────────────────────────────────


def test_session_london_inactive_at_22_utc(engineer):
    """Hour 22 UTC is outside the London session window."""
    bars = make_ohlcv(n=50, start="2024-01-02 22:00")
    result = engineer.transform(bars)
    assert (result["SESSION_LONDON"] == 0).all()


# ── T56 ───────────────────────────────────────────────────────────────────────


def test_session_overlap_active_at_14_utc(engineer):
    """Hour 14 UTC is inside the London/NY overlap window (12:00–16:00)."""
    bars = make_ohlcv(n=50, start="2024-01-02 14:00")
    result = engineer.transform(bars)
    assert (result["SESSION_OVERLAP_LONDON_NY"] == 1).all()


# ── T57 ───────────────────────────────────────────────────────────────────────


def test_time_sine_cosine_within_bounds(full_feature_df):
    assert (full_feature_df["TIME_SINE"] >= -1).all()
    assert (full_feature_df["TIME_SINE"] <= 1).all()
    assert (full_feature_df["TIME_COSINE"] >= -1).all()
    assert (full_feature_df["TIME_COSINE"] <= 1).all()


# ── T58 ───────────────────────────────────────────────────────────────────────


def test_hour_of_day_normalized_bounds(full_feature_df):
    col = full_feature_df["HOUR_OF_DAY_NORMALIZED"]
    assert (col >= 0).all()
    assert (col <= 1).all()


# ── T59 ───────────────────────────────────────────────────────────────────────


def test_minutes_to_news_is_placeholder_value(full_feature_df):
    assert (full_feature_df["MINUTES_TO_NEWS"] == _MINUTES_TO_NEWS_PLACEHOLDER).all()


# ── T60 ───────────────────────────────────────────────────────────────────────


def test_context_zeros_when_non_datetime_index(engineer):
    """
    _compute_context() is called directly with a RangeIndex.
    All CONTEXT columns should be zero-filled.
    """
    fe = pd.DataFrame(
        {"open": [1.1], "high": [1.101], "low": [1.099], "close": [1.1002], "volume": [1000.0]},
        index=[0],  # integer RangeIndex — not DatetimeIndex
    )
    result = engineer._compute_context(fe)
    for col in FEATURE_SET_BINARY_OPTIONS_AI["CONTEXT"]:
        assert col in result.columns
        assert result[col].iloc[0] == 0.0
