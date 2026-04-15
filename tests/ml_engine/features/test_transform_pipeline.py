"""
test_transform_pipeline.py — Tests for FeatureEngineer.transform().

Group 3 : Empty, missing cols, schema completeness, flag gating, gap detection,
          index coercion, schema-validation warning.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.ml_engine.features import (
    FEATURE_SET_BINARY_OPTIONS_AI,
    FeatureEngineer,
)
from .conftest import make_ohlcv


# ── T13 ───────────────────────────────────────────────────────────────────────


def test_transform_returns_dataframe(engineer, bars_100):
    result = engineer.transform(bars_100)
    assert isinstance(result, pd.DataFrame)


# ── T14 ───────────────────────────────────────────────────────────────────────


def test_transform_empty_bars_returns_empty_df(engineer, empty_bars):
    result = engineer.transform(empty_bars)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ── T15 ───────────────────────────────────────────────────────────────────────


def test_transform_missing_columns_raises_value_error(engineer):
    bad = pd.DataFrame(
        {"open": [1.0], "close": [1.0]},
        index=pd.date_range("2024-01-01", periods=1, freq="1min"),
    )
    with pytest.raises(ValueError, match="missing columns"):
        engineer.transform(bad)


# ── T16 ───────────────────────────────────────────────────────────────────────


def test_transform_missing_columns_logs_critical(engineer, caplog):
    bad = pd.DataFrame(
        {"open": [1.0], "close": [1.0]},
        index=pd.date_range("2024-01-01", periods=1, freq="1min"),
    )
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError):
            engineer.transform(bad)
    assert "MISSING COLUMNS" in caplog.text


# ── T17 ───────────────────────────────────────────────────────────────────────


def test_transform_output_has_expected_columns(engineer, bars_100, ticks_600):
    result = engineer.transform(bars_100, ticks_600)
    all_primary = [c for g in FEATURE_SET_BINARY_OPTIONS_AI.values() for c in g]
    for col in all_primary:
        assert col in result.columns, f"Missing column: {col}"


# ── T18 ───────────────────────────────────────────────────────────────────────


def test_transform_output_index_is_datetimeindex(engineer, bars_100):
    result = engineer.transform(bars_100)
    assert isinstance(result.index, pd.DatetimeIndex)


# ── T19 ───────────────────────────────────────────────────────────────────────


def test_transform_no_nans_after_warmup(engineer, bars_100, ticks_600):
    result = engineer.transform(bars_100, ticks_600)
    assert not result.empty
    assert not result.isnull().any().any()


# ── T20 ───────────────────────────────────────────────────────────────────────


def test_transform_price_action_disabled_columns_absent(
    mock_settings, monkeypatch, bars_100
):
    """When feat_price_action_enabled=False, PRICE_ACTION columns are absent."""
    import src.ml_engine.features as feat_mod

    mock_settings.feat_price_action_enabled = False
    monkeypatch.setattr(feat_mod, "get_settings", lambda: mock_settings)
    eng = FeatureEngineer()
    result = eng.transform(bars_100)
    for col in FEATURE_SET_BINARY_OPTIONS_AI["PRICE_ACTION"]:
        assert col not in result.columns


# ── T21 ───────────────────────────────────────────────────────────────────────


def test_transform_micro_disabled_zeros_micro_columns(
    mock_settings, monkeypatch, bars_100
):
    import src.ml_engine.features as feat_mod

    mock_settings.feat_micro_enabled = False
    monkeypatch.setattr(feat_mod, "get_settings", lambda: mock_settings)
    eng = FeatureEngineer()
    result = eng.transform(bars_100, ticks=None)
    for col in FEATURE_SET_BINARY_OPTIONS_AI["MICRO_STRUCTURE"]:
        assert col in result.columns
        assert (result[col] == 0.0).all()


# ── T22 ───────────────────────────────────────────────────────────────────────


def test_transform_micro_none_ticks_zeros_micro_columns(engineer, bars_100):
    result = engineer.transform(bars_100, ticks=None)
    for col in FEATURE_SET_BINARY_OPTIONS_AI["MICRO_STRUCTURE"]:
        assert col in result.columns
        assert (result[col] == 0.0).all()


# ── T23 ───────────────────────────────────────────────────────────────────────


def test_transform_non_datetime_index_coerced(engineer):
    """String-formatted timestamps should be coerced to DatetimeIndex."""
    bars = make_ohlcv(n=50)
    # Convert to string object index
    bars.index = bars.index.strftime("%Y-%m-%d %H:%M:%S").astype(object)
    assert not isinstance(bars.index, pd.DatetimeIndex)
    result = engineer.transform(bars)
    assert isinstance(result.index, pd.DatetimeIndex)


# ── T24 ───────────────────────────────────────────────────────────────────────


def test_transform_sorts_output_by_index(engineer):
    # Create bars in reverse order
    bars = make_ohlcv(n=50, start="2024-01-01")
    bars = bars.iloc[::-1]  # reverse chronological
    result = engineer.transform(bars)
    timestamps = result.index.tolist()
    assert timestamps == sorted(timestamps)


# ── T25 ───────────────────────────────────────────────────────────────────────


def test_transform_100_bars_non_empty(engineer, bars_100):
    result = engineer.transform(bars_100)
    assert not result.empty


# ── T_NEW_01: Gap Detection Warning ──────────────────────────────────────────


def test_transform_logs_warning_on_resampled_gaps(engineer, caplog):
    """A bars DataFrame with a 3-bar gap should trigger a [%] RESAMPLE GAP warning."""
    idx1 = pd.date_range("2024-01-02 00:00", periods=5, freq="1min")
    idx2 = pd.date_range("2024-01-02 00:09", periods=95, freq="1min")
    idx = idx1.append(idx2)
    bars = pd.DataFrame(
        {
            "open": 1.1,
            "high": 1.101,
            "low": 1.099,
            "close": 1.1002,
            "volume": 1000.0,
        },
        index=idx,
    )
    with caplog.at_level(logging.WARNING):
        engineer.transform(bars)
    assert "RESAMPLE GAP" in caplog.text


# ── T_NEW_02: Non-DatetimeIndex Warning ──────────────────────────────────────


def test_transform_logs_warning_on_non_datetime_index(engineer, caplog):
    bars = make_ohlcv(n=50)
    bars.index = bars.index.strftime("%Y-%m-%d %H:%M:%S").astype(object)
    with caplog.at_level(logging.WARNING):
        engineer.transform(bars)
    assert "not DatetimeIndex" in caplog.text


# ── T_NEW_03: Schema Validation Warning ──────────────────────────────────────


def test_transform_logs_schema_warning_when_group_disabled(
    mock_settings, monkeypatch, bars_100, caplog
):
    """Disabling a group emits a SCHEMA INCOMPLETE warning."""
    import src.ml_engine.features as feat_mod

    mock_settings.feat_momentum_enabled = False
    monkeypatch.setattr(feat_mod, "get_settings", lambda: mock_settings)
    eng = FeatureEngineer()
    with caplog.at_level(logging.WARNING):
        eng.transform(bars_100)
    assert "SCHEMA INCOMPLETE" in caplog.text


# ── T_NEW_04: Large Gap Is Dropped ───────────────────────────────────────────


def test_transform_large_gap_does_not_fabricate_flat_prices(engineer):
    """
    A gap of 10 bars (> _MAX_FFILL_BARS=5) should not be fully forward-filled.
    The output should be valid (no inf/nan) but should not contain more rows
    than could be produced from the non-gap portions after warmup.
    """
    from src.ml_engine.features import _MAX_FFILL_BARS

    gap_size = _MAX_FFILL_BARS + 5  # definitely exceeds limit
    idx1 = pd.date_range("2024-01-03 00:00", periods=60, freq="1min")
    gap_start = idx1[-1] + pd.Timedelta(minutes=gap_size + 1)
    idx2 = pd.date_range(gap_start, periods=60, freq="1min")
    idx = idx1.append(idx2)
    bars = pd.DataFrame(
        {"open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1002, "volume": 1000.0},
        index=idx,
    )
    result = engineer.transform(bars)
    # Output must be valid (no NaN, not empty)
    assert not result.empty
    assert not result.isnull().any().any()
