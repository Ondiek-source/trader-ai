"""
test_momentum.py — Tests for _compute_momentum().

Group 5: RSI bounds, MACD identity, log-return formula, finite values.
"""

import numpy as np
import pytest


# ── T36 ───────────────────────────────────────────────────────────────────────


def test_rsi_bounds(full_feature_df):
    rsi = full_feature_df["RSI"]
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


# ── T37 ───────────────────────────────────────────────────────────────────────


def test_macd_hist_equals_value_minus_signal(full_feature_df):
    computed = full_feature_df["MACD_VALUE"] - full_feature_df["MACD_SIGNAL"]
    actual = full_feature_df["MACD_HIST"]
    assert computed.sub(actual).abs().max() < 1e-6


# ── T38 ───────────────────────────────────────────────────────────────────────


def test_roc_5_finite(full_feature_df):
    col = full_feature_df["ROC_5"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()


# ── T39 ───────────────────────────────────────────────────────────────────────


def test_return_1_is_log_return(full_feature_df):
    """RETURN_1 must equal log(close / prev_close) for each row."""
    import pandas as pd

    close = full_feature_df["close"]
    expected = np.log(close / close.shift(1))
    actual = full_feature_df["RETURN_1"]
    # Compare overlapping rows only (skip first row where expected is NaN)
    mask = expected.notna() & actual.notna()
    assert np.allclose(actual[mask].values, expected[mask].values, atol=1e-6)


# ── T40 ───────────────────────────────────────────────────────────────────────


def test_momentum_oscillator_finite(full_feature_df):
    col = full_feature_df["MOMENTUM_OSCILLATOR"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()


# ── T41 ───────────────────────────────────────────────────────────────────────


def test_cci_14_finite(full_feature_df):
    col = full_feature_df["CCI"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()
