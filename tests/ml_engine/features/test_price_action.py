"""
test_price_action.py — Tests for _compute_price_action().

Group 4: Bounds, binary flags {0,1}, numeric validity post-warmup.
"""

import pytest


# ── T26 ───────────────────────────────────────────────────────────────────────


def test_body_to_range_ratio_bounds(full_feature_df):
    col = full_feature_df["BODY_TO_RANGE_RATIO"]
    assert (col >= 0).all()
    assert (col <= 1).all()


# ── T27 ───────────────────────────────────────────────────────────────────────


def test_doji_binary_is_zero_or_one(full_feature_df):
    assert set(full_feature_df["DOJI_BINARY"].unique()).issubset({0, 1})


# ── T28 ───────────────────────────────────────────────────────────────────────


def test_marubozu_binary_is_zero_or_one(full_feature_df):
    assert set(full_feature_df["MARUBOZU_BINARY"].unique()).issubset({0, 1})


# ── T29 ───────────────────────────────────────────────────────────────────────


def test_engulfing_binary_is_zero_or_one(full_feature_df):
    assert set(full_feature_df["ENGULFING_BINARY"].unique()).issubset({0, 1})


# ── T30 ───────────────────────────────────────────────────────────────────────


def test_pinbar_signal_is_zero_or_one(full_feature_df):
    assert set(full_feature_df["PINBAR_SIGNAL"].unique()).issubset({0, 1})


# ── T31 ───────────────────────────────────────────────────────────────────────


def test_two_bar_reversal_is_zero_or_one(full_feature_df):
    assert set(full_feature_df["TWO_BAR_REVERSAL"].unique()).issubset({0, 1})


# ── T32 ───────────────────────────────────────────────────────────────────────


def test_close_position_in_candle_bounds(full_feature_df):
    col = full_feature_df["CLOSE_POSITION_IN_CANDLE"]
    assert (col >= 0).all()
    assert (col <= 1).all()


# ── T33 ───────────────────────────────────────────────────────────────────────


def test_candle_position_in_day_bounds(full_feature_df):
    col = full_feature_df["CANDLE_POSITION_IN_DAY"]
    assert (col >= 0).all()
    assert (col <= 1).all()


# ── T34 ───────────────────────────────────────────────────────────────────────


def test_three_bar_slope_is_numeric(full_feature_df):
    import numpy as np

    col = full_feature_df["THREE_BAR_SLOPE"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()


# ── T35 ───────────────────────────────────────────────────────────────────────


def test_consecutive_bull_bars_non_negative(full_feature_df):
    assert (full_feature_df["CONSECUTIVE_BULL_BARS"] >= 0).all()
