"""
test_derived.py — Tests for _compute_derived().

Group 9: Finite values post-warmup, RVOL alias, SESSION_CONTEXT flag,
         zero-fill fallback when primary dependencies are absent.

_compute_derived is also tested by calling it directly to verify the
dep-absent branches without disabling full groups.
"""

import numpy as np
import pandas as pd
import pytest


# ── T61 ───────────────────────────────────────────────────────────────────────


def test_roc5_acceleration_finite_after_warmup(full_feature_df):
    col = full_feature_df["ROC_5_ACCELERATION"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()


# ── T62 ───────────────────────────────────────────────────────────────────────


def test_atr14_ratio_change_finite_after_warmup(full_feature_df):
    col = full_feature_df["ATR_RATIO_CHANGE"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()


# ── T63 ───────────────────────────────────────────────────────────────────────


def test_macd_hist_slope_finite(full_feature_df):
    col = full_feature_df["MACD_HIST_SLOPE"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()


# ── T64 ───────────────────────────────────────────────────────────────────────


def test_bb_width_ma20_non_negative(full_feature_df):
    assert (full_feature_df["BB_WIDTH_MA"] >= 0).all()


# ── T65 ───────────────────────────────────────────────────────────────────────


def test_rvol_alias_equals_relative_volume(full_feature_df):
    """RVOL is an alias for RELATIVE_VOLUME_RVOL — both columns must be equal."""
    assert "RVOL" in full_feature_df.columns
    assert "RELATIVE_VOLUME_RVOL" in full_feature_df.columns
    pd.testing.assert_series_equal(
        full_feature_df["RVOL"].reset_index(drop=True),
        full_feature_df["RELATIVE_VOLUME_RVOL"].reset_index(drop=True),
        check_names=False,
    )


# ── T66 ───────────────────────────────────────────────────────────────────────


def test_session_context_is_zero_or_one(full_feature_df):
    assert set(full_feature_df["SESSION_CONTEXT"].unique()).issubset({0, 1})


# ── T67 ───────────────────────────────────────────────────────────────────────


def test_derived_zero_filled_when_deps_missing(engineer):
    """
    Call _compute_derived directly with a DataFrame that has no ROC_5 or ATR_14.
    Absent-dependency columns should be zero-filled to 0.0.
    """
    fe = pd.DataFrame(
        {"close": [1.1002] * 10},
        index=pd.date_range("2024-01-01", periods=10, freq="1min"),
    )
    result = engineer._compute_derived(fe)
    assert np.allclose(result["ROC_5_ACCELERATION"], 0.0)
    assert np.allclose(result["ATR_RATIO_CHANGE"], 0.0)
    assert np.allclose(result["MACD_HIST_SLOPE"], 0.0)
    assert np.allclose(result["BB_WIDTH_MA"], 0.0)
    assert np.allclose(result["ATR_MA"], 0.0)
