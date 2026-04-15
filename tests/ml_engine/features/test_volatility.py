"""
test_volatility.py — Tests for _compute_volatility().

Group 6: Non-negative values, flat-market edge cases, ratio invariants.

NOTE on BB_PERCENT_B with flat market:
A perfectly flat price series (std=0) produces band_width=0, which is replaced
by _EPS. Since close == sma == lower_bb, BB_PERCENT_B = 0/_EPS = 0. The test
verifies the value is finite (not NaN/Inf), not that it equals 0.5 — that
only holds when close is equidistant between the bands (std > 0).
"""

import numpy as np
import pytest


# ── T42 ───────────────────────────────────────────────────────────────────────


def test_atr_14_non_negative(full_feature_df):
    assert (full_feature_df["ATR_14"] >= 0).all()


# ── T43 ───────────────────────────────────────────────────────────────────────


def test_natr_14_non_negative(full_feature_df):
    assert (full_feature_df["NATR_14"] >= 0).all()


# ── T44 ───────────────────────────────────────────────────────────────────────


def test_bb_width_non_negative(full_feature_df):
    assert (full_feature_df["BB_WIDTH"] >= 0).all()


# ── T45 ───────────────────────────────────────────────────────────────────────


def test_bb_percent_b_is_finite(full_feature_df):
    """
    For a flat market (std=0), BB_PERCENT_B resolves to 0 via the _EPS guard —
    not 0.5. We verify the value is finite and present, not a specific value.
    """
    col = full_feature_df["BB_PERCENT_B"]
    assert col.notna().all()
    assert np.isfinite(col.values).all()


# ── T46 ───────────────────────────────────────────────────────────────────────


def test_kc_width_non_negative(full_feature_df):
    assert (full_feature_df["KC_WIDTH"] >= 0).all()


# ── T47 ───────────────────────────────────────────────────────────────────────


def test_range_expansion_ratio_near_one_flat(full_feature_df):
    """
    For a flat market (constant high - low), current_range == rolling_mean(range),
    so RANGE_EXPANSION_RATIO should be exactly 1.0.
    """
    col = full_feature_df["RANGE_EXPANSION_RATIO"]
    assert np.allclose(col.values, 1.0, atol=1e-6)
