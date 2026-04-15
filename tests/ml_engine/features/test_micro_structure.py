"""
test_micro_structure.py — Tests for _compute_micro_structure().

Group 7: Non-negative bounds, order-flow range, zero-fill when disabled/ticks=None.
"""

import pytest
from src.ml_engine.features import FEATURE_SET_BINARY_OPTIONS_AI


# ── T48 ───────────────────────────────────────────────────────────────────────


def test_tick_velocity_non_negative(full_feature_df):
    assert (full_feature_df["TICK_VELOCITY"] >= 0).all()


# ── T49 ───────────────────────────────────────────────────────────────────────


def test_relative_volume_rvol_positive(full_feature_df):
    assert (full_feature_df["RELATIVE_VOLUME_RVOL"] > 0).all()


# ── T50 ───────────────────────────────────────────────────────────────────────


def test_spread_normalized_non_negative(full_feature_df):
    assert (full_feature_df["SPREAD_NORMALIZED"] >= 0).all()


# ── T51 ───────────────────────────────────────────────────────────────────────


def test_order_flow_imbalance_bounds(full_feature_df):
    col = full_feature_df["ORDER_FLOW_IMBALANCE"]
    assert (col >= -0.5).all()
    assert (col <= 0.5).all()


# ── T52 ───────────────────────────────────────────────────────────────────────


def test_micro_columns_zero_when_ticks_none(engineer, bars_100):
    result = engineer.transform(bars_100, ticks=None)
    for col in FEATURE_SET_BINARY_OPTIONS_AI["MICRO_STRUCTURE"]:
        assert col in result.columns
        assert (result[col] == 0.0).all(), f"{col} should be zero-filled"


# ── T53 ───────────────────────────────────────────────────────────────────────


def test_micro_columns_zero_when_flag_disabled(mock_settings, monkeypatch, bars_100):
    import src.ml_engine.features as feat_mod
    from src.ml_engine.features import FeatureEngineer

    mock_settings.feat_micro_enabled = False
    monkeypatch.setattr(feat_mod, "get_settings", lambda: mock_settings)
    eng = FeatureEngineer()
    result = eng.transform(bars_100, ticks=None)
    for col in FEATURE_SET_BINARY_OPTIONS_AI["MICRO_STRUCTURE"]:
        assert col in result.columns
        assert (result[col] == 0.0).all(), f"{col} should be zero-filled"
