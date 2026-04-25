"""
test_get_expiry_features.py — Tests for FeatureEngineer.get_expiry_features().

Group 11: Column subsets, alias resolution, invalid key rejection, missing-column warning.
"""

import logging

import pandas as pd
import pytest

from src.ml_engine.features import BINARY_EXPIRY_RULES


# ── T77 ───────────────────────────────────────────────────────────────────────


def test_get_expiry_features_1min_returns_correct_cols(engineer, full_feature_df):
    """1_MIN expiry maps to TICK_VELOCITY, RELATIVE_VOLUME_RVOL, BODY_TO_RANGE_RATIO."""
    result = engineer.get_expiry_features(full_feature_df, "1_MIN")
    assert "TICK_VELOCITY" in result.columns
    assert "RELATIVE_VOLUME_RVOL" in result.columns
    assert "BODY_TO_RANGE_RATIO" in result.columns


# ── T78 ───────────────────────────────────────────────────────────────────────


def test_get_expiry_features_invalid_key_raises(engineer, full_feature_df):
    with pytest.raises(ValueError, match="Unknown expiry key"):
        engineer.get_expiry_features(full_feature_df, "BAD_KEY")


# ── T79 ───────────────────────────────────────────────────────────────────────


def test_get_expiry_features_missing_cols_logs_warning(engineer, caplog):
    """A DataFrame missing an expected expiry column should log a [%] warning."""
    # 1_MIN needs TICK_VELOCITY — omit it to trigger the warning
    fe = pd.DataFrame(
        {
            "RELATIVE_VOLUME_RVOL": [1.5],
            "BODY_TO_RANGE_RATIO": [0.7],
            # TICK_VELOCITY intentionally absent
        }
    )
    with caplog.at_level(logging.WARNING):
        engineer.get_expiry_features(fe, "1_MIN")
    assert "missing columns" in caplog.text.lower() or "%" in caplog.text


# ── T80 ───────────────────────────────────────────────────────────────────────


def test_get_expiry_features_15min_resolves_session_context(engineer, full_feature_df):
    """15_MIN expiry references SESSION_CONTEXT which is created in _compute_derived."""
    result = engineer.get_expiry_features(full_feature_df, "15_MIN")
    assert "SESSION_CONTEXT" in result.columns


# ── T81 ───────────────────────────────────────────────────────────────────────


def test_get_expiry_features_returns_dataframe_subset(engineer, full_feature_df):
    result = engineer.get_expiry_features(full_feature_df, "5_MIN")
    assert isinstance(result, pd.DataFrame)
    # Subset: all returned columns must exist in the source
    for col in result.columns:
        assert col in full_feature_df.columns
