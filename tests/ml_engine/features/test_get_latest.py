"""
test_get_latest.py — Tests for FeatureEngineer.get_latest().

Group 12: FeatureVector shape/dtype, error wrapping, NaN/Inf guard,
          stage attribute on FeatureEngineerError, version constant.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml_engine.features import (
    FEATURE_SET_BINARY_OPTIONS_AI,
    FeatureEngineerError,
    FeatureVector,
    _VERSION,
)
from .conftest import make_ohlcv, make_ticks


# ── T82 ───────────────────────────────────────────────────────────────────────


def test_get_latest_returns_feature_vector(engineer, bars_100):
    result = engineer.get_latest(bars_100)
    assert isinstance(result, FeatureVector)


# ── T83 ───────────────────────────────────────────────────────────────────────


def test_get_latest_vector_is_float32(engineer, bars_100):
    fv = engineer.get_latest(bars_100)
    assert fv.vector.dtype == np.float32


# ── T84 ───────────────────────────────────────────────────────────────────────


def test_get_latest_vector_is_read_only(engineer, bars_100):
    fv = engineer.get_latest(bars_100)
    assert not fv.vector.flags.writeable


# ── T85 ───────────────────────────────────────────────────────────────────────


def test_get_latest_version_matches_module_constant(engineer, bars_100):
    fv = engineer.get_latest(bars_100)
    assert fv.version == _VERSION


# ── T86 ───────────────────────────────────────────────────────────────────────


def test_get_latest_feature_names_match_schema(engineer, bars_100):
    fv = engineer.get_latest(bars_100)
    all_schema = {c for g in FEATURE_SET_BINARY_OPTIONS_AI.values() for c in g}
    for name in fv.feature_names:
        assert name in all_schema, f"Unexpected feature name: {name}"


# ── T87 ───────────────────────────────────────────────────────────────────────


def test_get_latest_empty_bars_raises_feature_engineer_error(engineer, empty_bars):
    with pytest.raises(FeatureEngineerError):
        engineer.get_latest(empty_bars)


# ── T88 ───────────────────────────────────────────────────────────────────────


def test_get_latest_too_few_bars_raises_feature_engineer_error(engineer, bars_5):
    """5 bars is insufficient for MACD warmup → transform returns empty → raises."""
    with pytest.raises(FeatureEngineerError):
        engineer.get_latest(bars_5)


# ── T89 ───────────────────────────────────────────────────────────────────────


def test_get_latest_stage_set_on_error(engineer, bars_5):
    with pytest.raises(FeatureEngineerError) as exc_info:
        engineer.get_latest(bars_5)
    assert exc_info.value.stage == "get_latest"


# ── T90 ───────────────────────────────────────────────────────────────────────


def test_get_latest_nan_values_replaced_with_zero(engineer, bars_100, monkeypatch):
    """NaN in a feature column in the transform output is replaced with 0.0."""
    import pandas as pd

    original_transform = engineer.transform

    def patched_transform(bars, ticks=None):
        df = original_transform(bars, ticks)
        # Inject NaN into the first primary feature column
        first_col = list(FEATURE_SET_BINARY_OPTIONS_AI.values())[0][0]
        df = df.copy()
        df[first_col] = np.nan
        return df

    monkeypatch.setattr(engineer, "transform", patched_transform)
    fv = engineer.get_latest(bars_100)
    # No exception raised; NaN replaced with 0.0
    assert np.isfinite(fv.vector).all()


# ── T91 ───────────────────────────────────────────────────────────────────────


def test_get_latest_inf_values_replaced_with_zero(engineer, bars_100, monkeypatch):
    """Inf in a feature column in the transform output is replaced with 0.0."""
    original_transform = engineer.transform

    def patched_transform(bars, ticks=None):
        df = original_transform(bars, ticks)
        first_col = list(FEATURE_SET_BINARY_OPTIONS_AI.values())[0][0]
        df = df.copy()
        df[first_col] = np.inf
        return df

    monkeypatch.setattr(engineer, "transform", patched_transform)
    fv = engineer.get_latest(bars_100)
    assert np.isfinite(fv.vector).all()


# ── T92 ───────────────────────────────────────────────────────────────────────


def test_get_latest_transform_exception_wrapped(engineer, bars_100, monkeypatch):
    """Any exception inside transform() is wrapped into FeatureEngineerError."""

    def _raise(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(engineer, "transform", _raise)
    with pytest.raises(FeatureEngineerError) as exc_info:
        engineer.get_latest(bars_100)
    assert exc_info.value.stage == "transform"
    assert "boom" in str(exc_info.value)


# ── T_NEW_05: FeatureEngineerError repr ──────────────────────────────────────


def test_feature_engineer_error_repr():
    err = FeatureEngineerError("something broke", stage="transform")
    r = repr(err)
    assert "transform" in r
    assert "something broke" in r


# ── T_NEW_06: FeatureEngineerError stage attribute ───────────────────────────


def test_feature_engineer_error_stage_attribute():
    err = FeatureEngineerError("msg", stage="get_latest")
    assert err.stage == "get_latest"
    assert str(err) == "msg"
