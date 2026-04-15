"""
test_feature_vector.py — Tests for the FeatureVector dataclass.

Group 1: float32 coercion, read-only flag, frozen enforcement, to_dict, repr.
"""

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError
from src.ml_engine.features import FeatureVector


def _make_fv(**overrides):
    defaults = dict(
        timestamp=pd.Timestamp("2024-01-01 10:00"),
        vector=np.array([1.0, 2.0, 3.0]),
        feature_names=["A", "B", "C"],
        version="3.1.0-full-binary",
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


# ── T01 ───────────────────────────────────────────────────────────────────────


def test_feature_vector_vector_is_float32():
    fv = _make_fv(vector=np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert fv.vector.dtype == np.float32


# ── T02 ───────────────────────────────────────────────────────────────────────


def test_feature_vector_vector_is_read_only():
    fv = _make_fv()
    assert not fv.vector.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        fv.vector[0] = 99.0


# ── T03 ───────────────────────────────────────────────────────────────────────


def test_feature_vector_to_dict_keys():
    fv = _make_fv()
    d = fv.to_dict()
    assert set(d.keys()) == {"timestamp", "vector", "feature_names", "version"}


# ── T04 ───────────────────────────────────────────────────────────────────────


def test_feature_vector_to_dict_vector_is_list():
    fv = _make_fv()
    d = fv.to_dict()
    assert isinstance(d["vector"], list)


# ── T05 ───────────────────────────────────────────────────────────────────────


def test_feature_vector_repr_contains_n_features():
    fv = _make_fv(vector=np.array([1.0, 2.0, 3.0]))
    r = repr(fv)
    assert "n_features=3" in r


# ── T06 ───────────────────────────────────────────────────────────────────────


def test_feature_vector_frozen_raises_on_setattr():
    fv = _make_fv()
    with pytest.raises(FrozenInstanceError):
        fv.version = "x"  # type: ignore[misc]


# ── T07 ───────────────────────────────────────────────────────────────────────


def test_feature_vector_int_vector_coerced():
    fv = _make_fv(vector=np.array([1, 2, 3], dtype=np.int32))
    assert fv.vector.dtype == np.float32
    assert list(fv.vector) == pytest.approx([1.0, 2.0, 3.0])
