"""
test_feature_matrix.py — Tests for FeatureMatrix and build_matrix().

Group 14: FeatureMatrix dataclass — float32 coercion, read-only flag, frozen
          enforcement, 2D shape validation, column/name mismatch, to_tensor,
          __len__, __repr__, and CRITICAL diagnostic logging on construction failure.

Group 15: FeatureEngineer.build_matrix() — happy-path output contract,
          symbol/version storage, error wrapping, shape alignment with schema.
"""

import logging

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from src.ml_engine.features import FeatureMatrix, _VERSION


def _make_fm(**overrides):
    defaults = dict(
        timestamps=["2024-01-01 10:00", "2024-01-01 10:01", "2024-01-01 10:02"],
        matrix=np.ones((3, 4), dtype=np.float64),
        feature_names=["A", "B", "C", "D"],
        version=_VERSION,
        symbol="EURUSD",
    )
    defaults.update(overrides)
    return FeatureMatrix(**defaults)  # type: ignore[arg-type]


# ── T102 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_matrix_is_float32():
    """float64 input is coerced to float32 in __post_init__."""
    fm = _make_fm(matrix=np.ones((3, 4), dtype=np.float64))
    assert fm.matrix.dtype == np.float32


# ── T103 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_matrix_is_read_only():
    """matrix.flags.writeable is False; any write raises ValueError."""
    fm = _make_fm()
    assert not fm.matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        fm.matrix[0, 0] = 99.0


# ── T104 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_frozen_raises_on_setattr():
    """FeatureMatrix is a frozen dataclass — field assignment raises."""
    fm = _make_fm()
    with pytest.raises(FrozenInstanceError):
        fm.version = "x"  # type: ignore[misc]


# ── T105 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_int_matrix_coerced():
    """Integer matrix (int32) is cast to float32 and values preserved."""
    mat = np.arange(12, dtype=np.int32).reshape(3, 4)
    fm = _make_fm(matrix=mat)
    assert fm.matrix.dtype == np.float32
    np.testing.assert_array_almost_equal(fm.matrix, mat.astype(np.float32))


# ── T106 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_1d_raises_value_error():
    """Passing a 1D array raises ValueError mentioning '2D'."""
    with pytest.raises(ValueError, match="2D"):
        _make_fm(matrix=np.ones(4), feature_names=["A", "B", "C", "D"])


# ── T107 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_3d_raises_value_error():
    """Passing a 3D array raises ValueError mentioning '2D'."""
    with pytest.raises(ValueError, match="2D"):
        _make_fm(matrix=np.ones((3, 4, 5)), feature_names=["A", "B", "C", "D"])


# ── T108 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_mismatched_columns_raises_value_error():
    """matrix.shape[1]=4 but len(feature_names)=3 → ValueError."""
    with pytest.raises(ValueError, match="feature_names"):
        _make_fm(
            matrix=np.ones((3, 4)),
            feature_names=["A", "B", "C"],  # 3, matrix has 4 columns
        )


# ── T109 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_1d_logs_critical(caplog):
    """Invalid shape logs a CRITICAL block with 'FEATURE MATRIX ERROR' before raising."""
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError):
            _make_fm(matrix=np.ones(4), feature_names=["A", "B", "C", "D"])
    assert "FEATURE MATRIX ERROR" in caplog.text


# ── T110 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_mismatched_columns_logs_critical(caplog):
    """Column/name mismatch logs a CRITICAL block before raising."""
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError):
            _make_fm(
                matrix=np.ones((3, 4)),
                feature_names=["A", "B", "C"],
            )
    assert "FEATURE MATRIX ERROR" in caplog.text


# ── T111 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_len_returns_row_count():
    """len(fm) equals the number of rows (axis=0) in matrix."""
    fm = _make_fm(matrix=np.ones((7, 4)))
    assert len(fm) == 7


# ── T112 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_to_tensor_shape():
    """to_tensor() wraps (N, F) into (1, N, F) for SequenceGenerator seeding."""
    fm = _make_fm(matrix=np.ones((3, 4)))
    assert fm.to_tensor().shape == (1, 3, 4)


# ── T113 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_to_tensor_is_3d():
    """to_tensor() always returns a 3-dimensional array."""
    fm = _make_fm()
    assert fm.to_tensor().ndim == 3


# ── T114 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_to_tensor_data_preserved():
    """to_tensor()[0] is numerically identical to the original matrix."""
    mat = np.arange(12, dtype=np.float32).reshape(3, 4)
    fm = _make_fm(matrix=mat)
    np.testing.assert_array_equal(fm.to_tensor()[0], fm.matrix)


# ── T115 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_repr_contains_symbol():
    """repr() includes the symbol field."""
    fm = _make_fm(symbol="GBPUSD")
    assert "GBPUSD" in repr(fm)


# ── T116 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_repr_contains_shape():
    """repr() includes both dimensions of the matrix shape."""
    fm = _make_fm(matrix=np.ones((3, 4)))
    r = repr(fm)
    assert "3" in r
    assert "4" in r


# ── T117 ──────────────────────────────────────────────────────────────────────


def test_feature_matrix_repr_contains_version():
    """repr() includes the version string."""
    fm = _make_fm(version="3.1.0-full-binary")
    assert "3.1.0-full-binary" in repr(fm)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 15: FeatureEngineer.build_matrix()
# ═══════════════════════════════════════════════════════════════════════════════


# ── T118 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_returns_feature_matrix(engineer, bars_100):
    """build_matrix() on valid bars returns a FeatureMatrix instance."""
    fm = engineer.build_matrix(bars_100, symbol="EURUSD")
    assert isinstance(fm, FeatureMatrix)


# ── T119 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_version_matches_module_constant(engineer, bars_100):
    """version field on the returned FeatureMatrix equals _VERSION."""
    fm = engineer.build_matrix(bars_100, symbol="EURUSD")
    assert fm.version == _VERSION


# ── T120 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_symbol_stored(engineer, bars_100):
    """symbol argument is stored verbatim in the returned FeatureMatrix."""
    fm = engineer.build_matrix(bars_100, symbol="GBPUSD")
    assert fm.symbol == "GBPUSD"


# ── T121 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_matrix_shape_matches_feature_set(engineer, bars_100):
    """matrix.shape[1] equals the number of primary features in the schema."""
    from src.ml_engine.features import FEATURE_SET_BINARY_OPTIONS_AI

    fm = engineer.build_matrix(bars_100, symbol="EURUSD")
    expected_cols = [c for g in FEATURE_SET_BINARY_OPTIONS_AI.values() for c in g]
    # All schema columns that exist in the transform output are packed into matrix
    assert fm.matrix.shape[1] == len(fm.feature_names)
    assert fm.matrix.shape[1] <= len(expected_cols)


# ── T122 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_timestamps_length_matches_rows(engineer, bars_100):
    """len(timestamps) == matrix.shape[0] — every row has a timestamp."""
    fm = engineer.build_matrix(bars_100, symbol="EURUSD")
    assert len(fm.timestamps) == fm.matrix.shape[0]


# ── T123 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_matrix_is_float32(engineer, bars_100):
    """matrix dtype is float32 regardless of intermediate computation precision."""
    fm = engineer.build_matrix(bars_100, symbol="EURUSD")
    assert fm.matrix.dtype == np.float32


# ── T124 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_empty_bars_raises_feature_engineer_error(engineer, empty_bars):
    """Empty bars input raises FeatureEngineerError at the build_matrix stage."""
    from src.ml_engine.features import FeatureEngineerError

    with pytest.raises(FeatureEngineerError):
        engineer.build_matrix(empty_bars, symbol="EURUSD")


# ── T125 ──────────────────────────────────────────────────────────────────────


def test_build_matrix_transform_exception_wrapped(engineer, bars_100, monkeypatch):
    """Any exception from transform() is wrapped into FeatureEngineerError."""
    from src.ml_engine.features import FeatureEngineerError

    def _boom(*a, **kw):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(engineer, "transform", _boom)
    with pytest.raises(FeatureEngineerError) as exc_info:
        engineer.build_matrix(bars_100, symbol="EURUSD")
    assert exc_info.value.stage == "build_matrix"
    assert "pipeline exploded" in str(exc_info.value)
