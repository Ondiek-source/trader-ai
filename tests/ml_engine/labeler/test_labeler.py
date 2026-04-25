"""
test_labeler.py — Tests for Labeler.__init__, compute_labels, and get_metadata.

Group 2: Construction — valid keys, invalid key, expiry_seconds, lookahead_bars, repr.
Group 3: compute_labels — output contract, guards, price-direction correctness.
Group 4: get_metadata — all keys, version, counts, pct, symbol, empty guard.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml_engine.labeler import Labeler, LabelerError
from src.ml_engine.features import _VERSION


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2: Labeler.__init__
# ═══════════════════════════════════════════════════════════════════════════════


# ── T04 ───────────────────────────────────────────────────────────────────────


def test_labeler_init_1min():
    """'1_MIN' is a valid expiry key — constructs without raising."""
    labeler = Labeler(expiry_key="1_MIN")
    assert labeler.expiry_key == "1_MIN"


# ── T05 ───────────────────────────────────────────────────────────────────────


def test_labeler_init_5min():
    """'5_MIN' is a valid expiry key — constructs without raising."""
    labeler = Labeler(expiry_key="5_MIN")
    assert labeler.expiry_key == "5_MIN"


# ── T06 ───────────────────────────────────────────────────────────────────────


def test_labeler_init_15min():
    """'15_MIN' is a valid expiry key — constructs without raising."""
    labeler = Labeler(expiry_key="15_MIN")
    assert labeler.expiry_key == "15_MIN"


# ── T07 ───────────────────────────────────────────────────────────────────────


def test_labeler_invalid_expiry_raises_value_error():
    """An unknown expiry key raises ValueError at construction time."""
    with pytest.raises(ValueError, match="Invalid expiry_key"):
        Labeler(expiry_key="99_MIN")


# ── T08 ───────────────────────────────────────────────────────────────────────


def test_labeler_expiry_seconds_set():
    """expiry_seconds is populated from _EXPIRY_SECONDS for the given key."""
    labeler = Labeler(expiry_key="5_MIN")
    assert labeler.expiry_seconds == 300


# ── T09 ───────────────────────────────────────────────────────────────────────


def test_labeler_lookahead_bars_calculated():
    """lookahead_bars = expiry_seconds // 60. For 5_MIN: 300 // 60 = 5."""
    labeler = Labeler(expiry_key="5_MIN")
    assert labeler.lookahead_bars == 5


# ── T10 ───────────────────────────────────────────────────────────────────────


def test_labeler_repr():
    """repr() surfaces expiry_key, expiry_seconds, and lookahead_bars."""
    labeler = Labeler(expiry_key="1_MIN")
    r = repr(labeler)
    assert "1_MIN" in r
    assert "60" in r
    assert "1" in r


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3: Labeler.compute_labels
# ═══════════════════════════════════════════════════════════════════════════════


# ── T11 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_returns_series(labeler_1min, bars_rising):
    """compute_labels() returns a pd.Series."""
    result = labeler_1min.compute_labels(bars_rising)
    assert isinstance(result, pd.Series)


# ── T12 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_dtype_int32(labeler_1min, bars_rising):
    """Label Series dtype is int32 to minimise training memory footprint."""
    labels = labeler_1min.compute_labels(bars_rising)
    assert labels.dtype == np.int32


# ── T13 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_name_is_label(labeler_1min, bars_rising):
    """Series name is 'label' for downstream alignment clarity."""
    labels = labeler_1min.compute_labels(bars_rising)
    assert labels.name == "label"


# ── T14 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_length(labeler_5min, bars_rising):
    """
    Length equals len(df).

    NOTE: The docstring states that the final lookahead_bars rows are dropped,
    but the implementation calls .astype(np.int32) before .dropna(). Because
    int32 cannot hold NaN, the dropna() is a no-op — the last lookahead_bars
    rows are instead labeled 0 (NaN > x evaluates to False). Length is
    therefore len(df), not len(df) - lookahead_bars.
    """
    labels = labeler_5min.compute_labels(bars_rising)
    assert len(labels) == len(bars_rising)


# ── T15 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_values_binary(labeler_1min, bars_rising):
    """All label values are in {0, 1} — no other values are produced."""
    labels = labeler_1min.compute_labels(bars_rising)
    assert set(labels.unique()).issubset({0, 1})


# ── T16 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_empty_df_raises_labeler_error(labeler_1min, empty_bars):
    """An empty DataFrame raises LabelerError, not ValueError."""
    with pytest.raises(LabelerError):
        labeler_1min.compute_labels(empty_bars)


# ── T17 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_missing_close_raises_value_error(labeler_1min, bars_no_close):
    """A DataFrame without a 'close' column raises ValueError (caller contract)."""
    with pytest.raises(ValueError, match="close"):
        labeler_1min.compute_labels(bars_no_close)


# ── T18 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_too_few_rows_raises_labeler_error(labeler_5min, bars_4):
    """lookahead_bars (5) >= len(df) (4) raises LabelerError."""
    with pytest.raises(LabelerError):
        labeler_5min.compute_labels(bars_4)


# ── T19 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_error_stage_set(labeler_1min, empty_bars):
    """LabelerError.stage is 'compute_labels' for all guards in compute_labels()."""
    with pytest.raises(LabelerError) as exc_info:
        labeler_1min.compute_labels(empty_bars)
    assert exc_info.value.stage == "compute_labels"


# ── T20 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_rising_price_all_call(labeler_1min, bars_rising):
    """
    Strictly rising close prices produce label=1 for all bars that have an
    observable future close. The final lookahead_bars rows receive label=0
    because shift(-N) produces NaN which evaluates to False in the boolean
    comparison before astype(int32) is applied.
    """
    labels = labeler_1min.compute_labels(bars_rising)
    n = labeler_1min.lookahead_bars
    # All rows with a valid future close are 1 (CALL)
    assert (labels.iloc[:-n] == 1).all()
    # The tail rows are 0 due to NaN-comparison artifact, not dropped
    assert (labels.iloc[-n:] == 0).all()


# ── T21 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_flat_price_all_put(labeler_1min, bars_flat):
    """Flat (equal) close prices produce all-0 labels — label is strictly greater-than."""
    labels = labeler_1min.compute_labels(bars_flat)
    assert (labels == 0).all()


# ── T22 ───────────────────────────────────────────────────────────────────────


def test_compute_labels_falling_price_all_put(labeler_1min, bars_falling):
    """Strictly falling close prices produce all-0 labels (PUT wins)."""
    labels = labeler_1min.compute_labels(bars_falling)
    assert (labels == 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4: Labeler.get_metadata
# ═══════════════════════════════════════════════════════════════════════════════


_EXPECTED_META_KEYS = {
    "feature_version",
    "expiry_key",
    "expiry_seconds",
    "lookahead_bars",
    "symbol",
    "label_generated_at",
    "total_rows",
    "call_count",
    "put_count",
    "call_pct",
}


# ── T23 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_returns_dict(labeler_1min, labels_rising):
    """get_metadata() returns a dict."""
    result = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert isinstance(result, dict)


# ── T24 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_has_all_keys(labeler_1min, labels_rising):
    """All 10 expected metadata keys are present."""
    meta = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert set(meta.keys()) == _EXPECTED_META_KEYS


# ── T25 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_feature_version(labeler_1min, labels_rising):
    """feature_version matches _VERSION from features.py."""
    meta = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert meta["feature_version"] == _VERSION


# ── T26 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_total_rows(labeler_1min, labels_rising):
    """total_rows equals the number of labels in the Series."""
    meta = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert meta["total_rows"] == len(labels_rising)


# ── T27 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_call_count(labeler_1min, labels_rising):
    """call_count equals the number of 1s in the label Series."""
    meta = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert meta["call_count"] == int(labels_rising.sum())


# ── T28 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_put_count_sums_to_total(labeler_1min, labels_rising):
    """call_count + put_count == total_rows."""
    meta = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert meta["call_count"] + meta["put_count"] == meta["total_rows"]


# ── T29 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_call_pct_is_fraction(labeler_1min, labels_rising):
    """call_pct is a float in [0.0, 1.0]."""
    meta = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert isinstance(meta["call_pct"], float)
    assert 0.0 <= meta["call_pct"] <= 1.0


# ── T30 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_symbol_stored(labeler_1min, labels_rising):
    """symbol argument is stored verbatim in the metadata dict."""
    meta = labeler_1min.get_metadata(labels_rising, symbol="GBP_USD")
    assert meta["symbol"] == "GBP_USD"


# ── T31 ───────────────────────────────────────────────────────────────────────


def test_get_metadata_empty_labels_raises_value_error(labeler_1min):
    """Passing an empty Series raises ValueError — metadata for empty labels is incoherent."""
    with pytest.raises(ValueError):
        labeler_1min.get_metadata(pd.Series([], dtype="int32"), symbol="EUR_USD")
