"""
tests/data/storage/test_save_bar_batch.py

Tests for Storage.save_bar_batch() — the method the Historian calls after
every chunk fetch. Covers the empty-batch guard, mixed-symbol guard,
mixed-timeframe guard, and the happy path.
"""

import pytest
from datetime import datetime
from src.ml_engine.model import Bar, Timeframe


# ── Helpers ────────────────────────────────────────────────────────────────────


def _bar(symbol="EUR_USD", timeframe=Timeframe.M1, ts=None) -> Bar:
    return Bar(
        timestamp=ts or datetime(2026, 4, 12, 10, 0, 0),
        symbol=symbol,
        open=1.0850,
        high=1.0860,
        low=1.0845,
        close=1.0855,
        volume=100.0,
        is_complete=True,
        timeframe=timeframe,
    )


# ── Guard: empty batch ─────────────────────────────────────────────────────────


def test_save_bar_batch_empty_batch_returns_false(storage):
    """An empty list must return False immediately without touching the filesystem."""
    result = storage.save_bar_batch([])
    assert result is False


def test_save_bar_batch_empty_batch_creates_no_file(storage):
    result = storage.save_bar_batch([])
    parquet_files = list(storage.processed_dir.glob("*.parquet"))
    assert result is False
    assert len(parquet_files) == 0


# ── Guard: mixed symbols ───────────────────────────────────────────────────────


def test_save_bar_batch_mixed_symbols_raises(storage):
    """Bars from two different symbols in one call must raise ValueError."""
    bars = [_bar("EUR_USD"), _bar("GBP_USD")]
    with pytest.raises(ValueError, match="mixed symbols"):
        storage.save_bar_batch(bars)


# ── Guard: mixed timeframes ────────────────────────────────────────────────────


def test_save_bar_batch_mixed_timeframes_raises(storage):
    """Bars with two different timeframes in one call must raise ValueError."""
    bars = [_bar(timeframe=Timeframe.M1), _bar(timeframe=Timeframe.M5)]
    with pytest.raises(ValueError, match="mixed timeframes"):
        storage.save_bar_batch(bars)


# ── Happy path ─────────────────────────────────────────────────────────────────


def test_save_bar_batch_returns_true_on_success(storage):
    bars = [_bar()]
    assert storage.save_bar_batch(bars) is True


def test_save_bar_batch_creates_parquet_file(storage):
    bars = [_bar()]
    storage.save_bar_batch(bars)
    expected = storage.processed_dir / "EUR_USD_M1.parquet"
    assert expected.exists()


def test_save_bar_batch_persists_correct_count(storage):
    import pandas as pd

    bars = [
        _bar(ts=datetime(2026, 4, 12, 10, 0, 0)),
        _bar(ts=datetime(2026, 4, 12, 10, 1, 0)),
        _bar(ts=datetime(2026, 4, 12, 10, 2, 0)),
    ]
    storage.save_bar_batch(bars)
    df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
    assert len(df) == 3


def test_save_bar_batch_deduplicates_on_repeated_call(storage):
    """Calling save_bar_batch twice with the same bars must not duplicate rows."""
    import pandas as pd

    bars = [_bar(ts=datetime(2026, 4, 12, 10, 0, 0))]
    storage.save_bar_batch(bars)
    storage.save_bar_batch(bars)
    df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
    assert len(df) == 1


def test_save_bar_batch_appends_new_bars(storage):
    import pandas as pd

    storage.save_bar_batch([_bar(ts=datetime(2026, 4, 12, 10, 0, 0))])
    storage.save_bar_batch([_bar(ts=datetime(2026, 4, 12, 10, 1, 0))])
    df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
    assert len(df) == 2
