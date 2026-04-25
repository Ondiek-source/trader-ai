"""
test_save_bar.py -- Tests for Storage.save_bar().

Covers:
    Group 3 : Bar write operations, dedup, sorting, incomplete guard
"""

import pytest
import pandas as pd


def test_save_bar_returns_true(storage, valid_bar):
    assert storage.save_bar(valid_bar) is True


def test_save_bar_creates_correctly_named_file(storage, valid_bar):
    storage.save_bar(valid_bar)
    assert (storage.processed_dir / "EUR_USD_M1.parquet").exists()


def test_save_bar_deduplicates(storage, valid_bar):
    storage.save_bar(valid_bar)
    storage.save_bar(valid_bar)
    df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
    assert len(df) == 1


def test_save_bar_appends(storage, valid_bar, valid_bar_2):
    storage.save_bar(valid_bar)
    storage.save_bar(valid_bar_2)
    df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
    assert len(df) == 2


def test_save_bar_output_is_sorted(storage, valid_bar, valid_bar_2):
    storage.save_bar(valid_bar_2)
    storage.save_bar(valid_bar)
    df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_save_bar_rejects_incomplete_bar(storage, incomplete_bar):
    result = storage.save_bar(incomplete_bar)
    assert result is False
    assert not (storage.processed_dir / "EUR_USD_M1.parquet").exists()


def test_save_bar_uses_timeframe_value_not_repr(storage, valid_bar):
    storage.save_bar(valid_bar)
    files = list(storage.processed_dir.glob("*.parquet"))
    assert len(files) == 1
    assert "Timeframe" not in files[0].name
    assert "M1" in files[0].name
