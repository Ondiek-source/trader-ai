"""
test_save_tick_batch.py -- Tests for Storage.save_tick_batch().

Covers:
    Group 2 : Tick write operations, dedup, sorting, guards
"""

import pytest
import pandas as pd


def test_save_tick_batch_returns_true(storage, valid_tick, valid_tick_2):
    result = storage.save_tick_batch([valid_tick, valid_tick_2])
    assert result is True


def test_save_tick_batch_creates_file(storage, valid_tick):
    storage.save_tick_batch([valid_tick])
    assert (storage.raw_dir / "EUR_USD_ticks.parquet").exists()


def test_save_tick_batch_empty_returns_false(storage):
    result = storage.save_tick_batch([])
    assert result is False
    assert not (storage.raw_dir / "EUR_USD_ticks.parquet").exists()


def test_save_tick_batch_persists_correct_count(storage, valid_tick, valid_tick_2):
    storage.save_tick_batch([valid_tick, valid_tick_2])
    df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
    assert len(df) == 2


def test_save_tick_batch_deduplicates(storage, valid_tick):
    storage.save_tick_batch([valid_tick])
    storage.save_tick_batch([valid_tick])
    df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
    assert len(df) == 1


def test_save_tick_batch_appends_new_ticks(storage, valid_tick, valid_tick_2):
    storage.save_tick_batch([valid_tick])
    storage.save_tick_batch([valid_tick_2])
    df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
    assert len(df) == 2


def test_save_tick_batch_output_is_sorted(storage, valid_tick, valid_tick_2):
    storage.save_tick_batch([valid_tick_2, valid_tick])
    df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_save_tick_batch_rejects_mixed_symbols(storage, valid_tick, valid_tick_gbp):
    with pytest.raises(ValueError, match="mixed symbols"):
        storage.save_tick_batch([valid_tick, valid_tick_gbp])


def test_save_tick_batch_raises_on_write_failure(storage, valid_tick):
    from src.data.storage import StorageError
    with pytest.raises(StorageError):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(storage, "_atomic_upsert",
                       lambda *a, **kw: (_ for _ in ()).throw(StorageError("disk full")))
            storage.save_tick_batch([valid_tick])
