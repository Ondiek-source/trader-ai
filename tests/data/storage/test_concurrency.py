"""
test_concurrency.py -- Thread safety tests for Storage.

Covers:
    Group 8 : Concurrent writes, reads, lock integrity
"""

import pytest
import threading
import pandas as pd
from datetime import datetime
from src.ml_engine.model import Tick, Bar


def test_concurrent_tick_writes_correct_total(storage):
    num_threads = 10
    ticks_per_thread = 10
    barrier = threading.Barrier(num_threads)
    errors = []

    def write_batch(thread_id):
        batch = [
            Tick(
                timestamp=datetime(2024, 1, 15, 10, thread_id, i),
                symbol="EUR_USD",
                bid=1.0850,
                ask=1.0852,
                source="TWELVE",
            )
            for i in range(ticks_per_thread)
        ]
        barrier.wait()
        try:
            storage.save_tick_batch(batch)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=write_batch, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
    assert len(df) == num_threads * ticks_per_thread


def test_concurrent_bar_writes_correct_total(storage):
    num_threads = 10
    barrier = threading.Barrier(num_threads)
    errors = []

    def write_bar(thread_id):
        bar = Bar(
            timestamp=datetime(2024, 1, 15, 10, thread_id, 0),
            symbol="EUR_USD",
            open=1.0850,
            high=1.0865,
            low=1.0848,
            close=1.0860,
            volume=100 + thread_id,
        )
        barrier.wait()
        try:
            storage.save_bar(bar)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=write_bar, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    df = pd.read_parquet(storage.processed_dir / "EUR_USD_M1.parquet")
    assert len(df) == num_threads


def test_concurrent_read_write_no_corruption(storage):
    errors = []

    def write_ticks():
        for i in range(20):
            tick = Tick(
                timestamp=datetime(2024, 1, 15, 10, 0, i),
                symbol="EUR_USD",
                bid=1.0850,
                ask=1.0852,
                source="TWELVE",
            )
            try:
                storage.save_tick_batch([tick])
            except Exception as e:
                errors.append(e)

    def read_timestamps():
        for _ in range(20):
            try:
                storage.get_last_timestamp("EUR_USD")
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=write_ticks)
    t2 = threading.Thread(target=read_timestamps)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []


def test_concurrent_writes_different_symbols_no_interference(storage):
    errors = []

    def write_eur():
        for i in range(10):
            tick = Tick(
                timestamp=datetime(2024, 1, 15, 10, 0, i),
                symbol="EUR_USD",
                bid=1.0850,
                ask=1.0852,
                source="TWELVE",
            )
            storage.save_tick_batch([tick])

    def write_gbp():
        for i in range(10):
            tick = Tick(
                timestamp=datetime(2024, 1, 15, 10, 0, i),
                symbol="GBP_USD",
                bid=1.2650,
                ask=1.2652,
                source="QUOTEX",
            )
            storage.save_tick_batch([tick])

    t1 = threading.Thread(target=write_eur)
    t2 = threading.Thread(target=write_gbp)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    eur_df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
    gbp_df = pd.read_parquet(storage.raw_dir / "GBP_USD_ticks.parquet")
    assert len(eur_df) == 10
    assert len(gbp_df) == 10
    assert all(eur_df["symbol"] == "EUR_USD")
    assert all(gbp_df["symbol"] == "GBP_USD")


def test_no_data_loss_on_simultaneous_flush_and_write(storage):
    errors = []

    def write():
        for i in range(30):
            tick = Tick(
                timestamp=datetime(2024, 1, 15, 10, 0, i),
                symbol="EUR_USD",
                bid=1.0850,
                ask=1.0852,
                source="TWELVE",
            )
            try:
                storage.save_tick_batch([tick])
            except Exception as e:
                errors.append(e)

    def read_count():
        for _ in range(10):
            try:
                storage.get_tick_count("EUR_USD")
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=write)
    t2 = threading.Thread(target=read_count)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []


def test_list_symbols_safe_during_concurrent_writes(storage):
    errors = []

    def write_symbols():
        for i in range(5):
            tick = Tick(
                timestamp=datetime(2024, 1, 15, 10, 0, i),
                symbol="EUR_USD",
                bid=1.0850,
                ask=1.0852,
                source="TWELVE",
            )
            storage.save_tick_batch([tick])

    def list_repeatedly():
        for _ in range(10):
            try:
                storage.list_symbols()
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=write_symbols)
    t2 = threading.Thread(target=list_repeatedly)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []


def test_two_threads_same_timestamp_one_row(storage):
    ts = datetime(2024, 1, 15, 10, 0, 0)
    tick = Tick(timestamp=ts, symbol="EUR_USD", bid=1.0850, ask=1.0852, source="TWELVE")
    barrier = threading.Barrier(2)

    def write():
        barrier.wait()
        storage.save_tick_batch([tick])

    t1 = threading.Thread(target=write)
    t2 = threading.Thread(target=write)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    df = pd.read_parquet(storage.raw_dir / "EUR_USD_ticks.parquet")
    assert len(df) == 1


def test_lock_released_after_storage_error(storage, valid_tick):
    from src.data.storage import StorageError
    from unittest.mock import patch

    with patch.object(
        storage, "_atomic_upsert", side_effect=StorageError("forced failure")
    ):
        with pytest.raises(StorageError):
            storage.save_tick_batch([valid_tick])

    acquired = storage._lock.acquire(timeout=1)
    assert acquired, "Lock was not released after StorageError"
    storage._lock.release()
