"""
test_data_buffer.py -- Tests for the DataBuffer class.

Covers:
    Group 6 : DataBuffer construction and core logic
    Group 7 : DataBuffer thread safety
"""

import pytest
import threading
from src.ml_engine.model import DataBuffer


# ==============================================================================
# GROUP 6 -- DataBuffer: Construction & Core Logic
# ==============================================================================


def test_buffer_constructs_with_valid_flush_size():
    buf = DataBuffer(flush_size=5)
    assert len(buf) == 0


def test_buffer_rejects_zero_flush_size():
    with pytest.raises(ValueError, match="flush_size"):
        DataBuffer(flush_size=0)


def test_buffer_rejects_negative_flush_size():
    with pytest.raises(ValueError, match="flush_size"):
        DataBuffer(flush_size=-10)


def test_buffer_add_returns_empty_list_before_flush(valid_tick):
    buf = DataBuffer(flush_size=3)
    assert buf.add(valid_tick) == []
    assert buf.add(valid_tick) == []
    assert len(buf) == 2


def test_buffer_add_returns_batch_at_flush(valid_tick):
    buf = DataBuffer(flush_size=3)
    buf.add(valid_tick)
    buf.add(valid_tick)
    batch = buf.add(valid_tick)
    assert batch is not None
    assert len(batch) == 3
    assert len(buf) == 0


def test_buffer_clears_after_flush(valid_tick):
    buf = DataBuffer(flush_size=2)
    buf.add(valid_tick)
    buf.add(valid_tick)
    assert len(buf) == 0
    result = buf.add(valid_tick)
    assert result == []
    assert len(buf) == 1


def test_buffer_force_flush_drains_partial_batch(valid_tick):
    buf = DataBuffer(flush_size=10)
    buf.add(valid_tick)
    buf.add(valid_tick)
    buf.add(valid_tick)
    batch = buf.flush()
    assert batch is not None
    assert len(batch) == 3
    assert len(buf) == 0


def test_buffer_force_flush_empty_returns_empty_list():
    buf = DataBuffer(flush_size=5)
    assert buf.flush() == []


def test_buffer_overflow_raises_memory_error(valid_tick):
    flush_size = 3
    buf = DataBuffer(flush_size=flush_size)
    cap = flush_size * 10
    for _ in range(cap):
        buf._data.append(valid_tick)
    with pytest.raises(RuntimeError, match="Buffer overflow"):
        buf.add(valid_tick)


def test_buffer_len_is_accurate(valid_tick):
    buf = DataBuffer(flush_size=10)
    assert len(buf) == 0
    buf.add(valid_tick)
    assert len(buf) == 1
    buf.add(valid_tick)
    assert len(buf) == 2


# ==============================================================================
# GROUP 7 -- DataBuffer: Thread Safety
# ==============================================================================


def test_buffer_concurrent_adds_correct_total(valid_tick):
    buf = DataBuffer(flush_size=200)
    num_threads = 10
    adds_per_thread = 10
    barrier = threading.Barrier(num_threads)

    def add_ticks():
        barrier.wait()
        for _ in range(adds_per_thread):
            buf.add(valid_tick)

    threads = [threading.Thread(target=add_ticks) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(buf) == num_threads * adds_per_thread


def test_buffer_concurrent_flush_returns_exactly_one_batch(valid_tick):
    flush_size = 10
    buf = DataBuffer(flush_size=flush_size)
    batches = []
    lock = threading.Lock()

    def add_and_collect():
        result = buf.add(valid_tick)
        if result:
            with lock:
                batches.append(result)

    threads = [threading.Thread(target=add_and_collect) for _ in range(flush_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(batches) == 1
    assert len(batches[0]) == flush_size


def test_buffer_flush_concurrent_with_add(valid_tick):
    buf = DataBuffer(flush_size=1000)
    errors = []

    def stream_thread():
        for _ in range(50):
            try:
                buf.add(valid_tick)
            except Exception as e:
                errors.append(e)

    def shutdown_thread():
        for _ in range(5):
            buf.flush()

    t1 = threading.Thread(target=stream_thread)
    t2 = threading.Thread(target=shutdown_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []


def test_buffer_len_accurate_under_concurrency(valid_tick):
    buf = DataBuffer(flush_size=1000)
    errors = []

    def add_ticks():
        for _ in range(100):
            buf.add(valid_tick)

    def check_len():
        for _ in range(100):
            try:
                l = len(buf)
                assert l >= 0
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=add_ticks)
    t2 = threading.Thread(target=check_len)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []


def test_buffer_returns_copy_not_reference(valid_tick):
    buf = DataBuffer(flush_size=2)
    buf.add(valid_tick)
    batch = buf.add(valid_tick)
    assert batch != []
    batch.clear()
    result = buf.add(valid_tick)
    assert result == []
    assert len(buf) == 1


def test_buffer_multiple_flush_cycles(valid_tick):
    buf = DataBuffer(flush_size=3)
    buf.add(valid_tick)
    buf.add(valid_tick)
    batch1 = buf.add(valid_tick)
    assert batch1 is not None and len(batch1) == 3

    buf.add(valid_tick)
    buf.add(valid_tick)
    batch2 = buf.add(valid_tick)
    assert batch2 is not None and len(batch2) == 3

    assert len(buf) == 0
