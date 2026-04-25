"""
test_atomic_upsert.py -- Tests for Storage._atomic_upsert().

Covers:
    Group 7 : Schema guard, dedup, sort, last-in-wins, failure handling
"""

import pytest
import pandas as pd
from datetime import datetime
from src.data.storage import StorageError


def test_atomic_upsert_raises_on_missing_timestamp_column(storage, tmp_path):
    bad_df = pd.DataFrame({"price": [1.0, 2.0]})
    with pytest.raises(StorageError, match="missing required column"):
        storage._atomic_upsert(tmp_path / "test.parquet", bad_df)


def test_atomic_upsert_creates_new_file(storage, tmp_path):
    df = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
    path = tmp_path / "test.parquet"
    storage._atomic_upsert(path, df)
    assert path.exists()


def test_atomic_upsert_deduplicates(storage, tmp_path):
    ts = datetime(2024, 1, 1)
    df = pd.DataFrame({"timestamp": [ts], "value": [1.0]})
    path = tmp_path / "test.parquet"
    storage._atomic_upsert(path, df)
    storage._atomic_upsert(path, df)
    assert len(pd.read_parquet(path)) == 1


def test_atomic_upsert_last_in_wins(storage, tmp_path):
    ts = datetime(2024, 1, 1)
    path = tmp_path / "test.parquet"
    storage._atomic_upsert(path, pd.DataFrame({"timestamp": [ts], "value": [1.0]}))
    storage._atomic_upsert(path, pd.DataFrame({"timestamp": [ts], "value": [99.0]}))
    result = pd.read_parquet(path)
    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(99.0)


def test_atomic_upsert_output_sorted(storage, tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 3),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            ],
            "value": [3.0, 1.0, 2.0],
        }
    )
    path = tmp_path / "test.parquet"
    storage._atomic_upsert(path, df)
    result = pd.read_parquet(path)
    timestamps = result["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_atomic_upsert_raises_storage_error_on_read_failure(storage, tmp_path):
    path = tmp_path / "test.parquet"
    path.write_bytes(b"corrupted parquet")
    df = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
    with pytest.raises(StorageError, match="Parquet upsert failed"):
        storage._atomic_upsert(path, df)


def test_atomic_upsert_raises_storage_error_on_write_failure(storage, tmp_path):
    # Write path now uses pq.write_table() via _write_versioned_parquet,
    # not df.to_parquet(), so patch the pyarrow function instead.
    from unittest.mock import patch
    import pyarrow.parquet as pq

    df = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
    path = tmp_path / "test.parquet"
    with patch.object(pq, "write_table", side_effect=OSError("disk full")):
        with pytest.raises(StorageError, match="Parquet upsert failed"):
            storage._atomic_upsert(path, df)


def test_atomic_upsert_preserves_existing_data(storage, tmp_path):
    path = tmp_path / "test.parquet"
    storage._atomic_upsert(
        path, pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})
    )
    storage._atomic_upsert(
        path, pd.DataFrame({"timestamp": [datetime(2024, 1, 2)], "value": [2.0]})
    )
    result = pd.read_parquet(path)
    assert len(result) == 2
    assert set(result["value"].tolist()) == {1.0, 2.0}
