"""
test_read_operations.py -- Tests for Storage read/query methods.

Covers:
    Group 4 : get_last_timestamp
    Group 5 : get_bars
"""

import pytest
import pandas as pd
from datetime import datetime


# ==============================================================================
# GROUP 4 -- get_last_timestamp
# ==============================================================================


def test_get_last_timestamp_empty_file_returns_none(storage):
    """
    Test that get_last_timestamp returns None when the Parquet file exists but is empty.
    """
    import pandas as pd

    # Create an empty Parquet file with correct schema
    file_path = storage.raw_dir / "EUR_USD_ticks.parquet"
    empty_df = pd.DataFrame(columns=["timestamp", "symbol", "bid", "ask", "source"])
    empty_df.to_parquet(file_path, index=False)

    # Should return None for empty file
    result = storage.get_last_timestamp("EUR_USD")

    assert result is None


def test_get_last_timestamp_returns_none_when_no_file(storage):
    assert storage.get_last_timestamp("EUR_USD") is None


def test_get_last_timestamp_returns_max(storage, valid_tick, valid_tick_2):
    storage.save_tick_batch([valid_tick, valid_tick_2])
    result = storage.get_last_timestamp("EUR_USD")
    assert result == pd.Timestamp("2024-01-15 10:31:00")


def test_get_last_timestamp_raises_on_corrupted_file(storage):
    from src.data.storage import StorageError

    file_path = storage.raw_dir / "EUR_USD_ticks.parquet"
    file_path.write_bytes(b"this is not a valid parquet file")
    with pytest.raises(StorageError, match="corrupted"):
        storage.get_last_timestamp("EUR_USD")


def test_get_last_timestamp_returns_none_for_empty_file(storage):
    empty_df = pd.DataFrame(columns=["timestamp"])
    file_path = storage.raw_dir / "EUR_USD_ticks.parquet"
    empty_df.to_parquet(file_path, index=False)
    assert storage.get_last_timestamp("EUR_USD") is None


def test_get_last_timestamp_is_symbol_specific(storage, valid_tick_gbp):
    storage.save_tick_batch([valid_tick_gbp])
    assert storage.get_last_timestamp("EUR_USD") is None


def test_get_last_timestamp_reads_only_timestamp_column(storage, valid_tick):
    from unittest.mock import patch

    storage.save_tick_batch([valid_tick])
    with patch("pandas.read_parquet", wraps=pd.read_parquet) as mock_read:
        storage.get_last_timestamp("EUR_USD")
        call_kwargs = mock_read.call_args
        assert "columns" in str(call_kwargs)


def test_get_bars_empty_file_returns_none_with_warning(storage, caplog):
    """
    Test that get_bars returns None and logs a warning when the Parquet file exists but is empty.
    """
    import pandas as pd
    import logging

    # Create an empty Parquet file with correct schema
    file_path = storage.processed_dir / "EUR_USD_M1.parquet"
    empty_df = pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_complete",
            "timeframe",
        ]
    )
    empty_df.to_parquet(file_path, index=False)

    # Should return None for empty file
    with caplog.at_level(logging.WARNING, logger="src.data.storage"):
        result = storage.get_bars("EUR_USD", timeframe="M1")

    assert result is None
    assert "exists but is empty" in caplog.text
    assert "EUR_USD" in caplog.text
    assert "M1" in caplog.text


def test_get_bars_empty_file_different_timeframe(storage, caplog):
    """
    Test empty file handling for different timeframe (M5).
    """
    import pandas as pd
    import logging

    # Create an empty Parquet file for M5 timeframe
    file_path = storage.processed_dir / "EUR_USD_M5.parquet"
    empty_df = pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_complete",
            "timeframe",
        ]
    )
    empty_df.to_parquet(file_path, index=False)

    with caplog.at_level(logging.WARNING, logger="src.data.storage"):
        result = storage.get_bars("EUR_USD", timeframe="M5")

    assert result is None
    assert "exists but is empty" in caplog.text


# ==============================================================================
# GROUP 5 -- get_bars
# ==============================================================================


def test_get_bars_returns_none_when_no_file(storage):
    assert storage.get_bars("EUR_USD", timeframe="M1") is None


def test_get_bars_returns_dataframe(storage, valid_bar):
    storage.save_bar(valid_bar)
    df = storage.get_bars("EUR_USD", timeframe="M1")
    assert isinstance(df, pd.DataFrame)
    assert "open" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns


def test_get_bars_applies_max_rows_cap(storage, valid_bar, valid_bar_2):
    storage.save_bar(valid_bar)
    storage.save_bar(valid_bar_2)
    df = storage.get_bars("EUR_USD", timeframe="M1", max_rows=1)
    assert len(df) == 1
    assert df.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15 10:31:00")


def test_get_bars_returns_all_rows_when_no_cap(storage, valid_bar, valid_bar_2):
    storage.save_bar(valid_bar)
    storage.save_bar(valid_bar_2)
    df = storage.get_bars("EUR_USD", timeframe="M1", max_rows=None)
    assert len(df) == 2


def test_get_bars_raises_on_corrupted_file(storage):
    from src.data.storage import StorageError

    file_path = storage.processed_dir / "EUR_USD_M1.parquet"
    file_path.write_bytes(b"corrupted parquet content")
    with pytest.raises(StorageError, match="corrupted"):
        storage.get_bars("EUR_USD", timeframe="M1")


def test_get_bars_output_is_sorted(storage, valid_bar, valid_bar_2):
    storage.save_bar(valid_bar_2)
    storage.save_bar(valid_bar)
    df = storage.get_bars("EUR_USD", timeframe="M1")
    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_get_bars_defaults_to_m1(storage, valid_bar):
    storage.save_bar(valid_bar)
    df = storage.get_bars("EUR_USD")
    assert df is not None and len(df) == 1
