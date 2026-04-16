"""
test_diagnostics.py -- Tests for diagnostic and discovery methods.

Covers:
    Group 6 : get_tick_count, list_symbols
"""

import logging
import pytest
from pathlib import Path

# --- Group 6: get_tick_count ---


def test_get_tick_count_zero_when_no_file(storage):
    assert storage.get_tick_count("EUR_USD") == 0


def test_get_tick_count_correct_after_writes(storage, valid_tick, valid_tick_2):
    storage.save_tick_batch([valid_tick, valid_tick_2])
    assert storage.get_tick_count("EUR_USD") == 2


def test_get_tick_count_returns_zero_on_error(storage):
    file_path = storage.raw_dir / "EUR_USD_ticks.parquet"
    file_path.write_bytes(b"corrupted")
    assert storage.get_tick_count("EUR_USD") == 0


# --- Group 6: list_symbols ---


def test_list_symbols_empty_when_no_files(storage):
    assert storage.list_symbols() == []


def test_list_symbols_returns_all_symbols(storage, valid_tick, valid_tick_gbp):
    storage.save_tick_batch([valid_tick])
    storage.save_tick_batch([valid_tick_gbp])
    result = storage.list_symbols()
    assert "EUR_USD" in result
    assert "GBP_USD" in result


def test_list_symbols_is_sorted(storage, valid_tick, valid_tick_gbp):
    storage.save_tick_batch([valid_tick_gbp])
    storage.save_tick_batch([valid_tick])
    result = storage.list_symbols()
    assert result == sorted(result)


def test_list_symbols_with_mixed_files_and_directories(storage, valid_tick):
    """
    Test that list_symbols correctly filters only .parquet files and extracts symbols.
    """
    # Save a valid tick file
    storage.save_tick_batch([valid_tick])

    # Create noise in the raw directory
    (storage.raw_dir / "README.txt").write_text("not a parquet file")
    (storage.raw_dir / "EUR_USD_ticks.backup").write_text("backup")
    (storage.raw_dir / "subdir").mkdir()

    result = storage.list_symbols()

    assert "EUR_USD" in result
    # Should ignore README, backup, and subdir
    assert len(result) == 1


def test_list_symbols_handles_permission_error(storage, caplog, monkeypatch):
    """
    Test that list_symbols handles PermissionError (e.g. locked folder)
    and logs the appropriate warning.
    """

    def mock_glob_error(*args, **kwargs):
        raise PermissionError("Access denied")

    # Patch the class method to avoid "read-only" instance errors
    monkeypatch.setattr(Path, "glob", mock_glob_error)

    with caplog.at_level(logging.WARNING, logger="src.data.storage"):
        result = storage.list_symbols()

    assert result == []
    assert "Could not list symbols" in caplog.text
    assert "Access denied" in caplog.text


def test_list_symbols_handles_generic_exception(storage, caplog, monkeypatch):
    """
    Test that list_symbols handles unexpected SystemErrors or RuntimeErrors.
    """

    def mock_glob_crash(*args, **kwargs):
        raise RuntimeError("Unexpected system crash")

    monkeypatch.setattr(Path, "glob", mock_glob_crash)

    with caplog.at_level(logging.WARNING, logger="src.data.storage"):
        result = storage.list_symbols()

    assert result == []
    assert "Could not list symbols" in caplog.text
