"""
test_storage_init.py -- Tests for Storage construction and directory provisioning.

Covers:
    Group 1 : Construction, Hard-Exit on Failure, Path Helpers
"""

import threading
import pytest
import os
from unittest.mock import patch, MagicMock


def test_storage_provisions_directories_on_init(tmp_path, minimal_valid_env):
    """Verify that the internal provisioning logic creates the folder structure."""
    from src.data.storage import Storage

    with patch.dict(os.environ, minimal_valid_env, clear=True):
        store = Storage.__new__(Storage)
        store._lock = threading.Lock()
        store._settings = None  # type: ignore
        store.root_dir = tmp_path / "data"
        store.raw_dir = tmp_path / "data" / "raw"
        store.processed_dir = tmp_path / "data" / "processed"

        # This will create the folders and the .write_test files
        store._provision_infrastructure()

    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "processed").exists()


def test_storage_real_initialization(tmp_path):
    """
    Test that Storage.__init__ properly initializes with valid config.
    Patches get_settings() so the test is isolated from the real .env file
    and from whatever the settings singleton may have cached from prior tests.
    """
    from src.data.storage import Storage

    mock_cfg = MagicMock()
    mock_cfg.data_mode = "LOCAL"

    # Create a temporary directory structure to act as project root
    project_root = tmp_path
    mock_file_path = project_root / "src" / "data" / "storage.py"
    mock_file_path.parent.mkdir(parents=True, exist_ok=True)

    with patch("src.data.storage.get_settings", return_value=mock_cfg), \
         patch("src.data.storage.__file__", str(mock_file_path)):
        storage = Storage()

    # Verify __init__ set up the attributes correctly
    assert hasattr(storage, "_settings")
    assert hasattr(storage, "_lock")
    assert storage.root_dir == project_root / "data"
    assert storage.raw_dir.exists()
    assert storage._settings is not None


def test_storage_init_exits_on_permission_error(minimal_valid_env):
    """
    DICTATOR TEST: Verify the system terminates if it cannot create directories.
    """
    with patch.dict(os.environ, minimal_valid_env, clear=True):
        from src.data.storage import Storage

        # Force an OSError during directory creation
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(SystemExit) as excinfo:
                Storage()

            # Verify exit code is 1 (Fail-Fast)
            assert excinfo.value.code == 1


def test_storage_init_exits_on_write_check_failure(minimal_valid_env):
    """
    DICTATOR TEST: Verify the system terminates if directories exist but aren't writable.
    """
    with patch.dict(os.environ, minimal_valid_env, clear=True):
        from src.data.storage import Storage

        # Force an error when trying to touch the .write_test canary file
        with patch("pathlib.Path.touch", side_effect=OSError("Read-only file system")):
            with pytest.raises(SystemExit) as excinfo:
                Storage()

            assert excinfo.value.code == 1


def test_storage_provision_is_idempotent(storage):
    """Ensure running provisioning twice on existing folders doesn't crash."""
    # The fixture 'storage' already initialized once.
    storage._provision_infrastructure()
    assert storage.raw_dir.exists()
    assert storage.processed_dir.exists()


def test_raw_path_naming_convention(storage):
    """Verify symbol-to-parquet file mapping."""
    path = storage._raw_path("EUR_USD")
    assert path.name == "EUR_USD_ticks.parquet"
    assert path.parent == storage.raw_dir


def test_processed_path_naming_convention(storage):
    """Verify bar aggregation file mapping."""
    path = storage._processed_path("EUR_USD", "M1")
    assert path.name == "EUR_USD_M1.parquet"
    assert path.parent == storage.processed_dir


@pytest.mark.parametrize("tf", ["M1", "M5", "M15"])
def test_processed_path_all_timeframes(storage, tf):
    """Ensure all timeframe enums generate correct paths."""
    path = storage._processed_path("GBP_USD", tf)
    assert path.name == f"GBP_USD_{tf}.parquet"


def test_storage_error_has_attributes():
    """
    Verify StorageError still functions as a container (used for non-fatal
    runtime write errors later).
    """
    from src.data.storage import StorageError

    err = StorageError("test failure", symbol="EUR_USD", path="/data/raw/x.parquet")
    assert err.symbol == "EUR_USD"
    assert err.path == "/data/raw/x.parquet"
    assert "test failure" in str(err)
