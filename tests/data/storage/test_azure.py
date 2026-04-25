"""
tests/data/storage/test_azure.py

Tests for the Azure Blob integration in Storage:
    - _init_azure_client()  : CLOUD success path and failure/exit path
    - sync_to_azure()       : LOCAL skip, file-not-found, success, upload error
    - pull_from_azure()     : LOCAL skip, success, download error
"""
import pytest
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cloud_storage(tmp_path):
    """
    Return a Storage instance wired to a CLOUD mock config.
    __init__ is bypassed; _container_client is set manually so each test
    can control whether it is None (LOCAL) or a mock ContainerClient (CLOUD).
    """
    from src.data.storage import Storage

    store = Storage.__new__(Storage)
    mock_cfg = MagicMock()
    mock_cfg.data_mode = "CLOUD"
    mock_cfg.azure_storage_conn = "DefaultEndpointsProtocol=https;AccountName=test"
    mock_cfg.container_name = "traderai"
    store._settings = mock_cfg
    store._lock = threading.Lock()
    store.root_dir = tmp_path / "data"
    store.raw_dir = tmp_path / "data" / "raw"
    store.processed_dir = tmp_path / "data" / "processed"
    store._container_client = None  # overridden per-test
    return store


# ── _init_azure_client ─────────────────────────────────────────────────────────

def test_init_azure_client_returns_container_client_on_success(tmp_path):
    """CLOUD mode + reachable container → returns a ContainerClient."""
    store = _cloud_storage(tmp_path)
    mock_client = MagicMock()

    with patch("src.data.storage.BlobServiceClient") as mock_svc:
        mock_svc.from_connection_string.return_value.get_container_client.return_value = mock_client
        mock_client.get_container_properties.return_value = {}

        result = store._init_azure_client()

    assert result is mock_client


def test_init_azure_client_exits_on_connection_failure(tmp_path):
    """CLOUD mode + unreachable container → sys.exit(1) (Dictator pattern)."""
    store = _cloud_storage(tmp_path)

    with patch("src.data.storage.BlobServiceClient") as mock_svc:
        mock_svc.from_connection_string.return_value \
            .get_container_client.return_value \
            .get_container_properties.side_effect = Exception("network unreachable")

        with pytest.raises(SystemExit) as exc_info:
            store._init_azure_client()

    assert exc_info.value.code == 1


def test_init_azure_client_returns_none_in_local_mode(tmp_path):
    """LOCAL mode → returns None immediately, never touches Azure SDK."""
    store = _cloud_storage(tmp_path)
    store._settings.data_mode = "LOCAL"

    with patch("src.data.storage.BlobServiceClient") as mock_svc:
        result = store._init_azure_client()

    mock_svc.assert_not_called()
    assert result is None


# ── sync_to_azure ──────────────────────────────────────────────────────────────

def test_sync_to_azure_skips_when_local_mode(tmp_path):
    """_container_client is None (LOCAL) → returns False without SDK call."""
    store = _cloud_storage(tmp_path)
    store._container_client = None

    result = store.sync_to_azure(tmp_path / "some_file.parquet")
    assert result is False


def test_sync_to_azure_returns_false_when_file_missing(tmp_path):
    """File does not exist locally → returns False."""
    store = _cloud_storage(tmp_path)
    store._container_client = MagicMock()

    result = store.sync_to_azure(tmp_path / "nonexistent.parquet")
    assert result is False


def test_sync_to_azure_returns_true_on_success(tmp_path):
    """File exists + upload succeeds → returns True."""
    store = _cloud_storage(tmp_path)
    mock_container = MagicMock()
    store._container_client = mock_container

    local_file = tmp_path / "model.pkl"
    local_file.write_bytes(b"fake model data")

    with patch("builtins.open", mock_open(read_data=b"fake model data")):
        result = store.sync_to_azure(local_file)

    assert result is True
    mock_container.get_blob_client.assert_called_once()


def test_sync_to_azure_uses_custom_blob_name(tmp_path):
    """When blob_name is supplied it is used as the target key."""
    store = _cloud_storage(tmp_path)
    mock_container = MagicMock()
    store._container_client = mock_container

    local_file = tmp_path / "model.pkl"
    local_file.write_bytes(b"data")

    with patch("builtins.open", mock_open(read_data=b"data")):
        store.sync_to_azure(local_file, blob_name="models/production/model.pkl")

    mock_container.get_blob_client.assert_called_once_with("models/production/model.pkl")


def test_sync_to_azure_returns_false_on_upload_error(tmp_path):
    """SDK upload raises → returns False (non-fatal, logged)."""
    store = _cloud_storage(tmp_path)
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value.upload_blob.side_effect = Exception("network error")
    store._container_client = mock_container

    local_file = tmp_path / "model.pkl"
    local_file.write_bytes(b"data")

    with patch("builtins.open", mock_open(read_data=b"data")):
        result = store.sync_to_azure(local_file)

    assert result is False


# ── pull_from_azure ────────────────────────────────────────────────────────────

def test_pull_from_azure_skips_when_local_mode(tmp_path):
    """_container_client is None (LOCAL) → returns False without SDK call."""
    store = _cloud_storage(tmp_path)
    store._container_client = None

    result = store.pull_from_azure("models/model.pkl", tmp_path / "model.pkl")
    assert result is False


def test_pull_from_azure_returns_true_on_success(tmp_path):
    """Blob download succeeds → returns True and writes file."""
    store = _cloud_storage(tmp_path)
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value.download_blob.return_value.readall.return_value = b"model bytes"
    store._container_client = mock_container

    local_path = tmp_path / "model.pkl"

    with patch("builtins.open", mock_open()) as mocked_file:
        result = store.pull_from_azure("models/model.pkl", local_path)

    assert result is True
    mocked_file().write.assert_called_once_with(b"model bytes")


def test_pull_from_azure_returns_false_on_download_error(tmp_path):
    """SDK download raises → returns False (non-fatal, logged)."""
    store = _cloud_storage(tmp_path)
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value.download_blob.side_effect = Exception("blob not found")
    store._container_client = mock_container

    result = store.pull_from_azure("models/missing.pkl", tmp_path / "missing.pkl")
    assert result is False
