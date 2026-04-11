"""
log_storage.py — Persistent log writer to Azure Blob Storage.

Provides :class:`BlobLogHandler`, a :class:`logging.Handler` subclass that
buffers log records and periodically uploads them as a single blob to Azure
Blob Storage.  Each container instance gets its own daily log file to avoid
write conflicts.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient


class BlobLogHandler(logging.Handler):
    """
    Logging handler that appends records to a daily Azure Blob.

    Log files are stored under ``logs/YYYY-MM-DD/<prefix>-<instance>.log``.
    The *instance* suffix (a short UUID) ensures concurrent container restarts
    on the same day write to separate blobs.

    Uses ``sys.stderr`` for internal error reporting to avoid infinite
    recursion (since this handler is itself part of the logging pipeline).

    Args:
        conn_string: Azure Storage connection string.
        container_name: Blob container name.
        buffer_size: Number of records to accumulate before uploading.
        blob_prefix: Filename prefix (default ``"trader-ai-engine"``).

    Example::

        handler = BlobLogHandler(conn_string, "traderai", buffer_size=100)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
    """

    def __init__(
        self,
        conn_string: str,
        container_name: str,
        buffer_size: int = 10,
        blob_prefix: str = "trader-ai-engine",
    ) -> None:
        super().__init__()
        self._container_name: str = container_name
        self._blob_prefix: str = blob_prefix
        self._instance_id: str = uuid.uuid4().hex[:8]
        self._buffer_size: int = buffer_size
        self._buffer: list[str] = []
        self._client: BlobServiceClient | None = None

        try:
            self._client = BlobServiceClient.from_connection_string(conn_string)
        except Exception:
            # Use sys.stderr directly — logging here would recurse.
            sys.stderr.write("[BlobLogHandler] Failed to initialise blob client.\n")
            self._client = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _blob_name(self) -> str:
        """
        Return today's blob path.

        Changes automatically at UTC midnight since the date is computed
        on each access.

        Returns:
            Blob path like ``logs/2025-01-15/trader-ai-engine-a1b2c3d4.log``.
        """
        date_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"logs/{date_str}/{self._blob_prefix}-{self._instance_id}.log"

    @property
    def buffer_length(self) -> int:
        """Number of records currently buffered (not yet flushed)."""
        return len(self._buffer)

    # ── logging.Handler interface ─────────────────────────────────────────────

    def emit(self, record: logging.LogRecord) -> None:
        """
        Buffer a formatted log record and flush when *buffer_size* is reached.

        Args:
            record: Log record from the logging framework.
        """
        try:
            self._buffer.append(self.format(record) + "\n")
            if len(self._buffer) >= self._buffer_size:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """
        Upload buffered records to Azure Blob Storage.

        If the blob already exists (earlier records from the same instance),
        new records are appended by reading the existing content and
        re-uploading the concatenation.

        If the Azure client is unavailable, the buffer is silently dropped
        with a stderr warning.  On upload failure, records are retained in
        the buffer for the next flush attempt.
        """
        if not self._buffer:
            return

        if self._client is None:
            sys.stderr.write(
                f"[BlobLogHandler] Dropping {len(self._buffer)} records "
                f"— no blob client.\n"
            )
            self._buffer = []
            return

        blob_name: str = self._blob_name
        payload: str = "".join(self._buffer)

        try:
            blob_client = self._client.get_blob_client(
                container=self._container_name, blob=blob_name
            )

            existing: str = ""
            try:
                existing = blob_client.download_blob().readall().decode("utf-8")
            except ResourceNotFoundError:
                pass  # first write of the day — fine

            blob_client.upload_blob(existing + payload, overwrite=True)
            self._buffer = []

        except AzureError:
            # Don't clear buffer — retry on next flush
            sys.stderr.write(
                f"[BlobLogHandler] Azure error flushing "
                f"{len(self._buffer)} records — retaining for retry.\n"
            )
        except Exception:
            sys.stderr.write(
                f"[BlobLogHandler] Unexpected error flushing "
                f"{len(self._buffer)} records — retaining for retry.\n"
            )

    def close(self) -> None:
        """Flush remaining records and release resources."""
        try:
            self.flush()
        finally:
            super().close()

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BlobLogHandler(container={self._container_name!r}, "
            f"blob={self._blob_name!r}, buffered={self.buffer_length})"
        )
