"""
log_storage.py — Persistent log writer to Azure Blob Storage
"""

import logging
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient


class BlobLogHandler(logging.Handler):
    """Custom logging handler that writes logs to Azure Blob Storage."""

    def __init__(self, conn_string: str, container_name: str):
        super().__init__()
        self.conn_string = conn_string
        self.container_name = container_name
        self._client = None
        self._buffer = []
        self._buffer_size = 50

    def _init_client(self):
        try:
            self._client = BlobServiceClient.from_connection_string(self.conn_string)
        except Exception as e:
            print(f"Failed to init blob client: {e}")

    def emit(self, record):
        try:
            msg = self.format(record)
            self._buffer.append(msg + "\n")

            if len(self._buffer) >= self._buffer_size:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self):
        if not self._buffer or not self._client:
            return

        try:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            blob_name = f"logs/{date_str}/trader-ai-engine.log"

            blob_client = self._client.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            existing = ""
            try:
                existing_data = blob_client.download_blob().readall()
                existing = existing_data.decode("utf-8")
            except:
                pass

            new_content = existing + "".join(self._buffer)
            blob_client.upload_blob(new_content, overwrite=True)
            self._buffer = []
        except Exception as e:
            print(f"Failed to flush logs: {e}")

    def close(self):
        self.flush()
        super().close()
