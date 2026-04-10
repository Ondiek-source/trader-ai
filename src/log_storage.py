"""
log_storage.py — Persistent log writer to Azure Blob Storage
"""

import logging
import threading
import time
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient


class BlobLogHandler(logging.Handler):
    """Custom logging handler that writes logs to Azure Blob Storage."""

    def __init__(self, conn_string: str, container_name: str, flush_interval: int = 30):
        super().__init__()
        self.conn_string = conn_string
        self.container_name = container_name
        self.flush_interval = flush_interval  # Flush every 30 seconds
        self._client = None
        self._buffer = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self._init_client()
        self._start_timer()

    def _init_client(self):
        try:
            self._client = BlobServiceClient.from_connection_string(self.conn_string)
        except Exception as e:
            print(f"Failed to init blob client: {e}")

    def _start_timer(self):
        def timer_callback():
            while True:
                time.sleep(self.flush_interval)
                self.flush()

        thread = threading.Thread(target=timer_callback, daemon=True)
        thread.start()

    def emit(self, record):
        try:
            msg = self.format(record)
            with self._lock:
                self._buffer.append(msg + "\n")
                # Flush if buffer is large or time elapsed
                if (
                    len(self._buffer) >= 10
                    or (time.time() - self._last_flush) >= self.flush_interval
                ):
                    self._flush()
        except Exception:
            self.handleError(record)

    def _flush(self):
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
            self._last_flush = time.time()
        except Exception as e:
            print(f"Failed to flush logs: {e}")

    def flush(self):
        with self._lock:
            self._flush()

    def close(self):
        self.flush()
        super().close()
