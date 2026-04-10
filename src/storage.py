"""
storage.py — Azure Blob Storage interface with /tmp fallback.

Tick data layout  : data/{PAIR}/{YYYY-MM}.parquet
Result data layout: data/results/{PAIR}/{YYYY-MM}.parquet

Append operations are atomic at the Parquet level: read existing file (if any),
concat new rows, write back. Thread-safe per pair via threading.Lock.
On blob write failure: spill to /tmp/fallback/{blob_path} and queue a retry.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

# ── Schema definitions ────────────────────────────────────────────────────────

TICK_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ms", tz="UTC")),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("pair", pa.string()),
    ]
)

RESULT_SCHEMA = pa.schema(
    [
        pa.field("signal_time", pa.timestamp("ms", tz="UTC")),
        pa.field("pair", pa.string()),
        pa.field("direction", pa.string()),
        pa.field("confidence", pa.float64()),
        pa.field("expiry_seconds", pa.int32()),
        pa.field("result", pa.string()),
        pa.field("payout", pa.float64()),
    ]
)

FALLBACK_ROOT = Path("/tmp/fallback")


class StorageManager:
    """Thread-safe Parquet append manager backed by Azure Blob Storage."""

    def __init__(
        self, conn_string: str, container_name: str, flush_size: int = 500
    ) -> None:
        self._client = BlobServiceClient.from_connection_string(conn_string)
        self._container = container_name
        self._flush_size = flush_size

        # In-memory tick buffers: {pair -> list[dict]}
        self._tick_buffers: dict[str, list[dict]] = defaultdict(list)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Pending retry queue: list of (blob_path, pyarrow.Table)
        self._retry_queue: list[tuple[str, pa.Table]] = []
        self._retry_lock = threading.Lock()

        self._ensure_container()

    # ── Public API ─────────────────────────────────────────────────────────────

    def append_ticks(self, pair: str, df: pd.DataFrame) -> None:
        """Buffer ticks; flush to blob when buffer reaches flush_size."""
        if df.empty:
            return
        records = df.to_dict("records")
        with self._locks[pair]:
            self._tick_buffers[pair].extend(records)
            if len(self._tick_buffers[pair]) >= self._flush_size:
                self._flush_ticks(pair)

    def force_flush(self, pair: str) -> None:
        """Force an immediate flush regardless of buffer size."""
        with self._locks[pair]:
            if self._tick_buffers[pair]:
                self._flush_ticks(pair)

    def append_result(self, result: dict[str, Any]) -> None:
        """Write a single trade result record to blob storage."""
        pair = result["pair"]
        ts = result.get("signal_time", datetime.now(timezone.utc))
        if isinstance(ts, str):
            ts = pd.Timestamp(ts, tz="UTC")
        blob_path = self._result_blob_path(pair, ts)

        row = {
            "signal_time": pd.Timestamp(ts, tz="UTC"),
            "pair": str(pair),
            "direction": str(result.get("direction", "")),
            "confidence": float(result.get("confidence", 0.0)),
            "expiry_seconds": int(result.get("expiry_seconds", 0)),
            "result": str(result.get("result", "")),
            "payout": float(result.get("payout", 0.0)),
        }
        new_df = pd.DataFrame([row])
        self._append_parquet(blob_path, new_df, RESULT_SCHEMA)

    def read_ticks(self, pair: str, months: int = 12) -> pd.DataFrame:
        """Read up to `months` of tick data from blob for the given pair."""
        now = datetime.now(timezone.utc)
        frames: list[pd.DataFrame] = []
        for m in range(months):
            year = now.year if now.month - m > 0 else now.year - 1
            month = (now.month - m - 1) % 12 + 1
            blob_path = self._tick_blob_path(pair, year, month)
            df = self._read_parquet(blob_path)
            if df is not None:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "bid", "ask", "spread", "pair"])
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined

    def read_results(self, pair: str) -> pd.DataFrame:
        """Read all available result records for a pair."""
        now = datetime.now(timezone.utc)
        frames: list[pd.DataFrame] = []
        # Read last 12 months
        for m in range(12):
            year = now.year if now.month - m > 0 else now.year - 1
            month = (now.month - m - 1) % 12 + 1
            blob_path = self._result_blob_path(
                pair, datetime(year, month, 1, tzinfo=timezone.utc)
            )
            df = self._read_parquet(blob_path)
            if df is not None:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=list(RESULT_SCHEMA.names))
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values("signal_time")
            .reset_index(drop=True)
        )

    def blob_exists(self, blob_path: str) -> bool:
        """Check if a blob exists (used by backfill to skip already-downloaded data)."""
        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_path
            )
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False

    def write_raw_parquet(
        self, blob_path: str, df: pd.DataFrame, schema: pa.Schema | None = None
    ) -> None:
        """Write a DataFrame to a blob path directly (used by backfill)."""
        self._append_parquet(blob_path, df, schema)

    def flush_retry_queue(self) -> None:
        """Attempt to retry previously failed blob writes."""
        with self._retry_lock:
            if not self._retry_queue:
                return
            remaining: list[tuple[str, pa.Table]] = []
            for blob_path, table in self._retry_queue:
                try:
                    self._upload_table(blob_path, table)
                    logger.info({"event": "retry_success", "blob_path": blob_path})
                except Exception as exc:
                    logger.warning(
                        {
                            "event": "retry_failed",
                            "blob_path": blob_path,
                            "error": str(exc),
                        }
                    )
                    remaining.append((blob_path, table))
            self._retry_queue = remaining

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _flush_ticks(self, pair: str) -> None:
        """Must be called with self._locks[pair] held."""
        records = self._tick_buffers[pair]
        if not records:
            return
        df = pd.DataFrame(records)
        # Coerce timestamp to UTC
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["pair"] = pair

        now = datetime.now(timezone.utc)
        blob_path = self._tick_blob_path(pair, now.year, now.month)
        self._append_parquet(blob_path, df, TICK_SCHEMA)
        self._tick_buffers[pair] = []

    def _append_parquet(
        self, blob_path: str, new_df: pd.DataFrame, schema: pa.Schema | None
    ) -> None:
        """Read existing parquet from blob, concat new_df, write back."""
        existing = self._read_parquet(blob_path)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df.copy()

        # De-duplicate on timestamp within a pair (ticks)
        ts_col = "timestamp" if "timestamp" in combined.columns else "signal_time"
        if ts_col in combined.columns:
            combined = (
                combined.drop_duplicates(subset=[ts_col])
                .sort_values(ts_col)
                .reset_index(drop=True)
            )

        if schema is not None:
            try:
                table = pa.Table.from_pandas(
                    combined, schema=schema, preserve_index=False
                )
            except Exception:
                table = pa.Table.from_pandas(combined, preserve_index=False)
        else:
            table = pa.Table.from_pandas(combined, preserve_index=False)

        try:
            self._upload_table(blob_path, table)
            logger.debug(
                {
                    "event": "parquet_written",
                    "blob_path": blob_path,
                    "rows": len(combined),
                }
            )
        except (AzureError, Exception) as exc:
            logger.error(
                {
                    "event": "blob_write_failed",
                    "blob_path": blob_path,
                    "error": str(exc),
                    "action": "spilling_to_tmp",
                }
            )
            self._spill_to_tmp(blob_path, table)
            with self._retry_lock:
                self._retry_queue.append((blob_path, table))

    def _upload_table(self, blob_path: str, table: pa.Table) -> None:
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        blob_client = self._client.get_blob_client(
            container=self._container, blob=blob_path
        )
        blob_client.upload_blob(buf, overwrite=True)

    def _read_parquet(self, blob_path: str) -> pd.DataFrame | None:
        # First check /tmp fallback
        fallback = FALLBACK_ROOT / blob_path
        if fallback.exists():
            try:
                return pd.read_parquet(fallback)
            except Exception:
                pass
        # Read from Azure
        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_path
            )
            data = blob_client.download_blob().readall()
            buf = io.BytesIO(data)
            return pd.read_parquet(buf)
        except Exception:
            return None

    def _spill_to_tmp(self, blob_path: str, table: pa.Table) -> None:
        dest = FALLBACK_ROOT / blob_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Merge with existing tmp file if present
        if dest.exists():
            try:
                existing = pq.read_table(str(dest))
                table = pa.concat_tables([existing, table])
            except Exception:
                pass
        pq.write_table(table, str(dest), compression="snappy")
        logger.info({"event": "spilled_to_tmp", "path": str(dest)})

    def _ensure_container(self) -> None:
        try:
            container_client = self._client.get_container_client(self._container)
            container_client.get_container_properties()
        except Exception:
            try:
                self._client.create_container(self._container)
                logger.info(
                    {"event": "container_created", "container": self._container}
                )
            except Exception as exc:
                logger.warning({"event": "container_create_warning", "error": str(exc)})

    def save_model(
        self, model_data: bytes, model_name: str = "model_manager.pkl"
    ) -> None:
        """Save model to blob storage."""
        blob_path = f"models/{model_name}"
        blob_client = self._client.get_blob_client(
            container=self._container, blob=blob_path
        )
        blob_client.upload_blob(model_data, overwrite=True)
        logger.info({"event": "model_saved_to_blob", "blob_path": blob_path})

    def load_model(self, model_name: str = "model_manager.pkl") -> bytes | None:
        """Load model from blob storage."""
        blob_path = f"models/{model_name}"
        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_path
            )
            data = blob_client.download_blob().readall()
            logger.info({"event": "model_loaded_from_blob", "blob_path": blob_path})
            return data
        except Exception:
            logger.info({"event": "no_model_in_blob", "blob_path": blob_path})
            return None

    @staticmethod
    def _tick_blob_path(pair: str, year: int, month: int) -> str:
        return f"data/{pair}/{year}-{month:02d}.parquet"

    @staticmethod
    def _result_blob_path(pair: str, ts: datetime) -> str:
        return f"data/results/{pair}/{ts.year}-{ts.month:02d}.parquet"
