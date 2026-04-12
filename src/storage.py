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
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.core.pipeline.transport import RequestsTransport
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

# ── Schema definitions ────────────────────────────────────────────────────────

TICK_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ms", tz="UTC")),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("pair", pa.string()),
    ]
)

RESULT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("received_at", pa.timestamp("ms", tz="UTC")),
        pa.field("pair", pa.string()),
        pa.field("direction", pa.string()),
        pa.field("result", pa.string()),
        pa.field("payout", pa.float64()),
        pa.field("stake", pa.float64()),
        pa.field("signal_id", pa.string()),
    ]
)

FALLBACK_ROOT: Path = Path("/tmp/fallback")


class StorageManager:
    """
    Thread-safe Parquet append manager backed by Azure Blob Storage.

    Tick data is buffered in memory and flushed to blob storage when the
    buffer reaches *flush_size*.  Trade results are written immediately
    (no buffering).

    On blob write failure, data is spilled to ``/tmp/fallback/`` and queued
    for retry via :meth:`flush_retry_queue`.

    Args:
        conn_string: Azure Storage connection string.
        container_name: Blob container name (created if absent).
        flush_size: Number of tick records to buffer before auto-flush.
    """

    def __init__(
        self,
        conn_string: str,
        container_name: str,
        flush_size: int = 500,
    ) -> None:
        transport = RequestsTransport(
            pool_connections=20,  # Number of pooled connections
            pool_maxsize=50,  # Max total connections
        )
        self._client: BlobServiceClient = BlobServiceClient.from_connection_string(
            conn_string, transport=transport
        )
        self._container: str = container_name
        self._flush_size: int = flush_size

        # In-memory tick buffers: {pair -> list[dict]}
        self._tick_buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Pending retry queue: list of (blob_path, pyarrow.Table)
        self._retry_queue: list[tuple[str, pa.Table]] = []
        self._retry_lock: threading.Lock = threading.Lock()

        self._ensure_container()

    # ── Public API ─────────────────────────────────────────────────────────────

    def append_ticks(self, pair: str, df: pd.DataFrame) -> None:
        """
        Buffer tick records; auto-flush when buffer reaches :attr:`_flush_size`.

        Args:
            pair: Currency pair (e.g. ``"EUR_USD"``).
            df: DataFrame with tick columns matching :data:`TICK_SCHEMA`.
        """
        if df.empty:
            return
        records: list[dict[str, Any]] = cast(
            list[dict[str, Any]], df.to_dict("records")
        )
        with self._locks[pair]:
            self._tick_buffers[pair].extend(records)
            if len(self._tick_buffers[pair]) >= self._flush_size:
                self._flush_ticks(pair)

    def force_flush(self, pair: str) -> None:
        """
        Force an immediate flush regardless of buffer size.

        Args:
            pair: Currency pair to flush.
        """
        with self._locks[pair]:
            if self._tick_buffers[pair]:
                self._flush_ticks(pair)

    def force_flush_all(self) -> None:
        """Flush every buffered pair."""
        for pair in list(self._tick_buffers.keys()):
            self.force_flush(pair)

    def append_result(self, result: dict[str, Any]) -> None:
        """
        Write a single trade result record to blob storage.

        Accepts the dict format returned by
        :meth:`~quotex_reader.QuotexReader._make_result`::

            {
                "signal_id": "...",
                "pair": "EUR_USD",
                "direction": "UP",
                "result": "win",
                "payout": 5.50,
                "stake": 0.0,
                "received_at": "2025-01-15T12:00:00+00:00",
            }

        Args:
            result: Trade result dict.
        """
        pair: str = result.get("pair", "")
        if not pair:
            logger.warning({"event": "append_result_no_pair", "result": result})
            return

        # Parse received_at — could be ISO string or datetime
        ts_raw: Any = result.get("received_at", datetime.now(timezone.utc))
        ts: pd.Timestamp
        if isinstance(ts_raw, str):
            ts = pd.Timestamp(ts_raw, tz="UTC")
        elif isinstance(ts_raw, datetime):
            ts = pd.Timestamp(ts_raw, tz="UTC")
        else:
            ts = pd.Timestamp(datetime.now(timezone.utc), tz="UTC")

        blob_path: str = self._result_blob_path(pair, ts.to_pydatetime())

        row: dict[str, Any] = {
            "received_at": ts,
            "pair": str(pair),
            "direction": str(result.get("direction", "")),
            "result": str(result.get("result", "")),
            "payout": float(result.get("payout", 0.0)),
            "stake": float(result.get("stake", 0.0)),
            "signal_id": str(result.get("signal_id", "")),
        }
        new_df: pd.DataFrame = pd.DataFrame([row])
        self._append_parquet(blob_path, new_df, RESULT_SCHEMA)

    def read_ticks(self, pair: str, months: int = 12) -> pd.DataFrame:
        """
        Read up to *months* of tick data from blob for the given pair.

        Args:
            pair: Currency pair.
            months: How many recent months to load.

        Returns:
            Sorted DataFrame with columns matching :data:`TICK_SCHEMA`,
            or an empty DataFrame with those columns.
        """
        return self._read_months(
            pair,
            months,
            path_fn=lambda p, y, m: self._tick_blob_path(p, y, m),
            columns=list(TICK_SCHEMA.names),
            sort_col="timestamp",
        )

    def read_results(self, pair: str, months: int = 12) -> pd.DataFrame:
        """
        Read up to *months* of result records for the given pair.

        Args:
            pair: Currency pair.
            months: How many recent months to load.

        Returns:
            Sorted DataFrame with columns matching :data:`RESULT_SCHEMA`,
            or an empty DataFrame with those columns.
        """
        return self._read_months(
            pair,
            months,
            path_fn=lambda p, y, m: self._result_blob_path(
                p, datetime(y, m, 1, tzinfo=timezone.utc)
            ),
            columns=list(RESULT_SCHEMA.names),
            sort_col="received_at",
        )

    def blob_exists(self, blob_path: str) -> bool:
        """
        Check if a blob exists.

        Args:
            blob_path: Path inside the container.

        Returns:
            ``True`` if the blob can be stat'd, ``False`` otherwise.
        """
        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_path
            )
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception:
            return False

    def write_raw_parquet(
        self,
        blob_path: str,
        df: pd.DataFrame,
        schema: pa.Schema | None = None,
    ) -> None:
        """
        Write a DataFrame to a blob path directly (used by backfill).

        Unlike :meth:`_append_parquet`, this does **not** read the existing
        file first — it overwrites the blob entirely.

        Args:
            blob_path: Destination blob path.
            df: Data to write.
            schema: Optional PyArrow schema for type enforcement.
        """
        if df.empty:
            return

        if schema is not None:
            try:
                table: pa.Table = pa.Table.from_pandas(
                    df, schema=schema, preserve_index=False
                )
            except Exception:
                table = pa.Table.from_pandas(df, preserve_index=False)
        else:
            table = pa.Table.from_pandas(df, preserve_index=False)

        try:
            self._upload_table(blob_path, table)
            self._clear_fallback(blob_path)
            logger.debug(
                {
                    "event": "raw_parquet_written",
                    "blob_path": blob_path,
                    "rows": len(df),
                    "function": "write_raw_parquet",
                    "file": "storage.py",
                }
            )
        except Exception as exc:
            logger.error(
                {
                    "event": "raw_parquet_write_failed",
                    "blob_path": blob_path,
                    "error": str(exc),
                    "function": "write_raw_parquet",
                    "file": "storage.py",
                }
            )
            self._spill_to_tmp(blob_path, table)
            with self._retry_lock:
                self._retry_queue.append((blob_path, table))

    def flush_retry_queue(self) -> None:
        """
        Attempt to retry previously failed blob writes.

        Successfully retried entries are removed.  Entries that still fail
        stay in the queue for the next call.
        """
        with self._retry_lock:
            if not self._retry_queue:
                return
            remaining: list[tuple[str, pa.Table]] = []
            for blob_path, table in self._retry_queue:
                try:
                    self._upload_table(blob_path, table)
                    self._clear_fallback(blob_path)
                    logger.info({"event": "retry_success", "blob_path": blob_path})
                except Exception as exc:
                    logger.warning(
                        {
                            "event": "retry_failed",
                            "blob_path": blob_path,
                            "error": str(exc),
                            "function": "flush_retry_queue",
                            "file": "storage.py",
                        }
                    )
                    remaining.append((blob_path, table))
            self._retry_queue = remaining

    # ── Model persistence ─────────────────────────────────────────────────────

    def save_model(
        self,
        model_data: bytes,
        model_name: str = "model_manager.pkl",
    ) -> None:
        """
        Persist a serialised model to blob storage.

        Args:
            model_data: Raw bytes (e.g. ``pickle.dumps(model)``).
            model_name: Filename under ``models/``.
        """
        blob_path: str = f"models/{model_name}"
        blob_client = self._client.get_blob_client(
            container=self._container, blob=blob_path
        )
        blob_client.upload_blob(model_data, overwrite=True)
        logger.info({"event": "model_saved_to_blob", "blob_path": blob_path})

    def load_model(self, model_name: str = "model_manager.pkl") -> bytes | None:
        """
        Load a serialised model from blob storage.

        Args:
            model_name: Filename under ``models/``.

        Returns:
            Raw bytes, or ``None`` if the blob doesn't exist.
        """
        blob_path: str = f"models/{model_name}"
        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_path
            )
            data: bytes = blob_client.download_blob().readall()
            logger.info({"event": "model_loaded_from_blob", "blob_path": blob_path})
            return data
        except ResourceNotFoundError:
            logger.info({"event": "no_model_in_blob", "blob_path": blob_path})
            return None
        except Exception as exc:
            logger.warning(
                {
                    "event": "model_load_error",
                    "blob_path": blob_path,
                    "error": str(exc),
                    "function": "load_model",
                    "file": "storage.py",
                }
            )
            return None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _read_months(
        self,
        pair: str,
        months: int,
        path_fn: Callable[[str, int, int], str],
        columns: list[str],
        sort_col: str,
    ) -> pd.DataFrame:
        """
        Generic month-range reader shared by :meth:`read_ticks` and
        :meth:`read_results`.

        Iterates backwards from the current month, reading each month's
        parquet file and concatenating the results.

        Args:
            pair: Currency pair.
            months: Number of recent months to scan.
            path_fn: ``(pair, year, month) -> blob_path`` callable.
            columns: Column names for the empty-DataFrame fallback.
            sort_col: Column to sort the final result on.

        Returns:
            Sorted concatenated DataFrame, or empty DataFrame with *columns*.
        """
        now: datetime = datetime.now(timezone.utc)
        frames: list[pd.DataFrame] = []
        for m in range(months):
            year: int = now.year if now.month - m > 0 else now.year - 1
            month: int = (now.month - m - 1) % 12 + 1
            blob_path: str = path_fn(pair, year, month)
            df: pd.DataFrame | None = self._read_parquet(blob_path)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=columns)
        combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
        return combined.sort_values(sort_col).reset_index(drop=True)

    def _flush_ticks(self, pair: str) -> None:
        """
        Flush buffered tick records to blob.

        Must be called with ``_locks[pair]`` held.

        Args:
            pair: Currency pair to flush.
        """
        records: list[dict[str, Any]] = self._tick_buffers[pair]
        if not records:
            return
        df: pd.DataFrame = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["pair"] = pair

        now: datetime = datetime.now(timezone.utc)
        blob_path: str = self._tick_blob_path(pair, now.year, now.month)
        self._append_parquet(blob_path, df, TICK_SCHEMA)
        self._tick_buffers[pair] = []

    def _append_parquet(
        self,
        blob_path: str,
        new_df: pd.DataFrame,
        schema: pa.Schema | None,
    ) -> None:
        """
        Read existing parquet, append *new_df*, write back.

        Deduplicates on the timestamp column.  On blob write failure the
        table is spilled to ``/tmp/fallback/`` and queued for retry.

        Args:
            blob_path: Blob path to read/modify/write.
            new_df: New rows to append.
            schema: Optional PyArrow schema for type enforcement.
        """
        existing: pd.DataFrame | None = self._read_parquet(blob_path)
        if existing is not None and not existing.empty:
            combined: pd.DataFrame = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df.copy()

        ts_col: str = "timestamp" if "timestamp" in combined.columns else "received_at"
        if ts_col in combined.columns:
            combined = (
                combined.drop_duplicates(subset=[ts_col])
                .sort_values(ts_col)
                .reset_index(drop=True)
            )

        table: pa.Table
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
            self._clear_fallback(blob_path)
            logger.debug(
                {
                    "event": "parquet_written",
                    "blob_path": blob_path,
                    "rows": len(combined),
                    "function": "_append_parquet",
                    "file": "storage.py",
                }
            )
        except Exception as exc:
            logger.error(
                {
                    "event": "blob_write_failed",
                    "blob_path": blob_path,
                    "error": str(exc),
                    "action": "spilling_to_tmp",
                    "function": "_append_parquet",
                    "file": "storage.py",
                }
            )
            self._spill_to_tmp(blob_path, table)
            with self._retry_lock:
                self._retry_queue.append((blob_path, table))

    def _upload_table(self, blob_path: str, table: pa.Table) -> None:
        """
        Serialize *table* as Parquet and upload to *blob_path*.

        Args:
            blob_path: Destination blob path.
            table: PyArrow table to upload.

        Raises:
            AzureError: On any blob storage failure.
        """
        buf: io.BytesIO = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        blob_client = self._client.get_blob_client(
            container=self._container, blob=blob_path
        )
        blob_client.upload_blob(buf, overwrite=True)

    def _read_parquet(self, blob_path: str) -> pd.DataFrame | None:
        """
        Read a parquet file from blob (or fallback).

        If a fallback file exists for *blob_path* **and** the blob is
        unreachable, the fallback is returned.  If the blob is reachable
        the fallback is ignored in favour of the authoritative copy.

        Args:
            blob_path: Blob path to read.

        Returns:
            DataFrame, or ``None`` if neither source has data.
        """
        # Try blob first (authoritative source)
        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_path
            )
            data: bytes = blob_client.download_blob().readall()
            return pd.read_parquet(io.BytesIO(data))
        except ResourceNotFoundError:
            pass  # fall through to fallback
        except Exception as exc:
            logger.debug(
                {
                    "event": "blob_read_failed",
                    "blob_path": blob_path,
                    "error": str(exc),
                    "function": "_read_parquet",
                    "file": "storage.py",
                }
            )

        # Fallback to local tmp
        fallback: Path = FALLBACK_ROOT / blob_path
        if fallback.exists():
            try:
                return pd.read_parquet(fallback)
            except Exception as exc:
                logger.warning(
                    {
                        "event": "fallback_read_failed",
                        "path": str(fallback),
                        "error": str(exc),
                        "function": "_read_parquet",
                        "file": "storage.py",
                    }
                )
        return None

    def _spill_to_tmp(self, blob_path: str, table: pa.Table) -> None:
        """
        Write *table* to ``/tmp/fallback/{blob_path}``, merging with any
        existing local file.

        Args:
            blob_path: Relative path (used as subdirectory under fallback root).
            table: PyArrow table to spill.
        """
        dest: Path = FALLBACK_ROOT / blob_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            try:
                existing: pa.Table = pq.read_table(str(dest))
                table = pa.concat_tables([existing, table])
            except Exception:
                pass
        pq.write_table(table, str(dest), compression="snappy")
        logger.info({"event": "spilled_to_tmp", "path": str(dest)})

    def _clear_fallback(self, blob_path: str) -> None:
        """
        Remove a fallback file after a successful blob upload.

        Args:
            blob_path: Blob path whose fallback to remove.
        """
        fallback: Path = FALLBACK_ROOT / blob_path
        if fallback.exists():
            try:
                fallback.unlink()
                logger.debug({"event": "fallback_cleared", "path": str(fallback)})
            except OSError:
                pass

    def _ensure_container(self) -> None:
        """Create the blob container if it doesn't already exist."""
        try:
            container_client = self._client.get_container_client(self._container)
            container_client.get_container_properties()
        except ResourceNotFoundError:
            try:
                self._client.create_container(self._container)
                logger.info(
                    {
                        "event": "container_created",
                        "container": self._container,
                        "function": "_ensure_container",
                        "file": "storage.py",
                    }
                )
            except Exception as exc:
                logger.error(
                    {
                        "event": "container_create_failed",
                        "container": self._container,
                        "error": str(exc),
                        "function": "_ensure_container",
                        "file": "storage.py",
                    }
                )
                raise
        except Exception as exc:
            logger.warning({"event": "container_check_warning", "error": str(exc)})

    # ── Static path builders ──────────────────────────────────────────────────

    @staticmethod
    def _tick_blob_path(pair: str, year: int, month: int) -> str:
        """
        Return the blob path for a tick data file.

        Args:
            pair: Currency pair.
            year: Four-digit year.
            month: Month number (1–12).

        Returns:
            Blob path like ``data/EUR_USD/2025-01.parquet``.
        """
        return f"data/{pair}/{year}-{month:02d}.parquet"

    @staticmethod
    def _result_blob_path(pair: str, ts: datetime) -> str:
        """
        Return the blob path for a result data file.

        Args:
            pair: Currency pair.
            ts: Datetime used to determine the year/month partition.

        Returns:
            Blob path like ``data/results/EUR_USD/2025-01.parquet``.
        """
        return f"data/results/{pair}/{ts.year}-{ts.month:02d}.parquet"
