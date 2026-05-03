"""
storage.py — The Data Custodian.

Single point of contact between volatile RAM and the persistent data layer.
Enforces the Isolation Principle: no module outside Storage may know the
physical path, file format, or write strategy of the data store.

Operates under the 'Write-Once, Read-Many' vault philosophy. Every write
is atomic with respect to the threading lock, deduplicated by timestamp,
and committed in Snappy-compressed Parquet format for columnar ML loading.

Design Document: docs/Core/Storage/Storage.md
"""

from __future__ import annotations


import logging
import threading
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from typing import Optional, Literal
from azure.storage.blob import BlobServiceClient, ContainerClient
from core.config import get_settings
from core.exceptions import StorageError
from ml_engine.model import Bar, Tick

logger = logging.getLogger(__name__)

# ── Parquet Write Strategy ────────────────────────────────────────────────────
# Snappy is chosen over gzip/brotli for this workload:
#   - Fastest decompression (critical for ML training loops)
#   - Acceptable compression ratio for time-series float data
#   - Native support in PyArrow and Pandas without extra dependencies
_COMPRESSION: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy"

# Force pyarrow: industry standard for financial time-series efficiency
_ENGINE: Literal["pyarrow"] = "pyarrow"

# Timestamp column name used for deduplication across all Parquet files.
# Centralised here so a rename never causes a silent schema mismatch.
_TS_COLUMN: str = "timestamp"

# Schema version written into Parquet file metadata on every write.
# Increment when Bar or Tick fields are added, renamed, or removed.
# On read, a mismatch raises StorageError immediately rather than
# producing NaN columns or a silent pd.concat failure downstream.
_SCHEMA_VERSION: int = 1
_SCHEMA_VERSION_KEY: bytes = b"trader_ai_schema_version"


class Storage:
    """
    Thread-safe Parquet I/O manager for all market data persistence.

    Acts as the sole gateway between in-memory data structures (Tick, Bar)
    and the on-disk data vault. Enforces path isolation, atomic writes,
    timestamp deduplication, and auto-provisioning of the directory hierarchy.

    The Storage class follows the same Dictator pattern as config.py:
    it either initialises successfully with a fully verified directory
    structure, or it raises StorageError immediately at construction.
    There is no partially-initialised state.

    Attributes:
        root_dir:      Absolute path to project_root/data/
        raw_dir:       Absolute path to project_root/data/raw/
        processed_dir: Absolute path to project_root/data/processed/

    Thread Safety:
        A single threading.Lock serialises ALL write operations across all
        symbols. This is intentionally conservative: the lock cost is
        negligible compared to Parquet I/O, and it eliminates any risk of
        concurrent writes corrupting Parquet file headers.

    Example:
        >>> storage = Storage()
        >>> storage.save_tick_batch(ticks)
        True
        >>> storage.get_last_timestamp("EUR_USD")
        Timestamp('2024-01-15 10:30:00')
    """

    # Suppress repeated LOCAL-mode upload warnings — log once per process lifetime.
    _local_mode_warned: bool = False

    def __init__(self) -> None:
        """
        Initialise the Storage custodian.

        Resolves the data directory hierarchy relative to the project root,
        acquires validated configuration via get_settings(), and provisions
        all required directories. Fails immediately if any directory cannot
        be created or written to.

        Raises:
            StorageError: If directory provisioning fails due to permissions,
                disk space, or any other OS-level constraint.
        """
        self._settings = get_settings()
        # _file_locks maps resolved file path → Lock so that concurrent writes
        # to different symbols (different files) never block each other.
        # _locks_mutex guards the _file_locks dict itself against concurrent creation.
        self._locks_mutex: threading.Lock = threading.Lock()
        self._file_locks: dict[str, threading.Lock] = {}

        # ── Path Resolution ───────────────────────────────────────────────────
        # Resolve absolute project root from this file's location.
        # This file lives at src/core/storage.py, so parents[2] = project root.
        # Using resolve() ensures symlinks and relative paths are normalised,
        # making the system behave identically on Linux (VPS) and Windows (dev).
        self.root_dir: Path = Path(__file__).resolve().parents[2] / "data"
        self.raw_dir: Path = self.root_dir / "raw"
        self.processed_dir: Path = self.root_dir / "processed"

        # Provision directories immediately -- fail fast if not writable.
        self._provision_infrastructure()

        # ── Azure Blob Client (CLOUD mode only) ───────────────────────────────
        # In LOCAL mode this is None and all sync methods become no-ops.
        # In CLOUD mode this is a live ContainerClient; failure exits immediately.
        self._container_client: ContainerClient | None = self._init_azure_client()

    # ── Private: Infrastructure ───────────────────────────────────────────────

    def _provision_infrastructure(self) -> None:
        """
        Create the standard data directory hierarchy if it does not exist.

        Called once at construction time. Uses exist_ok=True so the call is
        idempotent -- running the system twice does not raise an error if
        directories already exist.

        Directory structure enforced:
            data/
            ├── raw/        <- Symbol_ticks.parquet (live + backfilled ticks)
            └── processed/  <- Symbol_M1.parquet (aggregated OHLCV bars)

        Raises:
            StorageError: If any directory cannot be created. Wraps the
                underlying OS exception to provide context in the log.
        """
        dirs_to_provision = [self.raw_dir, self.processed_dir]
        failed_path: Path | None = None

        try:
            for path in dirs_to_provision:
                failed_path = path
                path.mkdir(parents=True, exist_ok=True)
                # Proactive write-check: fail now rather than at first I/O.
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()

            logger.info({"event": "STORAGE_INIT_SUCCESS", "root": str(self.root_dir)})

        except OSError as e:
            raise StorageError(
                f"Cannot provision directory '{failed_path}': {e}. "
                f"Grant write permissions or check volume mounts.",
                path=str(failed_path),
            ) from e

    def _init_azure_client(self) -> ContainerClient | None:
        """
        Initialise the Azure Blob Storage container client.

        Called once from ``__init__``. Only active when ``DATA_MODE`` is
        ``'CLOUD'``; returns ``None`` immediately in ``'LOCAL'`` mode so
        the class can operate fully offline without touching Azure SDK code paths.

        Follows the Dictator Pattern: if CLOUD mode is configured but the
        connection cannot be established (bad connection string, missing container,
        network failure), the process exits immediately rather than continuing in a
        degraded state where sync calls would silently fail.

        Returns:
            ContainerClient if ``DATA_MODE`` is ``'CLOUD'``, authenticated and
            scoped to the configured container, ready for upload/download.
            None if ``DATA_MODE`` is ``'LOCAL'``.

        Raises:
            SystemExit: If ``DATA_MODE`` is ``'CLOUD'`` and the container cannot
                be reached — bad connection string, missing container, or network
                failure. The underlying Azure SDK exception is logged before exit.
        """
        if self._settings.data_mode != "CLOUD":
            logger.debug(
                {"event": "AZURE_BLOB_SKIPPED", "mode": self._settings.data_mode}
            )
            return None
        try:
            service: BlobServiceClient = BlobServiceClient.from_connection_string(
                self._settings.azure_storage_conn
            )
            client: ContainerClient = service.get_container_client(
                self._settings.container_name
            )
            # Dictator check: verify the container is reachable NOW, not at
            # first upload, so boot failures surface immediately.
            client.get_container_properties()

            logger.info(
                {
                    "event": "AZURE_BLOB_CONNECTED",
                    "container": self._settings.container_name,
                    "mode": self._settings.data_mode,
                }
            )
            return client

        except Exception as e:
            raise StorageError(
                f"Azure Blob unreachable (container='{self._settings.container_name}'): {e}. "
                f"Check AZURE_STORAGE_CONN or set DATA_MODE=LOCAL for offline use."
            ) from e

    def _get_file_lock(self, file_path: Path) -> threading.Lock:
        """
        Return the per-file threading.Lock for *file_path*, creating it on
        first access.

        Using one lock per file means concurrent writes to different symbols
        (EUR_USD_ticks.parquet vs GBP_USD_ticks.parquet) never block each
        other. The _locks_mutex guard ensures the dict itself is updated
        safely under concurrent access.

        Args:
            file_path: Resolved absolute path to the target Parquet file.

        Returns:
            threading.Lock: The lock that serialises I/O for this file.
        """
        key = str(file_path)
        # Fast-path: lock already exists (no mutex needed for read).
        if key in self._file_locks:
            return self._file_locks[key]
        with self._locks_mutex:
            # Re-check inside the mutex to avoid double-creation.
            if key not in self._file_locks:
                self._file_locks[key] = threading.Lock()
            return self._file_locks[key]

    def _raw_path(self, symbol: str) -> Path:
        """
        Resolve the canonical Parquet file path for a symbol's tick data.

        Centralises path construction so a naming convention change (e.g.,
        adding a date partition) only requires editing one place.

        Args:
            symbol: Pure currency pair name (e.g., "EUR_USD").

        Returns:
            Path: Absolute path to the symbol's tick Parquet file.
                    e.g., /project/data/raw/EUR_USD_ticks.parquet
        """
        return self.raw_dir / f"{symbol}_ticks.parquet"

    def _processed_path(self, symbol: str, timeframe: str) -> Path:
        """
        Resolve the canonical Parquet file path for a symbol's bar data.

        Args:
            symbol:    Pure currency pair name (e.g., "EUR_USD").
            timeframe: Timeframe string from the Timeframe enum (e.g., "M1").

        Returns:
            Path: Absolute path to the symbol's bar Parquet file.
                    e.g., /project/data/processed/EUR_USD_M1.parquet
        """
        return self.processed_dir / f"{symbol}_{timeframe}.parquet"

    def _check_schema_version(self, file_path: Path) -> None:
        """
        Read Parquet footer metadata and raise StorageError on version mismatch.

        Called before loading an existing file in _atomic_upsert so that a
        stale file (written by an older code version with different columns)
        raises a clear, actionable error instead of producing a silent NaN
        column or a confusing pd.concat failure.

        Args:
            file_path: Path to an existing Parquet file.

        Raises:
            StorageError: If the file has no schema version metadata, or if
                its version does not match _SCHEMA_VERSION.
        """
        meta = pq.read_metadata(file_path)
        stored: bytes | None = (meta.metadata or {}).get(_SCHEMA_VERSION_KEY)
        if stored is None or int(stored) != _SCHEMA_VERSION:
            got: str = stored.decode() if stored else "none"
            raise StorageError(
                f"Schema version mismatch in {file_path.name}: "
                f"file=v{got}, code=v{_SCHEMA_VERSION}. "
                f"Re-run backfill to regenerate this file.",
                path=str(file_path),
            )

    def _write_versioned_parquet(self, file_path: Path, df: pd.DataFrame) -> None:
        """
        Write a DataFrame to Parquet with _SCHEMA_VERSION embedded in metadata.

        Uses PyArrow directly (not pandas.to_parquet) so that custom key-value
        metadata survives the round-trip. The version key is read back by
        _check_schema_version on the next upsert cycle.

        Args:
            file_path: Destination path for the Parquet file.
            df:        DataFrame to write. Index is not preserved.
        """
        table = pa.Table.from_pandas(df, preserve_index=False)
        existing_meta: dict = table.schema.metadata or {}
        new_meta = {**existing_meta, _SCHEMA_VERSION_KEY: str(_SCHEMA_VERSION).encode()}
        table = table.replace_schema_metadata(new_meta)
        pq.write_table(table, file_path, compression=_COMPRESSION)

    def _atomic_upsert(self, file_path: Path, new_data: pd.DataFrame) -> None:
        """
        Perform an atomic read-deduplicate-write cycle on a Parquet file.

        This is the core write primitive used by all public save methods.
        It is called INSIDE the threading lock and must never be called
        outside of it.

        Strategy:
            1. If the file exists: verify schema version, read existing data,
                concatenate with new, deduplicate on timestamp (last-in-wins),
                write back with updated version metadata.
            2. If the file does not exist: write new_data directly with version
                metadata embedded.

        The 'last-in-wins' deduplication policy ensures that backfilled data
        which overlaps with live data is always superseded by the most recent
        write. This prevents stale backfill data from corrupting a live file.

        Args:
            file_path: Absolute path to the target Parquet file.
            new_data:  DataFrame to upsert. Must contain a 'timestamp' column.

        Raises:
            StorageError: If the file's schema version does not match the
                current code version, or if the Parquet read or write fails
                for any reason (disk full, corrupted header, etc.).
        """
        # Validate that the new data contains the required timestamp column before any I/O.
        if _TS_COLUMN not in new_data.columns:
            raise StorageError(
                f"DataFrame is missing required column '{_TS_COLUMN}'. "
                f"Cannot perform upsert.",
                path=str(file_path),
            )
        try:
            if file_path.exists():
                # Version gate: raises StorageError immediately on mismatch so
                # the caller sees a clear message rather than a pd.concat NaN.
                self._check_schema_version(file_path)
                existing_df = pd.read_parquet(file_path, engine=_ENGINE)  # type: ignore[arg-type]
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                # Free existing_df immediately — combined_df is the only copy needed.
                del existing_df
                # The Timeline Healer: Keep the latest data for any given timestamp
                combined_df.drop_duplicates(
                    subset=[_TS_COLUMN], keep="last", inplace=True
                )
                combined_df.sort_values(_TS_COLUMN, inplace=True)
                combined_df.reset_index(drop=True, inplace=True)
                self._write_versioned_parquet(file_path, combined_df)
                del combined_df
            else:
                new_data.sort_values(_TS_COLUMN, inplace=True)
                self._write_versioned_parquet(file_path, new_data)
        except StorageError:
            raise  # schema version errors propagate as-is without re-wrapping
        except Exception as e:
            logger.critical(
                {
                    "event": "STORAGE_WRITE_FAILURE",
                    "file": str(file_path),
                    "error": str(e),
                }
            )
            raise StorageError(
                f"Parquet upsert failed for {file_path}: {e}",
                path=str(file_path),
            ) from e

    # ── Public: Write Operations ──────────────────────────────────────────────

    def save_tick_batch(self, ticks: list[Tick]) -> bool:
        """
        Persist a batch of validated Tick objects to raw Parquet storage.

        Converts the batch to a DataFrame, then performs an atomic upsert
        into the symbol's tick file. All ticks in a batch must belong to the
        same symbol -- the symbol is read from ticks[0].

        This method is the primary write target for the DataBuffer. It is
        designed to be called with the output of DataBuffer.add() or
        DataBuffer.flush(), never with individual ticks.

        Args:
            ticks: Non-empty list of validated Tick instances. All ticks
                must share the same symbol. An empty list is a no-op.

        Returns:
            bool: True if the batch was committed successfully.
                    False if the batch was empty (no-op, not an error).

        Raises:
            StorageError: If the Parquet write fails. The caller (pipeline
                or engine) is responsible for deciding whether to retry,
                log, or halt on StorageError.

        Example:
            >>> batch = buffer.add(tick)  # returns batch at flush_size
            >>> if batch:
            ...     storage.save_tick_batch(batch)
        """
        if not ticks:
            logger.debug(
                "save_tick_batch_empty",
                extra={
                    "event": "STORAGE_BATCH_EMPTY",
                    "method": "save_tick_batch",
                },
            )
            return False

        symbol: str = ticks[0].symbol
        file_path: Path = self._raw_path(symbol)
        new_data: pd.DataFrame = pd.DataFrame([t.to_dict() for t in ticks])

        symbols_in_batch = {t.symbol for t in ticks}
        if len(symbols_in_batch) > 1:
            raise ValueError(
                f"save_tick_batch received mixed symbols: {symbols_in_batch}. "
                f"Each batch must contain ticks for a single symbol only."
            )

        logger.debug(
            {
                "event": "STORAGE_TICK_BATCH_COMMIT",
                "symbol": symbol,
                "count": len(ticks),
            }
        )

        with self._get_file_lock(file_path):
            self._atomic_upsert(file_path, new_data)

        logger.info(
            {
                "event": "STORAGE_TICK_BATCH_COMMITTED",
                "symbol": symbol,
                "count": len(ticks),
            }
        )
        # Sync to Azure if in CLOUD mode, but do not fail the write if the upload fails — the local file is the source of truth and can be re-synced later if needed.
        if self._container_client is not None:
            blob_name = f"raw/{symbol}_ticks.parquet"
            try:
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.debug(
                    {"event": "STORAGE_AZURE_SYNC", "blob": blob_name, "symbol": symbol}
                )
            except Exception as e:
                logger.warning(
                    {
                        "event": "STORAGE_AZURE_SYNC_FAILED",
                        "blob": blob_name,
                        "symbol": symbol,
                        "error": str(e),
                    }
                )
        return True

    def save_bar(self, bar: Bar) -> bool:
        """
        Persist a single validated Bar object to processed Parquet storage.

        Bars are stored in the processed directory, partitioned by symbol
        and timeframe. The file naming convention is:
            {symbol}_{timeframe}.parquet  (e.g., EUR_USD_M1.parquet)

        This method is called by the aggregation layer after a complete
        candle window has elapsed (bar.is_complete == True). Incomplete
        bars should be filtered by the caller before reaching Storage.

        Args:
            bar: A fully validated, complete Bar instance. The bar's
                timeframe is used to determine the target file path.

        Returns:
            bool: True if the bar was committed successfully.

        Raises:
            StorageError: If the Parquet write fails.

        Example:
            >>> if bar.is_complete:
            ...     storage.save_bar(bar)
        """
        file_path: Path = self._processed_path(bar.symbol, bar.timeframe.value)
        new_data: pd.DataFrame = pd.DataFrame([bar.to_dict()])
        if not bar.is_complete:
            logger.warning(
                {
                    "event": "STORAGE_INCOMPLETE_BAR",
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe.value,
                }
            )
            return False
        logger.debug(
            {
                "event": "STORAGE_BAR_COMMIT",
                "symbol": bar.symbol,
                "timeframe": bar.timeframe.value,
            }
        )

        with self._get_file_lock(file_path):
            self._atomic_upsert(file_path, new_data)

        logger.info(
            {
                "event": "STORAGE_BAR_COMMITTED",
                "symbol": bar.symbol,
                "timeframe": bar.timeframe.value,
            }
        )
        return True

    # ── Public: Azure Sync ────────────────────────────────────────────────────

    def sync_to_azure(self, local_path: Path, blob_name: str | None = None) -> bool:
        """
        Upload a local file to Azure Blob Storage.

        Used by the training workflow to push completed Parquet datasets and
        serialised model artefacts to the shared container so they can be
        pulled by the lean inference container on next deploy.

        No-op in ``LOCAL`` mode (returns ``False`` with a warning). In
        ``CLOUD`` mode, any upload failure is logged and returns ``False``
        rather than raising — the caller decides whether to retry or abort.

        Args:
            local_path: Absolute or relative path to the file to upload.
                        The file must exist before calling this method.
            blob_name:  Path within the container (e.g., ``"data/raw/EUR_USD_ticks.parquet"``
                        or ``"models/production_v1.pkl"``).
                        Defaults to the file's name if omitted.

        Returns:
            bool: ``True`` if the upload completed successfully, ``False``
                    otherwise (LOCAL mode, missing file, or Azure error).

        Example:
            >>> # Training workflow — push data then model
            >>> storage.sync_to_azure(Path("data/raw/EUR_USD_ticks.parquet"))
            >>> storage.sync_to_azure(Path("models/production_v1.pkl"), "models/production_v1.pkl")
        """
        if self._container_client is None:
            logger.warning({"event": "STORAGE_AZURE_SYNC_DISABLED", "mode": "LOCAL"})
            return False

        if not local_path.exists():
            logger.error(
                {"event": "STORAGE_AZURE_SYNC_FILE_MISSING", "path": str(local_path)}
            )

        target_blob: str = blob_name or local_path.name
        try:
            logger.info(
                {
                    "event": "STORAGE_AZURE_SYNC_START",
                    "local": local_path.name,
                    "blob": target_blob,
                }
            )
            blob_client = self._container_client.get_blob_client(target_blob)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            logger.info({"event": "STORAGE_AZURE_SYNC_COMPLETE", "blob": target_blob})
            return True

        except Exception as e:
            logger.error(
                {
                    "event": "STORAGE_AZURE_SYNC_FAILED",
                    "blob": target_blob,
                    "error": str(e),
                }
            )
            return False

    def pull_from_azure(self, blob_name: str, local_path: Path) -> bool:
        """
        Download a file from Azure Blob Storage to a local path.

        Used by the inference workflow to pull trained model artefacts and
        scalers from the shared container before the trading loop starts.
        Parent directories are created automatically if they do not exist.

        No-op in ``LOCAL`` mode (returns ``False`` with a warning). In
        ``CLOUD`` mode, any download failure is logged and returns ``False``
        rather than raising — the caller (e.g., pipeline.py) decides whether
        to abort or fall back to local files.

        Args:
            blob_name:  Path within the container (e.g., ``"models/production_v1.pkl"``).
                        Must match the name used during ``sync_to_azure``.
            local_path: Destination path on disk. Parent directories are
                        created automatically.

        Returns:
            bool: ``True`` if the download completed successfully, ``False``
                    otherwise (LOCAL mode, blob not found, or Azure error).

        Example:
            >>> # Inference workflow — pull model before trading
            >>> storage.pull_from_azure("models/production_v1.pkl", Path("models/production_v1.pkl"))
            >>> storage.pull_from_azure("models/scaler.pkl", Path("models/scaler.pkl"))
        """
        if self._container_client is None:
            logger.warning({"event": "STORAGE_AZURE_PULL_DISABLED", "mode": "LOCAL"})
            return False

        try:
            logger.info({"event": "STORAGE_AZURE_PULL_START", "blob": blob_name})
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            logger.info({"event": "STORAGE_AZURE_PULL_COMPLETE", "blob": blob_name})
            return True

        except Exception as e:
            logger.error(
                {
                    "event": "STORAGE_AZURE_PULL_FAILED",
                    "blob": blob_name,
                    "error": str(e),
                }
            )
            return False

    # ── Public: Model Persistence ─────────────────────────────────────────────

    def save_model(self, artifact_path: Path, metadata_path: Path) -> bool:
        """
        Upload a trained model artifact and its JSON metadata sidecar to Azure.

        Both files are written to the ``models/`` prefix in the shared
        container so that the inference container can pull them as a matched
        pair on next deploy. Uploading the sidecar alongside the artifact
        ensures the container never loads a model whose metadata is missing
        or belongs to a different training run.

        No-op in ``LOCAL`` mode (returns ``False`` with a warning). In
        ``CLOUD`` mode, any upload failure is logged and returns ``False``
        rather than raising — the caller decides whether to retry or abort.

        Args:
            artifact_path:  Local path to the serialised model artifact
                            (e.g., ``models/lstm_v3.pt`` or ``models/rf_v2.pkl``).
                            The file must exist before calling this method.
            metadata_path:  Local path to the JSON metadata sidecar
                            (e.g., ``models/lstm_v3_metadata.json``).
                            Must exist alongside the artifact.

        Returns:
            bool: ``True`` if both files uploaded successfully, ``False``
                    otherwise (LOCAL mode, missing file, or Azure error).

        Example:
            >>> storage.save_model(
            ...     Path("models/lstm_v3.pt"),
            ...     Path("models/lstm_v3_metadata.json"),
            ... )
            True
        """
        if self._container_client is None:
            if not Storage._local_mode_warned:
                logger.warning(
                    {
                        "event": "STORAGE_SAVE_MODEL_DISABLED",
                        "mode": "LOCAL",
                        "message": "Running in LOCAL mode — model uploads disabled for this session",
                    }
                )
                Storage._local_mode_warned = True
            return False

        for local_path in (artifact_path, metadata_path):
            if not local_path.exists():
                logger.error(
                    {
                        "event": "STORAGE_SAVE_MODEL_FILE_MISSING",
                        "path": str(local_path),
                    }
                )
                return False

        try:
            for local_path in (artifact_path, metadata_path):
                blob_name = f"models/{local_path.name}"
                logger.debug({"event": "STORAGE_MODEL_UPLOAD", "blob": blob_name})
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

            logger.info(
                {"event": "STORAGE_SAVE_MODEL_COMPLETE", "artifact": artifact_path.name}
            )
            return True

        except Exception as e:
            logger.error(
                {
                    "event": "STORAGE_SAVE_MODEL_FAILED",
                    "artifact": artifact_path.name,
                    "error": str(e),
                }
            )
            return False

    def load_model(
        self,
        artifact_filename: str,
        local_dir: Path | None = None,
    ) -> Path | None:
        """
        Download a model artifact and its JSON metadata sidecar from Azure.

        Pulls both the artifact and its ``<stem>.json`` sidecar
        from the ``models/`` prefix in the shared container. Both files are
        written into ``local_dir`` (or the project's ``models/`` directory
        by default). Returns the local artifact path on success so the
        caller can deserialise it immediately.

        No-op in ``LOCAL`` mode (returns ``None`` with a warning). In
        ``CLOUD`` mode, any download failure is logged and returns ``None``
        rather than raising — the caller (e.g., inference pipeline) decides
        whether to fall back to a cached model or abort.

        Args:
            artifact_filename: Filename of the artifact blob within the
                                ``models/`` prefix (e.g., ``"lstm_v3.pt"``).
                                The metadata sidecar is derived automatically
                                as ``<stem>.json``.
            local_dir:         Directory on disk where both files will be
                                written. Created automatically if absent.
                                Defaults to ``<project_root>/models/``.

        Returns:
            Path: Local path to the downloaded artifact on success.
            None: On LOCAL mode, blob not found, or Azure error.

        Example:
            >>> artifact = storage.load_model("lstm_v3.pt")
            >>> if artifact:
            ...     model = torch.load(artifact)
        """
        if self._container_client is None:
            logger.warning({"event": "STORAGE_LOAD_MODEL_DISABLED", "mode": "LOCAL"})
            return None

        dest_dir: Path = (
            local_dir
            if local_dir is not None
            else (Path(__file__).resolve().parents[2] / "models")
        )
        dest_dir.mkdir(parents=True, exist_ok=True)

        artifact_stem = Path(artifact_filename).stem
        metadata_filename = f"{artifact_stem}.json"

        try:
            for filename in (artifact_filename, metadata_filename):
                blob_name = f"models/{filename}"
                local_path = dest_dir / filename
                logger.debug({"event": "STORAGE_MODEL_PULL", "blob": blob_name})
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(local_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())

            local_artifact: Path = dest_dir / artifact_filename
            logger.info(
                {"event": "STORAGE_LOAD_MODEL_COMPLETE", "artifact": artifact_filename}
            )
            return local_artifact

        except Exception as e:
            logger.error(
                {
                    "event": "STORAGE_LOAD_MODEL_FAILED",
                    "artifact": artifact_filename,
                    "error": str(e),
                }
            )
            return None

    def save_processed_features(
        self,
        symbol: str,
        expiry_key: str,
        local_path: Path,
    ) -> bool:
        """
        Upload a processed feature matrix Parquet file to Azure Blob Storage.

        Writes to the ``processed/`` prefix using the canonical naming
        convention ``{symbol}_{expiry_key}_features.parquet``, matching the
        path pattern expected by the training pipeline when pulling feature
        matrices from the shared container.

        No-op in ``LOCAL`` mode (returns ``False`` with a warning). In
        ``CLOUD`` mode, any upload failure is logged and returns ``False``
        rather than raising.

        Args:
            symbol:      Currency pair name (e.g., ``"EUR_USD"``). Used to
                            construct the blob name.
            expiry_key:  Binary expiry rule key (e.g., ``"M1_1"``). Combined
                            with ``symbol`` to uniquely identify the feature set.
            local_path:  Local path to the Parquet file produced by
                            ``FeatureEngineer.build_matrix()``. Must exist.

        Returns:
            bool: ``True`` if the upload completed successfully, ``False``
                    otherwise (LOCAL mode, missing file, or Azure error).

        Example:
            >>> storage.save_processed_features(
            ...     symbol="EUR_USD",
            ...     expiry_key="M1_1",
            ...     local_path=Path("data/processed/EUR_USD_M1_1_features.parquet"),
            ... )
            True
        """
        if self._container_client is None:
            logger.warning(
                {"event": "STORAGE_FEATURES_UPLOAD_DISABLED", "mode": "LOCAL"}
            )
            return False

        if not local_path.exists():
            logger.error(
                {
                    "event": "STORAGE_FEATURES_UPLOAD_FILE_MISSING",
                    "symbol": symbol,
                    "expiry_key": expiry_key,
                }
            )
            return False

        blob_name = f"processed/{symbol}_{expiry_key}_features.parquet"
        try:
            logger.info(
                {
                    "event": "STORAGE_FEATURES_UPLOAD_START",
                    "symbol": symbol,
                    "expiry_key": expiry_key,
                    "blob": blob_name,
                }
            )
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            logger.info(
                {
                    "event": "STORAGE_FEATURES_UPLOAD_COMPLETE",
                    "symbol": symbol,
                    "expiry_key": expiry_key,
                }
            )
            return True

        except Exception as e:
            logger.error(
                {
                    "event": "STORAGE_FEATURES_UPLOAD_FAILED",
                    "symbol": symbol,
                    "expiry_key": expiry_key,
                    "error": str(e),
                }
            )
            return False

    # ── Public: Read / Query Operations ──────────────────────────────────────

    def get_last_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Return the most recent tick timestamp for a given symbol.

        Performs a columnar read of the timestamp column only, making this
        operation extremely fast even on files with millions of rows. This
        is the primary entry point for the Historian (backfill engine) to
        determine the data gap since the last session.

        Args:
            symbol: Pure currency pair name (e.g., "EUR_USD").

        Returns:
            pd.Timestamp: The maximum timestamp found in the tick file.
            None: If no tick file exists for the symbol (first run).

        Example:
            >>> last_ts = storage.get_last_timestamp("EUR_USD")
            >>> if last_ts is None:
            ...     # First run -- backfill from scratch
            ...     historian.backfill_full(symbol)
            >>> else:
            ...     # Resume -- backfill the gap only
            ...     historian.backfill_gap(symbol, since=last_ts)
        """
        file_path: Path = self._raw_path(symbol)

        with self._get_file_lock(file_path):
            # If local doesn't exist, try to pull from Azure FIRST
            if not file_path.exists() and self._container_client is not None:
                blob_name = f"raw/{symbol}_ticks.parquet"
                self.pull_from_azure(blob_name, file_path)

            if not file_path.exists():
                logger.info(
                    f"No tick file found for {symbol}. First run -- full backfill required.",
                    extra={
                        "event": "STORAGE_NO_TICK_FILE",
                        "symbol": symbol,
                        "first_run": True,
                    },
                )
                return None

            try:
                # Columnar read: load only the timestamp column.
                # PyArrow pushes this predicate to the Parquet reader,
                # avoiding full file deserialisation. Fast on any file size.
                df: pd.DataFrame = pd.read_parquet(file_path, engine=_ENGINE, columns=[_TS_COLUMN])  # type: ignore[arg-type]

                if df.empty:
                    logger.warning(
                        "Tick file for {symbol} exists but is empty.",
                        extra={
                            "event": "STORAGE_EMPTY_TICK_FILE",
                            "symbol": symbol,
                            "file": str(file_path),
                        },
                    )
                    return None

                last_ts: pd.Timestamp = df[_TS_COLUMN].max()
                logger.info(
                    f"Last known timestamp for {symbol}: {last_ts}",
                    extra={
                        "event": "STORAGE_LAST_TIMESTAMP",
                        "symbol": symbol,
                        "timestamp": str(last_ts),
                    },
                )
                return last_ts

            except Exception as e:
                logger.error(
                    "Cannot read timestamp index from tick file.",
                    extra={
                        "event": "STORAGE_TICK_FILE_CORRUPTED",
                        "symbol": symbol,
                        "file": str(file_path),
                        "error": str(e),
                    },
                )
                raise StorageError(
                    f"Tick file for {symbol} is corrupted and cannot be read.",
                    symbol=symbol,
                    path=str(file_path),
                ) from e

    def _pull_bar_file_from_azure(
        self, symbol: str, timeframe: str, file_path: Path
    ) -> None:
        """Pull a processed bar Parquet from Azure if it is absent locally."""
        if self._container_client is None:
            return
        blob_name = f"processed/{symbol}_{timeframe}.parquet"
        success = self.pull_from_azure(blob_name, file_path)
        if success:
            logger.debug(
                f"Pulled {blob_name} from Azure to {file_path}",
                extra={
                    "event": "STORAGE_BAR_FILE_PULLED",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "blob": blob_name,
                },
            )
        else:
            logger.debug(
                f"No blob found for {blob_name}, will need backfill",
                extra={
                    "event": "STORAGE_NO_BAR_BLOB",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "blob": blob_name,
                },
            )

    def _read_bar_df(
        self,
        pf: pq.ParquetFile,
        symbol: str,
        timeframe: str,
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        """Read a DataFrame from *pf*, honouring the *max_rows* row cap."""
        n_total: int = pf.metadata.num_rows
        if max_rows is not None and n_total > max_rows:
            row_groups: list[int] = []
            rows_accumulated: int = 0
            for i in range(pf.metadata.num_row_groups - 1, -1, -1):
                row_groups.insert(0, i)
                rows_accumulated += pf.metadata.row_group(i).num_rows
                if rows_accumulated >= max_rows:
                    break
            df = pf.read_row_groups(row_groups).to_pandas()
            logger.debug(
                f"Row cap applied: read {rows_accumulated} rows from "
                f"{len(row_groups)} row group(s), returning tail "
                f"{max_rows} for {symbol} [{timeframe}].",
                extra={
                    "event": "STORAGE_ROW_CAP_APPLIED",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "total_rows": n_total,
                    "returned_rows": rows_accumulated,
                },
            )
        else:
            df = pf.read().to_pandas()
        return df

    def _validate_and_index_bar_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        file_path: Path,
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        """Validate, sort, and optionally trim a bar DataFrame."""
        if _TS_COLUMN not in df.columns:
            logger.error(
                "missing_timestamp_column",
                extra={
                    "event": "STORAGE_MISSING_TIMESTAMP",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "file": str(file_path.name),
                    "columns": df.columns.tolist(),
                },
            )
            raise StorageError(
                f"Missing timestamp column in {file_path.name}",
                symbol=symbol,
                path=str(file_path),
            )
        df[_TS_COLUMN] = pd.to_datetime(df[_TS_COLUMN])
        df.set_index(_TS_COLUMN, inplace=True)
        df.sort_index(inplace=True)
        if max_rows is not None and len(df) > max_rows:
            df = df.tail(max_rows)
        return df

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "M1",
        max_rows: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load processed Bar data for a symbol and timeframe.

        Reads the full processed Parquet file for the given symbol and
        timeframe. If max_rows is specified, returns only the most recent
        N rows, sorted by timestamp. This is the primary data feed for
        the ML training loop and feature engineering pipeline.

        Args:
            symbol:    Pure currency pair name (e.g., "EUR_USD").
            timeframe: Timeframe string (e.g., "M1", "M5", "M15").
                        Defaults to "M1" as the primary training resolution.
            max_rows:  If provided, return only the most recent N rows.
                        Uses MAX_RF_ROWS from config if not specified by caller.
                        None returns all available rows.

        Returns:
            pd.DataFrame: Sorted DataFrame of Bar records, most recent last.
            None: If no processed file exists for the symbol/timeframe pair.

        Example:
            >>> df = storage.get_bars("EUR_USD", timeframe="M1", max_rows=50000)
            >>> model.train(df)
        """
        file_path: Path = self._processed_path(symbol, timeframe)

        if not file_path.exists():
            self._pull_bar_file_from_azure(symbol, timeframe, file_path)

        if not file_path.exists():
            logger.info(
                f"No processed bar file found for {symbol} [{timeframe}].",
                extra={
                    "event": "STORAGE_NO_BAR_FILE",
                    "symbol": symbol,
                    "timeframe": timeframe,
                },
            )
            return None

        with self._get_file_lock(file_path):
            try:
                pf = pq.ParquetFile(file_path)

                if pf.metadata.num_rows == 0:
                    logger.warning(
                        f"Bar file for {symbol} [{timeframe}] exists but is empty.",
                        extra={
                            "event": "STORAGE_EMPTY_BAR_FILE",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "file": str(file_path),
                        },
                    )
                    return None

                df = self._read_bar_df(pf, symbol, timeframe, max_rows)

                if df.empty:
                    logger.warning(
                        {
                            "event": "STORAGE_EMPTY_BAR_DATAFRAME",
                            "symbol": symbol,
                            "timeframe": timeframe,
                        }
                    )
                    return None

                df = self._validate_and_index_bar_df(
                    df, symbol, timeframe, file_path, max_rows
                )
                logger.info(
                    {
                        "event": "STORAGE_BARS_LOADED",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "count": len(df),
                    }
                )
                return df

            except StorageError:
                raise
            except Exception as e:
                logger.error(
                    {
                        "event": "STORAGE_BAR_FILE_CORRUPTED",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error": str(e),
                    }
                )
                raise StorageError(
                    f"Bar file for {symbol} [{timeframe}] is corrupted and cannot be read.",
                    symbol=symbol,
                    path=str(file_path),
                ) from e

    def save_bar_batch(self, bars: list[Bar]) -> bool:
        """
        Persist a batch of validated Bar objects to processed Parquet storage.

        Functionally mirrors :meth:`save_tick_batch` but targets the processed
        directory. Converts the entire batch to a single DataFrame and performs
        one atomic upsert, making it dramatically more efficient than calling
        :meth:`save_bar` in a loop for large historical datasets.

        All bars in a batch must share the same symbol **and** the same
        timeframe — one Parquet file is written per (symbol, timeframe) pair.
        Mixed-symbol or mixed-timeframe batches are rejected before any I/O.

        Args:
            bars: Non-empty list of validated Bar instances. All bars must
                share the same symbol and timeframe. An empty list is a no-op.

        Returns:
            bool: True if the batch was committed successfully.
                    False if the batch was empty (no-op, not an error).

        Raises:
            ValueError:    If the batch contains mixed symbols or timeframes.
            StorageError:  If the Parquet write fails.

        Example:
            >>> bars = historian.parse_chunk(api_response)
            >>> storage.save_bar_batch(bars)
            True
        """
        if not bars:
            logger.debug({"event": "STORAGE_BAR_BATCH_EMPTY"})
            return False

        # Guard: single symbol per batch (mirrors save_tick_batch contract)
        symbols_in_batch: set[str] = {b.symbol for b in bars}
        if len(symbols_in_batch) > 1:
            logger.error(
                {
                    "event": "STORAGE_MIXED_SYMBOLS_BAR_BATCH",
                    "symbols": list(symbols_in_batch),
                }
            )
            raise ValueError(
                f"save_bar_batch received mixed symbols: {symbols_in_batch}. "
                f"Each batch must contain bars for a single symbol only."
            )

        # Guard: single timeframe per batch (one file per symbol+timeframe)
        timeframes_in_batch: set[str] = {b.timeframe.value for b in bars}
        if len(timeframes_in_batch) > 1:
            logger.error(
                {
                    "event": "STORAGE_MIXED_TIMEFRAMES_BAR_BATCH",
                    "timeframes": list(timeframes_in_batch),
                }
            )
            raise ValueError(
                f"save_bar_batch received mixed timeframes: {timeframes_in_batch}. "
                f"Each batch must contain bars for a single timeframe only."
            )

        symbol: str = bars[0].symbol
        timeframe: str = bars[0].timeframe.value
        file_path: Path = self._processed_path(symbol, timeframe)
        new_data: pd.DataFrame = pd.DataFrame([b.to_dict() for b in bars])
        logger.debug(
            {
                "event": "STORAGE_BAR_BATCH_COMMIT",
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(bars),
            }
        )

        with self._get_file_lock(file_path):
            self._atomic_upsert(file_path, new_data)

        logger.info(
            {
                "event": "STORAGE_BAR_BATCH_COMMITTED",
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(bars),
            }
        )

        if self._container_client is not None:
            blob_name = f"processed/{symbol}_{timeframe}.parquet"
            try:
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.info(
                    {"event": "STORAGE_BAR_BATCH_AZURE_SYNC", "blob": blob_name}
                )
            except Exception as e:
                logger.warning(
                    {
                        "event": "STORAGE_BAR_BATCH_AZURE_SYNC_FAILED",
                        "blob": blob_name,
                        "error": str(e),
                    }
                )

        return True

    # ── Public: Diagnostics ───────────────────────────────────────────────────

    def get_tick_count(self, symbol: str) -> int:
        """
        Return the total number of ticks stored for a given symbol.

        Performs a columnar read of the timestamp column only to count rows
        without loading price data. Used for health checks, dashboard metrics,
        and pre-training data sufficiency validation.

        Args:
            symbol: Pure currency pair name (e.g., "EUR_USD").

        Returns:
            int: Total tick count. Returns 0 if no file exists or on error.

        Example:
            >>> count = storage.get_tick_count("EUR_USD")
            >>> if count < MIN_TRAINING_ROWS:
            ...     logger.warning("Insufficient data for training.")
        """
        file_path: Path = self._raw_path(symbol)

        if not file_path.exists():
            return 0

        try:
            df = pd.read_parquet(file_path, engine=_ENGINE, columns=[_TS_COLUMN])  # type: ignore[arg-type]
            count = len(df)
            logger.debug(
                "tick_count",
                extra={
                    "event": "STORAGE_TICK_COUNT",
                    "symbol": symbol,
                    "count": count,
                },
            )
            return count
        except Exception as e:
            logger.warning(
                f"[%] Could not count ticks for {symbol}: {e}",
                extra={
                    "event": "STORAGE_TICK_COUNT_FAILED",
                    "symbol": symbol,
                    "error": str(e),
                },
            )
            return 0

    def list_symbols(self) -> list[str]:
        """
        Return a list of all symbols that have tick data on disk.

        Scans the raw directory for Parquet files matching the naming
        convention and extracts the symbol names. Used by the dashboard
        and pipeline orchestrator to discover available data.

        Returns:
            list[str]: Sorted list of symbol names (e.g., ["EUR_USD", "GBP_USD"]).
                        Empty list if no tick files exist.

        Example:
            >>> symbols = storage.list_symbols()
            >>> for symbol in symbols:
            ...     pipeline.run(symbol)
        """
        try:
            symbols = sorted(
                [
                    f.stem.replace("_ticks", "")
                    for f in self.raw_dir.glob("*_ticks.parquet")
                ]
            )
            logger.debug(
                "list_symbols",
                extra={
                    "event": "STORAGE_LIST_SYMBOLS",
                    "count": len(symbols),
                    "symbols": symbols,
                },
            )
            return symbols
        except Exception as e:
            logger.warning(
                f"Could not list symbols from raw directory: {e}",
                extra={
                    "event": "STORAGE_LIST_SYMBOLS_FAILED",
                    "error": str(e),
                },
            )
            return []


# ── Singleton ────────────────────────────────────────────────────────────────

_storage_singleton: Storage | None = None


def get_storage() -> Storage:
    """
    Return the global Storage singleton instance.

    Ensures only one Storage instance exists across the entire application,
    so the Azure Blob client is initialized once and reused everywhere.
    """
    global _storage_singleton
    if _storage_singleton is None:
        _storage_singleton = Storage()
    return _storage_singleton
