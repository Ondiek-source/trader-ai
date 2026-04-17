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


import os
import sys
import logging
import threading
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from typing import Optional, Literal
from azure.storage.blob import BlobServiceClient, ContainerClient
from core.config import get_settings
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


class StorageError(Exception):
    """
    Raised when Storage cannot fulfil a critical I/O commitment.

    Distinct from ValueError (data shape) and RuntimeError (logic).
    Allows the engine to catch storage failures specifically without
    masking unrelated exceptions.

    Attributes:
        message: Human-readable description of the failure.
        symbol: The currency pair involved, if applicable.
        path: The file path involved, if applicable.
    """

    def __init__(
        self,
        message: str,
        symbol: str = "",
        path: str = "",
    ) -> None:
        self.symbol = symbol
        self.path = path
        super().__init__(message)


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

        try:
            for path in dirs_to_provision:
                path.mkdir(parents=True, exist_ok=True)
                # Proactive Dictator Check: Can we actually write here?
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
                debug_block = (
                    f"\n{'=' * 60}\n"
                    f"STORAGE INITIALIZATION SUCCESS\n"
                    f"Data directory structure is in place and writable.\n"
                    f"Root       : {self.root_dir}\n"
                    f"Raw        : {self.raw_dir}\n"
                    f"Processed  : {self.processed_dir}\n"
                    f"Path       : {'Writable' if all(p.is_dir() and os.access(p, os.W_OK) for p in dirs_to_provision) else 'Not Writable'}\n"
                    f"{'=' * 60}"
                )
                logger.info(debug_block)

        except OSError as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"STORAGE INITIALIZATION FAILURE\n"
                f"Cannot provision directory: {path}\n"
                f"OS Error: {e}\n\n"
                f"CONTEXT: Storage requires write access to the data directory.\n"
                f"  - On Linux VPS: Check that the process user owns /data/.\n"
                f"  - On Windows: Ensure the path is not read-only or OneDrive-synced.\n"
                f"  - In Docker: Confirm the volume mount is writable.\n"
                f"\nFIX: Grant write permissions to: {path}\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            # Storage is non-functional without a writable data directory, so we raise an error and exit immediately.
            sys.exit(1)

    def _init_azure_client(self) -> ContainerClient | None:
        """
        Initialise the Azure Blob Storage container client.

        Called once from ``__init__``. Only active when ``DATA_MODE`` is
        ``'CLOUD'``; returns ``None`` immediately in ``'LOCAL'`` mode so
        the class can operate fully offline without touching Azure SDK code
        paths.

        Follows the Dictator Pattern: if CLOUD mode is configured but the
        connection cannot be established (bad connection string, missing
        container, network failure), the process exits immediately rather
        than continuing in a degraded state where sync calls would silently
        fail.

        Returns:
            ContainerClient: Authenticated, container-scoped client ready for
                upload/download operations.
            None: When ``DATA_MODE`` is ``'LOCAL'``.

        Raises:
            SystemExit: If ``DATA_MODE`` is ``'CLOUD'`` and the container
                cannot be reached. Wraps the underlying Azure SDK exception
                in a structured diagnostic block before exiting.
        """
        if self._settings.data_mode != "CLOUD":
            debug_block = (
                f"\n{'=' * 60}\n"
                f"LOCAL MODE — Azure Blob client not initialised\n"
                f"Container : {self._settings.container_name}\n"
                f"Mode      : {self._settings.data_mode}\n"
                f"{'=' * 60}"
            )
            logger.debug(debug_block)
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

            info_block = (
                f"\n{'=' * 60}\n"
                f"AZURE BLOB CONNECTED\n"
                f"Container : {self._settings.container_name}\n"
                f"Mode      : {self._settings.data_mode}\n"
                f"{'=' * 60}"
            )
            logger.info(info_block)
            return client

        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"AZURE CONNECTION FAILURE\n"
                f"Container : {self._settings.container_name}\n"
                f"Error     : {e}\n\n"
                f"CONTEXT: DATA_MODE is 'CLOUD' but Azure Blob Storage is unreachable.\n"
                f"  [!] Check AZURE_STORAGE_CONN in your .env file.\n"
                f"  [!] Verify the container '{self._settings.container_name}' exists.\n"
                f"  [^] Confirm network connectivity from this host to Azure.\n"
                f"\nFIX: Correct AZURE_STORAGE_CONN or switch DATA_MODE=LOCAL for offline use.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            sys.exit(1)

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
            error_block = (
                f"\n{'%' * 60}\n"
                f"PARQUET WRITE FAILURE\n"
                f"File: {file_path}\n"
                f"Error: {e}\n\n"
                f"CONTEXT: This may indicate:\n"
                f"  [%] Schema mismatch between existing file and new data.\n"
                f"  [!] Disk full or permission denied at write time.\n"
                f"  [%] Corrupted Parquet header from a previous partial write.\n"
                f"\nFIX: Inspect the file manually. If corrupted, delete and re-backfill.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
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
            logger.debug("save_tick_batch called with empty list -- no-op.")
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

        logger.debug(f"Committing {len(ticks)} ticks for {symbol} -> {file_path.name}")

        with self._get_file_lock(file_path):
            self._atomic_upsert(file_path, new_data)
        logger.info(f"[+] Committed {len(ticks)} ticks for {symbol}.")
        # Sync to Azure if in CLOUD mode, but do not fail the write if the upload fails — the local file is the source of truth and can be re-synced later if needed.
        if self._container_client is not None:
            blob_name = f"raw/{symbol}_ticks.parquet"
            try:
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.debug(f"✓ Synced ticks to Azure: {blob_name}")
            except Exception as e:
                logger.warning(f"X Failed to sync ticks to Azure: {e}")
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
                f"[%] save_bar called with incomplete bar for {bar.symbol} "
                f"@ {bar.timestamp}. Incomplete bars are excluded from processed storage."
            )
            return False
        logger.debug(
            f"Committing bar for {bar.symbol} [{bar.timeframe.value}] "
            f"@ {bar.timestamp} -> {file_path.name}"
        )

        with self._get_file_lock(file_path):
            self._atomic_upsert(file_path, new_data)

        logger.info(
            f"[+] Committed bar for {bar.symbol} "
            f"[{bar.timeframe.value}] @ {bar.timestamp}."
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
            logger.warning(
                "[^] sync_to_azure called but DATA_MODE is LOCAL — skipping upload."
            )
            return False

        if not local_path.exists():
            error_block = (
                f"\n{'!' * 60}\n"
                f"SYNC TO AZURE FAILURE\n"
                f"File Not Found: {local_path}\n\n"
                f"CONTEXT: Cannot upload a file that does not exist locally.\n"
                f"\nFIX: Ensure the file is written to disk before calling sync_to_azure.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
            return False

        target_blob: str = blob_name or local_path.name
        try:
            logger.info(f"Syncing {local_path.name} -> Azure Blob '{target_blob}' ...")
            blob_client = self._container_client.get_blob_client(target_blob)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            info_block = (
                f"\n{'+' * 60}\n"
                f"AZURE SYNC COMPLETE\n"
                f"Local : {local_path}\n"
                f"Blob  : {target_blob}\n"
                f"{'+' * 60}"
            )
            logger.info(info_block)
            return True

        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"AZURE UPLOAD FAILURE\n"
                f"Local : {local_path}\n"
                f"Blob  : {target_blob}\n"
                f"Error : {e}\n\n"
                f"CONTEXT: Could not upload file to Azure Blob Storage.\n"
                f"  [^] Check network connectivity and storage account permissions.\n"
                f"\nFIX: Inspect the error above and re-run sync after resolving.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
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
            logger.warning(
                "[^] pull_from_azure called but DATA_MODE is LOCAL — skipping download."
            )
            return False

        try:
            logger.info(f"Pulling '{blob_name}' from Azure -> {local_path} ...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            info_block = (
                f"\n{'+' * 60}\n"
                f"AZURE PULL COMPLETE\n"
                f"Blob  : {blob_name}\n"
                f"Local : {local_path}\n"
                f"{'+' * 60}"
            )
            logger.info(info_block)
            return True

        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"AZURE DOWNLOAD FAILURE\n"
                f"Blob  : {blob_name}\n"
                f"Local : {local_path}\n"
                f"Error : {e}\n\n"
                f"CONTEXT: Could not download file from Azure Blob Storage.\n"
                f"  [!] Verify the blob name is correct and exists in the container.\n"
                f"  [^] Check network connectivity and storage account permissions.\n"
                f"\nFIX: Verify blob_name against your Azure Storage container contents.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
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
            logger.warning(
                "[^] save_model called but DATA_MODE is LOCAL — skipping upload."
            )
            return False

        for local_path in (artifact_path, metadata_path):
            if not local_path.exists():
                error_block = (
                    f"\n{'!' * 60}\n"
                    f"SAVE MODEL FAILURE\n"
                    f"File Not Found: {local_path}\n\n"
                    f"CONTEXT: Cannot upload a model file that does not exist locally.\n"
                    f"\nFIX: Ensure the trainer writes the artifact to disk before "
                    f"calling save_model.\n"
                    f"{'!' * 60}"
                )
                logger.error(error_block)
                return False

        try:
            for local_path in (artifact_path, metadata_path):
                blob_name = f"models/{local_path.name}"
                logger.info(
                    f"Uploading {local_path.name} -> Azure Blob '{blob_name}' ..."
                )
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

            info_block = (
                f"\n{'+' * 60}\n"
                f"SAVE MODEL COMPLETE\n"
                f"Artifact  : models/{artifact_path.name}\n"
                f"Metadata  : models/{metadata_path.name}\n"
                f"{'+' * 60}"
            )
            logger.info(info_block)
            return True

        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"SAVE MODEL UPLOAD FAILURE\n"
                f"Artifact  : {artifact_path}\n"
                f"Metadata  : {metadata_path}\n"
                f"Error     : {e}\n\n"
                f"CONTEXT: Could not upload model files to Azure Blob Storage.\n"
                f"  [^] Check network connectivity and storage account permissions.\n"
                f"\nFIX: Inspect the error above and re-run save_model after resolving.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
            return False

    def load_model(
        self,
        artifact_filename: str,
        local_dir: Path | None = None,
    ) -> Path | None:
        """
        Download a model artifact and its JSON metadata sidecar from Azure.

        Pulls both the artifact and its ``<stem>_metadata.json`` sidecar
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
                               as ``<stem>_metadata.json``.
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
            logger.warning(
                "[^] load_model called but DATA_MODE is LOCAL — skipping download."
            )
            return None

        dest_dir: Path = (
            local_dir
            if local_dir is not None
            else (Path(__file__).resolve().parents[2] / "models")
        )
        dest_dir.mkdir(parents=True, exist_ok=True)

        artifact_stem = Path(artifact_filename).stem
        metadata_filename = f"{artifact_stem}_metadata.json"

        try:
            for filename in (artifact_filename, metadata_filename):
                blob_name = f"models/{filename}"
                local_path = dest_dir / filename
                logger.info(f"Pulling '{blob_name}' from Azure -> {local_path} ...")
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(local_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())

            local_artifact: Path = dest_dir / artifact_filename
            info_block = (
                f"\n{'+' * 60}\n"
                f"LOAD MODEL COMPLETE\n"
                f"Artifact  : {local_artifact}\n"
                f"Metadata  : {dest_dir / metadata_filename}\n"
                f"{'+' * 60}"
            )
            logger.info(info_block)
            return local_artifact

        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"LOAD MODEL DOWNLOAD FAILURE\n"
                f"Artifact  : models/{artifact_filename}\n"
                f"Metadata  : models/{metadata_filename}\n"
                f"Error     : {e}\n\n"
                f"CONTEXT: Could not download model files from Azure Blob Storage.\n"
                f"  [!] Verify the blob names exist in the container under models/.\n"
                f"  [^] Check network connectivity and storage account permissions.\n"
                f"\nFIX: Verify artifact_filename against your Azure Storage container "
                f"contents.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
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
                "[^] save_processed_features called but DATA_MODE is LOCAL "
                "— skipping upload."
            )
            return False

        if not local_path.exists():
            error_block = (
                f"\n{'!' * 60}\n"
                f"SAVE PROCESSED FEATURES FAILURE\n"
                f"File Not Found: {local_path}\n\n"
                f"CONTEXT: Cannot upload a feature matrix that does not exist locally.\n"
                f"\nFIX: Ensure FeatureEngineer.build_matrix() has written the Parquet "
                f"file to disk before calling save_processed_features.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
            return False

        blob_name = f"processed/{symbol}_{expiry_key}_features.parquet"
        try:
            logger.info(f"Uploading {local_path.name} -> Azure Blob '{blob_name}' ...")
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            info_block = (
                f"\n{'+' * 60}\n"
                f"SAVE PROCESSED FEATURES COMPLETE\n"
                f"Symbol     : {symbol}\n"
                f"Expiry Key : {expiry_key}\n"
                f"Blob       : {blob_name}\n"
                f"{'+' * 60}"
            )
            logger.info(info_block)
            return True

        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"SAVE PROCESSED FEATURES UPLOAD FAILURE\n"
                f"Local  : {local_path}\n"
                f"Blob   : {blob_name}\n"
                f"Error  : {e}\n\n"
                f"CONTEXT: Could not upload feature matrix to Azure Blob Storage.\n"
                f"  [^] Check network connectivity and storage account permissions.\n"
                f"\nFIX: Inspect the error above and re-run save_processed_features "
                f"after resolving.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
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
            if not file_path.exists():
                logger.info(
                    f"No tick file found for {symbol}. "
                    f"First run -- full backfill required."
                )
                return None

            try:
                # Columnar read: load only the timestamp column.
                # PyArrow pushes this predicate to the Parquet reader,
                # avoiding full file deserialisation. Fast on any file size.
                df: pd.DataFrame = pd.read_parquet(file_path, engine=_ENGINE, columns=[_TS_COLUMN])  # type: ignore[arg-type]

                if df.empty:
                    logger.warning(
                        f"[%] Tick file for {symbol} exists but is empty. "
                        f"Treating as first run."
                    )
                    return None

                last_ts: pd.Timestamp = df[_TS_COLUMN].max()
                logger.info(f"Last known timestamp for {symbol}: {last_ts}")
                return last_ts

            except Exception as e:
                error_block = (
                    f"\n{'%' * 60}\n"
                    f"STORAGE QUERY FAILURE\n"
                    f"Symbol : {symbol}\n"
                    f"File   : {file_path}\n"
                    f"Error  : {e}\n\n"
                    f"CONTEXT: Cannot read timestamp index from tick file.\n"
                    f"  This may indicate a corrupted Parquet header or schema mismatch.\n"
                    f"\nFIX: Delete {file_path.name} and trigger a full backfill.\n"
                    f"{'%' * 60}"
                )
                logger.critical(error_block)
                raise StorageError(
                    f"Tick file for {symbol} is corrupted and cannot be read.",
                    symbol=symbol,
                    path=str(file_path),
                ) from e

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
            logger.info(
                f"No processed bar file found for {symbol} [{timeframe}]. "
                f"Training data not yet available."
            )
            return None
        with self._get_file_lock(file_path):
            try:
                pf = pq.ParquetFile(file_path)
                n_total: int = pf.metadata.num_rows

                if n_total == 0:
                    logger.warning(
                        f"[%] Bar file for {symbol} [{timeframe}] exists but is empty."
                    )
                    return None

                if max_rows is not None and n_total > max_rows:
                    # Read only the row groups that contain the tail max_rows rows,
                    # avoiding loading the entire file into RAM.
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
                        f"{max_rows} for {symbol} [{timeframe}]."
                    )
                else:
                    df = pf.read().to_pandas()

                if df.empty:
                    logger.warning(
                        f"[%] Bar file for {symbol} [{timeframe}] exists but is empty."
                    )
                    return None

                # Ensure timestamp column exists
                if _TS_COLUMN not in df.columns:
                    error_block = (
                        f"\n{'!' * 60}\n"
                        f"STORAGE ERROR: Missing timestamp column\n"
                        f"Symbol: {symbol}\n"
                        f"Timeframe: {timeframe}\n"
                        f"File: {file_path.name}\n"
                        f"Columns found: {df.columns.tolist()}\n"
                        f"Expected column: '{_TS_COLUMN}'\n"
                        f"{'!' * 60}"
                    )
                    logger.critical(error_block)
                    raise StorageError(
                        f"Missing timestamp column in {file_path.name}",
                        symbol=symbol,
                        path=str(file_path),
                    )

                # Convert timestamp to datetime and set as index
                df[_TS_COLUMN] = pd.to_datetime(df[_TS_COLUMN])
                df.set_index(_TS_COLUMN, inplace=True)
                # Sort chronologically — ML models require temporal order.
                df.sort_index(inplace=True)

                # Final trim in case the last row group contained more rows than needed.
                if max_rows is not None and len(df) > max_rows:
                    df = df.tail(max_rows).reset_index(drop=True)

                logger.info(f"Loaded {len(df)} bars for {symbol} [{timeframe}].")
                return df

            except StorageError:
                raise
            except Exception as e:
                error_block = (
                    f"\n{'%' * 60}\n"
                    f"STORAGE READ FAILURE\n"
                    f"Symbol    : {symbol}\n"
                    f"Timeframe : {timeframe}\n"
                    f"File      : {file_path}\n"
                    f"Error     : {e}\n\n"
                    f"CONTEXT: Cannot load bar data for training.\n"
                    f"\nFIX: Delete {file_path.name} and re-run the aggregation pipeline.\n"
                    f"{'%' * 60}"
                )
                logger.critical(error_block)
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
            debug_block = (
                f"\n{'-' * 60}\n"
                f"SAVE_BAR_BATCH: Empty batch received -- no-op."
                f"\n{'-' * 60}"
            )
            logger.debug(debug_block)
            return False

        # Guard: single symbol per batch (mirrors save_tick_batch contract)
        symbols_in_batch: set[str] = {b.symbol for b in bars}
        if len(symbols_in_batch) > 1:
            error_block = (
                f"\n{'!' * 60}\n"
                f"SAVE_BAR_BATCH FAILURE\n"
                f"Mixed symbols in batch: {symbols_in_batch}\n\n"
                f"CONTEXT: Each batch must contain bars for a single symbol only.\n"
                f"  This simplifies file management and ensures that each Parquet file\n"
                f"  corresponds to one symbol and one timeframe.\n"
                f"\nFIX: Split the batch into separate lists for each symbol and call save_bar_batch on each one.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
            raise ValueError(
                f"save_bar_batch received mixed symbols: {symbols_in_batch}. "
                f"Each batch must contain bars for a single symbol only."
            )

        # Guard: single timeframe per batch (one file per symbol+timeframe)
        timeframes_in_batch: set[str] = {b.timeframe.value for b in bars}
        if len(timeframes_in_batch) > 1:
            error_block = (
                f"\n{'!' * 60}\n"
                f"SAVE_BAR_BATCH FAILURE\n"
                f"Mixed timeframes in batch: {timeframes_in_batch}\n\n"
                f"CONTEXT: Each batch must contain bars for a single timeframe only.\n"
                f"  This ensures that each Parquet file corresponds to one symbol and one timeframe.\n"
                f"\nFIX: Split the batch into separate lists for each timeframe and call save_bar_batch on each one.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
            raise ValueError(
                f"save_bar_batch received mixed timeframes: {timeframes_in_batch}. "
                f"Each batch must contain bars for a single timeframe only."
            )

        symbol: str = bars[0].symbol
        timeframe: str = bars[0].timeframe.value
        file_path: Path = self._processed_path(symbol, timeframe)
        new_data: pd.DataFrame = pd.DataFrame([b.to_dict() for b in bars])
        debug_block = (
            f"\n{'-' * 60}\n"
            f"SAVE_BAR_BATCH: Committing batch of {len(bars)} bars for {symbol} [{timeframe}].\n"
            f"Target file: {file_path.name}\n"
            f"{'-' * 60}"
        )
        logger.debug(debug_block)

        with self._get_file_lock(file_path):
            self._atomic_upsert(file_path, new_data)

        info_block = (
            f"\n{'+' * 60}\n"
            f"SAVE_BAR_BATCH SUCCESS\n"
            f"Committed {len(bars)} bars for {symbol} [{timeframe}].\n"
            f"File: {file_path.name}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
        if self._container_client is not None:
            blob_name = f"processed/{symbol}_{timeframe}.parquet"
            try:
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.info(f"✓ Synced to Azure: {blob_name}")
            except Exception as e:
                logger.warning(f"X Failed to sync to Azure: {e}")

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
            return len(df)
        except Exception as e:
            logger.warning(f"[%] Could not count ticks for {symbol}: {e}")
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
            return sorted(
                [
                    f.stem.replace("_ticks", "")
                    for f in self.raw_dir.glob("*_ticks.parquet")
                ]
            )
        except Exception as e:
            logger.warning(f"[%] Could not list symbols from raw directory: {e}")
            return []
