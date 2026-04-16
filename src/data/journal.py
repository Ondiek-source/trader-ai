"""
data/journal.py — The Auditor.

Responsible for the persistent, append-only recording of internal trading
events and model signals. While Storage (storage.py) owns the market-price
vault, the Journal owns the bot's decision ledger — what was signalled, what
was traded, and what the outcome was. This separation preserves the Isolation
Principle: no module that needs market data must touch decision records, and
vice versa.

Design Principles:
    - Append-Only: The Journal never updates existing rows. Each write
        produces one new row in the relevant table. There is no deduplication.
    - Crash-Safe: Writes use an atomic tmp-then-rename strategy. A crash mid-
        write leaves the live file intact; only the .tmp file is corrupted.
    - Thread-Safe: A single lock serialises all writes. Reads also acquire
        the lock to prevent reads during an in-progress write cycle.
    - Typed: Both record schemas (TradeEntry, SignalEntry) are validated
        frozen dataclasses. Invalid records raise ValueError immediately at
        construction time — before any I/O is attempted.
    - Versioned: Every write embeds _SCHEMA_VERSION in Parquet metadata.
        A version mismatch on read raises JournalError before data is loaded.

Design Document: docs/data/Journal/Journal.md
"""

from __future__ import annotations

import sys
import logging
import threading
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ── Write Strategy ────────────────────────────────────────────────────────────

# Force pyarrow: industry standard for financial time-series efficiency.
_ENGINE: Literal["pyarrow"] = "pyarrow"

# Snappy chosen for fastest decompression at acceptable compression ratio.
_COMPRESSION: Literal["snappy"] = "snappy"

# Schema version embedded in every Parquet write.
# Increment when TradeEntry or SignalEntry fields are added, renamed, or removed.
# A mismatch on read raises JournalError rather than producing silent NaN columns.
_SCHEMA_VERSION: int = 1
_SCHEMA_VERSION_KEY: bytes = b"trader_ai_journal_schema_version"

# ── Allowed Values ────────────────────────────────────────────────────────────

# Valid values for TradeEntry.side and SignalEntry.direction.
# Binary-options terminology: CALL = buy (up), PUT = sell (down).
_VALID_SIDES: frozenset[str] = frozenset({"CALL", "PUT"})
_VALID_DIRECTIONS: frozenset[str] = frozenset({"CALL", "PUT"})


# ── Custom Exception ──────────────────────────────────────────────────────────


class JournalError(Exception):
    """
    Raised when the Journal cannot fulfil a write or read commitment.

    Distinct from ``ValueError`` (schema validation) and
    :class:`~data.storage.StorageError` (market-price vault failures).
    Allows the engine to catch journal-specific I/O failures without
    masking unrelated exceptions.

    Attributes:
        table: The journal table involved (e.g., ``"trades"``, ``"signals"``).
                Empty string if the failure is not table-specific.
        path:  The file path involved, if applicable. Empty string otherwise.
    """

    def __init__(
        self,
        message: str,
        table: str = "",
        path: str = "",
    ) -> None:
        self.table = table
        self.path = path
        super().__init__(message)


# ── Journal Schemas ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeEntry:
    """
    Immutable schema for a finalised binary-options trade execution record.

    Constructed by the execution layer immediately after a trade closes.
    Frozen to prevent mutation after construction — any amendment requires
    creating a new entry with a distinct ``signal_id``.

    Attributes:
        timestamp:        UTC datetime when the trade closed.
        symbol:           Pure currency pair name (e.g., ``"EUR_USD"``).
        side:             Trade direction. Must be ``"CALL"`` or ``"PUT"``.
        entry_price:      Price at which the trade was opened. Must be > 0.
        exit_price:       Price at which the trade expired. Must be > 0.
        result:           Profit (positive) or loss (negative) in account
                            currency units.
        duration_seconds: Trade window in seconds. Must be > 0.
        signal_id:        Unique identifier of the signal that generated
                            this trade. Must be a non-empty string.

    Raises:
        ValueError: If any field violates the constraints described above.

    Example:
        >>> entry = TradeEntry(
        ...     timestamp=datetime.utcnow(),
        ...     symbol="EUR_USD",
        ...     side="CALL",
        ...     entry_price=1.0850,
        ...     exit_price=1.0865,
        ...     result=8.50,
        ...     duration_seconds=60,
        ...     signal_id="sig-20240115-001",
        ... )
    """

    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    result: float
    duration_seconds: int
    signal_id: str

    def __post_init__(self) -> None:
        """
        Validate trade entry integrity at construction time.

        Enforces six rules:
            1. ``side`` must be one of :data:`_VALID_SIDES`.
            2. ``entry_price`` must be > 0.
            3. ``exit_price`` must be > 0.
            4. ``duration_seconds`` must be > 0.
            5. ``signal_id`` must be a non-empty string.
            6. If ``timestamp`` is timezone-aware, it must be UTC. A non-UTC
                aware datetime silently loses its offset if stripped — rejected.

        Raises:
            ValueError: If any rule is violated. All violations are collected
                and reported in a single message.
        """
        # Reject non-UTC timezone-aware timestamps (BL-03 compliance).
        if self.timestamp.tzinfo is not None:
            if self.timestamp.utcoffset() != timedelta(0):
                error_block = (
                    f"\n{'%' * 60}\n"
                    f"TRADE ENTRY CONSTRUCTION FAILURE: Non-UTC timezone rejected\n"
                    f"Symbol : {self.symbol}\n"
                    f"Side   : {self.side}\n"
                    f"Timestamp: {self.timestamp}\n"
                    f"Context: Record rejected at schema level to prevent ledger corruption.\n"
                    f"{'%' * 60}"
                )
                logger.critical(error_block)
                raise ValueError(
                    f"[!] Non-UTC timezone rejected for TradeEntry({self.symbol}): "
                    f"tzinfo={self.timestamp.tzinfo!r}. Convert to UTC before construction."
                )
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=None))

        violations: list[str] = []

        if self.side not in _VALID_SIDES:
            violations.append(
                f"  [!] Invalid side: '{self.side}' (must be one of {sorted(_VALID_SIDES)})"
            )
        if self.entry_price <= 0:
            violations.append(
                f"  [%] Invalid entry_price: {self.entry_price} (must be > 0)"
            )
        if self.exit_price <= 0:
            violations.append(
                f"  [%] Invalid exit_price: {self.exit_price} (must be > 0)"
            )
        if self.duration_seconds <= 0:
            violations.append(
                f"  [%] Invalid duration_seconds: {self.duration_seconds} (must be > 0)"
            )
        if not self.signal_id or not self.signal_id.strip():
            violations.append("  [!] signal_id must be a non-empty string.")

        if violations:
            error_block = (
                f"\n{'%' * 60}\n"
                f"TRADE ENTRY CONSTRUCTION FAILURE: {len(violations)} VIOLATION(S)\n"
                f"Symbol : {self.symbol}\n"
                f"Side   : {self.side}\n"
                f"\n".join(violations) + "\n"
                f"Context: Record rejected at schema level to prevent ledger corruption.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"TradeEntry({self.symbol}): {len(violations)} integrity violation(s). See logs."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this trade entry to a flat dictionary.

        Uses :func:`dataclasses.asdict` for field-by-field conversion.
        Compatible with ``pandas.DataFrame.from_records()`` and PyArrow
        for Parquet serialisation.

        Returns:
            dict[str, Any]: All ``TradeEntry`` fields as a flat mapping.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"TradeEntry({self.symbol} "
            f"side={self.side} "
            f"entry={self.entry_price:.5f} exit={self.exit_price:.5f} "
            f"result={self.result:+.2f} "
            f"dur={self.duration_seconds}s "
            f"ts={self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')})"
        )


@dataclass(frozen=True)
class SignalEntry:
    """
    Immutable schema for a model-generated trading signal record.

    Constructed by the signal layer immediately after the ML model produces
    a directional prediction. Stored independently of ``TradeEntry`` so that
    signals which do not result in a trade (e.g., below confidence threshold,
    martingale suppressed) are still auditable.

    Attributes:
        timestamp:     UTC datetime when the signal was generated.
        symbol:        Pure currency pair name (e.g., ``"EUR_USD"``).
        confidence:    Model confidence score. Must be in ``[0.0, 1.0]``.
        direction:     Signal direction. Must be ``"CALL"`` or ``"PUT"``.
        model_version: Identifier of the model version that produced this
                        signal (e.g., ``"rf-v3.2.1"``). Must be non-empty.
        metadata:      JSON-encoded snapshot of the feature vector and any
                        diagnostic context used by the model. May be empty
                        for lightweight deployments.

    Raises:
        ValueError: If ``confidence`` is outside ``[0.0, 1.0]``, ``direction``
            is not in :data:`_VALID_DIRECTIONS`, or ``model_version`` is empty.

    Example:
        >>> signal = SignalEntry(
        ...     timestamp=datetime.utcnow(),
        ...     symbol="EUR_USD",
        ...     confidence=0.78,
        ...     direction="CALL",
        ...     model_version="rf-v3.2.1",
        ...     metadata='{"rsi": 42.1, "macd": 0.0003}',
        ... )
    """

    timestamp: datetime
    symbol: str
    confidence: float
    direction: str
    model_version: str
    metadata: str

    def __post_init__(self) -> None:
        """
        Validate signal entry integrity at construction time.

        Enforces four rules:
            1. ``confidence`` must be in ``[0.0, 1.0]``.
            2. ``direction`` must be one of :data:`_VALID_DIRECTIONS`.
            3. ``model_version`` must be a non-empty string.
            4. If ``timestamp`` is timezone-aware, it must be UTC.

        Raises:
            ValueError: If any rule is violated.
        """
        # Reject non-UTC timezone-aware timestamps (BL-03 compliance).
        if self.timestamp.tzinfo is not None:
            if self.timestamp.utcoffset() != timedelta(0):
                error_block = (
                    f"\n{'%' * 60}\n"
                    f"SIGNAL ENTRY CONSTRUCTION FAILURE: Non-UTC timezone rejected\n"
                    f"Symbol : {self.symbol}\n"
                    f"Direction : {self.direction}\n"
                    f"Timestamp: {self.timestamp}\n"
                    f"Context: Record rejected at schema level to prevent ledger corruption.\n"
                    f"{'%' * 60}"
                )
                logger.critical(error_block)
                raise ValueError(
                    f"[!] Non-UTC timezone rejected for SignalEntry({self.symbol}): "
                    f"tzinfo={self.timestamp.tzinfo!r}. Convert to UTC before construction."
                )
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=None))

        violations: list[str] = []

        if not (0.0 <= self.confidence <= 1.0):
            violations.append(
                f"  [%] Invalid confidence: {self.confidence} (must be in [0.0, 1.0])"
            )
        if self.direction not in _VALID_DIRECTIONS:
            violations.append(
                f"  [!] Invalid direction: '{self.direction}' "
                f"(must be one of {sorted(_VALID_DIRECTIONS)})"
            )
        if not self.model_version or not self.model_version.strip():
            violations.append("  [!] model_version must be a non-empty string.")

        if violations:
            error_block = (
                f"\n{'%' * 60}\n"
                f"SIGNAL ENTRY CONSTRUCTION FAILURE: {len(violations)} VIOLATION(S)\n"
                f"Symbol    : {self.symbol}\n"
                f"Direction : {self.direction}\n"
                f"\n".join(violations) + "\n"
                f"Context: Record rejected at schema level to prevent ledger corruption.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"SignalEntry({self.symbol}): {len(violations)} integrity violation(s). See logs."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this signal entry to a flat dictionary.

        Returns:
            dict[str, Any]: All ``SignalEntry`` fields as a flat mapping.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"SignalEntry({self.symbol} "
            f"dir={self.direction} conf={self.confidence:.3f} "
            f"model={self.model_version} "
            f"ts={self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')})"
        )


# ── The Auditor ───────────────────────────────────────────────────────────────


class Journal:
    """
    Thread-safe, append-only Parquet ledger for trading events and signals.

    Acts as the sole gateway between in-memory decision records
    (``TradeEntry``, ``SignalEntry``) and the on-disk audit trail. Enforces
    path isolation and crash-safe atomic writes via a tmp-then-rename strategy:
    every write lands in a ``.tmp`` file first, then is atomically renamed to
    the live ``.parquet`` — a crash mid-write leaves the live file intact.

    Two separate tables are maintained:

    +----------+---------------------------+----------------------------------+
    | Table    | File                      | Source Record                    |
    +----------+---------------------------+----------------------------------+
    | trades   | journal/trades.parquet    | :class:`TradeEntry`              |
    | signals  | journal/signals.parquet   | :class:`SignalEntry`             |
    +----------+---------------------------+----------------------------------+

    Schema Versioning:
        Every write embeds :data:`_SCHEMA_VERSION` in the Parquet file
        metadata. Any read (for append or for query) first checks the stored
        version against the current constant; a mismatch raises
        :class:`JournalError` with a clear instruction to re-run the journal
        export rather than silently producing NaN columns.

    Thread Safety:
        All public methods acquire :attr:`_lock` before any file I/O.
        Reads and writes are fully serialised — no concurrent reader can
        observe a partially-written append.

    Attributes:
        _settings:    Validated frozen :class:`~core.config.Config`.
        journal_dir:  Absolute path to the journal Parquet directory.
        _lock:        Threading lock that serialises all I/O operations.

    Example:
        >>> journal = Journal()
        >>> journal.record_trade(trade_entry)
        True
        >>> df = journal.get_trade_history(limit=50)
    """

    def __init__(self) -> None:
        """
        Initialise the Journal and provision the journal directory.

        Acquires validated configuration via :func:`~core.config.get_settings`
        and derives the journal directory from the source file's location
        (``<project_root>/data/processed/journal/``), mirroring the path
        strategy used by :class:`~data.storage.Storage`.

        Follows the Dictator Pattern: if the directory cannot be created
        or verified (e.g., permission denied), the process exits immediately
        with a structured diagnostic block. A partially-provisioned Journal
        is more dangerous than no Journal — writes would silently fail.

        Raises:
            SystemExit: If the journal directory cannot be created or a
                write-permission check fails.
        """

        # Derive the project root from this file's location.
        # journal.py lives at src/data/journal.py → 3 parents up = project root.
        root_dir: Path = Path(__file__).resolve().parent.parent.parent
        self.journal_dir: Path = root_dir / "data" / "processed" / "journal"
        self._lock: threading.Lock = threading.Lock()

        try:
            self.journal_dir.mkdir(parents=True, exist_ok=True)
            # Write-permission smoke test — catch permission errors at boot,
            # not silently at first write (consistent with Storage.__init__).
            _probe: Path = self.journal_dir / ".write_probe"
            _probe.touch()
            _probe.unlink()
        except PermissionError as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"JOURNAL INITIALIZATION FAILURE: PERMISSION DENIED\n"
                f"Directory : {self.journal_dir}\n"
                f"Error     : {e}\n\n"
                f"CONTEXT: The journal directory exists but is not writable.\n"
                f"  [!] Check filesystem permissions for the data/processed/journal/ path.\n"
                f"  [^] If running in Docker, verify the volume mount is read-write.\n"
                f"\nFIX: Grant write access to {self.journal_dir} and restart.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            sys.exit(1)
        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"JOURNAL INITIALIZATION FAILURE\n"
                f"Directory : {self.journal_dir}\n"
                f"Error     : {e}\n\n"
                f"CONTEXT: Could not create or access the journal directory.\n"
                f"  [!] Possible causes: disk full, OS path limit, bad mount.\n"
                f"\nFIX: Resolve the filesystem condition and restart.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            sys.exit(1)

        info_block = (
            f"\n{'+' * 60}\n"
            f"JOURNAL READY\n"
            f"Directory : {self.journal_dir}\n"
            f"Schema    : v{_SCHEMA_VERSION}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)

    # ── Private: Schema Versioning ────────────────────────────────────────────

    def _check_schema_version(self, file_path: Path) -> None:
        """
        Read the Parquet footer metadata and raise on version mismatch.

        Called before loading any existing journal file so that a stale file
        (written by an older code version with different columns) raises a
        clear, actionable error rather than producing silent NaN columns or
        a confusing ``pd.concat`` failure.

        Args:
            file_path: Path to an existing Parquet journal file.

        Raises:
            JournalError: If the file has no schema version metadata, or if
                the stored version does not match :data:`_SCHEMA_VERSION`.
        """
        meta = pq.read_metadata(file_path)
        stored: bytes | None = (meta.metadata or {}).get(_SCHEMA_VERSION_KEY)
        if stored is None or int(stored) != _SCHEMA_VERSION:
            got: str = stored.decode() if stored else "none"
            error_block = (
                f"\n{'%' * 60}\n"
                f"SCHEMA VERSION MISMATCH\n"
                f"File    : {file_path.name}\n"
                f"Stored  : v{got}\n"
                f"Expected: v{_SCHEMA_VERSION}\n"
                f"Context : The journal file has a different schema version.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise JournalError(
                f"Schema version mismatch in {file_path.name}: "
                f"file=v{got}, code=v{_SCHEMA_VERSION}. "
                f"Re-export the journal to regenerate with current schema.",
                path=str(file_path),
            )

    def _write_versioned_parquet(self, file_path: Path, df: pd.DataFrame) -> None:
        """
        Write a DataFrame to a Parquet file with :data:`_SCHEMA_VERSION` in metadata.

        Uses PyArrow directly so that the custom key-value metadata survives
        the round-trip and is readable by :meth:`_check_schema_version` on
        the next append cycle.

        Args:
            file_path: Destination path for the Parquet file. Typically a
                        ``.tmp`` path — the caller handles the rename.
            df:        DataFrame to serialise. The index is not preserved.
        """
        table = pa.Table.from_pandas(df, preserve_index=False)
        existing_meta: dict = table.schema.metadata or {}
        new_meta = {
            **existing_meta,
            _SCHEMA_VERSION_KEY: str(_SCHEMA_VERSION).encode(),
        }
        table = table.replace_schema_metadata(new_meta)
        pq.write_table(table, file_path, compression=_COMPRESSION)

    # ── Private: Path Resolution ──────────────────────────────────────────────

    def _get_path(self, table_name: str) -> Path:
        """
        Resolve the canonical Parquet file path for a journal table.

        Centralises path construction so the naming convention can be changed
        in one place without hunting through every method.

        Args:
            table_name: Table identifier (e.g., ``"trades"``, ``"signals"``).

        Returns:
            Path: Absolute path to the table's Parquet file.
                    e.g., ``<project_root>/data/processed/journal/trades.parquet``
        """
        return self.journal_dir / f"{table_name}.parquet"

    # ── Public: Write Operations ──────────────────────────────────────────────

    def record_trade(self, entry: TradeEntry) -> bool:
        """
        Append a finalised trade execution record to the permanent ledger.

        Delegates to :meth:`_append_to_table` which performs a crash-safe
        atomic write. The entry is validated by :class:`TradeEntry.__post_init__`
        before this method is called — if an invalid entry reaches here, the
        ``ValueError`` from the dataclass propagates to the caller unchanged.

        Args:
            entry: A validated, frozen :class:`TradeEntry` instance.

        Returns:
            bool: ``True`` if the entry was committed successfully.

        Raises:
            JournalError: If the Parquet write fails (disk full, permission
                denied, schema version mismatch on existing file).

        Example:
            >>> success = journal.record_trade(trade_entry)
            >>> assert success is True
        """
        self._append_to_table("trades", [entry.to_dict()])
        info_block = (
            f"\n{'%' * 60}\n"
            f"TRADE ENTRY COMMITTED\n"
            f"Symbol : {entry.symbol}\n"
            f"Side   : {entry.side}\n"
            f"Result : {entry.result:+.2f}\n"
            f"Signal : {entry.signal_id}\n"
            f"{'%' * 60}"
        )
        logger.info(info_block)
        return True

    def record_signal(self, entry: SignalEntry) -> bool:
        """
        Append a model-generated signal record to the permanent ledger.

        Stores every signal regardless of whether it ultimately results in a
        trade. This allows post-hoc analysis of signal quality, confidence
        calibration, and suppression rates (e.g., martingale blocks).

        Args:
            entry: A validated, frozen :class:`SignalEntry` instance.

        Returns:
            bool: ``True`` if the entry was committed successfully.

        Raises:
            JournalError: If the Parquet write fails.

        Example:
            >>> success = journal.record_signal(signal_entry)
            >>> assert success is True
        """
        self._append_to_table("signals", [entry.to_dict()])
        info_block = (
            f"\n{'%' * 60}\n"
            f"SIGNAL ENTRY COMMITTED\n"
            f"Symbol : {entry.symbol}\n"
            f"Direction : {entry.direction}\n"
            f"Confidence : {entry.confidence:.3f}\n"
            f"Model Version : {entry.model_version}\n"
            f"{'%' * 60}"
        )
        logger.info(info_block)
        return True

    # ── Private: Atomic Append ────────────────────────────────────────────────

    def _append_to_table(self, table_name: str, data: list[dict]) -> None:
        """
        Perform a crash-safe atomic append to a Parquet journal table.

        Strategy:
            1. Check schema version of the existing file (if present).
            2. Load existing rows into memory.
            3. Concatenate with the new row(s).
            4. Write the combined DataFrame to a ``.tmp`` shadow file.
            5. Atomically rename ``.tmp`` → ``.parquet``.

        Steps 4 and 5 are what make the write crash-safe. A crash between
        steps 1–3 leaves the live ``.parquet`` intact. A crash mid-rename
        (step 5) leaves the ``.tmp`` file, which is cleaned up on the next
        call to this method.

        Args:
            table_name: Target table identifier (``"trades"`` or ``"signals"``).
            data:       List of dictionaries (one per record) to append.

        Raises:
            JournalError: If the schema version of the existing file does not
                match :data:`_SCHEMA_VERSION`, or if any filesystem operation
                (read, write, or rename) fails for any reason.
        """
        file_path: Path = self._get_path(table_name)
        tmp_path: Path = file_path.with_suffix(".tmp")

        new_df: pd.DataFrame = pd.DataFrame(data)

        with self._lock:
            # Clean up any stale .tmp file from a previous crashed write.
            if tmp_path.exists():
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"STALE .TMP FILE DETECTED\n"
                    f"Table : {table_name}\n"
                    f"File  : {tmp_path.name}\n"
                    f"Context: A .tmp file was found from a previous write attempt, "
                    f"indicating a possible crash during the last write cycle.\n"
                    f"  [!] The existing .parquet file is unaffected and will be used for this write.\n"
                    f"  [%] The stale .tmp file will be removed to prevent future conflicts.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)
                tmp_path.unlink()

            try:
                if file_path.exists():
                    # Version gate: raises JournalError on mismatch so the
                    # caller sees a clear message rather than a NaN column.
                    self._check_schema_version(file_path)
                    existing_df: pd.DataFrame = pd.read_parquet(
                        file_path, engine=_ENGINE  # type: ignore[arg-type]
                    )
                    final_df: pd.DataFrame = pd.concat(
                        [existing_df, new_df], ignore_index=True
                    )
                else:
                    final_df = new_df

                # Write to shadow file first, then atomically rename.
                self._write_versioned_parquet(tmp_path, final_df)
                tmp_path.replace(file_path)

            except JournalError:
                raise  # Schema errors propagate as-is without re-wrapping.
            except Exception as e:
                # Clean up any partially-written .tmp file before raising.
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass  # Best-effort cleanup; original error takes priority.

                error_block = (
                    f"\n{'%' * 60}\n"
                    f"JOURNAL WRITE FAILURE\n"
                    f"Table : {table_name}\n"
                    f"File  : {file_path}\n"
                    f"Error : {e}\n\n"
                    f"CONTEXT: This may indicate:\n"
                    f"  [!] Disk full or permission denied at write time.\n"
                    f"  [%] Corrupted Parquet header from a previous partial write.\n"
                    f"\nFIX: Inspect the file manually. If corrupted, delete and "
                    f"re-export from source.\n"
                    f"{'%' * 60}"
                )
                logger.critical(error_block)
                raise JournalError(
                    f"Journal write failed for table '{table_name}': {e}",
                    table=table_name,
                    path=str(file_path),
                ) from e

    # ── Public: Read / Query Operations ──────────────────────────────────────

    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve the most recent trade records from the permanent ledger.

        Performs a columnar full-file read inside the threading lock to
        prevent reads during an in-progress write cycle. Returns an empty
        ``DataFrame`` (not ``None``) when no trade file exists, so callers
        can always call ``df.empty`` without a ``None`` guard.

        Args:
            limit: Maximum number of most-recent rows to return.
                    Defaults to 100. Pass ``0`` for no limit.

        Returns:
            pd.DataFrame: The ``limit`` most recent trade rows, or an empty
                            DataFrame if no trade file exists yet.

        Raises:
            JournalError: If the existing file's schema version does not match
                :data:`_SCHEMA_VERSION`, or if the read fails for any reason.

        Example:
            >>> df = journal.get_trade_history(limit=50)
            >>> if not df.empty:
            ...     print(df["result"].sum())
        """
        return self._read_table("trades", limit)

    def get_signal_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve the most recent signal records from the permanent ledger.

        Mirrors :meth:`get_trade_history` exactly — see that method's
        documentation for full parameter and return value details.

        Args:
            limit: Maximum number of most-recent rows to return.
                    Defaults to 100. Pass ``0`` for no limit.

        Returns:
            pd.DataFrame: The ``limit`` most recent signal rows, or an empty
                            DataFrame if no signal file exists yet.

        Raises:
            JournalError: If the schema version mismatches or the read fails.

        Example:
            >>> df = journal.get_signal_history(limit=200)
            >>> high_conf = df[df["confidence"] >= 0.80]
        """
        return self._read_table("signals", limit)

    # ── Private: Table Reader ─────────────────────────────────────────────────

    def _read_table(self, table_name: str, limit: int) -> pd.DataFrame:
        """
        Thread-safe read of the most recent rows from a journal table.

        Acquires the lock before any filesystem operation so that reads are
        fully serialised with concurrent writes. Checks schema version before
        loading rows to catch stale files early.

        Args:
            table_name: Target table (``"trades"`` or ``"signals"``).
            limit:      Maximum rows to return (tail of the file). 0 = no limit.

        Returns:
            pd.DataFrame: Rows from the table, or an empty DataFrame if the
                            file does not exist.

        Raises:
            JournalError: On schema version mismatch or any read failure.
        """
        file_path: Path = self._get_path(table_name)

        with self._lock:
            if not file_path.exists():
                info_block = (
                    f"\n{'%' * 60}\n"
                    f"JOURNAL FILE NOT FOUND\n"
                    f"Table : {table_name}\n"
                    f"File  : {file_path.name}\n"
                    f"Context: No journal file exists yet for this table. Returning empty DataFrame.\n"
                    f"{'%' * 60}"
                )
                logger.info(info_block)
                return pd.DataFrame()

            try:
                self._check_schema_version(file_path)
                pf = pq.ParquetFile(file_path)
                n_total: int = pf.metadata.num_rows

                if limit > 0 and n_total > limit:
                    # Read only the tail row groups that contain the last `limit`
                    # rows, avoiding loading the entire journal into RAM.
                    row_groups: list[int] = []
                    rows_accumulated: int = 0
                    for i in range(pf.metadata.num_row_groups - 1, -1, -1):
                        row_groups.insert(0, i)
                        rows_accumulated += pf.metadata.row_group(i).num_rows
                        if rows_accumulated >= limit:
                            break
                    df: pd.DataFrame = pf.read_row_groups(row_groups).to_pandas()
                    return df.tail(limit)
                else:
                    df = pf.read().to_pandas()
                    return df

            except JournalError:
                raise  # Schema errors propagate as-is.
            except Exception as e:
                error_block = (
                    f"\n{'%' * 60}\n"
                    f"JOURNAL READ FAILURE\n"
                    f"Table : {table_name}\n"
                    f"File  : {file_path}\n"
                    f"Error : {e}\n\n"
                    f"CONTEXT: The Parquet file may be corrupted.\n"
                    f"  [%] Delete the file and re-export from source records.\n"
                    f"{'%' * 60}"
                )
                logger.error(error_block)
                raise JournalError(
                    f"Journal read failed for table '{table_name}': {e}",
                    table=table_name,
                    path=str(file_path),
                ) from e


# ── Singleton Gate ────────────────────────────────────────────────────────────
# The Journal is NOT instantiated at import time. Call get_journal() from your
# application entry point (pipeline.py or main.py) to trigger construction.
# This keeps imports safe for testing, linting, and doc generation.
_journal: Journal | None = None


def get_journal() -> Journal:
    """
    Return the global validated Journal instance (lazy singleton).

    Initialises on first call. Subsequent calls return the cached instance
    without re-constructing or re-provisioning the directory — mirroring the
    pattern used by :func:`~core.config.get_settings` and
    :func:`~data.historian.get_historian`.

    Using a singleton ensures that the threading lock is shared across all
    callers in the same process. Two independent ``Journal()`` instances would
    each have their own lock, allowing concurrent writes to corrupt the
    ``.tmp`` → ``.parquet`` rename sequence.

    Returns:
        Journal: The validated, initialised journal instance ready for use.

    Raises:
        SystemExit: If :class:`Journal.__init__` cannot provision the journal
            directory (delegated to the Dictator Pattern logic inside
            ``__init__``).

    Example:
        >>> journal = get_journal()
        >>> journal.record_signal(signal_entry)
    """
    global _journal
    if _journal is None:
        _journal = Journal()
    return _journal
