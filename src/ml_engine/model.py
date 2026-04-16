"""
ml/models.py — Core data structures for the Trader AI.

Standardized schemas for Ticks and Bars. Enforces data integrity via
high-visibility diagnostic blocks, matching the config.py pattern.
"""

from __future__ import annotations

import logging
import threading

from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any
from core.config import get_settings

logger = logging.getLogger(__name__)


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"


@dataclass(frozen=True)
class Tick:
    """
    Atomic market data unit. Validates price and source integrity.

    Represents a single bid/ask price snapshot captured live from
    Quotex or backfilled from Twelve Data. Frozen to prevent mutation
    after construction — any change requires creating a new instance.

    Attributes:
        timestamp: UTC datetime when the tick was captured.
        symbol: Pure currency pair name (e.g., "EUR_USD").
        bid: Current bid (sell) price. Must be > 0 and <= ask.
        ask: Current ask (buy) price. Must be > 0 and >= bid.
        source: Data origin identifier. Must be "TWELVE" or "QUOTEX".

    Raises:
        ValueError: If bid or ask is <= 0, bid > ask, or source is
            not in the allowed set.

    Example:
        >>> tick = Tick(
        ...     timestamp=datetime.utcnow(),
        ...     symbol="EUR_USD",
        ...     bid=1.0850,
        ...     ask=1.0852,
        ...     source="TWELVE",
        ... )
        >>> tick.mid_price
        1.0851
    """

    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    source: str

    def __post_init__(self) -> None:
        """
        Validate price and source integrity at construction time.

        Enforces three rules:
            1. bid > 0 and ask > 0 (no negative or zero prices)
            2. bid <= ask (no inverted spread)
            3. source in {"TWELVE", "QUOTEX"} (known data origin)

        Raises:
            ValueError: If any validation rule is violated. The error
                message uses the diagnostic language from config.py:
                '%' for logic/math violations, '!' for source errors.
        """

        # Reject non-UTC timezones before stripping. Silently stripping an
        # aware non-UTC datetime (e.g., US/Eastern) would store a corrupt
        # timestamp — 15:00 ET stored as if it were 15:00 UTC.
        if self.timestamp.tzinfo is not None:
            if self.timestamp.utcoffset() != timedelta(0):
                raise ValueError(
                    f"[!] Non-UTC timezone rejected for Tick({self.symbol}): "
                    f"tzinfo={self.timestamp.tzinfo!r}. Convert to UTC before construction."
                )
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=None))

        # Collect all violations to report them together in a critical log block
        violations = []
        settings = get_settings()
        if self.bid <= 0 or self.ask <= 0:
            violations.append(
                f"  [%] Invalid price: Bid={self.bid}, Ask={self.ask} (must be > 0)"
            )
        if self.bid > self.ask:
            violations.append(f"  [%] Inverted spread: Bid={self.bid} > Ask={self.ask}")
        if self.source not in settings.valid_sources:
            violations.append(
                f"  [!] Unknown source: '{self.source}' (must be one of {', '.join(settings.valid_sources)})"
            )

        # If there are any violations, log a critical error block and raise an exception
        if violations:
            error_block = (
                f"\n{'%' * 60}\n"
                f"MODEL CONSTRUCTION FAILURE: {len(violations)} VIOLATION(S)\n"
                f"Symbol: {self.symbol}\n"
                f"\n".join(violations) + "\n"
                f"Context: Data rejected at schema level to prevent ML corruption.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"Tick({self.symbol}): {len(violations)} integrity violation(s). See logs."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the tick to a flat dictionary.

        Uses dataclasses.asdict() for field-by-field conversion.
        Compatible with pandas.DataFrame.from_records() and
        pyarrow for Parquet serialization.

        Returns:
            dict[str, Any]: Dictionary with all Tick fields as keys.
        """
        return asdict(self)

    def __repr__(self) -> str:
        """
        The `__repr__` function returns a string representation of an object with specific attributes
        formatted in a certain way.

        Returns:
        The `__repr__` method returns a formatted string representation of an object of class `Tick`.
        It includes the symbol, bid price, ask price, source, and timestamp of the tick data. The timestamp
        is formatted to display hours, minutes, seconds, and milliseconds.
        """
        return (
            f"Tick({self.symbol} "
            f"bid={self.bid:.5f} ask={self.ask:.5f} "
            f"src={self.source} "
            f"ts={self.timestamp.strftime('%H:%M:%S.%f')[:-3]})"
        )

    @property
    def mid_price(self) -> float:
        """
        Calculate the mid-price from bid and ask.

        The mid-price (midpoint) is used as the primary price reference
        for signal generation and feature engineering. It represents the
        theoretical fair value between buying and selling pressure.

        Returns:
            float: (bid + ask) / 2
        """
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class Bar:
    """
    Aggregated OHLCV candle data, typically at M1 (1-minute) resolution.

    Used as the primary input for Technical Analysis indicators and
    Random Forest feature extraction. Each bar represents a fixed
    time window of aggregated tick activity.

    Attributes:
        timestamp: Start time of the candle window (UTC).
        symbol: Pure currency pair name (e.g., "EUR_USD").
        open: First traded price in the window.
        high: Highest traded price in the window.
        low: Lowest traded price in the window.
        close: Last traded price in the window.
        volume: Total tick count or traded volume in the window.
        is_complete: Whether the candle window has fully elapsed.
            Incomplete bars are excluded from training data.

    Example:
        >>> bar = Bar(
        ...     timestamp=datetime.utcnow(),
        ...     symbol="EUR_USD",
        ...     open=1.0850,
        ...     high=1.0865,
        ...     low=1.0848,
        ...     close=1.0860,
        ...     volume=342,
        ... )
        >>> bar.high - bar.low
        0.0017
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_complete: bool = True
    timeframe: Timeframe = Timeframe.M1

    def __post_init__(self) -> None:
        """
        Validate OHLC logic to ensure physical price integrity for candles.
        """
        # Logic constraints
        is_low_too_high: bool = self.low > self.high
        is_open_out_of_bounds: bool = self.open > self.high or self.open < self.low
        is_close_out_of_bounds: bool = self.close > self.high or self.close < self.low
        is_negative_volume: bool = self.volume < 0

        # Reject non-UTC timezones before stripping. Same rule as Tick —
        # a non-UTC aware datetime silently loses its offset information.
        if self.timestamp.tzinfo is not None:
            if self.timestamp.utcoffset() != timedelta(0):
                raise ValueError(
                    f"[!] Non-UTC timezone rejected for Bar({self.symbol}): "
                    f"tzinfo={self.timestamp.tzinfo!r}. Convert to UTC before construction."
                )
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=None))

        # If any integrity violation is detected, raise an exception
        if (
            is_low_too_high
            or is_open_out_of_bounds
            or is_close_out_of_bounds
            or is_negative_volume
        ):
            raise ValueError(
                f"[%] Bar integrity failure for {self.symbol} at {self.timestamp}: "
                f"O={self.open} H={self.high} L={self.low} C={self.close} V={self.volume}"
            )
        # Engine -- owns the logging and the decision to crash on invalid data

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the bar to a flat dictionary.

        Uses dataclasses.asdict() for field-by-field conversion.
        Compatible with pandas.DataFrame.from_records() and
        pyarrow for Parquet serialization.

        Returns:
            dict[str, Any]: Dictionary with all Bar fields as keys.
        """
        return asdict(self)

    def __repr__(self) -> str:
        """
        The function __repr__ returns a string representation of a Bar object with symbol, open price,
        high, low, close, volume, and completeness status.

        Returns:
        The code snippet implements the `__repr__` special method for a class. It returns a
        formatted string representation of an instance of the class. The returned string includes
        information about the symbol, open price, high price, low price, close price, volume, and
        whether the bar is complete or not.
        """
        return (
            f"Bar({self.symbol} "
            f"O={self.open:.5f} H={self.high:.5f} "
            f"L={self.low:.5f} C={self.close:.5f} "
            f"V={self.volume} complete={self.is_complete})"
        )


class DataBuffer:
    """
    Thread-safe in-memory buffer for high-frequency ticks.

    Accumulates ticks until a configured flush threshold is reached,
    then returns the batch for storage (disk or cloud). Designed to
    reduce I/O overhead by batching writes rather than writing each
    tick individually.

    Thread Safety:
        All public methods acquire an internal lock before modifying
        state. Safe for concurrent access from multiple stream sources.

    Attributes:
        _flush_size: Number of ticks that triggers a flush.
        _data: Internal list of accumulated ticks.
        _lock: Threading lock for concurrent access safety.

    Example:
        >>> buffer = DataBuffer(flush_size=3)
        >>> buffer.add(tick1)  # returns None
        >>> buffer.add(tick2)  # returns None
        >>> batch = buffer.add(tick3)  # returns [tick1, tick2, tick3]
        >>> len(buffer)
        0
    """

    def __init__(self, flush_size: int) -> None:
        """
        Initialize the buffer with a flush threshold.

        Args:
            flush_size: Number of ticks to accumulate before add()
                returns a batch. Must be a positive integer > 0.

        Raises:
            ValueError: If flush_size <= 0.
        """
        if flush_size <= 0:
            error_block = (
                f"\n{'%' * 60}\n"
                f"BUFFER CONFIGURATION ERROR: INVALID FLUSH SIZE\n"
                f"Value Received: {flush_size}\n"
                f"Constraint: flush_size must be a positive integer > 0.\n"
                f"Context: DataBuffer cannot function with zero or negative thresholds.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(f"[%] Invalid flush_size: {flush_size}")
        self._flush_size: int = flush_size
        self._data: list[Tick] = []
        self._lock: threading.Lock = threading.Lock()

    def add(self, tick: Tick) -> list[Tick]:
        """
        Add a tick to the buffer.

        Appends the tick to the internal list. If the list has reached
        or exceeded the flush threshold, the entire batch is returned
        and the buffer is cleared. Otherwise returns an empty list.
        Includes a safety cap to prevent memory exhaustion.

        Args:
            tick: A validated Tick instance to buffer.

        Returns:
            list[Tick]: The accumulated batch if flush threshold is
                reached, otherwise an empty list. Never returns None —
                callers can always do ``if batch:`` without an
                ``is not None`` guard.

        Thread Safety:
            Acquires self._lock for the duration of the append and
            potential flush operation.
        """
        with self._lock:
            # --- GUARDRAIL: Memory Safety Cap ---
            # If the buffer is 10x the flush size, something is wrong with the storage layer
            if len(self._data) >= (self._flush_size * 10):
                error_block = (
                    f"\n{'%' * 60}\n"
                    f"BUFFER OVERFLOW: STORAGE LAYER HANG DETECTED\n"
                    f"Current Size: {len(self._data)} | Max Cap: {self._flush_size * 10}\n"
                    f"Context: Buffer exceeded safety limits. Crashing to prevent RAM exhaustion.\n"
                    f"{'%' * 60}"
                )
                logger.critical(error_block)
                # We use RuntimeError to signal a hardware/resource exhaustion event
                raise RuntimeError(
                    f"[%] Buffer overflow: Storage layer failed to consume data. "
                    f"Size={len(self._data)}, Cap={self._flush_size * 10}"
                )

            self._data.append(tick)
            if len(self._data) >= self._flush_size:
                batch: list[Tick] = self._data.copy()
                self._data.clear()
                return batch
        return []

    def __len__(self) -> int:
        """
        Return the current number of buffered ticks.

        Returns:
            int: Count of ticks currently in the buffer.

        Thread Safety:
            Acquires self._lock before reading the list length.
        """
        with self._lock:
            return len(self._data)

    def flush(self) -> list[Tick]:
        """
        Force-flush the buffer regardless of size.
        Used during graceful shutdown or end-of-session drain.
        Returns the batch if any ticks exist, otherwise an empty list.
        Never returns None — callers can always do ``if batch:`` without
        an ``is not None`` guard.
        """
        with self._lock:
            if not self._data:
                return []
            batch = self._data.copy()
            self._data.clear()
            return batch
