"""
data/historian.py — The Data Archaeologist.

Single point of contact for fetching and persisting historical M1 bar data
from the Twelve Data REST API. Implements gap-detection, date-windowed
chunking, and inter-request rate-limiting to stay within free-tier API limits.

Operates in two modes:
    - Full backfill:  No existing bar data found → fetches BACKFILL_YEARS of
                    history walking forward from (now - years) to now.
    - Gap backfill:   Existing bars found → fetches only the window missing
                    since the last stored bar timestamp.

The Historian is the upstream supplier for the processed Parquet store.
Its output (validated Bar objects) is consumed by the ML training pipeline
via Storage.get_bars(). It does not touch the raw tick store.

Design Document: docs/data/Historian/Historian.md
"""

from __future__ import annotations

import asyncio
import logging
import dukascopy_python

from core.config import get_settings
from core.exceptions import HistorianError
from datetime import datetime, timedelta, timezone
from data.storage import Storage, StorageError, get_storage
from dukascopy_python import INTERVAL_MIN_1, OFFER_SIDE_BID
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_USD_CAD,
    INSTRUMENT_FX_MAJORS_USD_CHF,
    INSTRUMENT_FX_MAJORS_NZD_USD,
)

from ml_engine.model import Bar, Timeframe

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

# Calendar-day window covered by a single API request.
# Forex pairs trade ~24 hours/day on weekdays (no stock-market lunch break).
# 3 calendar days × 24 h × 60 min = 4320 M1 bars — safely within the 5000-bar cap.
# A 7-day window (≈ 5 trading days × 24 h × 60 min = 7200 bars) would exceed
# _BARS_PER_REQUEST and be silently truncated by Twelve Data, causing gaps.
_CHUNK_DAYS: int = 3

# Minimum volume floor applied to API bars before Bar construction.
# Forex/OTC bars from Twelve Data sometimes report volume=0 for synthetic
# instruments. A floor of 1.0 prevents zero-volume bars from being silently
# accepted or rejected depending on the Bar validation version in use.
_MIN_VOLUME: float = 1.0
# ISO datetime format for logging and API consistency
_ISO_DATETIME_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
_SECONDS_IN_DAY: int = 86400

# ── Dukascopy Configuration ────────────────────────────────────────────────────
# Dukascopy provides tick volume (real market activity) - essential for ATR
# Twelve Data will be removed entirely due to volume=0 corrupting features

# ── Dukascopy Configuration ────────────────────────────────────────────────────
# Check if Dukascopy is available (truthy check using sys.modules)
DUKASCOPY_AVAILABLE = True
# Map our symbol format to Dukascopy instruments
_DUKASCOPY_INSTRUMENTS = {
    "EUR_USD": INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBP_USD": INSTRUMENT_FX_MAJORS_GBP_USD,
    "USD_JPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "AUD_USD": INSTRUMENT_FX_MAJORS_AUD_USD,
    "USD_CAD": INSTRUMENT_FX_MAJORS_USD_CAD,
    "USD_CHF": INSTRUMENT_FX_MAJORS_USD_CHF,
    "NZD_USD": INSTRUMENT_FX_MAJORS_NZD_USD,
}

# Maximum number of consecutive chunk-save failures before aborting the backfill.
# A single StorageError may be transient (brief lock, permission flicker) and is
# tolerated. Three in a row indicates a systemic condition (disk full, mount lost)
# that will not self-resolve — continuing would burn API quota for data that
# cannot land on disk.
_MAX_CONSECUTIVE_STORAGE_FAILURES: int = 3


# ── Historian ─────────────────────────────────────────────────────────────────
class Historian:
    """
    Historical M1 bar backfill engine using the Twelve Data REST API.

    Fetches OHLCV bars for one or more currency pairs and persists them to
    the processed Parquet store via :class:`~data.storage.Storage`. Implements
    the gap-detection pattern from the Storage design document:

        1. Query Storage for the last known bar timestamp.
        2. If none exists, start from (now - BACKFILL_YEARS).
        3. Walk forward in 7-day chunks, fetching and saving each chunk.
        4. Stop when the current UTC time is reached.

    Rate Limiting:
        TwelveData free tier allows 8 requests/minute (800/day). An 8-second
        inter-request delay is enforced via :meth:`_enforce_rate_limit` before
        every API call. A 2-year backfill of a single pair (~100 requests)
        completes in roughly 15 minutes.

    Chunking:
        Each API call covers a 7-calendar-day window with up to 5000 M1 bars.
        Bars are persisted to the processed Parquet store immediately after
        each chunk is received — the system is crash-safe at chunk boundaries.

    Attributes:
        _settings:          Validated frozen :class:`~core.config.Config`.
        _storage:           :class:`~data.storage.Storage` instance for I/O.
        _last_request_time: Monotonic clock time of the last API call.
                            Used by :meth:`_enforce_rate_limit`.

    Example:
        >>> historian = Historian()
        >>> total = await historian.backfill("EUR_USD")
        >>> print(f"Backfilled {total} bars for EUR_USD.")

        >>> results = await historian.backfill_all()
        >>> for symbol, count in results.items():
        ...     print(f"{symbol}: {count} bars committed.")
    """

    def __init__(self) -> None:
        """
        Initialise the Historian.

        Acquires validated configuration via :func:`~core.config.get_settings`
        and instantiates the Storage custodian. Fails immediately if
        configuration is invalid or Storage cannot provision its directories.

        Raises:
            SystemExit:  If :func:`~core.config.get_settings` detects an
                invalid environment (delegated to config's fail-fast logic).
            StorageError: If Storage cannot provision its data directories.
        """
        self._settings = get_settings()
        self._storage: Storage = get_storage()

    # ── Dukascopy Fetch ─────────────────────────────────────────────────────

    def _get_dukascopy_instrument(self, symbol: str):
        """Get Dukascopy instrument for symbol, log error if unsupported."""
        instrument = _DUKASCOPY_INSTRUMENTS.get(symbol)
        if instrument is None:
            raise HistorianError(
                f"Unsupported symbol for Dukascopy backfill: {symbol}. "
                f"Supported symbols: {list(_DUKASCOPY_INSTRUMENTS.keys())}",
                symbol=symbol,
                source="DUKASCOPY",
            )
        return instrument

    def _df_row_to_bar(self, symbol: str, idx, row) -> Bar:
        """Convert a DataFrame row to a Bar object. Returns None if invalid."""
        try:
            timestamp = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            # Get volume, ensure minimum value (safety net)
            volume = max(float(row.get("volume", _MIN_VOLUME)), _MIN_VOLUME)

            return Bar(
                timestamp=timestamp,
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=volume,
                is_complete=True,
                timeframe=Timeframe.M1,
            )
        except (KeyError, ValueError) as e:
            raise HistorianError(
                f"Failed to parse Dukascopy bar for {symbol}: {e}. Row data: {row.to_dict() if hasattr(row, 'to_dict') else row}",
                symbol=symbol,
                source="DUKASCOPY",
            ) from e

    async def _fetch_from_dukascopy(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[Bar]:
        """
        Fetch M1 bars from Dukascopy with REAL tick volume.

        Dukascopy provides tick volume (number of price changes per minute),
        which is essential for ATR and volatility calculations.

        Args:
            symbol: Pure pair (e.g., "EUR_USD")
            start_dt: UTC start (inclusive)
            end_dt: UTC end (inclusive)

        Returns:
            List of Bar objects with real volume data (>0)

        Raises:
            HistorianError: If instrument not found, fetch fails, or data parsing fails
        """
        # Step 1: Get instrument (raises if unsupported)
        instrument = self._get_dukascopy_instrument(symbol)

        logger.debug(
            {
                "event": "DUKASCOPY_FETCH_START",
                "symbol": symbol,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            }
        )
        # Step 2: Fetch data in thread pool (raises on failure)
        loop = asyncio.get_event_loop()

        def _fetch():
            try:
                df = dukascopy_python.fetch(
                    instrument=instrument,
                    interval=INTERVAL_MIN_1,
                    offer_side=OFFER_SIDE_BID,
                    start=start_dt,
                    end=end_dt,
                )
                return df if not df.empty else None
            except Exception as e:
                raise HistorianError(
                    f"Dukascopy fetch failed for {symbol}: {e}",
                    symbol=symbol,
                    source="DUKASCOPY",
                ) from e

        df = await loop.run_in_executor(None, _fetch)
        # Step 3: Handle no data (weekend/holiday - not an error)
        if df is None or df.empty:
            logger.debug(
                {
                    "event": "DUKASCOPY_NO_DATA",
                    "symbol": symbol,
                    "range": f"{start_dt.date()} to {end_dt.date()}",
                    "message": "No data returned - possibly weekend, holiday, or illiquid period",
                }
            )
            return []
        # Step 4: Convert each row to Bar (raises on parse error)
        bars = []
        skipped = 0

        for idx, row in df.iterrows():
            bar = self._df_row_to_bar(symbol, idx, row)
            if bar:
                bars.append(bar)
            else:
                skipped += 1

        if skipped:
            logger.warning(
                {
                    "event": "DUKASCOPY_BARS_SKIPPED",
                    "symbol": symbol,
                    "skipped": skipped,
                    "total": len(df),
                }
            )

        logger.debug(
            {
                "event": "DUKASCOPY_FETCH_COMPLETE",
                "symbol": symbol,
                "bars": len(bars),
            }
        )

        return bars

    async def fetch_range(
        self, symbol: str, start_dt: datetime, end_dt: datetime
    ) -> int:
        """
        Fetch and persist M1 bars for an arbitrary UTC date range.

        Public wrapper around the internal _fetch_and_save() loop so that
        callers like Pipeline._repair_internal_gaps() can fill specific
        gaps without reimplementing chunking, rate-limiting, and retries.

        Args:
            symbol:   Pure currency pair name (e.g., "EUR_USD").
            start_dt: UTC start of the window (inclusive).
            end_dt:   UTC end of the window (inclusive).

        Returns:
            int: Total bars committed to storage. 0 if the API returned
                no data for the window (weekend/holiday).
        """
        return await self._fetch_and_save(symbol, start_dt, end_dt)

    async def backfill(self, symbol: str) -> int:
        """
        Bring a single symbol's bar data up to date.

        Dukascopy provides tick volume (real market activity) essential for ATR.

        Args:
            symbol: Pure currency pair name (e.g., ``"EUR_USD"``).

        Returns:
            int: Total number of bars successfully committed to storage.

        Raises:
            HistorianError: If backfill fails for any reason.
        """
        logger.info(
            {
                "event": "HISTORIAN_BACKFILL_START",
                "symbol": symbol,
                "source": "DUKASCOPY",
            }
        )

        try:
            count = await self._backfill_from_dukascopy(symbol)

            logger.info(
                {
                    "event": "HISTORIAN_BACKFILL_COMPLETE",
                    "symbol": symbol,
                    "source": "DUKASCOPY",
                    "bars_committed": count,
                }
            )
            return count

        except Exception as e:
            logger.error(
                {
                    "event": "HISTORIAN_BACKFILL_FAILED",
                    "symbol": symbol,
                    "source": "DUKASCOPY",
                    "error": str(e),
                }
            )
            raise HistorianError(
                f"Dukascopy backfill failed for {symbol}: {e}",
                symbol=symbol,
                source="DUKASCOPY",
            ) from e

    # ── Source Data Backfill ───────────────────────────────────────────────────

    async def _backfill_from_dukascopy(self, symbol: str) -> int:
        """
        Backfill M1 bars using Dukascopy as the data source.

        Dukascopy provides tick volume (real market activity), essential for ATR.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")

        Returns:
            Number of bars committed
        """
        now_utc = datetime.now(timezone.utc)
        start_dt = self._determine_start(symbol, now_utc)

        if start_dt >= now_utc:
            logger.info(
                {
                    "event": "DUKASCOPY_UP_TO_DATE",
                    "symbol": symbol,
                }
            )
            return 0

        gap_days = (now_utc - start_dt).total_seconds() / _SECONDS_IN_DAY
        logger.info(
            {
                "event": "DUKASCOPY_BACKFILL_START",
                "symbol": symbol,
                "start_date": start_dt.strftime(_ISO_DATETIME_FORMAT),
                "end_date": now_utc.strftime(_ISO_DATETIME_FORMAT),
                "gap_days": round(gap_days, 1),
            }
        )

        total_saved = await self._fetch_and_save(symbol, start_dt, now_utc)

        logger.info(
            {
                "event": "DUKASCOPY_BACKFILL_COMPLETE",
                "symbol": symbol,
                "bars_committed": total_saved,
            }
        )

        return total_saved

    # ── Private: Gap Detection ────────────────────────────────────────────────

    def _determine_start(self, symbol: str, now_utc: datetime) -> datetime:
        """
        Determine the UTC start datetime for a symbol's backfill window.

        Returns the oldest timestamp we need to fetch.
            - If no data: now_utc - BACKFILL_YEARS
            - If data exists: last_bar + 1 minute
        """

        years: int = self._settings.backfill_years

        # max_rows=1 returns the newest row, so we must read all rows and take index[0].
        df = self._storage.get_bars(symbol, timeframe="M1", max_rows=1)

        if df is None or df.empty:
            # First run: backfill from BACKFILL_YEARS ago at midnight UTC.
            # Use year replacement instead of timedelta(days=365*N) to avoid
            # accumulating leap-year error (each ignored Feb 29 shifts the
            # start date forward by one day relative to the calendar year).
            try:
                start: datetime = now_utc.replace(
                    year=now_utc.year - years,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            except ValueError:
                # now_utc is Feb 29 and (now_utc.year - years) is not a leap year.
                # Shift to Feb 28 — one day earlier is preferable to an exception.
                start = now_utc.replace(
                    year=now_utc.year - years,
                    month=2,
                    day=28,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            logger.info(
                {
                    "event": "NO_EXISTING_DATA",
                    "symbol": symbol,
                    "years": years,
                }
            )
            return start

        # Data exists
        # Resume from last bar + 1 minute
        raw_ts = df.index[-1]
        last_dt: datetime = (
            raw_ts.to_pydatetime() if hasattr(raw_ts, "to_pydatetime") else raw_ts
        )
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        start = last_dt + timedelta(minutes=1)

        logger.info(
            {
                "event": "DATA_EXISTS",
                "symbol": symbol,
                "last_bar": last_dt.strftime(_ISO_DATETIME_FORMAT),
            }
        )
        return start

    # ── Private: Fetch & Persist Loop ─────────────────────────────────────────

    async def _fetch_and_save(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> int:
        """
        Walk forward through date range in chunks, fetching from Dukascopy and saving.

        Dukascopy has no rate limits, but chunking is preserved for:
            1. Memory efficiency (large ranges)
            2. Crash safety (saved at each chunk)
            3. Progress visibility

        Args:
            symbol: Currency pair (e.g., "EUR_USD")
            start_dt: UTC start (inclusive)
            end_dt: UTC end (inclusive)

        Returns:
            Total bars saved
        """
        total_saved = 0
        chunk_start = start_dt
        consecutive_failures = 0

        while chunk_start < end_dt:
            chunk_end = min(
                chunk_start + timedelta(days=_CHUNK_DAYS),
                end_dt,
            )

            bars = await self._fetch_from_dukascopy(symbol, chunk_start, chunk_end)

            if bars:
                saved = self._save_bars(symbol, bars)
                if saved > 0:
                    total_saved += saved
                    consecutive_failures = 0
                    logger.info(
                        {
                            "event": "DUKASCOPY_CHUNK_SAVED",
                            "symbol": symbol,
                            "chunk_start": chunk_start.date().isoformat(),
                            "chunk_end": chunk_end.date().isoformat(),
                            "bars_saved": saved,
                            "running_total": total_saved,
                        }
                    )
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_STORAGE_FAILURES:
                        raise HistorianError(
                            f"Storage failed {consecutive_failures} times for {symbol}",
                            symbol=symbol,
                        )
            else:
                # No bars in this chunk (weekend/holiday) - not an error
                logger.debug(
                    {
                        "event": "DUKASCOPY_NO_BARS_IN_CHUNK",
                        "symbol": symbol,
                        "chunk": f"{chunk_start.date()} → {chunk_end.date()}",
                    }
                )

            # Advance to next chunk (add 1 minute to avoid overlap)
            chunk_start = chunk_end + timedelta(minutes=1)

            # Small delay to be kind to Dukascopy servers (no official rate limit)
            await asyncio.sleep(0.5)

        return total_saved

    # ── Private: Persistence ──────────────────────────────────────────────────

    def _save_bars(self, symbol: str, bars: list[Bar]) -> int:
        """
        Persist a chunk of validated Bar objects to the processed Parquet store.

        Delegates to :meth:`~data.storage.Storage.save_bar_batch` which
        performs a single atomic read-deduplicate-write cycle for the entire
        chunk. This is dramatically more efficient than calling
        :meth:`~data.storage.Storage.save_bar` in a loop, which would
        open, read, and rewrite the Parquet file once per bar.

        A :class:`~data.storage.StorageError` from the batch write aborts
        the entire chunk and is logged as a warning so the caller can decide
        whether to retry or advance to the next chunk.

        Args:
            symbol: Pure pair name used for log context only. The bars
                    themselves carry the symbol for Storage.
            bars:   List of validated :class:`~ml_engine.model.Bar` objects
                    from a single chunk, all sharing the same symbol and
                    timeframe.

        Returns:
            int: Number of bars in ``bars`` if the batch committed successfully.
                    0 if the batch was empty or a StorageError occurred.
        """
        if not bars:
            return 0

        try:
            self._storage.save_bar_batch(bars)
            return len(bars)
        except StorageError as exc:
            logger.warning(
                {
                    "event": "STORAGE_SAVE_ERROR",
                    "symbol": symbol,
                    "bars_in_batch": len(bars),
                    "error": str(exc),
                }
            )
            return 0


# ── Initialization Gate ───────────────────────────────────────────────────────
# The Historian is NOT instantiated at import time. Call get_historian() from
# your application entry point (pipeline.py or main.py) to trigger construction.
# This keeps imports safe for testing, linting, and doc generation.
#
# Using a singleton ensures that _last_request_time is shared across all callers
# in the same process. Multiple independent Historian instances would not share
# rate-limit state and could fire back-to-back API requests under 8 seconds.
_historian: Historian | None = None


def get_historian() -> Historian:
    """
    Return the global validated Historian instance.

    Initialises on first call (lazy singleton). Subsequent calls return the
    cached instance without re-constructing Storage or re-reading config.

    Using a single shared instance guarantees that ``_last_request_time`` is
    never duplicated across callers — a second independent ``Historian()``
    would have its own clock and could fire consecutive requests with no gap,
    violating the Twelve Data rate limit.

    Returns:
        Historian: The validated, initialised historian ready for backfill.

    Raises:
        SystemExit: If :func:`~core.config.get_settings` or
            :class:`~data.storage.Storage` fail during first-call construction.

    Example:
        >>> historian = get_historian()
        >>> await historian.backfill_all()
    """
    global _historian
    if _historian is None:
        _historian = Historian()
    return _historian
