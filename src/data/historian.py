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

import sys
import time
import asyncio
import logging
import aiohttp

from core.config import get_settings
from core.exceptions import HistorianError
from datetime import datetime, timedelta, timezone
from data.storage import Storage, StorageError, get_storage
from data.forex_calendar import is_forex_closed, get_gap_classification
from ml_engine.model import Bar, Timeframe

logger = logging.getLogger(__name__)

# ── API Constants ─────────────────────────────────────────────────────────────

# Twelve Data REST endpoint for OHLCV time-series data
_API_BASE: str = "https://api.twelvedata.com"

# Maximum bars per API request (hard cap enforced by Twelve Data)
_BARS_PER_REQUEST: int = 5000

# Free-tier rate limit: 8 req/min → 1 req per 8 s (with 1-second safety margin)
_REQUEST_INTERVAL_S: float = 10.0

# HTTP request timeout in seconds (Twelve Data is generally fast, 30 s is generous)
_HTTP_TIMEOUT_S: float = 45.0

# Number of retry attempts for each chunk on transient HTTP failure
_MAX_RETRIES: int = 5

# Initial backoff delay in seconds before the first retry; doubles on each attempt
_RETRY_BACKOFF_S: float = 30.0

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

# Maximum number of consecutive chunk-save failures before aborting the backfill.
# A single StorageError may be transient (brief lock, permission flicker) and is
# tolerated. Three in a row indicates a systemic condition (disk full, mount lost)
# that will not self-resolve — continuing would burn API quota for data that
# cannot land on disk.
_MAX_CONSECUTIVE_STORAGE_FAILURES: int = 3

_SECONDS_IN_DAY = 86400
# Import for PyQuotex
try:
    from pyquotex.stable_api import Quotex

    QUOTEX_AVAILABLE = True
except ImportError:
    QUOTEX_AVAILABLE = False
    logger.warning(
        {
            "event": "HISTORIAN_QUOTEX_UNAVAILABLE",
            "message": "pyquotex not available - Quotex backfill disabled",
        }
    )


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

        # Monotonic timestamp of the last API request.
        # Initialised to 0.0 so the first request fires immediately.
        self._last_request_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

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

        Routes to the appropriate backfill source based on BACKFILL_SOURCE config.

        Args:
            symbol: Pure currency pair name (e.g., ``"EUR_USD"``).

        Returns:
            int: Total number of bars successfully committed to storage.

        Raises:
            HistorianError: If backfill fails for any reason.
        """
        source = self._settings.backfill_source

        logger.info(
            {
                "event": "HISTORIAN_BACKFILL_START",
                "symbol": symbol,
                "source": source,
            }
        )

        # Route to appropriate backfill source
        if source == "QUOTEX":
            if not QUOTEX_AVAILABLE:
                raise HistorianError(
                    f"QUOTEX backfill requested for {symbol} but pyquotex not available",
                    symbol=symbol,
                    source=source,
                )
            count = await self._backfill_from_quotex(symbol)
        elif source == "TWELVE":
            count = await self._backfill_from_twelvedata(symbol)
        else:
            raise HistorianError(
                f"Unknown BACKFILL_SOURCE: {source}",
                symbol=symbol,
                source=source,
            )

        logger.info(
            {
                "event": "HISTORIAN_BACKFILL_COMPLETE",
                "symbol": symbol,
                "source": source,
                "bars_committed": count,
            }
        )
        return count

    async def backfill_all(self) -> dict[str, int]:
        """
        Run a gap backfill sequentially for every configured backfill pair.

        Reads the ``BACKFILL_PAIRS`` setting from config and calls
        :meth:`backfill` for each symbol in order. Pairs are processed
        sequentially (not concurrently) so the shared inter-request rate
        limiter functions correctly across all symbols.

        A :class:`HistorianError` for one symbol is caught and logged as an
        error — the remaining symbols continue to be processed.

        Returns:
            dict[str, int]: Mapping of ``{symbol: bars_committed}``.
                            A value of 0 indicates either up-to-date data
                            or a failed backfill (check logs for distinction).

        Example:
            >>> results = await historian.backfill_all()
            >>> for symbol, count in results.items():
            ...     print(f"{symbol}: {count} bars")
        """
        pairs: list[str] = self._settings.backfill_pairs
        results: dict[str, int] = {}

        logger.info(
            {
                "event": "HISTORIAN_BACKFILL_ALL_START",
                "pair_count": len(pairs),
                "pairs": pairs,
            }
        )

        for symbol in pairs:
            count: int = await self.backfill(symbol)
            results[symbol] = count

        logger.info(
            {
                "event": "HISTORIAN_BACKFILL_ALL_COMPLETE",
                "results": results,
            }
        )
        return results

    # ── Source Data Backfill ───────────────────────────────────────────────────

    async def _backfill_from_twelvedata(self, symbol: str) -> int:
        """
        Bring a single symbol's bar data up to date.

        Determines the last known bar timestamp via Storage, then fetches
        all missing M1 bars from the Twelve Data REST API in chronological
        7-day chunks. Bars are saved to the processed Parquet store as each
        chunk arrives — no in-memory accumulation of the full history.

        If bar data is already current (last bar timestamp is within the
        current minute), this method returns 0 without making any API calls.

        Args:
            symbol: Pure currency pair name (e.g., ``"EUR_USD"``).

        Returns:
            int: Total number of bars successfully committed to storage.
                    0 if the data was already up to date.

        Raises:
            HistorianError: If a chunk fetch fails after all retry attempts
                and the backfill cannot be completed.

        Example:
            >>> count = await historian.backfill("EUR_USD")
            >>> print(f"Committed {count} new bars.")
        """
        now_utc: datetime = datetime.now(timezone.utc)
        start_dt: datetime = self._determine_start(symbol, now_utc)

        if start_dt >= now_utc:
            logger.info(
                {
                    "event": "TWELVE_DATA_UP_TO_DATE",
                    "symbol": symbol,
                }
            )
            return 0

        gap_days: float = (now_utc - start_dt).total_seconds() / _SECONDS_IN_DAY
        logger.info(
            {
                "event": "TWELVE_DATA_BACKFILL_START",
                "symbol": symbol,
                "start_date": start_dt.isoformat(),
                "end_date": now_utc.isoformat(),
                "gap_days": round(gap_days, 1),
            }
        )

        total_saved: int = await self._fetch_and_save(symbol, start_dt, now_utc)

        logger.info(
            {
                "event": "TWELVE_DATA_BACKFILL_COMPLETE",
                "symbol": symbol,
                "bars_committed": total_saved,
            }
        )
        return total_saved

    async def _backfill_from_quotex(self, symbol: str) -> int:
        """
        Backfill historical OTC data directly from Quotex.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")

        Returns:
            int: Number of bars committed
        """
        otc_symbol = self._settings.quotex_symbols[symbol]

        # ── Gap Detection (same as Twelve Data) ──────────────────────────────
        now_utc: datetime = datetime.now(timezone.utc)
        start_dt: datetime = self._determine_start(symbol, now_utc)

        if start_dt >= now_utc:
            logger.info({"event": "QUOTEX_UP_TO_DATE", "symbol": symbol})
            return 0

        gap_days: float = (now_utc - start_dt).total_seconds() / 86400

        # Calculate only the missing gap (not full backfill years)
        start_ts: int = int(start_dt.timestamp())
        end_ts: int = int(now_utc.timestamp())
        duration_seconds: int = end_ts - start_ts
        timeframe_sec = 60  # M1 only

        logger.info({"event": "QUOTEX_BACKFILL_START", "symbol": symbol, "otc_symbol": otc_symbol, "gap_days": round(gap_days, 1)})

        client = Quotex(
            email=self._settings.quotex_email,
            password=self._settings.quotex_password,
            lang="en",
        )

        client.debug_ws_enable = False
        connected, _ = await client.connect()

        if not connected:
            raise HistorianError(
                f"Failed to connect to Quotex for {otc_symbol}",
            )

        try:
            # Get asset info (optional, but good for validation)
            _, asset_data = await client.get_available_asset(
                otc_symbol, force_open=True
            )
            if not asset_data[2]:
                logger.warning(
                    {
                        "event": "QUOTEX_ASSET_CLOSED",
                        "symbol": symbol,
                        "otc_symbol": otc_symbol,
                        "message": "Asset may be closed - attempting fetch anyway",
                    }
                )
            logger.info(
                {
                    "event": "QUOTEX_DEEP_FETCH_START",
                    "symbol": symbol,
                    "otc_symbol": otc_symbol,
                    "duration_seconds": duration_seconds,
                    "estimated_candles": duration_seconds // timeframe_sec,
                }
            )
            start_time = time.time()
            total_bars_committed = 0
            last_reported = [-1]  # mutable container so inner function can write to it

            def on_progress(fetched_secs, total_secs, count):
                elapsed = time.time() - start_time
                percent = (fetched_secs / total_secs) * 100 if total_secs > 0 else 0
                milestone = int(percent) // 10 * 10
                if milestone > last_reported[0]:
                    last_reported[0] = milestone
                    logger.info(
                        {
                            "event": "QUOTEX_FETCH_PROGRESS",
                            "percent": round(percent, 1),
                            "candles": count,
                            "elapsed_seconds": round(elapsed, 1),
                        }
                    )

            async def on_chunk(raw_candles: list) -> None:
                nonlocal total_bars_committed
                bars = []
                skipped = 0
                for c in raw_candles:
                    op = c.get("open")
                    hi = c.get("high")
                    lo = c.get("low")
                    cl = c.get("close")
                    if None in (op, hi, lo, cl):
                        skipped += 1
                        continue

                    ts = datetime.fromtimestamp(c["time"], tz=timezone.utc)

                    if ts < start_dt:
                        skipped += 1
                        continue

                    bars.append(
                        Bar(
                            timestamp=ts,
                            symbol=symbol,
                            open=float(op),
                            high=float(hi),
                            low=float(lo),
                            close=float(cl),
                            volume=float(c.get("ticks", 1.0)),
                            is_complete=True,
                            timeframe=Timeframe.M1,
                        )
                    )
                if skipped:
                    logger.warning(
                        {
                            "event": "QUOTEX_CANDLES_SKIPPED",
                            "symbol": symbol,
                            "skipped": skipped,
                            "chunk_size": len(raw_candles),
                        }
                    )
                if bars:
                    await asyncio.sleep(0)
                    self._storage.save_bar_batch(bars)
                    total_bars_committed += len(bars)
                    logger.info(
                        {
                            "event": "QUOTEX_CHUNK_SAVED",
                            "symbol": symbol,
                            "chunk_bars": len(bars),
                            "total_committed": total_bars_committed,
                        }
                    )

            # Use the patched get_candles_deep with chunk_callback for
            # incremental writes — survives disconnects mid-backfill.
            await client.get_candles_deep(
                otc_symbol,
                duration_seconds,
                timeframe_sec,
                progress_callback=on_progress,
                chunk_callback=on_chunk,
                chunk_size=10000,
            )

            fetch_time = time.time() - start_time

            logger.info(
                {
                    "event": "QUOTEX_DEEP_FETCH_COMPLETE",
                    "symbol": symbol,
                    "bars_committed": total_bars_committed,
                    "fetch_time_seconds": round(fetch_time, 1),
                    "gap_days": round(gap_days, 1),
                }
            )
            return total_bars_committed

        except Exception as e:
            raise HistorianError(
                f"Quotex backfill failed for {symbol}: {e}",
                symbol=symbol,
                error_type=type(e).__name__,
            ) from e
        finally:
            await client.close()

    # ── Private: Gap Detection ────────────────────────────────────────────────

    def _determine_start(self, symbol: str, now_utc: datetime) -> datetime:
        """
        Determine the UTC start datetime for a symbol's backfill window.

        Behavior depends on backfill source:
        - TWELVE (forward fetch): Start from oldest needed time, walk forward.
        - QUOTEX (backward fetch): Start from now, walk backward.

        For forward fetch (TWELVE):
            Returns the oldest timestamp we need to fetch.
            - If no data: now_utc - BACKFILL_YEARS
            - If data exists: last_bar + 1 minute

        For backward fetch (QUOTEX):
            Returns the oldest timestamp we have (or target oldest if no data).
            - If no data: now_utc - BACKFILL_YEARS
            - If data exists: oldest_bar - 1 minute (to get earlier data)
        """

        years: int = self._settings.backfill_years
        source = self._settings.backfill_source
        # TWELVE needs only the newest bar (max_rows=1) to find the forward resume point.
        # QUOTEX needs the oldest bar to find how far back the backward fetch already reached;
        # max_rows=1 returns the newest row, so we must read all rows and take index[0].
        if source == "QUOTEX":
            df = self._storage.get_bars(symbol, timeframe="M1")
        else:
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
            logger.info({"event": "NO_EXISTING_DATA", "symbol": symbol, "years": years, "source": source})
            return start

        # Data exists - behavior depends on source
        if source == "TWELVE":
            # Forward fetch: start from last bar + 1 minute
            # Resume: one minute after the last stored bar to avoid re-fetching it
            raw_ts = df.index[-1]  # last row after sort-by-timestamp
            last_dt: datetime = (
                raw_ts.to_pydatetime() if hasattr(raw_ts, "to_pydatetime") else raw_ts
            )
            # Ensure timezone-aware for arithmetic consistency
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)

            start = last_dt + timedelta(minutes=1)

            logger.info({"event": "DATA_EXISTING_FORWARD", "symbol": symbol, "last_bar": last_dt.isoformat()})
            return start
        # QUOTEX - backward fetch
        else:
            # Get oldest bar to continue fetching earlier data
            raw_ts = df.index[0]  # oldest bar
            oldest_dt: datetime = (
                raw_ts.to_pydatetime() if hasattr(raw_ts, "to_pydatetime") else raw_ts
            )
            if oldest_dt.tzinfo is None:
                oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)

            # Calculate target oldest time (BACKFILL_YEARS ago)
            try:
                target_oldest: datetime = now_utc.replace(
                    year=now_utc.year - years,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            except ValueError:
                # Feb 29 edge case
                target_oldest = now_utc.replace(
                    year=now_utc.year - years,
                    month=2,
                    day=28,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )

            # If oldest bar is older than target, we're done
            if oldest_dt <= target_oldest:
                # Already have data back to target, return now (no backfill needed)
                logger.info({"event": "QUOTEX_BACKFILL_ALREADY_COMPLETE", "symbol": symbol, "oldest_bar": oldest_dt.isoformat()})
                return now_utc  # Returns >= now_utc so backfill skips

            # Start from oldest bar - 1 minute to get earlier data
            start = oldest_dt - timedelta(minutes=1)
            logger.info({"event": "DATA_EXISTING_BACKWARD", "symbol": symbol, "oldest_bar": oldest_dt.isoformat()})
            return start

    # ── Private: Fetch & Persist Loop ─────────────────────────────────────────

    async def _fetch_and_save(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> int:
        """
        Walk forward through the date range in chunks, fetching and saving bars.

        Iterates from ``start_dt`` to ``end_dt`` in :data:`_CHUNK_DAYS` windows,
        issuing one API call per chunk. The inter-request rate limit is enforced
        inside :meth:`_fetch_chunk`. Bars from each chunk are saved to Storage
        immediately after receipt — the loop is crash-safe at chunk boundaries.

        Args:
            symbol:   Pure currency pair name (e.g., ``"EUR_USD"``).
            start_dt: UTC start of the backfill window (inclusive).
            end_dt:   UTC end of the backfill window (inclusive).

        Returns:
            int: Total number of bars committed across all chunks.

        Raises:
            HistorianError: Propagated from :meth:`_fetch_chunk` if a chunk
                            fails after all retries, or raised directly if
                            :data:`_MAX_CONSECUTIVE_STORAGE_FAILURES` consecutive
                            chunks fail to persist — indicating a systemic storage
                            condition that would otherwise burn API quota for data
                            that cannot land on disk.
        """
        total_saved: int = 0
        chunk_start: datetime = start_dt
        consecutive_failures: int = 0

        # Twelve Data uses slash-separated symbols (e.g., "EUR/USD")
        api_symbol: str = symbol.replace("_", "/")

        async with aiohttp.ClientSession() as session:
            while chunk_start < end_dt:
                chunk_end: datetime = min(
                    chunk_start + timedelta(days=_CHUNK_DAYS),
                    end_dt,
                )

                bars: list[Bar] = await self._fetch_chunk(
                    session=session,
                    api_symbol=api_symbol,
                    symbol=symbol,
                    start_dt=chunk_start,
                    end_dt=chunk_end,
                )

                if bars:
                    # ── Gap Validation ────────────────────────────────────
                    # Check if the API returned fewer bars than expected.
                    # This catches silent truncation or partial responses.
                    expected_bars = self._count_expected_bars(chunk_start, chunk_end)
                    actual_bars = len(bars)
                    if expected_bars > 0 and actual_bars < expected_bars * 0.8:
                        gap_pct = (1 - actual_bars / expected_bars) * 100
                        logger.warning(
                            {
                                "event": "CHUNK_BAR_DEFICIT",
                                "symbol": symbol,
                                "chunk_start": chunk_start.date().isoformat(),
                                "chunk_end": chunk_end.date().isoformat(),
                                "expected_bars": expected_bars,
                                "actual_bars": actual_bars,
                                "deficit_pct": round(gap_pct, 1),
                                "message": (
                                    f"API returned {actual_bars}/{expected_bars} expected bars "
                                    f"({gap_pct:.1f}% deficit). Possible data loss or API truncation."
                                ),
                            }
                        )

                    saved: int = self._save_bars(symbol, bars)

                    if saved > 0:
                        total_saved += saved
                        consecutive_failures = 0  # reset: this chunk committed cleanly
                        logger.info(
                            {
                                "event": "DATA_CHUNK_SAVED",
                                "symbol": symbol,
                                "chunk_start": chunk_start.date().isoformat(),
                                "chunk_end": chunk_end.date().isoformat(),
                                "bars_saved": saved,
                                "running_total": total_saved,
                            }
                        )
                    else:
                        # _save_bars returned 0 → StorageError was caught inside.
                        # Increment the consecutive-failure guard.
                        consecutive_failures += 1
                        logger.warning(
                            {
                                "event": "DATA_CHUNK_STORAGE_FAILURE",
                                "symbol": symbol,
                                "consecutive_failures": consecutive_failures,
                                "max_failures": _MAX_CONSECUTIVE_STORAGE_FAILURES,
                            }
                        )

                        if consecutive_failures >= _MAX_CONSECUTIVE_STORAGE_FAILURES:
                            raise HistorianError(
                                f"Storage failed on {consecutive_failures} consecutive chunks "
                                f"for {symbol}. Aborting...",
                                symbol=symbol,
                            )
                else:
                    # Expected for weekends, public holidays, or illiquid windows.
                    # Empty chunks do not count against the failure threshold.
                    logger.debug(
                        {
                            "event": "NO_BARS",
                            "symbol": symbol,
                            "chunk_start": chunk_start.date().isoformat(),
                            "chunk_end": chunk_end.date().isoformat(),
                        }
                    )

                # Advance to the minute after the end of this chunk
                chunk_start = chunk_end + timedelta(minutes=1)

        return total_saved

    async def _backoff_wait(self, attempt: int, backoff: float) -> float:
        """Sleep backoff seconds when retries remain; return doubled backoff."""
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(backoff)
            return backoff * 2
        return backoff

    async def _fetch_chunk(
        self,
        session: aiohttp.ClientSession,
        api_symbol: str,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[Bar]:
        """
        Fetch a single chunk of M1 bars from the Twelve Data REST API.

        Issues one HTTP GET to ``/time_series`` with the given date window.
        Enforces the inter-request rate-limit delay via
        :meth:`_enforce_rate_limit` before each attempt. Retries up to
        :data:`_MAX_RETRIES` times with exponential backoff on transient
        HTTP errors or rate-limit responses (HTTP 429).

        Args:
            session:    Shared :class:`aiohttp.ClientSession` for connection
                        pooling across chunks.
            api_symbol: Twelve Data symbol format (e.g., ``"EUR/USD"``).
            symbol:     Pure pair name for Bar construction and log messages
                        (e.g., ``"EUR_USD"``).
            start_dt:   Chunk start datetime (UTC, inclusive).
            end_dt:     Chunk end datetime (UTC, inclusive).

        Returns:
            list[Bar]: Validated :class:`~ml_engine.model.Bar` objects parsed
                        from the API response, in chronological order.
                        Empty list if the API returns no bars for the window
                        (weekend / holiday — not an error condition).

        Raises:
            HistorianError: If all :data:`_MAX_RETRIES` attempts are exhausted
                            without a successful response.
        """
        params: dict[str, str | int] = {
            "symbol": api_symbol,
            "interval": "1min",
            "apikey": self._settings.twelvedata_api_key,
            "outputsize": _BARS_PER_REQUEST,
            "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "UTC",
        }

        backoff: float = _RETRY_BACKOFF_S

        for attempt in range(1, _MAX_RETRIES + 1):
            # Respect free-tier rate limit before every attempt
            await self._enforce_rate_limit()

            try:
                async with session.get(
                    f"{_API_BASE}/time_series",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S),
                ) as resp:
                    # Record request time immediately after the call completes
                    self._last_request_time = time.monotonic()

                    if resp.status == 429:
                        # Twelve Data rate-limited this request explicitly
                        logger.warning(
                            {
                                "event": "DATA_RATE_LIMIT",
                                "symbol": symbol,
                                "attempt": attempt,
                                "backoff_seconds": backoff,
                            }
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    if resp.status != 200:
                        # Non-retryable HTTP error or transient server issue
                        error_text: str = await resp.text()
                        logger.warning(
                            {
                                "event": "DATA_HTTP_ERROR",
                                "symbol": symbol,
                                "status": resp.status,
                                "attempt": attempt,
                                "error": error_text[:200],
                            }
                        )
                        backoff = await self._backoff_wait(attempt, backoff)
                        continue

                    try:
                        data: dict = await resp.json()
                    except Exception as exc:
                        if await self._handle_json_parse_error(
                            exc, symbol, start_dt, end_dt, attempt, backoff
                        ):
                            backoff *= 2
                            continue
                        return []

                    if "values" not in data:
                        self._handle_api_body_error(data, symbol, start_dt, end_dt)
                        return []

                    return self._parse_bars(symbol, data["values"])

            except aiohttp.ClientError as exc:
                logger.warning(
                    {
                        "event": "DATA_NETWORK_ERROR",
                        "symbol": symbol,
                        "attempt": attempt,
                        "error": str(exc),
                    }
                )
                backoff = await self._backoff_wait(attempt, backoff)

        # All retry attempts exhausted
        raise HistorianError(
            f"All {_MAX_RETRIES} fetch attempts failed for {symbol} "
            f"({start_dt.date()} → {end_dt.date()}).",
            symbol=symbol,
        )

    async def _handle_json_parse_error(
        self,
        exc: Exception,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        attempt: int,
        backoff: float,
    ) -> bool:
        """
        Handle a JSON parse failure for a chunk response.

        Logs the error and, if retries remain, sleeps for ``backoff`` seconds.
        Returns True when the caller should retry (i.e. ``attempt < _MAX_RETRIES``),
        Raises HistorianError when retries are exhausted.
        """
        critical_block = {
            "event": "API_RESPONSE_ERROR",
            "symbol": symbol,
            "chunk": f"{start_dt.date()} to {end_dt.date()}",
            "error": str(exc),
            "attempt": f"{attempt}/{_MAX_RETRIES}",
        }
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(backoff)
            return True
        logger.critical(critical_block)
        raise HistorianError(
            f"JSON parse failed for {symbol} ({start_dt.date()} → {end_dt.date()}) "
            f"after {_MAX_RETRIES} attempts: {exc}",
            symbol=symbol,
        )

    def _handle_api_body_error(
        self,
        data: dict,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """
        Log an appropriate message when the API response body lacks a ``"values"`` key.

        Distinguishes configuration errors (unknown symbol) from expected empty
        windows (market closed, holiday).
        """
        api_message: str = data.get("message", str(data))

        message_lower = api_message.lower()
        is_symbol_error: bool = (
            "symbol" in message_lower
            or "not found" in message_lower
            or "invalid" in message_lower
        )

        is_data_unavailable: bool = (
            "no data" in message_lower or "specified dates" in message_lower
        )

        if is_symbol_error:
            logger.error({"event": "API_SYMBOL_NOT_FOUND", "symbol": symbol, "message": api_message[:100]})
        elif is_data_unavailable:
            logger.debug({"event": "API_NO_DATA_FOR_WINDOW", "symbol": symbol})
        else:
            logger.warning({"event": "API_UNKNOWN_ERROR", "symbol": symbol, "message": api_message[:100]})

    async def _enforce_rate_limit(self) -> None:
        """
        Sleep until the minimum inter-request interval has elapsed.

        TwelveData free tier permits 8 requests per minute. A fixed 8-second
        gap between requests (1 req/8 s) stays safely within this limit even
        under minor clock drift or OS scheduling jitter.

        Calculates the time elapsed since :attr:`_last_request_time` and
        sleeps only for the remaining portion of the interval if needed.
        No sleep occurs when the full interval has already elapsed naturally
        (e.g., slow network, large JSON payload).
        """
        elapsed: float = time.monotonic() - self._last_request_time
        remaining: float = _REQUEST_INTERVAL_S - elapsed

        if remaining > 0:
            logger.debug({"event": "RATE_LIMIT_WAIT", "seconds": round(remaining, 1)})
            await asyncio.sleep(remaining)

    # ── Private: Parsing ──────────────────────────────────────────────────────

    def _parse_bars(self, symbol: str, values: list[dict]) -> list[Bar]:
        """
        Parse raw Twelve Data API response dictionaries into Bar objects.

        Twelve Data returns bars in **reverse-chronological order** (newest
        first). This method reverses the input list before construction so
        the returned batch is chronological, matching the expected Storage
        write order.

        Each bar is constructed from the OHLCV fields. Invalid bars — those
        with OHLC violations, missing fields, or non-numeric values — are
        skipped with a warning rather than aborting the entire chunk. A volume
        floor of :data:`_MIN_VOLUME` is applied because Twelve Data sometimes
        returns ``volume=0`` for OTC/synthetic instruments.

        Args:
            symbol: Pure currency pair name for Bar construction
                    (e.g., ``"EUR_USD"``).
            values: Raw bar dictionaries from the API ``"values"`` field.
                    Each dict contains: ``datetime``, ``open``, ``high``,
                    ``low``, ``close``, ``volume``.

        Returns:
            list[Bar]: Chronologically ordered, validated
                        :class:`~ml_engine.model.Bar` objects.
                        Malformed or physically invalid bars are excluded.
        """
        bars: list[Bar] = []
        skipped: int = 0

        # Reverse to produce chronological order (API sends newest-first)
        for v in reversed(values):
            try:
                ts: datetime = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S")
                # Apply volume floor: OTC/synthetic pairs may report 0 ticks
                raw_volume: float = float(v.get("volume", _MIN_VOLUME))
                volume: float = max(raw_volume, _MIN_VOLUME)

                bar = Bar(
                    timestamp=ts,
                    symbol=symbol,
                    open=float(v["open"]),
                    high=float(v["high"]),
                    low=float(v["low"]),
                    close=float(v["close"]),
                    volume=volume,
                    is_complete=True,
                    timeframe=Timeframe.M1,
                )
                bars.append(bar)

            except (KeyError, ValueError) as exc:
                # KeyError  → missing required field in API response
                # ValueError → Bar OHLC validation failure or float() parse error
                skipped += 1
                logger.debug(
                    {
                        "event": "DATA_PARSE_ERROR",
                        "symbol": symbol,
                        "error": str(exc),
                        "bar_data": v,
                    }
                )

        if skipped:
            logger.warning(
                {
                    "event": "TWELVE_DATA_BARS_SKIPPED",
                    "symbol": symbol,
                    "skipped": skipped,
                    "total": len(values),
                    "percentage": round(skipped / len(values) * 100, 1),
                }
            )

        return bars

    # ── Private: Gap Validation ──────────────────────────────────────────────

    def _count_expected_bars(self, start_dt: datetime, end_dt: datetime) -> int:
        """
        Count expected bars in a time window, excluding weekends and holidays.

        Uses the forex calendar to determine how many trading minutes exist
        in the given window. Used to validate that API responses aren't
        silently truncated.

        Args:
            start_dt: Chunk start (UTC, inclusive).
            end_dt:   Chunk end (UTC, inclusive).

        Returns:
            Expected number of bars.
        """
        count = 0
        current = start_dt
        while current <= end_dt:
            if not is_forex_closed(current):
                count += 1
            current += timedelta(minutes=1)
        return count

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
                    "event": "TWELVE_DATA_SAVE_ERROR",
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
