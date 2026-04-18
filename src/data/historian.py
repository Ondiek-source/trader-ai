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
import sys
import time
from datetime import datetime, timedelta, timezone

import aiohttp

from core.config import get_settings
from data.storage import Storage, StorageError, get_storage
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


# ── Custom Exception ──────────────────────────────────────────────────────────


class HistorianError(Exception):
    """
    Raised when the Historian cannot fulfil a backfill commitment.

    Distinct from ``aiohttp.ClientError`` (network layer) and
    :class:`~data.storage.StorageError` (persistence layer). Allows the
    pipeline orchestrator to catch historian-specific failures and decide
    whether to halt, skip the symbol, or retry the session later.

    Attributes:
        symbol: The currency pair being backfilled when the error occurred.
                Empty string if the failure is not symbol-specific.
    """

    def __init__(self, message: str, symbol: str = "") -> None:
        self.symbol = symbol
        super().__init__(message)


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

    async def backfill(self, symbol: str) -> int:
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
            info_block = (
                f"\n{'+' * 60}\n"
                f"DATA UP TO DATE\n"
                f"Symbol: {symbol}\n"
                f"Last bar timestamp: {start_dt.isoformat()}\n"
                f"Current UTC time: {now_utc.isoformat()}\n"
                f"No backfill needed."
                f"\n{'+' * 60}"
            )
            logger.info(info_block)
            return 0

        gap_days: float = (now_utc - start_dt).total_seconds() / 86_400
        info_block = (
            f"\n{'+' * 60}\n"
            f"STARTING BACKFILL\n"
            f"Symbol: {symbol}\n"
            f"Last bar timestamp: {start_dt.isoformat()}\n"
            f"Current UTC time: {now_utc.isoformat()}\n"
            f"Gap: {gap_days:.1f} days "
            f"({start_dt.date()} → {now_utc.date()})."
            f"\n{'+' * 60}"
        )
        logger.info(info_block)

        total_saved: int = await self._fetch_and_save(symbol, start_dt, now_utc)

        info_block = (
            f"\n{'+' * 60}\n"
            f"BACKFILL COMPLETE\n"
            f"Symbol: {symbol}\n"
            f"Total bars committed: {total_saved}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
        return total_saved

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

        info_block = (
            f"\n{'+' * 60}\n"
            f"STARTING BACKFILL FOR ALL PAIRS\n"
            f"{len(pairs)} Pair(s): {pairs}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)

        for symbol in pairs:
            try:
                count: int = await self.backfill(symbol)
                results[symbol] = count
            except HistorianError as e:
                error_block = (
                    f"\n{'!' * 60}\n"
                    f"BACKFILL FAILURE, SKIPPED\n"
                    f"Symbol: {symbol}\n"
                    f"Reason: {str(e)}\n"
                    f"{'!' * 60}"
                )
                logger.error(error_block)
                results[symbol] = 0

        info_block = (
            f"\n{'+' * 60}\n"
            f"ALL BACKFILLS COMPLETE\n"
            f"Results: {results}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
        return results

    # ── Private: Gap Detection ────────────────────────────────────────────────

    def _determine_start(self, symbol: str, now_utc: datetime) -> datetime:
        """
        Determine the UTC start datetime for a symbol's backfill window.

        Delegates to Storage to find the last known bar timestamp in the
        processed Parquet store. Two outcomes:

        - **First run** (no existing bars): Returns a datetime
            ``BACKFILL_YEARS`` in the past, normalised to midnight UTC.
        - **Resume** (bars exist): Returns the last bar's timestamp plus one
            minute, so the next request begins immediately after the stored data.

        Args:
            symbol:  Pure currency pair name (e.g., ``"EUR_USD"``).
            now_utc: Current UTC datetime. Injected as a parameter so tests
                        can control the clock without patching ``datetime.now``.

        Returns:
            datetime: Timezone-aware UTC start datetime for the backfill window.
        """
        # Use get_bars with max_rows=1 (returns the most recent bar) to query
        # the processed store. This respects the Isolation Principle — the
        # Historian never accesses Parquet files directly.
        df = self._storage.get_bars(symbol, timeframe="M1", max_rows=1)

        if df is None or df.empty:
            # First run: backfill from BACKFILL_YEARS ago at midnight UTC.
            # Use year replacement instead of timedelta(days=365*N) to avoid
            # accumulating leap-year error (each ignored Feb 29 shifts the
            # start date forward by one day relative to the calendar year).
            years: int = self._settings.backfill_years
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
            info_block = (
                f"\n{'+' * 60}\n"
                f"NO EXISTING DATA\n"
                f"Symbol: {symbol}\n"
                f"Current UTC time: {now_utc.isoformat()}\n"
                f"Backfilling full history from {start.date()} "
                f"({years} year(s))."
                f"\n{'+' * 60}"
            )
            logger.info(info_block)
            return start

        # Resume: one minute after the last stored bar to avoid re-fetching it
        raw_ts = df["timestamp"].iloc[-1]  # last row after sort-by-timestamp
        last_dt: datetime = (
            raw_ts.to_pydatetime() if hasattr(raw_ts, "to_pydatetime") else raw_ts
        )
        # Ensure timezone-aware for arithmetic consistency
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        start = last_dt + timedelta(minutes=1)

        info_block = (
            f"\n{'+' * 60}\n"
            f"EXISTING DATA FOUND\n"
            f"Symbol: {symbol}\n"
            f"Last bar: {last_dt.isoformat()}\n"
            f"Gap backfill from {start.isoformat()}.\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
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
                    saved: int = self._save_bars(symbol, bars)

                    if saved > 0:
                        total_saved += saved
                        consecutive_failures = 0  # reset: this chunk committed cleanly
                        info_block = (
                            f"\n{'+' * 60}\n"
                            f"CHUNK SAVED\n"
                            f"Symbol: {symbol}\n"
                            f"Chunk: {chunk_start.date()} → {chunk_end.date()}\n"
                            f"Bars saved: {saved}\n"
                            f"Running total: {total_saved}\n"
                            f"{'+' * 60}"
                        )
                        logger.info(info_block)
                    else:
                        # _save_bars returned 0 → StorageError was caught inside.
                        # Increment the consecutive-failure guard.
                        consecutive_failures += 1
                        warning_block = (
                            f"\n{'!' * 60}\n"
                            f"CHUNK STORAGE FAILURE\n"
                            f"Symbol: {symbol}\n"
                            f"Chunk: {chunk_start.date()} → {chunk_end.date()}\n"
                            f"Consecutive failures: {consecutive_failures} / "
                            f"{_MAX_CONSECUTIVE_STORAGE_FAILURES}\n\n"
                            f"CONTEXT: The chunk was fetched but could not be written.\n"
                            f"  [!] Possible causes: disk full, mount lost, permissions revoked.\n"
                            f"  This gap is recoverable — re-run the backfill after resolving.\n"
                            f"{'!' * 60}"
                        )
                        logger.warning(warning_block)

                        if consecutive_failures >= _MAX_CONSECUTIVE_STORAGE_FAILURES:
                            error_block = (
                                f"\n{'!' * 60}\n"
                                f"BACKFILL ABORTED: PERSISTENT STORAGE FAILURE\n"
                                f"Symbol: {symbol}\n"
                                f"Consecutive chunk failures: {consecutive_failures}\n"
                                f"Last failed window: {chunk_start.isoformat()} → {chunk_end.isoformat()}\n\n"
                                f"CONTEXT: {consecutive_failures} consecutive chunks fetched "
                                f"successfully from Twelve Data but failed to persist.\n"
                                f"  [!] Continuing would burn API quota for data that cannot\n"
                                f"      land on disk. Aborting to preserve daily quota.\n"
                                f"  [!] Re-run after resolving the storage condition.\n"
                                f"      Gap detection will resume from the last good bar.\n"
                                f"  [^] To manually recover: re-run backfill for {symbol} after\n"
                                f"      resolving the storage issue. The gap detector will refetch\n"
                                f"      from the last successfully stored bar automatically.\n"
                                f"{'!' * 60}"
                            )
                            logger.critical(error_block)
                            raise HistorianError(
                                f"Storage failed on {consecutive_failures} consecutive chunks "
                                f"for {symbol}. Aborting to preserve API quota.",
                                symbol=symbol,
                            )
                else:
                    # Expected for weekends, public holidays, or illiquid windows.
                    # Empty chunks do not count against the failure threshold.
                    debug_block = (
                        f"\n{'+' * 60}\n"
                        f"NO BARS RETURNED\n"
                        f"Symbol: {symbol}\n"
                        f"Chunk: {chunk_start.date()} → {chunk_end.date()}\n"
                        f"{'+' * 60}"
                    )
                    logger.debug(debug_block)

                # Advance to the minute after the end of this chunk
                chunk_start = chunk_end + timedelta(minutes=1)

        return total_saved

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
                        warning_block = (
                            f"\n{'^' * 60}\n"
                            f"RATE LIMIT HIT\n"
                            f"Symbol: {symbol}\n"
                            f"Chunk: {start_dt.date()} → {end_dt.date()}\n"
                            f"HTTP 429 Too Many Requests from Twelve Data.\n"
                            f"Attempt {attempt}/{_MAX_RETRIES}.\n"
                            f"Sleeping for {backoff:.0f}s before retrying."
                            f"\n{'^' * 60}"
                        )
                        logger.warning(warning_block)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    if resp.status != 200:
                        # Non-retryable HTTP error or transient server issue
                        error_text: str = await resp.text()
                        warning_block = (
                            f"\n{'!' * 60}\n"
                            f"HTTP ERROR\n"
                            f"Symbol: {symbol}\n"
                            f"Chunk: {start_dt.date()} → {end_dt.date()}\n"
                            f"HTTP {resp.status} from Twelve Data.\n"
                            f"Attempt {attempt}/{_MAX_RETRIES}.\n"
                            f"Response body (truncated to 200 chars):\n"
                            f"{error_text[:200]}"
                            f"\n{'!' * 60}"
                        )
                        logger.warning(warning_block)
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                        continue

                    try:
                        data: dict = await resp.json()
                    except Exception as exc:
                        error_type = type(exc).__name__
                        critical_block = (
                            f"\n{'!' * 60}\n"
                            f"API RESPONSE ERROR\n"
                            f"Symbol: {symbol}\n"
                            f"Chunk: {start_dt.date()} → {end_dt.date()}\n"
                            f"API response mishapen {_HTTP_TIMEOUT_S}s\n"
                            f"Attempt {attempt}/{_MAX_RETRIES}.\n"
                            f"Sleeping for {backoff:.0f}s before retrying.\n"
                            f"{'!' * 60}"
                        )
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        else:
                            critical_block += (
                                f"MAX RETRIES EXHAUSTED - BACKFILL ABORTED\n"
                                f"System cannot recover. Container will exit.\n"
                                f"{'!' * 60}"
                            )
                            logger.critical(critical_block)
                            sys.exit(1)  # Force container exit

                    # Twelve Data returns an error status in the JSON body even
                    # on HTTP 200 when the request is semantically invalid
                    if "values" not in data:
                        api_status: str = data.get("status", "unknown")
                        api_message: str = data.get("message", str(data))
                        api_code: int = data.get("code", 0)

                        # Distinguish configuration errors (symbol unknown to API)
                        # from expected empty windows (market closed, holiday).
                        # Code 400 / message containing "symbol" indicates
                        # misconfiguration — this will never self-resolve.
                        message_lower = api_message.lower()
                        is_symbol_error: bool = (
                            api_code == 400
                            or "symbol" in message_lower
                            or "not found" in message_lower
                            or "invalid" in message_lower
                        )

                        if is_symbol_error:
                            error_block = (
                                f"\n{'!' * 60}\n"
                                f"API ERROR — SYMBOL NOT RECOGNISED\n"
                                f"Symbol: {symbol}\n"
                                f"Chunk: {start_dt.date()} → {end_dt.date()}\n"
                                f"Status: {api_status} | Code: {api_code}\n"
                                f"Message: {api_message[:200]}\n\n"
                                f"CONTEXT: This is a configuration error, not a market-closed\n"
                                f"  window. The symbol is unknown to the Twelve Data API.\n"
                                f"  [!] Check BACKFILL_PAIRS in your .env file.\n"
                                f"  [!] Verify the symbol is supported at your subscription tier.\n"
                                f"{'!' * 60}"
                            )
                            logger.error(error_block)
                        else:
                            warning_block = (
                                f"\n{'!' * 60}\n"
                                f"API — NO DATA FOR WINDOW (market closed / holiday)\n"
                                f"Symbol: {symbol}\n"
                                f"Chunk: {start_dt.date()} → {end_dt.date()}\n"
                                f"Status: {api_status}\n"
                                f"Message: {api_message[:200]}\n"
                                f"{'!' * 60}"
                            )
                            logger.warning(warning_block)

                        # Neither case is a retry candidate — return empty
                        return []

                    return self._parse_bars(symbol, data["values"])

            except aiohttp.ClientError as exc:
                warning_block = (
                    f"\n{'!' * 60}\n"
                    f"NETWORK ERROR\n"
                    f"Symbol: {symbol}\n"
                    f"Chunk: {start_dt.date()} → {end_dt.date()}\n"
                    f"Attempt {attempt}/{_MAX_RETRIES}.\n"
                    f"Error: {exc}"
                    f"\n{'!' * 60}"
                )
                logger.warning(warning_block)
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff *= 2

        # All retry attempts exhausted
        error_block = (
            f"\n{'!' * 60}\n"
            f"HISTORIAN FETCH FAILURE\n"
            f"Symbol  : {symbol}\n"
            f"Window  : {start_dt.date()} → {end_dt.date()}\n"
            f"Retries : {_MAX_RETRIES} attempts exhausted\n\n"
            f"CONTEXT: All retry attempts failed. Possible causes:\n"
            f"  [^] Network connectivity issue or Twelve Data outage.\n"
            f"  [!] Invalid API key — check TWELVEDATA_API_KEY in .env.\n"
            f"  [%] Daily quota exhausted (800 req/day on free tier).\n"
            f"\nFIX: Verify the API key, confirm service status at\n"
            f"     status.twelvedata.com, then re-run the backfill.\n"
            f"{'!' * 60}"
        )
        logger.critical(error_block)
        raise HistorianError(
            f"All {_MAX_RETRIES} fetch attempts failed for {symbol} "
            f"({start_dt.date()} → {end_dt.date()}).",
            symbol=symbol,
        )

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
            debug_block = (
                f"\n{'-' * 60}\n"
                f"ENFORCING RATE LIMIT\n"
                f"Time since last request: {elapsed:.2f}s\n"
                f"Sleeping for remaining {remaining:.2f}s to respect "
                f"Twelve Data free-tier limit of 8 req/min."
                f"\n{'-' * 60}"
            )
            logger.debug(debug_block)
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
                warning_block = (
                    f"\n{'!' * 60}\n"
                    f"MALFORMED BAR SKIPPED\n"
                    f"Symbol: {symbol}\n"
                    f"Bar data: {v}\n"
                    f"Error: {exc}\n"
                    f"{'!' * 60}"
                )
                logger.debug(warning_block)

        if skipped:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"BARS SKIPPED DUE TO VALIDATION/PARSE ERRORS\n"
                f"Symbol: {symbol}\n"
                f"Skipped {skipped} out of {len(values)} bars "
                f"({skipped / len(values) * 100:.1f}%).\n"
                f"Check debug logs for details on each skipped bar."
                f"\n{'%' * 60}"
            )
            logger.warning(warning_block)

        return bars

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
            warning_block = (
                f"\n{'%' * 60}\n"
                f"STORAGE ERROR: Failed to save bar batch\n"
                f"Symbol: {symbol}\n"
                f"Bars in batch: {len(bars)}\n"
                f"Error: {exc}\n"
                f"Batch was not saved. The backfill will continue with the next "
                f"chunk, but this window will be missing from storage.\n"
                f"Consider re-running the backfill later to attempt saving this "
                f"chunk again."
                f"\n{'%' * 60}"
            )
            logger.warning(warning_block)
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
