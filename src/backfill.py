"""
backfill.py — Historical M1 bar downloader using Twelve Data REST API.

Downloads 1-minute OHLCV bars via Twelve Data's ``/time_series`` endpoint
(same API key used for live streaming). Converts bars to synthetic tick
records compatible with the existing storage schema, then writes monthly
parquet blobs.

Strategy:
    - One REST request = up to 5000 M1 bars (~3.5 days of market hours)
    - Walks backwards from today, fetching 5000-bar chunks
    - Stops when target coverage is reached or 2 years covered
    - Free plan: 800 requests/day, 8/min — full 2yr backfill in ~150 requests
    - Skips months already present in blob storage (idempotent)

Rate limiting: 1 request per 10s to stay within the 8 req/min free tier.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

PAIR_TO_TD: dict[str, str] = {
    "EUR_USD": "EUR/USD",
    "GBP_USD": "GBP/USD",
    "USD_JPY": "USD/JPY",
    "AUD_USD": "AUD/USD",
    "USD_CAD": "USD/CAD",
    "USD_CHF": "USD/CHF",
    "NZD_USD": "NZD/USD",
    "XAU_USD": "XAU/USD",
}

TD_BASE: str = "https://api.twelvedata.com/time_series"
CHUNK_SIZE: int = 5000  # bars per request (Twelve Data max)
REQUEST_INTERVAL: float = 10.0  # seconds between requests (8/min = free tier limit)
DEFAULT_BACKFILL_YEARS: int = 2
RATE_LIMIT_WAIT: int = 60  # seconds to wait on 429
REQUEST_ERROR_WAIT: int = 30  # seconds to wait on connection error


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_params(pair: str, current_end: datetime, api_key: str) -> dict[str, Any]:
    """Constructs the URL parameters for the Twelve Data request."""
    symbol = PAIR_TO_TD.get(pair)
    # We fetch 7 days at a time to stay under the 5000 bar limit for M1
    current_start = current_end - timedelta(days=7)

    return {
        "symbol": symbol,
        "interval": "1min",
        "start_date": current_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": current_end.strftime("%Y-%m-%d %H:%M:%S"),
        "outputsize": 5000,
        "apikey": api_key,
        "timezone": "UTC",
        "format": "JSON",
    }


async def _handle_backfill_error(data: dict[str, Any] | None, pair: str) -> None:
    """Logs errors and handles mandatory sleep times for rate limits."""
    if data is None:
        logger.warning({"event": "backfill_no_data_received", "pair": pair})
        await asyncio.sleep(5)
        return

    code = data.get("code")
    msg = data.get("message", "Unknown error")

    if code == 429 or "rate" in msg.lower():
        logger.warning(
            {
                "event": "backfill_rate_limited",
                "pair": pair,
                "wait_time": 60,
                "function": "_handle_backfill_error",
                "file": "backfill.py",
            }
        )
        await asyncio.sleep(60)  # Twelve Data free tier cooldown
    else:
        logger.error(
            {
                "event": "backfill_api_error",
                "pair": pair,
                "code": code,
                "message": msg,
                "function": "_handle_backfill_error",
                "file": "backfill.py",
            }
        )
        await asyncio.sleep(10)


def _parse_bar_datetime(raw: str) -> datetime:
    """
    Parse a Twelve Data datetime string into a timezone-aware UTC datetime.

    Handles formats:
        - ``"2025-01-15 12:00:00"``
        - ``"2025-01-15 12:00:00.000"``
        - ``"2025-01-15T12:00:00"``

    Args:
        raw: Raw datetime string from Twelve Data.

    Returns:
        Timezone-aware UTC datetime.

    Raises:
        ValueError: If no known format matches.
    """
    try:
        # Strip milliseconds if present
        cleaned: str = raw.split(".")[0].replace("T", " ")
        return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except Exception as exc:
        logger.error(
            {
                "event": "datetime_parse_critical_error",
                "raw_input": raw,
                "error": str(exc),
                "function": "_parse_bar_datetime",
                "file": "backfill.py",
                "description": "Failed to parse datetime string from Twelve Data. This may indicate a change in API response format or unexpected data. Manual investigation required.",
            }
        )
        raise  # Re-raise because the main loop handles skipped bars


def _bars_to_ticks(bars: list[dict[str, Any]], pair: str) -> pd.DataFrame:
    """
    Convert M1 OHLCV bars to synthetic tick records.

    Each bar produces 4 synthetic ticks: open, high, low, close.
    Timestamps are spaced 15 s apart within the minute.
    ``bid == ask`` (zero spread) for historical bars — features degrade
    gracefully when spread is 0.

    Args:
        bars: List of dicts from Twelve Data ``values`` array.
            Each must have keys ``datetime``, ``open``, ``high``,
            ``low``, ``close``.
        pair: Internal pair name (``"EUR_USD"``).

    Returns:
        Sorted, deduplicated DataFrame with tick columns
        (``timestamp``, ``bid``, ``ask``, ``spread``, ``pair``).
    """
    rows: list[dict[str, Any]] = []
    for bar in bars:
        try:
            dt: datetime = _parse_bar_datetime(bar["datetime"])
            o: float = float(bar["open"])
            h: float = float(bar["high"])
            l_: float = float(bar["low"])
            c: float = float(bar["close"])
            for offset_s, price in [(0, o), (15, h), (30, l_), (45, c)]:
                rows.append(
                    {
                        "timestamp": pd.Timestamp(
                            dt + timedelta(seconds=offset_s), tz="UTC"
                        ),
                        "bid": price,
                        "ask": price,
                        "spread": 0.0,
                        "pair": pair,
                    }
                )
        except (KeyError, ValueError) as exc:
            logger.warning(
                {
                    "event": "bar_conversion_skipped",
                    "pair": pair,
                    "error": str(exc),
                    "bar_data": str(bar)[:100],
                    "function": "_bars_to_ticks",
                    "file": "backfill.py",
                }
            )
            continue

        for offset_s, price in [(0, o), (15, h), (30, l_), (45, c)]:
            rows.append(
                {
                    "timestamp": pd.Timestamp(
                        dt + timedelta(seconds=offset_s), tz="UTC"
                    ),
                    "bid": price,
                    "ask": price,
                    "spread": 0.0,
                    "pair": pair,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "spread", "pair"])

    df: pd.DataFrame = pd.DataFrame(rows)
    return (
        df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    )


def check_data_coverage(pair: str, storage: Any, min_days: int = 365) -> bool:
    """
    Check whether *pair* has at least *min_days* of data in blob storage.

    This checks blob **existence** only — it does NOT download any data.
    Each monthly blob is assumed to represent ~22 trading days.

    Args:
        pair: Internal pair name.
        storage: :class:`~storage.StorageManager` instance.
        min_days: Minimum coverage in calendar days.

    Returns:
        ``True`` if enough monthly blobs exist.
    """
    if storage is None:
        logger.error(
            {
                "event": "storage_not_initialized",
                "function": "check_data_coverage",
                "file": "backfill.py",
                "reason": "Storage manager instance is None. Cannot check data coverage without storage access.",
            }
        )
        return False

    months_needed: int = max(1, min_days // 22 + 1)
    now: datetime = datetime.now(timezone.utc)
    found: int = 0

    for m in range(months_needed):
        # Calculation for year/month offset
        target_date = now - timedelta(days=m * 30)
        blob_path: str = (
            f"data/{pair}/{target_date.year}-{target_date.month:02d}.parquet"
        )
        try:
            if storage.blob_exists(blob_path):
                found += 1
        except Exception as exc:
            logger.warning(
                {
                    "event": "storage_check_failed",
                    "path": blob_path,
                    "error": str(exc),
                    "function": "check_data_coverage",
                    "file": "backfill.py",
                }
            )
    logger.info(
        {
            "event": "coverage_check",
            "pair": pair,
            "months_found": found,
            "months_needed": months_needed,
            "coverage_percent": round(found / months_needed * 100, 2),
            "min_days": min_days,
            "function": "check_data_coverage",
            "file": "backfill.py",
        }
    )
    return found >= months_needed


# ── Main backfill logic ────────────────────────────────────────────────────────
async def _fetch_chunk(
    session: aiohttp.ClientSession, params: dict[str, Any]
) -> dict[str, Any] | None:
    """Handles the raw HTTP request to Twelve Data."""
    try:
        async with session.get(TD_BASE, params=params) as resp:
            if resp.status == 429:
                return {"status": "error", "code": 429, "message": "Rate limited"}
            resp.raise_for_status()
            return await resp.json()
    except Exception as exc:
        logger.error(
            {
                "event": "backfill_http_critical",
                "error": str(exc),
                "function": "_fetch_chunk",
                "file": "backfill.py",
            }
        )
        return None


def _process_and_save_chunk(
    bars: list, pair: str, storage: Any, written_months: set
) -> int:
    """Transforms bars and writes new months to storage. Returns count of bars processed."""
    tick_df = _bars_to_ticks(bars, pair)
    if tick_df.empty:
        return 0

    tick_df["_ym"] = tick_df["timestamp"].dt.to_period("M")
    bars_saved = 0

    for period, grp in tick_df.groupby("_ym"):
        p: Any = period
        period_year: int = int(p.year)
        period_month: int = int(p.month)
        month_key = f"{int(period_year)}-{int(period_month):02d}"
        blob_path = f"data/{pair}/{month_key}.parquet"

        if month_key in written_months:
            continue

        # Check Azure if not already in our local set
        if storage.blob_exists(blob_path):
            written_months.add(month_key)
            continue

        storage.write_raw_parquet(blob_path, grp.drop(columns=["_ym"]))
        written_months.add(month_key)
        bars_saved += len(grp)

        logger.info(
            {
                "event": "backfill_month_written",
                "pair": pair,
                "month": month_key,
                "function": "_process_and_save_chunk",
                "file": "backfill.py",
            }
        )

    # Memory Safety: Clear the large DataFrame once processing is done
    del tick_df
    return bars_saved


async def backfill_pair(
    pair: str, api_key: str, storage: Any, years_back: int = DEFAULT_BACKFILL_YEARS
) -> None:
    # ... [Same Anchor Month Check as before] ...

    written_months: set[str] = set()
    current_end = datetime.now(timezone.utc)
    start_date = current_end - timedelta(days=365 * years_back)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        while current_end > start_date:
            t0 = time.monotonic()
            params = _build_params(
                pair, current_end, api_key
            )  # Simple helper to pack dict

            data = await _fetch_chunk(session, params)

            if not data or data.get("status") == "error":
                # Handle Wait/Retry logic based on error code
                await _handle_backfill_error(data, pair)
                continue

            bars = data.get("values", [])
            if not bars:
                current_end -= timedelta(days=7)  # Slide window if gap in data
                continue

            # Process data
            _process_and_save_chunk(bars, pair, storage, written_months)

            # Move cursor back
            current_end = _parse_bar_datetime(bars[-1]["datetime"]) - timedelta(
                minutes=1
            )

            # Rate Limit Wait
            await asyncio.sleep(max(0, REQUEST_INTERVAL - (time.monotonic() - t0)))


async def backfill_all(
    pairs: list[str],
    storage: Any,
    years_back: int = DEFAULT_BACKFILL_YEARS,
    api_key: str = "",
) -> None:
    """
    Run backfill for all pairs sequentially (to respect rate limits).

    Args:
        pairs: List of internal pair names.
        storage: :class:`~storage.StorageManager` instance.
        years_back: How many years of history to fetch per pair.
        api_key: Twelve Data API key.  Falls back to
            ``TWELVEDATA_API_KEY`` env var if empty.
    """
    if not pairs:
        logger.warning(
            {
                "event": "backfill_no_pairs_provided",
                "function": "backfill_all",
                "file": "backfill.py",
            }
        )
        return
    if not api_key:
        import os

        api_key = os.environ.get("TWELVEDATA_API_KEY", "")

    if not api_key:
        logger.error(
            {
                "event": "backfill_no_api_key",
                "reason": "No API key provided and TWELVEDATA_API_KEY not set",
                "function": "backfill_all",
                "file": "backfill.py",
            }
        )
        return

    for pair in pairs:
        logger.info(
            {
                "event": "processing_pair",
                "pair": pair,
                "function": "backfill_all",
                "file": "backfill.py",
            }
        )
        await backfill_pair(pair, api_key, storage, years_back)

    logger.info(
        {
            "event": "backfill_all_complete",
            "pairs": pairs,
            "function": "backfill_all",
            "file": "backfill.py",
        }
    )
