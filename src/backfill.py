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
import os
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
    "XAU_USD": "XAU/USD",
}

TD_BASE = "https://api.twelvedata.com/time_series"
CHUNK_SIZE = 5000  # bars per request (Twelve Data max)
REQUEST_INTERVAL = 10.0  # seconds between requests (8/min = free tier limit)
DEFAULT_BACKFILL_YEARS = 2


# ── Helpers ────────────────────────────────────────────────────────────────────


def _bars_to_ticks(bars: list[dict[str, Any]], pair: str) -> pd.DataFrame:
    """
    Convert M1 OHLCV bars to synthetic tick records.

    Each bar produces 4 synthetic ticks: open, high, low, close.
    Timestamps are spaced 15 s apart within the minute.
    ``bid == ask`` (zero spread) for historical bars — features degrade
    gracefully when spread is 0.

    Args:
        bars: List of dicts from Twelve Data ``values`` array.
        pair: Internal pair name (``"EUR_USD"``).

    Returns:
        Sorted, deduplicated DataFrame with tick columns.
    """
    rows: list[dict[str, Any]] = []
    for bar in bars:
        try:
            dt = datetime.strptime(bar["datetime"], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            o = float(bar["open"])
            h = float(bar["high"])
            l_ = float(bar["low"])
            c = float(bar["close"])
        except (KeyError, ValueError):
            continue

        for offset_s, price in [(0, o), (15, h), (30, l_), (45, c)]:
            rows.append(
                {
                    "timestamp": pd.Timestamp(dt + timedelta(seconds=offset_s)),
                    "bid": price,
                    "ask": price,
                    "spread": 0.0,
                    "pair": pair,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "spread", "pair"])

    df = pd.DataFrame(rows)
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
    months_needed = max(1, min_days // 22 + 1)
    now = datetime.now(timezone.utc)
    found = 0

    for m in range(months_needed):
        year = now.year if now.month - m > 0 else now.year - 1
        month = (now.month - m - 1) % 12 + 1
        blob_path = f"data/{pair}/{year}-{month:02d}.parquet"
        if storage.blob_exists(blob_path):
            found += 1

    logger.info(
        {
            "event": "coverage_check",
            "pair": pair,
            "months_found": found,
            "months_needed": months_needed,
        }
    )
    return found >= months_needed


# ── Main backfill logic ────────────────────────────────────────────────────────


async def backfill_pair(
    pair: str,
    api_key: str,
    storage: Any,
    years_back: int = DEFAULT_BACKFILL_YEARS,
) -> None:
    """
    Fetch up to *years_back* years of M1 bars for *pair* via Twelve Data.

    Writes monthly parquet blobs to storage, skipping months whose blobs
    already exist (idempotent).

    Uses :mod:`aiohttp` for non-blocking HTTP so the caller's event loop
    is not stalled.

    Args:
        pair: Internal pair name (``"EUR_USD"``).
        api_key: Twelve Data API key.
        storage: :class:`~storage.StorageManager` instance.
        years_back: How many years of history to fetch.
    """
    symbol = PAIR_TO_TD.get(pair)
    if not symbol:
        logger.error({"event": "backfill_unknown_pair", "pair": pair})
        return

    # Quick existence check — no data downloaded
    if check_data_coverage(pair, storage, min_days=365 * years_back):
        logger.info(
            {
                "event": "backfill_skipped",
                "pair": pair,
                "reason": f"{years_back}+ years of data already in blob storage",
            }
        )
        return

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * years_back)

    logger.info(
        {
            "event": "backfill_start",
            "pair": pair,
            "symbol": symbol,
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "source": "TwelveData",
        }
    )

    # Track which months we've already written in this run to avoid
    # re-checking blob existence for every chunk.
    written_months: set[str] = set()

    total_bars = 0
    request_count = 0
    current_end = end_date

    async with aiohttp.ClientSession(
        headers={"User-Agent": "TraderAI/1.0"},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        while current_end > start_date:
            current_start = max(current_end - timedelta(days=7), start_date)

            params = {
                "symbol": symbol,
                "interval": "1min",
                "start_date": current_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": current_end.strftime("%Y-%m-%d %H:%M:%S"),
                "outputsize": CHUNK_SIZE,
                "apikey": api_key,
                "timezone": "UTC",
                "format": "JSON",
            }

            try:
                t0 = time.monotonic()
                async with session.get(TD_BASE, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                request_count += 1

                if data.get("status") == "error":
                    code = data.get("code", "")
                    msg = data.get("message", "")
                    if "429" in str(code) or "rate" in msg.lower():
                        logger.warning(
                            {
                                "event": "backfill_rate_limited",
                                "pair": pair,
                                "waiting_s": 60,
                            }
                        )
                        await asyncio.sleep(60)
                        continue
                    logger.warning(
                        {
                            "event": "backfill_api_error",
                            "pair": pair,
                            "code": code,
                            "msg": msg,
                        }
                    )
                    break

                bars = data.get("values", [])
                if not bars:
                    current_end = current_start - timedelta(minutes=1)
                    continue

                tick_df = _bars_to_ticks(bars, pair)
                if tick_df.empty:
                    current_end = current_start - timedelta(minutes=1)
                    continue

                total_bars += len(bars)

                # Split into monthly slices and write directly.
                tick_df["_ym"] = tick_df["timestamp"].dt.to_period("M")
                for period, grp in tick_df.groupby("_ym"):
                    # Pyright doesn't understand pandas Period attributes —
                    # use int() to extract scalar values explicitly.
                    period_year: int = int(period.year)  # type: ignore[union-attr]
                    period_month: int = int(period.month)  # type: ignore[union-attr]
                    month_key = f"{period_year}-{period_month:02d}"

                    # Skip months we've already written in this run
                    if month_key in written_months:
                        continue

                    blob_path = f"data/{pair}/{month_key}.parquet"

                    # Also skip if blob already existed before this run
                    if storage.blob_exists(blob_path):
                        written_months.add(month_key)
                        continue

                    storage.write_raw_parquet(blob_path, grp.drop(columns=["_ym"]))
                    written_months.add(month_key)
                    logger.info(
                        {
                            "event": "backfill_month_written",
                            "pair": pair,
                            "year": period_year,
                            "month": period_month,
                            "rows": len(grp),
                            "blob_path": blob_path,
                        }
                    )

                del tick_df  # free memory immediately

                logger.info(
                    {
                        "event": "backfill_chunk",
                        "pair": pair,
                        "bars_fetched": len(bars),
                        "window_start": current_start.strftime("%Y-%m-%d"),
                        "window_end": current_end.strftime("%Y-%m-%d"),
                        "total_bars": total_bars,
                        "requests": request_count,
                    }
                )

                # Walk backwards: oldest bar becomes new end
                oldest_dt_str = bars[-1]["datetime"]
                current_end = datetime.strptime(
                    oldest_dt_str, "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc) - timedelta(minutes=1)

                # Rate-limit: sleep remainder of REQUEST_INTERVAL
                elapsed = time.monotonic() - t0
                wait = max(0.0, REQUEST_INTERVAL - elapsed)
                if wait > 0:
                    await asyncio.sleep(wait)

            except aiohttp.ClientError as exc:
                logger.warning(
                    {
                        "event": "backfill_request_failed",
                        "pair": pair,
                        "error": str(exc),
                    }
                )
                await asyncio.sleep(30)
                continue

    logger.info(
        {
            "event": "backfill_complete",
            "pair": pair,
            "total_bars": total_bars,
            "requests": request_count,
        }
    )


async def backfill_all(
    pairs: list[str],
    storage: Any,
    years_back: int = DEFAULT_BACKFILL_YEARS,
) -> None:
    """
    Run backfill for all pairs sequentially (to respect rate limits).

    Args:
        pairs: List of internal pair names.
        storage: :class:`~storage.StorageManager` instance.
        years_back: How many years of history to fetch per pair.
    """
    api_key = os.environ.get("TWELVEDATA_API_KEY", "")
    if not api_key:
        logger.error(
            {"event": "backfill_no_api_key", "reason": "TWELVEDATA_API_KEY not set"}
        )
        return

    for pair in pairs:
        await backfill_pair(pair, api_key, storage, years_back)

    logger.info({"event": "backfill_all_complete", "pairs": pairs})
