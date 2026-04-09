"""
backfill.py — Historical M1 bar downloader using Twelve Data REST API.

Downloads 1-minute OHLCV bars via Twelve Data's /time_series endpoint
(same API key used for live streaming). Converts bars to synthetic tick
records compatible with the existing storage schema, then writes monthly
parquet blobs.

Strategy:
  - One REST request = up to 5000 M1 bars (~3.5 days of market hours)
  - Walks backwards from today, fetching 5000-bar chunks
  - Stops when target coverage is reached or 2 years covered
  - Free plan: 800 requests/day, 8/min → full 2yr backfill in ~150 requests
  - Skips months already present in blob storage (idempotent)

Rate limiting: 1 request per 8s to stay within 8 req/min free tier limit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Instrument name mapping ───────────────────────────────────────────────────

PAIR_TO_TD: dict[str, str] = {
    "EUR_USD": "EUR/USD",
    "GBP_USD": "GBP/USD",
    "USD_JPY": "USD/JPY",
    "XAU_USD": "XAU/USD",
}

TD_BASE = "https://api.twelvedata.com/time_series"
CHUNK_SIZE = 5000       # bars per request (Twelve Data max)
REQUEST_INTERVAL = 8.0  # seconds between requests (8/min = free tier limit)
BACKFILL_YEARS = 2      # how many years to try to fetch


def _bars_to_ticks(bars: list[dict], pair: str) -> pd.DataFrame:
    """
    Convert M1 OHLCV bars to synthetic tick records.

    Each bar produces 4 synthetic ticks: open, high, low, close.
    Timestamps are spaced 15s apart within the minute.
    bid == ask (zero spread) for historical bars — features degrade
    gracefully when spread is 0.
    """
    rows = []
    for bar in bars:
        try:
            # Twelve Data returns datetime as "YYYY-MM-DD HH:MM:SS"
            dt = datetime.strptime(bar["datetime"], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            o = float(bar["open"])
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])
        except (KeyError, ValueError):
            continue

        for offset_s, price in [(0, o), (15, h), (30, l), (45, c)]:
            rows.append({
                "timestamp": pd.Timestamp(dt + timedelta(seconds=offset_s)),
                "bid": price,
                "ask": price,
                "spread": 0.0,
                "pair": pair,
            })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "spread", "pair"])
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


async def backfill_pair(pair: str, api_key: str, storage, years_back: int = BACKFILL_YEARS) -> None:
    """
    Fetch up to `years_back` years of M1 bars for `pair` via Twelve Data.
    Writes monthly parquet blobs to storage, skipping months already present.
    """
    symbol = PAIR_TO_TD.get(pair)
    if not symbol:
        logger.error({"event": "backfill_unknown_pair", "pair": pair})
        return

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * years_back)

    logger.info({
        "event": "backfill_start",
        "pair": pair,
        "symbol": symbol,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "source": "TwelveData",
    })

    session = requests.Session()
    session.headers["User-Agent"] = "TraderAI/1.0"

    # Walk backwards in time, fetching chunks of CHUNK_SIZE bars
    current_end = end_date
    monthly_frames: dict[tuple[int, int], list[pd.DataFrame]] = {}
    total_bars = 0
    request_count = 0

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

        await asyncio.sleep(0)  # yield to event loop

        try:
            t0 = time.monotonic()
            resp = session.get(TD_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            request_count += 1

            if data.get("status") == "error":
                code = data.get("code", "")
                msg = data.get("message", "")
                if "429" in str(code) or "rate" in msg.lower():
                    logger.warning({"event": "backfill_rate_limited", "pair": pair, "waiting_s": 60})
                    await asyncio.sleep(60)
                    continue
                logger.warning({"event": "backfill_api_error", "pair": pair, "code": code, "msg": msg})
                break

            bars = data.get("values", [])
            if not bars:
                # No data in this window — move further back
                current_end = current_start - timedelta(minutes=1)
                continue

            tick_df = _bars_to_ticks(bars, pair)
            if not tick_df.empty:
                total_bars += len(bars)
                # Group into monthly buckets
                tick_df["_ym"] = tick_df["timestamp"].dt.to_period("M")
                for period, grp in tick_df.groupby("_ym"):
                    key = (period.year, period.month)
                    monthly_frames.setdefault(key, []).append(grp.drop(columns=["_ym"]))

            logger.info({
                "event": "backfill_chunk",
                "pair": pair,
                "bars_fetched": len(bars),
                "window_start": current_start.strftime("%Y-%m-%d"),
                "window_end": current_end.strftime("%Y-%m-%d"),
                "total_bars": total_bars,
                "requests": request_count,
            })

            # Oldest bar in this chunk becomes the new end
            oldest_dt_str = bars[-1]["datetime"]
            current_end = datetime.strptime(oldest_dt_str, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            ) - timedelta(minutes=1)

            # Rate-limit: sleep remainder of REQUEST_INTERVAL
            elapsed = time.monotonic() - t0
            wait = max(0.0, REQUEST_INTERVAL - elapsed)
            if wait > 0:
                await asyncio.sleep(wait)

        except requests.RequestException as exc:
            logger.warning({"event": "backfill_request_failed", "pair": pair, "error": str(exc)})
            await asyncio.sleep(30)
            continue

    session.close()

    # Flush all monthly frames to blob storage
    months_written = 0
    for (year, month), frames in sorted(monthly_frames.items()):
        blob_path = f"data/{pair}/{year}-{month:02d}.parquet"
        if storage.blob_exists(blob_path):
            logger.debug({"event": "backfill_month_exists", "pair": pair, "year": year, "month": month})
            continue
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )
        storage.write_raw_parquet(blob_path, combined)
        months_written += 1
        logger.info({
            "event": "backfill_month_written",
            "pair": pair, "year": year, "month": month,
            "rows": len(combined), "blob_path": blob_path,
        })

    logger.info({
        "event": "backfill_complete",
        "pair": pair,
        "total_bars": total_bars,
        "requests": request_count,
        "months_written": months_written,
    })


async def backfill_all(pairs: list[str], storage, years_back: int = BACKFILL_YEARS) -> None:
    """Run backfill for all pairs (sequentially to respect rate limits)."""
    # Get API key from storage config — passed via environment via config
    api_key = _get_api_key()
    if not api_key:
        logger.error({"event": "backfill_no_api_key", "reason": "TWELVEDATA_API_KEY not set"})
        return

    for pair in pairs:
        await backfill_pair(pair, api_key, storage, years_back)

    logger.info({"event": "backfill_all_complete", "pairs": pairs})


def _get_api_key() -> str:
    """Read TWELVEDATA_API_KEY from environment."""
    import os
    return os.environ.get("TWELVEDATA_API_KEY", "")


def check_data_coverage(pair: str, storage, min_days: int = 365) -> bool:
    """
    Returns True if there is at least min_days of data in blob storage.
    Used at startup to decide whether a backfill run is needed.
    """
    df = storage.read_ticks(pair, months=3)
    if df.empty:
        return False
    ts_col = "timestamp"
    if ts_col not in df.columns:
        return False
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    span = (df[ts_col].max() - df[ts_col].min()).days
    return span >= min_days
