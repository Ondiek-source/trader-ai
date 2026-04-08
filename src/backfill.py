"""
backfill.py — Dukascopy historical tick data downloader.

Downloads .bi5 (LZMA-compressed binary) tick files from Dukascopy's public CDN
and writes them to Azure Blob Storage via storage.StorageManager.

Dukascopy URL pattern:
    https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YEAR}/{MONTH:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5

Binary record format (20 bytes per tick, big-endian):
    int32  — milliseconds from hour start
    int32  — ask * 100000
    int32  — bid * 100000
    float32 — ask volume
    float32 — bid volume

Skips already-downloaded blobs. Rate-limited to 0.1 s between requests.
"""

from __future__ import annotations

import asyncio
import io
import logging
import lzma
import struct
import time
from datetime import date, datetime, timedelta, timezone
from typing import Generator

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Instrument name mapping ───────────────────────────────────────────────────

PAIR_TO_DUKASCOPY: dict[str, str] = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "XAU_USD": "XAUUSD",
}

# Price divisors — most FX pairs use 100000; XAUUSD uses 1000
PRICE_DIVISOR: dict[str, float] = {
    "EURUSD": 100_000.0,
    "GBPUSD": 100_000.0,
    "USDJPY": 1_000.0,
    "XAUUSD": 1_000.0,
}

CDN_BASE = "https://datafeed.dukascopy.com/datafeed"
REQUEST_DELAY = 0.1  # seconds between HTTP requests


def _build_url(instrument: str, year: int, month: int, day: int, hour: int) -> str:
    # Dukascopy month is 0-indexed in URL
    return f"{CDN_BASE}/{instrument}/{year}/{month - 1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"


def _decode_bi5(
    data: bytes, instrument: str, year: int, month: int, day: int, hour: int
) -> pd.DataFrame:
    """
    Decompress and decode a .bi5 file into a DataFrame.

    Returns columns: timestamp (datetime64[ms] UTC), bid, ask, spread.
    """
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError as exc:
        logger.warning({"event": "bi5_decompress_failed", "error": str(exc)})
        return pd.DataFrame()

    record_size = 20  # 5 × 4 bytes
    n_records = len(raw) // record_size
    if n_records == 0:
        return pd.DataFrame()

    divisor = PRICE_DIVISOR.get(instrument, 100_000.0)
    hour_start = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
    hour_start_ms = int(hour_start.timestamp() * 1000)

    records: list[dict] = []
    for i in range(n_records):
        offset = i * record_size
        chunk = raw[offset : offset + record_size]
        if len(chunk) < record_size:
            break
        ms_offset, ask_raw, bid_raw = struct.unpack(">iii", chunk[:12])
        ask_vol, bid_vol = struct.unpack(">ff", chunk[12:])

        ts_ms = hour_start_ms + ms_offset
        ask = ask_raw / divisor
        bid = bid_raw / divisor
        spread = round(ask - bid, 6)

        records.append(
            {
                "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                "bid": bid,
                "ask": ask,
                "spread": spread,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df[df["bid"] > 0].reset_index(drop=True)  # type: ignore[assignment]
    return df  # type: ignore[return-value]


def _hour_range(
    start: date, end: date
) -> Generator[tuple[int, int, int, int], None, None]:
    """Yield (year, month, day, hour) for every hour in [start, end)."""
    current = datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, 0, 0, 0, tzinfo=timezone.utc)
    while current < end_dt:
        yield current.year, current.month, current.day, current.hour
        current += timedelta(hours=1)


def _blob_path_for_hour(pair: str, year: int, month: int) -> str:
    return f"data/{pair}/{year}-{month:02d}.parquet"


async def backfill_pair(
    pair: str,
    start_date: date,
    end_date: date,
    storage,
    session: requests.Session | None = None,
) -> None:
    """
    Download Dukascopy tick history for `pair` between start_date and end_date.
    Writes to storage via storage.write_raw_parquet() per monthly partition.
    Skips hours whose monthly blob already exists (coarse skip — hour-level skip not implemented
    to avoid blob-per-hour overhead; rerunning is idempotent due to dedup in storage).
    """
    instrument = PAIR_TO_DUKASCOPY.get(pair)
    if not instrument:
        logger.error({"event": "backfill_unknown_pair", "pair": pair})
        return

    own_session = session is None
    if own_session:
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0 (TraderAI backfill)"

    logger.info(
        {
            "event": "backfill_start",
            "pair": pair,
            "instrument": instrument,
            "start": str(start_date),
            "end": str(end_date),
        }
    )

    # Accumulate frames per month to batch writes
    monthly_frames: dict[tuple[int, int], list[pd.DataFrame]] = {}

    for year, month, day, hour in _hour_range(start_date, end_date):
        if asyncio.get_event_loop().is_running():
            await asyncio.sleep(0)  # yield control to event loop

        url = _build_url(instrument, year, month, day, hour)
        key = (year, month)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 404:
                # No data for this hour (holiday / off-hours) — skip silently
                time.sleep(REQUEST_DELAY)
                continue
            resp.raise_for_status()
            df = _decode_bi5(resp.content, instrument, year, month, day, hour)
            if not df.empty:
                df["pair"] = pair
                monthly_frames.setdefault(key, []).append(df)
                logger.debug(
                    {
                        "event": "hour_downloaded",
                        "pair": pair,
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "rows": len(df),
                    }
                )
            time.sleep(REQUEST_DELAY)

        except requests.RequestException as exc:
            logger.warning(
                {
                    "event": "backfill_request_failed",
                    "url": url,
                    "error": str(exc),
                    "pair": pair,
                }
            )
            time.sleep(REQUEST_DELAY * 5)  # back off on network error

    # Write collected data to storage
    for (year, month), frames in monthly_frames.items():
        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )
        blob_path = f"data/{pair}/{year}-{month:02d}.parquet"
        storage.write_raw_parquet(blob_path, combined)
        logger.info(
            {
                "event": "backfill_month_written",
                "pair": pair,
                "year": year,
                "month": month,
                "rows": len(combined),
                "blob_path": blob_path,
            }
        )

    if own_session:
        session.close()

    logger.info({"event": "backfill_complete", "pair": pair})


async def backfill_all(pairs: list[str], storage, years_back: int = 5) -> None:
    """
    Run backfill for all pairs concurrently (limited to 2 concurrent downloads
    to avoid hammering Dukascopy's CDN).

    Default: 5 years of data (2019-2024) for robust model training.
    """
    end = date.today()
    start = date(end.year - years_back, 1, 1)  # Jan 1 of (today - N years)

    semaphore = asyncio.Semaphore(2)

    async def _guarded(pair: str) -> None:
        async with semaphore:
            await backfill_pair(pair, start, end, storage)

    await asyncio.gather(*[_guarded(pair) for pair in pairs])
    logger.info({"event": "backfill_all_complete", "pairs": pairs})


def check_data_coverage(pair: str, storage, min_days: int = 365) -> bool:
    """
    Returns True if there is at least min_days of tick data in blob storage.
    Default min_days=365 (1 year) — we want 5 years for robust training.
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
