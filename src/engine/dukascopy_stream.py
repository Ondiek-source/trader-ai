"""
engine/dukascopy_stream.py — Live M1 bar stream from Dukascopy.

Provides a real-time feed of completed M1 OHLCV bars using
dukascopy_python.live_fetch(). Each bar carries real tick volume
matching the historian's backfill data — no training/live mismatch.

A single background thread runs the blocking live_fetch iterator
and pushes completed bars into a thread-safe queue. The async
subscribe() generator yields bars to the LiveEngine.

Usage:
    stream = DukascopyLiveStream(symbol="EUR_USD")
    stream.connect()
    async for bar in stream.subscribe():
        engine.process_bar(bar)
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import AsyncGenerator, Any

import dukascopy_python
from dukascopy_python import OFFER_SIDE_BID
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

# ── Symbol → Dukascopy instrument mapping ───────────────────────────────────
_INSTRUMENT_MAP: dict[str, Any] = {
    "EUR_USD": INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBP_USD": INSTRUMENT_FX_MAJORS_GBP_USD,
    "USD_JPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "AUD_USD": INSTRUMENT_FX_MAJORS_AUD_USD,
    "USD_CAD": INSTRUMENT_FX_MAJORS_USD_CAD,
    "USD_CHF": INSTRUMENT_FX_MAJORS_USD_CHF,
    "NZD_USD": INSTRUMENT_FX_MAJORS_NZD_USD,
}

_MIN_VOLUME: float = 1.0
_END_OF_TIME = datetime(2099, 12, 31, tzinfo=timezone.utc)


class DukascopyLiveStream:
    """Live M1 bar feed from Dukascopy via background thread + queue."""

    def __init__(self, symbol: str) -> None:
        instrument = _INSTRUMENT_MAP.get(symbol)
        if instrument is None:
            raise ValueError(
                f"Unsupported symbol: {symbol}. "
                f"Supported: {list(_INSTRUMENT_MAP.keys())}"
            )

        self.symbol = symbol
        self._instrument = instrument
        self._stop = threading.Event()
        self._queue: queue.Queue[Bar] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._reconnect_count: int = 0
        self._max_reconnects: int = 10
        self._reconnect_delay: float = 5.0
        self._stop_event: asyncio.Event | None = None
        self._bars_since_reconnect: int = 0
        self._reset_reconnect_after: int = 60
        self._last_bar_time: datetime | None = None

    # ── Connection ─────────────────────────────────────────────────────────
    async def _handle_thread_failure(self) -> None:
        """Sleep with backoff, then restart the fetch thread. Raises if max retries exceeded."""
        self._reconnect_count += 1
        if self._reconnect_count > self._max_reconnects:
            with self._error_lock:
                err = self._error
            raise RuntimeError(
                f"Dukascopy live fetch failed after {self._max_reconnects} reconnects "
                f"for {self.symbol}: {err}"
            )

        delay = min(self._reconnect_delay * (2 ** (self._reconnect_count - 1)), 120)
        with self._error_lock:
            err = self._error
        logger.warning(
            {
                "event": "DUKASCOPY_LIVE_RECONNECTING",
                "symbol": self.symbol,
                "attempt": self._reconnect_count,
                "delay_seconds": delay,
                "error": str(err),
            }
        )
        with self._error_lock:
            self._error = None
        self._bars_since_reconnect = 0
        if self._stop_event is not None:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        else:
            await asyncio.sleep(delay)
        if not self._stop.is_set():
            self._thread = threading.Thread(target=self._run_fetch_loop, daemon=True)
            self._thread.start()

    def connect(self) -> bool:
        """Start the background fetch thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_fetch_loop, daemon=True)
        self._thread.start()
        logger.info({"event": "DUKASCOPY_LIVE_CONNECTED", "symbol": self.symbol})
        return True

    def disconnect(self) -> None:
        """Signal the background thread to exit."""
        self._stop.set()
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            logger.info({"event": "DUKASCOPY_LIVE_DISCONNECTED", "symbol": self.symbol})

    # ── Background fetch thread ────────────────────────────────────────────

    def _process_dataframe(self, df: Any) -> None:
        """Extract bars from a DataFrame and enqueue only new ones."""
        for idx, row in df.iterrows():
            bar = self._df_row_to_bar(idx, row)
            if self._last_bar_time is not None and bar.timestamp <= self._last_bar_time:
                continue
            self._last_bar_time = bar.timestamp
            self._queue.put(bar)

    def _run_fetch_loop(self) -> None:
        """Blocking loop: iterate live_fetch and push bars to the queue."""
        start = self._last_bar_time or datetime(
            datetime.now(timezone.utc).year,
            datetime.now(timezone.utc).month,
            datetime.now(timezone.utc).day,
            tzinfo=timezone.utc,
        )
        try:
            iterator = dukascopy_python.live_fetch(
                self._instrument,
                1,  # 1-minute interval
                dukascopy_python.TIME_UNIT_MIN,
                OFFER_SIDE_BID,
                start,
                _END_OF_TIME,
            )
            for df in iterator:
                if self._stop.is_set():
                    break
                if df is not None and not df.empty:
                    self._process_dataframe(df)
        except Exception as exc:
            with self._error_lock:
                self._error = exc
            logger.error(
                {
                    "event": "DUKASCOPY_LIVE_FETCH_ERROR",
                    "symbol": self.symbol,
                    "error": str(exc),
                }
            )
        finally:
            if not self._stop.is_set():
                with self._error_lock:
                    if self._error is None:
                        self._error = RuntimeError("Fetch loop exited unexpectedly")
                        logger.warning(
                            {
                                "event": "DUKASCOPY_LIVE_FETCH_EXITED",
                                "symbol": self.symbol,
                            }
                        )

    # ── DataFrame row → Bar ────────────────────────────────────────────────

    def _df_row_to_bar(self, idx: Any, row: Any) -> Bar:
        """Convert a DataFrame row to a Bar object."""
        timestamp = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        volume = max(float(row.get("volume", _MIN_VOLUME)), _MIN_VOLUME)  # type: ignore[arg-type]
        return Bar(
            timestamp=timestamp,
            symbol=self.symbol,
            open=float(row["open"]),  # type: ignore[arg-type]
            high=float(row["high"]),  # type: ignore[arg-type]
            low=float(row["low"]),  # type: ignore[arg-type]
            close=float(row["close"]),  # type: ignore[arg-type]
            volume=volume,
            is_complete=True,
            timeframe=Timeframe.M1,
        )

    # ── Async bar generator ────────────────────────────────────────────────

    async def _check_error_and_reconnect(self) -> bool:
        """If the fetch thread has failed, trigger reconnection. Returns True if handled."""
        with self._error_lock:
            has_error = self._error is not None
        if has_error:
            await self._handle_thread_failure()
            return True
        return False

    def _on_bar_delivered(self) -> None:
        """Track successful bar delivery; reset reconnect counter once after sustained recovery."""
        self._bars_since_reconnect += 1
        if (
            self._reconnect_count > 0
            and self._bars_since_reconnect >= self._reset_reconnect_after
        ):
            logger.info(
                {
                    "event": "DUKASCOPY_LIVE_RECOVERY_CONFIRMED",
                    "symbol": self.symbol,
                    "bars_delivered": self._bars_since_reconnect,
                }
            )
            self._reconnect_count = 0

    async def subscribe(self) -> AsyncGenerator[Bar, None]:
        """Yield completed M1 bars. Reconnects on thread failure with backoff.

        Raises:
            RuntimeError: If the fetch thread fails more than ``_max_reconnects``
                consecutive times without a sustained recovery.

        Example::

            stream = DukascopyLiveStream(symbol="EUR_USD")
            stream.connect()
            try:
                async for bar in stream.subscribe():
                    engine.process_bar(bar)
            except RuntimeError:
                logger.error("Live stream exhausted all reconnect attempts")
            finally:
                stream.disconnect()
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(
                f"subscribe() called before connect() for {self.symbol}. "
                "Call stream.connect() first."
            )

        loop = asyncio.get_event_loop()
        self._stop_event = asyncio.Event()
        logger.info({"event": "DUKASCOPY_LIVE_SUBSCRIBE", "symbol": self.symbol})

        while not self._stop.is_set():
            if await self._check_error_and_reconnect():
                continue

            try:
                bar: Bar = await loop.run_in_executor(
                    None, lambda: self._queue.get(block=True, timeout=1.0)
                )
                self._on_bar_delivered()
                yield bar
            except queue.Empty:
                await self._check_error_and_reconnect()
                continue

        logger.info({"event": "DUKASCOPY_LIVE_SUBSCRIBE_END", "symbol": self.symbol})
