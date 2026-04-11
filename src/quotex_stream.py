"""
quotex_stream.py — Quotex OTC tick streaming via pyquotex.

Drop-in alternative to :class:`~stream.TwelveDataStream`.  Polls Quotex for
live OTC prices and emits :class:`~stream.Tick` objects on the same shared
queue, so the signal generator, feature engineering, and storage pipeline
work unchanged.

Thread model mirrors TwelveDataStream:
    - Background thread runs an asyncio event loop.
    - Each price poll becomes a :class:`~stream.Tick`.
    - Ticks are buffered per pair and flushed to storage every N ticks.
    - On disconnect: logs warning, waits with backoff, reconnects forever.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from twelveticks_stream import Tick

logger = logging.getLogger(__name__)

# ── Quotex symbol mapping ─────────────────────────────────────────────────────
# Internal format : EUR_USD
# Quotex OTC sub  : EURUSD_otc   (lowercase _otc suffix)
# Webhook format  : EURUSD_otc   (same as subscription)


def _to_quotex_symbol(pair: str) -> str:
    """``EUR_USD`` → ``EURUSD_otc`` (Quotex OTC format)."""
    return pair.replace("_", "") + "_otc"


def _from_quotex_symbol(symbol: str) -> str:
    """``EURUSD_otc`` → ``EUR_USD``"""
    base = symbol.replace("_otc", "").replace("_OTC", "")
    if len(base) == 6:
        return f"{base[:3]}_{base[3:]}"
    return base


# ── Main stream class ─────────────────────────────────────────────────────────


class QuotexStream:
    """
    Streams live OTC prices from Quotex via pyquotex.

    Args:
        client: An authenticated ``Quotex`` instance from ``pyquotex.stable_api``.
        pairs: List of internal-format pairs (``["EUR_USD", ...]``).
        storage: :class:`~storage.StorageManager` instance for persistence.
        flush_size: Number of ticks to buffer before auto-flush to blob.
        tick_queue: Optional shared queue; created internally if ``None``.
        poll_interval: Seconds between price polls (default 1.0).
    """

    MAX_RETRIES: int = 5
    BACKOFF_BASE: float = 3.0

    def __init__(
        self,
        client: Any,
        pairs: list[str],
        storage: Any,
        flush_size: int = 500,
        tick_queue: queue.Queue[Tick] | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self._client = client
        self._pairs = pairs
        self._storage = storage
        self._flush_size = flush_size
        self._tick_queue: queue.Queue[Tick] = tick_queue or queue.Queue(maxsize=100_000)
        self._poll_interval = poll_interval

        self._buffers: dict[str, list[dict]] = {p: [] for p in pairs}
        self._buffer_lock = threading.Lock()
        self._last_price: dict[str, float] = {}

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks_received = 0
        self._connected = False

        logger.info(
            {
                "event": "quotex_stream_initialized",
                "class": "QuotexStream",
                "pairs": self._pairs,
                "poll_interval": self._poll_interval,
            }
        )

    # ── Properties (match TwelveDataStream interface) ──────────────────────────

    @property
    def tick_queue(self) -> queue.Queue[Tick]:
        """Shared queue of :class:`~stream.Tick` objects for the signal generator."""
        return self._tick_queue

    @property
    def ticks_received(self) -> int:
        """Total ticks received since start (across all pairs)."""
        return self._ticks_received

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background streaming thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="quotex-stream"
        )
        self._thread.start()
        logger.info(
            {
                "event": "stream_started",
                "provider": "Quotex",
                "pairs": self._pairs,
                "poll_interval": self._poll_interval,
            }
        )

    def stop(self) -> None:
        """Signal the stream to stop, wait for the thread, then flush remaining ticks."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._flush_all()
        logger.info({"event": "stream_stopped", "provider": "Quotex"})

    def force_flush(self) -> None:
        """Flush all buffered ticks to storage immediately."""
        self._flush_all()

    # ── Thread entry point ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Run an asyncio event loop inside the background thread."""
        asyncio.run(self._async_stream_loop())

    async def _async_stream_loop(self) -> None:
        """Outer reconnect loop — reconnects with exponential backoff."""
        reconnect_delay = self.BACKOFF_BASE
        while not self._stop_event.is_set():
            try:
                await self._connect()
                await self._poll_loop()
                reconnect_delay = self.BACKOFF_BASE
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.error(
                    {
                        "event": "quotex_disconnected",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "reconnect_in_seconds": reconnect_delay,
                    }
                )
                self._connected = False
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)

    # ── Connection ─────────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        """Connect and authenticate with Quotex."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                check, msg = await self._client.connect()
                if check:
                    self._connected = True
                    logger.info({"event": "quotex_connected", "message": msg})
                    return
                raise ConnectionError(msg)
            except Exception as exc:
                if attempt == self.MAX_RETRIES:
                    raise
                wait = self.BACKOFF_BASE * attempt
                logger.warning(
                    {
                        "event": "quotex_connect_retry",
                        "attempt": attempt,
                        "max": self.MAX_RETRIES,
                        "error": str(exc),
                        "retry_in": wait,
                    }
                )
                await asyncio.sleep(wait)

        raise ConnectionError("Quotex connection exhausted all retries")

    # ── Polling loop ───────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Poll each pair's OTC symbol forever.

        Uses the confirmed format: EURUSD_otc.
        """
        symbols = {p: _to_quotex_symbol(p) for p in self._pairs}

        logger.info(
            {
                "event": "quotex_polling_started",
                "symbols": symbols,
                "poll_interval": self._poll_interval,
            }
        )

        while not self._stop_event.is_set():
            for pair, symbol in symbols.items():
                if self._stop_event.is_set():
                    return

                try:
                    price = await self._get_price(symbol)
                    if price is None:
                        logger.warning(
                            {
                                "event": "quotex_no_price",
                                "pair": pair,
                                "symbol": symbol,
                            }
                        )
                        continue

                    tick = self._make_tick(pair, price)
                    self._ingest_tick(tick)

                except Exception as exc:
                    logger.error(
                        {
                            "event": "quotex_poll_error",
                            "pair": pair,
                            "symbol": symbol,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )

                await asyncio.sleep(self._poll_interval)

    # ── Price fetching ─────────────────────────────────────────────────────────

    async def _get_price(self, symbol: str) -> float | None:
        """
        Get current price for *symbol*.

        Primary: get_realtime_price (returns dict — we extract the price).
        Fallback: get_candles (returns list of dicts — we take the last close).
        """

        # ── Primary: get_realtime_price ─────────────────────────────────────
        try:
            result = await self._client.get_realtime_price(symbol)

            # Log the raw result once so we can see the schema
            if not hasattr(self, "_logged_raw_price"):
                logger.info(
                    {
                        "event": "quotex_raw_realtime_price",
                        "symbol": symbol,
                        "result_type": type(result).__name__,
                        "result_preview": str(result)[:400],
                    }
                )
                self._logged_raw_price = True

            if result is None:
                return None

            # Result is a dict — extract the price value
            if isinstance(result, dict):
                for key in ("price", "value", "close", "last", "bid", "ask"):
                    if key in result:
                        val = float(result[key])
                        if val > 0:
                            return val
                logger.warning(
                    {
                        "event": "quotex_price_dict_no_key",
                        "symbol": symbol,
                        "keys": list(result.keys())[:10],
                    }
                )
                return None

            # Result is a plain number
            if isinstance(result, (int, float)):
                val = float(result)
                return val if val > 0 else None

            logger.warning(
                {
                    "event": "quotex_price_unknown_type",
                    "symbol": symbol,
                    "type": type(result).__name__,
                }
            )
            return None

        except AttributeError:
            logger.error({"event": "quotex_no_realtime_price_method", "symbol": symbol})
        except Exception as exc:
            logger.error(
                {
                    "event": "quotex_realtime_price_error",
                    "symbol": symbol,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

        # ── Fallback: get_candles ───────────────────────────────────────────
        try:
            candles = await self._client.get_candles(symbol, 60, 1)

            if not candles or len(candles) == 0:
                return None

            candle = candles[-1]
            if not isinstance(candle, dict):
                return None

            close = float(candle.get("close", 0))
            return close if close > 0 else None

        except Exception as exc:
            logger.error(
                {
                    "event": "quotex_candles_error",
                    "symbol": symbol,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            return None

    # ── Tick construction ──────────────────────────────────────────────────────

    def _make_tick(self, pair: str, price: float) -> Tick:
        """Build a :class:`~stream.Tick` from a single price value."""
        spread = self._estimate_spread(pair, price)
        bid = round(price - spread / 2, 5)
        ask = round(price + spread / 2, 5)

        self._last_price[pair] = price

        return Tick(
            pair=pair,
            timestamp=datetime.now(timezone.utc),
            bid=bid,
            ask=ask,
            spread=round(spread, 6),
        )

    @staticmethod
    def _estimate_spread(pair: str, price: float) -> float:
        """Estimate a typical OTC spread for *pair*."""
        if "JPY" in pair:
            return 0.015
        elif "XAU" in pair or "GOLD" in pair:
            return 0.30
        elif "XAG" in pair or "SILVER" in pair:
            return 0.020
        else:
            return 0.00015

    # ── Ingestion / buffering (identical to TwelveDataStream) ──────────────────

    def _ingest_tick(self, tick: Tick) -> None:
        """Buffer a tick and optionally flush to storage."""
        with self._buffer_lock:
            self._buffers[tick.pair].append(tick.to_dict())
            self._ticks_received += 1

            if self._ticks_received % 100 == 0:
                logger.info(
                    {
                        "event": "tick_milestone",
                        "ticks": self._ticks_received,
                        "provider": "Quotex",
                        "pair": tick.pair,
                    }
                )

            if len(self._buffers[tick.pair]) >= self._flush_size:
                self._flush_pair(tick.pair)

        try:
            self._tick_queue.put_nowait(tick)
        except queue.Full:
            logger.warning(
                {"event": "tick_queue_full", "pair": tick.pair, "dropped": True}
            )

    def _flush_pair(self, pair: str) -> None:
        """Flush one pair's buffer to storage.  Must hold ``_buffer_lock``."""
        buf = self._buffers[pair]
        if not buf:
            return
        df = pd.DataFrame(buf)
        self._storage.append_ticks(pair, df)
        logger.debug(
            {
                "event": "ticks_flushed",
                "provider": "Quotex",
                "pair": pair,
                "count": len(buf),
            }
        )
        self._buffers[pair] = []

    def _flush_all(self) -> None:
        """Flush every pair's buffer.  Acquires ``_buffer_lock``."""
        with self._buffer_lock:
            for pair in list(self._buffers.keys()):
                self._flush_pair(pair)
