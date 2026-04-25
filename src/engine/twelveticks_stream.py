"""
twelveticks_stream.py — Twelve Data WebSocket tick streaming.

Connects to ``wss://ws.twelvedata.com/v1/quotes/price`` and subscribes to all
configured pairs. Each tick is:

- Appended to an in-memory buffer per pair (flushed to storage every N ticks)
- Put onto a shared queue for the signal generator to consume

On disconnect: logs warning, waits with backoff, reconnects (infinite retry).
Buffer is never lost on reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from ml_engine.model import Tick as ModelTick
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Pair format helpers ────────────────────────────────────────────────────────
# Internal format   : EUR_USD
# Twelve Data format: EUR/USD


def _to_td_symbol(pair: str) -> str:
    """
    Convert internal pair format to Twelve Data symbol.

    Args:
        pair: Internal format like ``"EUR_USD"``.

    Returns:
        Twelve Data symbol like ``"EUR/USD"``.
    """
    return pair.replace("_", "/")


def _from_td_symbol(symbol: str) -> str:
    """
    Convert Twelve Data symbol to internal pair format.

    Args:
        symbol: Twelve Data symbol like ``"EUR/USD"``.

    Returns:
        Internal format like ``"EUR_USD"``.
    """
    return symbol.replace("/", "_")


# ── Tick dataclass ─────────────────────────────────────────────────────────────


@dataclass
class Tick:
    """
    A single price update for one currency pair.

    Attributes:
        pair: Internal pair format (``"EUR_USD"``).
        timestamp: UTC datetime of the tick.
        bid: Bid price.
        ask: Ask price.
        spread: Ask minus bid.
    """

    pair: str
    timestamp: datetime
    bid: float
    ask: float
    spread: float

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a dict suitable for DataFrame construction.

        Returns:
            Dict with keys ``pair``, ``timestamp``, ``bid``, ``ask``,
            ``spread``.
        """
        return {
            "pair": self.pair,
            "timestamp": self.timestamp,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
        }


# ── Main stream class ─────────────────────────────────────────────────────────


class TwelveDataStream:
    """
    Streams live tick prices from Twelve Data WebSocket API.

    Thread model:
        - A background thread runs an asyncio event loop for the WebSocket.
        - Ticks are buffered per pair and flushed to storage every
          *flush_size* ticks.
        - Each tick is also placed on *tick_queue* for the signal generator.

    On disconnect, the stream reconnects with exponential backoff
    (2 s → 4 s → 8 s → ... → 60 s cap).  The buffer is never lost
    on reconnect.

    Args:
        api_key: Twelve Data API key.
        pairs: List of internal-format pairs (``["EUR_USD", ...]``).
        storage: :class:`~storage.StorageManager` instance for persistence.
        flush_size: Number of ticks to buffer before auto-flush to blob.
        tick_queue: Optional shared queue; created internally if ``None``.
    """

    WS_URL: str = "wss://ws.twelvedata.com/v1/quotes/price"
    MAX_RECONNECT_DELAY: float = 60.0
    WS_RECV_TIMEOUT: float = 30.0
    TICK_MILESTONE_INTERVAL: int = 100

    def __init__(
        self,
        api_key: str,
        pairs: list[str],
        storage: Any,
        flush_size: int = 500,
        tick_queue: queue.Queue[Tick] | None = None,
    ) -> None:
        self._api_key: str = api_key
        self._pairs: list[str] = pairs
        self._storage: Any = storage
        self._flush_size: int = flush_size
        self._tick_queue: queue.Queue[Tick] = tick_queue or queue.Queue(maxsize=100_000)

        self._buffers: dict[str, list[dict[str, Any]]] = {p: [] for p in pairs}
        self._buffer_lock: threading.Lock = threading.Lock()

        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks_received: int = 0

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def tick_queue(self) -> queue.Queue[Tick]:
        """
        Shared queue of :class:`Tick` objects.

        Consumed by the signal generator in :func:`main.signal_task`.
        """
        return self._tick_queue

    @property
    def ticks_received(self) -> int:
        """Total ticks received since start (across all pairs)."""
        return self._ticks_received

    # ── Async interface for live.py compatibility ─────────────────────────────

    async def connect(self) -> bool:
        """
        Start the background WebSocket thread.

        Bridges to the async interface expected by LiveEngine.create(),
        which calls ``await self._stream.connect()`` before the main loop.
        """
        await asyncio.sleep(1)
        self.start()
        return True

    async def subscribe(self):
        """
        Async generator that yields ModelTick objects from the tick queue.

        Bridges the threaded TwelveDataStream to the async interface
        expected by LiveEngine.run():
            ``async for tick in self._stream.subscribe():``

        Converts TwelveDataStream.Tick → ml_engine.model.Tick on the fly.
        """

        while not self._stop_event.is_set():
            try:
                # Non-blocking get with short timeout so we can check stop_event
                tick = self._tick_queue.get(timeout=0.5)
            except Exception:
                # queue.Empty or timeout — yield control and retry
                await asyncio.sleep(0.1)
                continue

            yield ModelTick(
                timestamp=tick.timestamp,
                symbol=tick.pair,
                bid=tick.bid,
                ask=tick.ask,
                source="TWELVE",
            )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the background streaming thread.

        The thread runs its own asyncio event loop and connects to the
        Twelve Data WebSocket.
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="twelvedata-stream",
        )
        self._thread.start()
        logger.info(
            {
                "event": "stream_started",
                "provider": "TwelveData",
                "pairs": self._pairs,
            }
        )

    def stop(self) -> None:
        """Signal the stream to stop, wait for the thread, then flush."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._flush_all()
        logger.info({"event": "stream_stopped"})

    def force_flush(self) -> None:
        """Flush all buffered ticks to storage immediately."""
        self._flush_all()

    # ── Thread entry point ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Run an asyncio event loop inside the background thread."""
        asyncio.run(self._async_stream_loop())

    async def _async_stream_loop(self) -> None:
        """
        Outer reconnect loop — reconnects with exponential backoff
        on any exception.
        """
        reconnect_delay: float = 2.0
        while not self._stop_event.is_set():
            try:
                await self._connect_and_stream()
                reconnect_delay = 2.0  # reset on clean disconnect
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    {
                        "event": "stream_disconnected",
                        "error": str(exc),
                        "reconnect_in_seconds": reconnect_delay,
                    }
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

    async def _connect_and_stream(self) -> None:
        """
        Open one WebSocket session and stream until error or stop.

        Tries ``websockets.asyncio.client.connect`` first (v12+),
        falls back to ``websockets.connect`` (v10/v11), then falls
        back to stub mode if neither is installed.
        """
        _ws_connect: Any
        try:
            from websockets.asyncio.client import connect as _ws_connect  # type: ignore[import-unresolved]
        except ImportError:
            try:
                from websockets import connect as _ws_connect  # type: ignore[import-unresolved]
            except ImportError:
                logger.warning(
                    {
                        "event": "websockets_not_installed",
                        "message": (
                            "pip install websockets>=12.0 — "
                            "falling back to stub mode"
                        ),
                    }
                )
                await self._stub_stream()
                return

        url: str = f"{self.WS_URL}?apikey={self._api_key}"
        symbols: str = ",".join(_to_td_symbol(p) for p in self._pairs)

        async with _ws_connect(url) as ws:
            subscribe_msg: str = json.dumps(
                {
                    "action": "subscribe",
                    "params": {"symbols": symbols},
                }
            )
            await ws.send(subscribe_msg)
            logger.info({"event": "stream_connected", "symbols": symbols})

            while not self._stop_event.is_set():
                try:
                    raw: str = await asyncio.wait_for(
                        ws.recv(), timeout=self.WS_RECV_TIMEOUT
                    )
                    self._handle_message(json.loads(raw))
                except asyncio.TimeoutError:
                    await ws.ping()

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """
        Process a single WebSocket message from Twelve Data.

        Heartbeats and subscription confirmations are logged and ignored.
        Price events are parsed into :class:`Tick` objects and ingested.

        Args:
            msg: Parsed JSON dict from the WebSocket.
        """
        event: str = msg.get("event", "")

        if event == "heartbeat":
            return
        if event in ("subscribe-status", "subscribe"):
            logger.debug({"event": "stream_subscribe_status", "msg": msg})
            return
        if event != "price":
            logger.debug(
                {
                    "event": "stream_unknown_msg",
                    "type": event,
                    "keys": list(msg.keys()),
                }
            )
            return

        try:
            symbol: str = msg.get("symbol", "")
            pair: str = _from_td_symbol(symbol)

            ask: float = float(msg.get("ask") or msg.get("price") or 0)
            bid: float = float(msg.get("bid") or ask)
            spread: float = round(ask - bid, 6)

            ts_raw: Any = msg.get("timestamp")
            ts: datetime
            if ts_raw:
                try:
                    ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
                except (ValueError, TypeError):
                    ts = pd.Timestamp(str(ts_raw), tz="UTC").to_pydatetime()
            else:
                ts = datetime.now(timezone.utc)

            tick = Tick(pair=pair, timestamp=ts, bid=bid, ask=ask, spread=spread)
            self._ingest_tick(tick)

        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                {
                    "event": "tick_parse_error",
                    "error": str(exc),
                    "raw": str(msg)[:200],
                }
            )

    # ── Ingestion / buffering ──────────────────────────────────────────────────

    def _ingest_tick(self, tick: Tick) -> None:
        """
        Buffer a tick and optionally flush to storage.

        The tick is also placed on the shared queue **outside** the lock
        to avoid holding the buffer lock while the queue operation runs.

        Args:
            tick: Fresh tick to ingest.
        """
        with self._buffer_lock:
            self._buffers[tick.pair].append(tick.to_dict())
            self._ticks_received += 1

            if self._ticks_received % self.TICK_MILESTONE_INTERVAL == 0:
                logger.info(
                    {
                        "event": "tick_milestone",
                        "ticks": self._ticks_received,
                        "pair": tick.pair,
                    }
                )

            if len(self._buffers[tick.pair]) >= self._flush_size:
                self._flush_pair(tick.pair)

        # Put on queue OUTSIDE the lock — avoids holding the lock if the
        # queue is full and put_nowait raises.
        try:
            self._tick_queue.put_nowait(tick)
        except queue.Full:
            logger.warning(
                {
                    "event": "tick_queue_full",
                    "pair": tick.pair,
                    "dropped": True,
                }
            )

    def _flush_pair(self, pair: str) -> None:
        """
        Flush one pair's buffer to storage.

        Must be called with :attr:`_buffer_lock` held.

        Args:
            pair: Internal pair format to flush.
        """
        buf: list[dict[str, Any]] = self._buffers[pair]
        if not buf:
            return
        df: pd.DataFrame = pd.DataFrame(buf)
        self._storage.append_ticks(pair, df)
        logger.debug({"event": "ticks_flushed", "pair": pair, "count": len(buf)})
        self._buffers[pair] = []

    def _flush_all(self) -> None:
        """Flush every pair's buffer.  Acquires :attr:`_buffer_lock`."""
        with self._buffer_lock:
            for pair in self._buffers.keys():
                self._flush_pair(pair)

    # ── Stub fallback ──────────────────────────────────────────────────────────

    async def _stub_stream(self) -> None:
        """
        Emit synthetic ticks when the websockets library is unavailable.

        Uses random walk on mid prices at ~10 ticks/sec.  Useful for
        local development and testing without an API key.
        """
        import random

        mid_prices: dict[str, float] = {
            "EUR_USD": 1.0850,
            "GBP_USD": 1.2650,
            "USD_JPY": 149.50,
            "XAU_USD": 2020.0,
        }
        while not self._stop_event.is_set():
            for pair in self._pairs:
                mid: float = mid_prices.get(pair, 1.0)
                mid += random.gauss(0, mid * 0.0001)
                mid_prices[pair] = mid
                spread: float = mid * 0.0001
                tick = Tick(
                    pair=pair,
                    timestamp=datetime.now(timezone.utc),
                    bid=round(mid - spread / 2, 5),
                    ask=round(mid + spread / 2, 5),
                    spread=round(spread, 6),
                )
                self._ingest_tick(tick)
            await asyncio.sleep(0.1)
