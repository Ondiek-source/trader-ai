"""
stream.py — Twelve Data WebSocket tick streaming.

Connects to wss://ws.twelvedata.com/v1/quotes/price and subscribes to all
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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Pair format helpers ────────────────────────────────────────────────────────
# Internal format:  EUR_USD
# Twelve Data format: EUR/USD


def _to_td_symbol(pair: str) -> str:
    """EUR_USD → EUR/USD"""
    return pair.replace("_", "/")


def _from_td_symbol(symbol: str) -> str:
    """EUR/USD → EUR_USD"""
    return symbol.replace("/", "_")


@dataclass
class Tick:
    pair: str
    timestamp: datetime
    bid: float
    ask: float
    spread: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "timestamp": self.timestamp,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
        }


class TwelveDataStream:
    """
    Streams live tick prices from Twelve Data WebSocket API.

    Thread model:
        - A background thread runs an asyncio event loop for the WebSocket.
        - Ticks are buffered per pair and flushed to storage every flush_size ticks.
        - Each tick is also placed on tick_queue for the signal generator to consume.
    """

    WS_URL = "wss://ws.twelvedata.com/v1/quotes/price"

    def __init__(
        self,
        api_key: str,
        pairs: list[str],
        storage: Any,
        flush_size: int = 500,
        tick_queue: queue.Queue | None = None,
    ) -> None:
        self._api_key = api_key
        self._pairs = pairs
        self._storage = storage
        self._flush_size = flush_size
        self._tick_queue: queue.Queue = tick_queue or queue.Queue(maxsize=100_000)

        self._buffers: dict[str, list[dict]] = {p: [] for p in pairs}
        self._buffer_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks_received = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def tick_queue(self) -> queue.Queue:
        return self._tick_queue

    @property
    def ticks_received(self) -> int:
        return self._ticks_received

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="twelvedata-stream"
        )
        self._thread.start()
        logger.info(
            {"event": "stream_started", "provider": "TwelveData", "pairs": self._pairs}
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._flush_all()
        logger.info({"event": "stream_stopped"})

    def force_flush(self) -> None:
        self._flush_all()

    # ── Thread entry point ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Run an asyncio event loop inside the background thread."""
        asyncio.run(self._async_stream_loop())

    async def _async_stream_loop(self) -> None:
        """Outer reconnect loop."""
        reconnect_delay = 2.0
        while not self._stop_event.is_set():
            try:
                await self._connect_and_stream()
                reconnect_delay = 2.0
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
                reconnect_delay = min(reconnect_delay * 2, 60.0)

    async def _connect_and_stream(self) -> None:
        """Open one WebSocket session and stream until error or stop."""
        try:
            from websockets.asyncio.client import connect as _ws_connect  # type: ignore[import-unresolved]
        except ImportError:
            try:
                from websockets import connect as _ws_connect  # type: ignore[import-unresolved]
            except ImportError:
                logger.warning(
                    {
                        "event": "websockets_not_installed",
                        "message": "pip install websockets>=12.0 — falling back to stub mode",
                    }
                )
                await self._stub_stream()
                return

        url = f"{self.WS_URL}?apikey={self._api_key}"
        symbols = ",".join(_to_td_symbol(p) for p in self._pairs)

        async with _ws_connect(url) as ws:
            # Subscribe to all pairs
            subscribe_msg = json.dumps(
                {
                    "action": "subscribe",
                    "params": {"symbols": symbols},
                }
            )
            await ws.send(subscribe_msg)
            logger.info({"event": "stream_connected", "symbols": symbols})

            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    self._handle_message(json.loads(raw))
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await ws.ping()

    def _handle_message(self, msg: dict) -> None:
        """Process a single WebSocket message from Twelve Data."""
        event = msg.get("event", "")

        if event == "heartbeat":
            return
        if event in ("subscribe-status", "subscribe"):
            logger.debug({"event": "stream_subscribe_status", "msg": msg})
            return
        if event != "price":
            logger.debug(
                {"event": "stream_unknown_msg", "type": event, "keys": list(msg.keys())}
            )
            return

        try:
            symbol = msg.get("symbol", "")
            pair = _from_td_symbol(symbol)

            # Twelve Data sends ask/bid as strings
            ask = float(msg.get("ask") or msg.get("price") or 0)
            bid = float(msg.get("bid") or ask)
            spread = round(ask - bid, 6)

            # Timestamp: "2024-01-01 12:00:00" UTC or unix
            ts_raw = msg.get("timestamp")
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
                {"event": "tick_parse_error", "error": str(exc), "raw": str(msg)[:200]}
            )

    # ── Ingestion / buffering ──────────────────────────────────────────────────

    def _ingest_tick(self, tick: Tick) -> None:
        with self._buffer_lock:
            if tick.pair not in self._buffers:
                self._buffers[tick.pair] = []
            self._buffers[tick.pair].append(tick.to_dict())
            self._ticks_received += 1

            if len(self._buffers[tick.pair]) >= self._flush_size:
                self._flush_pair(tick.pair)

        try:
            self._tick_queue.put_nowait(tick)
        except queue.Full:
            logger.warning(
                {"event": "tick_queue_full", "pair": tick.pair, "dropped": True}
            )

    def _flush_pair(self, pair: str) -> None:
        buf = self._buffers.get(pair, [])
        if not buf:
            return
        df = pd.DataFrame(buf)
        self._storage.append_ticks(pair, df)
        logger.debug({"event": "ticks_flushed", "pair": pair, "count": len(buf)})
        self._buffers[pair] = []

    def _flush_all(self) -> None:
        with self._buffer_lock:
            for pair in list(self._buffers.keys()):
                self._flush_pair(pair)

    # ── Stub fallback ──────────────────────────────────────────────────────────

    async def _stub_stream(self) -> None:
        """Emit synthetic ticks when websockets library is unavailable."""
        import random

        mid_prices = {
            "EUR_USD": 1.0850,
            "GBP_USD": 1.2650,
            "USD_JPY": 149.50,
            "XAU_USD": 2020.0,
        }
        while not self._stop_event.is_set():
            for pair in self._pairs:
                mid = mid_prices.get(pair, 1.0)
                mid += random.gauss(0, mid * 0.0001)
                mid_prices[pair] = mid
                spread = mid * 0.0001
                tick = Tick(
                    pair=pair,
                    timestamp=datetime.now(timezone.utc),
                    bid=round(mid - spread / 2, 5),
                    ask=round(mid + spread / 2, 5),
                    spread=round(spread, 6),
                )
                self._ingest_tick(tick)
            await asyncio.sleep(0.1)


# ── Backwards-compat alias (used in main.py) ───────────────────────────────────
OANDAStream = TwelveDataStream
