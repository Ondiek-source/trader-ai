"""
stream.py — OANDA v20 tick streaming.

Opens a PricingStream for all configured pairs. Each tick is:
  - Appended to an in-memory buffer per pair (flushed to storage every N ticks)
  - Put onto a shared asyncio-compatible Queue for downstream consumers

On stream disconnect: log warning, wait 2 s, reconnect (infinite retry).
Buffer is never lost on reconnect — ticks remain in self._buffers until flushed.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    import oandapyV20
    import oandapyV20.endpoints.pricing as pricing
    from oandapyV20 import API
    from oandapyV20.exceptions import StreamTerminated, V20Error
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    logger_import = logging.getLogger(__name__)
    logger_import.warning("oandapyV20 not installed — stream.py will run in stub mode.")

logger = logging.getLogger(__name__)


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


class OANDAStream:
    """
    Streams live tick prices from OANDA for all configured pairs.

    Thread model:
      - A single background thread runs _stream_loop().
      - Ticks are buffered per pair and flushed to storage every flush_size ticks.
      - Each tick is also placed on tick_queue for the signal generator to consume.
    """

    def __init__(
        self,
        token: str,
        account_id: str,
        environment: str,
        pairs: list[str],
        storage,
        flush_size: int = 500,
        tick_queue: queue.Queue | None = None,
    ) -> None:
        self._token = token
        self._account_id = account_id
        self._environment = environment  # "practice" | "live"
        self._pairs = pairs
        self._storage = storage
        self._flush_size = flush_size
        self._tick_queue: queue.Queue = tick_queue or queue.Queue(maxsize=100_000)

        # Per-pair in-memory buffers — never cleared on disconnect
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
        """Start the streaming thread (non-blocking)."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True, name="oanda-stream")
        self._thread.start()
        logger.info({"event": "stream_started", "pairs": self._pairs})

    def stop(self) -> None:
        """Signal the streaming thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        # Final flush of any remaining buffered ticks
        self._flush_all()
        logger.info({"event": "stream_stopped"})

    def force_flush(self) -> None:
        """Flush all in-memory buffers to storage immediately."""
        self._flush_all()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _stream_loop(self) -> None:
        """Infinite loop: stream ticks, reconnect on any failure."""
        reconnect_delay = 2.0
        while not self._stop_event.is_set():
            try:
                self._run_stream()
                reconnect_delay = 2.0  # reset on clean exit
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    {
                        "event": "stream_disconnected",
                        "error": str(exc),
                        "reconnect_in_seconds": reconnect_delay,
                        "buffered_ticks": sum(len(v) for v in self._buffers.values()),
                    }
                )
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)

    def _run_stream(self) -> None:
        """Open one streaming session. Returns when stream ends or errors."""
        if not OANDA_AVAILABLE:
            logger.warning({"event": "stream_stub_mode", "message": "oandapyV20 unavailable — emitting synthetic ticks"})
            self._stub_stream()
            return

        client = API(access_token=self._token, environment=self._environment)
        instruments = ",".join(self._pairs)
        params = {"instruments": instruments}
        request = pricing.PricingStream(accountID=self._account_id, params=params)

        logger.info({"event": "stream_connected", "environment": self._environment, "pairs": self._pairs})

        for response in client.request(request):
            if self._stop_event.is_set():
                break
            self._handle_message(response)

    def _handle_message(self, msg: dict) -> None:
        """Process a single streaming message from OANDA."""
        msg_type = msg.get("type", "")
        if msg_type == "HEARTBEAT":
            return  # ignore heartbeats
        if msg_type != "PRICE":
            logger.debug({"event": "stream_unknown_msg_type", "type": msg_type})
            return

        try:
            pair = msg["instrument"]
            ts_str = msg.get("time", "")
            ts = pd.Timestamp(ts_str, tz="UTC") if ts_str else pd.Timestamp.utcnow()

            bids = msg.get("bids", [{}])
            asks = msg.get("asks", [{}])
            bid = float(bids[0].get("price", 0)) if bids else 0.0
            ask = float(asks[0].get("price", 0)) if asks else 0.0
            spread = round(ask - bid, 6)

            tick = Tick(
                pair=pair,
                timestamp=ts.to_pydatetime(),
                bid=bid,
                ask=ask,
                spread=spread,
            )
            self._ingest_tick(tick)

        except (KeyError, IndexError, ValueError) as exc:
            logger.warning({"event": "tick_parse_error", "error": str(exc), "raw": str(msg)[:200]})

    def _ingest_tick(self, tick: Tick) -> None:
        """Buffer tick, flush if threshold reached, put onto queue."""
        with self._buffer_lock:
            self._buffers[tick.pair].append(tick.to_dict())
            self._ticks_received += 1

            if len(self._buffers[tick.pair]) >= self._flush_size:
                self._flush_pair(tick.pair)

        # Non-blocking put — drop if queue is full (backpressure protection)
        try:
            self._tick_queue.put_nowait(tick)
        except queue.Full:
            logger.warning({"event": "tick_queue_full", "pair": tick.pair, "dropped": True})

    def _flush_pair(self, pair: str) -> None:
        """Must be called with self._buffer_lock held (or equivalent)."""
        buf = self._buffers[pair]
        if not buf:
            return
        df = pd.DataFrame(buf)
        self._storage.append_ticks(pair, df)
        logger.debug({"event": "ticks_flushed", "pair": pair, "count": len(buf)})
        self._buffers[pair] = []

    def _flush_all(self) -> None:
        with self._buffer_lock:
            for pair in self._pairs:
                self._flush_pair(pair)

    def _stub_stream(self) -> None:
        """Generate synthetic ticks for testing when OANDA is unavailable."""
        import random
        mid_prices = {"EUR_USD": 1.0850, "GBP_USD": 1.2650, "USD_JPY": 149.50, "XAU_USD": 2020.0}
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
            time.sleep(0.1)


# ── asyncio-compatible wrapper ─────────────────────────────────────────────────

async def run_stream(stream: OANDAStream) -> None:
    """Run stream.start() in an executor so it doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stream.start)


async def get_tick_async(stream: OANDAStream, timeout: float = 1.0) -> Tick | None:
    """Non-blocking async tick consumer. Returns None on timeout."""
    loop = asyncio.get_running_loop()
    try:
        tick = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: stream.tick_queue.get(timeout=timeout)),
            timeout=timeout + 0.5,
        )
        return tick
    except (asyncio.TimeoutError, Exception):
        return None
