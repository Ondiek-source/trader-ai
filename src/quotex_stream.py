"""
quotex_stream.py — Quotex OTC tick streaming via pyquotex.

Drop-in alternative to :class:`~twelveticks_stream.TwelveDataStream`.
Polls Quotex for live OTC prices and emits :class:`~twelveticks_stream.Tick`
objects on the same shared queue, so the signal generator, feature
engineering, and storage pipeline work unchanged.

Thread model mirrors TwelveDataStream:
    - Background thread runs an asyncio event loop.
    - Each price poll becomes a :class:`~twelveticks_stream.Tick`.
    - Ticks are buffered per pair and flushed to storage every N ticks.
    - On disconnect: logs warning, waits with backoff, reconnects forever.

pyquotex API (from official docs):
    - Quotex(email, password, lang="en")
    - await client.connect() -> (bool, str)
    - await client.start_realtime_price(asset, period) -> None
    - await client.get_realtime_price(asset) -> list[dict] | None
      (each dict has "time" and "price" keys)
    - await client.close() -> None
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


def _to_quotex_symbol(pair: str) -> str:
    """
    Convert internal pair format to Quotex OTC subscription symbol.

    Args:
        pair: Internal format like ``"EUR_USD"``.

    Returns:
        Quotex OTC symbol like ``"EURUSD_otc"``.

    Examples:
        >>> _to_quotex_symbol("EUR_USD")
        'EURUSD_otc'
        >>> _to_quotex_symbol("XAU_USD")
        'XAUUSD_otc'
    """
    return pair.replace("_", "").upper() + "_otc"


# ── Main stream class ─────────────────────────────────────────────────────────


class QuotexStream:
    """
    Streams live OTC prices from Quotex via pyquotex.

    Drop-in replacement for :class:`~twelveticks_stream.TwelveDataStream`
    with the same public interface: ``tick_queue``, ``ticks_received``,
    ``start()``, ``stop()``.

    Args:
        client: An authenticated ``Quotex`` instance from
            ``pyquotex.stable_api``.
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
        self._client: Any = client
        self._pairs: list[str] = pairs
        self._storage: Any = storage
        self._flush_size: int = flush_size
        self._tick_queue: queue.Queue[Tick] = tick_queue or queue.Queue(maxsize=100_000)
        self._poll_interval: float = poll_interval

        self._buffers: dict[str, list[dict[str, Any]]] = {p: [] for p in pairs}
        self._buffer_lock: threading.Lock = threading.Lock()
        self._last_price: dict[str, float] = {}

        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks_received: int = 0
        self._connected: bool = False

        logger.info(
            {
                "event": "quotex_stream_initialized",
                "class": "QuotexStream",
                "pairs": self._pairs,
                "poll_interval": self._poll_interval,
                "symbols": {p: _to_quotex_symbol(p) for p in self._pairs},
            }
        )

    # ── Properties (match TwelveDataStream interface) ──────────────────────────

    @property
    def tick_queue(self) -> queue.Queue[Tick]:
        """
        Shared queue of :class:`~twelveticks_stream.Tick` objects.

        Consumed by the signal generator in :func:`main.signal_task`.
        """
        return self._tick_queue

    @property
    def ticks_received(self) -> int:
        """Total ticks received since start (across all pairs)."""
        return self._ticks_received

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the background streaming thread.

        The thread runs its own asyncio event loop and polls Quotex
        prices at :attr:`_poll_interval` for each pair.
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="quotex-stream",
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
        """Signal the stream to stop, wait for the thread, then flush."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
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
        """
        Outer reconnect loop — reconnects with exponential backoff
        on any exception.
        """
        reconnect_delay: float = self.BACKOFF_BASE
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
        """
        Connect and authenticate with Quotex.

        Retries up to :attr:`MAX_RETRIES` times with linear backoff.

        Raises:
            ConnectionError: If all retries are exhausted.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                check, msg = await self._client.connect()
                if check:
                    self._connected = True
                    logger.info({"event": "quotex_connected", "message": msg})
                    return
                raise ConnectionError(str(msg))
            except Exception as exc:
                if attempt == self.MAX_RETRIES:
                    raise
                wait: float = self.BACKOFF_BASE * attempt
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

    # ── Stream setup ───────────────────────────────────────────────────────────

    async def _start_realtime_streams(self, symbols: dict[str, str]) -> None:
        """
        Start real-time price streams for all symbols.

        Per pyquotex docs, ``start_realtime_price`` is async and must be
        awaited.  Without this, ``get_realtime_price`` returns ``None``.

        Args:
            symbols: Mapping of internal pair to Quotex symbol
                (e.g. ``{"EUR_USD": "EURUSD_otc"}``).
        """
        for pair, symbol in symbols.items():
            if self._stop_event.is_set():
                return
            try:
                await self._client.start_realtime_price(symbol, 60)
                logger.info(
                    {
                        "event": "quotex_realtime_stream_started",
                        "pair": pair,
                        "symbol": symbol,
                    }
                )
            except Exception as exc:
                logger.error(
                    {
                        "event": "quotex_realtime_stream_failed",
                        "pair": pair,
                        "symbol": symbol,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

    # ── Polling loop ───────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Start real-time streams, then poll for prices forever.

        On each tick:
            1. Call :meth:`_get_price` for each symbol.
            2. Construct a :class:`~twelveticks_stream.Tick`.
            3. Ingest into buffer and queue.
        """
        symbols: dict[str, str] = {p: _to_quotex_symbol(p) for p in self._pairs}

        # Start streams — get_realtime_price returns None without this.
        await self._start_realtime_streams(symbols)

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
                    price: float | None = await self._get_price(symbol)
                    if price is None:
                        continue

                    tick: Tick = self._make_tick(pair, price)
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

        Requires :meth:`_start_realtime_streams` to have been called first.

        Per pyquotex docs, ``get_realtime_price`` returns a **list of dicts**,
        each with ``"time"`` and ``"price"`` keys::

            prices = await client.get_realtime_price(asset)
            last_price = prices[-1]
            print(f"Time: {last_price['time']} Price: {last_price['price']}")

        Args:
            symbol: Quotex symbol (e.g. ``"EURUSD_otc"``).

        Returns:
            Price as a positive float, or ``None`` if unavailable.
        """
        try:
            result: Any = await self._client.get_realtime_price(symbol)

            if result is None:
                return None

            # List response — official format from pyquotex docs
            if isinstance(result, list):
                if not result:
                    return None
                last_entry: dict[str, Any] = result[-1]
                price_raw: Any = last_entry.get("price")
                if price_raw is not None:
                    try:
                        val: float = float(price_raw)
                        if val > 0:
                            return val
                    except (ValueError, TypeError):
                        logger.debug(
                            {
                                "event": "quotex_price_parse_error",
                                "symbol": symbol,
                                "price_raw": price_raw,
                            }
                        )
                return None

            # Dict response — some versions may return a single dict
            if isinstance(result, dict):
                if not result:
                    return None
                for key in ("price", "value", "close", "last"):
                    if key in result:
                        try:
                            val = float(result[key])
                            if val > 0:
                                return val
                        except (ValueError, TypeError):
                            continue
                return None

            # Plain number response
            if isinstance(result, (int, float)):
                val = float(result)
                return val if val > 0 else None

            # Unexpected type — log for debugging
            logger.debug(
                {
                    "event": "quotex_price_unparsed",
                    "symbol": symbol,
                    "type": type(result).__name__,
                    "repr": str(result)[:200],
                }
            )
            return None

        except Exception as exc:
            logger.error(
                {
                    "event": "quotex_realtime_price_error",
                    "symbol": symbol,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            return None

    # ── Tick construction ──────────────────────────────────────────────────────

    def _make_tick(self, pair: str, price: float) -> Tick:
        """
        Build a :class:`~twelveticks_stream.Tick` from a single price value.

        Quotex only provides a mid price, so bid/ask are estimated
        using :meth:`_estimate_spread`.

        Args:
            pair: Internal pair format (``"EUR_USD"``).
            price: Mid price from Quotex.

        Returns:
            A fully populated :class:`~twelveticks_stream.Tick`.
        """
        spread: float = self._estimate_spread(pair, price)
        bid: float = round(price - spread / 2, 5)
        ask: float = round(price + spread / 2, 5)

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
        """
        Estimate a typical OTC spread for *pair*.

        Args:
            pair: Internal pair format.
            price: Current mid price (unused but kept for API consistency).

        Returns:
            Estimated spread as a float.
        """
        if "JPY" in pair:
            return 0.015
        elif "XAU" in pair or "GOLD" in pair:
            return 0.30
        elif "XAG" in pair or "SILVER" in pair:
            return 0.020
        else:
            return 0.00015

    # ── Ingestion / buffering ──────────────────────────────────────────────────

    def _ingest_tick(self, tick: Tick) -> None:
        """
        Buffer a tick and optionally flush to storage.

        Puts the tick on :attr:`_tick_queue` for the signal generator.
        Flushes the pair's buffer to storage when it reaches
        :attr:`_flush_size`.

        Args:
            tick: Fresh tick to ingest.
        """
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
                {
                    "event": "tick_queue_full",
                    "pair": tick.pair,
                    "dropped": True,
                }
            )

    def _flush_pair(self, pair: str) -> None:
        """
        Flush one pair's buffer to storage.

        Must be called while holding :attr:`_buffer_lock`.

        Args:
            pair: Internal pair format to flush.
        """
        buf: list[dict[str, Any]] = self._buffers[pair]
        if not buf:
            return
        df: pd.DataFrame = pd.DataFrame(buf)
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
        """Flush every pair's buffer.  Acquires :attr:`_buffer_lock`."""
        with self._buffer_lock:
            for pair in list(self._buffers.keys()):
                self._flush_pair(pair)
