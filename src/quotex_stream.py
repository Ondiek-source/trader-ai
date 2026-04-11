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

from twelveticks_stream import Tick  # reuse the exact same dataclass

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
        self._working_symbols: dict[str, str] = {}

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
                reconnect_delay = self.BACKOFF_BASE  # reset on clean exit
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
                    await self._log_client_capabilities()
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

    # ── Client capability discovery ────────────────────────────────────────────

    async def _log_client_capabilities(self) -> None:
        """Probe the client to see what methods and data are available."""
        client_methods = [m for m in dir(self._client) if not m.startswith("_")]
        price_methods = [m for m in client_methods if "price" in m.lower()]
        candle_methods = [m for m in client_methods if "candle" in m.lower()]
        asset_methods = [m for m in client_methods if "asset" in m.lower()]

        logger.info(
            {
                "event": "quotex_client_capabilities",
                "price_methods": price_methods,
                "candle_methods": candle_methods,
                "asset_methods": asset_methods,
                "total_methods": len(client_methods),
            }
        )

        # Try to get assets list
        for method_name in [
            "get_all_assets",
            "get_available_asset",
            "get_all_asset_name",
        ]:
            method = getattr(self._client, method_name, None)
            if method is None:
                continue
            try:
                result = await method()
                if result:
                    sample = str(result)[:300]
                    logger.info(
                        {
                            "event": "quotex_assets_probe",
                            "method": method_name,
                            "count": (
                                len(result) if hasattr(result, "__len__") else "unknown"
                            ),
                            "sample": sample,
                        }
                    )
                    break
            except Exception as exc:
                logger.debug(
                    {
                        "event": "quotex_assets_probe_failed",
                        "method": method_name,
                        "error": str(exc),
                    }
                )

    # ── Symbol format discovery ────────────────────────────────────────────────

    async def _discover_symbol_format(self) -> None:
        """
        Try all symbol formats and log which one returns a price.

        Formats tried per pair:
            1. EURUSD_otc   (confirmed Quotex format)
            2. EURUSD-OTC   (hyphen variant)
            3. EURUSD       (plain, no suffix)
        """
        all_formats = [
            ("lowercase_otc", {p: _to_quotex_symbol(p) for p in self._pairs}),
            ("hyphen_OTC", {p: p.replace("_", "") + "-OTC" for p in self._pairs}),
            ("plain", {p: p.replace("_", "") for p in self._pairs}),
        ]

        self._working_symbols = {}

        for pair in self._pairs:
            for label, symbol_map in all_formats:
                symbol = symbol_map[pair]
                try:
                    price = await self._get_price(symbol)
                    if price is not None and price > 0:
                        self._working_symbols[pair] = symbol
                        logger.info(
                            {
                                "event": "quotex_symbol_format_found",
                                "pair": pair,
                                "symbol": symbol,
                                "format": label,
                                "price": price,
                            }
                        )
                        break
                    else:
                        logger.warning(
                            {
                                "event": "quotex_symbol_probe_none",
                                "pair": pair,
                                "symbol": symbol,
                                "format": label,
                                "price": price,
                            }
                        )
                except Exception as exc:
                    logger.warning(
                        {
                            "event": "quotex_symbol_probe_exception",
                            "pair": pair,
                            "symbol": symbol,
                            "format": label,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )

        if not self._working_symbols:
            logger.error(
                {
                    "event": "quotex_no_working_symbols",
                    "pairs": self._pairs,
                    "formats_tried": [label for label, _ in all_formats],
                    "note": (
                        "None of the symbol formats returned a price. "
                        "Check quotex_all_price_methods_failed logs for details."
                    ),
                }
            )
        else:
            failed_pairs = [p for p in self._pairs if p not in self._working_symbols]
            logger.info(
                {
                    "event": "quotex_symbol_discovery_complete",
                    "working": self._working_symbols,
                    "failed": failed_pairs if failed_pairs else "none",
                }
            )

    # ── Polling loop ───────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Discover working symbols, then poll forever."""
        await self._discover_symbol_format()

        if not self._working_symbols:
            logger.error(
                {
                    "event": "quotex_polling_aborted",
                    "reason": "no working symbols found",
                    "pairs": self._pairs,
                    "note": (
                        "Stream will sleep 30s and retry symbol discovery. "
                        "This usually means the symbol format is wrong or "
                        "the Quotex API is not returning price data for these assets."
                    ),
                }
            )
            await asyncio.sleep(30)
            return  # exits _poll_loop, outer loop reconnects

        logger.info(
            {
                "event": "quotex_polling_started",
                "symbols": self._working_symbols,
                "poll_interval": self._poll_interval,
            }
        )

        # Main poll loop
        while not self._stop_event.is_set():
            for pair, symbol in self._working_symbols.items():
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
        Try every known pyquotex method to get a price for *symbol*.

        Returns the first successful price, or None if all methods fail.
        Logs detailed errors for each failed method.
        """
        errors = []

        # ── Method 1: get_realtime_price ────────────────────────────────────
        try:
            price = await self._client.get_realtime_price(symbol)
            if price is not None:
                val = float(price)
                if val > 0:
                    logger.info(
                        {
                            "event": "quotex_price_hit",
                            "method": "get_realtime_price",
                            "symbol": symbol,
                            "price": val,
                        }
                    )
                    return val
                errors.append(f"get_realtime_price returned {val} (non-positive)")
            else:
                errors.append("get_realtime_price returned None")
        except AttributeError:
            errors.append("get_realtime_price: method not found on client")
        except Exception as exc:
            errors.append(f"get_realtime_price: {type(exc).__name__}: {exc}")

        # ── Method 2: get_candles ───────────────────────────────────────────
        try:
            candles = await self._client.get_candles(symbol, 60, 1)
            if candles and len(candles) > 0:
                candle = candles[-1]
                close = float(candle.get("close", 0))
                if close > 0:
                    logger.info(
                        {
                            "event": "quotex_price_hit",
                            "method": "get_candles",
                            "symbol": symbol,
                            "price": close,
                            "candle_keys": (
                                list(candle.keys())
                                if isinstance(candle, dict)
                                else type(candle).__name__
                            ),
                        }
                    )
                    return close
                errors.append(f"get_candles close={close} (non-positive)")
            else:
                errors.append(
                    f"get_candles returned {type(candles).__name__}: {str(candles)[:100]}"
                )
        except AttributeError:
            errors.append("get_candles: method not found on client")
        except Exception as exc:
            errors.append(f"get_candles: {type(exc).__name__}: {exc}")

        # ── Method 3: get_price ─────────────────────────────────────────────
        try:
            price = await self._client.get_price(symbol)
            if price is not None:
                val = float(price)
                if val > 0:
                    logger.info(
                        {
                            "event": "quotex_price_hit",
                            "method": "get_price",
                            "symbol": symbol,
                            "price": val,
                        }
                    )
                    return val
                errors.append(f"get_price returned {val} (non-positive)")
            else:
                errors.append("get_price returned None")
        except AttributeError:
            errors.append("get_price: method not found on client")
        except Exception as exc:
            errors.append(f"get_price: {type(exc).__name__}: {exc}")

        # ── Method 4: get_realtime_candles ──────────────────────────────────
        try:
            candles = await self._client.get_realtime_candles(symbol, 60, 1)
            if candles and len(candles) > 0:
                candle = candles[-1]
                close = float(candle.get("close", 0))
                if close > 0:
                    logger.info(
                        {
                            "event": "quotex_price_hit",
                            "method": "get_realtime_candles",
                            "symbol": symbol,
                            "price": close,
                        }
                    )
                    return close
                errors.append(f"get_realtime_candles close={close} (non-positive)")
            else:
                errors.append(
                    f"get_realtime_candles returned {type(candles).__name__}: {str(candles)[:100]}"
                )
        except AttributeError:
            errors.append("get_realtime_candles: method not found on client")
        except Exception as exc:
            errors.append(f"get_realtime_candles: {type(exc).__name__}: {exc}")

        # ── All methods failed ──────────────────────────────────────────────
        logger.error(
            {
                "event": "quotex_all_price_methods_failed",
                "symbol": symbol,
                "errors": errors,
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
