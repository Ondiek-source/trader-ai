"""
quotex_reader.py — Reads closed trade results from Quotex account.

Uses pyquotex (cleitonleonel/pyquotex) unofficial WebSocket API.
Module: pyquotex.stable_api (NOT quotexapi — package renamed in v1.0)
Install: pip install git+https://github.com/cleitonleonel/pyquotex.git

CONFIRMED API (from pyquotex source / research):
    - Quotex(email, password, lang="en")
    - await client.connect() → (bool, reason_str)
    - client.change_account("PRACTICE" | "REAL")
    - await client.get_balance() → float
    - Direction strings: "call" (UP/buy) | "put" (DOWN/sell)
    - Asset format: "EURUSD_otc" (OTC, 24/7) | "EURUSD" (market hours)

UNCONFIRMED (needs live introspection — raw logs will reveal):
    - Trade history method name (get_history / get_closed_deals / etc.)
    - WebSocket trade-close message schema field names

Strategy (no trade ID available — our bot places trades, not us):
    1. Balance-delta: monitor account balance; change at expiry = trade result.
    2. WebSocket introspection: log ALL messages from client.api for schema discovery.
    3. Time-windowed match: at expiry+2s, scan buffered WS messages for
    the asset+direction that matches our pending signal.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Asset name mapping ─────────────────────────────────────────────────────────
# Quotex uses "EURUSD_otc" on OTC (weekends / off-hours) and "EURUSD" on live
# The _otc suffix is tried first; both are checked on match.

PAIR_TO_ASSETS: dict[str, list[str]] = {
    "EUR_USD": ["EURUSD", "EURUSD_otc"],
    "GBP_USD": ["GBPUSD", "GBPUSD_otc"],
    "USD_JPY": ["USDJPY", "USDJPY_otc"],
    "XAU_USD": ["XAUUSD", "XAUUSD_otc"],
}

# Direction: our internal → Quotex API
DIRECTION_TO_QUOTEX: dict[str, str] = {"UP": "call", "DOWN": "put"}
QUOTEX_TO_DIRECTION: dict[str, str] = {"call": "UP", "put": "DOWN"}

# ── Library import ─────────────────────────────────────────────────────────────
_QuotexClient: Any = None  # replaced at import time if library is present
QUOTEX_LIB_AVAILABLE = False
try:
    from pyquotex.stable_api import Quotex as _QuotexClient  # type: ignore[import-unresolved]

    QUOTEX_LIB_AVAILABLE = True
except Exception as _qx_err:
    import sys as _sys

    print(
        f"[quotex_reader] pyquotex import failed: {type(_qx_err).__name__}: {_qx_err}",
        file=_sys.stderr,
    )


class QuotexReader:
    """
    Connects to Quotex WebSocket and reads closed trade results.

    Two complementary detection strategies run in parallel:
        A) Balance-delta: reliable signal that *some* trade closed.
        B) WebSocket stream: richer data (asset, direction, profit).

    Both are correlated against pending signals using expiry time window.

    Args:
        email: Quotex account email.
        password: Quotex account password.
        practice_mode: ``True`` for demo account, ``False`` for real.
    """

    RESULT_BUFFER_SECONDS = 2  # wait after signal expiry before checking
    BALANCE_POLL_INTERVAL = 1.5  # seconds between balance checks
    MAX_WS_MESSAGES = 300  # ring buffer size for raw WS messages
    MAX_RECONNECT_DELAY = 120  # seconds
    MAX_HISTORY_ATTEMPTS = 3  # cap history method calls per resolve

    def __init__(self, email: str, password: str, practice_mode: bool = True) -> None:
        self._email = email
        self._password = password
        self._practice_mode = practice_mode
        self._account_type = "PRACTICE" if practice_mode else "REAL"

        self._client: Any = None
        self._connected = False
        self._balance: float = 0.0
        self._prev_balance: float = 0.0

        # {signal_id → {signal, expiry_time, balance_before, attempts, resolved}}
        self._pending: dict[str, dict[str, Any]] = {}

        # Ring buffer of raw WebSocket messages for introspection
        self._ws_messages: list[dict[str, Any]] = []

        # Results ready for consumption by main loop
        self._result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Track which history methods actually work (populated at runtime)
        self._working_history_methods: list[str] = []

    # ── Connection ─────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Establish WebSocket connection to Quotex."""
        if not QUOTEX_LIB_AVAILABLE:
            return False
        if not self._email or not self._password:
            logger.warning({"event": "quotex_no_credentials"})
            return False

        try:
            self._client = _QuotexClient(
                email=self._email,
                password=self._password,
                lang="en",
            )
            ok, reason = await self._client.connect()
            if ok:
                self._connected = True
                # Switch to correct account type
                if hasattr(self._client, "change_account"):
                    result = self._client.change_account(self._account_type)
                    if asyncio.iscoroutine(result):
                        await result

                self._balance = await self._safe_get_balance()
                self._prev_balance = self._balance

                mode_tag = "[PRACTICE MODE]" if self._practice_mode else "[LIVE MODE]"
                logger.info(
                    {
                        "event": "quotex_connected",
                        "mode": mode_tag,
                        "account_type": self._account_type,
                        "balance": self._balance,
                        "reason": str(reason),
                    }
                )

                # Log all available client methods (aids schema discovery)
                self._introspect_client()
                return True
            else:
                logger.error({"event": "quotex_connect_failed", "reason": str(reason)})
                return False

        except Exception as exc:
            logger.error({"event": "quotex_connect_exception", "error": str(exc)})
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Gracefully close the Quotex WebSocket connection."""
        self._connected = False
        if self._client:
            try:
                if hasattr(self._client, "close"):
                    result = self._client.close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                pass

    def _introspect_client(self) -> None:
        """Log all public client methods — helps discover history/result APIs."""
        try:
            methods = [m for m in dir(self._client) if not m.startswith("_")]
            logger.info(
                {
                    "event": "quotex_client_methods_discovered",
                    "methods": methods,
                    "note": "Look for get_history / get_closed_deals / get_result / "
                    "closed_trades in this list",
                }
            )
            # Also log api sub-object if it exists
            if hasattr(self._client, "api"):
                api_methods = [
                    m for m in dir(self._client.api) if not m.startswith("_")
                ]
                logger.info(
                    {
                        "event": "quotex_api_object_methods",
                        "api_methods": api_methods,
                    }
                )
        except Exception as exc:
            logger.debug({"event": "introspect_failed", "error": str(exc)})

    # ── Signal registration ────────────────────────────────────────────────────

    def register_pending(
        self, signal_id: str, signal: dict[str, Any], expiry_time: datetime
    ) -> None:
        """
        Register a signal that needs result matching at *expiry_time*.

        Args:
            signal_id: UUID from :class:`SignalOrchestrator`.
            signal: Internal signal metadata dict (has ``pair``, ``direction``,
                ``confidence``, ``expiry_seconds``, ``fired_at``).
            expiry_time: UTC datetime when the option expires.
        """
        # Snapshot balance NOW so the delta is accurate later
        balance_snapshot = self._balance
        self._pending[signal_id] = {
            "signal": signal,
            "expiry_time": expiry_time,
            "balance_before": balance_snapshot,
            "attempts": 0,
            "resolved": False,
        }
        logger.debug(
            {
                "event": "pending_registered",
                "signal_id": signal_id,
                "pair": signal.get("pair"),
                "expiry_at": expiry_time.isoformat(),
                "balance_snapshot": balance_snapshot,
            }
        )

    # ── Main poll loop ─────────────────────────────────────────────────────────

    async def poll_results(self) -> None:
        """
        Background task: runs forever.

        - Polls balance every :attr:`BALANCE_POLL_INTERVAL` seconds.
        - At expiry + :attr:`RESULT_BUFFER_SECONDS`, attempts to match a result
          to each pending signal using balance-delta + WS message scan.
        - Puts matched results onto :attr:`_result_queue`.
        """
        balance_task: asyncio.Task[None] | None = None

        while True:
            if not self._connected:
                # Cancel balance monitor, reconnect, then restart it
                if balance_task is not None and not balance_task.done():
                    balance_task.cancel()
                    try:
                        await balance_task
                    except asyncio.CancelledError:
                        pass
                    balance_task = None

                await self._reconnect()
                if not self._connected:
                    await asyncio.sleep(2.0)
                    continue

                balance_task = asyncio.create_task(self._balance_monitor())

            # Ensure balance task is running
            if balance_task is None or balance_task.done():
                balance_task = asyncio.create_task(self._balance_monitor())

            now = datetime.now(timezone.utc)

            # Find pending signals that are due for resolution
            due = {
                sid: data
                for sid, data in self._pending.items()
                if not data["resolved"]
                and now
                >= data["expiry_time"]
                + timedelta(seconds=self.RESULT_BUFFER_SECONDS)
                and data["attempts"] < self.MAX_HISTORY_ATTEMPTS
            }

            for signal_id, data in due.items():
                data["attempts"] += 1
                result = await self._resolve_result(signal_id, data)
                if result:
                    data["resolved"] = True
                    await self._result_queue.put(result)
                elif data["attempts"] >= self.MAX_HISTORY_ATTEMPTS:
                    data["resolved"] = True
                    logger.warning(
                        {
                            "event": "result_unresolved",
                            "signal_id": signal_id,
                            "pair": data["signal"].get("pair"),
                            "message": "No matching trade found after "
                            f"{self.MAX_HISTORY_ATTEMPTS} attempts. "
                            "Check [QUOTEX_RAW_MSG] logs for schema hints.",
                        }
                    )

            # Clean up resolved entries older than 60 s
            self._pending = {
                sid: d
                for sid, d in self._pending.items()
                if not d["resolved"]
                or (now - d["expiry_time"]).total_seconds() < 60
            }

            await asyncio.sleep(0.5)

    # ── Balance monitor ────────────────────────────────────────────────────────

    async def _balance_monitor(self) -> None:
        """
        Continuously polls balance to detect trade closes via delta.

        Runs as a child task of :meth:`poll_results`.  Exits when
        :attr:`_connected` becomes ``False``.
        """
        # Set initial balance without logging a delta
        if self._balance == 0.0:
            self._balance = await self._safe_get_balance()
            self._prev_balance = self._balance

        while self._connected:
            await asyncio.sleep(self.BALANCE_POLL_INTERVAL)
            try:
                new_balance = await self._safe_get_balance()
                if abs(new_balance - self._balance) > 0.001:
                    delta = new_balance - self._balance
                    logger.info(
                        {
                            "event": "balance_delta_detected",
                            "prev": self._balance,
                            "new": new_balance,
                            "delta": round(delta, 4),
                            "likely_outcome": "win" if delta > 0 else "loss",
                        }
                    )
                    self._balance = new_balance
            except Exception as exc:
                logger.debug(
                    {"event": "balance_poll_error", "error": str(exc)}
                )

    async def _safe_get_balance(self) -> float:
        """Fetch balance with a 5-second timeout; returns cached value on failure."""
        if not self._client or not self._connected:
            return self._balance
        try:
            bal = await asyncio.wait_for(
                self._client.get_balance(), timeout=5.0
            )
            return float(bal) if bal is not None else self._balance
        except Exception:
            return self._balance

    # ── Result resolution ──────────────────────────────────────────────────────

    async def _resolve_result(
        self, signal_id: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Try all strategies to resolve a trade result for this signal.

        Returns a result dict or ``None`` if unresolved.
        """
        signal = data["signal"]
        pair = signal.get("pair", "")
        direction = signal.get("direction", "UP")
        expiry_time = data["expiry_time"]
        balance_before = data["balance_before"]

        # ── Strategy 1: WebSocket message scan ────────────────────────────────
        ws_result = self._scan_ws_messages(pair, direction, expiry_time)
        if ws_result:
            return {
                **ws_result,
                "signal_id": signal_id,
                "detection": "websocket",
            }

        # ── Strategy 2: Balance delta (fast, no API calls) ───────────────────
        current = await self._safe_get_balance()
        delta = current - balance_before
        if abs(delta) > 0.001:
            outcome = "win" if delta > 0 else "loss"
            logger.info(
                {
                    "event": "result_from_balance_delta",
                    "signal_id": signal_id,
                    "pair": pair,
                    "delta": round(delta, 4),
                    "outcome": outcome,
                    "note": "Balance delta match — pair/direction inferred",
                }
            )
            return self._make_result(
                signal_id, pair, direction, outcome, abs(delta)
            )

        # ── Strategy 3: Try history methods (expensive — last resort) ─────────
        if self._working_history_methods or data["attempts"] >= 2:
            api_result = await self._try_history_methods(
                pair, direction, expiry_time
            )
            if api_result:
                return {
                    **api_result,
                    "signal_id": signal_id,
                    "detection": "api_history",
                }

        return None

    # ── WebSocket message scanning ─────────────────────────────────────────────

    def _scan_ws_messages(
        self, pair: str, direction: str, expiry_time: datetime
    ) -> dict[str, Any] | None:
        """
        Scan buffered WebSocket messages for a trade result matching
        this signal's asset, direction, and time window.
        """
        asset_candidates = {
            a.upper()
            for a in PAIR_TO_ASSETS.get(pair, [pair.replace("_", "")])
        }
        quotex_direction = DIRECTION_TO_QUOTEX.get(direction, "call")
        window_start = expiry_time - timedelta(seconds=30)
        window_end = expiry_time + timedelta(seconds=10)

        for msg in reversed(self._ws_messages):
            try:
                parsed = self._parse_ws_message(msg)
                if parsed is None:
                    continue

                # Asset match
                msg_asset = str(parsed.get("asset", "")).upper()
                if not any(
                    candidate in msg_asset or msg_asset in candidate
                    for candidate in asset_candidates
                ):
                    continue

                # Direction match (skip if direction unknown in message)
                msg_direction = str(parsed.get("direction", "")).lower()
                if msg_direction and msg_direction != quotex_direction:
                    continue

                # Time proximity — only check if close_time was actually found
                msg_time = parsed.get("close_time")
                if msg_time is not None:
                    if not (window_start <= msg_time <= window_end):
                        continue

                outcome = str(parsed.get("result", "")).lower()
                payout = float(parsed.get("profit", 0.0))
                if outcome in ("win", "loss", "draw"):
                    return self._make_result("", pair, direction, outcome, payout)

            except Exception:
                continue
        return None

    def _parse_ws_message(self, msg: Any) -> dict[str, Any] | None:
        """
        Parse a raw WebSocket message into a normalised dict.

        Field names here are best-effort.  Run the system and check
        ``[QUOTEX_RAW_MSG]`` logs to verify/correct these.
        """
        if not isinstance(msg, dict):
            return None

        # Log raw for schema discovery
        msg_type = msg.get(
            "type", msg.get("name", msg.get("status", "unknown"))
        )
        logger.debug(
            {
                "event": "[QUOTEX_RAW_MSG]",
                "type": msg_type,
                "keys": list(msg.keys())[:15],
                "preview": str(msg)[:300],
            }
        )

        result_candidates: dict[str, Any] = {
            "profit": msg.get(
                "profit", msg.get("win_amount", msg.get("payout"))
            ),
            "result": msg.get(
                "result", msg.get("status", msg.get("outcome"))
            ),
            "asset": msg.get(
                "asset", msg.get("symbol", msg.get("active"))
            ),
            "direction": msg.get(
                "direction", msg.get("call_put", msg.get("option_type"))
            ),
            "close_time": None,
        }

        # Parse close time from any known timestamp field
        for time_key in (
            "close_time",
            "closed_at",
            "expiration",
            "exp_time",
            "time",
        ):
            raw_time = msg.get(time_key)
            if raw_time:
                try:
                    result_candidates["close_time"] = datetime.fromtimestamp(
                        float(raw_time), tz=timezone.utc
                    )
                    break
                except Exception:
                    pass

        # Normalise result string
        raw_result = result_candidates["result"]
        if raw_result is not None:
            raw_lower = str(raw_result).lower()
            if raw_lower in ("win", "1", "true", "won"):
                result_candidates["result"] = "win"
            elif raw_lower in ("loss", "lose", "lost", "0", "false"):
                result_candidates["result"] = "loss"
            elif raw_lower in ("draw", "tie", "equal"):
                result_candidates["result"] = "draw"

        # Normalise profit
        raw_profit = result_candidates["profit"]
        if raw_profit is not None:
            try:
                result_candidates["profit"] = float(raw_profit)
            except Exception:
                result_candidates["profit"] = 0.0

        return result_candidates

    # ── History method probing ─────────────────────────────────────────────────

    async def _try_history_methods(
        self,
        pair: str,
        direction: str,
        _expiry_time: datetime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """
        Try known history / result methods on the client object.

        On first success the working method is cached in
        :attr:`_working_history_methods` so future calls skip dead methods.
        """
        if not self._client:
            return None

        asset_candidates = PAIR_TO_ASSETS.get(pair, [pair.replace("_", "")])

        # Use cached working methods if we've found any; otherwise try all
        methods_to_try = (
            self._working_history_methods
            if self._working_history_methods
            else [
                "get_history",
                "get_history_v2",
                "get_closed_deals",
                "get_closed_options",
                "get_result",
                "get_trade_history",
                "get_user_deals",
            ]
        )

        call_count = 0
        for method_name in methods_to_try:
            if not hasattr(self._client, method_name):
                continue
            for asset in asset_candidates:
                try:
                    method = getattr(self._client, method_name)
                    # Try with just asset first (most likely to work)
                    try:
                        raw = await asyncio.wait_for(
                            method(asset), timeout=8.0
                        )
                        call_count += 1
                        logger.info(
                            {
                                "event": "[QUOTEX_RAW_RESPONSE]",
                                "method": method_name,
                                "args": (asset,),
                                "raw_type": type(raw).__name__,
                                "raw_preview": str(raw)[:600],
                            }
                        )
                        result = self._extract_from_history(
                            raw, pair, direction
                        )
                        if result:
                            # Cache this working method
                            if method_name not in self._working_history_methods:
                                self._working_history_methods.append(
                                    method_name
                                )
                                logger.info(
                                    {
                                        "event": "history_method_cached",
                                        "method": method_name,
                                    }
                                )
                            return result
                    except Exception:
                        pass

                    if call_count >= self.MAX_HISTORY_ATTEMPTS:
                        return None

                except Exception as exc:
                    logger.debug(
                        {
                            "event": "history_method_failed",
                            "method": method_name,
                            "error": str(exc),
                        }
                    )

        return None

    def _extract_from_history(
        self, raw: Any, pair: str, direction: str
    ) -> dict[str, Any] | None:
        """Extract a matching trade from a history response of unknown schema."""
        if raw is None:
            return None

        asset_candidates = {
            a.upper() for a in PAIR_TO_ASSETS.get(pair, [])
        }

        def _check_item(item: Any) -> dict[str, Any] | None:
            if not isinstance(item, dict):
                return None
            # Asset check
            item_asset = str(
                item.get(
                    "asset",
                    item.get("symbol", item.get("active", "")),
                )
            ).upper()
            if asset_candidates and not any(
                c in item_asset or item_asset in c
                for c in asset_candidates
            ):
                return None
            # Find profit/result
            profit: float | None = None
            for k in ("profit", "win", "payout", "income", "amount"):
                if k in item:
                    try:
                        profit = float(item[k])
                        break
                    except Exception:
                        pass
            if profit is None:
                return None
            if profit > 0:
                outcome = "win"
            elif profit == 0:
                outcome = "draw"
            else:
                outcome = "loss"
            return self._make_result("", pair, direction, outcome, profit)

        if isinstance(raw, list):
            for item in reversed(raw):
                r = _check_item(item)
                if r:
                    return r
        elif isinstance(raw, dict):
            # Might be wrapped: {"deals": [...], "history": [...]}
            for key in (
                "deals",
                "history",
                "trades",
                "operations",
                "data",
                "items",
            ):
                if key in raw and isinstance(raw[key], list):
                    for item in reversed(raw[key]):
                        r = _check_item(item)
                        if r:
                            return r
            return _check_item(raw)

        return None

    # ── Result construction ────────────────────────────────────────────────────

    def _make_result(
        self,
        signal_id: str,
        pair: str,
        direction: str,
        outcome: str,
        payout: float,
    ) -> dict[str, Any]:
        """Build a standardised result dict for :meth:`SignalOrchestrator.on_result`."""
        return {
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "result": outcome,
            "payout": round(abs(payout), 4),
            "stake": 0.0,
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public consumer ────────────────────────────────────────────────────────

    async def get_result(self, timeout: float = 0.1) -> dict[str, Any] | None:
        """
        Non-blocking read of the next resolved result.

        Returns:
            Result dict or ``None`` if nothing available within *timeout*.
        """
        try:
            return await asyncio.wait_for(
                self._result_queue.get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def add_ws_message(self, msg: dict[str, Any]) -> None:
        """
        Feed a raw WebSocket message into the introspection buffer.

        Call this from your WebSocket message handler::

            reader.add_ws_message(raw_ws_dict)
        """
        self._ws_messages.append(msg)
        if len(self._ws_messages) > self.MAX_WS_MESSAGES:
            self._ws_messages = self._ws_messages[-self.MAX_WS_MESSAGES :]

    def health(self) -> dict[str, Any]:
        """Return a serialisable health snapshot."""
        return {
            "connected": self._connected,
            "balance": self._balance,
            "pending_signals": len(self._pending),
            "ws_messages_buffered": len(self._ws_messages),
            "result_queue_size": self._result_queue.qsize(),
            "working_history_methods": self._working_history_methods,
        }

    # ── Reconnection ──────────────────────────────────────────────────────────

    async def _reconnect(self) -> None:
        """Exponential-backoff reconnection loop."""
        delay = 2.0
        while not self._connected:
            logger.info(
                {"event": "quotex_reconnecting", "delay_seconds": delay}
            )
            await asyncio.sleep(delay)
            try:
                await self.connect()
            except Exception as exc:
                logger.warning(
                    {"event": "quotex_reconnect_error", "error": str(exc)}
                )
            delay = min(delay * 2, self.MAX_RECONNECT_DELAY)


# ── Async runner (supervised task entry point) ─────────────────────────────────


async def run_quotex_reader(reader: QuotexReader) -> None:
    """
    Top-level entry point for :func:`asyncio.create_task`.

    If pyquotex is not installed, sleeps forever without reconnect spam.
    """
    if not QUOTEX_LIB_AVAILABLE:
        logger.info(
            {
                "event": "quotex_reader_disabled",
                "reason": "pyquotex_not_installed",
                "note": "Quotex result reading unavailable.",
            }
        )
        while True:
            await asyncio.sleep(3600)
    else:
        await reader.poll_results()
