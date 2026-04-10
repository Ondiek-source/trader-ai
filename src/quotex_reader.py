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
DIRECTION_TO_QUOTEX = {"UP": "call", "DOWN": "put"}
QUOTEX_TO_DIRECTION = {"call": "UP", "put": "DOWN"}

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
    """

    RESULT_BUFFER_SECONDS = 2  # wait after signal expiry before checking
    BALANCE_POLL_INTERVAL = 1.5  # seconds between balance checks
    MAX_WS_MESSAGES = 300  # ring buffer size for raw WS messages
    MAX_RECONNECT_DELAY = 120  # seconds

    def __init__(self, email: str, password: str, practice_mode: bool = True) -> None:
        self._email = email
        self._password = password
        self._practice_mode = practice_mode
        self._account_type = "PRACTICE" if practice_mode else "REAL"

        self._client: Any = None
        self._connected = False
        self._balance: float = 0.0
        self._prev_balance: float = 0.0

        # {signal_id → {signal, expiry_time, attempts, resolved}}
        self._pending: dict[str, dict] = {}

        # Ring buffer of raw WebSocket messages for introspection
        self._ws_messages: list[dict] = []

        # Results ready for consumption by main loop
        self._result_queue: asyncio.Queue = asyncio.Queue()

    # ── Connection ─────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
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

                mode_tag = "[PRACTICE MODE]" if self._practice_mode else "[LIVE MODE ⚠️]"
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
        if self._client:
            try:
                if hasattr(self._client, "close"):
                    result = self._client.close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                pass
        self._connected = False

    def _introspect_client(self) -> None:
        """Log all public client methods — helps discover history/result APIs."""
        try:
            methods = [m for m in dir(self._client) if not m.startswith("_")]
            logger.info(
                {
                    "event": "quotex_client_methods_discovered",
                    "methods": methods,
                    "note": "Look for get_history / get_closed_deals / get_result / closed_trades in this list",
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
        self, signal_id: str, signal: dict, expiry_time: datetime
    ) -> None:
        """Register a signal that needs result matching at expiry_time."""
        self._pending[signal_id] = {
            "signal": signal,
            "expiry_time": expiry_time,
            "balance_before": self._balance,
            "attempts": 0,
            "resolved": False,
        }
        logger.debug(
            {
                "event": "pending_registered",
                "signal_id": signal_id,
                "pair": signal.get("pair"),
                "expiry_at": expiry_time.isoformat(),
            }
        )

    # ── Main poll loop ─────────────────────────────────────────────────────────

    async def poll_results(self) -> None:
        """
        Background task: runs forever.
        - Polls balance every BALANCE_POLL_INTERVAL seconds.
        - At expiry + RESULT_BUFFER_SECONDS, attempts to match a result
        - to each pending signal using balance-delta + WS message scan.
        - Puts matched results onto self._result_queue.
        """
        balance_task = asyncio.create_task(self._balance_monitor())

        while True:
            if not self._connected:
                balance_task.cancel()
                await self._reconnect()
                balance_task = asyncio.create_task(self._balance_monitor())

            now = datetime.now(timezone.utc)
            due = {
                sid: data
                for sid, data in self._pending.items()
                if not data["resolved"]
                and now
                >= data["expiry_time"] + timedelta(seconds=self.RESULT_BUFFER_SECONDS)
                and data["attempts"] < 3
            }

            for signal_id, data in due.items():
                data["attempts"] += 1
                result = await self._resolve_result(signal_id, data)
                if result:
                    data["resolved"] = True
                    await self._result_queue.put(result)
                elif data["attempts"] >= 3:
                    data["resolved"] = True
                    logger.warning(
                        {
                            "event": "result_unresolved",
                            "signal_id": signal_id,
                            "pair": data["signal"].get("pair"),
                            "message": "No matching trade found after 3 attempts. "
                            "Review [QUOTEX_RAW_MSG] logs to improve _parse_ws_message().",
                        }
                    )

            # Clean up resolved entries older than 60s
            self._pending = {
                sid: d
                for sid, d in self._pending.items()
                if not d["resolved"] or (now - d["expiry_time"]).total_seconds() < 60
            }

            await asyncio.sleep(0.5)

    # ── Balance monitor ────────────────────────────────────────────────────────

    async def _balance_monitor(self) -> None:
        """Continuously polls balance to detect trade closes via delta."""
        # First, set initial balance without logging a delta
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
                logger.debug({"event": "balance_poll_error", "error": str(exc)})

    async def _safe_get_balance(self) -> float:
        if not self._client or not self._connected:
            return self._balance
        try:
            bal = await asyncio.wait_for(self._client.get_balance(), timeout=5.0)
            return float(bal) if bal is not None else self._balance
        except Exception:
            return self._balance

    # ── Result resolution ──────────────────────────────────────────────────────

    async def _resolve_result(self, signal_id: str, data: dict) -> dict | None:
        """
        Try all strategies to resolve a trade result for this signal.
        Returns a result dict or None if unresolved.
        """
        signal = data["signal"]
        pair = signal.get("pair", "")
        direction = signal.get("direction", "UP")
        expiry_time = data["expiry_time"]
        balance_before = data.get("balance_before", self._balance)

        # ── Strategy 1: WebSocket message scan ────────────────────────────────
        ws_result = self._scan_ws_messages(pair, direction, expiry_time)
        if ws_result:
            return {**ws_result, "signal_id": signal_id, "detection": "websocket"}

        # ── Strategy 2: Try known/discoverable history methods ────────────────
        api_result = await self._try_history_methods(pair, direction, expiry_time)
        if api_result:
            return {**api_result, "signal_id": signal_id, "detection": "api_history"}

        # ── Strategy 3: Balance delta (last resort) ───────────────────────────
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
                    "note": "Balance delta match — pair/direction inferred, not confirmed by Quotex",
                }
            )
            return self._make_result(signal_id, pair, direction, outcome, abs(delta))

        return None

    def _scan_ws_messages(
        self, pair: str, direction: str, expiry_time: datetime
    ) -> dict | None:
        """Scan buffered WebSocket messages for a trade result matching this signal."""
        asset_candidates = {
            a.upper() for a in PAIR_TO_ASSETS.get(pair, [pair.replace("_", "")])
        }
        quotex_direction = DIRECTION_TO_QUOTEX.get(direction, "call")
        window_start = expiry_time - timedelta(seconds=30)
        window_end = expiry_time + timedelta(seconds=10)

        for msg in reversed(self._ws_messages):
            try:
                parsed = self._parse_ws_message(msg)
                if parsed is None:
                    continue

                # Check asset match
                msg_asset = str(parsed.get("asset", "")).upper()
                if not any(
                    candidate in msg_asset or msg_asset in candidate
                    for candidate in asset_candidates
                ):
                    continue

                # Check direction match (optional — match on asset+time if direction unknown)
                msg_direction = str(parsed.get("direction", "")).lower()
                if msg_direction and msg_direction not in (quotex_direction, ""):
                    continue

                # Check time proximity
                msg_time = parsed.get("close_time")
                if msg_time and not (window_start <= msg_time <= window_end):
                    continue

                outcome = str(parsed.get("result", "")).lower()
                payout = float(parsed.get("profit", 0.0))
                if outcome in ("win", "loss", "draw"):
                    return self._make_result("", pair, direction, outcome, payout)

            except Exception:
                continue
        return None

    def _parse_ws_message(self, msg: Any) -> dict | None:
        """
        Parse a raw WebSocket message into a normalised dict.
        IMPORTANT: Field names here are based on best-effort research.
        Run the system and check [QUOTEX_RAW_MSG] logs to verify/correct these.
        """
        if not isinstance(msg, dict):
            return None

        # Log raw for schema discovery (first 50 unique message types)
        msg_type = msg.get("type", msg.get("name", msg.get("status", "unknown")))
        logger.debug(
            {
                "event": "[QUOTEX_RAW_MSG]",
                "type": msg_type,
                "keys": list(msg.keys())[:15],
                "preview": str(msg)[:300],
            }
        )

        # Known field patterns (update these as raw logs reveal actual schema):
        result_candidates = {
            "profit": msg.get("profit", msg.get("win_amount", msg.get("payout"))),
            "result": msg.get("result", msg.get("status", msg.get("outcome"))),
            "asset": msg.get("asset", msg.get("symbol", msg.get("active"))),
            "direction": msg.get(
                "direction", msg.get("call_put", msg.get("option_type"))
            ),
            "close_time": None,
        }

        # Parse close time
        for time_key in ("close_time", "closed_at", "expiration", "exp_time", "time"):
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

    async def _try_history_methods(
        self, pair: str, direction: str, _expiry_time: datetime  # noqa: ARG002
    ) -> dict | None:
        """
        Try all history/result methods discovered on the client object.
        Method names discovered via _introspect_client() logs at startup.
        """
        if not self._client:
            return None

        asset_candidates = PAIR_TO_ASSETS.get(pair, [pair.replace("_", "")])

        # Ordered list of method names to try (update from introspection logs)
        history_methods = [
            "get_history",
            "get_history_v2",
            "get_closed_deals",
            "get_closed_options",
            "get_result",
            "get_trade_history",
            "get_user_deals",
        ]

        for method_name in history_methods:
            if not hasattr(self._client, method_name):
                continue
            for asset in asset_candidates:
                try:
                    method = getattr(self._client, method_name)
                    # Try with just asset, or with asset + time offset
                    for call_args in [(asset,), (asset, 10), (asset, 3600)]:
                        try:
                            raw = await asyncio.wait_for(
                                method(*call_args), timeout=8.0
                            )
                            logger.info(
                                {
                                    "event": "[QUOTEX_RAW_RESPONSE - history method]",
                                    "method": method_name,
                                    "args": call_args,
                                    "raw_type": type(raw).__name__,
                                    "raw_preview": str(raw)[:600],
                                }
                            )
                            # Try to extract result from response
                            result = self._extract_from_history(raw, pair, direction)
                            if result:
                                return result
                            break
                        except Exception:
                            continue
                except Exception as exc:
                    logger.debug(
                        {
                            "event": "history_method_failed",
                            "method": method_name,
                            "error": str(exc),
                        }
                    )

        return None

    def _extract_from_history(self, raw: Any, pair: str, direction: str) -> dict | None:
        """Extract a matching trade from a history response of unknown schema."""
        if raw is None:
            return None

        asset_candidates = {a.upper() for a in PAIR_TO_ASSETS.get(pair, [])}

        def _check_item(item: Any) -> dict | None:
            if not isinstance(item, dict):
                return None
            # Asset check
            item_asset = str(
                item.get("asset", item.get("symbol", item.get("active", "")))
            ).upper()
            if asset_candidates and not any(
                c in item_asset or item_asset in c for c in asset_candidates
            ):
                return None
            # Find profit/result
            profit = None
            for k in ("profit", "win", "payout", "income", "amount"):
                if k in item:
                    try:
                        profit = float(item[k])
                        break
                    except Exception:
                        pass
            if profit is None:
                return None
            outcome = "win" if profit > 0 else ("draw" if profit == 0 else "loss")
            return self._make_result("", pair, direction, outcome, profit)

        if isinstance(raw, list):
            for item in reversed(raw):
                r = _check_item(item)
                if r:
                    return r
        elif isinstance(raw, dict):
            # Might be wrapped: {"deals": [...], "history": [...]}
            for key in ("deals", "history", "trades", "operations", "data", "items"):
                if key in raw and isinstance(raw[key], list):
                    for item in reversed(raw[key]):
                        r = _check_item(item)
                        if r:
                            return r
            return _check_item(raw)

        return None

    def _make_result(
        self, signal_id: str, pair: str, direction: str, outcome: str, payout: float
    ) -> dict:
        return {
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "result": outcome,
            "payout": round(abs(payout), 4),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public consumer ────────────────────────────────────────────────────────

    async def get_result(self, timeout: float = 0.1) -> dict | None:
        try:
            return await asyncio.wait_for(self._result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def add_ws_message(self, msg: dict) -> None:
        """Called by WebSocket hook to buffer messages for introspection."""
        self._ws_messages.append(msg)
        if len(self._ws_messages) > self.MAX_WS_MESSAGES:
            self._ws_messages = self._ws_messages[-self.MAX_WS_MESSAGES :]

    def health(self) -> dict:
        return {
            "connected": self._connected,
            "balance": self._balance,
            "pending_signals": len(self._pending),
            "ws_messages_buffered": len(self._ws_messages),
            "result_queue_size": self._result_queue.qsize(),
        }

    # ── Reconnection ──────────────────────────────────────────────────────────

    async def _reconnect(self) -> None:
        delay = 2.0
        while not self._connected:
            logger.info({"event": "quotex_reconnecting", "delay_seconds": delay})
            await asyncio.sleep(delay)
            try:
                await self.connect()
            except Exception as exc:
                logger.warning({"event": "quotex_reconnect_error", "error": str(exc)})
            delay = min(delay * 2, self.MAX_RECONNECT_DELAY)


# ── Async runner (supervised task entry point) ─────────────────────────────────


async def run_quotex_reader(reader: QuotexReader) -> None:
    if not QUOTEX_LIB_AVAILABLE:
        # pyquotex not installed — sleep forever without reconnect spam
        logger.info(
            {
                "event": "quotex_reader_disabled",
                "reason": "pyquotex_not_installed",
                "note": "Quotex result reading unavailable. Results tracked via balance delta if credentials are set.",
            }
        )
        while True:
            await asyncio.sleep(3600)
    else:
        await reader.poll_results()
