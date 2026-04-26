"""
quotex_reader.py — Reads closed trade results from Quotex account.

Uses pyquotex (cleitonleonel/pyquotex) unofficial WebSocket API.
Module: pyquotex.stable_api (NOT quotexapi — package renamed in v1.0)
Install: pip install git+https://github.com/cleitonleonel/pyquotex.git

CONFIRMED API (from pyquotex docs):
    - Quotex(email, password, lang="en")
    - await client.connect() → (bool, reason_str)
    - client.change_account("PRACTICE" | "REAL")
    - client.set_account_mode("PRACTICE" | "REAL")
    - await client.get_balance() → float
    - await client.get_history() → [{"ticket": id, "profitAmount": float, ...}, ...]
    - await client.get_result(operation_id) → ("win"|"loss", details)
    - await client.check_win(operation_id) → True (profit) | False (loss)
    - client.get_profit() → float (after check_win)
    - Direction strings: "call" (UP/buy) | "put" (DOWN/sell)
    - Asset format: "EURUSD_otc" (OTC, 24/7) | "EURUSD" (market hours)

Strategy (no trade ID available — our bot places trades, not us):
    1. Balance-delta: monitor account balance; change at expiry = trade result.
    2. get_history(): scan recent trades for matching asset + profitAmount.
    3. get_result(id): get win/loss for a specific operation ID from history.
"""

from __future__ import annotations

import asyncio
import logging


from typing import Any
from ml_engine.model import Tick
from core.config import get_settings
from core.exceptions import NotificationError
from datetime import datetime, timedelta, timezone


logger = logging.getLogger(__name__)


# ── Shared log-field literals (S1192: defined once, used in all log dicts) ─────
_CONNECT_METHOD = "connect(self) method"
_MODULE_FILE = "quotex_stream.py"

# ── Asset name mapping ─────────────────────────────────────────────────────────
# Quotex uses "EURUSD_otc" on OTC (weekends / off-hours) and "EURUSD" on live
# The _otc suffix is tried first; both are checked on match.

PAIR_TO_ASSETS: dict[str, list[str]] = {
    "EUR_USD": ["EURUSD", "EURUSD_otc"],
    "GBP_USD": ["GBPUSD", "GBPUSD_otc"],
    "USD_JPY": ["USDJPY", "USDJPY_otc"],
    "AUD_USD": ["AUDUSD", "AUDUSD_otc"],
    "USD_CAD": ["USDCAD", "USDCAD_otc"],
    "USD_CHF": ["USDCHF", "USDCHF_otc"],
    "NZD_USD": ["NZDUSD", "NZDUSD_otc"],
    "XAU_USD": ["XAUUSD", "XAUUSD_otc"],
}

# Direction: our internal → Quotex API
DIRECTION_TO_QUOTEX: dict[str, str] = {"UP": "call", "DOWN": "put"}
QUOTEX_TO_DIRECTION: dict[str, str] = {
    "call": "UP", "put": "DOWN",
    "0": "UP", "1": "DOWN",  # command field: 0=call, 1=put
}

# ── Library import ─────────────────────────────────────────────────────────────
_QuotexClient: Any = None
QUOTEX_LIB_AVAILABLE: bool = False
try:
    from pyquotex.stable_api import Quotex as _QuotexClient

    QUOTEX_LIB_AVAILABLE = True
except Exception as _qx_err:
    logger.critical(
        {
            "event": "pyquotex_import_failed",
            "error_type": type(_qx_err).__name__,
            "error": str(_qx_err),
            "message": "pyquotex library not available - Quotex streaming disabled",
        }
    )


class QuotexDataStream:
    """
    Connects to Quotex WebSocket and reads closed trade results.

    Detection strategies (tried in order):
        A) Balance-delta: fast, no API calls, reliable signal that *some* trade closed.
        B) get_history(): scan recent trades for matching asset + profitAmount.
        C) get_result(id): get win/loss for a specific operation from history.

    Args:
        email: Quotex account email.
        password: Quotex account password.
        practice_mode: ``True`` for demo account, ``False`` for real.
    """

    RESULT_BUFFER_SECONDS = 5  # wait after signal expiry before checking
    BALANCE_POLL_INTERVAL = 1.5  # seconds between balance checks
    MAX_HISTORY_ATTEMPTS = 3  # cap history method calls per resolve
    MAX_RECONNECT_DELAY = 120  # seconds
    STREAM_LOG_ONCE = True  # Circuit switch for unavailable stream

    def __init__(
        self,
        email: str,
        password: str,
        symbol: str,
        practice_mode: bool = True,
        client: Any = None,
    ) -> None:
        self._settings = get_settings()
        self.symbol = symbol

        self._email = email
        self._password = password
        self._practice_mode = practice_mode
        self._account_type = "PRACTICE" if practice_mode else "REAL"

        # Accept a pre-built shared client (e.g. from QuotexStream) so that
        # only one WebSocket session is opened per Quotex account. When None,
        # connect() creates its own client.
        self._client: Any = client
        self._connected = False
        self._balance: float = 0.0
        self._prev_balance: float = 0.0

        # {signal_id → {signal, expiry_time, balance_before, attempts, resolved}}
        self._pending: dict[str, dict[str, Any]] = {}

        # Cached history from last get_history() call
        self._cached_history: list[dict[str, Any]] = []
        self._history_cache_time: float = 0.0

        # Results ready for consumption by main loop
        self._result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    # ── Connection ─────────────────────────────────────────────────────────────

    async def _try_create_and_connect(
        self, attempt: int, max_retries: int
    ) -> tuple[bool, str]:
        """Create a new Quotex client and attempt one connection. Returns (ok, reason)."""
        logger.info(
            {
                "event": "quotex_connect_attempt",
                "attempt": attempt,
                "function": _CONNECT_METHOD,
                "file": _MODULE_FILE,
            }
        )
        self._client = _QuotexClient(
            email=self._email, password=self._password, lang="en"
        )
        if self._client is None:
            logger.warning(
                {
                    "event": "quotex_init_failed",
                    "attempt": attempt,
                    "function": _CONNECT_METHOD,
                    "file": _MODULE_FILE,
                }
            )
            return False, ""
        ok, reason = await self._client.connect()
        if not ok:
            logger.error(
                {
                    "event": "quotex_connect_failed",
                    "reason": f"{reason} | Attempt {attempt} of {max_retries}",
                    "function": _CONNECT_METHOD,
                    "file": _MODULE_FILE,
                }
            )
            self._client = None
            return False, ""
        return True, str(reason)

    async def _finalize_connection(self, reason: str) -> None:
        """Set account mode, fetch balance, and log successful connection."""
        self._connected = True
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
                "reason": reason,
                "shared_client": reason == "shared_client",
                "function": _CONNECT_METHOD,
                "file": _MODULE_FILE,
            }
        )

    async def _connect_one_attempt(
        self, attempt: int, max_retries: int, retry_delay: float
    ) -> str | None:
        """Run one connection attempt; return reason string on success, None on failure.

        Sleeps retry_delay before returning None when a retry should follow.
        """
        if self._client is None:
            ok, reason = await self._try_create_and_connect(attempt, max_retries)
            if not ok:
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                return None
        else:
            reason = "shared_client"
        return reason

    async def _connect_attempt_with_exception_guard(
        self, attempt: int, max_retries: int, retry_delay: float
    ) -> str | None:
        """Wrap _connect_one_attempt with exception handling. Returns reason or None."""
        try:
            return await self._connect_one_attempt(attempt, max_retries, retry_delay)
        except Exception as exc:
            logger.error(
                {
                    "event": "quotex_connect_exception",
                    "error": str(exc),
                    "file": _MODULE_FILE,
                    "function": _CONNECT_METHOD,
                }
            )
            self._client = None
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                self._connected = False
            return None

    async def connect(self) -> bool:
        """Establish WebSocket connection to Quotex.

        If a shared client was injected at construction time (e.g. the same
        client used by QuotexStream), reuses it — only sets the account mode
        and fetches the initial balance without opening a second WebSocket.
        """
        if not QUOTEX_LIB_AVAILABLE:
            return False
        if not self._email or not self._password:
            logger.warning(
                {
                    "event": "quotex_no_credentials",
                }
            )
            return False

        max_retries = 3
        retry_delay = 5

        for attempt in range(1, max_retries + 1):
            reason = await self._connect_attempt_with_exception_guard(
                attempt, max_retries, retry_delay
            )
            if reason is not None:
                await self._finalize_connection(reason)
                return True
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
            except Exception as exc:
                logger.debug(
                    {
                        "event": "quotex_disconnect_error",
                        "error": str(exc),
                    }
                )

    # ── Signal registration ────────────────────────────────────────────────────

    def register_pending(
        self, signal_id: str, signal: dict[str, Any], expiry_time: datetime
    ) -> None:
        """Register a signal that needs result matching at *expiry_time*."""
        balance_snapshot = self._balance
        self._pending[signal_id] = {
            "signal": signal,
            "expiry_time": expiry_time,
            "balance_before": balance_snapshot,
            "attempts": 0,
            "resolved": False,
        }
        logger.info(
            {
                "event": "pending_registered",
                "signal_id": signal_id,
                "pair": signal.get("pair"),
                "direction": signal.get("direction"),
                "confidence": signal.get("confidence"),
                "fired_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "expiry_at": expiry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "balance_snapshot": balance_snapshot,
            }
        )

    # ── Main poll loop ─────────────────────────────────────────────────────────

    async def _manage_balance_task(
        self, balance_task: "asyncio.Task[None] | None"
    ) -> "asyncio.Task[None] | None":
        """Ensure balance monitor task is running; reconnect if disconnected.

        Returns the active task, or None if still disconnected after reconnect.
        """
        if not self._connected:
            if balance_task is not None and not balance_task.done():
                balance_task.cancel()
                await balance_task
                balance_task = None
            await self._reconnect()
        if balance_task is None or balance_task.done():
            balance_task = asyncio.create_task(self._balance_monitor())
        return balance_task

    async def _process_due_signals(self, now: datetime) -> None:
        """Resolve all pending signals whose expiry window has passed."""
        due = {
            sid: data
            for sid, data in self._pending.items()
            if not data["resolved"]
            and now
            >= data["expiry_time"] + timedelta(seconds=self.RESULT_BUFFER_SECONDS)
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
                        "message": (
                            f"No matching trade found after "
                            f"{self.MAX_HISTORY_ATTEMPTS} attempts."
                        ),
                    }
                )

    def _cleanup_resolved(self, now: datetime) -> None:
        """Remove resolved pending entries older than 60 s."""
        self._pending = {
            sid: d
            for sid, d in self._pending.items()
            if not d["resolved"] or (now - d["expiry_time"]).total_seconds() < 60
        }

    async def poll_results(self) -> None:
        """
        Background task: runs forever.

        - Polls balance every BALANCE_POLL_INTERVAL seconds.
        - At expiry + RESULT_BUFFER_SECONDS, attempts to match a result
            to each pending signal.
        - Puts matched results onto _result_queue.
        """
        balance_task: asyncio.Task[None] | None = None

        while True:
            balance_task = await self._manage_balance_task(balance_task)
            if balance_task is None:
                continue  # still disconnected — _manage_balance_task already slept
            now = datetime.now(timezone.utc)
            await self._process_due_signals(now)
            self._cleanup_resolved(now)
            await asyncio.sleep(0.5)

    # ── Balance monitor ────────────────────────────────────────────────────────

    async def _balance_monitor(self) -> None:
        """Continuously polls balance to detect trade closes via delta.

        Also pushes the fresh balance directly to ``status_store`` on every
        poll so the dashboard updates every BALANCE_POLL_INTERVAL (1.5 s)
        rather than waiting for health_task's 10-second tick.
        """
        if not self._balance:
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
                # Always update so health() returns fresh balance
                self._balance = new_balance
            except Exception as exc:
                logger.debug(
                    {
                        "event": "balance_poll_error",
                        "error": str(exc),
                    }
                )

    async def _safe_get_balance(self) -> float:
        """Fetch balance with a 5-second timeout; returns cached value on failure."""
        if not self._client or not self._connected:
            return self._balance
        try:
            async with asyncio.timeout(5.0):
                bal = await self._client.get_balance()
            return float(bal) if bal is not None else self._balance
        except Exception:
            return self._balance

    # ── Result resolution ──────────────────────────────────────────────────────

    async def _resolve_result(
        self, signal_id: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Try to resolve trade result for this signal.

        """
        signal = data["signal"]
        pair = signal.get("pair", "")
        direction = signal.get("direction", "UP")
        expiry_time = data["expiry_time"]

        history_result = await self._match_from_history(pair, direction, expiry_time)
        if history_result:
            return {
                **history_result,
                "signal_id": signal_id,
                "detection": "api_history",
            }

        # Return None - signal is dead, do NOT create a result
        return None

    # ── History-based matching ─────────────────────────────────────────────────

    def _trade_time_diff(
        self,
        trade: dict,
        pair: str,
        direction: str,
        target_time: datetime,
    ) -> float | None:
        """Return seconds between trade close time and target_time if trade matches, else None."""
        trade_asset = trade.get("symbol") or trade.get("asset", "")
        if not trade_asset or not self._assets_match(pair, trade_asset):
            return None
        # Normalize signal direction to Quotex format ("call"/"put").
        # TradeSignal uses "CALL"/"PUT"; legacy paths may use "UP"/"DOWN".
        # DIRECTION_TO_QUOTEX covers UP/DOWN; unknown values fall through to
        # direction.lower() which handles CALL→"call", PUT→"put" directly.
        signal_quotex = DIRECTION_TO_QUOTEX.get(direction.upper(), direction.lower())
        # Check direction via multiple field names; map integer command 0/1.
        for field in ("directionType", "direction", "command"):
            raw = trade.get(field)
            if raw is not None:
                trade_dir = {"0": "call", "1": "put"}.get(str(raw).lower(), str(raw).lower())
                if trade_dir != signal_quotex:
                    return None
                break
        # Prefer Unix epoch timestamp — available in all Quotex history records
        # as close_time_timestamp. Fall back to string parsing if absent.
        close_ts = trade.get("close_time_timestamp")
        if close_ts is not None:
            trade_time = datetime.fromtimestamp(float(close_ts), tz=timezone.utc).replace(tzinfo=None)
            return abs((trade_time - target_time).total_seconds())
        close_time_str = trade.get("close_time") or trade.get("closeTime", "")
        if not close_time_str:
            return None
        try:
            trade_time = datetime.strptime(str(close_time_str)[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        return abs((trade_time - target_time).total_seconds())

    def _find_closest_trade(
        self,
        history: list,
        pair: str,
        direction: str,
        target_time: datetime,
    ) -> tuple[dict | None, float]:
        """Scan history for the trade closest in time to target_time.

        Returns (trade_dict, time_diff_seconds) or (None, inf) if no match.
        """
        closest_trade: dict | None = None
        smallest_diff: float = float("inf")

        for trade in history:
            diff = self._trade_time_diff(trade, pair, direction, target_time)
            if diff is not None and diff < 10 and diff < smallest_diff:
                smallest_diff = diff
                closest_trade = trade

        return closest_trade, smallest_diff

    @staticmethod
    def _outcome_from_profit(profit: float) -> tuple[str, float, float]:
        """Return (outcome, payout, stake) from a raw profit value."""
        if profit > 0:
            return "win", profit, 0.0
        if profit < 0:
            return "loss", 0.0, abs(profit)
        return "draw", 0.0, 0.0

    async def _match_from_history(
        self, pair: str, direction: str, expiry_time: datetime
    ) -> dict[str, Any] | None:
        """
        Call get_history() and match against our pending signal.

        Per docs, get_history() returns a list like:
            [{"ticket": 12345, "profitAmount": 5.50, "asset": "EURUSD_otc", ...}, ...]
        """
        if not self._client:
            return None

        try:
            async with asyncio.timeout(8.0):
                history = await self._client.get_history()
            if not history or not isinstance(history, list):
                return None

            target_time = expiry_time.astimezone(timezone.utc).replace(tzinfo=None)
            closest_trade, smallest_diff = self._find_closest_trade(
                history, pair, direction, target_time
            )

            if not closest_trade:
                logger.warning(
                    {
                        "event": "NO_TRADE_NEAR_EXPIRY",
                        "pair": pair,
                        "direction": direction,
                        "expiry_time": expiry_time.isoformat(),
                    }
                )
                return None

            profit_raw = closest_trade.get("profitAmount", closest_trade.get("profit"))
            if profit_raw is None:
                logger.warning(
                    {
                        "event": "TRADE_PROFIT_MISSING",
                        "pair": pair,
                        "msg": "Likely trade did not go through",
                        "ticket": closest_trade.get("ticket"),
                    }
                )
                return None

            profit = float(profit_raw)
            outcome, payout, stake = self._outcome_from_profit(profit)
            open_price = float(closest_trade.get("open_price") or 0)
            close_price = float(closest_trade.get("close_price") or 0)

            logger.info(
                {
                    "event": "QUOTEX_HISTORY_MATCH",
                    "pair": pair,
                    "direction": direction,
                    "profit": profit,
                    "outcome": outcome,
                    "ticket": closest_trade.get("ticket"),
                    "open_time": closest_trade.get("open_time"),
                    "close_time": closest_trade.get("close_time"),
                    "expected_expiry": expiry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_diff_seconds": round(smallest_diff, 2),
                }
            )

            return self._make_result(
                "", pair, direction, outcome, payout=payout, stake=stake,
                open_price=open_price, close_price=close_price,
            )

        except TimeoutError:
            logger.warning(
                {
                    "event": "QUOTEX_HISTORY_TIMEOUT",
                    "pair": pair,
                }
            )
        except Exception as exc:
            logger.error(
                {
                    "event": "QUOTEX_HISTORY_ERROR",
                    "pair": pair,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
        return None

    def _assets_match(self, signal_pair: str, trade_asset: str) -> bool:
        """
        Check if a signal pair matches a Quotex trade asset using PAIR_TO_ASSETS.

        Direct comparison fails because signal pairs use underscore format
        (e.g. "EUR_USD") while Quotex trades use concatenated format
        (e.g. "EURUSD_otc"). The PAIR_TO_ASSETS mapping bridges this.
        """
        if not trade_asset:
            return False
        trade_upper = trade_asset.upper().strip()
        valid_assets = PAIR_TO_ASSETS.get(signal_pair, [])
        return trade_upper in [a.upper() for a in valid_assets]

    # ── Result construction ────────────────────────────────────────────────────

    def _make_result(
        self,
        signal_id: str,
        pair: str,
        direction: str,
        outcome: str,
        payout: float,
        stake: float = 0.0,
        open_price: float = 0.0,
        close_price: float = 0.0,
    ) -> dict[str, Any]:
        """Build a standardised result dict for SignalOrchestrator.on_result."""
        logger.info(
            {
                "event": "RESULT_MADE",
                "signal_id": signal_id,
                "pair": pair,
                "direction": direction,
                "outcome": outcome,
                "payout": round(abs(payout), 4),
                "stake": round(stake, 4),
            }
        )
        return {
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "result": outcome,
            # payout = amount won (wins) or 0 (losses/draws)
            "payout": round(abs(payout), 4) if outcome == "win" else 0.0,
            # stake = amount risked — used by DailySession.record_loss to
            # deduct from net_profit. For balance-delta results, stake equals
            # the absolute delta (the money lost). For history results, stake
            # equals the absolute profit amount when outcome is loss.
            "stake": round(abs(stake), 4),
            "open_price": open_price,
            "close_price": close_price,
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public consumer ────────────────────────────────────────────────────────

    async def get_result(self) -> dict[str, Any]:
        """Await the next resolved result from the queue. Callers should wrap with asyncio.timeout()."""
        return await self._result_queue.get()

    def health(self) -> dict[str, Any]:
        """Return a serialisable health snapshot."""
        return {
            "connected": self._connected,
            "balance": self._balance,
            "pending_signals": len(self._pending),
            "result_queue_size": self._result_queue.qsize(),
        }

    # ── Subscribe for LiveEngine ─────────────────────────────────────────────

    def _build_tick(self, price_point: dict) -> Tick:
        price = float(price_point["price"])
        return Tick(
            timestamp=datetime.fromtimestamp(price_point["time"], tz=timezone.utc),
            symbol=self.symbol,
            bid=price,
            ask=price,
            source="QUOTEX",
        )

    def _check_realtime_capability(self, otc_asset: str, state: str) -> str:
        """Log a one-time error if start_realtime_price is unavailable. Returns new state."""
        if state != "degraded":
            logger.error({"event": "STREAM_REALTIME_UNAVAILABLE", "symbol": otc_asset})
        return "degraded"

    def _extract_price_points(self, result: Any, otc_asset: str) -> list | None:
        """Return price_points list from a start_realtime_price result, or None."""
        if result and isinstance(result, dict):
            return result.get(otc_asset)
        return None

    def _handle_price_points(self, otc_asset: str, state: str) -> str:
        """Log stream-up transition. Returns updated state."""
        if state == "degraded":
            logger.info({"event": "STREAM_UP", "symbol": otc_asset})
            state = "ok"
        return state

    async def _poll_once(self, otc_asset: str, state: str) -> tuple[str, list[Tick]]:
        """
        One polling iteration. Handles all sleeps internally so the caller
        (subscribe) stays a thin generator loop with no branching.
        Returns (updated_state, ticks_to_yield).
        """
        try:
            if not hasattr(self._client, "start_realtime_price"):
                state = self._check_realtime_capability(otc_asset, state)
                await asyncio.sleep(1)
                return state, []

            result = await self._client.start_realtime_price(otc_asset, 1)
            price_points = self._extract_price_points(result, otc_asset)

            if price_points:
                state = self._handle_price_points(otc_asset, state)
                ticks = [self._build_tick(pp) for pp in price_points]
                await asyncio.sleep(1)
                return state, ticks

            if state != "degraded":
                logger.warning({"event": "STREAM_OFFLINE", "symbol": otc_asset})
                state = "degraded"
            await asyncio.sleep(4)
            return state, []

        except asyncio.CancelledError:
            raise
        except Exception as e:
            if state != "degraded":
                logger.error({"event": "STREAM_ERROR", "symbol": otc_asset, "error_type": type(e).__name__, "error": str(e)})
                state = "degraded"
            if not self._connected:
                logger.info(
                    {"event": "quotex_stream_reconnecting", "symbol": otc_asset}
                )
                await self._reconnect()
                if self._connected:
                    state = "ok"
                    logger.info(
                        {"event": "quotex_stream_reconnected", "symbol": otc_asset}
                    )
            await asyncio.sleep(1)
            return state, []

    async def subscribe(self):
        """
        Async generator that yields real-time price ticks for Quotex OTC pairs.
        Since pyquotex doesn't support a real price stream, this method polls
        the latest price every 1000 ms using start_realtime_price() with batch_size=1.
        """
        otc_asset = self.symbol
        state = "ok"

        while self._connected:
            state, ticks = await self._poll_once(otc_asset, state)
            for tick in ticks:
                yield tick

        if hasattr(self._client, "stop_candles_stream"):
            self._client.stop_candles_stream(otc_asset)

    # ── Reconnection ──────────────────────────────────────────────────────────

    async def _reconnect(self) -> None:
        """Exponential-backoff reconnection loop."""
        delay = 2.0
        while not self._connected:
            logger.info({"event": "QUOTEX_RECONNECTING", "delay_seconds": delay})
            await asyncio.sleep(delay)
            try:
                await self.connect()
            except Exception as exc:
                logger.warning({"event": "QUOTEX_RECONNECT_ERROR", "error": str(exc)})
            delay = min(delay * 2, self.MAX_RECONNECT_DELAY)


# ── Async runner (supervised task entry point) ─────────────────────────────────


async def run_quotex_reader(reader: QuotexDataStream) -> None:
    """Top-level entry point for asyncio.create_task.

    Does NOT call reader.connect() — main() already connected the reader
    before launching this task. Calling connect() a second time opens a
    duplicate WebSocket session on the same Quotex account, which kills
    the first. If the initial connect failed, poll_results()'s internal
    _reconnect() loop handles recovery.
    """
    if not QUOTEX_LIB_AVAILABLE:
        logger.info(
            {
                "event": "QUOTEX_READER_DISABLED",
                "reason": "pyquotex_not_installed",
                "note": "Quotex result reading unavailable.",
            }
        )
        while True:
            await asyncio.sleep(3600)
    else:
        await reader.poll_results()
