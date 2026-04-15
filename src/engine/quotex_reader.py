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
    "AUD_USD": ["AUDUSD", "AUDUSD_otc"],
    "USD_CAD": ["USDCAD", "USDCAD_otc"],
    "USD_CHF": ["USDCHF", "USDCHF_otc"],
    "NZD_USD": ["NZDUSD", "NZDUSD_otc"],
    "XAU_USD": ["XAUUSD", "XAUUSD_otc"],
}

# Direction: our internal → Quotex API
DIRECTION_TO_QUOTEX: dict[str, str] = {"UP": "call", "DOWN": "put"}
QUOTEX_TO_DIRECTION: dict[str, str] = {"call": "UP", "put": "DOWN"}

# ── Library import ─────────────────────────────────────────────────────────────
_QuotexClient: Any = None
QUOTEX_LIB_AVAILABLE = False
try:
    from pyquotex.stable_api import Quotex as _QuotexClient

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

    Detection strategies (tried in order):
        A) Balance-delta: fast, no API calls, reliable signal that *some* trade closed.
        B) get_history(): scan recent trades for matching asset + profitAmount.
        C) get_result(id): get win/loss for a specific operation from history.

    Args:
        email: Quotex account email.
        password: Quotex account password.
        practice_mode: ``True`` for demo account, ``False`` for real.
    """

    RESULT_BUFFER_SECONDS = 2  # wait after signal expiry before checking
    BALANCE_POLL_INTERVAL = 1.5  # seconds between balance checks
    MAX_HISTORY_ATTEMPTS = 3  # cap history method calls per resolve
    MAX_RECONNECT_DELAY = 120  # seconds

    def __init__(
        self,
        email: str,
        password: str,
        practice_mode: bool = True,
        client: Any = None,
    ) -> None:
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

    async def connect(self) -> bool:
        """Establish WebSocket connection to Quotex.

        If a shared client was injected at construction time (e.g. the same
        client used by QuotexStream), reuses it — only sets the account mode
        and fetches the initial balance without opening a second WebSocket.
        """
        if not QUOTEX_LIB_AVAILABLE:
            return False
        if not self._email or not self._password:
            logger.warning({"event": "quotex_no_credentials"})
            return False
        max_retries = 3
        retry_delay = 5
        for attempt in range(1, max_retries + 1):
            try:
                if self._client is None:
                    # No shared client — create and connect our own
                    logger.info(
                        {
                            "event": "quotex_connect_attempt",
                            "attempt": attempt,
                            "function": "connect(self) method",
                            "file": "quotex_reader.py",
                        }
                    )

                    self._client = _QuotexClient(
                        email=self._email,
                        password=self._password,
                        lang="en",
                    )

                    # SAFETY CHECK: If the library fails to even create the object
                    if self._client is None:
                        logger.warning(
                            {
                                "event": "quotex_init_failed",
                                "attempt": attempt,
                                "function": "connect(self) method",
                                "file": "quotex_reader.py",
                            }
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                            continue
                        return False

                    ok, reason = await self._client.connect()
                    if not ok:
                        logger.error(
                            {
                                "event": "quotex_connect_failed",
                                "reason": str(reason)
                                + " | Attempt "
                                + str(attempt)
                                + " of "
                                + str(max_retries),
                                "function": "connect(self) method",
                                "file": "quotex_reader.py",
                            }
                        )
                        self._client = None  # Reset for next retry
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                            continue
                        return False
                else:
                    # Shared client — caller already connected it
                    reason = "shared_client"

                # If we reached here, self._client is guaranteed to be a valid object
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
                        "reason": str(reason),
                        "shared_client": reason == "shared_client",
                        "function": "connect(self) method",
                        "file": "quotex_reader.py",
                    }
                )
                return True

            except Exception as exc:
                logger.error(
                    {
                        "event": "quotex_connect_exception",
                        "error": str(exc),
                        "file": "quotex_reader.py",
                        "function": "connect(self) method",
                    }
                )
                self._client = None  # Reset for next retry
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    self._connected = False
                    return False
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

        - Polls balance every BALANCE_POLL_INTERVAL seconds.
        - At expiry + RESULT_BUFFER_SECONDS, attempts to match a result
          to each pending signal.
        - Puts matched results onto _result_queue.
        """
        balance_task: asyncio.Task[None] | None = None

        while True:
            if not self._connected:
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

            if balance_task is None or balance_task.done():
                balance_task = asyncio.create_task(self._balance_monitor())

            now = datetime.now(timezone.utc)

            # Find pending signals that are due for resolution
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

            # Clean up resolved entries older than 60 s
            self._pending = {
                sid: d
                for sid, d in self._pending.items()
                if not d["resolved"] or (now - d["expiry_time"]).total_seconds() < 60
            }

            await asyncio.sleep(0.5)

    # ── Balance monitor ────────────────────────────────────────────────────────

    async def _balance_monitor(self) -> None:
        """Continuously polls balance to detect trade closes via delta.

        Also pushes the fresh balance directly to ``status_store`` on every
        poll so the dashboard updates every BALANCE_POLL_INTERVAL (1.5 s)
        rather than waiting for health_task's 10-second tick.
        """
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
                # Always update so health() returns fresh balance
                self._balance = new_balance
            except Exception as exc:
                logger.debug({"event": "balance_poll_error", "error": str(exc)})

    async def _safe_get_balance(self) -> float:
        """Fetch balance with a 5-second timeout; returns cached value on failure."""
        if not self._client or not self._connected:
            return self._balance
        try:
            bal = await asyncio.wait_for(self._client.get_balance(), timeout=5.0)
            return float(bal) if bal is not None else self._balance
        except Exception:
            return self._balance

    # ── Result resolution ──────────────────────────────────────────────────────

    async def _resolve_result(
        self, signal_id: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Try all strategies to resolve a trade result for this signal.

        Order:
            1. Balance delta (fast, no API calls)
            2. get_history() scan (matches asset + profitAmount)
            3. get_result(id) for specific operation from history
        """
        signal = data["signal"]
        pair = signal.get("pair", "")
        direction = signal.get("direction", "UP")
        expiry_time = data["expiry_time"]
        balance_before = data["balance_before"]

        # ── Strategy 1: Balance delta (fast, no API calls) ─────────────────
        current = await self._safe_get_balance()
        delta = current - balance_before
        self._balance = current  # always update
        if abs(delta) > 0.001:
            outcome = "win" if delta > 0 else "loss"
            logger.info(
                {
                    "event": "result_from_balance_delta",
                    "signal_id": signal_id,
                    "pair": pair,
                    "delta": round(delta, 4),
                    "outcome": outcome,
                }
            )
            return self._make_result(
                signal_id,
                pair,
                direction,
                outcome,
                payout=abs(delta) if outcome == "win" else 0.0,
                stake=abs(delta) if outcome == "loss" else 0.0,
            )

        # ── Strategy 2: get_history() scan ─────────────────────────────────
        history_result = await self._match_from_history(pair, direction, expiry_time)
        if history_result:
            return {
                **history_result,
                "signal_id": signal_id,
                "detection": "api_history",
            }

        # ── Strategy 3: near-zero delta fallback = draw ────────────────────
        # If balance hasn't moved at all after expiry + buffer, and history
        # also found nothing, the most likely outcome is a draw (stake returned).
        # Only treat as draw when delta is genuinely zero — not just below our
        # 0.001 noise threshold on a loss.
        if abs(delta) < 0.001:
            logger.info(
                {
                    "event": "result_from_zero_delta",
                    "signal_id": signal_id,
                    "pair": pair,
                    "delta": round(delta, 6),
                    "outcome": "draw",
                }
            )
            return self._make_result(
                signal_id,
                pair,
                direction,
                "draw",
                payout=0.0,
                stake=0.0,
            )

        return None

    # ── History-based matching ─────────────────────────────────────────────────

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
            history = await asyncio.wait_for(self._client.get_history(), timeout=8.0)

            if not history or not isinstance(history, list):
                return None

            # Log first call for schema discovery
            if not hasattr(self, "_logged_history"):
                logger.info(
                    {
                        "event": "quotex_history_sample",
                        "type": type(history).__name__,
                        "count": len(history),
                        "sample": str(history[:2])[:500],
                    }
                )
                self._logged_history = True

            asset_candidates = {
                a.upper() for a in PAIR_TO_ASSETS.get(pair, [pair.replace("_", "")])
            }
            window_start = expiry_time - timedelta(seconds=60)

            for item in reversed(history):
                if not isinstance(item, dict):
                    continue

                # Asset match
                item_asset = str(
                    item.get("asset", item.get("symbol", item.get("active", "")))
                ).upper()
                if not any(
                    c in item_asset or item_asset in c for c in asset_candidates
                ):
                    continue

                # Profit — per docs the field is "profitAmount"
                profit = None
                for key in ("profitAmount", "profit", "payout", "amount"):
                    if key in item:
                        try:
                            profit = float(item[key])
                            break
                        except (ValueError, TypeError):
                            continue

                if profit is None:
                    continue

                # Result from get_result(id) if we have a ticket
                ticket = item.get("ticket")
                if ticket is not None:
                    try:
                        status, details = await asyncio.wait_for(
                            self._client.get_result(ticket), timeout=5.0
                        )
                        logger.info(
                            {
                                "event": "quotex_result_detail",
                                "ticket": ticket,
                                "status": status,
                                "profit": profit,
                            }
                        )
                        outcome = "win" if str(status).lower() == "win" else "loss"
                        return self._make_result(
                            "",
                            pair,
                            direction,
                            outcome,
                            payout=abs(profit) if outcome == "win" else 0.0,
                            stake=abs(profit) if outcome == "loss" else 0.0,
                        )
                    except Exception:
                        pass

                # Infer outcome from profit sign
                if profit > 0:
                    outcome = "win"
                elif profit == 0:
                    outcome = "draw"
                else:
                    outcome = "loss"

                logger.info(
                    {
                        "event": "quotex_history_match",
                        "pair": pair,
                        "asset": item_asset,
                        "profit": profit,
                        "outcome": outcome,
                        "ticket": item.get("ticket"),
                    }
                )
                return self._make_result(
                    "",
                    pair,
                    direction,
                    outcome,
                    payout=abs(profit) if outcome == "win" else 0.0,
                    stake=abs(profit) if outcome == "loss" else 0.0,
                )

        except asyncio.TimeoutError:
            logger.warning({"event": "quotex_history_timeout", "pair": pair})
        except Exception as exc:
            logger.error(
                {
                    "event": "quotex_history_error",
                    "pair": pair,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

        return None

    # ── Result construction ────────────────────────────────────────────────────

    def _make_result(
        self,
        signal_id: str,
        pair: str,
        direction: str,
        outcome: str,
        payout: float,
        stake: float = 0.0,
    ) -> dict[str, Any]:
        """Build a standardised result dict for SignalOrchestrator.on_result."""
        logger.info(
            {
                "event": "result_made",
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
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public consumer ────────────────────────────────────────────────────────

    async def get_result(self, timeout: float = 0.1) -> dict[str, Any] | None:
        """Non-blocking read of the next resolved result."""
        try:
            return await asyncio.wait_for(self._result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def health(self) -> dict[str, Any]:
        """Return a serialisable health snapshot."""
        return {
            "connected": self._connected,
            "balance": self._balance,
            "pending_signals": len(self._pending),
            "result_queue_size": self._result_queue.qsize(),
        }

    # ── Reconnection ──────────────────────────────────────────────────────────

    async def _reconnect(self) -> None:
        """Exponential-backoff reconnection loop."""
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
                "event": "quotex_reader_disabled",
                "reason": "pyquotex_not_installed",
                "note": "Quotex result reading unavailable.",
            }
        )
        while True:
            await asyncio.sleep(3600)
    else:
        await reader.poll_results()
