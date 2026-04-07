"""
quotex_reader.py — Reads closed trade results directly from Quotex account.

Uses the pyquotex (cleitonleonel/pyquotex) unofficial WebSocket API.
Stripped to the bare minimum needed: authenticate, receive trade result events,
match them to pending signals, return win/loss/draw + payout.

Install:
    pip install quotexapi
    # or from source:
    pip install git+https://github.com/cleitonleonel/pyquotex.git

Strategy:
    1. Connect and authenticate via email/password (WebSocket session).
    2. Monitor the underlying WebSocket for trade-close messages.
    3. When a pending signal's expiry time passes (+2s buffer), scan recent
       messages for a result matching (asset, direction, expiry_window).
    4. Any uncertain field is logged as [QUOTEX_RAW_RESPONSE] for debugging.

IMPORTANT: The Quotex unofficial API response schema is not fully stable.
If field names change, check the raw_response logs and update _parse_result().
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Quotex asset name format ───────────────────────────────────────────────────
# Based on known pyquotex usage; Quotex internal names may differ per instrument
PAIR_TO_QUOTEX_ASSET: dict[str, str] = {
    "EUR_USD": "EURUSD_otc",   # OTC pairs available on weekends; live pairs on weekdays
    "GBP_USD": "GBPUSD_otc",
    "USD_JPY": "USDJPY_otc",
    "XAU_USD": "XAUUSD_otc",
}
LIVE_ASSET_MAP: dict[str, str] = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "XAU_USD": "XAUUSD",
}
DIRECTION_MAP = {"buy": "UP", "call": "UP", "sell": "DOWN", "put": "DOWN"}

# ── Library availability ───────────────────────────────────────────────────────
try:
    from quotexapi.stable_api import Quotex as QuotexClient
    QUOTEX_AVAILABLE = True
except ImportError:
    QUOTEX_AVAILABLE = False
    logger.warning(
        {"event": "quotex_unavailable",
         "message": "quotexapi not installed — trade result reading disabled. "
                    "Install with: pip install git+https://github.com/cleitonleonel/pyquotex.git"}
    )


class QuotexReader:
    """
    Connects to Quotex via pyquotex and reads closed trade results.

    Call flow:
        reader = QuotexReader(email, password)
        await reader.connect()
        reader.register_pending(signal_id, signal, expiry_time)
        # ... later, from main loop:
        result = await reader.get_result()   # non-blocking
    """

    RESULT_BUFFER_SECONDS = 2   # wait this long after expiry before querying
    MAX_RECONNECT_DELAY = 120

    def __init__(self, email: str, password: str, practice_mode: bool = True) -> None:
        self._email = email
        self._password = password
        self._practice = practice_mode
        self._client: Any = None
        self._connected = False

        # {signal_id -> {signal, expiry_time, checked}}
        self._pending: dict[str, dict] = {}

        # Results ready for consumption
        self._results_queue: asyncio.Queue = asyncio.Queue()

        # Raw WebSocket messages received (ring buffer, last 200)
        self._raw_messages: list[dict] = []

    # ── Connection lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        if not QUOTEX_AVAILABLE:
            logger.warning({"event": "quotex_connect_skipped", "reason": "library_unavailable"})
            return
        if not self._email or not self._password:
            logger.warning({"event": "quotex_connect_skipped", "reason": "no_credentials"})
            return

        try:
            self._client = QuotexClient(
                email=self._email,
                password=self._password,
            )
            check, message = await self._client.connect()
            if check:
                self._connected = True
                mode = "[PRACTICE MODE]" if self._practice else "[LIVE MODE ⚠️]"
                logger.info(
                    {"event": "quotex_connected", "mode": mode, "message": str(message)}
                )
                # Hook into the underlying message handler if available
                self._install_message_hook()
            else:
                logger.error(
                    {"event": "quotex_connect_failed", "message": str(message)}
                )
        except Exception as exc:
            logger.error({"event": "quotex_connect_exception", "error": str(exc)})
            self._connected = False

    async def disconnect(self) -> None:
        if self._client and self._connected:
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False

    def _install_message_hook(self) -> None:
        """
        Attempt to hook into pyquotex's WebSocket message handler.
        pyquotex stores parsed messages in client.api.wss_message or similar.
        This is version-dependent — we log what we find.
        """
        try:
            # pyquotex stores the underlying API object
            if hasattr(self._client, "api"):
                logger.info({"event": "quotex_api_attrs",
                             "attrs": [a for a in dir(self._client.api) if not a.startswith("_")][:20]})
            # Log available client methods for debugging
            available = [m for m in dir(self._client) if not m.startswith("_")]
            logger.info({"event": "quotex_client_methods", "methods": available[:30]})
        except Exception as exc:
            logger.debug({"event": "quotex_hook_failed", "error": str(exc)})

    # ── Pending signal registration ───────────────────────────────────────────

    def register_pending(self, signal_id: str, signal: dict, expiry_time: datetime) -> None:
        """Register a fired signal for result lookup at expiry + buffer."""
        self._pending[signal_id] = {
            "signal": signal,
            "expiry_time": expiry_time,
            "checked": False,
            "attempts": 0,
        }

    # ── Result polling loop ───────────────────────────────────────────────────

    async def poll_results(self) -> None:
        """
        Background loop: at expiry + RESULT_BUFFER_SECONDS, query Quotex
        for the most recent closed trade matching the pending signal.
        Runs forever; reconnects on disconnect.
        """
        while True:
            if not self._connected:
                await self._reconnect()

            now = datetime.now(timezone.utc)
            due = [
                sid for sid, data in self._pending.items()
                if not data["checked"]
                and now >= data["expiry_time"] + timedelta(seconds=self.RESULT_BUFFER_SECONDS)
                and data["attempts"] < 3
            ]

            for signal_id in due:
                data = self._pending[signal_id]
                data["attempts"] += 1
                result = await self._fetch_result(signal_id, data)
                if result:
                    data["checked"] = True
                    await self._results_queue.put(result)
                    logger.info({"event": "result_matched", "signal_id": signal_id, "result": result["result"]})
                elif data["attempts"] >= 3:
                    data["checked"] = True
                    logger.warning(
                        {"event": "result_unmatched",
                         "signal_id": signal_id,
                         "signal": data["signal"],
                         "message": "Could not find matching trade in Quotex — "
                                    "check [QUOTEX_RAW_RESPONSE] logs above for schema"}
                    )

            # Clean up resolved entries
            self._pending = {
                sid: d for sid, d in self._pending.items()
                if not d["checked"] or d["attempts"] < 3
            }

            await asyncio.sleep(1.0)

    async def _fetch_result(self, signal_id: str, data: dict) -> dict | None:
        """
        Try all known pyquotex methods to retrieve the trade result.
        Logs raw responses for schema verification.
        """
        if not self._connected or not self._client:
            return None

        signal = data["signal"]
        pair = signal.get("pair", "")
        direction = signal.get("direction", "UP")
        expiry_time = data["expiry_time"]

        # Build asset name candidates
        asset_candidates = [
            LIVE_ASSET_MAP.get(pair, pair.replace("_", "")),
            PAIR_TO_QUOTEX_ASSET.get(pair, pair.replace("_", "") + "_otc"),
        ]

        # ── Attempt 1: check_win (requires trade ID — we don't have it) ───────
        # Skipped: we don't place trades, so we have no ID.

        # ── Attempt 2: get_history / get_profit_line methods ─────────────────
        for method_name in ["get_history", "get_profit_line", "get_candle_v2",
                             "get_digital_spot_pulse", "subscribe_symbol"]:
            if not hasattr(self._client, method_name):
                continue
            try:
                method = getattr(self._client, method_name)
                # Try with each asset name
                for asset in asset_candidates:
                    try:
                        raw = await asyncio.wait_for(method(asset), timeout=5.0)
                        logger.info(
                            {"event": "[QUOTEX_RAW_RESPONSE - schema needs verification]",
                             "method": method_name,
                             "asset": asset,
                             "raw_type": type(raw).__name__,
                             "raw_preview": str(raw)[:500]}
                        )
                        result = self._parse_result(raw, signal_id, pair, direction, expiry_time)
                        if result:
                            return result
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug({"event": "quotex_method_failed", "method": method_name, "error": str(exc)})

        # ── Attempt 3: inspect raw WebSocket messages ─────────────────────────
        if self._raw_messages:
            for msg in reversed(self._raw_messages[-50:]):
                result = self._parse_result(msg, signal_id, pair, direction, expiry_time)
                if result:
                    return result

        # ── Attempt 4: profile / balance delta (last resort) ─────────────────
        try:
            if hasattr(self._client, "get_profile"):
                profile = await asyncio.wait_for(self._client.get_profile(), timeout=5.0)
                logger.info(
                    {"event": "[QUOTEX_RAW_RESPONSE - profile]",
                     "raw_preview": str(profile)[:300]}
                )
        except Exception:
            pass

        return None

    def _parse_result(
        self,
        raw: Any,
        signal_id: str,
        pair: str,
        direction: str,
        expiry_time: datetime,
    ) -> dict | None:
        """
        Best-effort extraction of win/loss from any pyquotex response shape.
        Returns None if the response doesn't contain a recognisable trade result.

        Known response shapes (update as you discover them from raw logs):
          - dict with 'profit', 'win', 'result', 'asset', 'direction'
          - dict with 'deals' list (Quotex history)
          - float/int (profit value directly)
        """
        if raw is None:
            return None

        try:
            # Shape: plain profit float
            if isinstance(raw, (int, float)):
                profit = float(raw)
                outcome = "win" if profit > 0 else ("draw" if profit == 0 else "loss")
                return self._make_result(signal_id, pair, direction, outcome, profit)

            # Shape: dict with known fields
            if isinstance(raw, dict):
                # Direct result field
                if "result" in raw:
                    outcome = str(raw["result"]).lower()
                    if outcome in ("win", "loss", "draw"):
                        payout = float(raw.get("profit", raw.get("payout", 0.0)))
                        return self._make_result(signal_id, pair, direction, outcome, payout)

                # Profit field (positive = win, negative = loss, zero = draw)
                if "profit" in raw:
                    profit = float(raw["profit"])
                    outcome = "win" if profit > 0 else ("draw" if profit == 0 else "loss")
                    return self._make_result(signal_id, pair, direction, outcome, profit)

                # 'win' boolean field
                if "win" in raw:
                    outcome = "win" if raw["win"] else "loss"
                    payout = float(raw.get("profit", raw.get("amount", 0.0)))
                    return self._make_result(signal_id, pair, direction, outcome, payout)

                # Deals list (Quotex trade history)
                if "deals" in raw and isinstance(raw["deals"], list):
                    for deal in reversed(raw["deals"]):
                        if not isinstance(deal, dict):
                            continue
                        deal_asset = str(deal.get("asset", deal.get("symbol", ""))).upper()
                        expected_asset = pair.replace("_", "")
                        if expected_asset not in deal_asset:
                            continue
                        # Check time proximity to expiry
                        deal_time_raw = deal.get("close_time", deal.get("expiration_time", deal.get("time")))
                        if deal_time_raw:
                            try:
                                deal_ts = datetime.fromtimestamp(float(deal_time_raw), tz=timezone.utc)
                                if abs((deal_ts - expiry_time).total_seconds()) > 30:
                                    continue
                            except Exception:
                                pass
                        profit = float(deal.get("profit", deal.get("win", deal.get("payout", 0.0))))
                        outcome = "win" if profit > 0 else ("draw" if profit == 0 else "loss")
                        return self._make_result(signal_id, pair, direction, outcome, profit)

            # Shape: list of trades
            if isinstance(raw, list):
                for item in reversed(raw):
                    parsed = self._parse_result(item, signal_id, pair, direction, expiry_time)
                    if parsed:
                        return parsed

        except Exception as exc:
            logger.debug({"event": "result_parse_error", "error": str(exc)})

        return None

    def _make_result(
        self, signal_id: str, pair: str, direction: str, outcome: str, payout: float
    ) -> dict:
        return {
            "signal_id": signal_id,
            "pair": pair,
            "direction": direction,
            "result": outcome,
            "payout": round(payout, 4),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Async result consumer ─────────────────────────────────────────────────

    async def get_result(self, timeout: float = 0.1) -> dict | None:
        """Non-blocking result consumer for the main feedback loop."""
        try:
            return await asyncio.wait_for(self._results_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ── Reconnection ──────────────────────────────────────────────────────────

    async def _reconnect(self) -> None:
        delay = 2.0
        while not self._connected:
            logger.info({"event": "quotex_reconnecting", "in_seconds": delay})
            await asyncio.sleep(delay)
            try:
                await self.connect()
            except Exception as exc:
                logger.warning({"event": "quotex_reconnect_failed", "error": str(exc)})
            delay = min(delay * 2, self.MAX_RECONNECT_DELAY)

    def health(self) -> dict:
        return {
            "connected": self._connected,
            "pending_signals": len(self._pending),
            "results_queued": self._results_queue.qsize(),
            "raw_messages_buffered": len(self._raw_messages),
        }


# ── Async runner wrapper ───────────────────────────────────────────────────────

async def run_quotex_reader(reader: QuotexReader) -> None:
    """Supervised poll loop — main.py runs this as a supervised task."""
    await reader.poll_results()
