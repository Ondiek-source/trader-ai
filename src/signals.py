"""
signals.py — Signal orchestration with martingale-aware confidence gating.

Responsibilities:
  - Per-pair 60-second cooldown enforcement
  - Trading window management (default 19h/day)
  - Daily target: stop after N wins OR target profit reached
  - Martingale integration: raise confidence threshold progressively after losses,
    reset after a win or after max consecutive losses
  - Pending signal tracking for result matching
  - Result processing: update MartingaleTracker and DailySession

Direction mapping: "UP" → "buy", "DOWN" → "sell" (Quotex convention)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DIRECTION_MAP = {"UP": "buy", "DOWN": "sell"}

# Pair normalization: internal format → webhook symbol format
PAIR_TO_SYMBOL: dict[str, str] = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "XAU_USD": "XAUUSD",
}


def normalize_symbol(pair: str) -> str:
    """Convert internal pair format to webhook symbol (e.g. EUR_USD → EURUSD)."""
    return PAIR_TO_SYMBOL.get(pair, pair.replace("_", "").replace("/", ""))


# ── Daily Session ─────────────────────────────────────────────────────────────

class DailySession:
    """
    Tracks a single 19-hour trading session.

    Stops issuing signals once:
      - wins >= target_wins, OR
      - net_profit >= target_profit (if configured)
    """

    def __init__(
        self,
        target_wins: int = 10,
        target_profit: float | None = 20.0,
        window_hours: int = 19,
    ) -> None:
        self.target_wins = target_wins
        self.target_profit = target_profit
        self.window_hours = window_hours

        self.session_start: datetime | None = None
        self.wins: int = 0
        self.losses: int = 0
        self.net_profit: float = 0.0
        self.signals_fired: int = 0

    def start(self) -> None:
        self.session_start = datetime.now(timezone.utc)
        self.wins = 0
        self.losses = 0
        self.net_profit = 0.0
        self.signals_fired = 0
        logger.info(
            {
                "event": "session_started",
                "session_start": self.session_start.isoformat(),
                "target_wins": self.target_wins,
                "target_profit": self.target_profit,
                "window_hours": self.window_hours,
            }
        )

    def is_active(self) -> bool:
        """True if within the trading window."""
        if self.session_start is None:
            return False
        elapsed = datetime.now(timezone.utc) - self.session_start
        return elapsed < timedelta(hours=self.window_hours)

    def is_target_reached(self) -> bool:
        if self.wins >= self.target_wins:
            return True
        if self.target_profit is not None and self.net_profit >= self.target_profit:
            return True
        return False

    def record_win(self, payout: float = 0.0) -> None:
        self.wins += 1
        self.net_profit += payout
        logger.info(
            {
                "event": "session_win",
                "wins": self.wins,
                "target_wins": self.target_wins,
                "net_profit": round(self.net_profit, 2),
                "target_profit": self.target_profit,
            }
        )

    def record_loss(self, stake: float = 0.0) -> None:
        self.losses += 1
        self.net_profit -= stake
        logger.info(
            {
                "event": "session_loss",
                "losses": self.losses,
                "net_profit": round(self.net_profit, 2),
            }
        )

    def record_draw(self) -> None:
        logger.info({"event": "session_draw"})

    def reset(self) -> None:
        self.start()

    def summary(self) -> dict:
        now = datetime.now(timezone.utc)
        elapsed = (now - self.session_start).total_seconds() if self.session_start else 0
        return {
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "elapsed_minutes": round(elapsed / 60, 1),
            "wins": self.wins,
            "losses": self.losses,
            "signals_fired": self.signals_fired,
            "net_profit": round(self.net_profit, 2),
            "target_wins": self.target_wins,
            "target_profit": self.target_profit,
            "window_hours": self.window_hours,
            "is_active": self.is_active(),
            "target_reached": self.is_target_reached(),
        }


# ── Signal Orchestrator ───────────────────────────────────────────────────────

class SignalOrchestrator:
    """
    Central gatekeeper between model predictions and webhook delivery.

    Signal flow (4 sequential gates):
      1. Session is active (within trading window)
      2. Daily target not yet reached
      3. Per-pair cooldown (60 s)
      4. Confidence >= martingale_tracker.current_threshold

    If all gates pass: build webhook payload, return it.
    """

    COOLDOWN_SECONDS = 60

    def __init__(
        self,
        config,
        martingale_tracker,
        daily_session: DailySession,
        webhook_key: str = "Ondiek",
    ) -> None:
        self._config = config
        self._mt = martingale_tracker
        self._session = daily_session
        self._webhook_key = webhook_key

        # {pair -> last_signal_time}
        self._last_signal: dict[str, datetime] = {}
        # {signal_id -> signal_payload}
        self.pending_signals: dict[str, dict] = {}

        self._stopped: bool = False  # manual stop via Telegram

    # ── Session management ────────────────────────────────────────────────────

    def ensure_session_active(self) -> None:
        """Call at the start of each processing cycle to auto-start/reset session."""
        if self._session.session_start is None:
            self._session.start()
        elif not self._session.is_active():
            logger.info({"event": "session_window_expired", "summary": self._session.summary()})
            # Session expired — auto-reset for next window (or wait for new day)
            # Do not auto-start; let main loop decide based on scheduling

    # ── Core signal gate ──────────────────────────────────────────────────────

    def try_signal(self, prediction: dict) -> dict | None:
        """
        Evaluate a model prediction against all gates.

        Returns webhook payload dict if signal fires, or None if suppressed.
        """
        pair = prediction.get("pair", "")
        confidence = float(prediction.get("confidence", 0.0))
        direction = prediction.get("direction", "")
        expiry_seconds = int(prediction.get("expiry_seconds", self._config.expiry_seconds))

        # Gate 0: manual stop
        if self._stopped:
            logger.info({"event": "signal_suppressed", "reason": "manual_stop", "pair": pair})
            return None

        # Gate 1: session active
        if not self._session.is_active():
            logger.debug({"event": "signal_suppressed", "reason": "session_inactive", "pair": pair})
            return None

        # Gate 2: daily target not reached
        if self._session.is_target_reached():
            logger.info(
                {
                    "event": "signal_suppressed",
                    "reason": "daily_target_reached",
                    "summary": self._session.summary(),
                }
            )
            return None

        # Gate 3: per-pair cooldown
        now = datetime.now(timezone.utc)
        last = self._last_signal.get(pair)
        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < self.COOLDOWN_SECONDS:
                logger.debug(
                    {
                        "event": "signal_suppressed",
                        "reason": "cooldown",
                        "pair": pair,
                        "wait_seconds": round(self.COOLDOWN_SECONDS - elapsed, 1),
                    }
                )
                return None

        # Gate 4: martingale confidence threshold
        threshold = self._mt.current_threshold
        if confidence < threshold:
            logger.info(
                {
                    "event": "signal_suppressed",
                    "reason": "confidence_below_threshold",
                    "pair": pair,
                    "confidence": confidence,
                    "threshold": threshold,
                    "streak": self._mt.current_streak,
                }
            )
            return None

        # All gates passed — build payload
        signal_id = str(uuid.uuid4())
        symbol = normalize_symbol(pair)
        side = DIRECTION_MAP.get(direction.upper(), "buy")

        # Exact payload the webhook receiver expects — nothing more
        payload = {
            "side": side,
            "symbol": symbol,
            "key": self._webhook_key,
        }

        # Track internally (not sent to webhook)
        self._last_signal[pair] = now
        self._session.signals_fired += 1
        self.pending_signals[signal_id] = {
            "signal_id": signal_id,
            "payload": payload,
            "pair": pair,
            "direction": direction,
            "confidence": confidence,
            "expiry_seconds": expiry_seconds,
            "fired_at": now,
            "symbol": symbol,
        }

        logger.info(
            {
                "event": "signal_fired",
                "signal_id": signal_id,
                "pair": pair,
                "side": side,
                "symbol": symbol,
                "confidence": confidence,
                "threshold": threshold,
                "streak": self._mt.current_streak,
                "session_wins": self._session.wins,
                "session_losses": self._session.losses,
            }
        )
        return payload

    # ── Result feedback ───────────────────────────────────────────────────────

    def on_result(self, result: dict) -> None:
        """
        Process a trade result from Quotex / external feedback.

        result dict:
          {pair, direction, result: "win"|"loss"|"draw", payout: float,
           signal_id: str (optional)}
        """
        outcome = result.get("result", "").lower()
        payout = float(result.get("payout", 0.0))
        pair = result.get("pair", "")

        if outcome == "win":
            self._mt.record_result(win=True)
            self._session.record_win(payout=payout)
        elif outcome == "loss":
            self._mt.record_result(win=False)
            self._session.record_loss(stake=0.0)  # stake tracked by Quotex bot
        elif outcome == "draw":
            self._session.record_draw()
        else:
            logger.warning({"event": "unknown_result_outcome", "result": result})
            return

        # Remove from pending
        signal_id = result.get("signal_id")
        if signal_id and signal_id in self.pending_signals:
            del self.pending_signals[signal_id]

        logger.info(
            {
                "event": "result_processed",
                "pair": pair,
                "outcome": outcome,
                "payout": payout,
                "streak": self._mt.current_streak,
                "threshold": self._mt.current_threshold,
                "session": self._session.summary(),
            }
        )

        # Check if daily target now reached
        if self._session.is_target_reached():
            logger.info({"event": "daily_target_reached", "session": self._session.summary()})

    # ── Manual controls (Telegram commands) ──────────────────────────────────

    def stop(self) -> str:
        self._stopped = True
        logger.warning({"event": "manual_stop", "source": "telegram"})
        return "Signal generation stopped."

    def resume(self) -> str:
        self._stopped = False
        logger.info({"event": "manual_resume", "source": "telegram"})
        return "Signal generation resumed."

    def start_session(self) -> str:
        self._stopped = False
        self._session.start()
        return f"New session started. Target: {self._session.target_wins} wins / ${self._session.target_profit}."

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "stopped": self._stopped,
            "martingale_streak": self._mt.current_streak,
            "confidence_threshold": self._mt.current_threshold,
            "session": self._session.summary(),
            "pending_signals": len(self.pending_signals),
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def create_orchestrator(config, martingale_tracker) -> SignalOrchestrator:
    """Wire up SignalOrchestrator from config."""
    session = DailySession(
        target_wins=config.daily_trade_target,
        target_profit=config.target_net_profit,
        window_hours=config.trading_window_hours,
    )
    session.start()
    webhook_key = getattr(config, "webhook_key", "Ondiek")
    return SignalOrchestrator(
        config=config,
        martingale_tracker=martingale_tracker,
        daily_session=session,
        webhook_key=webhook_key,
    )
