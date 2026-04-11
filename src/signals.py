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
from typing import Any

logger = logging.getLogger(__name__)

DIRECTION_MAP: dict[str, str] = {"UP": "buy", "DOWN": "sell"}

# Pair normalization: internal format → webhook symbol format
PAIR_TO_SYMBOL: dict[str, str] = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "XAU_USD": "XAUUSD",
}

# How long a signal stays in the pending queue before expiring.
PENDING_SIGNAL_TTL_SECONDS = 300  # 5 minutes


def normalize_symbol(pair: str, otc: bool = False) -> str:
    """Convert internal pair format to webhook symbol (``EUR_USD`` → ``EURUSD``)."""
    base = PAIR_TO_SYMBOL.get(pair, pair.replace("_", "").replace("/", ""))
    return f"{base}_otc" if otc else base


# ── Daily Session ─────────────────────────────────────────────────────────────


class DailySession:
    """
    Tracks a single trading session.

    Stops issuing signals once:
        - wins >= target_wins, OR
        - net_profit >= target_profit (if configured)

    Args:
        target_wins: Number of wins that ends the session.
        target_profit: Dollar profit target (``None`` = disabled).
        window_hours: Maximum session duration in hours.
    """

    def __init__(
        self,
        target_wins: int = 30,
        target_profit: float | None = 450.0,
        window_hours: int = 19,
    ) -> None:
        self.target_wins = target_wins
        self.target_profit = target_profit
        self.window_hours = window_hours

        self.session_start: datetime | None = None
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0
        self.net_profit: float = 0.0
        self.signals_fired: int = 0

    def start(self) -> None:
        """Begin a new session, resetting all counters."""
        self.session_start = datetime.now(timezone.utc)
        self.wins = 0
        self.losses = 0
        self.draws = 0
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
        """Return ``True`` if the session window has not expired."""
        if self.session_start is None:
            return False
        elapsed = datetime.now(timezone.utc) - self.session_start
        return elapsed < timedelta(hours=self.window_hours)

    def is_target_reached(self) -> bool:
        """Return ``True`` if the win or profit target has been hit."""
        if self.wins >= self.target_wins:
            return True
        if self.target_profit is not None and self.net_profit >= self.target_profit:
            return True
        return False

    def record_win(self, payout: float = 0.0) -> None:
        """Record a winning trade."""
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
        """Record a losing trade."""
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
        """Record a draw (stake returned)."""
        self.draws += 1
        logger.info({"event": "session_draw", "draws": self.draws})

    def reset(self) -> None:
        """Alias for :meth:`start`."""
        self.start()

    def summary(self) -> dict[str, Any]:
        """Return a serialisable snapshot of session state."""
        now = datetime.now(timezone.utc)
        elapsed = (
            (now - self.session_start).total_seconds() if self.session_start else 0.0
        )
        return {
            "session_start": (
                self.session_start.isoformat() if self.session_start else None
            ),
            "elapsed_minutes": round(elapsed / 60, 1),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
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

    Args:
        config: Application :class:`~config.Config`.
        martingale_tracker: :class:`MartingaleTracker` instance.
        daily_session: :class:`DailySession` instance.
        webhook_key: Static auth key included in every webhook payload.
    """

    COOLDOWN_SECONDS = 60

    def __init__(
        self,
        config: Any,
        martingale_tracker: Any,
        daily_session: DailySession,
        webhook_key: str = "Ondiek",
    ) -> None:
        self._config = config
        self._mt = martingale_tracker
        self._session = daily_session
        self._webhook_key = webhook_key

        # {pair -> last_signal_time}
        self._last_signal: dict[str, datetime] = {}
        # {signal_id -> signal_metadata}
        self.pending_signals: dict[str, dict[str, Any]] = {}

        self._stopped: bool = False

    # ── Session management ────────────────────────────────────────────────────

    def ensure_session_active(self) -> None:
        """Auto-start session if it has never started or has expired."""
        if self._session.session_start is None:
            self._session.start()
        elif not self._session.is_active():
            logger.info(
                {
                    "event": "session_window_expired",
                    "summary": self._session.summary(),
                }
            )
            # Do not auto-restart — let the scheduling loop or /start command
            # decide when to begin a new session.

    # ── Pending signal cleanup ────────────────────────────────────────────────

    def _expire_old_signals(self) -> None:
        """Remove pending signals older than ``PENDING_SIGNAL_TTL_SECONDS``."""
        now = datetime.now(timezone.utc)
        expired = [
            sid
            for sid, sig in self.pending_signals.items()
            if (now - sig["fired_at"]).total_seconds() > PENDING_SIGNAL_TTL_SECONDS
        ]
        for sid in expired:
            del self.pending_signals[sid]
        if expired:
            logger.info({"event": "pending_signals_expired", "count": len(expired)})

    # ── Core signal gate ──────────────────────────────────────────────────────

    def try_signal(self, prediction: dict[str, Any]) -> dict[str, Any] | None:
        """
        Evaluate a model prediction against all gates.

        Args:
            prediction: Dict from :meth:`ModelManager.predict` with keys
                ``pair``, ``direction``, ``confidence``, ``expiry_seconds``.

        Returns:
            Webhook payload dict if signal fires, or ``None`` if suppressed.
        """
        pair = prediction.get("pair", "")
        confidence = float(prediction.get("confidence", 0.0))
        direction = prediction.get("direction", "")
        expiry_seconds = int(
            prediction.get("expiry_seconds", self._config.expiry_seconds)
        )

        # Diagnostic: log every attempt so we can see which gate blocks
        logger.info(
            {
                "event": "signal_attempt",
                "pair": pair,
                "direction": direction,
                "confidence": confidence,
                "stopped": self._stopped,
                "session_active": self._session.is_active(),
                "target_reached": self._session.is_target_reached(),
                "threshold": self._mt.current_threshold,
                "streak": self._mt.current_streak,
            }
        )

        # Gate 0: manual stop
        if self._stopped:
            logger.debug(
                {"event": "signal_suppressed", "reason": "manual_stop", "pair": pair}
            )
            return None

        # Gate 1: session active
        if not self._session.is_active():
            logger.debug(
                {
                    "event": "signal_suppressed",
                    "reason": "session_inactive",
                    "pair": pair,
                }
            )
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

        # ── All gates passed — build payload ──────────────────────────────
        signal_id = str(uuid.uuid4())
        otc = bool(prediction.get("otc", False))
        symbol = normalize_symbol(pair, otc=otc)
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

        # Expire stale pending signals
        self._expire_old_signals()

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

    def on_result(self, result: dict[str, Any]) -> None:
        """
        Process a trade result from Quotex / external feedback.

        Args:
            result: Dict with keys ``pair``, ``direction``,
                ``result`` (``"win"`` / ``"loss"`` / ``"draw"``),
                ``payout`` (float), ``stake`` (float, optional),
                ``signal_id`` (str, optional).
        """
        outcome = result.get("result", "").lower()
        payout = float(result.get("payout", 0.0))
        stake = float(result.get("stake", 0.0))
        pair = result.get("pair", "")

        if outcome == "win":
            self._mt.record_result(win=True)
            self._session.record_win(payout=payout)
        elif outcome == "loss":
            self._mt.record_result(win=False)
            self._session.record_loss(stake=stake)
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
                "stake": stake,
                "streak": self._mt.current_streak,
                "threshold": self._mt.current_threshold,
                "session": self._session.summary(),
            }
        )

        if self._session.is_target_reached():
            logger.info(
                {"event": "daily_target_reached", "session": self._session.summary()}
            )

    # ── Manual controls (Telegram commands) ────────────────────────────────────

    def stop(self) -> str:
        """Halt signal generation immediately."""
        self._stopped = True
        logger.warning({"event": "manual_stop", "source": "telegram"})
        return "Signal generation stopped."

    def resume(self) -> str:
        """Resume signal generation, restarting the session if needed."""
        self._stopped = False
        if not self._session.is_active():
            self._session.start()
            logger.info(
                {"event": "manual_resume_with_new_session", "source": "telegram"}
            )
            return (
                f"Signal generation resumed. "
                f"New session started: {self._session.target_wins} wins / "
                f"${self._session.target_profit} target."
            )
        logger.info({"event": "manual_resume", "source": "telegram"})
        return "Signal generation resumed."

    def start_session(self) -> str:
        """Force-start a new session and resume signal generation."""
        self._stopped = False
        self._session.start()
        self._last_signal.clear()
        return (
            f"New session started. "
            f"Target: {self._session.target_wins} wins / "
            f"${self._session.target_profit}."
        )

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return a serialisable snapshot of orchestrator state."""
        return {
            "stopped": self._stopped,
            "martingale_streak": self._mt.current_streak,
            "confidence_threshold": self._mt.current_threshold,
            "session": self._session.summary(),
            "pending_signals": len(self.pending_signals),
        }


# ── Factory ───────────────────────────────────────────────────────────────────


def create_orchestrator(config: Any, martingale_tracker: Any) -> SignalOrchestrator:
    """
    Wire up :class:`SignalOrchestrator` from config.

    Args:
        config: Application :class:`~config.Config`.
        martingale_tracker: :class:`MartingaleTracker` instance.

    Returns:
        A started :class:`SignalOrchestrator` ready for use.
    """
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
