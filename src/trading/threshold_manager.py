"""
trading/threshold_manager.py — Dynamic confidence-threshold state machine.

ThresholdManager tracks the current Martingale streak and adjusts the
confidence threshold that SignalGenerator must clear before a trade is fired.

Rules
-----
- After each **loss**: streak += 1; effective_threshold += step
- After each **win**: streak resets to 0; threshold resets to base_threshold
- When streak reaches max_streak: halted = True (kill switch)

The manager is pure state — no I/O, no threading, no logging side-effects.
All mutations are single-threaded by design; the caller (LiveTrader) is
responsible for sequencing calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ThresholdManager:
    """
    Stateful confidence-threshold controller for the Martingale circuit breaker.

    Args:
        base_threshold: Starting (and reset) confidence threshold (e.g. 0.58).
        step:           Amount added to the threshold per consecutive loss.
        max_streak:     Number of consecutive losses that trigger the kill switch.

    Attributes:
        effective_threshold: Current threshold SignalGenerator must clear.
        streak:              Current consecutive-loss count.
        halted:              True when the kill switch has been activated.
    """

    base_threshold: float
    step: float
    max_streak: int

    # mutable state — set via __post_init__ so frozen=False is not required
    effective_threshold: float = 0.0
    streak: int = 0
    halted: bool = False

    _history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.effective_threshold = self.base_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def on_win(self) -> None:
        """Record a winning trade and reset streak + threshold."""
        self._history.append(
            {
                "event": "win",
                "streak_before": self.streak,
                "threshold_before": self.effective_threshold,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.streak = 0
        self.effective_threshold = self.base_threshold
        self.halted = False

    def on_loss(self) -> None:
        """
        Record a losing trade, advance the streak, and raise the threshold.

        Activates the kill switch when ``streak == max_streak``.
        """
        self._history.append(
            {
                "event": "loss",
                "streak_before": self.streak,
                "threshold_before": self.effective_threshold,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.streak += 1
        self.effective_threshold = min(
            self.base_threshold + self.streak * self.step,
            0.999,  # hard cap — never exactly 1.0
        )
        if self.streak >= self.max_streak:
            self.halted = True

    def reset(self) -> None:
        """
        Unconditional full reset (e.g. on session start or manual operator override).

        Clears streak, resets threshold, deactivates kill switch, and purges history.
        """
        self.streak = 0
        self.effective_threshold = self.base_threshold
        self.halted = False
        self._history.clear()

    def get_threshold(self) -> float:
        """Return the current effective confidence threshold."""
        return self.effective_threshold

    def is_halted(self) -> bool:
        """Return True when the kill switch is active."""
        return self.halted

    def get_state(self) -> dict[str, Any]:
        """
        Return a snapshot of the current manager state suitable for dashboard
        injection or structured logging.

        Returns:
            dict with keys: streak, effective_threshold, base_threshold,
            step, max_streak, halted, history_len.
        """
        return {
            "streak": self.streak,
            "effective_threshold": self.effective_threshold,
            "base_threshold": self.base_threshold,
            "step": self.step,
            "max_streak": self.max_streak,
            "halted": self.halted,
            "history_len": len(self._history),
        }
