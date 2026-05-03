"""
src/ml_engine/labeler.py — The Ground Truth.

Role: Compute static binary classification labels for Classical ML and
Deep Learning models, and dynamic reward signals for Reinforcement
Learning environments.

Design rationale
----------------
Labels and rewards are deliberately separated into two classes:

    Labeler           — produces a static Y column (0 or 1) aligned to a
                        FeatureMatrix for supervised training. Used by all
                        Phase 1 (Classical ML) and Phase 2 (Deep Learning)
                        trainers via DataShaper.

    RewardCalculator  — produces a scalar reward at each RL environment
                        step. Used exclusively by Phase 3 (RL) trainers.
                        Wraps the same price-direction logic as Labeler
                        but is called dynamically at runtime rather than
                        pre-computed over a batch. No gate compliance
                        checks — gates have been removed from the system.

Expiry alignment
----------------
BINARY_EXPIRY_RULES in features.py maps expiry keys to the feature
columns relevant for each expiry window. It does NOT store seconds.
This file owns the authoritative seconds-per-expiry mapping via
_EXPIRY_SECONDS, which must be kept in sync with any changes to the
expiry taxonomy in features.py.

Version safety
--------------
Every label batch produced by Labeler.compute_labels() is accompanied
by a metadata dict that records the feature engineering version string
(_VERSION from features.py). If features.py is updated and _VERSION
bumps, any label set generated against the old version is stale. The
trainer must validate metadata["feature_version"] == _VERSION before
using a cached label set.

Public API
----------
    Labeler(expiry_key)                          -> Labeler
    Labeler.compute_labels(df)                   -> pd.Series
    Labeler.get_metadata(labels, symbol)         -> dict
    RewardCalculator(payout_ratio)               -> RewardCalculator
    RewardCalculator.calculate_reward(...)       -> float
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml_engine.features import BINARY_EXPIRY_RULES, _VERSION, _ATR_PERIOD
from core.exceptions import LabelerError

logger = logging.getLogger(__name__)


# ── Expiry Seconds Map ───────────────────────────────────────────────────────
# Authoritative mapping from expiry key to duration in seconds.
# BINARY_EXPIRY_RULES in features.py maps the same keys to feature column
# lists — it does not store durations. These two dicts are complementary.
# If a new expiry key is added to BINARY_EXPIRY_RULES, add it here too.

_EXPIRY_SECONDS: dict[str, int] = {
    "1_MIN": 60,
    "5_MIN": 300,
    "15_MIN": 900,
}

# M1 bars are 60-second candles. Lookahead in bars = seconds / bar_duration.
_M1_BAR_SECONDS: int = 60


# ── Taxonomy Drift Guard ─────────────────────────────────────────────────────
# Fail at import time if BINARY_EXPIRY_RULES and _EXPIRY_SECONDS have drifted
# out of sync. This converts a silent runtime failure in Labeler.__init__()
# into an immediate ImportError that is caught by any test or boot sequence.
assert set(BINARY_EXPIRY_RULES.keys()) == set(_EXPIRY_SECONDS.keys()), (
    f"TAXONOMY DRIFT DETECTED: BINARY_EXPIRY_RULES and _EXPIRY_SECONDS "
    f"have different keys.\n"
    f"  BINARY_EXPIRY_RULES keys : {sorted(BINARY_EXPIRY_RULES.keys())}\n"
    f"  _EXPIRY_SECONDS keys     : {sorted(_EXPIRY_SECONDS.keys())}\n"
    f"FIX: When adding a new expiry key to features.py BINARY_EXPIRY_RULES, "
    f"add the matching duration entry to _EXPIRY_SECONDS in labeler.py."
)


# ── Labeler ──────────────────────────────────────────────────────────────────


class Labeler:
    """
    Source of truth for binary classification labels across the system.

    Computes a static Y column over a historical bar DataFrame by
    comparing each bar's close price to the close price N bars ahead,
    where N is derived from the expiry key. The result is a binary
    series: 1 if price rose (CALL win), 0 if price fell or was flat
    (PUT win or neutral).

    SKIP is not a label produced here. SKIP is an inference-time
    decision made by the model when its output confidence falls below
    a threshold. The Labeler only produces the ground-truth binary
    outcome of the trade direction.

    The expiry key must be one of the keys defined in both
    BINARY_EXPIRY_RULES (features.py) and _EXPIRY_SECONDS (this file).
    These two dicts are complementary — BINARY_EXPIRY_RULES owns the
    feature column lists, _EXPIRY_SECONDS owns the durations.

    Attributes:
        expiry_key:      One of "1_MIN", "5_MIN", "15_MIN".
        expiry_seconds:  Duration of the expiry window in seconds.
        lookahead_bars:  Number of M1 bars to look ahead for the label.

    Example:
        >>> labeler = Labeler(expiry_key="5_MIN")
        >>> labels = labeler.compute_labels(bar_dataframe)
        >>> meta = labeler.get_metadata(labels, symbol="EUR_USD")
    """

    def __init__(self, expiry_key: str = "1_MIN", atr_threshold: float = 0.0) -> None:
        """
        Initialise the Labeler for a specific expiry window.

        Validates that the expiry key exists in both BINARY_EXPIRY_RULES
        and _EXPIRY_SECONDS to catch any taxonomy drift between files
        at construction time rather than silently at label generation.

        Args:
            expiry_key: Expiry window identifier. Must be one of
                "1_MIN", "5_MIN", "15_MIN".

        Raises:
            ValueError: If expiry_key is not in BINARY_EXPIRY_RULES or
                not in _EXPIRY_SECONDS.
        """
        if expiry_key not in BINARY_EXPIRY_RULES:
            raise ValueError(
                f"[!] Invalid expiry_key: '{expiry_key}'. "
                f"Must be one of {list(BINARY_EXPIRY_RULES.keys())}. "
                f"Verify BINARY_EXPIRY_RULES in features.py."
            )
        if expiry_key not in _EXPIRY_SECONDS:  # pragma: no cover — taxonomy drift guard
            # Only reachable if a key is added to BINARY_EXPIRY_RULES in
            # features.py without a corresponding entry in _EXPIRY_SECONDS here.
            # Both dicts must be kept in sync; this guard catches the drift at
            # construction time rather than silently at label generation.
            raise ValueError(
                f"[!] expiry_key '{expiry_key}' exists in BINARY_EXPIRY_RULES "
                f"but is missing from _EXPIRY_SECONDS in labeler.py. "
                f"Add the duration mapping before proceeding."
            )

        self.expiry_key: str = expiry_key
        self.expiry_seconds: int = _EXPIRY_SECONDS[expiry_key]
        self.lookahead_bars: int = self.expiry_seconds // _M1_BAR_SECONDS
        self.atr_threshold: float = atr_threshold
        logger.debug(
            "Labeler initialised: expiry_key=%s, expiry_seconds=%d, "
            "lookahead_bars=%d, atr_threshold=%s",
            self.expiry_key,
            self.expiry_seconds,
            self.lookahead_bars,
            self.atr_threshold,
        )

    # ── Label Generation ─────────────────────────────────────────────────────

    def compute_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute binary classification labels over a bar DataFrame.

        If atr_threshold > 0, only labels moves where the price change
        magnitude exceeds atr_threshold × ATR are kept. Rows with smaller
        moves are marked as -1 (SKIP) and should be filtered out during training.

        Args:
            df: OHLCV DataFrame with DatetimeIndex. Must have 'close' column.
            If atr_threshold > 0, must also have 'ATR' column.

        Returns:
            pd.Series: Integer series with values in {0, 1, -1} where:
                1 = CALL (price rose significantly)
                0 = PUT (price fell significantly)
                -1 = SKIP (move too small to be reliable)
        """
        if df.empty:
            raise LabelerError(
                "compute_labels() received an empty DataFrame.",
                stage="compute_labels",
            )

        if "close" not in df.columns:
            raise ValueError(
                "compute_labels() requires a 'close' column. "
                f"Columns present: {list(df.columns)}"
            )

        if self.lookahead_bars >= len(df):
            raise LabelerError(
                f"lookahead_bars ({self.lookahead_bars}) >= "
                f"DataFrame length ({len(df)}). "
                f"Provide more historical bars for expiry_key='{self.expiry_key}'.",
                stage="compute_labels",
            )

        # Future close price (N bars ahead)
        future_close = df["close"].shift(-self.lookahead_bars)

        # Direction label: 1 = up, 0 = down or flat
        direction_labels = (future_close > df["close"]).astype(np.int32)

        # If no filtering, return standard labels (drop NaN tail)
        if self.atr_threshold <= 0:
            labels = direction_labels.dropna().rename("label")
            logger.debug(f"Standard labels: {len(labels):,} rows")
            return labels

        if "ATR" not in df.columns:
            tr = pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df = df.copy()
            df["ATR"] = tr.ewm(
                alpha=1 / _ATR_PERIOD, min_periods=_ATR_PERIOD, adjust=False
            ).mean()

        # Calculate move magnitude (absolute price change)
        move_magnitude = (future_close - df["close"]).abs()

        # Minimum move required to keep the label
        min_move = df["ATR"] * self.atr_threshold

        # Apply filter: -1 for moves below threshold
        labels = (
            direction_labels.where(move_magnitude >= min_move, other=-1)
            .dropna()
            .rename("label")
        )

        labels = labels.astype(np.int32)

        kept = (labels >= 0).sum()
        skipped = (labels == -1).sum()
        logger.info(
            f"Labels with threshold {self.atr_threshold}×ATR: "
            f"{kept:,} kept ({kept/len(labels)*100:.1f}%), "
            f"{skipped:,} skipped ({skipped/len(labels)*100:.1f}%)"
        )

        return labels

    # ── Metadata ─────────────────────────────────────────────────────────────

    def get_metadata(self, labels: pd.Series, symbol: str) -> dict[str, Any]:
        """
        Produce a versioned metadata snapshot for a computed label set.

        Records the feature engineering version, class balance, and
        provenance information alongside the label set. Trainers must
        validate metadata["feature_version"] against the current
        _VERSION before using any cached label set — a version mismatch
        means the labels were generated against a different feature
        schema and must be regenerated.

        Args:
            labels: Label Series produced by compute_labels().
            symbol: Currency pair the labels were generated for,
                    e.g. "EUR_USD".

        Returns:
            dict[str, Any]: Metadata dict with the following keys:
                feature_version    (str)   — _VERSION from features.py
                expiry_key         (str)   — e.g. "5_MIN"
                expiry_seconds     (int)   — e.g. 300
                lookahead_bars     (int)   — e.g. 5
                symbol             (str)   — e.g. "EUR_USD"
                label_generated_at (str)   — UTC ISO-8601 timestamp
                total_rows         (int)   — number of labeled bars
                call_count         (int)   — number of CALL labels (1)
                put_count          (int)   — number of PUT labels (0)
                call_pct           (float) — fraction of CALL labels

        Raises:
            ValueError: If labels is empty.
        """
        if labels.empty:
            raise ValueError(
                "[!] get_metadata() received an empty label Series. "
                "Run compute_labels() before requesting metadata."
            )

        total: int = len(labels)
        call_count: int = int(labels.sum())
        put_count: int = total - call_count
        call_pct: float = round(call_count / total, 4) if total > 0 else 0.0

        metadata: dict[str, Any] = {
            "feature_version": _VERSION,
            "expiry_key": self.expiry_key,
            "expiry_seconds": self.expiry_seconds,
            "lookahead_bars": self.lookahead_bars,
            "atr_threshold": self.atr_threshold,
            "symbol": symbol,
            "label_generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_rows": total,
            "call_count": call_count,
            "put_count": put_count,
            "call_pct": call_pct,
        }

        logger.info(
            "[^] Label metadata: symbol=%s expiry=%s total=%d "
            "calls=%d (%.1f%%) puts=%d (%.1f%%) version=%s",
            symbol,
            self.expiry_key,
            total,
            call_count,
            call_pct * 100,
            put_count,
            (1 - call_pct) * 100,
            _VERSION,
        )

        return metadata

    def __repr__(self) -> str:
        return (
            f"Labeler("
            f"expiry_key={self.expiry_key!r}, "
            f"expiry_seconds={self.expiry_seconds}, "
            f"lookahead_bars={self.lookahead_bars})"
        )


# ── RewardCalculator ─────────────────────────────────────────────────────────


class RewardCalculator:
    """
    Dynamic reward signal generator for Reinforcement Learning environments.

    Called at each step of an RL training episode to return a scalar
    reward based on the agent's action and whether the direction was
    correct.

    This class does not produce labels — it produces rewards. The
    distinction is critical: labels are pre-computed over a static
    batch (Labeler); rewards are computed dynamically at each RL
    environment step based on the agent's live action.

    Action space contract (must match the RL environment definition):
        0 = CALL  (predict price rises)
        1 = PUT   (predict price falls)
        2 = SKIP  (abstain from trading this bar)

    Reward structure:
        SKIP action:          0.0  (no gain, no loss)
        Correct direction:   +1.0 * payout_ratio
        Wrong direction:     -1.0 (full stake loss)

    Attributes:
        payout_ratio: Broker payout on a winning binary option trade.
            Typical range 0.70 to 0.92 depending on asset and broker.

    Example:
        >>> rc = RewardCalculator(payout_ratio=0.85)
        >>> reward = rc.calculate_reward(action=0, is_correct=True)
        >>> reward  # 0.85
        0.85
    """

    def __init__(self, payout_ratio: float = 0.85) -> None:
        """
        Initialise the RewardCalculator with a broker payout ratio.

        Args:
            payout_ratio: Fraction of stake returned as profit on a
                winning trade. Must be in the range (0.0, 1.0].
                Default 0.85 represents an 85% payout (stake back
                plus 85% profit).

        Raises:
            ValueError: If payout_ratio is not in (0.0, 1.0].
        """
        if not (0.0 < payout_ratio <= 1.0):
            raise ValueError(
                f"[%] Invalid payout_ratio: {payout_ratio}. "
                f"Must be in the range (0.0, 1.0]. "
                f"Typical broker range is 0.70 to 0.92."
            )
        self.payout_ratio: float = payout_ratio

        logger.debug(
            "[^] RewardCalculator initialised: payout_ratio=%.2f",
            self.payout_ratio,
        )

    # ── Reward Computation ───────────────────────────────────────────────────

    def calculate_reward(
        self,
        action: int,
        is_correct: bool,
    ) -> float:
        """
        Compute the scalar reward for a single RL environment step.

        Applies the reward structure defined in the class docstring.

        Args:
            action:     Agent action. Must be 0 (CALL), 1 (PUT), or
                        2 (SKIP). Any other value raises ValueError.
            is_correct: True if the agent's direction prediction
                        matched the actual price movement at expiry.
                        Ignored when action == 2 (SKIP).

        Returns:
            float: Scalar reward. Positive values indicate a
                beneficial outcome; negative values indicate a loss
                or a penalised action.

        Raises:
            ValueError: If action is not in {0, 1, 2}.
        """
        if action not in (0, 1, 2):
            raise ValueError(
                f"[!] Invalid action: {action}. "
                f"Must be 0 (CALL), 1 (PUT), or 2 (SKIP)."
            )

        # SKIP: agent abstains. No financial outcome, no gate adjustment.
        if action == 2:
            return 0.0

        # Direction reward: correct = payout, wrong = full loss.
        reward: float = self.payout_ratio if is_correct else -1.0

        logger.debug(
            f"calculate_reward(): action={action} is_correct={is_correct} reward={reward:.4f}",
        )

        return float(reward)

    def __repr__(self) -> str:
        return f"RewardCalculator(payout_ratio={self.payout_ratio:.2f})"
