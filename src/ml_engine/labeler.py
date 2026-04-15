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
                        pre-computed over a batch.

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

from ml_engine.features import BINARY_EXPIRY_RULES, _VERSION

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


# ── Custom Exception ─────────────────────────────────────────────────────────


class LabelerError(Exception):
    """
    Raised when the Labeler cannot produce a valid label set.

    Distinct from ValueError (which signals a caller contract violation
    such as a malformed input DataFrame) — LabelerError signals a
    runtime failure inside the labeling pipeline that the trainer must
    handle explicitly.

    Attributes:
        stage: The pipeline stage that failed (e.g. "compute_labels").
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return f"LabelerError(stage={self.stage!r}, message={str(self)!r})"


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

    def __init__(self, expiry_key: str = "1_MIN") -> None:
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

        logger.debug(
            "[^] Labeler initialised: expiry_key=%s, expiry_seconds=%d, "
            "lookahead_bars=%d",
            self.expiry_key,
            self.expiry_seconds,
            self.lookahead_bars,
        )

    # ── Label Generation ─────────────────────────────────────────────────────

    def compute_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute binary classification labels over a bar DataFrame.

        For each bar at index t, the label is 1 if the close price at
        t + lookahead_bars is strictly greater than the close price at
        t, and 0 otherwise. The final lookahead_bars rows are dropped
        because no future close exists to compare against — training on
        these rows would introduce NaN-derived corruption.

        The returned Series index is aligned to the input DataFrame
        index, minus the tail rows that were dropped. Callers must
        align the FeatureMatrix timestamps to this index before
        passing both to DataShaper.

        Args:
            df: OHLCV DataFrame with a DatetimeIndex and at minimum a
                "close" column. Typically the output of
                FeatureEngineer.transform() cast to a DataFrame, or
                the raw bars DataFrame before feature engineering.

        Returns:
            pd.Series: Integer series of dtype int32 with values in
                {0, 1}. Index matches the input DataFrame index minus
                the final lookahead_bars rows. Name is set to "label"
                for downstream alignment clarity.

        Raises:
            ValueError: If "close" column is absent from df.
            LabelerError: If df is empty, or if all rows are dropped
                after NaN removal (e.g. lookahead_bars >= len(df)).
        """
        if df.empty:
            raise LabelerError(
                "compute_labels() received an empty DataFrame. "
                "Ensure historical bars are loaded before labeling.",
                stage="compute_labels",
            )

        if "close" not in df.columns:
            raise ValueError(
                "[!] compute_labels() requires a 'close' column. "
                f"Columns present: {list(df.columns)}"
            )

        if self.lookahead_bars >= len(df):
            raise LabelerError(
                f"[%] lookahead_bars ({self.lookahead_bars}) >= "
                f"DataFrame length ({len(df)}). "
                f"Provide more historical bars for expiry_key='{self.expiry_key}'.",
                stage="compute_labels",
            )

        # Shift close price backwards by lookahead_bars to align the
        # future close with the current bar index. shift(-N) moves values
        # N positions earlier in the index, so df["close"].shift(-N).iloc[t]
        # gives the close price at t + N.
        future_close: pd.Series = df["close"].shift(-self.lookahead_bars)

        # Binary label: 1 = CALL win (price rose), 0 = PUT win or flat.
        # Cast to int32 to minimise memory footprint across large matrices.
        raw_labels: pd.Series = (future_close > df["close"]).astype(np.int32)

        # Drop the tail rows where future_close is NaN. These are the last
        # lookahead_bars rows where no outcome is observable.
        labels: pd.Series = raw_labels.dropna().rename("label")

        # Sanity guard: if dropna removed everything, the DataFrame was too
        # short relative to the lookahead window.
        # NOTE: In practice this branch is unreachable — astype(np.int32) is
        # applied before dropna(), which converts NaN comparisons to 0 (False).
        # int32 cannot hold NaN so dropna() is a no-op. The lookahead_bars >=
        # len(df) guard above already blocks the edge case that would otherwise
        # make labels empty. Retained as a defensive contract.
        if labels.empty:  # pragma: no cover — defensive contract; see note above
            raise LabelerError(
                f"[%] All rows dropped after NaN removal in compute_labels(). "
                f"DataFrame had {len(df)} rows, lookahead_bars={self.lookahead_bars}. "
                f"Provide a longer history window.",
                stage="compute_labels",
            )

        dropped: int = len(df) - len(labels)
        logger.debug(
            "[^] compute_labels(): %d labels produced, %d tail rows dropped "
            "(lookahead_bars=%d, expiry_key=%s).",
            len(labels),
            dropped,
            self.lookahead_bars,
            self.expiry_key,
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
    reward based on the agent's action, whether the direction was
    correct, and whether the trade eligibility gates were satisfied.

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
        Gate passed bonus:   +0.1 (trade was in a high-quality setup)
        Gate failed penalty: -0.2 (agent traded through a bad setup)

    The gate compliance bonus and penalty apply on top of the
    direction reward. A correct trade through a failing gate yields
    payout_ratio + (-0.2). A wrong trade through a passing gate
    yields -1.0 + 0.1. This structure teaches the agent to avoid
    trading in low-quality market conditions even when it guesses
    the direction correctly.

    Attributes:
        payout_ratio: Broker payout on a winning binary option trade.
            Typical range 0.70 to 0.92 depending on asset and broker.

    Example:
        >>> rc = RewardCalculator(payout_ratio=0.85)
        >>> reward = rc.calculate_reward(action=0, is_correct=True, gate_passed=True)
        >>> reward  # 0.85 + 0.1 = 0.95
        0.95
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
        gate_passed: bool,
    ) -> float:
        """
        Compute the scalar reward for a single RL environment step.

        Applies the reward structure defined in the class docstring.
        The gate compliance adjustment is always applied regardless
        of whether the direction was correct — the agent must learn
        that gate quality matters independently of outcome.

        Args:
            action:     Agent action. Must be 0 (CALL), 1 (PUT), or
                        2 (SKIP). Any other value raises ValueError.
            is_correct: True if the agent's direction prediction
                        matched the actual price movement at expiry.
                        Ignored when action == 2 (SKIP).
            gate_passed: True if all four TradeEligibility gates
                        passed for this bar (BB_WIDTH, ATR, RVOL,
                        SPREAD). Sourced from
                        FeatureEngineer.evaluate_gates().
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

        # Gate compliance adjustment applied on top of direction reward.
        # Teaches the agent to respect market quality conditions.
        if gate_passed:
            reward += 0.1
        else:
            reward -= 0.2

        logger.debug(
            "[^] calculate_reward(): action=%d is_correct=%s "
            "gate_passed=%s reward=%.4f",
            action,
            is_correct,
            gate_passed,
            reward,
        )

        return float(reward)

    def __repr__(self) -> str:
        return f"RewardCalculator(payout_ratio={self.payout_ratio:.2f})"
