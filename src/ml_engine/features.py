"""
src/ml_engine/features.py — The Translator.

Role: Transform raw Vault data (Bar / Tick DataFrames) into the full
FEATURE_SET_BINARY_OPTIONS_AI feature matrix ready for model inference.

Design document: docs/ml_engine/Features.md
Test harness:    docs/ml_engine/Features_Test_Harness.md

Architecture
------------
Five primary feature groups are computed in dependency order:
    1. PRICE_ACTION   — candlestick geometry and pattern flags
    2. MOMENTUM       — RSI, MACD, ROC, log-return series
    3. VOLATILITY     — ATR, Bollinger Bands, Keltner Channels
    4. MICRO_STRUCTURE — tick-level velocity, spread, order-flow (optional)
    5. CONTEXT        — session flags, cyclical time encoding

A sixth DERIVED pass computes gate thresholds and top-weighted helpers that
depend on the five primary groups having already been written into the frame.

Public API
----------
    get_feature_engineer() -> FeatureEngineer          (singleton)
    FeatureEngineer.transform(bars, ticks)             -> pd.DataFrame
    FeatureEngineer.get_latest(bars, ticks)            -> FeatureVector
    FeatureEngineer.build_matrix(bars, symbol, ticks)  -> FeatureMatrix
    FeatureEngineer.get_expiry_features(fe, k)         -> pd.DataFrame
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.config import get_settings
from data.forex_calendar import is_forex_closed, get_forex_holidays

logger = logging.getLogger(__name__)


# ── Custom Exception ────────────────────────────────────────────────────────


class FeatureEngineerError(Exception):
    """
    Raised when the FeatureEngineer cannot produce a valid output.

    Distinct from ValueError (which signals a caller contract violation
    such as a malformed input) — FeatureEngineerError signals a runtime
    failure inside the pipeline that the engine layer must handle.

    Attributes:
        stage: The pipeline stage that failed (e.g. "transform", "get_latest").
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return f"FeatureEngineerError(stage={self.stage!r}, message={str(self)!r})"


# ── Module Constants ────────────────────────────────────────────────────────

# Tiny denominator guard — prevents division-by-zero without altering values
# measurably (1e-9 is orders of magnitude smaller than any valid price move).
_EPS: float = 1e-9

# Maps internal timeframe keys to pandas resample offset aliases.
# M1 is the canonical base for binary-options features in this system.
_RESAMPLE_MAP: dict[str, str] = {"M1": "1min", "M5": "5min", "M15": "15min"}

# Feature engineering schema version. Increment when the column set or
# computation logic changes in a way that invalidates cached vectors.
_VERSION: str = "3.3.0-full-binary"

# ── Indicator Periods ───────────────────────────────────────────────────────
# All look-back periods are defined here so they can be found and changed in
# one place without hunting through the computation methods.

# Momentum
_RSI_PERIOD: int = 5  # Wilder RSI look-back
_CCI_PERIOD: int = 10  # Commodity Channel Index look-back
_MACD_FAST: int = 6  # MACD fast EMA span
_MACD_SLOW: int = 18  # MACD slow EMA span — longest warmup period
_MACD_SIGNAL_PERIOD: int = 5  # MACD signal EMA span
_MOMENTUM_PERIOD: int = 3  # Classic momentum oscillator look-back

# Volatility
_ATR_PERIOD: int = 5  # Wilder ATR look-back
_BB_PERIOD: int = 10  # Bollinger Band SMA period
_BB_MULTIPLIER: float = 2.0  # Bollinger Band standard-deviation width
_KC_MULTIPLIER: float = 1.5  # Keltner Channel ATR width

# Gap-filling policy
# Maximum consecutive M1 bars to forward-fill after a resample gap.
# Gaps longer than this are left as NaN and eliminated by the final dropna().
# At 5 bars (= 5 minutes), short broker hiccups are healed without silently
# fabricating multi-hour flat-price periods (weekends, session breaks).
_MAX_FFILL_BARS: int = 5

# Context
_MINUTES_TO_NEWS_PLACEHOLDER: int = 90  # Default until economic calendar API is wired


# ── Feature Schema ──────────────────────────────────────────────────────────

FEATURE_SET_BINARY_OPTIONS_AI = {
    "MOMENTUM": [
        "RSI",
        "MACD_VALUE",
        "MACD_SIGNAL",
        "MACD_HIST",
        "ROC_5",
        "ROC_10",
        "ROC_20",
        "RETURN_1",
        "RETURN_3",
        "RETURN_5",
        "MOMENTUM_OSCILLATOR",
        "CCI",
    ],
    "VOLATILITY": [
        "ATR",
        "BB_UPPER_DIST",
        "BB_LOWER_DIST",
        "BB_WIDTH",
        "BB_PERCENT_B",
        "KC_WIDTH",
        "NATR",
        "RANGE_EXPANSION_RATIO",
        "RELATIVE_VOLUME_RVOL",
    ],
    "PRICE_ACTION": [
        "BODY_TO_RANGE_RATIO",
        "UPPER_WICK_TO_BODY_RATIO",
        "LOWER_WICK_TO_BODY_RATIO",
        "ENGULFING_BINARY",
        "THREE_BAR_SLOPE",
        "PINBAR_SIGNAL",
        "DOJI_BINARY",
        "MARUBOZU_BINARY",
        "CONSECUTIVE_BULL_BARS",
        "CONSECUTIVE_BEAR_BARS",
        "CLOSE_POSITION_IN_CANDLE",
        "HIGH_LOW_RANGE_NORMALIZED",
        "CANDLE_POSITION_IN_DAY",
        "PREVIOUS_CANDLE_DIRECTION",
        "TWO_BAR_REVERSAL",
    ],
    "MICRO_STRUCTURE": [
        "TICK_VELOCITY",
        "SPREAD_NORMALIZED",
        "TICK_DELTA_CUMULATIVE",
        "ORDER_FLOW_IMBALANCE",
    ],
    "CONTEXT": [
        "SESSION_LONDON",
        "SESSION_NEWYORK",
        "SESSION_TOKYO",
        "SESSION_OVERLAP_LONDON_NY",
        "TIME_SINE",
        "TIME_COSINE",
        "DAY_OF_WEEK_SINE",
        "DAY_OF_WEEK_COSINE",
        "MINUTES_TO_NEWS",
        "HOUR_OF_DAY_NORMALIZED",
        "IS_GAP_OPEN",
        "GAP_MINUTES",
        "GAP_PIPS",
        "IS_MONDAY_OPEN",
    ],
}

# ── Derived / Gate Features ─────────────────────────────────────────────────
# These are computed internally for eligibility gates and expiry rules
# but are not part of the primary schema fed to the model.

TOP_WEIGHTED_FEATURES = [
    "BODY_TO_RANGE_RATIO",
    "UPPER_WICK_TO_BODY_RATIO",
    "TICK_VELOCITY",
    "RELATIVE_VOLUME_RVOL",
    "BB_WIDTH",
    "ROC_5_ACCELERATION",
    "ENGULFING_BINARY",
    "ATR_RATIO_CHANGE",
    "THREE_BAR_SLOPE",
    "SPREAD_NORMALIZED",
]

BINARY_EXPIRY_RULES = {
    "1_MIN": ["TICK_VELOCITY", "RVOL", "BODY_TO_RANGE_RATIO"],
    "5_MIN": ["ROC_5", "BB_WIDTH", "THREE_BAR_SLOPE"],
    "15_MIN": ["MACD_HIST_SLOPE", "ATR", "SESSION_CONTEXT"],
}


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureVector:
    """
    Single-row feature vector ready for model inference.

    Immutable after construction (frozen=True). The numpy array held in
    ``vector`` is coerced to float32 and write-locked in ``__post_init__``
    so that downstream model code cannot silently mutate the feature snapshot.

    Attributes:
        timestamp: UTC timestamp of the source bar (naive, assumed UTC).
        vector:    Flattened float32 array of feature values, index-aligned
                   to ``feature_names``.
        feature_names: Ordered column names matching each position in
                   ``vector``. Length must equal ``len(vector)``.
        version:   Feature engineering version string (mirrors ``_VERSION``).
    """

    timestamp: pd.Timestamp
    vector: np.ndarray
    feature_names: list[str]
    version: str

    def __post_init__(self) -> None:
        # Coerce to float32 via a new array object, then lock it read-only.
        # object.__setattr__ is required because the dataclass is frozen=True;
        # direct assignment would raise FrozenInstanceError.
        object.__setattr__(self, "vector", np.asarray(self.vector, dtype=np.float32))
        # Setting numpy flags is not a field reassignment — it mutates the
        # array object in-place, which is permitted even on a frozen dataclass.
        self.vector.flags.writeable = False

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot (vector as list)."""
        return {
            "timestamp": str(self.timestamp),
            "vector": self.vector.tolist(),
            "feature_names": self.feature_names,
            "version": self.version,
        }

    def __repr__(self) -> str:
        return (
            f"FeatureVector("
            f"timestamp={self.timestamp!r}, "
            f"n_features={len(self.vector)}, "
            f"version={self.version!r})"
        )


@dataclass(frozen=True)
class FeatureMatrix:
    """
    Batch feature output for model training.

    Produced by calling FeatureEngineer.transform() on a full
    historical bar DataFrame. Represents the complete 2D input
    matrix for Classical ML training and the source data for
    SequenceGenerator (Deep Learning) and RL environments.

    Attributes:
        timestamps: UTC timestamps aligned to each row in matrix.
        matrix:     float32 array of shape (N, F) where N is the
                    number of bars and F is the number of features.
        feature_names: Ordered column names matching axis=1 of matrix.
                    Must match FeatureVector.feature_names exactly.
        version:    Feature engineering version string. Must match
                    _VERSION to ensure label/matrix alignment.
        symbol:     Currency pair this matrix was built from.
    """

    timestamps: list
    matrix: np.ndarray
    feature_names: list[str]
    version: str
    symbol: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", np.asarray(self.matrix, dtype=np.float32))
        self.matrix.flags.writeable = False
        if self.matrix.ndim != 2:
            error_block = (
                f"\n{'!' * 60}\n"
                f"FEATURE MATRIX ERROR: INVALID SHAPE\n"
                f"Expected 2D array, got shape {self.matrix.shape}.\n"
                f"Verify the output of FeatureEngineer.transform() and the "
                f"data source formatting.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"FeatureMatrix requires 2D array, got shape {self.matrix.shape}"
            )
        if self.matrix.shape[1] != len(self.feature_names):
            error_block = (
                f"\n{'!' * 60}\n"
                f"FEATURE MATRIX ERROR: INCOMPATIBLE DIMENSIONS\n"
                f"Matrix columns ({self.matrix.shape[1]}) != "
                f"feature_names length ({len(self.feature_names)})\n"
                f"Verify the output of FeatureEngineer.transform() and the "
                f"data source formatting.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"matrix columns ({self.matrix.shape[1]}) != "
                f"feature_names length ({len(self.feature_names)})"
            )

    def to_tensor(self) -> "np.ndarray":
        """
        Return matrix as a 3D tensor-ready array (1, N, F).
        Convenience method for SequenceGenerator seeding.
        """
        return self.matrix[np.newaxis, :, :]

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __repr__(self) -> str:
        return (
            f"FeatureMatrix("
            f"symbol={self.symbol!r}, "
            f"shape={self.matrix.shape}, "
            f"version={self.version!r})"
        )


# ── The Engineer ─────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    The Translator: raw OHLCV Bars + optional Ticks → FEATURE_SET_BINARY_OPTIONS_AI.

    Implements the complete feature pipeline for binary-options AI inference.
    All five primary feature groups (PRICE_ACTION, MOMENTUM, VOLATILITY,
    MICRO_STRUCTURE, CONTEXT) plus a DERIVED pass are applied in strict
    dependency order inside ``transform()``.

    Each group is individually gated by a config flag
    (``feat_price_action_enabled``, ``feat_momentum_enabled``, etc.) so
    expensive groups can be disabled during unit testing or staged rollout.

    Thread safety
    -------------
    ``FeatureEngineer`` holds no mutable state after construction — all
    intermediate DataFrames are local to each call. It is safe to call
    ``transform()`` and ``get_latest()`` concurrently from multiple threads.

    Lifecycle
    ---------
    Do not instantiate directly — use the ``get_feature_engineer()`` singleton
    factory. Multiple instances share no state, but the singleton ensures
    ``get_settings()`` is called exactly once.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Master Pipeline ──────────────────────────────────────────────────────

    def transform(
        self,
        bars: pd.DataFrame,
        ticks: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        The Master Pipeline: Bars + Ticks -> Normalized Features DataFrame.

        Applies all five feature groups in dependency order. Each group
        is guarded by a config flag. The returned DataFrame has one row
        per input bar (after dropna) with all computed feature columns
        appended.

        Args:
            bars: OHLCV DataFrame with DatetimeIndex. Must contain
                columns: open, high, low, close, volume.
            ticks: Optional tick DataFrame with DatetimeIndex and
                columns: bid, ask. Required for MICRO_STRUCTURE features.

        Returns:
            pd.DataFrame: Feature matrix aligned to bar index.
        """
        # Input validation and early exit for empty bars DataFrame
        if bars.empty:
            logger.warning(
                "Input bars DataFrame is empty. Returning empty features.",
                extra={"event": "FEATURE ENGINEERING WARNING: EMPTY BARS DATAFRAME"},
            )
            return pd.DataFrame()

        # 1. Normalize Frequency using the Map
        # We target M1 as the base for binary options features
        target_freq = _RESAMPLE_MAP.get("M1", "1min")

        # Ensure index is DatetimeIndex for resampling.
        # A silent pd.to_datetime() on an integer index (e.g. epoch seconds vs
        # epoch milliseconds) produces wrong timestamps without any error.
        # We log a warning so the caller can fix the data source, and raise
        # immediately if the conversion itself fails.
        if not isinstance(bars.index, pd.DatetimeIndex):
            logger.warning(
                "Input bars DataFrame is empty. Returning empty features.",
                extra={
                    "event": "FEATURE ENGINEERING: bars index is not DatetimeIndex",
                    "message": f"(type={type(bars.index).__name__}) — coercing to datetime. Verify source data formatting.",
                },
            )

        try:
            bars = bars.copy()
            bars.index = pd.to_datetime(bars.index)
        except Exception as exc:  # pragma: no cover — pathological index type
            logger.critical(
                "[!] FEATURE ENGINEERING: bars index could not be coerced "
                "to DatetimeIndex (type=%s, error=%s). "
                "Verify the data source — Storage always returns DatetimeIndex.",
                type(bars.index).__name__,
                exc,
            )
            raise ValueError(
                f"bars index cannot be coerced to DatetimeIndex: {exc}"
            ) from exc

        # Resample to M1 grid. After resampling, missing bars (weekends, session
        # breaks, broker outages) appear as NaN rows. We forward-fill up to
        # _MAX_FFILL_BARS consecutive gaps only — short broker hiccups are healed
        # without fabricating multi-hour flat-price periods that would corrupt all
        # rolling indicators (RSI, ATR, MACD etc.). Gaps beyond the limit stay NaN
        # and are eliminated by the final dropna().
        fe = bars.sort_index().resample(target_freq).last()
        n_gaps = int(fe["close"].isna().sum())
        if n_gaps > 0:
            self._log_gap_info(fe, n_gaps)
        fe = fe.ffill(limit=_MAX_FFILL_BARS)

        # Guard against missing columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(fe.columns)
        if missing:
            logger.critical(
                "wtf!? Missing columns?",
                extra={
                    "event": "FEATURE ENGINEERING FAILURE: MISSING COLUMNS",
                    "Required": sorted(required),
                    "Missing": sorted(missing),
                },
            )
            raise ValueError(f"Input bars missing columns: {missing}")

        # 1. PRICE ACTION
        if self._settings.feat_price_action_enabled:
            fe = self._compute_price_action(fe)

        # 2. MOMENTUM
        if self._settings.feat_momentum_enabled:
            fe = self._compute_momentum(fe)

        # 3. VOLATILITY
        if self._settings.feat_volatility_enabled:
            fe = self._compute_volatility(fe)

        # 4. MICRO STRUCTURE
        if self._settings.feat_micro_enabled and ticks is not None:
            fe = self._compute_micro_structure(fe, ticks, target_freq)
        else:
            # Zero-fill micro structure columns so the schema is always complete
            for col in FEATURE_SET_BINARY_OPTIONS_AI["MICRO_STRUCTURE"]:
                if col not in fe.columns:
                    fe[col] = 0.0

        # 5. CONTEXT
        if self._settings.feat_context_enabled:
            fe = self._compute_context(
                fe
            )  # always runs — feeds gates and top-weighted features

        # 6. DERIVED (fe)
        fe = self._compute_derived(fe)

        self._warn_missing_schema(fe)

        before = len(fe)
        fe = fe.dropna()
        after = len(fe)
        dropped = before - after
        if dropped > 0:
            # Rolling window warmup routinely drops the first ~26 bars (MACD span).
            # This is expected and not an error — log at DEBUG so it doesn't
            # pollute INFO streams in production.
            logger.debug(
                "[^] FEATURE ENGINEERING: %d rows dropped (%.1f%% of %d) "
                "— expected rolling-window warmup.",
                dropped,
                dropped / before * 100,
                before,
            )
        return fe.dropna()

    # ── Private: Pipeline Helpers ────────────────────────────────────────────

    def _log_gap_info(self, fe: pd.DataFrame, n_gaps: int) -> None:
        """Log gap diagnostics after M1 resampling."""
        # After resample().last(), fe already has a complete minute index —
        # gaps show up as NaN rows, not missing timestamps.  We need to
        # filter the NaN rows themselves by their timestamp.
        gap_timestamps = fe.index[fe["close"].isna()].to_series()
        total_len = len(fe) + n_gaps
        if gap_timestamps.empty or total_len == 0:
            logger.info(
                "No gap timestamp found.",
                extra={
                    "event": "BACKFILL_GAP_CHECK",
                    "data": f"data={str(fe.shape)}",
                },
            )
            return

        dow = gap_timestamps.dt.dayofweek
        hour = gap_timestamps.dt.hour
        # Count gaps that fall within standard forex market-closed windows:
        # Friday 21:00 UTC through Sunday 21:00 UTC
        weekend_closed_mask = (
            (dow == 5)  # Saturday
            | ((dow == 6) & (hour < 21))  # Sunday before 21:00
            | ((dow == 4) & (hour >= 21))  # Friday after 21:00
        )

        # Use the forex holiday calendar to identify known market closures.
        # This replaces the ad-hoc 48-hour region heuristic with proper
        # holiday awareness (Christmas, New Year, Easter, etc.).
        holiday_mask = gap_timestamps.apply(
            lambda ts: is_forex_closed(ts.to_pydatetime())
            and not weekend_closed_mask.loc[ts]
        )

        market_closed_mask = weekend_closed_mask | holiday_mask
        market_closed_count = int(market_closed_mask.sum())
        unexpected_gaps = n_gaps - market_closed_count

        if unexpected_gaps > 0:
            logger.info(
                "[%%] UNEXPECTED DATA GAP: %d M1 bars missing (%.1f%% of %d total) "
                "outside normal weekend closures. Forward-filling up to %d bars. "
                "Check data source for API failures or broker outages.",
                unexpected_gaps,
                unexpected_gaps / (len(fe) + n_gaps) * 100,
                len(fe) + n_gaps,
                _MAX_FFILL_BARS,
            )
        else:
            logger.info(
                "[^] Weekend/holiday gaps: %d M1 bars (%.1f%% of %d total) — normal. "
                "Forward-filling up to %d consecutive bars.",
                n_gaps,
                n_gaps / (len(fe) + n_gaps) * 100,
                len(fe) + n_gaps,
                _MAX_FFILL_BARS,
            )

    def _warn_missing_schema(self, fe: pd.DataFrame) -> None:
        """Warn if any primary feature columns expected by the schema are absent."""
        all_primary = [
            col for group in FEATURE_SET_BINARY_OPTIONS_AI.values() for col in group
        ]
        missing_schema = [col for col in all_primary if col not in fe.columns]
        if missing_schema:
            logger.warning(
                "[%%] SCHEMA INCOMPLETE: %d of %d expected feature columns "
                "absent from output: %s. Verify feat_*_enabled config flags "
                "match the schema the model was trained on.",
                len(missing_schema),
                len(all_primary),
                missing_schema,
            )

    # ── 1. PRICE ACTION ─────────────────────────────────────────────────────

    def _compute_price_action(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all PRICE_ACTION features from raw OHLCV data.

        Features computed:
            BODY_TO_RANGE_RATIO, UPPER_WICK_TO_BODY_RATIO,
            LOWER_WICK_TO_BODY_RATIO, DOJI_BINARY, MARUBOZU_BINARY,
            THREE_BAR_SLOPE, CONSECUTIVE_BULL_BARS, CONSECUTIVE_BEAR_BARS,
            CLOSE_POSITION_IN_CANDLE, PREVIOUS_CANDLE_DIRECTION,
            ENGULFING_BINARY, PINBAR_SIGNAL, HIGH_LOW_RANGE_NORMALIZED,
            CANDLE_POSITION_IN_DAY, TWO_BAR_REVERSAL
        """

        hi_lo = (fe["high"] - fe["low"]).replace(0, _EPS)
        body_raw = fe["close"] - fe["open"]
        body_abs = body_raw.abs()

        # ── Core Ratios ──────────────────────────────────────────────────
        fe["BODY_TO_RANGE_RATIO"] = body_abs / hi_lo

        upper_wick = fe["high"] - fe[["open", "close"]].max(axis=1)
        lower_wick = fe[["open", "close"]].min(axis=1) - fe["low"]

        fe["UPPER_WICK_TO_BODY_RATIO"] = upper_wick / body_abs.replace(0, _EPS)
        fe["LOWER_WICK_TO_BODY_RATIO"] = lower_wick / body_abs.replace(0, _EPS)

        # ── Candlestick Pattern Binary Flags ─────────────────────────────

        # Doji: body is tiny relative to range (< 10 %)
        fe["DOJI_BINARY"] = (fe["BODY_TO_RANGE_RATIO"] < 0.1).astype(int)

        # Marubozu: body dominates range (> 90 %)
        fe["MARUBOZU_BINARY"] = (fe["BODY_TO_RANGE_RATIO"] > 0.9).astype(int)

        # Engulfing: current bar's body fully contains previous bar's body
        prev_open = fe["open"].shift(1)
        prev_close = fe["close"].shift(1)

        # Bullish engulfing: prev bear, curr bull, curr body engulfs prev body
        bullish_engulf = (
            (prev_close < prev_open)  # previous bar bearish
            & (fe["close"] > fe["open"])  # current bar bullish
            & (fe["open"] <= prev_close)  # current open at/below prev close
            & (fe["close"] >= prev_open)  # current close at/above prev open
        )

        # Bearish engulfing: prev bull, curr bear, curr body engulfs prev body
        bearish_engulf = (
            (prev_close > prev_open)  # previous bar bullish
            & (fe["close"] < fe["open"])  # current bar bearish
            & (fe["open"] >= prev_close)  # current open at/above prev close
            & (fe["close"] <= prev_open)  # current close at/below prev open
        )

        fe["ENGULFING_BINARY"] = (bullish_engulf | bearish_engulf).astype(int)

        # Pin bar: one wick is >= 2x the body, other wick is small
        # Bullish pin: long lower wick, small upper wick
        # Bearish pin: long upper wick, small lower wick
        is_bullish_pin = (lower_wick >= 2 * body_abs.replace(0, _EPS)) & (
            upper_wick <= 0.3 * lower_wick.replace(0, _EPS)
        )
        is_bearish_pin = (upper_wick >= 2 * body_abs.replace(0, _EPS)) & (
            lower_wick <= 0.3 * upper_wick.replace(0, _EPS)
        )
        fe["PINBAR_SIGNAL"] = (is_bullish_pin | is_bearish_pin).astype(int)

        # Two-bar reversal: bar1 bearish, bar2 bullish, bar2 close > bar1 open
        #   OR bar1 bullish, bar2 bearish, bar2 close < bar1 open
        prev_body = body_raw.shift(1)
        two_bar_bull = (
            (prev_body < 0) & (body_raw > 0) & (fe["close"] > fe["open"].shift(1))
        )
        two_bar_bear = (
            (prev_body > 0) & (body_raw < 0) & (fe["close"] < fe["open"].shift(1))
        )
        fe["TWO_BAR_REVERSAL"] = (two_bar_bull | two_bar_bear).astype(int)

        # ── Slope & Consecutive ──────────────────────────────────────────

        fe["THREE_BAR_SLOPE"] = fe["close"].diff(3) / 3

        bull = (fe["close"] > fe["open"]).astype(int)
        fe["CONSECUTIVE_BULL_BARS"] = bull.groupby(
            (bull != bull.shift()).cumsum()
        ).cumsum()

        bear = (fe["close"] < fe["open"]).astype(int)
        fe["CONSECUTIVE_BEAR_BARS"] = bear.groupby(
            (bear != bear.shift()).cumsum()
        ).cumsum()

        # ── Position Features ────────────────────────────────────────────

        fe["CLOSE_POSITION_IN_CANDLE"] = (fe["close"] - fe["low"]) / hi_lo

        fe["PREVIOUS_CANDLE_DIRECTION"] = pd.Series(
            np.sign(body_raw.shift(1)), index=fe.index
        ).fillna(0)

        # High-low range normalized by rolling average range (_BB_PERIOD bars)
        bar_range = fe["high"] - fe["low"]
        avg_range = bar_range.rolling(_BB_PERIOD).mean().replace(0, _EPS)
        fe["HIGH_LOW_RANGE_NORMALIZED"] = bar_range / avg_range

        # Candle position within the trading day (0.0 = day open, 1.0 = day close)
        if isinstance(fe.index, pd.DatetimeIndex):
            day_start = fe.index.normalize()  # midnight of each bar's day
            # Approximate trading day as 00:00–23:59
            minutes_since_midnight = (fe.index - day_start).total_seconds() / 60.0
            fe["CANDLE_POSITION_IN_DAY"] = minutes_since_midnight / 1440.0
        else:  # pragma: no cover — index is always DatetimeIndex when called via transform()
            fe["CANDLE_POSITION_IN_DAY"] = 0.0

        return fe

    # ── 2. MOMENTUM ─────────────────────────────────────────────────────────

    def _compute_momentum(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all MOMENTUM features.

        Features computed:
            RSI, CCI, MACD_VALUE, MACD_SIGNAL, MACD_HIST,
            ROC_5, ROC_10, ROC_20, RETURN_1, RETURN_3, RETURN_5,
            MOMENTUM_OSCILLATOR
        """

        # ── RSI (Wilder-smoothed) ────────────────────────────────────────
        delta = fe["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing: EWM with alpha=1/period approximates the recursive
        # smoothed-average formula. min_periods=_RSI_PERIOD ensures the first
        # value is a true SMA, not a partial-window estimate.
        avg_gain = gain.ewm(
            alpha=1 / _RSI_PERIOD, min_periods=_RSI_PERIOD, adjust=False
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / _RSI_PERIOD, min_periods=_RSI_PERIOD, adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, _EPS)
        fe["RSI"] = 100 - (100 / (1 + rs))

        # ── CCI ──────────────────────────────────────────────────────────
        tp = (fe["high"] + fe["low"] + fe["close"]) / 3
        tp_mean = tp.rolling(_CCI_PERIOD).mean()
        tp_std = tp.rolling(_CCI_PERIOD).std().replace(0, _EPS)
        fe["CCI"] = (tp - tp_mean) / (0.015 * tp_std)

        # ── MACD ─────────────────────────────────────────────────────────
        ema_fast = fe["close"].ewm(span=_MACD_FAST, adjust=False).mean()
        ema_slow = fe["close"].ewm(span=_MACD_SLOW, adjust=False).mean()
        fe["MACD_VALUE"] = ema_fast - ema_slow
        fe["MACD_SIGNAL"] = (
            fe["MACD_VALUE"].ewm(span=_MACD_SIGNAL_PERIOD, adjust=False).mean()
        )
        fe["MACD_HIST"] = fe["MACD_VALUE"] - fe["MACD_SIGNAL"]

        # ── Rate of Change ───────────────────────────────────────────────
        # fill_method=None: NaN gaps from the resample have already been
        # handled by the _MAX_FFILL_BARS strategy; we don't want pct_change
        # to silently forward-fill them again here.
        for n in [5, 10, 20]:
            fe[f"ROC_{n}"] = fe["close"].pct_change(n, fill_method=None)

        # ── Returns (log) ────────────────────────────────────────────────
        for n in [1, 3, 5]:
            fe[f"RETURN_{n}"] = np.log(
                fe["close"] / fe["close"].shift(n).replace(0, _EPS)
            )

        # ── Momentum Oscillator ──────────────────────────────────────────
        # Classic price momentum: close minus close N bars ago.
        fe["MOMENTUM_OSCILLATOR"] = fe["close"] - fe["close"].shift(_MOMENTUM_PERIOD)

        return fe

    # ── 3. VOLATILITY ───────────────────────────────────────────────────────

    def _compute_volatility(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all VOLATILITY features.

        Features computed:
            ATR, NATR, BB_WIDTH, BB_UPPER_DIST, BB_LOWER_DIST,
            BB_PERCENT_B, KC_WIDTH, RANGE_EXPANSION_RATIO,
            RELATIVE_VOLUME_RVOL
        """

        # ── True Range & ATR ─────────────────────────────────────────────
        tr_components = pd.concat(
            [
                fe["high"] - fe["low"],
                (fe["high"] - fe["close"].shift()).abs(),
                (fe["low"] - fe["close"].shift()).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)
        fe["ATR"] = tr.ewm(
            alpha=1 / _ATR_PERIOD, min_periods=_ATR_PERIOD, adjust=False
        ).mean()
        fe["NATR"] = (fe["ATR"] / fe["close"].replace(0, _EPS)) * 100

        # ── Bollinger Bands ──────────────────────────────────────────────
        sma_bb = fe["close"].rolling(_BB_PERIOD).mean()
        std_bb = fe["close"].rolling(_BB_PERIOD).std()
        upper_bb = sma_bb + _BB_MULTIPLIER * std_bb
        lower_bb = sma_bb - _BB_MULTIPLIER * std_bb
        band_width = upper_bb - lower_bb

        fe["BB_WIDTH"] = band_width / sma_bb.replace(0, _EPS)
        fe["BB_UPPER_DIST"] = (upper_bb - fe["close"]) / fe["close"].replace(0, _EPS)
        fe["BB_LOWER_DIST"] = (fe["close"] - lower_bb) / fe["close"].replace(0, _EPS)
        fe["BB_PERCENT_B"] = (fe["close"] - lower_bb) / band_width.replace(0, _EPS)

        # ── Keltner Channels ─────────────────────────────────────────────
        ema_kc = fe["close"].ewm(span=_BB_PERIOD, adjust=False).mean()
        kc_upper = ema_kc + _KC_MULTIPLIER * fe["ATR"]
        kc_lower = ema_kc - _KC_MULTIPLIER * fe["ATR"]
        fe["KC_WIDTH"] = (kc_upper - kc_lower) / ema_kc.replace(0, _EPS)

        # ── Range Expansion Ratio ────────────────────────────────────────
        # Current range / rolling-average range — detects breakout expansions.
        current_range = fe["high"] - fe["low"]
        avg_range = current_range.rolling(_BB_PERIOD).mean().replace(0, _EPS)
        fe["RANGE_EXPANSION_RATIO"] = current_range / avg_range

        # ── Relative Volume (RVOL) ───────────────────────────────────────
        # Bar volume relative to its rolling average — detects volume surges.
        vol_ma = fe["volume"].rolling(_BB_PERIOD).mean().replace(0, _EPS)
        fe["RELATIVE_VOLUME_RVOL"] = fe["volume"] / vol_ma

        return fe

    # ── 4. MICRO STRUCTURE ──────────────────────────────────────────────────

    def _compute_micro_structure(
        self,
        fe: pd.DataFrame,
        ticks: pd.DataFrame,
        target_freq: str,
    ) -> pd.DataFrame:
        """
        Compute all MICRO_STRUCTURE features from tick-level data.

        Features computed:
            TICK_VELOCITY, SPREAD_NORMALIZED,
            TICK_DELTA_CUMULATIVE, ORDER_FLOW_IMBALANCE

        Args:
            fe: Bar DataFrame (index is DatetimeIndex).
            ticks: Tick DataFrame with columns: bid, ask.
        """

        # ── Tick Velocity ────────────────────────────────────────────────
        # Number of ticks per bar window (1-min resample)
        tick_counts = ticks["bid"].resample(target_freq).count()
        fe["TICK_VELOCITY"] = tick_counts.reindex(fe.index).fillna(0)

        # ── Spread Normalized ────────────────────────────────────────────
        # Tick-level ask-bid spread averaged per bar, then normalised by close.
        tick_spread = ticks["ask"] - ticks["bid"]
        avg_spread: pd.Series = tick_spread.resample(target_freq).mean()
        fe["SPREAD_NORMALIZED"] = avg_spread.reindex(fe.index).ffill().fillna(0) / fe[
            "close"
        ].replace(0, _EPS)

        # ── Tick Delta Cumulative & Order Flow Imbalance ─────────────────
        #
        # IMPORTANT — APPROXIMATION NOTE:
        # True order-flow features require per-trade aggressor-side data
        # (i.e. whether the initiator was the buyer or the seller). Neither
        # Quotex streaming nor the Twelve Data REST feed provides this.
        #
        # We approximate using mid-price direction:
        #   sign(mid.diff()) == +1 → price moved up → inferred buy pressure
        #   sign(mid.diff()) == -1 → price moved down → inferred sell pressure
        #
        # This is a well-known approximation (sometimes called the "tick rule")
        # and is standard when aggressor-side data is unavailable. It is noisy
        # on low-activity bars but directionally informative on high-velocity
        # bars. Features derived from it should be treated as weak signals and
        # not relied on as the sole trade trigger.
        #
        # ORDER_FLOW_IMBALANCE falls back to 0.5 → 0.0 (centred, neutral) when
        # no ticks are present in the window — not 0.5 — because the feature is
        # shifted by -0.5 to be zero-centred for the model.
        tick_mid = (ticks["bid"] + ticks["ask"]) / 2
        tick_delta_raw: pd.Series = pd.Series(
            np.sign(tick_mid.diff()), index=tick_mid.index
        ).fillna(0)
        tick_delta_resampled: pd.Series = tick_delta_raw.resample(target_freq).sum()
        fe["TICK_DELTA_CUMULATIVE"] = (
            tick_delta_resampled.reindex(fe.index).fillna(0).rolling(_BB_PERIOD).sum()
        )

        buy_ticks: pd.Series = (tick_delta_raw > 0).resample(target_freq).sum()
        total_ticks: pd.Series = (
            tick_delta_raw.abs().resample(target_freq).sum().replace(0, _EPS)
        )
        # Shift by -0.5 so neutral (50/50 buy/sell) → 0.0, buyer dominance → +0.5,
        # seller dominance → -0.5. fillna(0.5) before shift → fillna(0.0) net.
        fe["ORDER_FLOW_IMBALANCE"] = (buy_ticks / total_ticks).reindex(fe.index).fillna(
            0.5
        ) - 0.5

        return fe

    # ── 5. CONTEXT ──────────────────────────────────────────────────────────

    def _compute_context(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all CONTEXT features from the DatetimeIndex.

        Features computed:
            SESSION_LONDON, SESSION_NEWYORK, SESSION_TOKYO,
            SESSION_OVERLAP_LONDON_NY, TIME_SINE, TIME_COSINE,
            DAY_OF_WEEK_SINE, DAY_OF_WEEK_COSINE, MINUTES_TO_NEWS,
            HOUR_OF_DAY_NORMALIZED
        """
        if not isinstance(fe.index, pd.DatetimeIndex):
            logger.warning("[%] Context features require DatetimeIndex — zero-filling.")
            for col in FEATURE_SET_BINARY_OPTIONS_AI["CONTEXT"]:
                fe[col] = 0.0
            return fe

        h = fe.index.hour
        m = fe.index.minute
        d = fe.index.dayofweek

        # ── Session Flags (UTC hours) ────────────────────────────────────
        fe["SESSION_LONDON"] = ((h >= 7) & (h <= 16)).astype(int)
        fe["SESSION_NEWYORK"] = ((h >= 12) & (h <= 21)).astype(int)
        fe["SESSION_TOKYO"] = ((h >= 0) & (h <= 9)).astype(int)

        # Overlap: both London and New York are active (12:00–16:00 UTC)
        fe["SESSION_OVERLAP_LONDON_NY"] = ((h >= 12) & (h <= 16)).astype(int)

        # ── Cyclical Time Encoding ───────────────────────────────────────
        # Hour + minute fractional hour for sub-hourly resolution
        fractional_hour = h + m / 60.0

        fe["TIME_SINE"] = np.sin(2 * np.pi * fractional_hour / 24)
        fe["TIME_COSINE"] = np.cos(2 * np.pi * fractional_hour / 24)

        fe["DAY_OF_WEEK_SINE"] = np.sin(2 * np.pi * d / 7)
        fe["DAY_OF_WEEK_COSINE"] = np.cos(2 * np.pi * d / 7)

        # ── Hour of Day Normalized ───────────────────────────────────────
        # Maps 0–23 → 0.0–1.0
        fe["HOUR_OF_DAY_NORMALIZED"] = fractional_hour / 24.0

        # ── Minutes to News (placeholder) ────────────────────────────────
        # Default: assume no imminent news. Override when economic calendar
        # API is integrated (see backlog BL-XX: news-feed integration).
        fe["MINUTES_TO_NEWS"] = _MINUTES_TO_NEWS_PLACEHOLDER

        # ── Gap-Aware Features ───────────────────────────────────────────
        # These describe the time gap between consecutive bars so the model
        # learns that post-weekend / post-holiday bars are a different regime.
        # No fabricated price data is needed — the gap itself is the signal.

        time_deltas = fe.index.to_series().diff().dt.total_seconds().div(60)

        # Flag: did this bar open after a gap longer than 2 minutes?
        fe["IS_GAP_OPEN"] = (time_deltas > 2).astype(int)

        # How many minutes was the gap? Capped at 1 week (10080 min) to
        # prevent extreme outliers from distorting the model.
        fe["GAP_MINUTES"] = time_deltas.clip(upper=10080).fillna(0)

        # Price gap in pips (open vs previous close). Zero on non-gap bars.
        prev_close = fe["close"].shift(1)
        raw_gap_pips = (fe["open"] - prev_close).abs() / 0.0001
        fe["GAP_PIPS"] = raw_gap_pips.where(fe["IS_GAP_OPEN"] == 1, other=0.0).fillna(0)

        # Flag: Monday open (post-weekend regime — typically gappy and noisy)
        fe["IS_MONDAY_OPEN"] = ((fe.index.dayofweek == 0) & (fe.index.hour < 4)).astype(
            int
        )

        return fe

    # ── 6. DERIVED (Gates + Top-Weighted + Expiry Helpers) ──────────────────

    def _compute_derived(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features referenced by TOP_WEIGHTED_FEATURES,
        and BINARY_EXPIRY_RULES.

        These are computed AFTER all primary groups so they can
        reference ATR, BB_WIDTH, ROC_5, MACD_HIST, etc.
        """

        # ── ROC_5 Acceleration ───────────────────────────────────────────
        # Second derivative of price: change in ROC_5 over 1 bar.
        if "ROC_5" in fe.columns:
            roc5: pd.Series = fe["ROC_5"]
            fe["ROC_5_ACCELERATION"] = roc5.diff()
        else:
            fe["ROC_5_ACCELERATION"] = 0.0

        # ── ATR Ratio Change ──────────────────────────────────────────
        # Percentage change in ATR over _ATR_PERIOD bars — detects
        # volatility regime shifts (expansion / contraction transitions).
        if "ATR" in fe.columns:
            atr_prev: pd.Series = fe["ATR"].shift(_ATR_PERIOD).replace(0, _EPS)
            fe["ATR_RATIO_CHANGE"] = (fe["ATR"] - atr_prev) / atr_prev
        else:
            fe["ATR_RATIO_CHANGE"] = 0.0

        # ── MACD Histogram Slope ─────────────────────────────────────────
        # Change in MACD_HIST over 3 bars — used in 15-min expiry rules.
        if "MACD_HIST" in fe.columns:
            fe["MACD_HIST_SLOPE"] = fe["MACD_HIST"].diff(3) / 3
        else:
            fe["MACD_HIST_SLOPE"] = 0.0

        # ── Session Context (aggregate flag for 15-min expiry) ───────────
        session_cols = [
            "SESSION_LONDON",
            "SESSION_NEWYORK",
            "SESSION_TOKYO",
        ]
        available_sessions = [c for c in session_cols if c in fe.columns]
        if available_sessions:
            fe["SESSION_CONTEXT"] = fe[available_sessions].max(axis=1)
        else:
            fe["SESSION_CONTEXT"] = 0.0

        return fe

    # ── Expiry-Specific Feature Selector ────────────────────────────────────

    def get_expiry_features(
        self,
        fe: pd.DataFrame,
        expiry_key: str,
    ) -> pd.DataFrame:
        """
        Return only the columns relevant to a specific binary expiry.

        Maps the BINARY_EXPIRY_RULES keys ("1_MIN", "5_MIN", "15_MIN")
        to the actual columns present in the feature DataFrame, resolving
        aliases (e.g., "RVOL" → "RELATIVE_VOLUME_RVOL").

        Args:
            fe: Full feature DataFrame.
            expiry_key: One of "1_MIN", "5_MIN", "15_MIN".

        Returns:
            pd.DataFrame: Subset of fe with only the expiry-relevant columns.

        Raises:
            ValueError: If expiry_key is not in BINARY_EXPIRY_RULES.
        """
        if expiry_key not in BINARY_EXPIRY_RULES:
            raise ValueError(
                f"Unknown expiry key: '{expiry_key}'. "
                f"Valid keys: {list(BINARY_EXPIRY_RULES.keys())}"
            )

        # Resolve alias names to actual column names
        alias_map = {
            "RVOL": "RELATIVE_VOLUME_RVOL",
            "SESSION_CONTEXT": "SESSION_CONTEXT",
        }

        raw_cols = BINARY_EXPIRY_RULES[expiry_key]
        resolved = [alias_map.get(c, c) for c in raw_cols]
        available = [c for c in resolved if c in fe.columns]

        missing = set(resolved) - set(available)
        if missing:
            logger.warning(
                f"[%] Expiry '{expiry_key}' missing columns: {missing}. "
                f"Returning available subset only."
            )

        return fe[available]

    # ── Live Inference ──────────────────────────────────────────────────────

    def get_latest(
        self,
        bars: pd.DataFrame,
        ticks: pd.DataFrame | None = None,
    ) -> FeatureVector:
        """
        The Live Inference Gate.

        Transforms the latest bars (+ optional ticks) into a single
        FeatureVector ready for model.predict(). Raises FeatureEngineerError
        if the pipeline cannot produce a valid vector — the engine layer is
        responsible for deciding whether to retry or skip the signal.

        Args:
            bars:  Recent OHLCV bars with DatetimeIndex (minimum ~30 rows
                   for rolling-window warmup; fewer rows yield an empty
                   feature frame after dropna).
            ticks: Optional recent ticks for MICRO_STRUCTURE features.
                   Pass None to zero-fill those columns.

        Returns:
            FeatureVector: immutable, float32, read-only numpy vector.

        Raises:
            FeatureEngineerError: If transform produces an empty frame, or
                if any unexpected exception occurs inside the pipeline.
        """
        try:
            full_df = self.transform(bars, ticks)
        except Exception as e:
            raise FeatureEngineerError(
                f"transform() failed: {e}", stage="transform"
            ) from e

        if full_df.empty:
            raise FeatureEngineerError(
                "transform() returned an empty DataFrame — insufficient bars "
                "for rolling-window warmup (minimum ~30 required).",
                stage="get_latest",
            )

        # Flatten schema: preserve FEATURE_SET ordering for model stability.
        target_cols = [
            col for group in FEATURE_SET_BINARY_OPTIONS_AI.values() for col in group
        ]
        available_cols = [c for c in target_cols if c in full_df.columns]

        last_row = full_df.iloc[-1][available_cols]

        # Guard against NaN / Inf that can propagate from rolling warmup edges.
        values: np.ndarray = np.asarray(last_row.values, dtype=np.float32)
        finite_mask: np.ndarray = np.isfinite(values)
        if not finite_mask.all():
            bad_names = [
                available_cols[i] for i, ok in enumerate(finite_mask) if not ok
            ]
            logger.warning(
                "[%%] NON-FINITE FEATURE VALUES: %s — replacing with 0.0 for model safety.",
                bad_names,
            )
            values = np.where(finite_mask, values, 0.0).astype(np.float32)

        ts_value: pd.Timestamp = full_df.index[-1]
        return FeatureVector(
            timestamp=ts_value,
            vector=values,
            feature_names=available_cols,
            version=_VERSION,
        )

    # ── Batch Matrix Builder ───────────────────────────────────────────────

    def build_matrix(
        self,
        bars: pd.DataFrame,
        symbol: str,
        ticks: pd.DataFrame | None = None,
    ) -> FeatureMatrix:
        """
        Batch mode: transform full historical bars into a FeatureMatrix.

        Runs the same pipeline as transform() but packages the result
        as an immutable FeatureMatrix ready for trainer.py consumption.
        Use this for offline training data preparation only — not for
        live inference (use get_latest() for that).

        Args:
            bars:   Full historical OHLCV DataFrame with DatetimeIndex.
            symbol: Currency pair identifier stored in matrix metadata.
            ticks:  Optional tick DataFrame for MICRO_STRUCTURE features.

        Returns:
            FeatureMatrix: immutable, float32, version-stamped batch matrix.

        Raises:
            FeatureEngineerError: If transform() fails or returns empty.
        """
        try:
            full_df = self.transform(bars, ticks)
        except Exception as e:
            raise FeatureEngineerError(
                f"build_matrix() failed at transform: {e}", stage="build_matrix"
            ) from e

        if full_df.empty:
            raise FeatureEngineerError(
                "build_matrix() got empty DataFrame after transform. "
                "Check bar history length.",
                stage="build_matrix",
            )

        target_cols = [
            col for group in FEATURE_SET_BINARY_OPTIONS_AI.values() for col in group
        ]
        available_cols = [c for c in target_cols if c in full_df.columns]
        matrix = full_df[available_cols].values.astype(np.float32)

        return FeatureMatrix(
            timestamps=list(full_df.index),
            matrix=matrix,
            feature_names=available_cols,
            version=_VERSION,
            symbol=symbol,
        )


# ── Singleton ───────────────────────────────────────────────────────────────

_engineer: FeatureEngineer | None = None


def get_feature_engineer() -> FeatureEngineer:
    """
    Return the module-level FeatureEngineer singleton.

    Lazy initialisation — the instance is created on the first call and
    reused on every subsequent call. This ensures ``get_settings()`` is
    invoked exactly once and that callers across the engine share the
    same configuration snapshot.

    **Thread safety**: construction is not guarded by a lock. In practice
    the engine layer calls this once at startup before spawning worker
    threads. If concurrent initialisation is a concern, call this function
    during the main-thread boot sequence to pre-warm the singleton.

    Returns:
        FeatureEngineer: the shared instance.
    """
    global _engineer
    if _engineer is None:
        _engineer = FeatureEngineer()
    return _engineer
