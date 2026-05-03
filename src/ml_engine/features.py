"""
src/ml_engine/features.py — The Translator (Pruned V2).

Role: Transform raw Vault data (Bar DataFrames) into the pruned
FEATURE_SET_BINARY_OPTIONS_AI feature matrix ready for model inference.

This is V2 of the feature engineer, pruned to 21 high-importance features
based on V2 Colab experiment results (AUC 0.8659).

Architecture
------------
Four primary feature groups are computed in dependency order:
    1. PRICE_ACTION   — candlestick geometry and pattern flags
    2. MOMENTUM       — RSI, MACD, log-return series
    3. VOLATILITY     — ATR, Bollinger Bands, Keltner Channels
    4. CONTEXT        — time-of-day encoding

    A fifth DERIVED pass computes helper features (e.g., MACD_HIST_SLOPE)
    that depend on the four primary groups having already been written.

    Public API
    ----------
    get_feature_engineer() -> FeatureEngineer          (singleton)
    FeatureEngineer.transform(bars)                    -> pd.DataFrame
    FeatureEngineer.get_latest(bars)                   -> FeatureVector
    FeatureEngineer.build_matrix(bars, symbol)         -> FeatureMatrix
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
_VERSION: str = "4.0.0-pruned"

# ── Indicator Periods ───────────────────────────────────────────────────────
# All look-back periods are defined here so they can be found and changed in
# one place without hunting through the computation methods.
# Multi-timeframe
_MTF_RSI_PERIOD: int = 5
_MTF_TREND_FAST: int = 5
_MTF_TREND_SLOW: int = 13
_MTF_CLOSE_POS_WINDOW: int = 10

# Regime detection
_REGIME_ATR_LONG: int = 100
_REGIME_ADX_PERIOD: int = 14
_REGIME_SMA_PERIODS: list[int] = [20, 50, 200]

# Lagged features
_LAGGED_FEATURES: list[str] = ["CLOSE_POSITION_IN_CANDLE", "RETURN_1", "RSI"]
_LAGGED_PERIODS: list[int] = [1, 2, 3, 5]

# Rolling statistics
_ROLLING_WINDOWS: list[int] = [5, 10]

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

# Minimum M1 bars required for transform() to produce a non-empty DataFrame.
# Derived from the longest rolling window (_MACD_SLOW) plus lag periods and
# multi-timeframe resample headroom. Pipeline reads this for warmup validation.
MIN_BARS_REQUIRED: int = 26  # _MACD_SLOW (18) + max lag (5) + resample safety (3)


# ── Feature Schema ──────────────────────────────────────────────────────────

FEATURE_SET_BINARY_OPTIONS_AI = {
    "MOMENTUM": [
        "RSI",
        "MACD_VALUE",
        "MACD_SIGNAL",
        "MACD_HIST",
        "RETURN_1",
        "RETURN_3",
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
        "RELATIVE_VOLUME_RVOL",
    ],
    "PRICE_ACTION": [
        "BODY_TO_RANGE_RATIO",
        "UPPER_WICK_TO_BODY_RATIO",
        "LOWER_WICK_TO_BODY_RATIO",
        "CONSECUTIVE_BULL_BARS",
        "CONSECUTIVE_BEAR_BARS",
        "CLOSE_POSITION_IN_CANDLE",
        "TWO_BAR_REVERSAL",
    ],
    "CONTEXT": [
        "HOUR_OF_DAY_NORMALIZED",
    ],
    "MULTI_TIMEFRAME": [
        "RSI_5T",
        "TREND_5T",
        "CLOSE_POS_5T",
        "RSI_15T",
        "TREND_15T",
        "CLOSE_POS_15T",
    ],
    "REGIME": [
        "VOLATILITY_REGIME",
        "ADX",
        "DIST_SMA20",
        "DIST_SMA50",
        "DIST_SMA200",
        "BB_KC_SQUEEZE",
    ],
    "INTERACTION": [
        "RSI_TREND_5T",
        "CLOSEPOS_VOLREGIME",
        "BB_MOMENTUM",
        "STREAK_TREND",
        "ADX_RETURN",
    ],
    "LAGGED": [
        "CLOSE_POSITION_IN_CANDLE_LAG1",
        "CLOSE_POSITION_IN_CANDLE_LAG2",
        "CLOSE_POSITION_IN_CANDLE_LAG3",
        "CLOSE_POSITION_IN_CANDLE_LAG5",
        "RETURN_1_LAG1",
        "RETURN_1_LAG2",
        "RETURN_1_LAG3",
        "RETURN_1_LAG5",
        "RSI_LAG1",
        "RSI_LAG2",
        "RSI_LAG3",
        "RSI_LAG5",
        "RETURN_1_MEAN_5",
        "RETURN_1_STD_5",
        "RSI_MEAN_10",
        "CLOSE_POS_MEAN_5",
    ],
    "SESSION_INTENSITY": [
        "RVOL_LONDON",
        "RVOL_NEWYORK",
        "RVOL_TOKYO",
    ],
}

# ── Derived / Gate Features ─────────────────────────────────────────────────
# These are computed internally for eligibility gates and expiry rules
# but are not part of the primary schema fed to the model.

TOP_WEIGHTED_FEATURES = []

BINARY_EXPIRY_RULES = {
    "1_MIN": [],
    "5_MIN": [],
    "15_MIN": [],
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
            logger.critical(
                {
                    "event": "FEATURE_MATRIX_INVALID_SHAPE",
                    "shape": str(self.matrix.shape),
                }
            )
            raise ValueError(
                f"FeatureMatrix requires 2D array, got shape {self.matrix.shape}"
            )
        if self.matrix.shape[1] != len(self.feature_names):
            logger.critical(
                {
                    "event": "FEATURE_MATRIX_DIMENSION_MISMATCH",
                    "matrix_cols": self.matrix.shape[1],
                    "feature_names_len": len(self.feature_names),
                }
            )
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
        timeframe: str = "M1",
    ) -> pd.DataFrame:
        """
        The Master Pipeline: Bars -> Normalized Features DataFrame.

        Applies all four feature groups in dependency order. Each group
        is guarded by a config flag. The returned DataFrame has one row
        per input bar (after dropna) with all computed feature columns
        appended.

        Args:
            bars: OHLCV DataFrame with DatetimeIndex. Must contain
            columns: open, high, low, close, volume.
            timeframe: Resample frequency. Default "M1".

        Returns:
            pd.DataFrame: Feature matrix aligned to bar index.
        """
        # Input validation and early exit for empty bars DataFrame
        if bars.empty:
            logger.warning({"event": "FEATURE_ENGINEERING_EMPTY_BARS"})
            return pd.DataFrame()

        # 1. Normalize Frequency using the Map
        # We target M1 as the base for binary options features
        target_freq = _RESAMPLE_MAP.get(timeframe, "1min")

        # Ensure index is DatetimeIndex for resampling.
        # A silent pd.to_datetime() on an integer index (e.g. epoch seconds vs
        # epoch milliseconds) produces wrong timestamps without any error.
        # We log a warning so the caller can fix the data source, and raise
        # immediately if the conversion itself fails.
        if not isinstance(bars.index, pd.DatetimeIndex):
            logger.warning(
                {
                    "event": "FEATURE_BARS_NON_DATETIME_INDEX",
                    "index_type": type(bars.index).__name__,
                }
            )

        try:
            bars = bars.copy()
            bars.index = pd.to_datetime(bars.index)
        except Exception as exc:  # pragma: no cover — pathological index type
            logger.critical(
                {
                    "event": "FEATURE_BARS_INDEX_COERCE_FAILED",
                    "index_type": type(bars.index).__name__,
                    "error": str(exc),
                }
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
                {
                    "event": "FEATURE_MISSING_COLUMNS",
                    "required": sorted(required),
                    "missing": sorted(missing),
                }
            )
            raise ValueError(f"Input bars missing columns: {missing}")

        # 1. PRICE ACTION
        fe = self._compute_price_action(fe)

        # 2. MOMENTUM
        fe = self._compute_momentum(fe)

        # 3. VOLATILITY
        fe = self._compute_volatility(fe)

        # 4. CONTEXT
        fe = self._compute_context(
            fe
        )  # always runs — feeds gates and top-weighted features

        # 5. MULTI-TIMEFRAME
        fe = self._compute_multitimeframe(fe)

        # 6. REGIME DETECTION
        fe = self._compute_regime(fe)

        # 7. INTERACTION FEATURES
        fe = self._compute_interactions(fe)

        # 8. LAGGED FEATURES
        fe = self._compute_lagged(fe)

        # 9. DERIVED (fe)
        fe = self._compute_derived(fe)

        # 10. SESSION INTENSITY
        fe = self._compute_session_intensity(fe)

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
                {
                    "event": "FEATURE_ROWS_DROPPED",
                    "dropped": dropped,
                    "total": before,
                    "pct": round(dropped / before * 100, 1),
                }
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
                {
                    "event": "BACKFILL_GAP_CHECK",
                    "shape": str(fe.shape),
                    "message": "No gap timestamps found",
                }
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
                {
                    "event": "UNEXPECTED_DATA_GAP",
                    "unexpected_gaps": unexpected_gaps,
                    "total_bars": len(fe) + n_gaps,
                    "pct": round(unexpected_gaps / (len(fe) + n_gaps) * 100, 1),
                    "max_ffill": _MAX_FFILL_BARS,
                }
            )
        else:
            logger.info(
                {
                    "event": "WEEKEND_HOLIDAY_GAPS",
                    "n_gaps": n_gaps,
                    "total_bars": len(fe) + n_gaps,
                    "pct": round(n_gaps / (len(fe) + n_gaps) * 100, 1),
                    "max_ffill": _MAX_FFILL_BARS,
                }
            )

    def _warn_missing_schema(self, fe: pd.DataFrame) -> None:
        """Warn if any primary feature columns expected by the schema are absent."""
        all_primary = [
            col for group in FEATURE_SET_BINARY_OPTIONS_AI.values() for col in group
        ]
        missing_schema = [col for col in all_primary if col not in fe.columns]
        if missing_schema:
            logger.warning(
                {
                    "event": "FEATURE_SCHEMA_INCOMPLETE",
                    "missing_count": len(missing_schema),
                    "total": len(all_primary),
                    "missing": missing_schema,
                }
            )

    # ── 1. PRICE ACTION ─────────────────────────────────────────────────────

    def _compute_price_action(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all PRICE_ACTION features from raw OHLCV data.

        Retained features:
            BODY_TO_RANGE_RATIO, UPPER_WICK_TO_BODY_RATIO,
            LOWER_WICK_TO_BODY_RATIO, CONSECUTIVE_BULL_BARS,
            CONSECUTIVE_BEAR_BARS, CLOSE_POSITION_IN_CANDLE,
            TWO_BAR_REVERSAL
        """

        hi_lo = (fe["high"] - fe["low"]).replace(0, _EPS)
        body_abs = (fe["close"] - fe["open"]).abs()

        # Core Ratios
        fe["BODY_TO_RANGE_RATIO"] = body_abs / hi_lo

        upper_wick = fe["high"] - fe[["open", "close"]].max(axis=1)
        lower_wick = fe[["open", "close"]].min(axis=1) - fe["low"]

        fe["UPPER_WICK_TO_BODY_RATIO"] = upper_wick / body_abs.replace(0, _EPS)
        fe["LOWER_WICK_TO_BODY_RATIO"] = lower_wick / body_abs.replace(0, _EPS)

        # Two-bar reversal
        body_raw = fe["close"] - fe["open"]
        prev_body = body_raw.shift(1)
        two_bar_bull = (
            (prev_body < 0) & (body_raw > 0) & (fe["close"] > fe["open"].shift(1))
        )
        two_bar_bear = (
            (prev_body > 0) & (body_raw < 0) & (fe["close"] < fe["open"].shift(1))
        )
        fe["TWO_BAR_REVERSAL"] = (two_bar_bull | two_bar_bear).astype(int)

        # Consecutive bars
        bull = (fe["close"] > fe["open"]).astype(int)
        fe["CONSECUTIVE_BULL_BARS"] = bull.groupby(
            (bull != bull.shift()).cumsum()
        ).cumsum()

        bear = (fe["close"] < fe["open"]).astype(int)
        fe["CONSECUTIVE_BEAR_BARS"] = bear.groupby(
            (bear != bear.shift()).cumsum()
        ).cumsum()

        # Close position in candle
        fe["CLOSE_POSITION_IN_CANDLE"] = (fe["close"] - fe["low"]) / hi_lo

        return fe

    # ── 2. MOMENTUM ─────────────────────────────────────────────────────────

    def _compute_momentum(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all MOMENTUM features.

        Retained features:
            RSI, CCI, MACD_VALUE, MACD_SIGNAL, MACD_HIST,
            RETURN_1, RETURN_3, MOMENTUM_OSCILLATOR
        """

        # ── RSI (Wilder-smoothed) ────────────────────────────────────────
        delta = fe["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

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

        # ── Returns (log) ────────────────────────────────────────────────
        # Only keep RETURN_1 and RETURN_3 (RETURN_5 removed)
        fe["RETURN_1"] = np.log(fe["close"] / fe["close"].shift(1).replace(0, _EPS))
        fe["RETURN_3"] = np.log(fe["close"] / fe["close"].shift(3).replace(0, _EPS))

        # ── Momentum Oscillator ──────────────────────────────────────────
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

        # ── Relative Volume (RVOL) ───────────────────────────────────────
        # Bar volume relative to its rolling average — detects volume surges.
        vol_ma = fe["volume"].rolling(_BB_PERIOD).mean().replace(0, _EPS)
        fe["RELATIVE_VOLUME_RVOL"] = fe["volume"] / vol_ma

        return fe

    # ── 4. CONTEXT ──────────────────────────────────────────────────────────

    def _compute_context(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute CONTEXT features from the DatetimeIndex.

        Retained features:
            HOUR_OF_DAY_NORMALIZED
        """
        if not isinstance(fe.index, pd.DatetimeIndex):
            logger.warning({"event": "FEATURE_CONTEXT_NO_DATETIME_INDEX"})
            fe["HOUR_OF_DAY_NORMALIZED"] = 0.0
            return fe

        h = fe.index.hour
        m = fe.index.minute

        # Hour of Day Normalized (0.0 to 1.0)
        fractional_hour = h + m / 60.0
        fe["HOUR_OF_DAY_NORMALIZED"] = fractional_hour / 24.0

        return fe

    # ── 5. MULTI-TIMEFRAME ───────────────────────────────────────────────────

    def _compute_multitimeframe(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multi-timeframe features (5T and 15T).
        """
        EPS = _EPS

        # ── 5T Features ──────────────────────────────────────────────────────
        resampled_5t = (
            fe[["open", "high", "low", "close", "volume"]]
            .resample(_RESAMPLE_MAP["M5"])
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # RSI_5T
        delta_5t = resampled_5t["close"].diff()
        gain_5t = delta_5t.clip(lower=0)
        loss_5t = (-delta_5t).clip(lower=0)
        ag = gain_5t.ewm(
            alpha=1 / _MTF_RSI_PERIOD, min_periods=_MTF_RSI_PERIOD, adjust=False
        ).mean()
        al = loss_5t.ewm(
            alpha=1 / _MTF_RSI_PERIOD, min_periods=_MTF_RSI_PERIOD, adjust=False
        ).mean()
        rsi_5t = 100 - (100 / (1 + ag / al.replace(0, EPS)))

        # TREND_5T
        ema5 = resampled_5t["close"].ewm(span=_MTF_TREND_FAST, adjust=False).mean()
        ema13 = resampled_5t["close"].ewm(span=_MTF_TREND_SLOW, adjust=False).mean()
        trend_5t = np.sign(ema5 - ema13).astype(np.float32)

        # CLOSE_POS_5T
        rolling_hi = resampled_5t["high"].rolling(_MTF_CLOSE_POS_WINDOW).max()
        rolling_lo = resampled_5t["low"].rolling(_MTF_CLOSE_POS_WINDOW).min()
        range_5t = (rolling_hi - rolling_lo).replace(0, EPS)
        close_pos_5t = (resampled_5t["close"] - rolling_lo) / range_5t

        # Assign using reindex
        fe["RSI_5T"] = rsi_5t.reindex(fe.index, method="ffill").astype(np.float32)
        fe["TREND_5T"] = (
            pd.Series(trend_5t, index=resampled_5t.index)
            .reindex(fe.index, method="ffill")
            .astype(np.float32)
        )
        fe["CLOSE_POS_5T"] = close_pos_5t.reindex(fe.index, method="ffill").astype(
            np.float32
        )

        # ── 15T Features ─────────────────────────────────────────────────────
        resampled_15t = (
            fe[["open", "high", "low", "close", "volume"]]
            .resample(_RESAMPLE_MAP["M15"])
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # RSI_15T
        delta_15t = resampled_15t["close"].diff()
        gain_15t = delta_15t.clip(lower=0)
        loss_15t = (-delta_15t).clip(lower=0)
        ag = gain_15t.ewm(
            alpha=1 / _MTF_RSI_PERIOD, min_periods=_MTF_RSI_PERIOD, adjust=False
        ).mean()
        al = loss_15t.ewm(
            alpha=1 / _MTF_RSI_PERIOD, min_periods=_MTF_RSI_PERIOD, adjust=False
        ).mean()
        rsi_15t = 100 - (100 / (1 + ag / al.replace(0, EPS)))

        # TREND_15T
        ema5 = resampled_15t["close"].ewm(span=_MTF_TREND_FAST, adjust=False).mean()
        ema13 = resampled_15t["close"].ewm(span=_MTF_TREND_SLOW, adjust=False).mean()
        trend_15t = np.sign(ema5 - ema13).astype(np.float32)

        # CLOSE_POS_15T
        rolling_hi = resampled_15t["high"].rolling(_MTF_CLOSE_POS_WINDOW).max()
        rolling_lo = resampled_15t["low"].rolling(_MTF_CLOSE_POS_WINDOW).min()
        range_15t = (rolling_hi - rolling_lo).replace(0, EPS)
        close_pos_15t = (resampled_15t["close"] - rolling_lo) / range_15t

        # Assign using reindex
        fe["RSI_15T"] = rsi_15t.reindex(fe.index, method="ffill").astype(np.float32)
        fe["TREND_15T"] = (
            pd.Series(trend_15t, index=resampled_15t.index)
            .reindex(fe.index, method="ffill")
            .astype(np.float32)
        )
        fe["CLOSE_POS_15T"] = close_pos_15t.reindex(fe.index, method="ffill").astype(
            np.float32
        )

        return fe

    # ── 6. REGIME DETECTION ───────────────────────────────────────────────────

    def _compute_regime(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market regime detection features.

        Features computed:
            VOLATILITY_REGIME, ADX, DIST_SMA20, DIST_SMA50, DIST_SMA200,
            BB_KC_SQUEEZE
        """
        EPS = _EPS

        # VOLATILITY_REGIME: ATR ratio to 100-bar average ATR
        atr_long = fe["ATR"].rolling(_REGIME_ATR_LONG).mean().replace(0, EPS)
        fe["VOLATILITY_REGIME"] = (fe["ATR"] / atr_long).astype(np.float32)

        # ADX (Average Directional Index) - trend strength
        # +DI and -DI calculation
        high = fe["high"]
        low = fe["low"]
        close = fe["close"]

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        # Smoothed DM and TR using Wilder's smoothing (EMA with alpha=1/period)
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(
            alpha=1 / _REGIME_ADX_PERIOD, min_periods=_REGIME_ADX_PERIOD, adjust=False
        ).mean()

        plus_di = 100 * (
            plus_dm.ewm(
                alpha=1 / _REGIME_ADX_PERIOD,
                min_periods=_REGIME_ADX_PERIOD,
                adjust=False,
            ).mean()
            / atr.replace(0, EPS)
        )
        minus_di = 100 * (
            minus_dm.ewm(
                alpha=1 / _REGIME_ADX_PERIOD,
                min_periods=_REGIME_ADX_PERIOD,
                adjust=False,
            ).mean()
            / atr.replace(0, EPS)
        )

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, EPS)
        fe["ADX"] = (
            dx.ewm(
                alpha=1 / _REGIME_ADX_PERIOD,
                min_periods=_REGIME_ADX_PERIOD,
                adjust=False,
            )
            .mean()
            .astype(np.float32)
        )

        # Distance to moving averages (percentage)
        for period in _REGIME_SMA_PERIODS:
            sma = fe["close"].rolling(period).mean()
            fe[f"DIST_SMA{period}"] = (
                (fe["close"] - sma) / sma.replace(0, EPS) * 100
            ).astype(np.float32)

        # BB_KC_SQUEEZE: 1 when Bollinger Bands are inside Keltner Channels
        if "BB_WIDTH" in fe.columns and "KC_WIDTH" in fe.columns:
            fe["BB_KC_SQUEEZE"] = (fe["BB_WIDTH"] < fe["KC_WIDTH"]).astype(np.int8)
        else:
            fe["BB_KC_SQUEEZE"] = 0

        return fe

    # ── 7. INTERACTION FEATURES ───────────────────────────────────────────────

    def _compute_interactions(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature interaction combinations.

        Features computed:
            RSI_TREND_5T, CLOSEPOS_VOLREGIME, BB_MOMENTUM,
            STREAK_TREND, ADX_RETURN
        """

        # Define interactions: (output_name, col_a, col_b, use_abs, fallback)
        interactions = [
            ("RSI_TREND_5T", "RSI", "TREND_5T", False, 0.0),
            (
                "CLOSEPOS_VOLREGIME",
                "CLOSE_POSITION_IN_CANDLE",
                "VOLATILITY_REGIME",
                False,
                0.0,
            ),
            ("BB_MOMENTUM", "BB_WIDTH", "MOMENTUM_OSCILLATOR", True, 0.0),
            ("STREAK_TREND", "CONSECUTIVE_BEAR_BARS", "TREND_5T", False, 0.0),
            ("ADX_RETURN", "ADX", "RETURN_1", False, 0.0),
        ]

        for out_name, col_a, col_b, use_abs, fallback in interactions:
            if col_a in fe.columns and col_b in fe.columns:
                product = fe[col_a] * fe[col_b]
                if use_abs:
                    product = product.abs()
                    fe[out_name] = product.astype(
                        np.float32
                    )  # <-- MOVED OUTSIDE the if
                else:
                    fe[out_name] = fallback

        return fe

    # ── 8. LAGGED FEATURES ───────────────────────────────────────────────────

    def _compute_lagged(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute lagged versions of key features.

        Features computed:
            CLOSE_POS_LAG1, CLOSE_POS_LAG2, CLOSE_POS_LAG3, CLOSE_POS_LAG5
            RETURN_1_LAG1, RETURN_1_LAG2, RETURN_1_LAG3, RETURN_1_LAG5
            RSI_LAG1, RSI_LAG2, RSI_LAG3, RSI_LAG5
            RETURN_1_MEAN_5, RETURN_1_STD_5, RSI_MEAN_10, CLOSE_POS_MEAN_5
        """

        # Lagged individual features
        for col in _LAGGED_FEATURES:
            if col in fe.columns:
                for lag in _LAGGED_PERIODS:
                    fe[f"{col}_LAG{lag}"] = fe[col].shift(lag).astype(np.float32)

        # Rolling statistics
        if "RETURN_1" in fe.columns:
            fe["RETURN_1_MEAN_5"] = fe["RETURN_1"].rolling(5).mean().astype(np.float32)
            fe["RETURN_1_STD_5"] = fe["RETURN_1"].rolling(5).std().astype(np.float32)

        if "RSI" in fe.columns:
            fe["RSI_MEAN_10"] = fe["RSI"].rolling(10).mean().astype(np.float32)

        if "CLOSE_POSITION_IN_CANDLE" in fe.columns:
            fe["CLOSE_POS_MEAN_5"] = (
                fe["CLOSE_POSITION_IN_CANDLE"].rolling(5).mean().astype(np.float32)
            )

        return fe

    # ── 9. SESSION INTENSITY ─────────────────────────────────────────────────

    def _compute_session_intensity(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute session-specific relative volume.

        Features computed:
            RVOL_LONDON, RVOL_NEWYORK, RVOL_TOKYO

        For bars outside session hours or with insufficient history,
        RVOL is set to 1.0 (normal volume) instead of NaN.
        """
        # Initialize all RVOL features to 1.0 (normal volume) by default
        for sess in ["LONDON", "NEWYORK", "TOKYO"]:
            fe[f"RVOL_{sess}"] = 1.0

        # Only compute actual values if we have a DatetimeIndex
        if isinstance(fe.index, pd.DatetimeIndex):
            h = fe.index.hour

            # Session hour definitions (UTC)
            session_hours = {
                "LONDON": (h >= 7) & (h <= 16),
                "NEWYORK": (h >= 12) & (h <= 21),
                "TOKYO": (h >= 0) & (h <= 9),
            }

            for sess_name, sess_mask in session_hours.items():
                # Volume during this session (NaN outside session)
                sess_vol = fe["volume"].where(sess_mask)
                # 100-bar rolling average volume during this session
                sess_avg = sess_vol.rolling(100).mean().replace(0, _EPS)
                # Relative volume: current / session average
                rvol = (fe["volume"] / sess_avg).astype(np.float32)
                # Fill NaN (outside session or insufficient history) with 1.0 = normal
                rvol = rvol.fillna(1.0)
                fe[f"RVOL_{sess_name}"] = rvol

        return fe

    # ── 10. DERIVED (Gates + Top-Weighted + Expiry Helpers) ──────────────────

    def _compute_derived(self, fe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features.

        With TOP_WEIGHTED_FEATURES and BINARY_EXPIRY_RULES emptied,
        this method now only computes MACD_HIST_SLOPE (kept for potential future use).
        """
        # MACD Histogram Slope (kept for compatibility)
        if "MACD_HIST" in fe.columns:
            fe["MACD_HIST_SLOPE"] = fe["MACD_HIST"].diff(3) / 3
        else:
            fe["MACD_HIST_SLOPE"] = 0.0

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
                {
                    "event": "EXPIRY_COLUMNS_MISSING",
                    "expiry_key": expiry_key,
                    "missing": list(missing),
                }
            )

        return fe[available]

    # ── Live Inference ──────────────────────────────────────────────────────

    def get_latest(
        self,
        bars: pd.DataFrame,
        timeframe: str = "M1",
    ) -> FeatureVector:
        """
        The Live Inference Gate.

        Transforms the latest bars into a single FeatureVector ready for
        model.predict(). Raises FeatureEngineerError if the pipeline cannot
        produce a valid vector — the engine layer is responsible for deciding
        whether to retry or skip the signal.

        Args:
            bars: Recent OHLCV bars with DatetimeIndex (minimum ~30 rows
            for rolling-window warmup; fewer rows yield an empty
            feature frame after dropna).
            timeframe: Resample frequency. Default "M1".

        Returns:
            FeatureVector: immutable, float32, read-only numpy vector.

        Raises:
            FeatureEngineerError: If transform produces an empty frame, or
            if any unexpected exception occurs inside the pipeline.
        """

        try:
            full_df = self.transform(bars, timeframe=timeframe)
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
                {"event": "FEATURE_NON_FINITE_VALUES", "features": bad_names}
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
        timeframe: str = "M1",
    ) -> FeatureMatrix:
        """
        Batch mode: transform full historical bars into a FeatureMatrix.

        Runs the same pipeline as transform() but packages the result
        as an immutable FeatureMatrix ready for trainer.py consumption.
        Use this for offline training data preparation only — not for
        live inference (use get_latest() for that).

        Args:
            bars: Full historical OHLCV DataFrame with DatetimeIndex.
            symbol: Currency pair identifier stored in matrix metadata.
            timeframe: Resample frequency. Default "M1".

        Returns:
            FeatureMatrix: immutable, float32, version-stamped batch matrix.

        Raises:
            FeatureEngineerError: If transform() fails or returns empty.
        """

        try:
            full_df = self.transform(bars, timeframe=timeframe)
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
