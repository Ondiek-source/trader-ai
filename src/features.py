"""
features.py — Feature engineering from tick data.

All indicators implemented in pure pandas/numpy — no ta-lib dependency.
Produces a feature DataFrame ready for ML training and live inference.

Workflow:
    1. ``resample_to_1min(tick_df)``              → 1-minute OHLCV bars
    2. ``compute_features(tick_df, expiry_seconds)`` → full feature + label DataFrame
    3. ``get_feature_columns()``                   → list of feature column names
    4. ``extract_live_features(bars, ticks)``      → single-row feature Series for inference
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MEMORY_SAVER_MODE: bool = os.environ.get("MEMORY_SAVER_MODE", "false").lower() == "true"

# ── Constants ─────────────────────────────────────────────────────────────────

SESSION_BOUNDS: dict[str, tuple[int, int]] = {
    "london": (7, 16),
    "new_york": (12, 21),
    "tokyo": (0, 9),
    "london_ny_overlap": (12, 16),
}

MOMENTUM_WINDOWS_S: list[int] = [5, 30, 60] if MEMORY_SAVER_MODE else [1, 5, 15, 30, 60]
TICK_VELOCITY_WINDOWS_S: list[int] = [10, 30] if MEMORY_SAVER_MODE else [5, 10, 30]

_TICK_FEATURE_COLS: list[str] = (
    [
        "spread_zscore",
        "spread_mean_reversion",
        "tick_velocity_10s",
        "tick_velocity_30s",
        "price_momentum_5s",
        "price_momentum_30s",
        "price_momentum_60s",
    ]
    if MEMORY_SAVER_MODE
    else [
        "spread_zscore",
        "spread_mean_reversion",
        "tick_velocity_5s",
        "tick_velocity_10s",
        "tick_velocity_30s",
        "price_momentum_1s",
        "price_momentum_5s",
        "price_momentum_15s",
        "price_momentum_30s",
        "price_momentum_60s",
    ]
)

# ── Public API ─────────────────────────────────────────────────────────────────


def resample_to_1min(tick_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a tick DataFrame into 1-minute OHLCV bars.

    Args:
        tick_df: DataFrame with columns ``timestamp``, ``bid``, ``ask``, ``spread``.

    Returns:
        DataFrame with columns: ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``spread_mean``, ``spread_max``, ``mid``.
    """
    df = tick_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    df["mid"] = (df["bid"] + df["ask"]) / 2

    ohlcv = df["mid"].resample("1min").ohlc()  # type: ignore[assignment]
    ohlcv.columns = pd.Index(["open", "high", "low", "close"])
    ohlcv["volume"] = df["mid"].resample("1min").count()
    ohlcv["spread_mean"] = df["spread"].resample("1min").mean()
    ohlcv["spread_max"] = df["spread"].resample("1min").max()
    ohlcv["mid"] = df["mid"].resample("1min").last()
    ohlcv = ohlcv.dropna(subset=["close"]).reset_index()  # type: ignore[assignment]
    return ohlcv


def compute_features(tick_df: pd.DataFrame, expiry_seconds: int = 60) -> pd.DataFrame:
    """
    Build a full feature + label DataFrame from raw tick data.

    Steps:
        1. Resample ticks to 1-min bars
        2. Compute technical indicators (RSI, MACD, Bollinger, EMA, Stochastic, ATR, etc.)
        3. Add session / time features
        4. Add tick-level aggregated features (velocity, spread z-score, momentum)
        5. Generate binary label (1 = price up at T + expiry_bars, 0 = down)
        6. Drop rows with NaN labels or features

    Args:
        tick_df: Raw tick DataFrame with ``timestamp``, ``bid``, ``ask``, ``spread``.
        expiry_seconds: Option expiry in seconds (60, 120, or 300).

    Returns:
        Feature DataFrame with ``label`` column.  Empty if insufficient data.
    """
    if tick_df.empty:
        logger.warning({"event": "empty_tick_df"})
        return pd.DataFrame()

    bars = resample_to_1min(tick_df)
    if len(bars) < 50:
        logger.warning({"event": "insufficient_bars", "count": len(bars)})
        return pd.DataFrame()

    bars = bars.set_index("timestamp")

    # ── Technical indicators ───────────────────────────────────────────────
    bars = _add_rsi(bars, period=14)
    bars = _add_macd(bars, fast=12, slow=26, signal=9)
    bars = _add_bollinger_bands(bars, period=20, std_dev=2)
    bars = _add_ema_crossover(bars, fast=9, slow=21)
    bars = _add_stochastic(bars, k_period=14, d_period=3)
    bars = _add_atr(bars, period=14)
    bars = _add_williams_r(bars, period=14)
    bars = _add_cci(bars, period=20)
    bars = _add_momentum(bars, period=10)
    bars = _add_volume_momentum(bars, short_period=5, long_period=20)

    # ── Session / time features ────────────────────────────────────────────
    bars = _add_session_features(bars)

    # ── Tick-level aggregated features ─────────────────────────────────────
    bars = _add_tick_features(bars, tick_df)

    # ── Label generation ───────────────────────────────────────────────────
    expiry_bars = max(1, expiry_seconds // 60)
    bars["label"] = (bars["close"].shift(-expiry_bars) > bars["close"]).astype(float)
    # Last expiry_bars rows can't have a label
    bars.loc[bars.index[-expiry_bars:], "label"] = np.nan

    bars = bars.reset_index()
    bars = bars.dropna(subset=["label"]).copy()
    bars["label"] = bars["label"].astype(int)

    # Downcast float64 → float32 to halve memory
    float_cols = bars.select_dtypes(include=["float64"]).columns
    bars = bars.astype({col: "float32" for col in float_cols})

    return bars


def get_feature_columns() -> list[str]:
    """
    Return the canonical list of feature column names.

    The returned list respects ``MEMORY_SAVER_MODE`` — in saver mode
    tick-velocity and momentum columns that are excluded at build time
    are also excluded here.

    Excludes: ``label``, ``timestamp``, OHLCV, ``volume``, ``spread_mean``,
    ``spread_max``, ``mid``.
    """
    indicator_cols = [
        # RSI
        "rsi",
        # MACD
        "macd",
        "macd_signal",
        "macd_hist",
        # Bollinger Bands
        "bb_upper",
        "bb_lower",
        "bb_mid",
        "bb_pct_b",
        "bb_bandwidth",
        # EMA crossover
        "ema_fast",
        "ema_slow",
        "ema_cross",
        # Stochastic
        "stoch_k",
        "stoch_d",
        # ATR
        "atr",
        "atr_pct",
        # Williams %R
        "williams_r",
        # CCI
        "cci",
        # Momentum
        "momentum",
        # Volume momentum
        "vol_momentum",
        # Session features
        "hour_of_day",
        "day_of_week",
        "session_london",
        "session_new_york",
        "session_tokyo",
        "session_london_ny_overlap",
        "is_high_volatility_session",
    ]

    return indicator_cols + _TICK_FEATURE_COLS


def get_indicator_feature_groups() -> dict[str, list[str]]:
    """
    Map indicator names to their feature columns.

    Used for per-indicator confidence scoring.  Tick-velocity columns
    respect ``MEMORY_SAVER_MODE``.
    """
    return {
        "rsi": ["rsi"],
        "macd": ["macd", "macd_signal", "macd_hist"],
        "bollinger": ["bb_pct_b", "bb_bandwidth"],
        "ema_cross": ["ema_cross", "ema_fast", "ema_slow"],
        "stochastic": ["stoch_k", "stoch_d"],
        "atr": ["atr", "atr_pct"],
        "williams_r": ["williams_r"],
        "cci": ["cci"],
        "momentum": ["momentum"],
        "volume_momentum": [
            "vol_momentum",
            *[c for c in _TICK_FEATURE_COLS if c.startswith("tick_velocity_")],
        ],
    }


# ── Technical indicator implementations (pure pandas/numpy) ───────────────────


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()  # type: ignore[return-value]


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()  # type: ignore[return-value]


def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)
    return df


def _add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = _ema(df["close"], fast)
    ema_slow = _ema(df["close"], slow)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], signal)
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def _add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: upper, lower, mid, %B, bandwidth."""
    sma = _sma(df["close"], period)
    std = df["close"].rolling(window=period, min_periods=period).std()
    df["bb_mid"] = sma
    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    band_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan).astype(float)
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / band_range
    df["bb_bandwidth"] = band_range / sma.replace(0, np.nan).astype(float)
    return df


def _add_ema_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    """EMA crossover signal: +1 bullish, -1 bearish, 0 flat."""
    df["ema_fast"] = _ema(df["close"], fast)
    df["ema_slow"] = _ema(df["close"], slow)
    df["ema_cross"] = np.sign(df["ema_fast"] - df["ema_slow"])
    return df


def _add_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator %K and %D."""
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_min) / denom
    df["stoch_d"] = df["stoch_k"].rolling(window=d_period, min_periods=d_period).mean()
    df["stoch_k"] = df["stoch_k"].fillna(50.0)
    df["stoch_d"] = df["stoch_d"].fillna(50.0)
    return df


def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range and ATR as percentage of close."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)
    return df


def _add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R oscillator."""
    high_max = df["high"].rolling(window=period, min_periods=period).max()
    low_min = df["low"].rolling(window=period, min_periods=period).min()
    denom = (high_max - low_min).replace(0, np.nan)
    df["williams_r"] = -100 * (high_max - df["close"]) / denom
    df["williams_r"] = df["williams_r"].fillna(-50.0)
    return df


def _add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    df["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    df["cci"] = df["cci"].fillna(0.0)
    return df


def _add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Rate of change: close - close[N periods ago]."""
    df["momentum"] = df["close"].diff(period)
    return df


def _add_volume_momentum(
    df: pd.DataFrame, short_period: int = 5, long_period: int = 20
) -> pd.DataFrame:
    """Short/long volume ratio — high value = volume spike."""
    short_vol = (
        df["volume"].rolling(window=short_period, min_periods=short_period).mean()
    )
    long_vol = df["volume"].rolling(window=long_period, min_periods=long_period).mean()
    df["vol_momentum"] = (short_vol - long_vol) / long_vol.replace(0, np.nan)
    df["vol_momentum"] = df["vol_momentum"].fillna(0.0)
    return df


def _add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trading session flags and time-of-day features."""
    idx = pd.DatetimeIndex(df.index)

    hour = idx.hour.to_numpy()
    df["hour_of_day"] = hour
    df["day_of_week"] = idx.dayofweek.to_numpy()

    lo_s, lo_e = SESSION_BOUNDS["london"]
    ny_s, ny_e = SESSION_BOUNDS["new_york"]
    tk_s, tk_e = SESSION_BOUNDS["tokyo"]
    ov_s, ov_e = SESSION_BOUNDS["london_ny_overlap"]

    df["session_london"] = ((hour >= lo_s) & (hour < lo_e)).astype(int)
    df["session_new_york"] = ((hour >= ny_s) & (hour < ny_e)).astype(int)
    df["session_tokyo"] = ((hour >= tk_s) & (hour < tk_e)).astype(int)
    df["session_london_ny_overlap"] = ((hour >= ov_s) & (hour < ov_e)).astype(int)
    df["is_high_volatility_session"] = df["session_london_ny_overlap"]
    return df


def _add_tick_features(bars: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tick-level features and join to 1-min bar DataFrame.

    Features computed (all respect ``MEMORY_SAVER_MODE``):
        - ``spread_zscore``, ``spread_mean_reversion``
        - ``tick_velocity_{w}s`` for each window in ``TICK_VELOCITY_WINDOWS_S``
        - ``price_momentum_{w}s`` for each window in ``MOMENTUM_WINDOWS_S``

    Args:
        bars: 1-min OHLCV DataFrame (index = timestamp).
        tick_df: Raw tick DataFrame.

    Returns:
        *bars* with tick feature columns added.
    """
    if tick_df.empty:
        for col in _TICK_FEATURE_COLS:
            bars[col] = 0.0
        return bars

    tk = tick_df.copy()
    tk["timestamp"] = pd.to_datetime(tk["timestamp"], utc=True)
    tk = tk.sort_values("timestamp").set_index("timestamp")
    tk["mid"] = (tk["bid"] + tk["ask"]) / 2

    # Spread z-score on 60-second rolling window
    spread_roll = tk["spread"].rolling("60s")
    tk["spread_zscore"] = (
        tk["spread"] - spread_roll.mean()
    ) / spread_roll.std().replace(0, np.nan)
    tk["spread_mean_reversion"] = tk["spread"] - spread_roll.mean()

    # Tick velocity: count ticks per second, then rolling sum
    tick_count_1s = tk["mid"].resample("1s").count()
    for w in TICK_VELOCITY_WINDOWS_S:
        col = f"tick_velocity_{w}s"
        velocity = tick_count_1s.rolling(w, min_periods=1).sum()
        tk[col] = velocity.reindex(tk.index, method="ffill").fillna(0)

    # Price momentum over multiple windows
    for w in MOMENTUM_WINDOWS_S:
        col = f"price_momentum_{w}s"
        tk[col] = tk["mid"].diff(periods=max(1, w))

    # Resample tick features to 1-min alignment and join
    tick_features_1min = tk[_TICK_FEATURE_COLS].resample("1min").last()
    bars = bars.join(tick_features_1min, how="left")
    bars[_TICK_FEATURE_COLS] = bars[_TICK_FEATURE_COLS].fillna(0.0)
    return bars


# ── Live feature extraction ───────────────────────────────────────────────────


def extract_live_features(
    recent_bars: pd.DataFrame,
    recent_ticks: pd.DataFrame | None = None,
    expiry_seconds: int = 60,
) -> pd.Series | None:
    """
    Extract a single feature row from recent 1-min bars for live inference.

    Requires at least 30 bars of OHLCV history.  Optionally includes
    recent tick data for tick-level features.

    Args:
        recent_bars: DataFrame with OHLCV columns and a ``timestamp`` column
            (or a DatetimeIndex).
        recent_ticks: Optional raw tick DataFrame with ``timestamp``, ``bid``,
            ``ask``, ``spread``.
        expiry_seconds: Option expiry for label (not used for inference,
            but passed to ``compute_features``).

    Returns:
        A :class:`~pandas.Series` of feature values, or ``None`` if data
        is insufficient.
    """
    if len(recent_bars) < 30:
        logger.debug(
            {"event": "insufficient_bars_for_live_features", "count": len(recent_bars)}
        )
        return None

    try:
        bars = recent_bars.copy()
        if "timestamp" not in bars.columns:
            bars = bars.reset_index()

        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp")

        # Compute indicators directly on the pre-built bars
        bars = _add_rsi(bars, period=14)
        bars = _add_macd(bars, fast=12, slow=26, signal=9)
        bars = _add_bollinger_bands(bars, period=20, std_dev=2)
        bars = _add_ema_crossover(bars, fast=9, slow=21)
        bars = _add_stochastic(bars, k_period=14, d_period=3)
        bars = _add_atr(bars, period=14)
        bars = _add_williams_r(bars, period=14)
        bars = _add_cci(bars, period=20)
        bars = _add_momentum(bars, period=10)
        bars = _add_volume_momentum(bars, short_period=5, long_period=20)
        bars = _add_session_features(bars)

        # Tick features from recent ticks (if available)
        if recent_ticks is not None and not recent_ticks.empty:
            bars = _add_tick_features(bars, recent_ticks)
        else:
            for col in _TICK_FEATURE_COLS:
                bars[col] = 0.0

        feature_cols = get_feature_columns()
        available = [c for c in feature_cols if c in bars.columns]
        last_row = bars.iloc[-1][available]

        if last_row.isna().any():
            logger.debug({"event": "live_features_contain_nan"})
            return None

        return last_row

    except Exception as exc:
        logger.warning({"event": "live_feature_extraction_failed", "error": str(exc)})
        return None
