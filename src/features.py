"""
features.py — Feature engineering from tick data.

All indicators implemented in pure pandas/numpy — no ta-lib dependency.
Produces a feature DataFrame ready for ML training and live inference.

Workflow:
    1. resample_to_1min(tick_df)     → 1-minute OHLCV bars
    2. compute_features(tick_df, expiry_seconds) → full feature + label DataFrame
    3. get_feature_columns()         → list of feature column names (no label cols)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)
MEMORY_SAVER_MODE = os.environ.get("MEMORY_SAVER_MODE", "false").lower() == "true"


# ── Constants ─────────────────────────────────────────────────────────────────

SESSION_BOUNDS = {
    "london": (7, 16),  # UTC hours [start, end)
    "new_york": (12, 21),
    "tokyo": (0, 9),
    "london_ny_overlap": (12, 16),
}

if MEMORY_SAVER_MODE:
    MOMENTUM_WINDOWS_S = [5, 30, 60]      # Reduced
    TICK_VELOCITY_WINDOWS_S = [10, 30]    # Reduced
else:
    MOMENTUM_WINDOWS_S = [1, 5, 15, 30, 60]   # Full
    TICK_VELOCITY_WINDOWS_S = [5, 10, 30]     # Full
    

# ── Public API ─────────────────────────────────────────────────────────────────


def resample_to_1min(tick_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a tick DataFrame into 1-minute OHLCV bars.

    Input columns required: timestamp, bid, ask, spread
    Output columns: timestamp (bar open), open, high, low, close,
                    volume (tick count), spread_mean, spread_max, mid
    """
    df = tick_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    df["mid"] = (df["bid"] + df["ask"]) / 2

    ohlcv: pd.DataFrame = df["mid"].resample("1min").ohlc()  # type: ignore[assignment]
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
        1. Resample to 1-min bars
        2. Compute all technical indicators
        3. Add session / time features
        4. Add tick-level aggregated features (velocity, spread z-score, imbalance, momentum)
        5. Generate binary label (1 = price up at T + expiry_bars, 0 = down)
        6. Drop NaN rows that cannot form a complete label or feature window

    Returns a DataFrame ready for ML. Label column: 'label'.
    """
    if tick_df.empty:
        logger.warning({"event": "empty_tick_df"})
        return pd.DataFrame()

    bars = resample_to_1min(tick_df)
    if len(bars) < 50:
        logger.warning({"event": "insufficient_bars", "count": len(bars)})
        return pd.DataFrame()

    bars = bars.set_index("timestamp")

    # ── Technical indicators ──────────────────────────────────────────────────
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

    # ── Session / time features ───────────────────────────────────────────────
    bars = _add_session_features(bars)

    # ── Tick-level aggregated features ────────────────────────────────────────
    # Resample tick features to 1-min alignment
    bars = _add_tick_features(bars, tick_df)

    # ── Label generation ─────────────────────────────────────────────────────
    expiry_bars = max(1, expiry_seconds // 60)
    bars["label"] = (bars["close"].shift(-expiry_bars) > bars["close"]).astype(float)
    # Rows where label cannot be computed (last expiry_bars rows) → NaN
    bars.loc[bars.index[-expiry_bars:], "label"] = np.nan

    bars = bars.reset_index()
    bars = bars.dropna(subset=["label"]).copy()
    bars["label"] = bars["label"].astype(int)
    bars = bars.astype({col: 'float32' for col in bars.select_dtypes(include=['float64']).columns})
    return bars


def get_feature_columns() -> list[str]:
    """Return the canonical list of feature column names (excludes label, timestamp, OHLCV)."""
    return [
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
        # Tick-level aggregated features
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


def get_indicator_feature_groups() -> dict[str, list[str]]:
    """Map indicator names to their feature columns — used for per-indicator confidence."""
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
            "tick_velocity_5s",
            "tick_velocity_10s",
            "tick_velocity_30s",
        ],
    }


# ── Technical indicator implementations (pure pandas/numpy) ───────────────────


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()  # type: ignore[return-value]


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()  # type: ignore[return-value]


def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)
    return df


def _add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = _ema(df["close"], fast)
    ema_slow = _ema(df["close"], slow)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], signal)
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def _add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    sma = _sma(df["close"], period)
    std = df["close"].rolling(window=period, min_periods=1).std()
    df["bb_mid"] = sma
    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    band_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / band_range
    df["bb_bandwidth"] = band_range / sma.replace(0, np.nan)
    return df


def _add_ema_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    df["ema_fast"] = _ema(df["close"], fast)
    df["ema_slow"] = _ema(df["close"], slow)
    # 1 = fast above slow (bullish), -1 = fast below slow (bearish)
    df["ema_cross"] = np.sign(df["ema_fast"] - df["ema_slow"])
    return df


def _add_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    low_min = df["low"].rolling(window=k_period, min_periods=1).min()
    high_max = df["high"].rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_min) / denom
    df["stoch_d"] = df["stoch_k"].rolling(window=d_period, min_periods=1).mean()
    df["stoch_k"] = df["stoch_k"].fillna(50.0)
    df["stoch_d"] = df["stoch_d"].fillna(50.0)
    return df


def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(com=period - 1, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)
    return df


def _add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_max = df["high"].rolling(window=period, min_periods=1).max()
    low_min = df["low"].rolling(window=period, min_periods=1).min()
    denom = (high_max - low_min).replace(0, np.nan)
    df["williams_r"] = -100 * (high_max - df["close"]) / denom
    df["williams_r"] = df["williams_r"].fillna(-50.0)
    return df


def _add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    df["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    df["cci"] = df["cci"].fillna(0.0)
    return df


def _add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df["momentum"] = df["close"].diff(period)
    return df


def _add_volume_momentum(
    df: pd.DataFrame, short_period: int = 5, long_period: int = 20
) -> pd.DataFrame:
    short_vol = df["volume"].rolling(window=short_period, min_periods=1).mean()
    long_vol = df["volume"].rolling(window=long_period, min_periods=1).mean()
    df["vol_momentum"] = (short_vol - long_vol) / long_vol.replace(0, np.nan)
    df["vol_momentum"] = df["vol_momentum"].fillna(0.0)
    return df


def _add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    # Always convert to DatetimeIndex for reliable .hour / .dayofweek access
    try:
        idx = pd.DatetimeIndex(df.index)
    except Exception:
        for col in [
            "hour_of_day",
            "day_of_week",
            "session_london",
            "session_new_york",
            "session_tokyo",
            "session_london_ny_overlap",
            "is_high_volatility_session",
        ]:
            df[col] = 0
        return df

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
    Aggregate tick-level features (velocity, spread z-score, momentum)
    and join them to the 1-min bar DataFrame.
    """
    if tick_df.empty:
        for col in [
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
        ]:
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

    # Tick velocity: count ticks per second in rolling windows
    # Resample to 1-second count, then take rolling sums
    tick_count_1s = tk["mid"].resample("1s").count()
    for w in TICK_VELOCITY_WINDOWS_S:
        col = f"tick_velocity_{w}s"
        velocity = tick_count_1s.rolling(w, min_periods=1).sum()
        tk[col] = velocity.reindex(tk.index, method="ffill").fillna(0)  # type: ignore[call-arg]

    # Price momentum over multiple windows
    for w in MOMENTUM_WINDOWS_S:
        col = f"price_momentum_{w}s"
        tk[col] = tk["mid"].diff(periods=max(1, w))

    # Resample all tick features to 1-min (use last value per bar)
    tick_features_1min = (
        tk[
            [
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
        ]
        .resample("1min")
        .last()
    )

    bars = bars.join(tick_features_1min, how="left")

    # Fill any NaNs from join
    tick_feature_cols = [
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
    bars[tick_feature_cols] = bars[tick_feature_cols].fillna(0.0)
    return bars


# ── Live feature extraction for a single bar ──────────────────────────────────


def extract_live_features(
    recent_bars: pd.DataFrame, recent_ticks: pd.DataFrame | None = None
) -> pd.Series | None:
    """
    Extract a single feature row from the most recent 1-min bars.
    Used during live inference — requires at least 30 bars of history.

    Returns a pd.Series with feature values, or None if insufficient data.
    """
    if len(recent_bars) < 30:
        logger.debug(
            {"event": "insufficient_bars_for_live_features", "count": len(recent_bars)}
        )
        return None

    tick_df = recent_ticks if recent_ticks is not None else pd.DataFrame()
    try:
        feature_df = compute_features(
            (
                recent_bars
                if "timestamp" in recent_bars.columns
                else recent_bars.reset_index()
            ),
            expiry_seconds=60,
        )
        if feature_df.empty:
            return None
        feature_cols = get_feature_columns()
        available = [c for c in feature_cols if c in feature_df.columns]
        return feature_df.iloc[-1][available]
    except Exception as exc:
        logger.warning({"event": "live_feature_extraction_failed", "error": str(exc)})
        return None
