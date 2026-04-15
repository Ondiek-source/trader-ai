"""
tests/ml_engine/features/conftest.py

Shared fixtures for the FeatureEngineer test suite.
All fixtures are function-scoped (default) unless noted.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.ml_engine.features import FeatureEngineer


# ── Data Factories ────────────────────────────────────────────────────────────


def make_ohlcv(
    n: int = 100,
    start: str = "2024-01-01",
    freq: str = "1min",
    open_: float = 1.1000,
    high_delta: float = 0.0010,
    low_delta: float = 0.0010,
    close_delta: float = 0.0002,
    volume: float = 1000.0,
) -> pd.DataFrame:
    """
    Build a synthetic M1 OHLCV DataFrame with a DatetimeIndex.

    All prices are offset from ``open_`` by fixed deltas so the market is
    perfectly flat. Volume is constant so RVOL-based tests are predictable.
    Use ``n >= 100`` for tests that require all rolling-window warmup to
    be consumed (MACD requires 26 bars; derived MAs require 20 more).
    """
    idx = pd.date_range(start=start, periods=n, freq=freq)
    return pd.DataFrame(
        {
            "open": open_,
            "high": open_ + high_delta,
            "low": open_ - low_delta,
            "close": open_ + close_delta,
            "volume": volume,
        },
        index=idx,
    )


def make_ticks(
    n: int = 600,
    start: str = "2024-01-01",
    freq: str = "100ms",
    bid: float = 1.0999,
    ask: float = 1.1001,
) -> pd.DataFrame:
    """Build a synthetic tick DataFrame with bid/ask columns."""
    idx = pd.date_range(start=start, periods=n, freq=freq)
    return pd.DataFrame({"bid": bid, "ask": ask}, index=idx)


# ── Settings Mock ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_settings():
    """
    Settings mock with all feature flags enabled and
    gate thresholds matching test expectations.
    """
    s = MagicMock()
    s.feat_price_action_enabled = True
    s.feat_momentum_enabled = True
    s.feat_volatility_enabled = True
    s.feat_micro_enabled = True
    s.feat_context_enabled = True
    s.gate_min_rvol = 1.5
    s.gate_max_spread = 0.0005
    return s


# ── Engineer Fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def engineer(mock_settings, monkeypatch):
    """
    FeatureEngineer with mocked settings.

    Monkeypatches get_settings() at the module level so the real config
    file is never read and all feature flags are on by default.
    """
    import src.ml_engine.features as feat_mod

    monkeypatch.setattr(feat_mod, "get_settings", lambda: mock_settings)
    return FeatureEngineer()


# ── Bar / Tick Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def bars_100():
    """100-row M1 OHLCV DataFrame — enough for all rolling windows."""
    return make_ohlcv(n=100)


@pytest.fixture
def bars_30():
    """30-row M1 OHLCV DataFrame — marginal warmup coverage."""
    return make_ohlcv(n=30)


@pytest.fixture
def bars_5():
    """5-row M1 OHLCV DataFrame — too few for any rolling window."""
    return make_ohlcv(n=5)


@pytest.fixture
def ticks_600():
    """600 ticks at 100 ms intervals — covers the first bar window."""
    return make_ticks(n=600)


@pytest.fixture
def empty_bars():
    """Empty OHLCV DataFrame with correct column schema."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def full_feature_df(engineer, bars_100, ticks_600):
    """
    Full feature matrix produced by transform() with all groups enabled.
    Used across multiple test groups to avoid duplicating the transform call.
    """
    return engineer.transform(bars_100, ticks_600)
