"""
tests/ml_engine/labeler/conftest.py

Shared fixtures for the Labeler and RewardCalculator test suite.
All fixtures are function-scoped (default) unless noted.
"""

import pytest
import pandas as pd

from src.ml_engine.labeler import Labeler, RewardCalculator


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_bars(n: int = 20, close_vals: list | None = None) -> pd.DataFrame:
    """
    Build a minimal OHLCV bar DataFrame with a DatetimeIndex.

    All five OHLCV columns are present so tests that downstream to the
    features pipeline don't fail on missing columns. ``close_vals``
    overrides the close column; defaults to flat 1.1000.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = close_vals if close_vals is not None else [1.1000] * n
    return pd.DataFrame(
        {
            "open": 1.1000,
            "high": 1.1010,
            "low": 1.0990,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


# ── Bar fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def bars_rising():
    """20 bars with strictly rising close: each future bar > current bar."""
    return make_bars(n=20, close_vals=[1.1000 + i * 0.0001 for i in range(20)])


@pytest.fixture
def bars_flat():
    """20 bars with identical close: future == current → all labels = 0."""
    return make_bars(n=20)


@pytest.fixture
def bars_falling():
    """20 bars with strictly falling close: future < current → all labels = 0."""
    return make_bars(n=20, close_vals=[1.1000 - i * 0.0001 for i in range(20)])


@pytest.fixture
def bars_4():
    """4-row bar DataFrame — too few for a 5-bar lookahead (5_MIN expiry)."""
    return make_bars(n=4)


@pytest.fixture
def empty_bars():
    """Empty DataFrame with the correct columns but no rows."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def bars_no_close():
    """Bar DataFrame missing the 'close' column."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1min")
    return pd.DataFrame({"open": 1.1, "volume": 1000.0}, index=idx)


# ── Labeler fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def labeler_1min():
    return Labeler(expiry_key="1_MIN")


@pytest.fixture
def labeler_5min():
    return Labeler(expiry_key="5_MIN")


@pytest.fixture
def labeler_15min():
    return Labeler(expiry_key="15_MIN")


@pytest.fixture
def labels_rising(labeler_1min, bars_rising):
    """19 labels from 20 rising bars under 1-MIN expiry — all 1."""
    return labeler_1min.compute_labels(bars_rising)


@pytest.fixture
def labels_flat(labeler_1min, bars_flat):
    """19 labels from 20 flat bars under 1-MIN expiry — all 0."""
    return labeler_1min.compute_labels(bars_flat)


# ── RewardCalculator fixtures ─────────────────────────────────────────────────


@pytest.fixture
def rc():
    """RewardCalculator with payout_ratio=0.85."""
    return RewardCalculator(payout_ratio=0.85)
