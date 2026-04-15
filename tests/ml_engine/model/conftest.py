"""
conftest.py -- Shared fixtures for ml_engine tests.

Provides pre-built valid Tick and Bar instances as both direct
fixtures and factory fixtures for field-level override tests.
"""

import pytest
from datetime import datetime, timezone
from typing import Any, cast
from src.ml_engine.model import Tick, Bar, Timeframe


VALID_TS_NAIVE = datetime(2024, 1, 15, 10, 30, 0)
VALID_TS_AWARE = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def valid_tick():
    """A known-good Tick that passes all validation."""
    return Tick(
        timestamp=VALID_TS_NAIVE,
        symbol="EUR_USD",
        bid=1.08500,
        ask=1.08520,
        source="TWELVE",
    )


@pytest.fixture
def valid_bar():
    """A known-good Bar that passes all validation."""
    return Bar(
        timestamp=VALID_TS_NAIVE,
        symbol="EUR_USD",
        open_price=1.0850,
        high=1.0865,
        low=1.0848,
        close=1.0860,
        volume=342,
    )


@pytest.fixture
def valid_tick_factory():
    """
    Returns a factory function for creating valid Ticks with
    optional field overrides. Uses explicit typing to satisfy Pylance.
    """

    def _factory(**overrides: Any) -> Tick:
        # Extract values with explicit types
        return Tick(
            timestamp=cast(datetime, overrides.get("timestamp", VALID_TS_NAIVE)),
            symbol=cast(str, overrides.get("symbol", "EUR_USD")),
            bid=cast(float, overrides.get("bid", 1.08500)),
            ask=cast(float, overrides.get("ask", 1.08520)),
            source=cast(str, overrides.get("source", "TWELVE")),
        )

    return _factory


@pytest.fixture
def valid_bar_factory():
    """
    Returns a factory function for creating valid Bars with
    optional field overrides.
    """

    def _factory(**overrides: Any):
        return Bar(
            timestamp=cast(datetime, overrides.get("timestamp", VALID_TS_NAIVE)),
            symbol=cast(str, overrides.get("symbol", "EUR_USD")),
            open_price=cast(float, overrides.get("open_price", 1.0850)),
            high=cast(float, overrides.get("high", 1.0865)),
            low=cast(float, overrides.get("low", 1.0848)),
            close=cast(float, overrides.get("close", 1.0860)),
            volume=cast(float, overrides.get("volume", 342.0)),
            is_complete=cast(bool, overrides.get("is_complete", True)),
            timeframe=cast(Timeframe, overrides.get("timeframe", Timeframe.M1)),
        )

    return _factory
