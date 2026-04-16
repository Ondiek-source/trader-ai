"""
conftest.py -- Shared fixtures for storage tests.

All I/O is wired to tmp_path -- the real data/ directory is never touched.
"""

import pytest
import threading
from datetime import datetime
from src.ml_engine.model import Tick, Bar, Timeframe


VALID_TS = datetime(2024, 1, 15, 10, 30, 0)
VALID_TS_2 = datetime(2024, 1, 15, 10, 31, 0)
VALID_TS_3 = datetime(2024, 1, 15, 10, 32, 0)


@pytest.fixture
def valid_tick():
    return Tick(
        timestamp=VALID_TS,
        symbol="EUR_USD",
        bid=1.08500,
        ask=1.08520,
        source="TWELVE",
    )


@pytest.fixture
def valid_tick_2():
    return Tick(
        timestamp=VALID_TS_2,
        symbol="EUR_USD",
        bid=1.08510,
        ask=1.08530,
        source="TWELVE",
    )


@pytest.fixture
def valid_tick_gbp():
    return Tick(
        timestamp=VALID_TS,
        symbol="GBP_USD",
        bid=1.26500,
        ask=1.26520,
        source="QUOTEX",
    )


@pytest.fixture
def valid_bar():
    return Bar(
        timestamp=VALID_TS,
        symbol="EUR_USD",
        open=1.0850,
        high=1.0865,
        low=1.0848,
        close=1.0860,
        volume=342,
    )


@pytest.fixture
def valid_bar_2():
    return Bar(
        timestamp=VALID_TS_2,
        symbol="EUR_USD",
        open=1.0860,
        high=1.0875,
        low=1.0858,
        close=1.0870,
        volume=280,
    )


@pytest.fixture
def incomplete_bar():
    return Bar(
        timestamp=VALID_TS,
        symbol="EUR_USD",
        open=1.0850,
        high=1.0865,
        low=1.0848,
        close=1.0860,
        volume=342,
        is_complete=False,
    )


@pytest.fixture
def storage(tmp_path, minimal_valid_env):
    """
    A Storage instance wired to tmp_path. Never touches the real data/ dir.
    Uses minimal_valid_env from root conftest.py for config bootstrapping.
    """
    import os
    from unittest.mock import patch
    from src.data.storage import Storage

    with patch.dict(os.environ, minimal_valid_env, clear=True):
        store = Storage.__new__(Storage)
        store._settings = None  # type: ignore
        store._lock = threading.Lock()
        store.root_dir = tmp_path / "data"
        store.raw_dir = tmp_path / "data" / "raw"
        store.processed_dir = tmp_path / "data" / "processed"
        store._provision_infrastructure()
        return store
