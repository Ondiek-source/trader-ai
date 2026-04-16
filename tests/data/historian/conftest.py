import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from src.data.historian import Historian, HistorianError
from src.ml_engine.model import Bar, Timeframe


# â”€â”€ Shared constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

VALID_SYMBOL = "EUR_USD"
VALID_API_SYMBOL = "EUR/USD"
NOW_UTC = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
LAST_BAR_TS = datetime(2026, 4, 12, 23, 59, 0)  # naive UTC


# â”€â”€ Settings mock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@pytest.fixture
def mock_settings():
    """Minimal mock of Config with historian-relevant fields."""
    settings = MagicMock()
    settings.twelvedata_api_key = "test_api_key_32chars_000000000000"
    settings.backfill_years = 2
    settings.backfill_pairs = ["EUR_USD", "GBP_USD"]
    return settings


# â”€â”€ Storage mock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@pytest.fixture
def mock_storage():
    """Mock Storage with get_bars and save_bar_batch pre-configured."""
    storage = MagicMock()
    storage.get_bars.return_value = None  # default: no existing data
    storage.save_bar_batch.return_value = True
    return storage


# â”€â”€ Historian under test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@pytest.fixture
def historian(mock_settings, mock_storage):
    """
    Historian instance with mocked settings and storage.

    Bypasses get_settings() and Storage() construction so tests
    do not require a real .env file or filesystem.
    """
    with patch("src.data.historian.get_settings", return_value=mock_settings), patch(
        "src.data.historian.Storage", return_value=mock_storage
    ):
        h = Historian()
    return h


# â”€â”€ API response factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def make_api_values(n: int = 3, symbol: str = VALID_SYMBOL) -> list[dict]:
    """
    Build a list of n valid Twelve Data bar dicts, newest-first.
    Timestamps descend from 2026-04-12 10:02:00 down by 1 minute.
    """
    base = datetime(2026, 4, 12, 10, 2, 0)
    return [
        {
            "datetime": (base - pd.Timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": "1.08500",
            "high": "1.08550",
            "low": "1.08480",
            "close": "1.08520",
            "volume": "342",
        }
        for i in range(n)
    ]


@pytest.fixture
def api_values():
    """Three valid bar dicts in newest-first order."""
    return make_api_values(3)


@pytest.fixture
def valid_bar():
    """A known-good Bar for use in _save_bars and batch tests."""
    return Bar(
        timestamp=datetime(2026, 4, 12, 10, 0, 0),
        symbol=VALID_SYMBOL,
        open=1.08500,
        high=1.08550,
        low=1.08480,
        close=1.08520,
        volume=342.0,
        is_complete=True,
        timeframe=Timeframe.M1,
    )
