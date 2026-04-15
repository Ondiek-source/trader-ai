"""
test_properties.py -- Tests for the Config.quotex_symbols property.

Covers:
    Group 11 : Symbol translation, OTC override, length correctness
"""

import os
import pytest
from unittest.mock import patch


def test_quotex_symbols_auto_translation(minimal_valid_env):
    env = {**minimal_valid_env, "PAIRS": "EUR_USD,XAU_USD"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        cfg = load_config()
        symbols = cfg.quotex_symbols
        assert symbols["EUR_USD"] == "EURUSD_otc"
        assert symbols["XAU_USD"] == "XAUUSD_otc"


def test_quotex_symbols_uses_otc_override(minimal_valid_env):
    env = {
        **minimal_valid_env,
        "PAIRS": "EUR_USD,GBP_USD",
        "OTC_PAIRS": "EURUSD-OTC,GBPUSD-OTC",
    }
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        cfg = load_config()
        symbols = cfg.quotex_symbols
        assert symbols["EUR_USD"] == "EURUSD-OTC"
        assert symbols["GBP_USD"] == "GBPUSD-OTC"


def test_quotex_symbols_length_matches_pairs(minimal_valid_env):
    env = {**minimal_valid_env, "PAIRS": "EUR_USD,GBP_USD,USD_JPY"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        cfg = load_config()
        assert len(cfg.quotex_symbols) == 3
        assert set(cfg.quotex_symbols.keys()) == {"EUR_USD", "GBP_USD", "USD_JPY"}


def test_quotex_symbols_never_reached_on_mismatch(minimal_valid_env):
    env = {
        **minimal_valid_env,
        "PAIRS": "EUR_USD,GBP_USD",
        "OTC_PAIRS": "EURUSD_otc",
    }
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        with pytest.raises(ValueError, match="OTC_PAIRS length"):
            load_config()
