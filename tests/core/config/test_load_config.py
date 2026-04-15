"""
test_load_config.py -- Integration tests for load_config().

Covers:
    Group 9 : Full boot path, required vars, defaults, immutability
"""

import os
import pytest
from unittest.mock import patch
from dataclasses import FrozenInstanceError


def test_missing_twelvedata_key_raises(minimal_valid_env):
    env = {k: v for k, v in minimal_valid_env.items() if k != "TWELVEDATA_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        with pytest.raises(ValueError, match="TWELVEDATA_API_KEY"):
            load_config()


def test_missing_webhook_url_raises(minimal_valid_env):
    env = {k: v for k, v in minimal_valid_env.items() if k != "WEBHOOK_URL"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        with pytest.raises(ValueError, match="WEBHOOK_URL"):
            load_config()


def test_missing_quotex_email_raises(minimal_valid_env):
    env = {**minimal_valid_env, "QUOTEX_EMAIL": ""}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        with pytest.raises(ValueError, match="Quotex"):
            load_config()


def test_missing_quotex_password_raises(minimal_valid_env):
    env = {**minimal_valid_env, "QUOTEX_PASSWORD": ""}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        with pytest.raises(ValueError, match="Quotex"):
            load_config()


def test_load_config_happy_path(minimal_valid_env):
    with patch.dict(os.environ, minimal_valid_env, clear=True):
        from src.core.config import load_config
        cfg = load_config()
        assert cfg is not None
        assert cfg.pairs == ["EUR_USD"]
        assert cfg.practice_mode is True
        assert cfg.data_mode == "LOCAL"


def test_config_is_frozen(minimal_valid_env):
    with patch.dict(os.environ, minimal_valid_env, clear=True):
        from src.core.config import load_config
        cfg = load_config()
        with pytest.raises(FrozenInstanceError):
            cfg.practice_mode = False


def test_pairs_defaults_to_eur_usd(minimal_valid_env):
    env = {k: v for k, v in minimal_valid_env.items() if k != "PAIRS"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        cfg = load_config()
        assert cfg.pairs == ["EUR_USD"]


def test_daily_net_profit_target_defaults_to_none(minimal_valid_env):
    env = {k: v for k, v in minimal_valid_env.items()
           if k != "DAILY_NET_PROFIT_TARGET"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config
        cfg = load_config()
        assert cfg.daily_net_profit_target is None
