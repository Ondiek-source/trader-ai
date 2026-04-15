"""
test_validation.py -- Tests for Config.__post_init__ cross-field validation.

Covers:
    Group 5 : Security  (webhook key)
    Group 6 : Data integrity  (OTC length, cloud mode)
    Group 7 : Trading logic  (thresholds, expiry, martingale)
    Group 8 : Infrastructure  (port range, log level)
"""

import os
import pytest
from unittest.mock import patch


# ==============================================================================
# GROUP 5 -- Security: webhook key enforcement
# ==============================================================================


def test_webhook_key_required_when_url_set(minimal_valid_env):
    env = {**minimal_valid_env, "WEBHOOK_KEY": ""}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="WEBHOOK_KEY"):
            load_config()


def test_webhook_key_present_allows_boot(minimal_valid_env):
    with patch.dict(os.environ, minimal_valid_env, clear=True):
        from src.core.config import load_config

        cfg = load_config()
        assert cfg.webhook_key == minimal_valid_env["WEBHOOK_KEY"]


def test_webhook_key_has_no_hardcoded_default(minimal_valid_env):
    env = {k: v for k, v in minimal_valid_env.items() if k != "WEBHOOK_KEY"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError):
            load_config()
    import src.core.config as cfg_module

    with patch.dict(os.environ, {}, clear=True):
        result = cfg_module._optional("WEBHOOK_KEY", "")
        assert result == ""
        assert result != "Ondiek"


# ==============================================================================
# GROUP 6 -- Data integrity
# ==============================================================================


def test_otc_pairs_length_mismatch_raises(minimal_valid_env):
    env = {
        **minimal_valid_env,
        "PAIRS": "EUR_USD,GBP_USD,USD_JPY",
        "OTC_PAIRS": "EURUSD_otc,GBPUSD_otc",
    }
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="OTC_PAIRS length"):
            load_config()


def test_otc_pairs_matching_length_accepted(minimal_valid_env):
    env = {
        **minimal_valid_env,
        "PAIRS": "EUR_USD,GBP_USD",
        "OTC_PAIRS": "EURUSD_otc,GBPUSD_otc",
    }
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        cfg = load_config()
        assert len(cfg.otc_pairs) == 2


def test_cloud_mode_requires_azure_conn(minimal_valid_env):
    env = {**minimal_valid_env, "DATA_MODE": "CLOUD", "AZURE_STORAGE_CONN": ""}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="AZURE_STORAGE_CONN"):
            load_config()


def test_cloud_mode_with_azure_conn_accepted(minimal_valid_env):
    env = {
        **minimal_valid_env,
        "DATA_MODE": "CLOUD",
        "AZURE_STORAGE_CONN": "DefaultEndpointsProtocol=https;AccountName=test",
    }
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        cfg = load_config()
        assert cfg.data_mode == "CLOUD"


def test_invalid_data_mode_raises(minimal_valid_env):
    env = {**minimal_valid_env, "DATA_MODE": "S3"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="DATA_MODE"):
            load_config()


def test_practice_mode_logs_info(minimal_valid_env, caplog):
    import logging

    with patch.dict(os.environ, minimal_valid_env, clear=True):
        from src.core.config import load_config

        with caplog.at_level(logging.INFO):
            load_config()
    assert "PRACTICE MODE" in caplog.text


def test_parse_pairs_empty_returns_default():
    from src.core.config import _parse_and_validate_pairs

    result = _parse_and_validate_pairs("PAIRS", "")
    assert result == ["EUR_USD"]


def test_live_mode_logs_warning(minimal_valid_env, caplog):
    import logging

    env = {**minimal_valid_env, "PRACTICE_MODE": "False"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with caplog.at_level(logging.WARNING):
            load_config()
    assert "LIVE MODE" in caplog.text


def test_tick_flush_size_zero_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "TICK_FLUSH_SIZE": "0"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="TICK_FLUSH_SIZE"):
            load_config()


def test_tick_flush_size_negative_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "TICK_FLUSH_SIZE": "-10"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="TICK_FLUSH_SIZE"):
            load_config()


# ==============================================================================
# GROUP 7 -- Trading logic
# ==============================================================================


def test_confidence_threshold_lower_boundary(minimal_valid_env):
    env = {**minimal_valid_env, "CONFIDENCE_THRESHOLD": "0.51"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        cfg = load_config()
        assert cfg.confidence_threshold == pytest.approx(0.51)


def test_confidence_threshold_at_050_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "CONFIDENCE_THRESHOLD": "0.50"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="Confidence"):
            load_config()


def test_confidence_threshold_at_100_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "CONFIDENCE_THRESHOLD": "1.0"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="Confidence"):
            load_config()


@pytest.mark.parametrize("expiry", [60, 120, 300])
def test_valid_expiry_seconds_accepted(minimal_valid_env, expiry):
    env = {**minimal_valid_env, "EXPIRY_SECONDS": str(expiry)}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        cfg = load_config()
        assert cfg.expiry_seconds == expiry


def test_invalid_expiry_seconds_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "EXPIRY_SECONDS": "45"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="expiry"):
            load_config()


def test_martingale_streak_above_6_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "MARTINGALE_MAX_STREAK": "7"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="Martingale"):
            load_config()


def test_martingale_streak_at_6_accepted(minimal_valid_env):
    env = {**minimal_valid_env, "MARTINGALE_MAX_STREAK": "6"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        cfg = load_config()
        assert cfg.martingale_max_streak == 6


# ==============================================================================
# GROUP 8 -- Infrastructure
# ==============================================================================


def test_dashboard_port_below_1_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "DASHBOARD_PORT": "0"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="port"):
            load_config()


def test_dashboard_port_above_65535_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "DASHBOARD_PORT": "65536"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="port"):
            load_config()


def test_invalid_log_level_rejected(minimal_valid_env):
    env = {**minimal_valid_env, "LOG_LEVEL": "VERBOSE"}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        with pytest.raises(ValueError, match="LOG_LEVEL"):
            load_config()


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_valid_log_levels_accepted(minimal_valid_env, level):
    env = {**minimal_valid_env, "LOG_LEVEL": level}
    with patch.dict(os.environ, env, clear=True):
        from src.core.config import load_config

        cfg = load_config()
        assert cfg.log_level == level
