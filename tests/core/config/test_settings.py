"""
test_settings.py -- Tests for the get_settings() lazy singleton.

Covers:
    Group 10 : Singleton caching, fail-fast, import safety

IMPORTANT: Each test resets _settings = None to ensure isolation.
           Without this, singleton state bleeds between tests.
"""

import os
import sys
import importlib
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after every test in this file."""
    import src.core.config as cfg_module
    cfg_module._settings = None
    yield
    cfg_module._settings = None


def test_get_settings_returns_config(minimal_valid_env):
    import src.core.config as cfg_module
    with patch.dict(os.environ, minimal_valid_env, clear=True):
        result = cfg_module.get_settings()
        assert isinstance(result, cfg_module.Config)


def test_get_settings_is_singleton(minimal_valid_env):
    import src.core.config as cfg_module
    with patch.dict(os.environ, minimal_valid_env, clear=True):
        first = cfg_module.get_settings()
        second = cfg_module.get_settings()
        assert first is second


def test_get_settings_exits_on_failure():
    import src.core.config as cfg_module
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            cfg_module.get_settings()
        assert exc_info.value.code == 1


def test_import_does_not_boot_config():
    """
    Importing config with an empty environment must not crash.
    Only get_settings() should trigger validation.
    """
    with patch.dict(os.environ, {}, clear=True):
        import src.core.config as cfg_module
        importlib.reload(cfg_module)
        assert cfg_module._settings is None
