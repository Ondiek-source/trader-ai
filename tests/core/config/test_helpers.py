"""
test_helpers.py -- Unit tests for config.py helper functions.

Covers:
    Group 1 : _require()
    Group 2 : _parse_float()
    Group 3 : _bool() and _int()
    Group 4 : _parse_and_validate_pairs() and _parse_otc_pairs()
"""

import os
import logging
import pytest
from unittest.mock import patch


# ==============================================================================
# GROUP 1 -- _require()
# ==============================================================================

def test_require_returns_stripped_value(monkeypatch):
    monkeypatch.setenv("MY_VAR", "  hello_world  ")
    from src.core.config import _require
    assert _require("MY_VAR") == "hello_world"


def test_require_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    from src.core.config import _require
    with pytest.raises(ValueError, match="MISSING_VAR"):
        _require("MISSING_VAR")


def test_require_raises_when_empty(monkeypatch):
    monkeypatch.setenv("EMPTY_VAR", "")
    from src.core.config import _require
    with pytest.raises(ValueError, match="EMPTY_VAR"):
        _require("EMPTY_VAR")


def test_require_raises_when_whitespace_only(monkeypatch):
    monkeypatch.setenv("WHITESPACE_VAR", "   ")
    from src.core.config import _require
    with pytest.raises(ValueError, match="WHITESPACE_VAR"):
        _require("WHITESPACE_VAR")


# ==============================================================================
# GROUP 2 -- _parse_float()
# ==============================================================================

def test_parse_float_valid(monkeypatch):
    monkeypatch.setenv("MY_FLOAT", "0.75")
    from src.core.config import _parse_float
    assert _parse_float("MY_FLOAT") == 0.75


def test_parse_float_missing_returns_default(monkeypatch):
    monkeypatch.delenv("MY_FLOAT", raising=False)
    from src.core.config import _parse_float
    assert _parse_float("MY_FLOAT", default=1.5) == 1.5


def test_parse_float_missing_returns_none(monkeypatch):
    monkeypatch.delenv("MY_FLOAT", raising=False)
    from src.core.config import _parse_float
    assert _parse_float("MY_FLOAT") is None


def test_parse_float_invalid_raises(monkeypatch):
    monkeypatch.setenv("MY_FLOAT", "not_a_number")
    from src.core.config import _parse_float
    with pytest.raises(ValueError, match="MY_FLOAT"):
        _parse_float("MY_FLOAT")


def test_parse_float_uses_tilde_symbol(monkeypatch, caplog):
    monkeypatch.setenv("MY_FLOAT", "bad_value")
    from src.core.config import _parse_float
    with caplog.at_level(logging.CRITICAL, logger="src.core.config"):
        with pytest.raises(ValueError):
            _parse_float("MY_FLOAT")
    assert "~~~" in caplog.text
    assert "###" not in caplog.text


# ==============================================================================
# GROUP 3 -- _bool() and _int()
# ==============================================================================

@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on"])
def test_bool_truthy_variants(monkeypatch, value):
    monkeypatch.setenv("MY_BOOL", value)
    from src.core.config import _bool
    assert _bool("MY_BOOL", default=False) is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "maybe"])
def test_bool_falsy_variants(monkeypatch, value):
    monkeypatch.setenv("MY_BOOL", value)
    from src.core.config import _bool
    assert _bool("MY_BOOL", default=True) is False


def test_bool_absent_returns_default(monkeypatch):
    monkeypatch.delenv("MY_BOOL", raising=False)
    from src.core.config import _bool
    assert _bool("MY_BOOL", default=True) is True
    assert _bool("MY_BOOL", default=False) is False


def test_int_absent_returns_default(monkeypatch):
    monkeypatch.delenv("MY_INT", raising=False)
    from src.core.config import _int
    assert _int("MY_INT", default=42) == 42


def test_int_empty_returns_default(monkeypatch):
    monkeypatch.setenv("MY_INT", "")
    from src.core.config import _int
    assert _int("MY_INT", default=42) == 42


def test_int_invalid_raises(monkeypatch):
    monkeypatch.setenv("MY_INT", "3.14")
    from src.core.config import _int
    with pytest.raises(ValueError, match="MY_INT"):
        _int("MY_INT", default=0)


# ==============================================================================
# GROUP 4 -- Pair parsing helpers
# ==============================================================================

def test_parse_pairs_valid():
    from src.core.config import _parse_and_validate_pairs
    result = _parse_and_validate_pairs("PAIRS", "EUR_USD,GBP_USD")
    assert result == ["EUR_USD", "GBP_USD"]


def test_parse_pairs_case_insensitive():
    from src.core.config import _parse_and_validate_pairs
    result = _parse_and_validate_pairs("PAIRS", "eur_usd,gbp_usd")
    assert result == ["EUR_USD", "GBP_USD"]


def test_parse_pairs_rejects_invalid():
    from src.core.config import _parse_and_validate_pairs
    with pytest.raises(ValueError, match="BTC_USD"):
        _parse_and_validate_pairs("PAIRS", "EUR_USD,BTC_USD")


def test_parse_pairs_semicolon_delimiter():
    from src.core.config import _parse_and_validate_pairs
    result = _parse_and_validate_pairs("PAIRS", "EUR_USD;GBP_USD")
    assert result == ["EUR_USD", "GBP_USD"]


def test_parse_otc_pairs_delimiter_only_raises():
    from src.core.config import _parse_otc_pairs
    with pytest.raises(ValueError, match="no valid pairs"):
        _parse_otc_pairs(",,,")
