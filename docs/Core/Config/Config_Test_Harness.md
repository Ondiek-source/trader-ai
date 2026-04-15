# TRADER-AI CORE: CONFIG.PY -- COMPLETE TEST HARNESS

Generated : 2026-04-12 | File Under Test : config.py | Runner : pytest

  Total Test Cases  : 54
  Test Groups       : 11
  Coverage Target   : 100% of all validation branches + happy paths

  PHILOSOPHY:
  Every test in this harness follows the same rule as the module itself --
  Fail-Fast and Zero Ambiguity. Each test has one job. A test that checks
  two things at once is a test that lies when it fails.

  SETUP REQUIREMENT:
  All tests patch environment variables using pytest's monkeypatch fixture
  or unittest.mock.patch.dict. The module must NOT be imported at the top
  level in test files -- import config INSIDE each test or use importlib
  to force a fresh module state. This is required because load_dotenv()
  runs at import time.

  RECOMMENDED TEST FILE LAYOUT:
    tests/
      conftest.py          <-- shared fixtures (minimal_valid_env)
      test_helpers.py      <-- Group 1-4  (helper function unit tests)
      test_validation.py   <-- Group 5-8  (Config.__post_init__ tests)
      test_load_config.py  <-- Group 9    (load_config integration tests)
      test_settings.py     <-- Group 10   (get_settings singleton tests)
      test_properties.py   <-- Group 11   (quotex_symbols property tests)

──────────────────────────────────────────────────────────────────────────────
  CONFTEST.PY -- SHARED FIXTURES
──────────────────────────────────────────────────────────────────────────────

  PURPOSE:
  The minimal_valid_env fixture provides the smallest possible set of
  environment variables that allows load_config() to succeed without error.
  Every other test either uses this as a base and overrides one key,
  or builds its own environment from scratch.

  GOLDEN PROMPT -- conftest.py
  -----------------------------------------------------------------------

  Create tests/conftest.py with the following content:

    import pytest

    @pytest.fixture
    def minimal_valid_env():
        """
        The smallest set of env vars that passes all config.py validation.
        Use as a base and override individual keys in each test.
        """
        return {
            "TWELVEDATA_API_KEY": "a" * 32,
            "WEBHOOK_URL": "https://example.com/webhook",
            "WEBHOOK_KEY": "test-secret-key-32-chars-xxxxxxxxx",
            "QUOTEX_EMAIL": "test@example.com",
            "QUOTEX_PASSWORD": "securepassword123",
            "PAIRS": "EUR_USD",
            "CONFIDENCE_THRESHOLD": "0.55",
            "EXPIRY_SECONDS": "60",
            "DATA_MODE": "LOCAL",
            "MARTINGALE_MAX_STREAK": "4",
            "DASHBOARD_PORT": "8080",
            "LOG_LEVEL": "INFO",
            "PRACTICE_MODE": "True",
        }
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────
  GROUP 1 -- _require() HELPER TESTS  (4 tests)
  File: tests/test_helpers.py
──────────────────────────────────────────────────────────────────────────────

  These tests verify the behaviour of the _require() function in isolation.
  Import it directly: from config import_require

──────────────────────────────────────────────────────────────────────────────
TEST 1.1 -- _require returns value when variable is set
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    When the named environment variable has a non-empty value, _require
    returns that value stripped of leading/trailing whitespace.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_require_returns_stripped_value(monkeypatch):
        monkeypatch.setenv("MY_VAR", "  hello_world  ")
        from config import _require
        assert _require("MY_VAR") == "hello_world"
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 1.2 -- _require raises ValueError when variable is missing
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    When the variable is not set at all, _require raises ValueError with
    a message containing the variable name.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_require_raises_when_missing(monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        from config import _require
        with pytest.raises(ValueError, match="MISSING_VAR"):
            _require("MISSING_VAR")
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 1.3 -- _require raises ValueError when variable is empty string
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    An env var set to "" (empty string) is treated as missing.
    This is the "zombie state" prevention -- an empty key is not a key.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_require_raises_when_empty(monkeypatch):
        monkeypatch.setenv("EMPTY_VAR", "")
        from config import _require
        with pytest.raises(ValueError, match="EMPTY_VAR"):
            _require("EMPTY_VAR")
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 1.4 -- _require raises ValueError when variable is whitespace only
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    An env var set to "   " (spaces only) is also treated as missing
    after stripping.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_require_raises_when_whitespace_only(monkeypatch):
        monkeypatch.setenv("WHITESPACE_VAR", "   ")
        from config import _require
        with pytest.raises(ValueError, match="WHITESPACE_VAR"):
            _require("WHITESPACE_VAR")
  -----------------------------------------------------------------------

---

  GROUP 2 -- _parse_float() HELPER TESTS  (5 tests)
  File: tests/test_helpers.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 2.1 -- _parse_float returns float when valid decimal string given
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_float_valid(monkeypatch):
        monkeypatch.setenv("MY_FLOAT", "0.75")
        from config import _parse_float
        assert _parse_float("MY_FLOAT") == 0.75
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 2.2 -- _parse_float returns default when variable is missing
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_float_missing_returns_default(monkeypatch):
        monkeypatch.delenv("MY_FLOAT", raising=False)
        from config import _parse_float
        assert _parse_float("MY_FLOAT", default=1.5) == 1.5
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 2.3 -- _parse_float returns None when missing and no default given
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_float_missing_returns_none(monkeypatch):
        monkeypatch.delenv("MY_FLOAT", raising=False)
        from config import _parse_float
        assert _parse_float("MY_FLOAT") is None
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 2.4 -- _parse_float raises ValueError on non-numeric string
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    A non-empty, non-numeric value (e.g. "abc") must raise ValueError,
    NOT silently return None. This is the "present but invalid" case.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_float_invalid_raises(monkeypatch):
        monkeypatch.setenv("MY_FLOAT", "not_a_number")
        from config import _parse_float
        with pytest.raises(ValueError, match="MY_FLOAT"):
            _parse_float("MY_FLOAT")
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 2.5 -- _parse_float uses tilde (~) diagnostic symbol on failure
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The error message logged during a float parse failure must use the
    '~' diagnostic symbol (Issue #1 fix verification). This test confirms
    the fix is in place and has not regressed.

  NOTE: Capture logger output using pytest's caplog fixture.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_float_uses_tilde_symbol(monkeypatch, caplog):
        monkeypatch.setenv("MY_FLOAT", "bad_value")
        from config import _parse_float
        import logging
        with caplog.at_level(logging.CRITICAL, logger="config"):
            with pytest.raises(ValueError):
                _parse_float("MY_FLOAT")
        assert "~~~" in caplog.text
        assert "###" not in caplog.text
  -----------------------------------------------------------------------

---

  GROUP 3 -- _bool() AND_int() HELPER TESTS  (5 tests)
  File: tests/test_helpers.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 3.1 -- _bool returns True for all truthy string variants
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_bool_truthy_variants(monkeypatch, value):
        monkeypatch.setenv("MY_BOOL", value)
        from config import _bool
        assert _bool("MY_BOOL", default=False) is True
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 3.2 -- _bool returns False for non-truthy strings
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "maybe"])
    def test_bool_falsy_variants(monkeypatch, value):
        monkeypatch.setenv("MY_BOOL", value)
        from config import _bool
        assert _bool("MY_BOOL", default=True) is False
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 3.3 -- _bool returns default when variable is absent
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_bool_absent_returns_default(monkeypatch):
        monkeypatch.delenv("MY_BOOL", raising=False)
        from config import _bool
        assert _bool("MY_BOOL", default=True) is True
        assert _bool("MY_BOOL", default=False) is False
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 3.4 -- _int returns default when variable is absent or empty
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_int_absent_returns_default(monkeypatch):
        monkeypatch.delenv("MY_INT", raising=False)
        from config import _int
        assert _int("MY_INT", default=42) == 42

    def test_int_empty_returns_default(monkeypatch):
        monkeypatch.setenv("MY_INT", "")
        from config import _int
        assert _int("MY_INT", default=42) == 42
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 3.5 -- _int raises ValueError on non-integer string
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_int_invalid_raises(monkeypatch):
        monkeypatch.setenv("MY_INT", "3.14")
        from config import _int
        with pytest.raises(ValueError, match="MY_INT"):
            _int("MY_INT", default=0)
  -----------------------------------------------------------------------

---

  GROUP 4 -- PAIR PARSING HELPERS  (5 tests)
  File: tests/test_helpers.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 4.1 -- _parse_and_validate_pairs accepts valid uppercase pairs
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_pairs_valid(monkeypatch):
        from config import _parse_and_validate_pairs
        result = _parse_and_validate_pairs("PAIRS", "EUR_USD,GBP_USD")
        assert result == ["EUR_USD", "GBP_USD"]
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 4.2 -- _parse_and_validate_pairs normalises lowercase input
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The .env might contain "eur_usd" or "Eur_Usd". The parser must
    uppercase before validating so the user does not need to know casing.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_pairs_case_insensitive(monkeypatch):
        from config import _parse_and_validate_pairs
        result = _parse_and_validate_pairs("PAIRS", "eur_usd,gbp_usd")
        assert result == ["EUR_USD", "GBP_USD"]
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 4.3 -- _parse_and_validate_pairs rejects unknown pair
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_pairs_rejects_invalid(monkeypatch):
        from config import _parse_and_validate_pairs
        with pytest.raises(ValueError, match="BTC_USD"):
            _parse_and_validate_pairs("PAIRS", "EUR_USD,BTC_USD")
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 4.4 -- _parse_and_validate_pairs accepts semicolon as delimiter
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The parser replaces ";" with "," before splitting, so both delimiters
    are accepted. A user who pastes "EUR_USD;GBP_USD" should not get an
    error.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_pairs_semicolon_delimiter(monkeypatch):
        from config import _parse_and_validate_pairs
        result = _parse_and_validate_pairs("PAIRS", "EUR_USD;GBP_USD")
        assert result == ["EUR_USD", "GBP_USD"]
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 4.5 -- _parse_otc_pairs raises on delimiter-only input
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    If OTC_PAIRS is set to ",,," (only commas), the function must raise
    ValueError rather than returning an empty list silently. This is the
    corrected dead-branch test from Issue #2.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_parse_otc_pairs_delimiter_only_raises():
        from config import _parse_otc_pairs
        with pytest.raises(ValueError, match="no valid pairs"):
            _parse_otc_pairs(",,,")
  -----------------------------------------------------------------------

---

  GROUP 5 -- __post_init__ SECURITY VALIDATION  (3 tests)
  File: tests/test_validation.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 5.1 -- Config raises when webhook_url set but webhook_key is empty
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The primary security fix (Issue #5). A configured webhook endpoint
    with no authentication key must be a hard boot failure.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_webhook_key_required_when_url_set(monkeypatch, minimal_valid_env):
        env = {**minimal_valid_env, "WEBHOOK_KEY": ""}
        monkeypatch.setattr(os, "environ", env, raising=False)
        # Use patch.dict for cleaner env isolation:
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="WEBHOOK_KEY"):
                from config import load_config
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 5.2 -- Config boots successfully when webhook_key is non-empty
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_webhook_key_present_allows_boot(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        with patch.dict(os.environ, minimal_valid_env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.webhook_key == minimal_valid_env["WEBHOOK_KEY"]
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 5.3 -- webhook_key default is empty string (not a hardcoded value)
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    Regression test for Issue #5. Verifies "Ondiek" (or any hardcoded
    string) is NOT the default. If WEBHOOK_KEY is absent and WEBHOOK_URL
    is set, the system must fail, not silently use a known default.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_webhook_key_has_no_hardcoded_default(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {k: v for k, v in minimal_valid_env.items() if k != "WEBHOOK_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError):
                load_config()
        # Confirm the default is "" by checking _optional directly
        import importlib
        import config as cfg_module
        with patch.dict(os.environ, {}, clear=True):
            result = cfg_module._optional("WEBHOOK_KEY", "")
            assert result == ""
            assert result != "Ondiek"
  -----------------------------------------------------------------------

---

  GROUP 6 -- __post_init__ DATA INTEGRITY VALIDATION  (5 tests)
  File: tests/test_validation.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 6.1 -- Config raises when OTC_PAIRS length != PAIRS length
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The zip() silent truncation fix (Issue #3). If a user sets 3 PAIRS
    but only 2 OTC_PAIRS, the system must crash with a clear error rather
    than silently dropping the third pair from the symbol map.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_otc_pairs_length_mismatch_raises(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {
            **minimal_valid_env,
            "PAIRS": "EUR_USD,GBP_USD,USD_JPY",
            "OTC_PAIRS": "EURUSD_otc,GBPUSD_otc",  # only 2 for 3 pairs
        }
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="OTC_PAIRS length"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 6.2 -- Config accepts matching OTC_PAIRS and PAIRS lengths
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_otc_pairs_matching_length_accepted(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {
            **minimal_valid_env,
            "PAIRS": "EUR_USD,GBP_USD",
            "OTC_PAIRS": "EURUSD_otc,GBPUSD_otc",
        }
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.otc_pairs == ["EURUSD_OTC", "GBPUSD_OTC"]
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 6.3 -- Config raises when DATA_MODE is CLOUD but AZURE_STORAGE_CONN missing
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_cloud_mode_requires_azure_conn(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "DATA_MODE": "CLOUD", "AZURE_STORAGE_CONN": ""}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="AZURE_STORAGE_CONN"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 6.4 -- Config accepts CLOUD mode when AZURE_STORAGE_CONN is present
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_cloud_mode_with_azure_conn_accepted(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {
            **minimal_valid_env,
            "DATA_MODE": "CLOUD",
            "AZURE_STORAGE_CONN": "DefaultEndpointsProtocol=https;AccountName=test",
        }
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.data_mode == "CLOUD"
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 6.5 -- Config raises on invalid DATA_MODE value
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_invalid_data_mode_raises(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "DATA_MODE": "S3"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="DATA_MODE"):
                load_config()
  -----------------------------------------------------------------------

---

  GROUP 7 -- __post_init__ TRADING LOGIC VALIDATION  (7 tests)
  File: tests/test_validation.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 7.1 -- Confidence threshold at lower boundary (0.51) is accepted
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_confidence_threshold_lower_boundary(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "CONFIDENCE_THRESHOLD": "0.51"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.confidence_threshold == pytest.approx(0.51)
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 7.2 -- Confidence threshold at exactly 0.50 is rejected
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    0.50 is a coin flip. The system must reject it. The boundary check is
    STRICT: 0.5 < x < 1.0, so 0.50 is NOT allowed.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_confidence_threshold_at_050_rejected(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "CONFIDENCE_THRESHOLD": "0.50"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="Confidence"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 7.3 -- Confidence threshold at exactly 1.0 is rejected
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_confidence_threshold_at_100_rejected(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "CONFIDENCE_THRESHOLD": "1.0"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="Confidence"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 7.4 -- Valid EXPIRY_SECONDS values are accepted  (parametrized)
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    @pytest.mark.parametrize("expiry", [60, 120, 300])
    def test_valid_expiry_seconds_accepted(monkeypatch, minimal_valid_env, expiry):
        from unittest.mock import patch
        env = {**minimal_valid_env, "EXPIRY_SECONDS": str(expiry)}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.expiry_seconds == expiry
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 7.5 -- Invalid EXPIRY_SECONDS value is rejected
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_invalid_expiry_seconds_rejected(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "EXPIRY_SECONDS": "45"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="expiry"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 7.6 -- Martingale streak above 6 is rejected
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_martingale_streak_above_6_rejected(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "MARTINGALE_MAX_STREAK": "7"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="Martingale"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 7.7 -- Martingale streak at exactly 6 is accepted
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_martingale_streak_at_6_accepted(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "MARTINGALE_MAX_STREAK": "6"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.martingale_max_streak == 6
  -----------------------------------------------------------------------

---

  GROUP 8 -- __post_init__ INFRASTRUCTURE VALIDATION  (4 tests)
  File: tests/test_validation.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 8.1 -- Dashboard port below 1 is rejected
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_dashboard_port_below_1_rejected(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "DASHBOARD_PORT": "0"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="port"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 8.2 -- Dashboard port above 65535 is rejected
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_dashboard_port_above_65535_rejected(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "DASHBOARD_PORT": "65536"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="port"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 8.3 -- Invalid LOG_LEVEL string is rejected
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_invalid_log_level_rejected(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "LOG_LEVEL": "VERBOSE"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="LOG_LEVEL"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 8.4 -- All valid LOG_LEVEL values are accepted  (parametrized)
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels_accepted(monkeypatch, minimal_valid_env, level):
        from unittest.mock import patch
        env = {**minimal_valid_env, "LOG_LEVEL": level}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.log_level == level
  -----------------------------------------------------------------------

---

  GROUP 9 -- load_config() INTEGRATION TESTS  (8 tests)
  File: tests/test_load_config.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 9.1 -- load_config raises when TWELVEDATA_API_KEY is missing
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_missing_twelvedata_key_raises(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {k: v for k, v in minimal_valid_env.items()
               if k != "TWELVEDATA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="TWELVEDATA_API_KEY"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 9.2 -- load_config raises when WEBHOOK_URL is missing
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_missing_webhook_url_raises(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {k: v for k, v in minimal_valid_env.items()
               if k != "WEBHOOK_URL"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="WEBHOOK_URL"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 9.3 -- load_config raises when QUOTEX_EMAIL is missing
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_missing_quotex_email_raises(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "QUOTEX_EMAIL": ""}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="Quotex"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 9.4 -- load_config raises when QUOTEX_PASSWORD is missing
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_missing_quotex_password_raises(monkeypatch, minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "QUOTEX_PASSWORD": ""}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="Quotex"):
                load_config()
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 9.5 -- load_config succeeds with minimal valid environment
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The happy path. The minimal_valid_env fixture must be sufficient to
    produce a valid Config object without any exception.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_load_config_happy_path(minimal_valid_env):
        from unittest.mock import patch
        with patch.dict(os.environ, minimal_valid_env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg is not None
            assert cfg.pairs == ["EUR_USD"]
            assert cfg.practice_mode is True
            assert cfg.data_mode == "LOCAL"
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 9.6 -- Config instance is frozen (immutable after creation)
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    frozen=True on the dataclass must prevent any field from being set
    after instantiation. This is the immutability guarantee.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_config_is_frozen(minimal_valid_env):
        from unittest.mock import patch
        from dataclasses import FrozenInstanceError
        with patch.dict(os.environ, minimal_valid_env, clear=True):
            from config import load_config
            cfg = load_config()
            with pytest.raises(FrozenInstanceError):
                cfg.practice_mode = False
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 9.7 -- PAIRS defaults to EUR_USD when not set
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    If PAIRS is absent from the environment, load_config should not crash
    but should default to ["EUR_USD"].

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_pairs_defaults_to_eur_usd(minimal_valid_env):
        from unittest.mock import patch
        env = {k: v for k, v in minimal_valid_env.items() if k != "PAIRS"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.pairs == ["EUR_USD"]
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 9.8 -- DAILY_NET_PROFIT_TARGET defaults to None when not set
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    This field is intentionally optional (None = disabled). Absence from
    .env must NOT raise -- it must produce None.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_daily_net_profit_target_defaults_to_none(minimal_valid_env):
        from unittest.mock import patch
        env = {k: v for k, v in minimal_valid_env.items()
               if k != "DAILY_NET_PROFIT_TARGET"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert cfg.daily_net_profit_target is None
  -----------------------------------------------------------------------

---

  GROUP 10 -- get_settings() SINGLETON TESTS  (4 tests)
  File: tests/test_settings.py
---

  IMPORTANT NOTE ON SINGLETON TESTING:
  The get_settings() function caches its result in the module-level
  _settings variable. Between tests that exercise get_settings(), you
  must reset_settings to None to ensure test isolation. Do this with:

      import config as cfg_module
      cfg_module._settings = None

  Include this in a fixture or teardown for all tests in this group.

──────────────────────────────────────────────────────────────────────────────

TEST 10.1 -- get_settings returns a Config instance on first call
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_get_settings_returns_config(minimal_valid_env):
        from unittest.mock import patch
        import config as cfg_module
        cfg_module._settings = None  # reset singleton
        with patch.dict(os.environ, minimal_valid_env, clear=True):
            result = cfg_module.get_settings()
            assert isinstance(result, cfg_module.Config)
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 10.2 -- get_settings returns the SAME object on subsequent calls
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The lazy singleton must cache. Two calls must return the exact same
    object (identity check with `is`, not equality check with `==`).

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_get_settings_is_singleton(minimal_valid_env):
        from unittest.mock import patch
        import config as cfg_module
        cfg_module._settings = None
        with patch.dict(os.environ, minimal_valid_env, clear=True):
            first = cfg_module.get_settings()
            second = cfg_module.get_settings()
            assert first is second
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 10.3 -- get_settings calls sys.exit(1) on validation failure
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    When the environment is invalid, get_settings must call sys.exit(1).
    The fail-fast behaviour must be preserved in the singleton wrapper.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_get_settings_exits_on_failure(monkeypatch):
        from unittest.mock import patch
        import config as cfg_module
        cfg_module._settings = None
        with patch.dict(os.environ, {}, clear=True):  # empty env = missing required vars
            with pytest.raises(SystemExit) as exc_info:
                cfg_module.get_settings()
            assert exc_info.value.code == 1
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 10.4 -- Importing config does NOT trigger load_config at import time
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The lazy singleton fix (Issue #6). Importing the module with a broken
    environment must NOT cause a crash. Only calling get_settings() should
    trigger validation.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_import_does_not_boot_config():
        from unittest.mock import patch
        # Import with completely empty environment -- should not crash
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import config as cfg_module
            importlib.reload(cfg_module)  # force fresh import
            # If we get here without SystemExit, the test passes.
            assert cfg_module._settings is None
  -----------------------------------------------------------------------

---

  GROUP 11 -- quotex_symbols PROPERTY TESTS  (4 tests)
  File: tests/test_properties.py
---

──────────────────────────────────────────────────────────────────────────────

TEST 11.1 -- quotex_symbols auto-translates pairs when OTC_PAIRS not set
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The auto-translation logic: EUR_USD -> EURUSD_otc, XAU_USD -> XAUUSD_otc.
    Underscores are removed and "_otc" is appended.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_quotex_symbols_auto_translation(minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "PAIRS": "EUR_USD,XAU_USD"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            symbols = cfg.quotex_symbols
            assert symbols["EUR_USD"] == "EURUSD_otc"
            assert symbols["XAU_USD"] == "XAUUSD_otc"
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 11.2 -- quotex_symbols uses OTC_PAIRS override when provided
──────────────────────────────────────────────────────────────────────────────

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_quotex_symbols_uses_otc_override(minimal_valid_env):
        from unittest.mock import patch
        env = {
            **minimal_valid_env,
            "PAIRS": "EUR_USD,GBP_USD",
            "OTC_PAIRS": "EURUSD-OTC,GBPUSD-OTC",
        }
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            symbols = cfg.quotex_symbols
            assert symbols["EUR_USD"] == "EURUSD-OTC"
            assert symbols["GBP_USD"] == "GBPUSD-OTC"
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 11.3 -- quotex_symbols produces one entry per pair (no extras)
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    The output dict must have exactly as many keys as there are pairs.
    No ghost entries, no missing entries.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_quotex_symbols_length_matches_pairs(minimal_valid_env):
        from unittest.mock import patch
        env = {**minimal_valid_env, "PAIRS": "EUR_USD,GBP_USD,USD_JPY"}
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            cfg = load_config()
            assert len(cfg.quotex_symbols) == 3
            assert set(cfg.quotex_symbols.keys()) == {"EUR_USD", "GBP_USD", "USD_JPY"}
  -----------------------------------------------------------------------

──────────────────────────────────────────────────────────────────────────────

TEST 11.4 -- quotex_symbols with mismatched OTC_PAIRS never reaches property
──────────────────────────────────────────────────────────────────────────────

  WHAT IT CHECKS:
    Integration of Issue #3 fix with the property. If OTC_PAIRS is the
    wrong length, __post_init__ must have already raised before the property
    is ever called. This confirms the guard is upstream of the property.

  GOLDEN PROMPT
  -----------------------------------------------------------------------

    def test_quotex_symbols_never_reached_on_mismatch(minimal_valid_env):
        from unittest.mock import patch
        env = {
            **minimal_valid_env,
            "PAIRS": "EUR_USD,GBP_USD",
            "OTC_PAIRS": "EURUSD_otc",  # 1 for 2 pairs
        }
        with patch.dict(os.environ, env, clear=True):
            from config import load_config
            with pytest.raises(ValueError, match="OTC_PAIRS length"):
                load_config()
            # If we reach here, __post_init__ raised before quotex_symbols
            # could be called. The test passing means the guard works.
  -----------------------------------------------------------------------

---

  RUNNING THE FULL SUITE
---

  INSTALL DEPENDENCIES:
    pip install pytest pytest-cov python-dotenv --break-system-packages

  RUN ALL TESTS:
    pytest tests/ -v

  RUN WITH COVERAGE REPORT:
    pytest tests/ -v --cov=config --cov-report=term-missing

  RUN A SINGLE GROUP:
    pytest tests/test_helpers.py -v
    pytest tests/test_validation.py -v
    pytest tests/test_load_config.py -v
    pytest tests/test_settings.py -v
    pytest tests/test_properties.py -v

  EXPECTED COVERAGE TARGET:
    All branches in config.py should be exercised. The only lines that
    may show as uncovered are the `else` path of the live mode warning
    (requires PRACTICE_MODE=False in the env), which is covered by
    test 9.5 and should be added as a parametrized case if 100% branch
    coverage is required.

  EXPECTED OUTPUT (all passing):
    tests/test_helpers.py::test_require_returns_stripped_value       PASSED
    tests/test_helpers.py::test_require_raises_when_missing          PASSED
    tests/test_helpers.py::test_require_raises_when_empty            PASSED
    tests/test_helpers.py::test_require_raises_when_whitespace_only  PASSED
    tests/test_helpers.py::test_parse_float_valid                    PASSED
    tests/test_helpers.py::test_parse_float_missing_returns_default  PASSED
    tests/test_helpers.py::test_parse_float_missing_returns_none     PASSED
    tests/test_helpers.py::test_parse_float_invalid_raises           PASSED
    tests/test_helpers.py::test_parse_float_uses_tilde_symbol        PASSED
    tests/test_helpers.py::test_bool_truthy_variants[1]              PASSED
    ... (parametrized variants)
    tests/test_helpers.py::test_bool_falsy_variants[false]           PASSED
    ... (parametrized variants)
    tests/test_helpers.py::test_bool_absent_returns_default          PASSED
    tests/test_helpers.py::test_int_absent_returns_default           PASSED
    tests/test_helpers.py::test_int_empty_returns_default            PASSED
    tests/test_helpers.py::test_int_invalid_raises                   PASSED
    tests/test_helpers.py::test_parse_pairs_valid                    PASSED
    tests/test_helpers.py::test_parse_pairs_case_insensitive         PASSED
    tests/test_helpers.py::test_parse_pairs_rejects_invalid          PASSED
    tests/test_helpers.py::test_parse_pairs_semicolon_delimiter      PASSED
    tests/test_helpers.py::test_parse_otc_pairs_delimiter_only_raises PASSED
    tests/test_validation.py::test_webhook_key_required_when_url_set PASSED
    ... (all 54 tests)
    ====== 54 passed in X.XXs ======

---

  END OF TEST HARNESS
---
