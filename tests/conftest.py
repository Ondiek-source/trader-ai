"""
conftest.py -- Shared pytest fixtures for all TRADER-AI tests.

Run pytest from the project root:
    pytest tests/ -v
"""

import pytest


@pytest.fixture
def minimal_valid_env():
    """
    The smallest set of env vars that passes all config.py validation.
    Use as a base in each test and override one key at a time to exercise
    a specific failure branch.
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
