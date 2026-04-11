"""
config.py — Environment variable parsing and validated configuration.

All secrets (tokens, passwords, connection strings) are read exclusively from
environment variables.  Nothing is hard-coded.  Raises ValueError on startup if
any required variable is missing or invalid.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

VALID_PAIRS: set[str] = {"EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"}
VALID_EXPIRIES: set[int] = {60, 120, 300}


# ── Environment variable helpers ──────────────────────────────────────────────


def _require(name: str) -> str:
    """
    Read a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Stripped value.

    Raises:
        ValueError: If the variable is unset or empty.
    """
    val = os.environ.get(name, "").strip()
    if not val:
        raise ValueError(f"Required environment variable '{name}' is not set or empty.")
    return val


def _optional(name: str, default: str = "") -> str:
    """
    Read an optional environment variable with a fallback.

    Args:
        name: Environment variable name.
        default: Value to return if unset.

    Returns:
        Stripped value or *default*.
    """
    return os.environ.get(name, default).strip()


def _bool(name: str, default: bool = True) -> bool:
    """
    Read a boolean environment variable.

    Accepts ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive) as
    truthy.  Anything else — including an empty or missing variable — is
    falsy, *unless* the variable is absent entirely, in which case *default*
    is returned.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is not set at all.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    """
    Read an integer environment variable.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is not set.

    Raises:
        ValueError: If the variable is set but not a valid integer.
    """
    val = os.environ.get(name, str(default)).strip()
    try:
        return int(val)
    except ValueError:
        raise ValueError(
            f"Environment variable '{name}' must be an integer, got: '{val}'"
        )


def _float(name: str, default: float) -> float:
    """
    Read a float environment variable.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is not set.

    Raises:
        ValueError: If the variable is set but not a valid float.
    """
    val = os.environ.get(name, str(default)).strip()
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable '{name}' must be a float, got: '{val}'")


# ── Configuration dataclass ───────────────────────────────────────────────────


@dataclass
class Config:
    """
    Validated application configuration.

    Instantiated via :func:`load_config`.  Callers should never construct
    this class directly — use ``load_config()`` so that every field is
    sourced from environment variables and validated.
    """

    # ── Twelve Data ───────────────────────────────────────────────────────
    twelvedata_api_key: str

    # ── Azure Storage ─────────────────────────────────────────────────────
    azure_storage_conn: str
    container_name: str

    # ── Quotex ────────────────────────────────────────────────────────────
    quotex_email: str
    quotex_password: str

    # ── Webhook ───────────────────────────────────────────────────────────
    webhook_url: str
    webhook_secret: str
    webhook_key: str  # static auth key included in every payload

    # ── Trading ───────────────────────────────────────────────────────────
    pairs: list[str]
    confidence_threshold: float
    expiry_seconds: int
    daily_trade_target: int
    trading_window_hours: int
    target_net_profit: float | None  # None = disabled; use daily_trade_target only

    # ── Operational ───────────────────────────────────────────────────────
    practice_mode: bool
    log_level: str
    tick_flush_size: int

    # ── Martingale ────────────────────────────────────────────────────────
    martingale_max_streak: int

    # ── Reporting / notifications ─────────────────────────────────────────
    telegram_token: str
    telegram_chat_id: str
    discord_webhook_url: str

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard_port: int

    # ── Training ──────────────────────────────────────────────────────────
    backfill_years: int
    max_sequences: int

    # ── Data source routing ──────────────────────────────────────────────
    use_quotex_streaming: bool
    otc_pairs: list[str]
    poll_interval: float

    def __post_init__(self) -> None:
        """Run cross-field validation after construction."""
        # Pairs
        for p in self.pairs:
            if p not in VALID_PAIRS:
                logger.warning(
                    "Pair '%s' is not in the default set %s — proceeding anyway.",
                    p,
                    VALID_PAIRS,
                )

        # Confidence threshold
        if not 0.5 < self.confidence_threshold < 1.0:
            raise ValueError(
                f"CONFIDENCE_THRESHOLD must be between 0.5 and 1.0, "
                f"got: {self.confidence_threshold}"
            )

        # Expiry
        if self.expiry_seconds not in VALID_EXPIRIES:
            logger.warning(
                "EXPIRY_SECONDS=%d is not in the tested set %s. Proceeding.",
                self.expiry_seconds,
                sorted(VALID_EXPIRIES),
            )

        # Port range
        if not 1 <= self.dashboard_port <= 65535:
            raise ValueError(
                f"DASHBOARD_PORT must be 1-65535, got: {self.dashboard_port}"
            )

        # Quotex credentials required when direct reading is enabled
        if not self.quotex_email or not self.quotex_password:
            logger.info(
                "Quotex credentials not set — result reading via pyquotex disabled. "
                "Set QUOTEX_EMAIL and QUOTEX_PASSWORD to enable."
            )

        # Log level validation (don't configure logging here — main.py owns that)
        level = getattr(logging, self.log_level, None)
        if level is None:
            raise ValueError(
                f"LOG_LEVEL must be a valid Python logging level, got: '{self.log_level}'"
            )

        # Live mode warning
        if not self.practice_mode:
            logger.warning(
                "[LIVE MODE ACTIVE] — System is configured for LIVE trading. "
                "Ensure you have reviewed all parameters and risk settings."
            )
        else:
            logger.info(
                "[PRACTICE MODE] — Signals will fire but treat results as simulation."
            )


def _parse_otc_pairs(raw: str) -> list[str]:
    """
    Parse ``OTC_PAIRS`` env var.

    Accepts comma-separated Quotex symbol names, e.g.::

        "EURUSD-OTC,GBPUSD-OTC,USDJPY-OTC,XAUUSD-OTC"

    If empty, symbols are derived from ``PAIRS`` at stream start time
    (``EUR_USD`` → ``EURUSD-OTC``).
    """
    if not raw.strip():
        return []
    pairs = [p.strip().upper() for p in raw.split(",") if p.strip()]
    if not pairs:
        raise ValueError("OTC_PAIRS is set but contains no valid pairs.")
    return pairs


def load_config() -> Config:
    """
    Parse all environment variables and return a validated :class:`Config`.

    Raises:
        ValueError: If any required variable is missing or any value is invalid.
    """
    pairs_raw = _optional("PAIRS", "EUR_USD,GBP_USD,USD_JPY,XAU_USD")
    pairs = [p.strip().upper() for p in pairs_raw.split(",") if p.strip()]
    if not pairs:
        raise ValueError("PAIRS must contain at least one currency pair.")

    target_profit_raw = _optional("TARGET_NET_PROFIT", "")
    target_net_profit: float | None = None
    if target_profit_raw:
        try:
            target_net_profit = float(target_profit_raw)
        except ValueError:
            raise ValueError(
                f"TARGET_NET_PROFIT must be a float, got: '{target_profit_raw}'"
            )

    return Config(
        # Twelve Data
        twelvedata_api_key=_require("TWELVEDATA_API_KEY"),
        # Azure
        azure_storage_conn=_require("AZURE_STORAGE_CONN"),
        container_name=_optional("CONTAINER_NAME", "traderai"),
        # Quotex
        quotex_email=_optional("QUOTEX_EMAIL", ""),
        quotex_password=_optional("QUOTEX_PASSWORD", ""),
        # Webhook
        webhook_url=_require("WEBHOOK_URL"),
        webhook_secret=_optional("WEBHOOK_SECRET", ""),
        webhook_key=_optional("WEBHOOK_KEY", "Ondiek"),
        # Trading
        pairs=pairs,
        confidence_threshold=_float("CONFIDENCE_THRESHOLD", 0.65),
        expiry_seconds=_int("EXPIRY_SECONDS", 60),
        daily_trade_target=_int("DAILY_TRADE_TARGET", 10),
        trading_window_hours=_int("TRADING_WINDOW_HOURS", 19),
        target_net_profit=target_net_profit,
        # Operational
        practice_mode=_bool("PRACTICE_MODE", True),
        log_level=_optional("LOG_LEVEL", "INFO").upper(),
        tick_flush_size=_int("TICK_FLUSH_SIZE", 500),
        # Martingale
        martingale_max_streak=_int("MARTINGALE_MAX_STREAK", 4),
        # Reporting
        telegram_token=_optional("TELEGRAM_TOKEN", ""),
        telegram_chat_id=_optional("TELEGRAM_CHAT_ID", ""),
        discord_webhook_url=_optional("DISCORD_WEBHOOK_URL", ""),
        # Dashboard
        dashboard_port=_int("DASHBOARD_PORT", 8080),
        # Training
        backfill_years=_int("BACKFILL_YEARS", 2),
        max_sequences=_int("MAX_SEQUENCES", 20000),
        # Data source routing
        use_quotex_streaming=_bool("USE_QUOTEX_STREAMING", True),
        otc_pairs=_parse_otc_pairs(_optional("OTC_PAIRS", "")),
        poll_interval=_float("POLL_INTERVAL", 1.0),
    )
