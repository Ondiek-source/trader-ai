"""
config.py — Environment variable parsing and validated configuration.

All secrets (tokens, passwords, connection strings) are read exclusively from
environment variables. Nothing is hard-coded. Raises ValueError on startup if
any required variable is missing or invalid.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _require(name: str) -> str:
    """Return env var value or raise ValueError."""
    val = os.environ.get(name, "").strip()
    if not val:
        raise ValueError(f"Required environment variable '{name}' is not set or empty.")
    return val


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool(name: str, default: bool = True) -> bool:
    val = os.environ.get(name, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    val = os.environ.get(name, str(default)).strip()
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable '{name}' must be an integer, got: '{val}'")


def _float(name: str, default: float) -> float:
    val = os.environ.get(name, str(default)).strip()
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable '{name}' must be a float, got: '{val}'")


@dataclass
class Config:
    # ── OANDA ─────────────────────────────────────────────────────────────────
    oanda_token: str
    oanda_account_id: str
    oanda_env: str  # "practice" | "live"

    # ── Azure Storage ─────────────────────────────────────────────────────────
    azure_storage_conn: str
    container_name: str

    # ── Webhook ───────────────────────────────────────────────────────────────
    webhook_url: str
    webhook_secret: str  # may be empty string if not configured

    # ── Quotex ────────────────────────────────────────────────────────────────
    quotex_email: str
    quotex_password: str

    # ── Trading ───────────────────────────────────────────────────────────────
    pairs: list[str]
    confidence_threshold: float
    expiry_seconds: int
    daily_trade_target: int
    trading_window_hours: int
    target_net_profit: float | None  # None = disabled; use daily_trade_target only

    # ── Operational ───────────────────────────────────────────────────────────
    practice_mode: bool
    log_level: str
    tick_flush_size: int

    # ── Martingale ────────────────────────────────────────────────────────────
    martingale_max_streak: int  # reset after this many consecutive losses

    # ── Webhook ───────────────────────────────────────────────────────────────
    webhook_key: str  # static auth key sent in payload (e.g. "Ondiek")

    # ── Reporting / notifications ─────────────────────────────────────────────
    telegram_token: str       # Telegram Bot API token (empty = disabled)
    telegram_chat_id: str     # Telegram chat ID to send messages to
    discord_webhook_url: str  # Discord incoming webhook URL (empty = disabled)

    # ── Result callback / Quotex reader ──────────────────────────────────────
    result_callback_port: int  # port for the local HTTP result receiver
    quotex_read_results: bool  # True = use Quotex API to read trade results

    # ── Training ──────────────────────────────────────────────────────────────
    backfill_years: int         # years of Dukascopy history to download
    optimize_expiry: bool       # True = test 60/120/300s and pick best per pair

    # ── Derived / convenience ─────────────────────────────────────────────────
    oanda_environment: str = field(init=False)

    def __post_init__(self) -> None:
        # Validate OANDA env
        if self.oanda_env not in ("practice", "live"):
            raise ValueError(f"OANDA_ENV must be 'practice' or 'live', got: '{self.oanda_env}'")
        self.oanda_environment = "practice" if self.oanda_env == "practice" else "live"

        # Validate pairs
        valid_pairs = {"EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"}
        for p in self.pairs:
            if p not in valid_pairs:
                logger.warning("Pair '%s' is not in the default set %s — proceeding anyway.", p, valid_pairs)

        # Validate confidence threshold
        if not 0.5 < self.confidence_threshold < 1.0:
            raise ValueError(f"CONFIDENCE_THRESHOLD must be between 0.5 and 1.0, got: {self.confidence_threshold}")

        # Validate expiry
        if self.expiry_seconds not in (60, 120, 300):
            logger.warning(
                "EXPIRY_SECONDS=%d is not in the tested set [60, 120, 300]. Proceeding.", self.expiry_seconds
            )

        # Warn loudly on live mode
        if not self.practice_mode:
            logger.warning(
                "⚠️  [LIVE MODE ACTIVE] — System is configured for LIVE trading. "
                "Ensure you have reviewed all parameters and risk settings."
            )
        else:
            logger.info("[PRACTICE MODE] — All operations run against the OANDA practice environment.")


def load_config() -> Config:
    """Parse all environment variables and return a validated Config instance."""
    pairs_raw = _optional("PAIRS", "EUR_USD")  # start with EUR/USD only; add more when validated
    pairs = [p.strip().upper() for p in pairs_raw.split(",") if p.strip()]
    if not pairs:
        raise ValueError("PAIRS must contain at least one currency pair.")

    target_profit_raw = _optional("TARGET_NET_PROFIT", "")
    target_net_profit: float | None = None
    if target_profit_raw:
        try:
            target_net_profit = float(target_profit_raw)
        except ValueError:
            raise ValueError(f"TARGET_NET_PROFIT must be a float, got: '{target_profit_raw}'")

    return Config(
        # OANDA
        oanda_token=_require("OANDA_TOKEN"),
        oanda_account_id=_require("OANDA_ACCOUNT_ID"),
        oanda_env=_optional("OANDA_ENV", "practice").lower(),
        # Azure
        azure_storage_conn=_require("AZURE_STORAGE_CONN"),
        container_name=_optional("CONTAINER_NAME", "traderai"),
        # Webhook
        webhook_url=_require("WEBHOOK_URL"),
        webhook_secret=_optional("WEBHOOK_SECRET", ""),
        # Quotex (required for result reading; can be blank if QUOTEX_READ_RESULTS=false)
        quotex_email=_optional("QUOTEX_EMAIL", ""),
        quotex_password=_optional("QUOTEX_PASSWORD", ""),
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
        # Webhook key (static auth sent in payload)
        webhook_key=_optional("WEBHOOK_KEY", "Ondiek"),
        # Reporting / notifications
        telegram_token=_optional("TELEGRAM_TOKEN", ""),
        telegram_chat_id=_optional("TELEGRAM_CHAT_ID", ""),
        discord_webhook_url=_optional("DISCORD_WEBHOOK_URL", ""),
        # Result callback HTTP server + Quotex direct reading
        result_callback_port=_int("RESULT_CALLBACK_PORT", 8080),
        quotex_read_results=_bool("QUOTEX_READ_RESULTS", True),
        # Training parameters
        backfill_years=_int("BACKFILL_YEARS", 5),
        optimize_expiry=_bool("OPTIMIZE_EXPIRY", True),
    )
