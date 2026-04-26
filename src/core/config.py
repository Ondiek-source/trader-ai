"""
core/config.py — Environment variable parsing and validated configuration.

All secrets (tokens, passwords, connection strings) are read exclusively from
environment variables.  Nothing is hard-coded.  Raises ValueError on startup if
any required variable is missing or invalid.
"""

from __future__ import annotations


import os
import sys
import logging

from datetime import datetime, timezone, timedelta
from typing import overload
from dotenv import load_dotenv
from dataclasses import dataclass, field
from core.exceptions import (
    ConfigExpiryError,
    ConfigLengthMismatchError,
    ConfigMissingError,
    ConfigInvalidValueError,
    ConfigPairError,
    ConfigPortError,
    ConfigTypeError,
    ConfigValidationError,
    ConfigRangeError,
    ConfigRequiredError,
    ConfigCeilingError,
)


# Load .env file for local development
load_dotenv()

logger = logging.getLogger(__name__)

# ── Global Diagnostic Snapshot ──────────────────────────────────────────────
# to provide context for any failures during environment variable parsing
CWD = os.getcwd()
ENV_FILE_EXISTS = os.path.exists(".env")
SYSTEM_USER = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
PYTHON_VER = sys.version.split()[0]

# ── Validation Constants ─────────────────────────────────────────────────────

VALID_PAIRS: set[str] = {
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
    "USD_CHF",
    "XAU_USD",
}

VALID_OTC_BASE_PAIRS: set[str] = {
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "XAUUSD",
}

VALID_EXPIRIES: set[int] = {60, 300, 900}
VALID_SOURCES: set[str] = {"TWELVE", "QUOTEX"}

# ── Value truncation helper ───────────────────────────────────────────────────


def _trunc(val: object, max_len: int = 120) -> str:
    """
    Truncate *val* to *max_len* characters for safe inclusion in log messages.

    Prevents oversized log lines when large values (Azure connection strings,
    certificates, API keys) are echoed back in error blocks. The truncation
    marker shows how many characters were omitted so the reader knows the
    value was cut.

    Args:
        val:     Any value; converted to str via str().
        max_len: Maximum number of characters to include. Default 120 keeps
                    error blocks scannable while covering most legitimate values.

    Returns:
        The full string if len <= max_len, otherwise the first max_len
        characters followed by an ellipsis and the omitted character count.

    Example:
        >>> _trunc("x" * 200)
        'xxxxxxx...xxxxxxx…(+80 chars)'
    """
    s = str(val)
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"…(+{len(s) - max_len} chars)"


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
    val: str = os.environ.get(name, "").strip()
    if not val:
        raise ConfigMissingError(field_name=name)
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


# TYPE NARROWING: When a float default is given, return is guaranteed
# float. When no default (or None) is given, return may be None.
# These stubs are for static type checkers (mypy/pyright) only and
# have no runtime effect.
@overload
def _parse_float(name: str, default: float) -> float: ...


@overload
def _parse_float(name: str, default: None = None) -> float | None: ...


def _parse_float(name: str, default: float | None = None) -> float | None:
    """
    Read an optional float.
    Returns None if missing/empty.
    Crashes with diagnostics if data is present but invalid.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:

        raise ConfigTypeError(name, _trunc(raw.strip()), "float")


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

    Returns:
        ``True`` if the variable is set to a truthy value, ``False`` if set
        to any other non-empty value, or *default* if absent.
    """
    raw: str | None = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    """
    Read an integer environment variable.

    If the variable is absent **or** set to an empty string, *default* is
    returned.  This mirrors the behaviour of :func:`_bool` which treats
    empty strings as falsy rather than raising.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is not set or empty.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the variable is set to a non-empty, non-integer value.
    """
    raw: str | None = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ConfigTypeError(name, raw.strip(), "int")


# ── List Parsing Helpers  ─────────────────────────────────────────────────────


def _parse_and_validate_pairs(name: str, raw: str) -> list[str]:
    """
    Parse currency pairs and validate them against a set of supported pairs.

    Args:
        name (str): The name of the environment variable.
        raw (str): The raw string containing comma-separated pair names.

    Returns:
        list[str]: A list of validated currency pairs.

    Raises:
        ValueError: If any pair is not in the set of supported pairs.
    """
    if not raw.strip():
        return ["EUR_USD"]

    pairs = [p.strip().upper() for p in raw.replace(";", ",").split(",") if p.strip()]

    for p in pairs:
        if p not in VALID_PAIRS:
            raise ConfigPairError(name, p, list(VALID_PAIRS))

    return pairs


def _parse_otc_pairs(raw: str) -> list[str]:
    """
    Parse and validated ``OTC_PAIRS`` from env var.

    Process:
        1. Split by comma, strip whitespace
        2. Remove any suffix (_OTC, _otc, -OTC, etc.) to get base pair
        3. Convert base pair to UPPERCASE for validation
        4. Validate base pair against allowed list
        5. Reconstruct with CORRECT format: basepair_otc (lowercase)

    Accepts valid comma-separated Quotex otc symbol names, e.g.::

        "EURUSD_otc,GBPUSD_otc,USDJPY_otc,XAUUSD_otc"

    If empty, symbols are derived from ``PAIRS`` at stream start time
    (``EUR_USD`` → ``EURUSD_otc``).

    Args:
        raw: Raw comma-separated string from the environment variable.

    Returns:
        List of validated OTC symbol names, or empty list if unset.
    """
    if not raw.strip():
        return []

    # Split and clean
    raw_symbols = [p.strip().replace("_", "") for p in raw.split(",") if p.strip()]
    if not raw_symbols:
        raise ConfigValidationError(
            field_name="OTC_PAIRS",
            value=raw,
            reason=(
                "String contains no valid symbols (only delimiters?),\n"
                "OTC_PAIRS must contain at least one symbol name.\n"
                "Example: OTC_PAIRS=EURUSD_otc,GBPUSD_otc"
            ),
        )

    validated_pairs = []
    for raw_symbol in raw_symbols:
        # Step 1: Remove any suffix to get base pair
        base_pair = raw_symbol.upper()  # Start with uppercase for processing

        # Remove common suffixes
        for suffix in ["OTC", "otc", "OTC", "otc", "Otc", "oTC"]:
            if base_pair.endswith(suffix):
                base_pair = base_pair[: -len(suffix)]
                break

        # Step 2: Validate base pair
        if base_pair not in VALID_OTC_BASE_PAIRS:
            raise ConfigPairError(
                field_name="OTC_PAIRS",
                pair=raw_symbol,
                valid_pairs=sorted(VALID_OTC_BASE_PAIRS),
            )

        # Step 3: Reconstruct with CORRECT format (uppercase _otc)
        correct_symbol = f"{base_pair.upper()}_otc"
        validated_pairs.append(correct_symbol)

    # NOTE: do not log here — _parse_otc_pairs is called from load_config()
    # before logging is configured. Normalization count is visible in the
    # validated pair list returned to the caller.
    return validated_pairs


# ── Configuration dataclass ───────────────────────────────────────────────────


@dataclass(frozen=True)
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
    daily_net_profit_target: (
        float | None
    )  # None = disabled; use daily_trade_target only
    trading_window_hours: int

    # ── Operational ───────────────────────────────────────────────────────
    practice_mode: bool
    log_level: str
    log_verbose_events: str  # comma-separated list of events to log (empty = minimal)
    tick_flush_size: int
    martingale_max_streak: int
    martingale_step: float  # per-loss confidence-threshold increment

    # ── Reporting / notifications ─────────────────────────────────────────
    telegram_token: str
    telegram_chat_id: str
    discord_webhook_url: str

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard_port: int
    display_tz: int  # UTC offset in hours applied to all display timestamps

    # ── ML / Training ──────────────────────────────────────────────────────────
    backfill_source: str  # "TWELVE" or "QUOTEX"
    backfill_years: int
    backfill_pairs: list[str]
    max_sequences: int
    max_rf_rows: int
    memory_saver_mode: bool
    gpu_enabled: bool

    # Feature Toggles
    feat_momentum_enabled: bool
    feat_volatility_enabled: bool
    feat_price_action_enabled: bool
    feat_micro_enabled: bool
    feat_context_enabled: bool

    # Gate Thresholds
    gate_min_rvol: float
    gate_max_spread: float

    # ── Data source routing ──────────────────────────────────────────────
    use_quotex_streaming: bool
    otc_pairs: list[str]
    poll_interval: float

    # ──Infrastructure ───────────────────────────────────────────────────────
    data_mode: str
    data_dir: str = "/app/data"
    model_dir: str = "/app/models"
    allow_stale_models: bool = False

    # Pipeline orchestration
    train_on_full_history: bool = True
    model_retrain_interval: int = 3600
    model_save_interval: int = 1800

    # This field is not sourced from the environment. It is derived from
    # VALID_SOURCES and exists to provide a single source of truth for valid
    # data sources that can be referenced throughout the codebase, including
    # in the Model class for validating incoming tick data.
    valid_sources: set[str] = field(default_factory=lambda: VALID_SOURCES)

    def __post_init__(self) -> None:
        """
        Run cross-field validation after construction.

        Raises:
            ValueError: If any configured value fails cross-field validation.
        """
        # Validate tick flush size
        if self.tick_flush_size <= 0:
            raise ConfigRangeError(
                field_name="TICK_FLUSH_SIZE",
                value=self.tick_flush_size,
                min_val=1.0,
                max_val=500.0,
            )

        # Webhook key must be set if a webhook URL is configured
        if self.webhook_url and not self.webhook_key:
            raise ConfigRequiredError(
                field_name="WEBHOOK_KEY",
                reason="WEBHOOK_URL is configured but WEBHOOK_KEY is missing",
            )

        # Validate DATA_MODE
        if self.data_mode not in {"LOCAL", "CLOUD"}:
            raise ConfigValidationError(
                field_name="DATA_MODE",
                value=self.data_mode,
                reason=f"Must be 'LOCAL' or 'CLOUD', got '{self.data_mode}'",
            )

        # Conditional Requirement: Azure
        if self.data_mode == "CLOUD" and not self.azure_storage_conn:
            raise ConfigRequiredError(
                field_name="AZURE_STORAGE_CONN",
                reason="DATA_MODE is 'CLOUD' but AZURE_STORAGE_CONN is missing",
            )

        self._validate_martingale_thresholds()
        self._validate_model_intervals()

        # Expiry
        if self.expiry_seconds not in VALID_EXPIRIES:
            raise ConfigExpiryError(
                expiry=self.expiry_seconds, valid_expiries=sorted(VALID_EXPIRIES)
            )

        # Port range
        if not 1 <= self.dashboard_port <= 65535:
            raise ConfigPortError(port=self.dashboard_port)

        self._validate_credentials_and_pairs()

        # Log level validation (don't configure logging here — main.py owns that)
        level: int | None = getattr(logging, self.log_level, None)
        if level is None:
            raise ConfigValidationError(
                field_name="LOG_LEVEL",
                value=self.log_level,
                reason="Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
            )

        # NOTE: do not log here — logging is not yet configured when __post_init__
        # runs. main.py emits CONFIG_LIVE_MODE / CONFIG_PRACTICE_MODE after
        # _configure_logging() so the event reaches the structured handler.

    def _validate_martingale_thresholds(self) -> None:
        if self.martingale_max_streak > 6:
            raise ConfigRangeError(
                field_name="MARTINGALE_MAX_STREAK",
                value=self.martingale_max_streak,
                min_val=0,  # or 1 if zero not allowed
                max_val=6,
            )

        if not 0.005 <= self.martingale_step <= 0.10:
            raise ConfigRangeError(
                field_name="MARTINGALE_STEP",
                value=self.martingale_step,
                min_val=0.005,
                max_val=0.10,
            )

        _threshold_ceiling = (
            self.confidence_threshold
            + self.martingale_max_streak * self.martingale_step
        )
        if _threshold_ceiling >= 1.0:
            raise ConfigCeilingError(
                threshold=self.confidence_threshold,
                max_streak=self.martingale_max_streak,
                step=self.martingale_step,
                ceiling=_threshold_ceiling,
            )

        if not 0.5 < self.confidence_threshold < 1.0:
            raise ConfigRangeError(
                field_name="CONFIDENCE_THRESHOLD",
                value=self.confidence_threshold,
                min_val=0.5,
                max_val=1.0,
            )

    def _validate_model_intervals(self) -> None:
        if self.model_retrain_interval < 60:
            raise ConfigRangeError(
                field_name="MODEL_RETRAIN_INTERVAL",
                value=float(self.model_retrain_interval),
                min_val=60.0,
                max_val=float("inf"),
            )

        if self.model_save_interval < 60:
            raise ConfigRangeError(
                field_name="MODEL_SAVE_INTERVAL",
                value=float(self.model_save_interval),
                min_val=60.0,
                max_val=float("inf"),
            )

        if self.model_save_interval > self.model_retrain_interval:
            raise ConfigRangeError(
                field_name="MODEL_SAVE_INTERVAL",
                value=float(self.model_save_interval),
                min_val=60.0,
                max_val=float(self.model_retrain_interval),
            )

    def _validate_credentials_and_pairs(self) -> None:
        if not self.quotex_email or not self.quotex_password:
            raise ConfigRequiredError(
                field_name="QUOTEX_EMAIL/QUOTEX_PASSWORD",
                reason="Both QUOTEX_EMAIL and QUOTEX_PASSWORD are required for Quotex streaming",
            )

        if self.otc_pairs and len(self.otc_pairs) != len(self.pairs):
            raise ConfigLengthMismatchError(
                field1="OTC_PAIRS",
                len1=len(self.otc_pairs),
                field2="PAIRS",
                len2=len(self.pairs),
            )

    @property
    def quotex_symbols(self) -> dict[str, str]:
        """
        TRANSLATOR: This sits ready for the Trading Engine.
        It maps the 'Pure' names to 'Broker' names.

        For example, if PAIRS contains "EUR_USD", this property will provide
        the corresponding Quotex symbol, which is typically "EURUSD_otc".
        """
        # If you explicitly provided OTC_PAIRS in .env, map them 1:1
        if self.otc_pairs:
            return dict(zip(self.pairs, self.otc_pairs))

        # Otherwise, auto-translate: EUR_USD -> EURUSD_otc
        return {p: f"{p.replace('_', '')}_otc" for p in self.pairs}


def load_config() -> Config:
    """
    Parse all environment variables and return a validated :class:`Config`.

    Raises:
        ValueError: If any required variable is missing or any value is
            invalid.
    """
    # load_config() now uses the Dictator pattern: each variable is read, parsed, and validated in isolation, with
    # detailed error messages that include the variable name, the invalid value, and contextual information to aid debugging.
    pairs: list[str] = _parse_and_validate_pairs("PAIRS", _optional("PAIRS", "EUR_USD"))
    daily_net_profit_target = _parse_float("DAILY_NET_PROFIT_TARGET", default=None)

    return Config(
        # Azure
        azure_storage_conn=_optional("AZURE_STORAGE_CONN"),
        container_name=_optional("CONTAINER_NAME", "traderai"),
        # Dashboard
        dashboard_port=_int("DASHBOARD_PORT", 8080),
        display_tz=_int("DISPLAY_TZ", 0),
        # Data source routing
        otc_pairs=_parse_otc_pairs(_optional("OTC_PAIRS", "")),
        poll_interval=_parse_float("POLL_INTERVAL", default=1.0),
        use_quotex_streaming=_bool("USE_QUOTEX_STREAMING", True),
        # Martingales
        martingale_max_streak=_int("MARTINGALE_MAX_STREAK", 4),
        martingale_step=_parse_float("MARTINGALE_STEP", default=0.02),
        # Operational overrides
        log_level=_optional("LOG_LEVEL", "INFO").upper(),
        log_verbose_events=_optional("LOG_VERBOSE_EVENTS", ""),
        practice_mode=_bool("PRACTICE_MODE", True),
        tick_flush_size=_int("TICK_FLUSH_SIZE", 500),
        data_mode=_optional("DATA_MODE", "LOCAL").upper(),
        max_rf_rows=_int("MAX_RF_ROWS", 50000),
        memory_saver_mode=_bool("MEMORY_SAVER_MODE", False),
        data_dir=_optional("DATA_DIR", "/app/data"),
        model_dir=_optional("MODEL_DIR", "/app/models"),
        # Quotex
        quotex_email=_optional("QUOTEX_EMAIL", ""),
        quotex_password=_optional("QUOTEX_PASSWORD", ""),
        # Reporting
        discord_webhook_url=_optional("DISCORD_WEBHOOK_URL", ""),
        telegram_chat_id=_optional("TELEGRAM_CHAT_ID", ""),
        telegram_token=_optional("TELEGRAM_TOKEN", ""),
        # Trading
        confidence_threshold=_parse_float("CONFIDENCE_THRESHOLD", default=0.55),
        daily_trade_target=_int("DAILY_TRADE_TARGET", 10),
        expiry_seconds=_int("EXPIRY_SECONDS", 60),
        pairs=pairs,
        daily_net_profit_target=daily_net_profit_target,
        trading_window_hours=_int("TRADING_WINDOW_HOURS", 19),
        # Training
        backfill_source=_optional("BACKFILL_SOURCE", "TWELVE").upper(),
        backfill_pairs=_parse_and_validate_pairs(
            "BACKFILL_PAIRS", _optional("BACKFILL_PAIRS", "EUR_USD")
        ),
        gpu_enabled=_bool("GPU_ENABLED", False),
        backfill_years=_int("BACKFILL_YEARS", 2),
        max_sequences=_int("MAX_SEQUENCES", 100000),
        feat_momentum_enabled=_bool("FEAT_MOMENTUM_ENABLED", True),
        feat_volatility_enabled=_bool("FEAT_VOLATILITY_ENABLED", True),
        feat_price_action_enabled=_bool("FEAT_PRICE_ACTION_ENABLED", True),
        feat_micro_enabled=_bool("FEAT_MICRO_ENABLED", True),
        feat_context_enabled=_bool("FEAT_CONTEXT_ENABLED", True),
        gate_min_rvol=_parse_float("GATE_MIN_RVOL", 1.5),
        gate_max_spread=_parse_float("GATE_MAX_SPREAD", 0.0005),
        allow_stale_models=_bool("ALLOW_STALE_MODELS", False),
        train_on_full_history=_bool("TRAIN_ON_FULL_HISTORY", True),
        model_retrain_interval=_int("MODEL_RETRAIN_INTERVAL", 3600),
        model_save_interval=_int("MODEL_SAVE_INTERVAL", 1800),
        # Twelve Data
        twelvedata_api_key=_require("TWELVEDATA_API_KEY"),
        # Webhook
        webhook_key=_optional("WEBHOOK_KEY", ""),
        webhook_secret=_optional("WEBHOOK_SECRET", ""),
        webhook_url=_require("WEBHOOK_URL"),
    )


# ── Initialization Gate ──────────────────────────────────────────────────────
# settings is NOT loaded at import time. Call get_settings() from your
# application entrypoint (main.py) to trigger validation. This keeps
# imports safe for testing, linting, and doc generation.
_settings: Config | None = None


def get_settings() -> Config:
    """
    Return the global validated Config instance.

    Initializes on first call (lazy singleton). Subsequent calls return
    the cached instance. Exits with code 1 on validation failure.

    Returns:
        Config: The validated, frozen configuration object.
    """
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def local_now() -> datetime:
    """
    Return current time shifted by DISPLAY_TZ hours for display purposes.

    All log timestamps, dashboard activity entries, and reporter timestamps
    use this function so the operator sees local time rather than UTC.
    Structured log records show local clock time; no timezone suffix is
    appended so the output stays compact.
    """
    return datetime.now(timezone.utc) + timedelta(hours=get_settings().display_tz)
