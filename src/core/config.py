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

from typing import overload
from dotenv import load_dotenv
from dataclasses import dataclass, field

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
        # Gather diagnostic details

        error_msg = (
            f"\n{'!'*60}\n"
            f"FATAL CONFIGURATION ERROR\n"
            f"Missing Required Variable: {name}\n"
            f"Context Details:\n"
            f"  - Working Directory: {CWD}\n"
            f"  - .env File Found: {ENV_FILE_EXISTS}\n"
            f"  - System User: {SYSTEM_USER}\n"
            f"  - Hint: If running in Docker, ensure the .env is passed via 'env_file'.\n"
            f"{'!'*60}"
        )

        # Log to the critical path (File + Console)
        logger.critical(error_msg)
        # Raise and Fail Fast with a clear message
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
        error_block = (
            f"\n{'~'*60}\n"
            f"TYPE MISMATCH ERROR: {name}\n"
            f"Expected: FLOAT (Decimal)\n"
            f"Received: '{_trunc(raw.strip())}'\n\n"
            f"SYSTEM SNAPSHOT:\n"
            f"  - Working Directory: {CWD}\n"
            f"  - .env File Present: {ENV_FILE_EXISTS}\n"
            f"  - System User:      {SYSTEM_USER}\n"
            f"  - Python Version:   {PYTHON_VER}\n"
            f"\nFIX: In your .env, ensure '{name}' uses a period for decimals (e.g., 0.53).\n"
            f"{'~'*60}"
        )
        logger.critical(error_block)
        raise ValueError(f"Optional variable '{name}' must be a float or empty.")


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
        error_block = (
            f"\n{'#'*60}\n"
            f"TYPE MISMATCH ERROR: {name}\n"
            f"Expected: INTEGER\n"
            f"Received: '{_trunc(raw.strip())}'\n\n"
            f"SYSTEM SNAPSHOT:\n"
            f"  - Working Directory: {CWD}\n"
            f"  - .env File Present: {ENV_FILE_EXISTS}\n"
            f"  - System User:      {SYSTEM_USER}\n"
            f"\nFIX: Open your .env file and ensure '{name}' is a whole number.\n"
            f"{'#'*60}"
        )
        logger.critical(error_block)
        raise ValueError(
            f"Environment variable '{name}' must be an integer, got: '{raw.strip()}'"
        )


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
            error_block = (
                f"\n{'='*60}\n"
                f"VALIDATION ERROR: UNSUPPORTED PAIR\n"
                f"Variable: {name}\n"
                f"Invalid Pair Found: '{p}'\n\n"
                f"SUPPORTED SET: {sorted(list(VALID_PAIRS))}\n"
                f"\nFIX: Remove '{p}' from your {name} list in .env.\n"
                f"{'='*60}"
            )
            logger.critical(error_block)
            raise ValueError(f"Unsupported currency pair: {p}")

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
        error_block = (
            f"\n{'='*60}\n"
            f"PARSING ERROR: OTC_PAIRS\n"
            f"Input Received: '{raw}'\n"
            f"Status: String contains no valid symbols (only delimiters?).\n\n"
            f"FIX: OTC_PAIRS must contain at least one symbol name.\n"
            f"     Example: OTC_PAIRS=EURUSD_otc,GBPUSD_otc\n"
            f"     If you do not need OTC overrides, leave OTC_PAIRS blank.\n"
            f"{'='*60}"
        )
        logger.critical(error_block)
        raise ValueError("OTC_PAIRS is set but contains no valid pairs.")

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
            error_block = (
                f"\n{'!'*60}\n"
                f"VALIDATION ERROR: UNSUPPORTED BASE PAIR IN OTC_PAIRS\n"
                f"Raw input     : '{raw_symbol}'\n"
                f"Extracted base: '{base_pair}'\n"
                f"Valid pairs   : {sorted(VALID_OTC_BASE_PAIRS)}\n\n"
                f"CONTEXT: The OTC symbol could not be mapped to a valid forex pair.\n"
                f"  - Check for typos in your .env file.\n"
                f"  - Supported pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, XAUUSD\n"
                f"\nFIX: Correct OTC_PAIRS in .env or remove it to use auto-translation.\n"
                f"{'!'*60}"
            )
            logger.critical(error_block)
            raise ValueError(f"Unsupported base pair in OTC_PAIRS: '{base_pair}'")

        # Step 3: Reconstruct with CORRECT format (uppercase _otc)
        correct_symbol = f"{base_pair.upper()}_otc"
        validated_pairs.append(correct_symbol)

        # Log the transformation for transparency
        if raw_symbol != correct_symbol:
            logger.info(
                f"[CONFIG] OTC symbol normalized: '{raw_symbol}' -> '{correct_symbol}'"
            )

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
    tick_flush_size: int
    martingale_max_streak: int
    martingale_step: float  # per-loss confidence-threshold increment

    # ── Reporting / notifications ─────────────────────────────────────────
    telegram_token: str
    telegram_chat_id: str
    discord_webhook_url: str

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard_port: int

    # ── ML / Training ──────────────────────────────────────────────────────────
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
            error_block = (
                f"\n{'%'*60}\n"
                f"CONFIGURATION ERROR: TICK_FLUSH_SIZE\n"
                f"Value Received: {_trunc(self.tick_flush_size)}\n\n"
                f"CONTEXT: TICK_FLUSH_SIZE determines how many ticks are stored in memory before being flushed to disk. \n"
                f"A non-positive value would cause the system to never flush, leading to unbounded memory growth and potential crashes.\n"
                f"\nFIX: Set TICK_FLUSH_SIZE to a positive integer (e.g., 500) in your .env file.\n"
                f"{'%'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"TICK_FLUSH_SIZE must be > 0, got {_trunc(self.tick_flush_size)}"
            )

        # Webhook key must be set if a webhook URL is configured
        if self.webhook_url and not self.webhook_key:
            error_block = (
                f"\n{'!'*60}\n"
                f"SECURITY ERROR: MISSING WEBHOOK_KEY\n"
                f"WEBHOOK_URL is set but WEBHOOK_KEY is empty.\n\n"
                f"CONTEXT: Sending trade signals to an unauthenticated endpoint\n"
                f"creates a security vulnerability. Any actor with the URL\n"
                f"could trigger or intercept trade execution.\n"
                f"\nFIX: Set WEBHOOK_KEY to a strong secret string in your .env.\n"
                f"     Example: WEBHOOK_KEY=<generate a random 32-char string>\n"
                f"{'!'*60}"
            )
            logger.critical(error_block)
            raise ValueError("WEBHOOK_KEY must be set when WEBHOOK_URL is configured.")

        # Validate DATA_MODE
        if self.data_mode not in {"LOCAL", "CLOUD"}:
            error_block = (
                f"\n{'!'*60}\n"
                f"CONFIGURATION ERROR: INVALID DATA_MODE\n"
                f"Value Received: '{_trunc(self.data_mode)}'\n\n"
                f"ALLOWED VALUES: 'LOCAL' or 'CLOUD'\n"
                f"CONTEXT: DATA_MODE determines where the system reads/writes data. \n"
                f"An invalid setting can cause failures in data access and storage.\n"
                f"\nFIX: Set DATA_MODE to either 'LOCAL' or 'CLOUD' in your .env file.\n"
                f"{'!'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"DATA_MODE must be 'LOCAL' or 'CLOUD', got: '{_trunc(self.data_mode)}'"
            )

        # Conditional Requirement: Azure
        if self.data_mode == "CLOUD" and not self.azure_storage_conn:
            error_block = (
                f"\n{'!'*60}\n"
                f"CONFIGURATION ERROR: MISSING AZURE_STORAGE_CONN\n"
                f"DATA_MODE is 'CLOUD' but AZURE_STORAGE_CONN is empty.\n"
                f"CONTEXT: Cloud mode requires a valid Azure storage connection.\n"
                f"\nFIX: Set AZURE_STORAGE_CONN in your .env file.\n"
                f"{'!'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                "AZURE_STORAGE_CONN is required when DATA_MODE is 'CLOUD'."
            )

        # Martingale max streak sanity check
        if self.martingale_max_streak > 6:
            error_block = (
                f"\n{'%'*60}\n"
                f"CONFIGURATION ERROR: MARTINGALE_MAX_STREAK\n"
                f"Value Received: {_trunc(self.martingale_max_streak)}\n\n"
                f"CONTEXT: Setting a Martingale max streak above 6 can lead to \n"
                f"catastrophic losses due to exponential bet sizing. This is a \n"
                f"safety check to prevent misconfiguration.\n"
                f"\nFIX: Set MARTINGALE_MAX_STREAK to 6 or below in .env.\n"
                f"{'%'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                "Martingale max streak cannot exceed 6 due to risk of catastrophic losses."
            )

        # Martingale step range
        if not 0.005 <= self.martingale_step <= 0.10:
            error_block = (
                f"\n{'%'*60}\n"
                f"CONFIGURATION ERROR: MARTINGALE_STEP\n"
                f"Value Received: {_trunc(self.martingale_step)}\n\n"
                f"ALLOWED RANGE: 0.005 – 0.10\n"
                f"CONTEXT: MARTINGALE_STEP controls how much the confidence threshold\n"
                f"rises after each consecutive loss. A value below 0.005 is too small\n"
                f"to have any protective effect; a value above 0.10 would push the\n"
                f"threshold unreachably high after just a few losses.\n"
                f"\nFIX: Set MARTINGALE_STEP between 0.005 and 0.10 in .env.\n"
                f"Recommended: 0.02 (raises threshold 2 pp per loss).\n"
                f"{'%'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"MARTINGALE_STEP must be between 0.005 and 0.10, "
                f"got {_trunc(self.martingale_step)}."
            )

        # Ceiling check: base_threshold + max_streak * step must be < 1.0
        _threshold_ceiling = (
            self.confidence_threshold
            + self.martingale_max_streak * self.martingale_step
        )
        if _threshold_ceiling >= 1.0:
            error_block = (
                f"\n{'%'*60}\n"
                f"CONFIGURATION ERROR: MARTINGALE CEILING BREACH\n"
                f"CONFIDENCE_THRESHOLD   : {_trunc(self.confidence_threshold)}\n"
                f"MARTINGALE_STEP        : {_trunc(self.martingale_step)}\n"
                f"MARTINGALE_MAX_STREAK  : {_trunc(self.martingale_max_streak)}\n"
                f"Computed ceiling       : {_trunc(_threshold_ceiling)} (must be < 1.0)\n\n"
                f"CONTEXT: After {_trunc(self.martingale_max_streak)} consecutive losses the\n"
                f"threshold would reach {_trunc(_threshold_ceiling)}, which is unreachable\n"
                f"and would permanently halt all trading.\n"
                f"\nFIX: Reduce MARTINGALE_STEP or MARTINGALE_MAX_STREAK so that\n"
                f"     CONFIDENCE_THRESHOLD + MAX_STREAK * STEP < 1.0.\n"
                f"{'%'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"Martingale ceiling breach: {_trunc(self.confidence_threshold)} + "
                f"{_trunc(self.martingale_max_streak)} × {_trunc(self.martingale_step)} "
                f"= {_trunc(_threshold_ceiling)} >= 1.0."
            )

        # Confidence threshold
        if not 0.5 < self.confidence_threshold < 1.0:
            error_block = (
                f"\n{'%'*60}\n"
                f"LOGIC VALIDATION ERROR: CONFIDENCE_THRESHOLD\n"
                f"Range Required: 0.5 < Value < 1.0\n"
                f"Value Received: {_trunc(self.confidence_threshold)}\n\n"
                f"CONTEXT:\n"
                f"  - 0.50 means 50% (Coin flip/Random)\n"
                f"  - 1.00 means 100% (Impossible/Overfitted)\n"
                f"  - Current setting is mathematically invalid for trading logic.\n"
                f"\nFIX: Set CONFIDENCE_THRESHOLD between 0.51 and 0.99 in .env.\n"
                f"{'%'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"Invalid Confidence Threshold: {_trunc(self.confidence_threshold)}"
            )

        # Model retrain interval sanity check
        if self.model_retrain_interval < 60:
            error_block = (
                f"\n{'%' * 60}\n"
                f"CONFIGURATION ERROR: MODEL_RETRAIN_INTERVAL\n"
                f"Value Received: {_trunc(self.model_retrain_interval)} seconds\n\n"
                f"CONTEXT: MODEL_RETRAIN_INTERVAL controls how frequently the\n"
                f"pipeline triggers a full model retrain. Values below 60 seconds\n"
                f"would trigger continuous retraining, exhausting CPU/GPU resources\n"
                f"and preventing the inference loop from running stably.\n"
                f"\nFIX: Set MODEL_RETRAIN_INTERVAL to 60 or above in .env.\n"
                f"Recommended: 3600 (1 hour) for live trading.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"MODEL_RETRAIN_INTERVAL must be >= 60 seconds, "
                f"got {_trunc(self.model_retrain_interval)}."
            )

        if self.model_save_interval < 60:
            error_block = (
                f"\n{'%' * 60}\n"
                f"CONFIGURATION ERROR: MODEL_SAVE_INTERVAL\n"
                f"Value Received: {_trunc(self.model_save_interval)} seconds\n\n"
                f"CONTEXT: MODEL_SAVE_INTERVAL controls how frequently the\n"
                f"pipeline checkpoints the current model to disk and Blob.\n"
                f"Values below 60 seconds would cause continuous I/O that\n"
                f"would saturate disk and network bandwidth.\n"
                f"\nFIX: Set MODEL_SAVE_INTERVAL to 60 or above in .env.\n"
                f"Recommended: 1800 (30 minutes) for live trading.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"MODEL_SAVE_INTERVAL must be >= 60 seconds, "
                f"got {_trunc(self.model_save_interval)}."
            )

        if self.model_save_interval > self.model_retrain_interval:
            error_block = (
                f"\n{'%' * 60}\n"
                f"CONFIGURATION ERROR: INTERVAL ORDERING\n"
                f"MODEL_SAVE_INTERVAL   : {_trunc(self.model_save_interval)}s\n"
                f"MODEL_RETRAIN_INTERVAL: {_trunc(self.model_retrain_interval)}s\n\n"
                f"CONTEXT: MODEL_SAVE_INTERVAL must be <= MODEL_RETRAIN_INTERVAL.\n"
                f"Saving a checkpoint more frequently than a full retrain is\n"
                f"valid, but saving LESS frequently means the last retrain result\n"
                f"may never be persisted before the next retrain overwrites it.\n"
                f"\nFIX: Set MODEL_SAVE_INTERVAL <= MODEL_RETRAIN_INTERVAL in .env.\n"
                f"{'%' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"MODEL_SAVE_INTERVAL ({_trunc(self.model_save_interval)}s) must be "
                f"<= MODEL_RETRAIN_INTERVAL ({_trunc(self.model_retrain_interval)}s)."
            )

        # Expiry
        if self.expiry_seconds not in VALID_EXPIRIES:
            error_block = (
                f"\n{'?'*60}\n"
                f"VALIDATION ERROR: UNTESTED EXPIRY TIME\n"
                f"Value Received: {_trunc(self.expiry_seconds)} seconds\n\n"
                f"ALLOWED SET: {sorted(list(VALID_EXPIRIES))}\n"
                f"CONTEXT: High-frequency trading models are trained on specific \n"
                f"timeframes. Using {_trunc(self.expiry_seconds)}s may lead to signal drift.\n"
                f"\nFIX: Change EXPIRY_SECONDS to one of {sorted(list(VALID_EXPIRIES))} in .env.\n"
                f"{'?'*60}"
            )
            logger.critical(error_block)
            raise ValueError(f"Untested expiry time: {_trunc(self.expiry_seconds)}")

        # Port range
        if not 1 <= self.dashboard_port <= 65535:
            error_block = (
                f"\n{'^'*60}\n"
                f"NETWORK ERROR: INVALID DASHBOARD PORT\n"
                f"Value Received: {_trunc(self.dashboard_port)}\n\n"
                f"ALLOWED RANGE: 1 - 65535\n"
                f"CONTEXT: The system cannot bind the dashboard to a non-existent \n"
                f"or restricted networking port.\n"
                f"\nFIX: Set DASHBOARD_PORT to a valid number (usually 8000-9000).\n"
                f"{'^'*60}"
            )
            logger.critical(error_block)
            raise ValueError(f"Invalid network port: {_trunc(self.dashboard_port)}")

        # Quotex credentials required
        if not self.quotex_email or not self.quotex_password:
            error_block = (
                f"\n{'#'*60}\n"
                f"CONFIGURATION ERROR: QUOTEX CREDENTIALS\n"
                f"QUOTEX_EMAIL or QUOTEX_PASSWORD is not set.\n\n"
                f"CONTEXT: Quotex streaming is enabled, but credentials are missing. \n"
                f"Without these, the system cannot connect to Quotex for live data.\n"
                f"\nFIX: Set QUOTEX_EMAIL and QUOTEX_PASSWORD in your .env file.\n"
                f"{'#'*60}"
            )
            logger.critical(error_block)
            raise ValueError("Quotex credentials are required.")

        # OTC pair count must match trading pair count
        if self.otc_pairs and len(self.otc_pairs) != len(self.pairs):
            error_block = (
                f"\n{'='*60}\n"
                f"CONFIGURATION ERROR: OTC_PAIRS LENGTH MISMATCH\n"
                f"PAIRS count     : {len(self.pairs)} -> {_trunc(self.pairs)}\n"
                f"OTC_PAIRS count : {len(self.otc_pairs)} -> {_trunc(self.otc_pairs)}\n\n"
                f"CONTEXT: When OTC_PAIRS is set, each entry maps 1-to-1 with\n"
                f"PAIRS. A mismatch causes silent truncation via zip(), meaning\n"
                f"some pairs will be dropped from the symbol map with no error.\n"
                f"\nFIX: Ensure OTC_PAIRS has exactly one symbol per entry in PAIRS.\n"
                f"     PAIRS={','.join(self.pairs)}\n"
                f"     OTC_PAIRS must have {len(self.pairs)} entries.\n"
                f"{'='*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"OTC_PAIRS length ({len(self.otc_pairs)}) must match "
                f"PAIRS length ({len(self.pairs)})."
            )

        # Log level validation (don't configure logging here — main.py owns that)
        level: int | None = getattr(logging, self.log_level, None)
        if level is None:
            error_block = (
                f"\n{'#'*60}\n"
                f"CONFIGURATION ERROR: INVALID LOG_LEVEL\n"
                f"Value Received: '{_trunc(self.log_level)}'\n\n"
                f"ALLOWED VALUES: DEBUG, INFO, WARNING, ERROR, CRITICAL\n"
                f"CONTEXT: An invalid log level can cause logging to fail silently or \n"
                f"produce unreadable logs, hindering debugging and monitoring.\n"
                f"\nFIX: Set LOG_LEVEL to a valid logging level in .env.\n"
                f"{'#'*60}"
            )
            logger.critical(error_block)
            raise ValueError(
                f"LOG_LEVEL must be a valid Python logging level, "
                f"got: '{_trunc(self.log_level)}'"
            )

        # Live mode warning
        if not self.practice_mode:
            logger.warning(
                "[LIVE MODE ACTIVE] — System is configured for LIVE trading. "
                "Ensure you have reviewed all parameters and risk settings."
            )
        else:
            logger.info(
                "[PRACTICE MODE] — Signals will fire but treat results as "
                "simulation."
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
        # Data source routing
        otc_pairs=_parse_otc_pairs(_optional("OTC_PAIRS", "")),
        poll_interval=_parse_float("POLL_INTERVAL", default=1.0),
        use_quotex_streaming=_bool("USE_QUOTEX_STREAMING", True),
        # Martingales
        martingale_max_streak=_int("MARTINGALE_MAX_STREAK", 4),
        martingale_step=_parse_float("MARTINGALE_STEP", default=0.02),
        # Operational overrides
        log_level=_optional("LOG_LEVEL", "INFO").upper(),
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
        try:
            _settings = load_config()
        except Exception as e:
            print(f"\nCRITICAL BOOTSTRAP FAILURE: {e}", file=sys.stderr)
            sys.exit(1)
    return _settings
