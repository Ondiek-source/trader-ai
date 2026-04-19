"""
core/exceptions.py — Centralized exception hierarchy with event naming.

All exceptions raised in the codebase inherit from TraderAIError, which
carries an event name for logging control via LOG_VERBOSE_EVENTS.
"""

from __future__ import annotations
from typing import Any


class TraderAIError(Exception):
    """Base exception for all application-specific errors."""

    def __init__(self, event: str, message: str, *args, **kwargs):
        self.event = event
        self.message = message
        self.field_name: str | None = None
        self.value: Any | None = None
        super().__init__(message, *args, **kwargs)

    def to_log_entry(self) -> dict:
        """Convert to structured log entry."""
        entry = {
            "event": self.event,
            "error": self.message,
            "type": self.__class__.__name__,
        }
        if self.field_name:
            entry["field_name"] = self.field_name
        if self.value:
            entry["value"] = self.value
        return entry


# ── Configuration Errors ──────────────────────────────────────────────────────


class ConfigError(TraderAIError):
    """Configuration validation error (missing env vars, invalid values)."""

    def __init__(self, message: str, field_name: str | None = None):
        event = "CONFIG_ERROR"
        if field_name:
            event = f"CONFIG_ERROR_{field_name}"
        super().__init__(event=event, message=message)
        self.field_name = field_name


class ConfigMissingError(ConfigError):
    """Required environment variable is missing."""

    def __init__(self, field_name: str):
        super().__init__(
            message=f"Missing required environment variable: '{field_name}'",
            field_name=field_name,
        )
        self.event = "CONFIG_MISSING"


class ConfigInvalidValueError(ConfigError):
    """Environment variable has invalid value."""

    def __init__(self, field_name: str, value: str, expected: str):
        super().__init__(
            message=f"Invalid value for '{field_name}': '{value}'. Expected: {expected}",
            field_name=field_name,
        )
        self.event = "CONFIG_INVALID_VALUE"
        self.value = value
        self.expected = expected


class ConfigTypeError(ConfigError):
    """Environment variable has wrong type."""

    def __init__(self, field_name: str, value: str, expected_type: str):
        super().__init__(
            message=f"Type mismatch for '{field_name}': '{value}' is not {expected_type}",
            field_name=field_name,
        )
        self.event = "CONFIG_TYPE_ERROR"
        self.value = value
        self.expected_type = expected_type


class ConfigValidationError(ConfigError):
    """General validation failure (pair not supported, etc)."""

    def __init__(self, field_name: str, value: str, reason: str):
        super().__init__(
            message=f"Validation failed for '{field_name}': '{value}' - {reason}",
            field_name=field_name,
        )
        self.event = "CONFIG_VALIDATION_ERROR"
        self.value = value
        self.reason = reason


class ConfigPairError(ConfigError):
    """Currency pair validation error."""

    def __init__(self, field_name: str, pair: str, valid_pairs: list[str]):
        super().__init__(
            message=f"Unsupported pair '{pair}' in {field_name}. Supported: {valid_pairs}",
            field_name=field_name,
        )
        self.event = "CONFIG_PAIR_ERROR"
        self.pair = pair
        self.valid_pairs = valid_pairs


class ConfigLengthMismatchError(ConfigError):
    """Length mismatch between two config lists (e.g., PAIRS vs OTC_PAIRS)."""

    def __init__(self, field1: str, len1: int, field2: str, len2: int):
        super().__init__(
            message=f"{field1} length ({len1}) must match {field2} length ({len2})",
            field_name=f"{field1},{field2}",
        )
        self.event = "CONFIG_LENGTH_MISMATCH"
        self.field1 = field1
        self.len1 = len1
        self.field2 = field2
        self.len2 = len2


class ConfigExpiryError(ConfigError):
    """Invalid expiry seconds value."""

    def __init__(self, expiry: int, valid_expiries: list[int]):
        super().__init__(
            message=f"EXPIRY_SECONDS must be one of {valid_expiries}, got {expiry}",
            field_name="EXPIRY_SECONDS",
        )
        self.event = "CONFIG_EXPIRY_ERROR"
        self.expiry = expiry
        self.valid_expiries = valid_expiries


class ConfigPortError(ConfigError):
    """Invalid port number."""

    def __init__(self, port: int):
        super().__init__(
            message=f"DASHBOARD_PORT must be 1-65535, got {port}",
            field_name="DASHBOARD_PORT",
        )
        self.event = "CONFIG_PORT_ERROR"
        self.port = port


class ConfigRangeError(ConfigError):
    """Value outside allowed range."""

    def __init__(self, field_name: str, value: float, min_val: float, max_val: float):
        super().__init__(
            message=f"{field_name} must be between {min_val} and {max_val}, got {value}",
            field_name=field_name,
        )
        self.event = "CONFIG_RANGE_ERROR"
        self.value = value
        self.min_val = min_val
        self.max_val = max_val


class ConfigCeilingError(ConfigError):
    """Martingale ceiling breach."""

    def __init__(self, threshold: float, max_streak: int, step: float, ceiling: float):
        super().__init__(
            message=f"Martingale ceiling breach: {threshold} + {max_streak} × {step} = {ceiling} >= 1.0",
            field_name="MARTINGALE",
        )
        self.event = "CONFIG_CEILING_ERROR"
        self.threshold = threshold
        self.max_streak = max_streak
        self.step = step
        self.ceiling = ceiling


class ConfigRequiredError(ConfigError):
    """Required field missing (already handled by ConfigMissingError but for non-env fields)."""

    def __init__(self, field_name: str, reason: str):
        super().__init__(
            message=f"Required field '{field_name}' is missing: {reason}",
            field_name=field_name,
        )
        self.event = "CONFIG_REQUIRED_ERROR"


# ── Pipeline Errors ────────────────────────────────────────────────────────────


class PipelineError(TraderAIError):
    """
    Raised when a critical stage of the boot sequence fails.

    Propagates to main.py which logs it and exits. Detailed diagnostic
    logging is always emitted by the stage method before raising so the
    operator sees exactly which stage and component failed.

    Attributes:
        stage: The boot stage that failed, e.g. "storage", "historian_sync".
    """

    def __init__(self, message: str, stage: str = ""):
        event = f"PIPELINE_ERROR_{stage.upper()}" if stage else "PIPELINE_ERROR"
        super().__init__(event=event, message=message)
        self.stage = stage


class NotificationError(PipelineError):
    """Failed to send boot notification."""

    def __init__(self, message: str, channel: str = ""):
        event = (
            f"NOTIFICATION_ERROR_{channel.upper()}" if channel else "NOTIFICATION_ERROR"
        )
        super().__init__(message=message, stage="notification")
        self.event = event
        self.channel = channel


# ── Runtime Errors ────────────────────────────────────────────────────────────


class TradingError(TraderAIError):
    """Trading execution error."""

    def __init__(self, message: str, pair: str | None = None):
        event = "TRADING_ERROR"
        if pair:
            event = f"TRADING_ERROR_{pair}"
        super().__init__(event=event, message=message)
        self.pair = pair


class ConnectionError(TraderAIError):
    """Network/WebSocket connection error."""

    def __init__(self, message: str, source: str):
        super().__init__(event=f"CONNECTION_ERROR_{source.upper()}", message=message)
        self.source = source


class DataError(TraderAIError):
    """Data fetching/parsing error."""

    def __init__(self, message: str, source: str):
        super().__init__(event=f"DATA_ERROR_{source.upper()}", message=message)
        self.source = source


class ModelError(TraderAIError):
    """ML model inference/training error."""

    def __init__(self, message: str, operation: str):
        super().__init__(event=f"MODEL_ERROR_{operation.upper()}", message=message)
        self.operation = operation


class RateLimitError(TraderAIError):
    """Rate limit exceeded."""

    def __init__(self, message: str, resource: str):
        super().__init__(event="RATE_LIMIT_ERROR", message=message)
        self.resource = resource


# ── Recovery / Retry Errors ───────────────────────────────────────────────────


class RetryableError(TraderAIError):
    """Error that can be retried (transient failure)."""

    def __init__(self, message: str, event: str = "RETRYABLE_ERROR"):
        super().__init__(event=event, message=message)


class FatalError(TraderAIError):
    """Non-recoverable error that should crash the application."""

    def __init__(self, message: str, event: str = "FATAL_CRASH"):
        super().__init__(event=event, message=message)
