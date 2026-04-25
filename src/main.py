"""
main.py — Thin entry point for the Trader AI signal engine.

Delegates all orchestration to core.pipeline.Pipeline which owns the
five-stage boot sequence, LiveEngine lifecycle, and retrain scheduling.

This file handles exactly three things:
    1. Configure structured JSON logging.
    2. Construct and run the Pipeline.
    3. Catch and log fatal errors with clean exit codes.
"""

from __future__ import annotations

import sys
import json
import asyncio
import logging
import traceback


from core.config import get_settings
from datetime import datetime, timezone
from core.exceptions import TraderAIError, ConfigError, FatalError


# ── JSON structured logging ────────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def __init__(self, verbose_events: set[str]):
        super().__init__()
        self._verbose_events = verbose_events
        self._allow_all = "ALL" in verbose_events
        self._minimal_mode = len(verbose_events) == 0
        # Always allow these critical events even in minimal mode
        self._critical_events = {"BOOT", "SHUTDOWN", "FATAL_CRASH"}

    def format(self, record: logging.LogRecord) -> str:
        # Handle dict messages directly — avoids str(dict) round-trip which
        # produces single-quoted keys that json.loads() rejects silently.
        if isinstance(record.msg, dict):
            payload: dict = record.msg
        else:
            msg: str = record.getMessage()
            try:
                payload = json.loads(msg) if msg.startswith("{") else {"message": msg}
            except json.JSONDecodeError:
                payload = {"message": msg}

        event = payload.get("event")
        if self._minimal_mode:
            if event not in self._critical_events:
                return ""
        elif not self._allow_all:
            # Verbose mode: only log events explicitly listed; pass through
            # plain-string records that have no event key.
            if event and event not in self._verbose_events:
                return ""

        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            **payload,
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _configure_logging(verbose_events: set[str], log_level: str) -> None:
    """Set up root logger with JSON formatter and suppress noisy SDKs."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter(verbose_events))

    root = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for name in (
        "azure.core.pipeline.policies.http_logging_policy",
        "azure",
        "azure.storage.blob",
        "urllib3",
        "pyquotex.ws.client",
        "websockets.client",
        "websockets.server",
        "asyncio",
        "charset_normalizer",
        "PIL",
        "matplotlib",
        "numexpr",
        "numba",
        "websocket",
        "websocket-client",
    ):
        logging.getLogger(name).disabled = True


# ── Entry point ────────────────────────────────────────────────────────────────


async def _run() -> None:
    """Import and run the Pipeline (deferred so logging is configured first) always cleanly exits to main() for error handling."""
    from core.pipeline import Pipeline

    pipeline = Pipeline()
    await pipeline.run()


def main() -> None:
    """CLI entry point. Configures logging, runs the async pipeline."""
    try:
        config = get_settings()

        # Parse verbose events from config
        verbose_events = set()
        if config.log_verbose_events:
            verbose_events = {
                e.strip() for e in config.log_verbose_events.split(",") if e.strip()
            }

        _configure_logging(verbose_events, config.log_level)

        # Emit mode event now that the structured handler is in place.
        _mode_logger = logging.getLogger("config")
        if config.practice_mode:
            _mode_logger.info(
                {
                    "event": "CONFIG_PRACTICE_MODE",
                    "message": "Signals will fire but treat results as simulation.",
                }
            )
        else:
            _mode_logger.warning(
                {
                    "event": "CONFIG_LIVE_MODE",
                    "message": "LIVE trading mode — review all parameters and risk settings.",
                }
            )

    except ConfigError as e:
        # Config errors are fatal at startup - always log them
        print(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "CRITICAL",
                    "component": "main",
                    "event": e.event,
                    "error": e.message,
                    "type": e.__class__.__name__,
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    except Exception as e:
        # Unknown error during config load
        print(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "CRITICAL",
                    "component": "main",
                    "event": "CONFIG_LOAD_FAILED",
                    "error": str(e),
                    "type": type(e).__name__,
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    logger = logging.getLogger("main")

    try:
        logger.info({"event": "BOOT", "message": "Trader AI starting"})
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info({"event": "SHUTDOWN", "message": "KeyboardInterrupt"})
    except SystemExit:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    except (TraderAIError, FatalError) as exc:
        # Structured app exceptions — use .to_log_entry() to preserve the
        # event name (e.g. PIPELINE_ERROR_STORAGE) and stage/field context.
        logger.critical(exc.to_log_entry(), exc_info=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
    except Exception as exc:
        logger.critical(
            {"event": "FATAL_CRASH", "error": str(exc), "type": type(exc).__name__},
            exc_info=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
