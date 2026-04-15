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

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone

# ── JSON structured logging ────────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        msg: str = record.getMessage()
        try:
            payload: dict = (
                json.loads(msg) if msg.startswith("{") else {"message": msg}
            )
        except (json.JSONDecodeError, ValueError):
            payload = {"message": msg}

        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            **payload,
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _configure_logging() -> None:
    """Set up root logger with JSON formatter and suppress noisy SDKs."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for name in (
        "azure.core.pipeline.policies.http_logging_policy",
        "azure",
        "azure.storage.blob",
        "urllib3",
        "pyquotex.ws.client",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Entry point ────────────────────────────────────────────────────────────────


async def _run() -> None:
    """Import and run the Pipeline (deferred so logging is configured first)."""
    from core.pipeline import Pipeline

    pipeline = Pipeline()
    await pipeline.run()


def main() -> None:
    """CLI entry point. Configures logging, runs the async pipeline."""
    _configure_logging()
    logger = logging.getLogger("main")

    try:
        logger.info({"event": "boot", "message": "Trader AI starting"})
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info({"event": "shutdown", "message": "KeyboardInterrupt"})
    except SystemExit:
        raise
    except Exception as exc:
        logger.critical(
            {"event": "fatal_crash", "error": str(exc), "type": type(exc).__name__},
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
