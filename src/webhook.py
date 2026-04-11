"""
webhook.py — Webhook signal delivery.

Sends the EXACT payload the Quotex trading bot expects::

    {"side": "buy"|"sell", "symbol": "EURUSD", "key": "Ondiek"}

Nothing extra is added. The payload is passed through as-is.

Retry logic: 3 attempts, exponential backoff (1 s → 2 s → 4 s).
Optional HMAC-SHA256 signing via X-Signature header.
Structured JSON logging for every attempt.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)


class WebhookError(Exception):
    """Raised when all retry attempts are exhausted."""


class WebhookSender:
    """
    Delivers signal payloads to the configured webhook URL.

    The payload POSTed is exactly what is passed in — no fields are added
    or modified.  This guarantees the Quotex bot receives its expected schema.

    Reuses a single :class:`requests.Session` for connection pooling.
    Call :meth:`close` when done to release resources.

    Args:
        url: Webhook endpoint URL.
        secret: Optional HMAC-SHA256 secret for ``X-Signature`` header.

    Example::

        sender = WebhookSender(url="https://example.com/hook", secret="s3cret")
        result = sender.send({"side": "buy", "symbol": "EURUSD", "key": "Ondiek"})
        sender.close()
    """

    MAX_ATTEMPTS = 3
    BACKOFF_SECONDS = [1, 2, 4]
    TIMEOUT_SECONDS = 5

    def __init__(self, url: str, secret: str = "") -> None:
        self._url = url
        self._secret = secret
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST *payload* to the webhook URL with retry.

        Args:
            payload: Must match the schema the Quotex bot expects
                (``{"side": ..., "symbol": ..., "key": ...}``).

        Returns:
            ``{success, status_code, latency_ms, attempts}``.

        Raises:
            WebhookError: If all retry attempts fail.
        """
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers: dict[str, str] = {}
        if self._secret:
            headers["X-Signature"] = self._sign(body)

        last_error: Exception | None = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            start = time.monotonic()
            status_code: int | None = None
            try:
                resp = self._session.post(
                    self._url,
                    data=body,
                    headers=headers,
                    timeout=self.TIMEOUT_SECONDS,
                )
                status_code = resp.status_code
                latency_ms = (time.monotonic() - start) * 1000
                self._log_attempt(payload, attempt, status_code, latency_ms)

                if 200 <= status_code < 300:
                    return {
                        "success": True,
                        "status_code": status_code,
                        "latency_ms": round(latency_ms, 1),
                        "attempts": attempt,
                    }

                # Non-2xx — retry
                last_error = requests.HTTPError(f"HTTP {status_code}")

            except requests.RequestException as exc:
                latency_ms = (time.monotonic() - start) * 1000
                self._log_attempt(
                    payload, attempt, None, latency_ms, error=exc
                )
                last_error = exc

            if attempt < self.MAX_ATTEMPTS:
                wait = self.BACKOFF_SECONDS[attempt - 1]
                logger.info(
                    {
                        "event": "webhook_retry_wait",
                        "attempt": attempt,
                        "wait_seconds": wait,
                        "symbol": payload.get("symbol"),
                    }
                )
                time.sleep(wait)

        # All attempts failed
        logger.error(
            {
                "event": "webhook_all_attempts_failed",
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "attempts": self.MAX_ATTEMPTS,
                "error": str(last_error),
            }
        )
        raise WebhookError(
            f"Webhook failed after {self.MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def _sign(self, body: bytes) -> str:
        """
        Compute HMAC-SHA256 signature over raw body bytes.

        Returns:
            ``"sha256=<hex>"`` string for the ``X-Signature`` header.
        """
        sig = hmac.new(
            self._secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return f"sha256={sig}"

    def _log_attempt(
        self,
        payload: dict[str, Any],
        attempt: int,
        status: int | None,
        latency_ms: float,
        error: Exception | None = None,
    ) -> None:
        """Log a structured record for one send attempt."""
        record: dict[str, Any] = {
            "event": "webhook_attempt",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "attempt": attempt,
            "http_status": status,
            "latency_ms": round(latency_ms, 1),
            "success": status is not None and 200 <= status < 300,
        }
        if error:
            record["error"] = str(error)
        logger.info(record)


# ── Module-level sender (singleton) ───────────────────────────────────────────

_sender: WebhookSender | None = None


def _get_sender(url: str, secret: str = "") -> WebhookSender:
    """Return a module-level singleton :class:`WebhookSender`."""
    global _sender
    if _sender is None or _sender._url != url:
        if _sender is not None:
            _sender.close()
        _sender = WebhookSender(url=url, secret=secret)
    return _sender


def send_signal(payload: dict[str, Any], config: Any) -> dict[str, Any]:
    """
    Convenience wrapper: create (or reuse) a :class:`WebhookSender` from
    *config* and fire once.

    Args:
        payload: Must already be in the exact format the bot expects::

            {"side": "buy"|"sell", "symbol": "EURUSD", "key": "Ondiek"}

        config: Application :class:`~config.Config` (needs ``webhook_url``
            and ``webhook_secret``).

    Returns:
        ``{success, status_code, latency_ms, attempts}``.

    Raises:
        WebhookError: If all retry attempts fail.
    """
    sender = _get_sender(
        url=config.webhook_url,
        secret=getattr(config, "webhook_secret", ""),
    )
    return sender.send(payload)


def close_sender() -> None:
    """Close the module-level singleton sender (call on shutdown)."""
    global _sender
    if _sender is not None:
        _sender.close()
        _sender = None
