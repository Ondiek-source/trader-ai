"""
webhook.py — Webhook signal delivery.

Sends the EXACT payload the Quotex trading bot expects::

    {"side": "buy"|"sell", "symbol": "EURUSD_otc", "key": "Ondiek"}

Nothing extra is added. The payload is passed through as-is.

Retry logic: 3 attempts, exponential backoff (1 s → 2 s → 4 s).
Optional HMAC-SHA256 signing via X-Signature header.
Structured JSON logging for every attempt.
"""

from __future__ import annotations

import asyncio
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
    Supports context manager protocol (``with WebhookSender(...) as sender:``).

    Supports both sync and async usage:
        - :meth:`send` — synchronous (blocks the current thread).
        - :meth:`send_async` — async (uses ``asyncio.sleep`` for retries,
            does not block the event loop).

    Args:
        url: Webhook endpoint URL.
        secret: Optional HMAC-SHA256 secret for ``X-Signature`` header.

    Example::

        with WebhookSender(url="https://example.com/hook", secret="s3cret") as sender:
            result = sender.send({"side": "buy", "symbol": "EURUSD_otc", "key": "Ondiek"})
    """

    MAX_ATTEMPTS: int = 3
    BACKOFF_SECONDS: list[int] = [1, 2, 4]
    TIMEOUT_SECONDS: int = 5

    def __init__(self, url: str, secret: str = "") -> None:
        self._url: str = url
        self._secret: str = secret
        self._session: requests.Session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Context Manager ───────────────────────────────────────────────────────

    def __enter__(self) -> WebhookSender:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ── Core Retry Engine (shared by sync and async) ──────────────────────────

    def _build_headers(self, body: bytes) -> dict[str, str]:
        """Build request headers, including optional HMAC signature."""
        headers: dict[str, str] = {}
        if self._secret:
            headers["X-Signature"] = self._sign(body)
        return headers

    def _evaluate_response(
        self,
        resp: requests.Response,
        start: float,
        payload: dict[str, Any],
        attempt: int,
    ) -> dict[str, Any] | None:
        """
        Evaluate an HTTP response. Returns result dict on success (2xx),
        None on failure (to trigger retry).
        """
        status_code: int = resp.status_code
        latency_ms: float = (time.monotonic() - start) * 1000

        is_success: bool = 200 <= status_code < 300
        self._log_attempt(payload, attempt, status_code, latency_ms, success=is_success)

        if is_success:
            return {
                "success": True,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 1),
                "attempts": attempt,
            }
        return None

    def _handle_exception(
        self,
        exc: requests.RequestException,
        start: float,
        payload: dict[str, Any],
        attempt: int,
    ) -> None:
        """Log a failed attempt due to a network exception."""
        latency_ms: float = (time.monotonic() - start) * 1000
        self._log_attempt(payload, attempt, None, latency_ms, success=False, error=exc)

    # ── Async send (preferred in asyncio context) ─────────────────────────────

    async def send_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST *payload* to the webhook URL with retry (async-safe).

        Uses ``asyncio.sleep`` for backoff so the event loop is never
        blocked by retry waits.

        Args:
            payload: Must match the schema the Quotex bot expects
                (``{"side": ..., "symbol": ..., "key": ...}``).

        Returns:
            Dict with keys ``success``, ``status_code``, ``latency_ms``,
            ``attempts``.

        Raises:
            WebhookError: If all retry attempts fail.
        """
        body: bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers: dict[str, str] = self._build_headers(body)

        last_error: Exception | None = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            start: float = time.monotonic()
            try:
                resp: (
                    requests.Response
                ) = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._session.post(
                        self._url,
                        data=body,
                        headers=headers,
                        timeout=self.TIMEOUT_SECONDS,
                    ),
                )

                result = self._evaluate_response(resp, start, payload, attempt)
                if result is not None:
                    return result

                last_error = requests.HTTPError(f"HTTP {resp.status_code}")

            except requests.RequestException as exc:
                self._handle_exception(exc, start, payload, attempt)
                last_error = exc

            if attempt < self.MAX_ATTEMPTS:
                wait: int = self.BACKOFF_SECONDS[attempt - 1]
                logger.info(
                    {
                        "event": "webhook_retry_wait",
                        "attempt": attempt,
                        "wait_seconds": wait,
                        "symbol": payload.get("symbol"),
                    }
                )
                await asyncio.sleep(wait)

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

    # ── Sync send (blocks current thread) ─────────────────────────────────────

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST *payload* to the webhook URL with retry (synchronous).

        **Warning:** Uses ``time.sleep`` for backoff.  Do not call from
        an async function — use :meth:`send_async` instead.

        Args:
            payload: Must match the schema the Quotex bot expects
                (``{"side": ..., "symbol": ..., "key": ...}``).

        Returns:
            Dict with keys ``success``, ``status_code``, ``latency_ms``,
            ``attempts``.

        Raises:
            WebhookError: If all retry attempts fail.
        """
        body: bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers: dict[str, str] = self._build_headers(body)

        last_error: Exception | None = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            start: float = time.monotonic()
            try:
                resp = self._session.post(
                    self._url,
                    data=body,
                    headers=headers,
                    timeout=self.TIMEOUT_SECONDS,
                )

                result = self._evaluate_response(resp, start, payload, attempt)
                if result is not None:
                    return result

                last_error = requests.HTTPError(f"HTTP {resp.status_code}")

            except requests.RequestException as exc:
                self._handle_exception(exc, start, payload, attempt)
                last_error = exc

            if attempt < self.MAX_ATTEMPTS:
                wait: int = self.BACKOFF_SECONDS[attempt - 1]
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

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP session and release connections."""
        if hasattr(self, "_session") and self._session is not None:
            self._session.close()

    # ── TradeSignal adapter ────────────────────────────────────────────────────

    async def fire(self, signal: Any) -> dict[str, Any]:
        """
        Translate a TradeSignal into the Quotex payload and send it.

        This is the interface live.py calls — it accepts a TradeSignal
        dataclass and builds the exact payload the Quotex bot expects:
            {"side": "buy"|"sell", "symbol": "EURUSD_otc", "key": "<key>"}

        Direction translation:
            CALL -> "buy"
            PUT  -> "sell"

        Symbol resolution:
            Uses ``Config.quotex_symbols`` exclusively. If the signal's
            symbol is not found in the mapping, raises ValueError — no
            silent fallback to a guessed symbol.

        Args:
            signal: TradeSignal instance from trading/signals.py.
                    Must have .direction, .symbol, and .expiry_key fields.

        Returns:
            Result dict from send_async() with success, status_code,
            latency_ms, attempts keys.

        Raises:
            WebhookError: If all retry attempts are exhausted.
            ValueError:   If direction is not CALL/PUT, or symbol is
                          not in the configured pairs mapping.
        """
        from core.config import get_settings

        settings = get_settings()

        direction: str = signal.direction.upper()
        if direction not in ("CALL", "PUT"):
            raise ValueError(
                f"WebhookSender.fire() received invalid direction: "
                f"'{direction}'. Only CALL and PUT are supported."
            )

        side: str = "buy" if direction == "CALL" else "sell"

        # Delegate symbol resolution entirely to Config.quotex_symbols.
        # This respects explicit OTC_PAIRS if set, or auto-translates.
        symbol_map: dict[str, str] = settings.quotex_symbols
        if signal.symbol not in symbol_map:
            raise ValueError(
                f"Signal symbol '{signal.symbol}' is not in the configured "
                f"pairs. Known symbols: {list(symbol_map.keys())}. "
                f"Check PAIRS and OTC_PAIRS in your .env."
            )
        quotex_symbol: str = symbol_map[signal.symbol]

        payload: dict[str, Any] = {
            "side": side,
            "symbol": quotex_symbol,
            "key": settings.webhook_key,
        }

        logger.info(
            {
                "event": "webhook_fire",
                "symbol": signal.symbol,
                "quotex_symbol": quotex_symbol,
                "side": side,
                "confidence": round(signal.confidence, 4),
                "expiry": signal.expiry_key,
                "model": signal.model_name,
            }
        )

        return await self.send_async(payload)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _sign(self, body: bytes) -> str:
        """
        Compute HMAC-SHA256 signature over raw body bytes.

        Args:
            body: Raw request body bytes.

        Returns:
            ``"sha256=<hex>"`` string for the ``X-Signature`` header.
        """
        sig: str = hmac.new(
            self._secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return f"sha256={sig}"

    def _log_attempt(
        self,
        payload: dict[str, Any],
        attempt: int,
        status: int | None,
        latency_ms: float,
        success: bool = False,
        error: Exception | None = None,
    ) -> None:
        """
        Log a structured record for one send attempt.

        Args:
            payload: The payload that was sent.
            attempt: 1-indexed attempt number.
            status: HTTP status code, or ``None`` if the request failed
                before receiving a response.
            latency_ms: Round-trip latency in milliseconds.
            success: Whether the HTTP response was 2xx.
            error: Exception if the request failed, ``None`` otherwise.
        """
        record: dict[str, Any] = {
            "event": "webhook_attempt",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "attempt": attempt,
            "http_status": status,
            "latency_ms": round(latency_ms, 1),
            "success": success,
        }
        if error:
            record["error"] = str(error)
        logger.info(record)
