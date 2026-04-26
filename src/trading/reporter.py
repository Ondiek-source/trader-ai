"""
reporter.py — Unified reporter facade + Telegram bot + Discord reports.

Provides a single Reporter class that live.py depends on, wrapping
DiscordReporter and TelegramBot behind a consistent interface.

Telegram bot commands:
    /status    — Current session state (wins, losses, streak, profit)
    /stop      — Halt signal generation immediately
    /start     — Resume signal generation / start new session
    /report    — Send the current session HTML report now
    /threshold — Show current confidence threshold

Discord integration:
    - Sends HTML-formatted session summary to a Discord webhook URL.

Telegram integration:
    - Sends Markdown/HTML session summary to a Telegram chat.
    - Responds to commands via long-polling.

Both channels send at end of session window (or on demand via /report).
"""

from __future__ import annotations

import aiohttp
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from core.dashboard import status_store

logger = logging.getLogger(__name__)


# ── Orchestrator protocol ─────────────────────────────────────────────────────


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Interface the Telegram bot expects from the trading orchestrator."""

    def get_status(self) -> dict[str, Any]:
        """Return current session state and configuration."""
        ...

    def stop(self) -> str:
        """Halt signal generation; return a human-readable confirmation."""
        ...

    def resume(self) -> str:
        """Resume signal generation; return a human-readable confirmation."""
        ...

    def start_session(self) -> str:
        """Start a new session; return a human-readable confirmation."""
        ...


# ── HTML report generator ─────────────────────────────────────────────────────


def build_html_report(session_summary: dict[str, Any], status: dict[str, Any]) -> str:
    """
    Build an HTML session performance report.

    Suitable for Discord embed description (truncated) or Telegram HTML mode.

    Args:
        session_summary: Dict with keys like wins, losses, signals_fired,
            net_profit, daily_trade_target, daily_net_profit_target,
            elapsed_minutes, is_active, target_reached.
        status: Dict with keys like martingale_streak, confidence_threshold.

    Returns:
        A self-contained HTML string with inline styles.
    """
    now: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    wins: int = session_summary.get("wins", 0)
    losses: int = session_summary.get("losses", 0)
    fired: int = session_summary.get("signals_fired", 0)
    net: float = session_summary.get("net_profit", 0.0)
    target_wins: int = session_summary.get("daily_trade_target", 10)
    target_profit: float | None = session_summary.get("daily_net_profit_target")
    elapsed: int = session_summary.get("elapsed_minutes", 0)
    streak: int = status.get("martingale_streak", 0)
    threshold: float = status.get("confidence_threshold", 0.65)
    active: bool = session_summary.get("is_active", False)
    reached: bool = session_summary.get("target_reached", False)

    win_rate: float = round(wins / max(wins + losses, 1) * 100, 1)

    if active:
        status_badge: str = "ACTIVE"
    elif reached:
        status_badge = "TARGET REACHED"
    else:
        status_badge = "CLOSED"

    profit_color: str = "#00ff88" if net >= 0 else "#ff4444"
    streak_color: str = "#ff8800" if streak > 0 else "#00ff88"

    # Handle disabled profit target (None)
    profit_target_str: str = (
        f"${target_profit:.2f}" if target_profit is not None else "DISABLED"
    )

    html: str = f"""
<html><body style="font-family:monospace;background:#0d0d0d;color:#e0e0e0;padding:20px;">
<h2 style="color:#00ff88;">Trader AI - Daily Session Report</h2>
<p style="color:#888;">{now}</p>

<table style="width:100%;border-collapse:collapse;">
  <tr><td style="padding:6px;color:#aaa;">Session Status</td>
      <td style="padding:6px;color:#fff;font-weight:bold;">{status_badge}</td></tr>
  <tr><td style="padding:6px;color:#aaa;">Elapsed</td>
      <td style="padding:6px;">{elapsed} min</td></tr>
  <tr><td style="padding:6px;color:#aaa;">Signals Fired</td>
      <td style="padding:6px;">{fired}</td></tr>
  <tr style="background:#1a1a1a;">
    <td style="padding:6px;color:#00ff88;">Wins</td>
    <td style="padding:6px;color:#00ff88;font-weight:bold;">{wins} / {target_wins}</td></tr>
  <tr style="background:#1a1a1a;">
    <td style="padding:6px;color:#ff4444;">Losses</td>
    <td style="padding:6px;color:#ff4444;font-weight:bold;">{losses}</td></tr>
  <tr><td style="padding:6px;color:#aaa;">Win Rate</td>
      <td style="padding:6px;">{win_rate}%</td></tr>
  <tr style="background:#1a1a1a;">
    <td style="padding:6px;color:#aaa;">Net Profit</td>
    <td style="padding:6px;color:{profit_color};font-weight:bold;">
      ${net:.2f} / {profit_target_str}</td></tr>
  <tr><td style="padding:6px;color:#aaa;">Martingale Streak</td>
      <td style="padding:6px;color:{streak_color};">{streak} consecutive losses</td></tr>
  <tr><td style="padding:6px;color:#aaa;">Confidence Threshold</td>
      <td style="padding:6px;">{threshold:.0%}</td></tr>
</table>

<p style="margin-top:16px;color:#555;font-size:11px;">
  Generated by Trader AI - Practice mode only until validated
</p>
</body></html>
"""
    return html.strip()


def build_telegram_message(
    session_summary: dict[str, Any], status: dict[str, Any]
) -> str:
    """
    Build a compact Telegram message in HTML format.

    Uses HTML (not Markdown) to avoid issues with special characters
    inside values.

    Args:
        session_summary: Session stats dict (wins, losses, net_profit, etc.).
        status: Global status dict (martingale_streak, confidence_threshold).

    Returns:
        HTML-formatted string ready for Telegram sendMessage.
    """
    wins: int = session_summary.get("wins", 0)
    losses: int = session_summary.get("losses", 0)
    net: float = session_summary.get("net_profit", 0.0)
    target_wins: int = session_summary.get("daily_trade_target", 10)
    target_profit: float | None = session_summary.get("daily_net_profit_target")
    streak: int = status.get("martingale_streak", 0)
    threshold: float = status.get("confidence_threshold", 0.65)
    elapsed: int = session_summary.get("elapsed_minutes", 0)

    win_rate: float = round(wins / max(wins + losses, 1) * 100, 1)
    profit_icon: str = "🟢" if net >= 0 else "🔴"
    streak_icon: str = "🟠" if streak > 0 else "✅"

    profit_target_str: str = (
        f"<code>${target_profit:.2f}</code>"
        if target_profit is not None
        else "<code>DISABLED</code>"
    )

    return (
        f"<b>📊 Trader AI Session Report</b>\n"
        f"🕐 Elapsed: <code>{elapsed} min</code>\n\n"
        f"✅ Wins: <code>{wins}/{target_wins}</code>\n"
        f"❌ Losses: <code>{losses}</code>\n"
        f"📈 Win Rate: <code>{win_rate}%</code>\n"
        f"{profit_icon} Net Profit: <code>${net:.2f}</code> / "
        f"{profit_target_str}\n"
        f"{streak_icon} Streak: <code>{streak}</code> consecutive losses\n"
        f"🎯 Threshold: <code>{threshold:.0%}</code>\n"
    )


def build_discord_message(
    session_summary: dict[str, Any], status: dict[str, Any]
) -> str:
    """
    Build a plain-text session summary for Discord embeds.

    Discord embed descriptions support markdown, not HTML. This function
    produces a clean string with no HTML tags.

    Args:
        session_summary: Session stats dict.
        status: Global status dict.

    Returns:
        Plain-text string safe for Discord embed description.
    """
    wins: int = session_summary.get("wins", 0)
    losses: int = session_summary.get("losses", 0)
    net: float = session_summary.get("net_profit", 0.0)
    target_wins: int = session_summary.get("daily_trade_target", 10)
    target_profit: float | None = session_summary.get("daily_net_profit_target")
    streak: int = status.get("martingale_streak", 0)
    threshold: float = status.get("confidence_threshold", 0.65)
    elapsed: int = session_summary.get("elapsed_minutes", 0)

    win_rate: float = round(wins / max(wins + losses, 1) * 100, 1)
    profit_icon: str = "🟢" if net >= 0 else "🔴"
    streak_icon: str = "🟠" if streak > 0 else "✅"

    profit_target_str: str = (
        f"`${target_profit:.2f}`" if target_profit is not None else "`DISABLED`"
    )

    return (
        f"**📊 Trader AI Session Report**\n"
        f"🕐 Elapsed: `{elapsed} min`\n\n"
        f"✅ Wins: `{wins}/{target_wins}`\n"
        f"❌ Losses: `{losses}`\n"
        f"📈 Win Rate: `{win_rate}%`\n"
        f"{profit_icon} Net Profit: `${net:.2f}` / {profit_target_str}\n"
        f"{streak_icon} Streak: `{streak}` consecutive losses\n"
        f"🎯 Threshold: `{threshold:.0%}`\n"
    )


# ── Discord reporter ───────────────────────────────────────────────────────────


class DiscordReporter:
    """
    Sends session reports and alerts to a Discord webhook.

    Uses a shared aiohttp session for connection reuse. Call :meth:`close`
    when done to release resources.
    """

    ALERT_COLORS: dict[str, int] = {
        "info": 0x5865F2,
        "warning": 0xFF8800,
        "error": 0xFF4444,
    }
    MESSAGE_TTL_SECONDS: int = 20 * 3600  # auto-delete after 20 hours

    def __init__(self, webhook_url: str) -> None:
        self._url: str = webhook_url
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return a reusable aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=90,
                    connect=30,
                    sock_read=30,
                    sock_connect=30,
                )
            )
        return self._session

    async def _delete_after(self, message_id: str, delay: int) -> None:
        """Delete a webhook message after *delay* seconds."""
        await asyncio.sleep(delay)
        try:
            session = await self._get_session()
            async with session.delete(f"{self._url}/messages/{message_id}"):
                pass
        except Exception as exc:
            logger.debug({"event": "discord_delete_failed", "message_id": message_id, "error": str(exc)})

    async def send_report_async(
        self, session_summary: dict[str, Any], status: dict[str, Any]
    ) -> None:
        """Send a rich embed summarising the current trading session."""
        if not self._url:
            return

        text: str = build_discord_message(session_summary, status)
        net: float = session_summary.get("net_profit", 0)
        payload: dict[str, Any] = {
            "embeds": [
                {
                    "title": "📊 Trader AI — Daily Session Report",
                    "description": text,
                    "color": 0x00FF88 if net >= 0 else 0xFF4444,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "footer": {"text": "Trader AI Signal Engine"},
                }
            ]
        }

        try:
            session = await self._get_session()
            async with session.post(f"{self._url}?wait=true", json=payload) as resp:
                data = await resp.json()
            message_id = str(data.get("id", "")) if isinstance(data, dict) else ""
            if message_id:
                asyncio.create_task(self._delete_after(message_id, self.MESSAGE_TTL_SECONDS))
            logger.info({"event": "discord_report_sent"})
        except Exception as exc:
            logger.warning({"event": "discord_report_failed", "error": str(exc)})

    async def send_alert_async(self, message: str, level: str = "info") -> None:
        """Send a short alert embed to Discord."""
        if not self._url:
            return

        payload: dict[str, Any] = {
            "embeds": [
                {
                    "description": message,
                    "color": self.ALERT_COLORS.get(level, self.ALERT_COLORS["info"]),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }

        try:
            session = await self._get_session()
            async with session.post(f"{self._url}?wait=true", json=payload) as resp:
                data = await resp.json()
            message_id = str(data.get("id", "")) if isinstance(data, dict) else ""
            if message_id:
                asyncio.create_task(self._delete_after(message_id, self.MESSAGE_TTL_SECONDS))
        except Exception as exc:
            logger.warning({"event": "discord_alert_failed", "error": str(exc)})

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ── Telegram bot ───────────────────────────────────────────────────────────────


class TelegramBot:
    """
    Simple long-polling Telegram bot that exposes trading-orchestrator controls.

    Commands:
        /status    — session status
        /stop      — halt signal generation
        /start     — resume / start new session
        /report    — send report now
        /threshold — show current confidence threshold

    Reuses a single aiohttp.ClientSession for all API calls.
    """

    BASE_URL: str = "https://api.telegram.org/bot{token}/{method}"
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 6.0
    MESSAGE_TTL_SECONDS: int = 20 * 3600  # auto-delete after 20 hours

    def __init__(
        self,
        token: str,
        chat_id: str,
        orchestrator: OrchestratorProtocol,
        discord_reporter: DiscordReporter | None = None,
    ) -> None:
        self._token: str = token
        self._chat_id: str = str(chat_id)
        self._orchestrator: OrchestratorProtocol = orchestrator
        self._discord: DiscordReporter | None = discord_reporter
        self._offset: int = 0
        self._running: bool = False
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return a reusable aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=90,
                    connect=30,
                    sock_read=30,
                    sock_connect=30,
                )
            )
        return self._session

    # ── Low-level helpers ──────────────────────────────────────────────────

    async def _parse_retry_after(self, resp: aiohttp.ClientResponse) -> int:
        """Return the retry_after value from a 429 response body."""
        try:
            body: dict[str, Any] = await resp.json()
            return int(body.get("parameters", {}).get("retry_after", 5))
        except Exception:
            return 5

    def _log_attempt_error(self, exc: Exception, method: str, attempt: int) -> None:
        """Log a per-attempt request failure with the appropriate event key."""
        if isinstance(exc, asyncio.TimeoutError):
            logger.warning(
                {"event": "telegram_timeout", "method": method, "attempt": attempt}
            )
        elif isinstance(exc, aiohttp.ClientError):
            logger.warning(
                {
                    "event": "telegram_client_error",
                    "method": method,
                    "attempt": attempt,
                    "error": str(exc),
                }
            )
        else:
            logger.warning(
                {
                    "event": "telegram_request_failed",
                    "method": method,
                    "attempt": attempt,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    async def _api_async(
        self,
        method: str,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Call a Telegram Bot API method with retry on rate limit."""
        if max_retries is None:
            max_retries = self.MAX_RETRIES

        url: str = self.BASE_URL.format(token=self._token, method=method)

        for attempt in range(1, max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=kwargs) as resp:
                    if resp.status == 429:
                        retry_after: int = await self._parse_retry_after(resp)
                        logger.warning(
                            {
                                "event": "telegram_rate_limited",
                                "method": method,
                                "attempt": attempt,
                                "retry_after": retry_after,
                            }
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(retry_after)
                            continue
                        return None

                    data: dict[str, Any] = await resp.json()
                    if not data.get("ok"):
                        logger.warning(
                            {
                                "event": "telegram_api_error",
                                "method": method,
                                "response": data,
                            }
                        )
                    return data

            except Exception as exc:
                self._log_attempt_error(exc, method, attempt)

            if attempt < max_retries:
                await asyncio.sleep(self.RETRY_DELAY)

        return None

    # ── Auto-delete helper ─────────────────────────────────────────────────

    async def _delete_after(self, message_id: int, delay: int) -> None:
        """Delete a message after *delay* seconds."""
        await asyncio.sleep(delay)
        await self._api_async(
            "deleteMessage",
            max_retries=1,
            chat_id=self._chat_id,
            message_id=message_id,
        )

    # ── Public interface ───────────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "HTML") -> None:
        """Send a text message to the configured chat."""
        if not text:
            logger.warning({"event": "send_message_empty_text"})
            return
        data = await self._api_async(
            "sendMessage",
            chat_id=self._chat_id,
            text=text,
            parse_mode=parse_mode,
        )
        if data and data.get("ok"):
            message_id: int | None = data.get("result", {}).get("message_id")
            if message_id:
                asyncio.create_task(self._delete_after(message_id, self.MESSAGE_TTL_SECONDS))

    async def send_report(
        self,
        session_summary: dict[str, Any] | None = None,
        status: dict[str, Any] | None = None,
    ) -> None:
        """
        Build and send the current session report to Telegram (and Discord).

        If session_summary/status are not provided, pulls them from the
        orchestrator via get_status().

        Args:
            session_summary: Optional pre-built session stats dict.
            status: Optional pre-built global status dict.
        """
        resolved_status: dict[str, Any] = (
            status if status is not None else self._orchestrator.get_status()
        )
        resolved_session: dict[str, Any] = (
            session_summary
            if session_summary is not None
            else dict(resolved_status.get("session", {}))
        )

        msg: str = build_telegram_message(resolved_session, resolved_status)
        await self.send_message(msg)

        if self._discord:
            await self._discord.send_report_async(resolved_session, resolved_status)

    async def _process_updates(self, updates: list[dict[str, Any]]) -> None:
        """Dispatch each update from a getUpdates batch."""
        for update in updates:
            self._offset = update["update_id"] + 1
            msg: dict[str, Any] = update.get("message", {})
            if str(msg.get("chat", {}).get("id", "")) != self._chat_id:
                continue
            text: str = msg.get("text", "")
            if not text.startswith("/"):
                continue
            reply: str = await self._handle_command(text)
            if reply:
                await self.send_message(reply)

    async def poll_loop(self) -> None:
        """Long-polling loop — run as an asyncio.Task."""
        self._running = True
        logger.info({"event": "telegram_bot_started", "chat_id": self._chat_id})

        while self._running:
            try:
                data: dict[str, Any] | None = await self._api_async(
                    "getUpdates",
                    offset=self._offset,
                    timeout=25,
                    allowed_updates=["message"],
                )

                if not data or not data.get("ok"):
                    await asyncio.sleep(5)
                    continue

                await self._process_updates(data.get("result", []))

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning({"event": "telegram_poll_error", "error": str(exc)})
                await asyncio.sleep(10)

    async def stop(self) -> None:
        """Signal the polling loop to exit and close the session."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Command dispatch ───────────────────────────────────────────────────

    async def _handle_command(self, text: str) -> str:
        """Parse a slash-command and return a reply string."""
        cmd: str = text.strip().split()[0].lower().lstrip("/")

        if cmd == "status":
            status: dict[str, Any] = self._orchestrator.get_status()
            session: dict[str, Any] = status.get("session", {})
            return build_telegram_message(session, status)

        if cmd == "stop":
            result: str = self._orchestrator.stop()
            return result if result else "Signals halted."

        if cmd in ("start", "resume"):
            try:
                result = self._orchestrator.resume()
            except AttributeError:
                result = self._orchestrator.start_session()
            return result if result else "Session started."

        if cmd == "report":
            await self.send_report()
            return ""

        if cmd == "threshold":
            status = self._orchestrator.get_status()
            t: float = status.get("confidence_threshold", 0.65)
            streak: int = status.get("martingale_streak", 0)
            return f"Current threshold: <code>{t:.0%}</code> (streak: {streak} losses)"

        return (
            "Available commands:\n"
            "/status - session overview\n"
            "/stop - halt signals\n"
            "/start - resume / new session\n"
            "/report - send report now\n"
            "/threshold - show confidence threshold"
        )


# ── Unified Reporter Facade ───────────────────────────────────────────────────


class Reporter:
    """
    Unified facade that live.py depends on.

    Wraps DiscordReporter and TelegramBot behind a single ``notify()``
    interface. If neither channel is configured, all methods are silent
    no-ops — the engine degrades gracefully.

    Args:
        telegram_token: Bot token from BotFather (empty = disabled).
        telegram_chat_id: Authorized chat ID (empty = disabled).
        discord_webhook_url: Discord webhook URL (empty = disabled).
        orchestrator: Trading orchestrator for Telegram bot commands.
    """

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook_url: str = "",
        orchestrator: OrchestratorProtocol | None = None,
    ) -> None:
        self._discord: DiscordReporter | None = None
        self._telegram: TelegramBot | None = None
        self._telegram_task: asyncio.Task[None] | None = None
        self._journal: Any = None
        self._session: dict[str, Any] = {}
        self._started_at: str | None = None

        if discord_webhook_url:
            self._discord = DiscordReporter(webhook_url=discord_webhook_url)
            logger.info({"event": "REPORTER_DISCORD_ENABLED"})

        if telegram_token and telegram_chat_id and orchestrator:
            self._telegram = TelegramBot(
                token=telegram_token,
                chat_id=telegram_chat_id,
                orchestrator=orchestrator,
                discord_reporter=self._discord,
            )
            logger.info({"event": "REPORTER_TELEGRAM_ENABLED"})

        if not self._discord and not self._telegram:
            logger.info({"event": "REPORTER_NO_CHANNELS"})

    def set_journal(self, journal: Any) -> None:
        """Inject the Journal instance for Recent Trades queries."""
        self._journal = journal

    def push_dashboard(self, context: dict[str, Any]) -> None:
        """
        Build and push a complete dashboard snapshot.

        Called by LiveEngine after every pipeline event. Fetches all
        metrics (threshold, session, journal, connection) and pushes
        a single atomic update to status_store. No 10-second polling
        needed — every event triggers an immediate full refresh.

        Args:
            context: Dict from LiveEngine._build_context() containing
                event_type, symbol, is_running, ticks, signals, connected,
                balance, threshold_state, signal, result, session, started_at.
        """
        try:
            event_type = context.get("event_type", "tick")
            symbol = context.get("symbol", "")
            expiry_key = context.get("expiry_key", "")
            connected = context.get("connected", True)
            balance = context.get("balance", 0.0)
            is_running = context.get("is_running", False)
            ticks = context.get("ticks_processed", 0)
            signals = context.get("signals_fired", 0)
            threshold_state = context.get("threshold_state", {})
            signal = context.get("signal")
            result = context.get("result")
            # Elapsed time — uses reporter-owned _started_at
            elapsed_minutes: float = 0.0
            if self._started_at:
                started_at = datetime.fromisoformat(self._started_at)
                elapsed_minutes = (
                    datetime.now(timezone.utc) - started_at
                ).total_seconds() / 60.0

            # Session — uses reporter-owned _session
            session = dict(self._session)

            # Base update — always pushed regardless of event type
            update: dict[str, Any] = {
                "stream": {
                    "connected": connected,
                    "ticks_received": ticks,
                },
                "quotex": {
                    "connected": connected,
                    "balance": balance,
                },
                "martingale_streak": threshold_state.get("streak", 0),
                "martingale_max_streak": threshold_state.get("max_streak", 4),
                "confidence_threshold": threshold_state.get(
                    "effective_threshold", 0.58
                ),
                "base_confidence_threshold": threshold_state.get(
                    "base_threshold", 0.58
                ),
                "kill_switch_active": threshold_state.get("halted", False),
                "pending_signals": context.get("pending_signals", 0),
                "practice_mode": context.get("practice_mode", True),
            }

            # Session base
            session["elapsed_minutes"] = round(elapsed_minutes, 1)
            session["is_active"] = is_running
            session["signals_fired"] = signals

            # Event-specific updates
            last_event: str = ""

            if event_type == "signal" and signal is not None:
                last_event = (
                    f"{symbol} {signal.direction} @ "
                    f"{signal.confidence:.1%} [{expiry_key}]"
                )

            elif event_type == "skip" and signal is not None:
                threshold = threshold_state.get("effective_threshold", 0.58)
                last_event = (
                    f"{symbol} SKIP — {signal.confidence:.1%} < {threshold:.1%} threshold"
                )

            elif event_type == "cooldown" and signal is not None:
                last_event = (
                    f"{symbol} COOLDOWN ({signal.direction} "
                    f"{signal.confidence:.1%})"
                )

            elif event_type == "result" and result is not None:
                outcome = result.get("result", "")
                if outcome == "win":
                    session["wins"] = session.get("wins", 0) + 1
                    session["net_profit"] = session.get("net_profit", 0.0) + result.get(
                        "payout", 0.0
                    )
                elif outcome == "loss":
                    session["losses"] = session.get("losses", 0) + 1
                    session["net_profit"] = session.get("net_profit", 0.0) - result.get(
                        "stake", 0.0
                    )
                elif outcome == "draw":
                    session["draws"] = session.get("draws", 0) + 1
                pnl = result.get("payout", 0) or -result.get("stake", 0)
                last_event = f"{symbol} {outcome.upper()} ${pnl:+.2f}"

            elif event_type == "kill_switch":
                session["is_active"] = False
                streak = threshold_state.get("streak", 0)
                last_event = f"Kill switch: {symbol} — {streak} consecutive losses"

            elif event_type == "session_reset":
                session = {
                    "is_active": True,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "net_profit": 0.0,
                    "signals_fired": 0,
                    "elapsed_minutes": 0.0,
                }

                self._session = session

                self._started_at = datetime.now(timezone.utc).isoformat()
                update["started_at"] = self._started_at
                last_event = f"Session started for {symbol}"

            elif event_type == "fatal_error":
                errors = context.get("consecutive_errors", 0)
                session["is_active"] = False
                last_event = f"Engine fatal: {errors} consecutive errors — terminating"

            elif event_type == "engine_crash":
                session["is_active"] = False
                last_event = f"Engine crashed: {symbol}"

            elif event_type == "webhook_failed":
                session["is_active"] = False
                sig = context.get("signal")
                last_event = (
                    f"Webhook failed: {getattr(sig, 'symbol', '?')} — shutting down"
                )

            elif event_type == "journal_failed":
                session["is_active"] = False
                sig = context.get("signal")
                last_event = (
                    f"Journal failed: {getattr(sig, 'symbol', '?')} — shutting down"
                )

            if last_event:
                update["last_event"] = last_event

            update["session"] = session
            self._session = session

            # Recent trades from journal (field names match dashboard HTML)
            if self._journal is not None and event_type in ("result", "session_reset"):
                try:
                    trades_df = self._journal.get_trade_history(limit=10)
                    if trades_df is not None and not trades_df.empty:
                        recent: list[dict[str, Any]] = []
                        for _, row in trades_df.iterrows():
                            pnl = float(row.get("result", 0))
                            if pnl > 0:
                                result_label = "win"
                            elif pnl < 0:
                                result_label = "loss"
                            else:
                                result_label = "draw"
                            recent.append(
                                {
                                    "time": str(row.get("timestamp", ""))[:19],
                                    "symbol": str(row.get("symbol", "")),
                                    "side": str(row.get("side", "")),
                                    "result": result_label,
                                    "pnl": pnl,
                                    "duration": (
                                        str(int(row.get("duration_seconds", 0))) + "s"
                                    ),
                                }
                            )
                        update["recent_trades"] = list(reversed(recent))
                except Exception:
                    pass

            status_store.update(update)

            # Route last_event into the activity log's recent_events list so the
            # dashboard shows signals, results, and kill events — not just the
            # infrastructure events that pipeline.py adds via add_event() directly.
            if last_event:
                if event_type == "result" and result is not None:
                    _outcome = result.get("result", "")
                    _css = "win" if _outcome == "win" else "loss" if _outcome == "loss" else "info"
                elif event_type == "signal":
                    _css = "signal"
                elif event_type == "skip":
                    _css = "info"
                elif event_type in ("kill_switch", "fatal_error", "engine_crash", "webhook_failed", "journal_failed"):
                    _css = "kill"
                else:
                    _css = "info"
                status_store.add_event(last_event, _css)

        except Exception as exc:
            logger.warning({"event": "DASHBOARD_PUSH_FAILED", "error": str(exc)})

    # ── The interface live.py calls ────────────────────────────────────────

    async def notify(self, signal: Any) -> None:
        """
        Send a trade execution notification to all configured channels.

        Called by live.py._execute() for every executable signal.
        Failures are warning-only — a missed notification is preferable
        to a crashed engine.

        Args:
            signal: TradeSignal with .symbol, .direction, .confidence,
                    .expiry_key, .model_name, .timestamp fields.
        """
        symbol = getattr(signal, "symbol", "?")
        direction = getattr(signal, "direction", "?")
        confidence = getattr(signal, "confidence", 0)
        expiry_key = getattr(signal, "expiry_key", "?")
        model_name = getattr(signal, "model_name", "?")

        if self._discord:
            discord_msg = (
                f"⚡ **Trade Placed**\n"
                f"Symbol: `{symbol}`\n"
                f"Direction: `{direction}`\n"
                f"Confidence: `{confidence:.2%}`\n"
                f"Expiry: `{expiry_key}`\n"
                f"Model: `{model_name}`"
            )
            try:
                await self._discord.send_alert_async(discord_msg, level="info")
            except Exception as exc:
                logger.warning(
                    {"event": "reporter_discord_notify_failed", "error": str(exc)}
                )

        if self._telegram:
            telegram_msg = (
                f"⚡ <b>Trade Placed</b>\n"
                f"Symbol: <code>{symbol}</code>\n"
                f"Direction: <code>{direction}</code>\n"
                f"Confidence: <code>{confidence:.2%}</code>\n"
                f"Expiry: <code>{expiry_key}</code>\n"
                f"Model: <code>{model_name}</code>"
            )
            try:
                await self._telegram.send_message(telegram_msg, parse_mode="HTML")
            except Exception as exc:
                logger.warning(
                    {"event": "reporter_telegram_notify_failed", "error": str(exc)}
                )

    async def notify_result(self, result: dict[str, Any]) -> None:
        """
        Send a trade result notification to all configured channels.

        Called by live.py._consume_results() after every resolved trade.

        Args:
            result: Dict with keys result, pair, direction, payout, stake.
        """
        outcome: str = result.get("result", "?")
        pair: str = result.get("pair", "?")
        direction: str = result.get("direction", "?")
        payout: float = float(result.get("payout", 0))
        stake: float = float(result.get("stake", 0))
        pnl: float = payout if outcome == "win" else (-stake if outcome == "loss" else 0.0)

        icon = "✅" if outcome == "win" else "❌" if outcome == "loss" else "➖"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        level = "info" if outcome == "win" else "warning"

        if self._discord:
            discord_msg = (
                f"{icon} **Trade Result: {outcome.upper()}**\n"
                f"Pair: `{pair}`\n"
                f"Direction: `{direction}`\n"
                f"P&L: `{pnl_str}`"
            )
            try:
                await self._discord.send_alert_async(discord_msg, level=level)
            except Exception as exc:
                logger.warning(
                    {"event": "reporter_discord_result_failed", "error": str(exc)}
                )

        if self._telegram:
            telegram_msg = (
                f"{icon} <b>Trade Result: {outcome.upper()}</b>\n"
                f"Pair: <code>{pair}</code>\n"
                f"Direction: <code>{direction}</code>\n"
                f"P&amp;L: <code>{pnl_str}</code>"
            )
            try:
                await self._telegram.send_message(telegram_msg, parse_mode="HTML")
            except Exception as exc:
                logger.warning(
                    {"event": "reporter_telegram_result_failed", "error": str(exc)}
                )

    async def send_session_report(
        self,
        session_summary: dict[str, Any],
        status: dict[str, Any],
    ) -> None:
        """
        Send a session report to all configured channels.

        Called at end-of-session or on demand via Telegram /report command.

        Args:
            session_summary: Session stats dict.
            status: Global status dict.
        """
        if self._discord:
            try:
                await self._discord.send_report_async(session_summary, status)
            except Exception as exc:
                logger.warning(
                    {"event": "reporter_discord_report_failed", "error": str(exc)}
                )

        if self._telegram:
            try:
                await self._telegram.send_report(session_summary, status)
            except Exception as exc:
                logger.warning(
                    {"event": "reporter_telegram_report_failed", "error": str(exc)}
                )

    async def start_telegram_polling(self) -> None:
        """
        Start the Telegram bot long-polling loop as a background task.

        No-op if Telegram is not configured.
        """
        await asyncio.sleep(0)
        if self._telegram is None:
            return
        self._telegram_task = asyncio.create_task(self._telegram.poll_loop())
        logger.info({"event": "REPORTER_TELEGRAM_POLLING_STARTED"})

    async def close(self) -> None:
        """Shut down all channels and release resources."""
        if self._telegram:
            await self._telegram.stop()
        if self._telegram_task and not self._telegram_task.done():
            self._telegram_task.cancel()
            await asyncio.gather(self._telegram_task, return_exceptions=True)
        if self._discord:
            await self._discord.close()
        logger.info({"event": "REPORTER_CLOSED"})

    def __repr__(self) -> str:
        channels: list[str] = []
        if self._discord:
            channels.append("Discord")
        if self._telegram:
            channels.append("Telegram")
        return f"Reporter(channels={channels or ['NONE']})"
