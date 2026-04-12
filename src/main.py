"""
main.py — Asyncio orchestrator for the Trader AI signal engine.

Task topology:
    stream_task      — Tick stream (Quotex OTC or Twelve Data)
    signal_task      — Feature engineering + model inference → webhook
    feedback_task    — Consumes Quotex trade results → updates orchestrator
    health_task      — System heartbeat every 60 s
    backfill_task    — One-shot historical data download at startup
    telegram_task    — Telegram bot long-polling for commands
    report_task      — Scheduled Telegram/Discord reports (every 60 min)
    dashboard_task   — FastAPI dashboard server

Each long-running task is wrapped in :func:`supervised` which catches
exceptions, logs them, sleeps 5 s, and restarts.  One-shot tasks use
:func:`_safe` which logs failures without restarting.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import queue
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine

import numpy as np
import pandas as pd
import psutil

# ── JSON structured logging ────────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        msg: str = record.getMessage()
        try:
            payload: dict[str, Any] = (
                json.loads(msg) if msg.startswith("{") else {"message": msg}
            )
        except (json.JSONDecodeError, ValueError):
            payload = {"message": msg}

        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            **payload,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _configure_logging(level: str = "INFO") -> None:
    """
    Configure root logger with JSON formatter and suppress noisy SDK logs.

    Args:
        level: Python logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    # Suppress noisy Azure SDK HTTP transport logs
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure").setLevel(logging.WARNING)


# ── Import application modules ─────────────────────────────────────────────────
# (after logging config so imports can log safely)

from config import load_config  # noqa: E402
from storage import StorageManager  # noqa: E402
from twelveticks_stream import TwelveDataStream  # noqa: E402
from quotex_stream import QuotexStream  # noqa: E402
from backfill import backfill_all, check_data_coverage  # noqa: E402
from features import compute_features, resample_to_1min  # noqa: E402
from model import MartingaleTracker, ModelManager, SEQ_LEN  # noqa: E402
from signals import create_orchestrator, SignalOrchestrator  # noqa: E402
from webhook import WebhookSender, WebhookError  # noqa: E402
from quotex_reader import QuotexReader, run_quotex_reader  # noqa: E402
from reporter import DiscordReporter, TelegramBot, scheduled_report_loop  # noqa: E402
from dashboard import run_dashboard, status_store  # noqa: E402
from log_storage import BlobLogHandler  # noqa: E402

logger = logging.getLogger("main")

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_SAVE_PATH: str = "/app/models"
HEALTH_LOG_INTERVAL: int = 10  # seconds between health snapshots
SIGNAL_POLL_MS: float = 0.02  # 20 ms between tick queue polls
MAX_RECENT_TICKS: int = 15_000  # ~25 min at ~10 ticks/sec

# FIX G: Indicator Convergence Guard (The "Warm-up" Fix).
# OLD: MIN_BARS_FOR_FEATURES = 30
#   RSI and EMA use smoothing that takes longer to "settle."  30 bars of
#   history results in different indicator values than the 2 years used
#   in training, causing "Value Drift."
# NEW: Strict 80-bar minimum before inference.
MIN_TICKS_FOR_FEATURES: int = 120  # ~2 min of ticks
MIN_BARS_FOR_FEATURES: int = 80  # minimum 1-min bars for converged indicators

TRAINING_RETRY_SECONDS: int = 120  # re-check coverage every 2 min
SUPERVISOR_RESTART_DELAY: int = 5  # seconds before supervised restart


# ── Dashboard helper ──────────────────────────────────────────────────────────


def _push_dashboard(
    orchestrator: SignalOrchestrator,
    config: Any,
    **extra: Any,
) -> None:
    """
    Push the latest session + engine state to the dashboard status store.

    Called after every significant event so the dashboard never shows
    stale values for wins, losses, streak, profit, etc.

    Args:
        orchestrator: The live orchestrator whose :meth:`get_status`
            provides the canonical session state.
        config: Application config (for confidence_threshold, etc.).
        **extra: Additional keys to merge (e.g. ``last_event``, ``stream``).
    """
    status: dict[str, Any] = orchestrator.get_status()
    update: dict[str, Any] = {
        "stopped": status.get("stopped", False),
        "martingale_streak": status.get("martingale_streak", 0),
        "confidence_threshold": config.confidence_threshold,
        "pending_signals": status.get("pending_signals", 0),
        "session": status.get("session", {}),
    }
    update.update(extra)
    status_store.update(update)


# ── Task supervision ───────────────────────────────────────────────────────────


async def _safe(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    name: str,
) -> None:
    """
    Run a one-shot coroutine factory, logging failures without restarting.

    Use for fire-and-forget tasks like backfill and initial training that
    should run once and not restart on failure.

    Args:
        coro_factory: A callable returning a coroutine (not a bare coroutine
            — allows the supervisor to invoke it fresh each time).
        name: Human-readable task name for structured logs.
    """
    try:
        await coro_factory()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(
            {"event": "task_failed", "task": name, "error": str(exc)},
            exc_info=True,
        )


async def supervised(
    name: str,
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
) -> None:
    """
    Wrap a coroutine factory in an infinite restart-on-error loop.

    On exception: logs the error, sleeps :data:`SUPERVISOR_RESTART_DELAY`
    seconds, then re-invokes the factory.  Exits cleanly on
    :class:`asyncio.CancelledError`.

    Args:
        name: Human-readable task name for structured logs.
        coro_factory: A callable returning a coroutine to supervise.
    """
    while True:
        try:
            logger.info({"event": "task_start", "task": name})
            await coro_factory()
        except asyncio.CancelledError:
            logger.info({"event": "task_cancelled", "task": name})
            break
        except Exception as exc:
            logger.error(
                {"event": "task_crashed", "task": name, "error": str(exc)},
                exc_info=True,
            )
            await asyncio.sleep(SUPERVISOR_RESTART_DELAY)
            logger.info({"event": "task_restarting", "task": name})


# ── Individual task implementations ───────────────────────────────────────────


async def stream_task(stream: TwelveDataStream | QuotexStream) -> None:
    """
    Start the tick stream background thread and keep this task alive.

    The actual streaming runs in the stream's daemon thread.  This
    coroutine just prevents the :func:`supervised` wrapper from
    restarting it instantly by sleeping until the stop event is set.

    Args:
        stream: An active :class:`TwelveDataStream` or :class:`QuotexStream`.
    """
    stream.start()
    while not stream._stop_event.is_set():
        await asyncio.sleep(5)


async def signal_task(
    stream: TwelveDataStream | QuotexStream,
    model_manager: ModelManager,
    orchestrator: SignalOrchestrator,
    webhook_sender: WebhookSender,
    storage: StorageManager,
    config: Any,
    quotex_reader: QuotexReader | None = None,
) -> None:
    """
    Main signal generation loop — evaluates on every completed 1-minute bar.

    Flow per pair:
        1. Accumulate ticks in a rolling in-memory window.
        2. Detect minute-boundary crossing (new bar closed).
        3. Compute features from recent ticks.
        4. Run model inference (always — logs below-threshold results).
        5. Pass prediction through :meth:`SignalOrchestrator.try_signal`.
        6. If gate passes: fire webhook, register pending with reader.
        7. Trigger non-blocking retrain when enough new results exist.

    Args:
        stream: Active tick stream providing :attr:`tick_queue`.
        model_manager: Loaded :class:`ModelManager` with trained models.
        orchestrator: :class:`SignalOrchestrator` gating signals.
        webhook_sender: :class:`WebhookSender` for HTTP delivery.
        storage: :class:`StorageManager` for reading historical ticks.
        config: Application config with pairs, expiry, thresholds.
        quotex_reader: Optional :class:`QuotexReader` for result matching.
    """
    recent_ticks: dict[str, list[dict[str, Any]]] = {p: [] for p in config.pairs}
    last_bar_minute: dict[str, int] = {p: -1 for p in config.pairs}

    # FIX F: Rolling History Buffer (The "Bridge" Fix).
    # OLD: Extract one row → predict.  Only provided the current snapshot,
    #   preventing sequence models from working as intended.
    # NEW: Maintain a rolling window of feature rows per pair so that every
    #   inference call has a full 30-minute "context" ready for LSTM /
    #   Transformer.
    feature_history: dict[str, list[np.ndarray]] = {p: [] for p in config.pairs}

    logger.info(
        {
            "event": "signal_task_started",
            "pairs": list(recent_ticks.keys()),
            "min_ticks": MIN_TICKS_FOR_FEATURES,
            "min_bars": MIN_BARS_FOR_FEATURES,
        }
    )
    while True:
        # ── Drain tick queue ───────────────────────────────────────────────
        try:
            tick = stream.tick_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(SIGNAL_POLL_MS)
            continue

        pair: str = tick.pair
        if pair not in config.pairs:
            continue

        # Append to rolling window, trim to budget
        recent_ticks[pair].append(tick.to_dict())
        if len(recent_ticks[pair]) > MAX_RECENT_TICKS:
            recent_ticks[pair] = recent_ticks[pair][-MAX_RECENT_TICKS:]

        # ── Detect new 1-min bar ───────────────────────────────────────────
        tick_minute: int = tick.timestamp.minute
        if tick_minute == last_bar_minute.get(pair, -1):
            continue
        last_bar_minute[pair] = tick_minute

        # ── New bar closed — full evaluation cycle ─────────────────────────
        was_active: bool = (
            orchestrator.get_status().get("session", {}).get("is_active", False)
        )
        orchestrator.ensure_session_active()
        now_active: bool = (
            orchestrator.get_status().get("session", {}).get("is_active", False)
        )
        if not was_active and now_active:
            _push_dashboard(
                orchestrator,
                config,
                last_event="session_started",
            )

        # Non-blocking retrain if enough new results since last training
        if model_manager.should_retrain(pair):
            _pair: str = pair  # capture for closure

            async def _retrain(p: str = _pair) -> None:
                try:
                    tick_df_full = storage.read_ticks(p, months=6)
                    tick_df_recent = storage.read_ticks(p, months=3)

                    if not tick_df_full.empty:
                        feature_df_full = compute_features(
                            tick_df_full, config.expiry_seconds
                        )
                        feature_df_recent = (
                            compute_features(tick_df_recent, config.expiry_seconds)
                            if not tick_df_recent.empty
                            else None
                        )

                        if not feature_df_full.empty:
                            await asyncio.get_running_loop().run_in_executor(
                                None,
                                lambda: model_manager.train(
                                    p,
                                    config.expiry_seconds,
                                    feature_df_full,
                                    feature_df_recent,
                                ),
                            )
                            model_manager.save(MODEL_SAVE_PATH, storage=storage)
                            logger.info({"event": "retrain_complete", "pair": p})
                            _push_dashboard(
                                orchestrator,
                                config,
                                last_event=f"retrained: {p}",
                            )
                except Exception as exc:
                    logger.warning(
                        {
                            "event": "retrain_failed",
                            "pair": p,
                            "error": str(exc),
                        }
                    )

            asyncio.create_task(_retrain())

        # ── Build feature row from recent in-memory ticks ──────────────────
        tick_df: pd.DataFrame = pd.DataFrame(recent_ticks[pair])
        if len(tick_df) < MIN_TICKS_FOR_FEATURES:
            logger.debug(
                {
                    "event": "signal_debug",
                    "pair": pair,
                    "stage": "too_few_ticks",
                    "count": len(tick_df),
                    "need": MIN_TICKS_FOR_FEATURES,
                }
            )
            continue

        try:
            bars = resample_to_1min(tick_df)
            if len(bars) < MIN_BARS_FOR_FEATURES:
                logger.debug(
                    {
                        "event": "signal_debug",
                        "pair": pair,
                        "stage": "too_few_bars",
                        "bars": len(bars),
                        "need": MIN_BARS_FOR_FEATURES,
                    }
                )
                continue
            feature_df: pd.DataFrame = compute_features(tick_df, config.expiry_seconds)
            if feature_df.empty:
                logger.debug(
                    {"event": "signal_debug", "pair": pair, "stage": "feature_df_empty"}
                )
                continue
            feature_row: pd.Series[Any] = feature_df.iloc[-1]
            del feature_df, tick_df
            gc.collect()
        except Exception as exc:
            logger.warning(
                {
                    "event": "feature_extraction_failed",
                    "pair": pair,
                    "error": str(exc),
                }
            )
            continue

        # ── FIX F: Update rolling history buffer ──────────────────────────
        from features import get_feature_columns

        _feature_cols: list[str] = get_feature_columns()
        available_cols: list[str] = [c for c in _feature_cols if c in feature_row.index]
        feature_history[pair].append(
            feature_row[available_cols].to_numpy(dtype=np.float64)
        )
        # Keep at least SEQ_LEN rows for sequence models
        if len(feature_history[pair]) > SEQ_LEN + 50:
            feature_history[pair] = feature_history[pair][-(SEQ_LEN + 50) :]

        # ── Model inference ────────────────────────────────────────────────
        # FIX F + FIX G: Build 3D history array only when we have enough
        # converged bars (MIN_BARS_FOR_FEATURES = 80).
        history_array: np.ndarray | None = None
        if len(feature_history[pair]) >= SEQ_LEN:
            history_array = np.array(feature_history[pair][-SEQ_LEN:])

        prediction: dict[str, Any] | None = model_manager.predict(
            pair,
            config.expiry_seconds,
            feature_row,
            feature_history=history_array,
        )
        if prediction is None:
            logger.debug(
                {"event": "signal_debug", "pair": pair, "stage": "prediction_none"}
            )
            continue
        logger.info(
            {
                "event": "signal_debug",
                "pair": pair,
                "stage": "prediction_result",
                "direction": prediction.get("direction"),
                "confidence": prediction.get("confidence"),
            }
        )
        prediction["otc"] = config.use_quotex_streaming

        # ── Signal gate ────────────────────────────────────────────────────
        payload: dict[str, Any] | None = orchestrator.try_signal(prediction)
        if payload is None:
            # Gate rejected — push confidence for visibility
            _push_dashboard(
                orchestrator,
                config,
                last_event=(
                    f"gate_rejected: {pair} "
                    f"conf={prediction.get('confidence', 0.0):.2f}"
                ),
            )
            logger.debug(
                {
                    "event": "signal_debug",
                    "pair": pair,
                    "stage": "gate_rejected",
                    "confidence": prediction.get("confidence"),
                }
            )
            continue

        # ── Fire webhook ───────────────────────────────────────────────────
        try:
            result: dict[str, Any] = webhook_sender.send(payload)
            logger.info(
                {
                    "event": "signal_sent",
                    "symbol": payload.get("symbol"),
                    "side": payload.get("side"),
                    "confidence": prediction.get("confidence"),
                    "http_status": result.get("status_code"),
                    "latency_ms": result.get("latency_ms"),
                }
            )

            # Register pending with quotex reader for result matching.
            # The orchestrator stores signal_id in pending_signals keyed by
            # the UUID it generated inside try_signal().  We look it up by
            # matching pair + symbol so the reader and orchestrator share
            # the same signal_id.
            if quotex_reader is not None:
                signal_id: str | None = None
                for sid, entry in orchestrator.pending_signals.items():
                    if entry.get("pair") == pair and entry.get("symbol") == payload.get(
                        "symbol"
                    ):
                        signal_id = sid
                        break

                if signal_id is None:
                    logger.warning({"event": "pending_signal_not_found", "pair": pair})
                else:
                    expiry_time: datetime = tick.timestamp + timedelta(
                        seconds=config.expiry_seconds
                    )
                    quotex_reader.register_pending(
                        signal_id,
                        {
                            "pair": pair,
                            "direction": prediction.get("direction", "UP"),
                            "signal_id": signal_id,
                            **payload,
                        },
                        expiry_time,
                    )

            # Push to dashboard with fresh session state
            _push_dashboard(
                orchestrator,
                config,
                last_event=(
                    f"signal_fired: {pair} "
                    f"{prediction.get('direction')} @ "
                    f"{prediction.get('confidence', 0.0):.2f}"
                ),
            )

        except WebhookError as exc:
            logger.error({"event": "webhook_failed", "error": str(exc), "pair": pair})
            _push_dashboard(
                orchestrator,
                config,
                last_event=f"webhook_failed: {pair}",
            )


async def feedback_task(
    quotex_reader: QuotexReader,
    orchestrator: SignalOrchestrator,
    model_manager: ModelManager,
    storage: StorageManager,
    config: Any,
) -> None:
    """
    Consume trade results from Quotex and feed them back.

    Polls :meth:`QuotexReader.get_result` in a loop.  For each result:
        1. Forward to :meth:`SignalOrchestrator.on_result` (martingale + session).
        2. Record in :class:`ModelManager` for retrain triggers.
        3. Persist to :class:`StorageManager`.
        4. Push to dashboard status store with **fresh session state**.

    If pyquotex is not installed, sleeps forever (no-op).

    Args:
        quotex_reader: Connected :class:`QuotexReader`.
        orchestrator: Signal orchestrator for result processing.
        model_manager: Model manager for retrain tracking.
        storage: Persistent storage for result history.
        config: Application config (for confidence_threshold, etc.).
    """
    from quotex_reader import QUOTEX_LIB_AVAILABLE

    if not QUOTEX_LIB_AVAILABLE:
        logger.info(
            {
                "event": "feedback_task_disabled",
                "reason": "pyquotex_not_installed",
            }
        )
        while True:
            await asyncio.sleep(3600)

    while True:
        result: dict[str, Any] | None = await quotex_reader.get_result(timeout=1.0)
        if result is None:
            # get_result already waited up to timeout — yield briefly before
            # retrying to avoid spinning at ~10 calls/s when idle.
            await asyncio.sleep(0.1)
            continue

        orchestrator.on_result(result)

        pair: str = result.get("pair", "")
        outcome: str = result.get("result", "unknown")
        payout: float = float(result.get("payout", 0.0))

        if pair:
            model_manager.record_result(pair)

        storage.append_result(result)

        # Push fresh session state immediately — don't wait for health tick
        _push_dashboard(
            orchestrator,
            config,
            last_event=f"{outcome}: {pair} ${payout:.2f}",
        )

        # Check if session target reached
        status: dict[str, Any] = orchestrator.get_status()
        session: dict[str, Any] = status.get("session", {})
        if session.get("target_reached"):
            _push_dashboard(
                orchestrator,
                config,
                last_event=(
                    f"target_reached: {session.get('wins', 0)} wins, "
                    f"${session.get('net_profit', 0.0):.2f} profit"
                ),
            )


async def health_task(
    stream: TwelveDataStream | QuotexStream,
    orchestrator: SignalOrchestrator,
    config: Any,
    quotex_reader: QuotexReader | None = None,
) -> None:
    """
    Log system health every :data:`HEALTH_LOG_INTERVAL` seconds.

    Reports uptime, tick count, session state, martingale streak,
    memory usage (RSS), CPU percent, and Quotex connection status.
    Also pushes live data to the dashboard status store as a
    **backup** — the primary push happens in :func:`feedback_task`
    after every result.

    Args:
        stream: Active tick stream for tick counter.
        orchestrator: Signal orchestrator for session/martingale state.
        config: Application config (for confidence_threshold, etc.).
        quotex_reader: Optional Quotex reader for connection health.
    """
    start_time: float = time.monotonic()

    while True:
        await asyncio.sleep(HEALTH_LOG_INTERVAL)

        uptime: int = int(time.monotonic() - start_time)
        status: dict[str, Any] = orchestrator.get_status()
        qhealth: dict[str, Any] = (
            quotex_reader.health()
            if quotex_reader
            else {"connected": False, "balance": 0.0}
        )

        # Memory and CPU metrics — cpu_percent(interval=1) blocks for 1 s;
        # run it in an executor so the event loop stays responsive.
        process = psutil.Process()
        memory_mb: float = process.memory_info().rss / 1024 / 1024
        cpu_percent: float = await asyncio.get_running_loop().run_in_executor(
            None, lambda: process.cpu_percent(interval=1)
        )

        logger.info(
            {
                "event": "health",
                "uptime_seconds": uptime,
                "ticks_received": stream.ticks_received,
                "session": status.get("session"),
                "martingale_streak": status.get("martingale_streak"),
                "confidence_threshold": config.confidence_threshold,
                "pending_signals": status.get("pending_signals"),
                "memory_usage_mb": round(memory_mb, 1),
                "memory_usage_gb": round(memory_mb / 1024, 2),
                "cpu_percent": cpu_percent,
            }
        )

        # Backup push — feedback_task is the primary pusher for session state.
        # Do NOT include quotex= here: _balance_monitor pushes balance every
        # 1.5 s directly to status_store.  Overwriting it here with a 10 s
        # snapshot would show a stale balance on the dashboard.
        _push_dashboard(
            orchestrator,
            config,
            stream={
                "connected": True,
                "ticks_received": stream.ticks_received,
            },
        )
        # Update connection status only — preserve balance already set by _balance_monitor
        if quotex_reader:
            connected = quotex_reader._connected
            with status_store._lock:
                existing_quotex = status_store._data.get("quotex", {})
                status_store._data["quotex"] = {
                    **existing_quotex,
                    "connected": connected,
                }


async def backfill_task(config: Any, storage: StorageManager) -> None:
    """
    One-shot historical data download at startup.

    Only downloads for :attr:`config.backfill_pairs`.  Skips pairs that
    already have >= 365 days of coverage.  Logs blob parquet file count
    when skipping (diagnostic for Azure storage verification).

    Args:
        config: Application config with ``backfill_pairs`` and
            ``backfill_years``.
        storage: :class:`StorageManager` for checking coverage and writing.
    """
    pairs: list[str] = config.backfill_pairs

    needs_backfill: bool = any(
        not check_data_coverage(pair, storage, min_days=365) for pair in pairs
    )

    if not needs_backfill:
        # Log blob parquet count for diagnostics
        parquet_count: int = 0
        try:
            from azure.storage.blob import BlobServiceClient

            conn_string: str = storage._client.connection_string
            logger.info(
                {
                    "event": "blob_conn_string_present",
                    "length": len(conn_string),
                }
            )

            client = BlobServiceClient.from_connection_string(conn_string)
            container_client = client.get_container_client(storage._container)
            blobs = list(container_client.list_blobs(name_starts_with="data/"))
            parquet_count = len([b for b in blobs if b.name.endswith(".parquet")])
            logger.info({"event": "blob_count_success", "count": parquet_count})
        except ImportError as exc:
            logger.error({"event": "blob_import_error", "error": str(exc)})
        except Exception as exc:
            logger.error(
                {
                    "event": "blob_count_failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

        logger.info(
            {
                "event": "backfill_skipped",
                "reason": "sufficient_data",
                "parquet_files": parquet_count,
            }
        )
        status_store.update({"last_event": "backfill_skipped: sufficient data"})
        return

    logger.info(
        {
            "event": "backfill_starting",
            "pairs": pairs,
            "years": config.backfill_years,
        }
    )
    status_store.update({"last_event": f"backfill_started: {', '.join(pairs)}"})

    await backfill_all(
        pairs,
        storage,
        years_back=config.backfill_years,
        api_key=config.twelvedata_api_key,
    )

    logger.info({"event": "backfill_done"})
    status_store.update({"last_event": "backfill_complete"})


async def initial_training(
    config: Any,
    storage: StorageManager,
    model_manager: ModelManager,
) -> None:
    """
    Load saved models or train fresh from historical data.

    Retries every :data:`TRAINING_RETRY_SECONDS` until all pairs have a
    trained model.  Backfill may still be running when this first executes,
    so the loop waits for sufficient coverage (>= 365 days per pair).

    After training, models are saved to :data:`MODEL_SAVE_PATH` and also
    persisted to Azure via :meth:`ModelManager.save`.

    Args:
        config: Application config with pairs and expiry.
        storage: :class:`StorageManager` for reading historical ticks.
        model_manager: :class:`ModelManager` to train and persist.
    """
    model_manager.load(MODEL_SAVE_PATH, storage=storage)

    while True:
        all_trained: bool = True

        for pair in config.pairs:
            expiry: int = config.expiry_seconds
            models_exist: bool = bool(model_manager._models.get(pair, {}).get(expiry))
            if models_exist:
                logger.info({"event": "model_already_trained", "pair": pair})
                continue

            # Read all available data (up to 60 months)
            tick_df_full: pd.DataFrame = storage.read_ticks(pair, months=60)
            tick_df_recent: pd.DataFrame = storage.read_ticks(pair, months=3)

            if tick_df_full.empty:
                logger.info(
                    {
                        "event": "training_waiting_for_data",
                        "pair": pair,
                        "retry_in_s": TRAINING_RETRY_SECONDS,
                    }
                )
                status_store.update(
                    {"last_event": (f"training_waiting: {pair} (no data)")}
                )
                all_trained = False
                continue

            # Check coverage span
            tick_df_full["timestamp"] = pd.to_datetime(
                tick_df_full["timestamp"], utc=True
            )
            span_days: int = (
                tick_df_full["timestamp"].max() - tick_df_full["timestamp"].min()
            ).days

            if span_days < 365:
                logger.info(
                    {
                        "event": "training_waiting_for_coverage",
                        "pair": pair,
                        "span_days": span_days,
                        "target_days": 365,
                        "retry_in_s": TRAINING_RETRY_SECONDS,
                    }
                )
                status_store.update(
                    {
                        "last_event": (
                            f"training_waiting: {pair} " f"({span_days}/365 days)"
                        )
                    }
                )
                all_trained = False
                continue

            feature_df_full: pd.DataFrame = compute_features(tick_df_full, expiry)
            feature_df_recent: pd.DataFrame = compute_features(tick_df_recent, expiry)

            if feature_df_full.empty or len(feature_df_full) < 200:
                logger.warning(
                    {
                        "event": "insufficient_features",
                        "pair": pair,
                        "bars": len(feature_df_full),
                    }
                )
                status_store.update(
                    {
                        "last_event": (
                            f"training_insufficient: {pair} "
                            f"({len(feature_df_full)} bars)"
                        )
                    }
                )
                all_trained = False
                continue

            logger.info(
                {
                    "event": "initial_training_start",
                    "pair": pair,
                    "expiry": expiry,
                    "bars": len(feature_df_full),
                    "span_days": span_days,
                }
            )
            status_store.update(
                {
                    "last_event": (
                        f"training_started: {pair} " f"({len(feature_df_full)} bars)"
                    )
                }
            )

            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda p=pair, e=expiry, df=feature_df_full, df_r=feature_df_recent: model_manager.train(
                    p, e, df, df_r
                ),
            )

            logger.info(
                {
                    "event": "initial_training_complete",
                    "pair": pair,
                    "bars": len(feature_df_full),
                }
            )
            status_store.update(
                {
                    "last_event": (
                        f"training_complete: {pair} " f"({len(feature_df_full)} bars)"
                    )
                }
            )

        model_manager.save(MODEL_SAVE_PATH, storage=storage)

        if all_trained:
            logger.info({"event": "all_models_trained"})
            status_store.update({"last_event": "all_models_trained"})
            return

        await asyncio.sleep(TRAINING_RETRY_SECONDS)


# ── Main entry point ───────────────────────────────────────────────────────────


async def main() -> None:
    """
    Application entry point — initialise all components and launch tasks.

    Startup sequence:
        1. Load config and configure logging.
        2. Initialise storage, martingale tracker, model manager, orchestrator.
        3. Select data source (Quotex OTC or Twelve Data).
        4. Connect Quotex reader (if credentials available).
        5. Launch supervised tasks (stream, signal, health, dashboard).
        6. Launch one-shot tasks (backfill, initial training).
        7. Launch Quotex-dependent tasks (reader, feedback) if available.
        8. Launch optional tasks (telegram, scheduled reports).
        9. Await all tasks; on shutdown: cancel tasks, disconnect, save models.
    """
    config = load_config()
    _configure_logging(config.log_level)

    logger.info(
        {
            "event": "startup",
            "pairs": config.pairs,
            "expiry_seconds": config.expiry_seconds,
            "confidence_threshold": config.confidence_threshold,
            "practice_mode": config.practice_mode,
            "daily_trade_target": config.daily_trade_target,
            "target_net_profit": config.target_net_profit,
        }
    )

    # ── Core components ─────────────────────────────────────────────────────
    storage: StorageManager = StorageManager(
        conn_string=config.azure_storage_conn,
        container_name=config.container_name,
        flush_size=config.tick_flush_size,
    )

    # Persistent blob logging (best-effort, non-fatal on failure)
    try:
        blob_handler = BlobLogHandler(
            conn_string=config.azure_storage_conn,
            container_name=config.container_name,
        )
        blob_handler.setFormatter(_JSONFormatter())
        logging.getLogger().addHandler(blob_handler)
        logger.info({"event": "persistent_logging_enabled"})
    except Exception as exc:
        logger.warning({"event": "persistent_logging_failed", "error": str(exc)})

    container_id: str = str(uuid.uuid4())[:8]
    logger.info({"event": "container_start", "container_id": container_id})

    martingale: MartingaleTracker = MartingaleTracker(
        base_threshold=config.confidence_threshold,
        max_streak=config.martingale_max_streak,
    )
    model_manager: ModelManager = ModelManager(
        config=config, martingale_tracker=martingale
    )
    orchestrator: SignalOrchestrator = create_orchestrator(
        config=config, martingale_tracker=martingale
    )

    # ── Data source selection ───────────────────────────────────────────────
    stream: TwelveDataStream | QuotexStream

    # Single shared Quotex client — Quotex only supports one active WebSocket
    # session per account. QuotexStream (tick data) and QuotexReader (result
    # reading) must share this client, not each create their own connection.
    shared_qx_client: Any = None

    if config.use_quotex_streaming and config.quotex_email:
        from pyquotex.stable_api import Quotex

        shared_qx_client = Quotex(
            config.quotex_email,
            config.quotex_password,
            lang="en",
        )
        stream = QuotexStream(
            client=shared_qx_client,
            pairs=config.pairs,
            storage=storage,
            flush_size=config.tick_flush_size,
            poll_interval=config.poll_interval,
        )
        logger.info({"event": "data_source_selected", "provider": "Quotex OTC"})
    else:
        stream = TwelveDataStream(
            api_key=config.twelvedata_api_key,
            pairs=config.pairs,
            storage=storage,
            flush_size=config.tick_flush_size,
        )
        logger.info({"event": "data_source_selected", "provider": "Twelve Data"})

    # ── Webhook, Quotex reader, reporters ───────────────────────────────────
    webhook_sender: WebhookSender = WebhookSender(
        url=config.webhook_url,
        secret=config.webhook_secret,
    )

    has_quotex_creds: bool = bool(config.quotex_email and config.quotex_password)

    # QuotexReader reuses shared_qx_client when Quotex streaming is active so
    # there is only ever one WebSocket session. When using Twelve Data for
    # ticks, shared_qx_client is None and QuotexReader self-connects.
    quotex_reader: QuotexReader = QuotexReader(
        email=config.quotex_email,
        password=config.quotex_password,
        practice_mode=config.practice_mode,
        client=shared_qx_client,
    )
    if has_quotex_creds:
        await quotex_reader.connect()

    discord: DiscordReporter = DiscordReporter(
        webhook_url=getattr(config, "discord_webhook_url", "")
    )

    telegram: TelegramBot | None = None
    if getattr(config, "telegram_token", "") and getattr(
        config, "telegram_chat_id", ""
    ):
        telegram = TelegramBot(
            token=config.telegram_token,
            chat_id=config.telegram_chat_id,
            orchestrator=orchestrator,
            discord_reporter=discord,
        )

    # Seed dashboard with config and initial state
    status_store.update(
        {
            "practice_mode": config.practice_mode,
            "confidence_threshold": config.confidence_threshold,
            "last_event": "starting_up",
        }
    )

    # ── Launch supervised tasks ─────────────────────────────────────────────
    tasks: list[asyncio.Task[Any]] = [
        asyncio.create_task(
            supervised(
                "dashboard",
                lambda: run_dashboard(port=config.dashboard_port),
            )
        ),
        asyncio.create_task(supervised("stream", lambda: stream_task(stream))),
        asyncio.create_task(
            supervised(
                "signal",
                lambda: signal_task(
                    stream,
                    model_manager,
                    orchestrator,
                    webhook_sender,
                    storage,
                    config,
                    quotex_reader if has_quotex_creds else None,
                ),
            )
        ),
        asyncio.create_task(
            supervised(
                "health",
                lambda: health_task(stream, orchestrator, config, quotex_reader),
            )
        ),
    ]

    # One-shot tasks (no restart on failure)
    tasks.append(
        asyncio.create_task(_safe(lambda: backfill_task(config, storage), "backfill"))
    )
    tasks.append(
        asyncio.create_task(
            _safe(
                lambda: initial_training(config, storage, model_manager),
                "initial_training",
            )
        )
    )

    # Quotex-dependent tasks (only when credentials available)
    if has_quotex_creds:
        tasks.append(
            asyncio.create_task(
                supervised(
                    "quotex_reader",
                    lambda: run_quotex_reader(quotex_reader),
                )
            )
        )
        tasks.append(
            asyncio.create_task(
                supervised(
                    "feedback",
                    lambda: feedback_task(
                        quotex_reader,
                        orchestrator,
                        model_manager,
                        storage,
                        config,
                    ),
                )
            )
        )

    # Optional Telegram tasks
    if telegram:
        tasks.append(
            asyncio.create_task(supervised("telegram", lambda: telegram.poll_loop()))
        )
        tasks.append(
            asyncio.create_task(
                supervised(
                    "scheduled_report",
                    lambda: scheduled_report_loop(telegram),
                )
            )
        )

    logger.info({"event": "all_tasks_started", "count": len(tasks)})

    # ── Run until shutdown ──────────────────────────────────────────────────
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info({"event": "shutdown_requested"})
    finally:
        for task in tasks:
            task.cancel()
        stream.stop()
        if has_quotex_creds:
            await quotex_reader.disconnect()
        model_manager.save(MODEL_SAVE_PATH, storage=storage)
        logger.info({"event": "shutdown_complete"})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
