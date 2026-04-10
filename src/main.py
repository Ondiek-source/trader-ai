"""
main.py — Asyncio orchestrator for the Trader AI signal engine.

Task topology:
    stream_task      — Tick stream (thread executor)
    signal_task      — Feature engineering + model inference → webhook
    result_task      — HTTP server receiving trade results from bot
    feedback_task    — Consumes results → updates orchestrator + storage
    report_task      — Scheduled Telegram/Discord reports (every 60 min)
    telegram_task    — Telegram bot long-polling for commands
    health_task      — System heartbeat every 60 s
    backfill_task    — One-shot historical data download at startup

Each task is wrapped in a supervisor that catches any exception,
logs it, sleeps 5 s, and restarts the task.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import sys
import time
import gc
import uuid
import psutil
from datetime import datetime, timedelta, timezone

import pandas as pd


# ── JSON structured logging ────────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        try:
            payload = json.loads(msg) if msg.startswith("{") else {"message": msg}
        except (json.JSONDecodeError, ValueError):
            payload = {"message": msg}

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            **payload,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    # Suppress noisy Azure SDK HTTP transport logs (404s from blob existence checks)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure").setLevel(logging.WARNING)


# ── Import application modules ─────────────────────────────────────────────────

from config import load_config
from storage import StorageManager
from stream import OANDAStream
from backfill import backfill_all, check_data_coverage
from features import compute_features, get_feature_columns, resample_to_1min
from model import MartingaleTracker, ModelManager
from signals import create_orchestrator
from webhook import WebhookSender, WebhookError
from quotex_reader import QuotexReader, run_quotex_reader
from reporter import DiscordReporter, TelegramBot, scheduled_report_loop
from dashboard import run_dashboard, status_store
from log_storage import BlobLogHandler

logger = logging.getLogger("main")

# ── Constants ──────────────────────────────────────────────────────────────────

SIGNAL_EVAL_TICK_INTERVAL = 50  # re-evaluate model every N ticks per pair
MODEL_SAVE_PATH = "/app/models"
HEALTH_LOG_INTERVAL = 60  # seconds


# ── Task supervisor ────────────────────────────────────────────────────────────


async def _safe(coro, name: str) -> None:
    """Run a one-shot coroutine, logging but never propagating exceptions."""
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(
            {"event": "task_failed", "task": name, "error": str(exc)}, exc_info=True
        )


async def supervised(name: str, coro_factory) -> None:
    """Wrap a coroutine factory in infinite restart-on-error loop."""
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
            await asyncio.sleep(5)
            logger.info({"event": "task_restarting", "task": name})


# ── Individual task implementations ───────────────────────────────────────────


async def stream_task(stream: OANDAStream) -> None:
    """
    Start the OANDA stream background thread, then hold this asyncio task alive
    until the stop_event is set. The actual streaming runs in stream's daemon thread;
    this coroutine just keeps the supervised() wrapper from restarting it instantly.
    """
    stream.start()
    # Keep the task alive — the stream runs in its own daemon thread
    while not stream._stop_event.is_set():
        await asyncio.sleep(5)


async def signal_task(
    stream: OANDAStream,
    model_manager: ModelManager,
    orchestrator,
    webhook_sender: WebhookSender,
    storage: StorageManager,
    config,
    quotex_reader: QuotexReader | None = None,
) -> None:
    """
    Main signal generation loop — evaluates on every completed 1-minute bar.

    Architecture:
    - Accumulates ticks per pair in a rolling window
    - Detects when a new 1-min bar has closed (minute boundary crossing)
    - On every new bar: compute features → run inference → gate → fire
    - Retrain in executor (non-blocking) when enough new results accumulate
    - Always analyzes, always predicts — suppression is handled by the gate
    """
    import pandas as pd

    recent_ticks: dict[str, list] = {p: [] for p in config.pairs}
    last_bar_minute: dict[str, int] = {p: -1 for p in config.pairs}
    MAX_RECENT_TICKS = 15_000  # ~25 min of ticks at ~10 ticks/sec

    while True:
        try:
            tick = stream.tick_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.02)  # 20ms poll — high-frequency
            continue

        pair = tick.pair
        if pair not in config.pairs:
            continue

        recent_ticks[pair].append(tick.to_dict())
        if len(recent_ticks[pair]) > MAX_RECENT_TICKS:
            recent_ticks[pair] = recent_ticks[pair][-MAX_RECENT_TICKS:]

        # Detect new 1-min bar boundary (evaluate once per bar, not per tick)
        tick_minute = tick.timestamp.minute
        if tick_minute == last_bar_minute.get(pair, -1):
            continue
        last_bar_minute[pair] = tick_minute

        # ── New bar closed — full evaluation cycle ─────────────────────────
        orchestrator.ensure_session_active()

        # Non-blocking retrain if enough new results since last training
        if model_manager.should_retrain(pair):

            async def _retrain(p: str = pair) -> None:
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
                except Exception as exc:
                    logger.warning(
                        {"event": "retrain_failed", "pair": p, "error": str(exc)}
                    )

            asyncio.create_task(_retrain())

        # Build feature row from recent in-memory ticks
        tick_df_recent = pd.DataFrame(recent_ticks[pair])
        if len(tick_df_recent) < 120:  # need ~2 min of ticks minimum
            continue

        try:
            bars = resample_to_1min(tick_df_recent)
            if len(bars) < 30:
                continue
            feature_df = compute_features(tick_df_recent, config.expiry_seconds)
            if feature_df.empty:
                continue
            feature_row = feature_df.iloc[-1]
            # Force garbage collection to free memory from temporary DataFrames
            gc.collect()
        except Exception as exc:
            logger.warning(
                {"event": "feature_extraction_failed", "pair": pair, "error": str(exc)}
            )
            continue

        # Predict — always runs, logs below-threshold results, only fires when gate passes
        prediction = model_manager.predict(pair, config.expiry_seconds, feature_row)
        if prediction is None:
            continue

        payload = orchestrator.try_signal(prediction)
        if payload is None:
            continue

        # Fire webhook
        try:
            result = webhook_sender.send(payload)
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
            # Register with quotex reader so result can be matched at expiry
            if quotex_reader is not None:
                signal_id = f"{pair}_{tick.timestamp.isoformat()}"
                expiry_time = tick.timestamp + timedelta(seconds=config.expiry_seconds)
                quotex_reader.register_pending(
                    signal_id,
                    {
                        "pair": pair,
                        "direction": prediction.get("direction", "UP"),
                        **payload,
                    },
                    expiry_time,
                )
            # After webhook sends successfully
            status_store.update(
                {
                    "last_event": f"signal_fired: {pair} {prediction.get('direction')} @ {prediction.get('confidence'):.2f}"
                }
            )

        except WebhookError as exc:
            logger.error({"event": "webhook_failed", "error": str(exc), "pair": pair})


async def feedback_task(
    quotex_reader: QuotexReader, orchestrator, model_manager, storage
) -> None:
    """Consume trade results and feed them back to orchestrator + storage."""
    while True:
        result = await quotex_reader.get_result(timeout=1.0)
        if result is None:
            continue

        orchestrator.on_result(result)
        pair = result.get("pair", "")
        outcome = result.get("result", "unknown")
        payout = result.get("payout", 0.0)
        if pair:
            model_manager.record_result(pair)
        storage.append_result(result)
        status_store.update({"last_event": f"{outcome}: {pair} ${payout:.2f}"})


async def health_task(
    stream: OANDAStream, orchestrator, quotex_reader: QuotexReader | None = None
) -> None:
    """Log system health every HEALTH_LOG_INTERVAL seconds."""
    start_time = time.monotonic()
    while True:
        await asyncio.sleep(HEALTH_LOG_INTERVAL)
        uptime = int(time.monotonic() - start_time)
        status = orchestrator.get_status()
        qhealth = (
            quotex_reader.health()
            if quotex_reader
            else {"connected": False, "balance": 0.0}
        )
        # ── Memory and CPU metrics ─────────────────────────────────────────
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=1)

        logger.info(
            {
                "event": "health",
                "uptime_seconds": uptime,
                "ticks_received": stream.ticks_received,
                "session": status.get("session"),
                "martingale_streak": status.get("martingale_streak"),
                "confidence_threshold": status.get("confidence_threshold"),
                "pending_signals": status.get("pending_signals"),
                "memory_usage_mb": round(memory_mb, 1),
                "memory_usage_gb": round(memory_mb / 1024, 2),
                "cpu_percent": cpu_percent,
            }
        )

        # Push live data to dashboard
        status_store.update(
            {
                "stopped": status.get("stopped", False),
                "martingale_streak": status.get("martingale_streak", 0),
                "confidence_threshold": status.get("confidence_threshold", 0.65),
                "pending_signals": status.get("pending_signals", 0),
                "session": status.get("session", {}),
                "stream": {"connected": True, "ticks_received": stream.ticks_received},
                "quotex": qhealth,
            }
        )


async def backfill_task(config, storage) -> None:
    """One-shot backfill at startup if less than 1 year of data exists."""
    needs_backfill = any(
        not check_data_coverage(pair, storage, min_days=365) for pair in config.pairs
    )
    if not needs_backfill:
        # Count parquet files for logging (optional but helpful)
        try:
            from azure.storage.blob import BlobServiceClient

            conn_string = storage._client.connection_string
            logger.info(
                {"event": "blob_conn_string_present", "length": len(conn_string)}
            )

            client = BlobServiceClient.from_connection_string(conn_string)
            container_client = client.get_container_client(storage._container)

            # List blobs with timeout
            import sys

            sys.stderr.write("Listing blobs...\n")

            blobs = list(container_client.list_blobs(name_starts_with="data/"))
            parquet_count = len([b for b in blobs if b.name.endswith(".parquet")])
            logger.info(
                {
                    "event": "blob_count_success",
                    "count": parquet_count,
                }
            )
        except ImportError as e:
            logger.error(
                {
                    "event": "blob_import_error",
                    "error": str(e),
                }
            )
            parquet_count = 0
        except Exception as e:
            logger.error(
                {
                    "event": "blob_count_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            parquet_count = 0

        logger.info(
            {
                "event": "backfill_skipped",
                "reason": "sufficient_data_5yr",
                "parquet_files": parquet_count,
            }
        )
        return

    logger.info(
        {
            "event": "backfill_starting",
            "pairs": config.pairs,
            "years": config.backfill_years,
        }
    )
    await backfill_all(config.pairs, storage, years_back=config.backfill_years)
    logger.info({"event": "backfill_done"})


# ── Startup: train models from historical data ────────────────────────────────


async def initial_training(config, storage, model_manager) -> None:
    """
    Load saved models or train fresh from historical data.
    Retries every 5 minutes until all pairs have a trained model
    (backfill may still be running when this first executes).
    """
    model_manager.load(MODEL_SAVE_PATH, storage=storage)

    # Wait until backfill has written sufficient data before training.
    # Check every 2 minutes. Train once coverage reaches ≥ 365 days per pair.
    while True:
        all_trained = True

        for pair in config.pairs:
            expiry = config.expiry_seconds
            models_exist = bool(model_manager._models.get(pair, {}).get(expiry))
            if models_exist:
                logger.info({"event": "model_already_trained", "pair": pair})
                continue

            # Read all available data (up to 60 months)
            # tick_df = storage.read_ticks(pair, months=60)
            tick_df_full = storage.read_ticks(pair, months=60)  # For tree models
            tick_df_recent = storage.read_ticks(pair, months=3)  # For LSTM/Transformer

            if tick_df_full.empty:
                logger.info(
                    {
                        "event": "training_waiting_for_data",
                        "pair": pair,
                        "retry_in_s": 120,
                    }
                )
                all_trained = False
                continue

            # Check coverage span
            tick_df_full["timestamp"] = pd.to_datetime(
                tick_df_full["timestamp"], utc=True
            )
            span_days = (
                tick_df_full["timestamp"].max() - tick_df_full["timestamp"].min()
            ).days

            if span_days < 365:
                logger.info(
                    {
                        "event": "training_waiting_for_coverage",
                        "pair": pair,
                        "span_days": span_days,
                        "target_days": 365,
                        "retry_in_s": 120,
                    }
                )
                all_trained = False
                continue

            # feature_df = compute_features(tick_df, expiry)
            feature_df_full = compute_features(tick_df_full, expiry)
            feature_df_recent = compute_features(tick_df_recent, expiry)
            if feature_df_full.empty or len(feature_df_full) < 200:
                logger.warning(
                    {
                        "event": "insufficient_features",
                        "pair": pair,
                        "bars": len(feature_df_full),
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
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda p=pair, e=expiry, df=feature_df_full, df_recent=feature_df_recent: model_manager.train(
                    p, e, df, df_recent
                ),
            )
            logger.info(
                {
                    "event": "initial_training_complete",
                    "pair": pair,
                    "bars": len(feature_df_full),
                }
            )

        model_manager.save(MODEL_SAVE_PATH, storage=storage)

        if all_trained:
            logger.info({"event": "all_models_trained"})
            return  # signal_task retrain loop handles ongoing updates

        await asyncio.sleep(120)  # check again in 2 minutes


# ── Main entry point ───────────────────────────────────────────────────────────


async def main() -> None:
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

    # ── Initialise components ─────────────────────────────────────────────────
    storage = StorageManager(
        conn_string=config.azure_storage_conn,
        container_name=config.container_name,
        flush_size=config.tick_flush_size,
    )
    
    # ── Add persistent blob logging ──────────────────────────────────────────
    try:
        blob_handler = BlobLogHandler(
            conn_string=config.azure_storage_conn, container_name=config.container_name
        )
        blob_handler.setFormatter(_JSONFormatter())
        root_logger = logging.getLogger()
        root_logger.addHandler(blob_handler)
        logger.info({"event": "persistent_logging_enabled"})
    except Exception as e:
        logger.warning({"event": "persistent_logging_failed", "error": str(e)})

    # ── Log container start with unique ID ───────────────────────────────────
    container_id = str(uuid.uuid4())[:8]
    logger.info({"event": "container_start", "container_id": container_id})

    martingale = MartingaleTracker(
        base_threshold=config.confidence_threshold,
        max_streak=config.martingale_max_streak,
    )
    model_manager = ModelManager(config=config, martingale_tracker=martingale)
    orchestrator = create_orchestrator(config=config, martingale_tracker=martingale)

    stream = OANDAStream(
        api_key=config.twelvedata_api_key,
        pairs=config.pairs,
        storage=storage,
        flush_size=config.tick_flush_size,
    )

    webhook_sender = WebhookSender(url=config.webhook_url, secret=config.webhook_secret)
    quotex_reader = QuotexReader(
        email=getattr(config, "quotex_email", ""),
        password=getattr(config, "quotex_password", ""),
        practice_mode=config.practice_mode,
    )
    if getattr(config, "quotex_email", "") and getattr(config, "quotex_password", ""):
        await quotex_reader.connect()

    discord = DiscordReporter(webhook_url=getattr(config, "discord_webhook_url", ""))
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

    # ── Seed dashboard with config ────────────────────────────────────────────
    status_store.update(
        {
            "practice_mode": config.practice_mode,
            "confidence_threshold": config.confidence_threshold,
        }
    )

    # ── Launch all tasks immediately — dashboard is live on port 8080 ─────────
    # Backfill and initial training run as background tasks so nothing blocks.
    # signal_task waits for model_manager to have a trained model before firing.
    tasks = [
        asyncio.create_task(supervised("dashboard", lambda: run_dashboard(port=8080))),
        asyncio.create_task(_safe(backfill_task(config, storage), "backfill")),
        asyncio.create_task(
            _safe(initial_training(config, storage, model_manager), "initial_training")
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
                    quotex_reader,
                ),
            )
        ),
        asyncio.create_task(
            supervised("quotex_reader", lambda: run_quotex_reader(quotex_reader))
        ),
        asyncio.create_task(
            supervised(
                "feedback",
                lambda: feedback_task(
                    quotex_reader, orchestrator, model_manager, storage
                ),
            )
        ),
        asyncio.create_task(
            supervised(
                "health", lambda: health_task(stream, orchestrator, quotex_reader)
            )
        ),
    ]

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

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info({"event": "shutdown_requested"})
    finally:
        for task in tasks:
            task.cancel()
        stream.stop()
        await quotex_reader.disconnect()
        model_manager.save(MODEL_SAVE_PATH, storage=storage)
        storage.force_flush if hasattr(storage, "force_flush") else None
        logger.info({"event": "shutdown_complete"})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
