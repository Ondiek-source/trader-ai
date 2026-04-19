"""
src/core/pipeline.py — The Orchestrator.

Role: Execute the strict five-stage boot sequence, verify data continuity,
load and inject model artifacts, launch LiveEngine tasks, and manage the
background retrain scheduler.

Architecture
------------
Pipeline is the single entry point for the entire trading system. main.py
constructs one Pipeline instance and calls await pipeline.run(). Everything
else — Storage, Historian, FeatureEngineer, ModelManager, LiveEngine,
retrain scheduling — is owned and orchestrated from here.

Boot sequence
-------------
The five stages execute in strict dependency order. Any failure in Stages
1-3 or Stage 5 raises PipelineError and halts the system immediately.
Stage 4 (model load) degrades gracefully — a missing model puts the engine
in SKIP-only mode rather than aborting boot.

    Stage 1: Storage Link         — verify Parquet I/O and Azure connectivity
    Stage 2: Historian Sync       — fill the time gap since last session
    Stage 3: Feature Warmup       — verify >=30 bars and smoke-test transform()
    Stage 4: Model Load           — cold-start pull + artifact injection
    Stage 5: Ignition             — construct and configure LiveEngines

Sync failure policy
-------------------
Stage 2 aborts on HistorianError. There is no degraded-start option.
A time-gap in M1 bars silently corrupts every rolling indicator (RSI,
MACD, ATR, BB) for up to 26 bars after the gap, producing confident but
wrong FeatureVectors. Trading on corrupted features is more dangerous
than not trading at all. Restart the pipeline when connectivity is restored.

Thread safety
-------------
All asyncio tasks share the event loop. Storage owns its own threading.Lock
for Parquet I/O. FeatureEngineer is stateless and safe to share. Each
LiveEngine owns its SignalGenerator — no cross-engine state is shared.
reload_model() on a LiveEngine is awaited from the retrain scheduler and
acquires the engine's internal asyncio.Lock before mutating state.

Public API
----------
    Pipeline()
    Pipeline.run()   -> coroutine, blocks until stop() or fatal error
    Pipeline.stop()  -> graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.reporter import Reporter
from core.config import get_settings
from core.exceptions import PipelineError, NotificationError
from core.dashboard import status_store, run_dashboard
from data.historian import Historian, HistorianError, get_historian
from data.storage import Storage, StorageError, get_storage
from engine.live import LiveEngine, LiveEngineError
from ml_engine.features import FeatureEngineerError, get_feature_engineer
from ml_engine.model_manager import ModelManager
from ml_engine.labeler import Labeler

from ml_engine.trainer import (
    DataShaper,
    XGBoostTrainer,
    CatBoostTrainer,
    RandomForestTrainer,
    GRUTrainer,
    TCNTrainer,
    LightGBMTrainer,
    LSTMTrainer,
)


# How many seconds before midnight (UTC) to fire the daily report.
_DAILY_REPORT_OFFSET_S: int = 600  # 23:50 UTC

logger = logging.getLogger(__name__)

# ── Module Constants ──────────────────────────────────────────────────────────

# Bars pre-fetched in Stage 3 warmup verification.
# Must be >= _MIN_BARS_REQUIRED in live.py (30). 100 gives safe headroom.
_WARMUP_BARS: int = 100

# How often the retrain scheduler wakes to check if the interval has elapsed.
# Actual retraining only fires when model_retrain_interval seconds have passed.
_RETRAIN_CHECK_INTERVAL_S: int = 60

# Canonical mapping of expiry key to seconds.
# Must stay in sync with _EXPIRY_SECONDS in labeler.py — the assert in
# labeler.py will catch any drift at import time.
_EXPIRY_KEY_MAP: dict[str, int] = {
    "1_MIN": 60,
    "5_MIN": 300,
    "15_MIN": 900,
}

# Inverse map: seconds -> expiry key. Derived once at module load.
_SECONDS_TO_EXPIRY_KEY: dict[int, str] = {v: k for k, v in _EXPIRY_KEY_MAP.items()}


# ── Pipeline ──────────────────────────────────────────────────────────────────


class Pipeline:
    """
    The Orchestrator: boot sequence, continuity handshake, model injection,
    live engine launch, and retrain scheduling.

    Instantiate once in main.py and call await pipeline.run(). Do not
    construct Storage, Historian, or LiveEngine outside this class —
    Pipeline owns those lifecycles.

    Attributes:
        _settings:              Validated Config singleton.
        _storage:               Stage 1 Storage instance shared across stages.
        _engines:               List of LiveEngine instances launched in Stage 5.
        _tasks:                 List of asyncio Tasks for stop() to cancel.
        _shutdown_requested:    True after stop() is called.
        _last_retrain_time:     UTC datetime of the last completed retrain cycle.
    """

    def __init__(self) -> None:
        """
        Initialise the Pipeline.

        Reads config via get_settings(). get_settings() calls sys.exit(1)
        internally on validation failure — no try/except is needed here.
        Component construction is deferred to run() so stage attribution
        in logs is accurate.
        """
        self._settings = get_settings()
        self._storage: Storage | None = None
        self._engines: list[LiveEngine] = []
        self._tasks: list[asyncio.Task] = []
        self._shutdown_requested: bool = False
        self._last_retrain_time: datetime = datetime.now(timezone.utc)
        # Reporter initialised in _run_task_group after engines are ready.
        self._reporter: Any = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Execute the five-stage boot sequence then launch the async task group.

        Registers SIGINT and SIGTERM handlers before Stage 1 so that a
        keyboard interrupt during boot triggers a clean shutdown rather than
        an abrupt kill. Each stage is a private method that logs its own
        diagnostic blocks and raises PipelineError on failure.

        Raises:
            PipelineError: If any of Stages 1, 2, 3, or 5 fail.
            Exception:     If an unexpected error bypasses stage guards.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        try:
            logger.info(
                {
                    "event": "PIPELINE_STARTUP_INITIATED",
                    "message": "Pipeline has started",
                }
            )
            self._storage = await self._stage_storage()
            await self._stage_historian_sync(self._storage)
            await self._stage_feature_warmup(self._storage)
            model_map = await self._stage_model_load()
            await self._stage_ignition(model_map)
            await self._run_task_group()

        except PipelineError:
            # Already has proper event - just propagate to main.py
            raise

        except Exception as exc:
            raise PipelineError(
                message=f"An Unhandled pipeline crash occured: {exc}", stage="unknown"
            ) from exc

        finally:
            logger.info({"event": "PIPELINE_EXIT", "symbols": self._settings.pairs})

    async def stop(self) -> None:
        """
        Initiate a graceful shutdown.

        Sets the shutdown flag, stops all LiveEngines (which sets their own
        is_running=False), and cancels all background asyncio tasks. Safe to
        call multiple times — subsequent calls are no-ops.
        """
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        logger.info(
            {
                "event": "PIPELINE_SHUTDOWN_INITIATED",
                "message": "Pipeline stopping gracefully",
            }
        )

        for engine in self._engines:
            engine.stop()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Push stopped state to dashboard
        try:
            snapshot = status_store.get()
            status_store.update(
                {
                    "stopped": True,
                    "session": {**snapshot.get("session", {}), "is_active": False},
                }
            )
            logger.info(
                {
                    "event": "DASHBOARD_UPDATED_SHUTDOWN",
                    "message": "Status store updated with is active flag false and stopped true.",
                }
            )
        except Exception:
            pass

        # Close reporter — cancels Telegram poll_loop task and closes aiohttp sessions
        if self._reporter is not None:
            try:
                await self._reporter.close()
            except Exception:
                pass

        logger.info(
            {
                "event": "SHUTDOWN_COMPLETE",
                "message": "All engines stopped, all tasks cancelled",
            }
        )

    # ── Stage 1: Storage Link ─────────────────────────────────────────────────

    async def _stage_storage(self) -> Storage:
        """
        Construct and verify the Storage custodian.

        Storage provisions the data directory hierarchy and (in CLOUD mode)
        establishes the Azure Blob connection. Both operations happen at
        construction time and fail immediately if they cannot complete.

        Returns:
            Storage: The initialised, verified Storage instance. Passed to
                        all downstream stages that require it.

        Raises:
            PipelineError: If Storage construction fails.
        """
        try:
            storage = get_storage()
            logger.info(
                {
                    "event": "STAGE_COMPLETE",
                    "stage": "storage",
                    "data_mode": self._settings.data_mode,
                    "data_dir": self._settings.data_dir,
                }
            )
            return storage

        except StorageError as exc:
            raise PipelineError(str(exc), stage="storage") from exc

    # ── Stage 2: Historian Sync ───────────────────────────────────────────────

    async def _stage_historian_sync(self, storage: Storage) -> dict[str, int]:
        """
        Fill the time gap since the last session for every backfill pair.

        Calls historian.backfill(symbol) directly per pair so that
        HistorianError is caught and treated as a fatal abort. This is
        intentionally different from historian.backfill_all() which catches
        HistorianError internally and returns 0 — making failure indistinct
        from an up-to-date result.

        A return value of 0 from backfill() means the data was already
        current (start_dt >= now_utc). This is not a failure.

        Args:
            storage: Stage 1 Storage instance (passed for audit context only —
                        the Historian constructs its own internal Storage via
                        get_historian() which shares the same singleton).

        Returns:
            dict[str, int]: Mapping of {symbol: bars_committed}.

        Raises:
            PipelineError: If backfill fails for any pair.
        """
        historian: Historian = get_historian()
        results: dict[str, int] = {}

        for symbol in self._settings.backfill_pairs:
            try:
                count: int = await historian.backfill(symbol)
                results[symbol] = count
                logger.info(
                    {
                        "event": "HISTORIAN_SYNC",
                        "symbol": symbol,
                        "bars_committed": count,
                    }
                )

            except HistorianError as exc:
                raise PipelineError(str(exc), stage="historian_sync") from exc

        logger.info(
            {
                "event": "STAGE_COMPLETE",
                "stage": "historian_sync",
                "results": results,
            }
        )
        return results

    # ── Stage 3: Feature Warmup ───────────────────────────────────────────────

    async def _stage_feature_warmup(self, storage: Storage) -> dict[str, int]:
        """
        Verify that each trading pair has enough bars for feature engineering.

        Checks that at least 30 M1 bars exist (the MACD slow span minimum)
        and runs a single transform() smoke-test to confirm the bar data
        schema is valid. This is a READ-ONLY check — no data is stored in
        memory. The LiveEngine loads bars fresh from storage on each tick.

        Args:
            storage: Stage 1 Storage instance.

        Returns:
            dict[str, int]: Mapping of {symbol: bar_count_available}.

        Raises:
            PipelineError: If any pair has insufficient bars or if the
                            smoke-test transform() raises.
        """
        engineer = get_feature_engineer()
        results: dict[str, int] = {}

        for symbol in self._settings.pairs:
            bars_df = storage.get_bars(symbol, timeframe="M1", max_rows=_WARMUP_BARS)
            bar_count: int = len(bars_df) if bars_df is not None else 0

            if bar_count < 30:
                raise PipelineError(
                    f"Insufficient bars for {symbol}: {bar_count}/30 required",
                    stage="feature_warmup",
                )

            try:
                # Smoke-test: transform the most recent 30 bars.
                # If the bar data schema is broken this raises immediately
                # rather than silently at the first live tick.
                assert bars_df is not None  # guaranteed: bar_count >= 30 above
                _ = engineer.transform(bars_df.tail(30))
                results[symbol] = bar_count

            except FeatureEngineerError as exc:
                raise PipelineError(str(exc), stage="feature_warmup") from exc

        logger.info(
            {
                "event": "STAGE_COMPLETE",
                "stage": "feature_warmup",
                "results": results,
            }
        )
        return results

    # ── Stage 4: Model Load ───────────────────────────────────────────────────

    async def _stage_model_load(self) -> dict[tuple[str, str], Any]:
        """
        Pull model artifacts from Blob and load them into memory.

        For each (symbol, expiry_key) pair: attempts a cold-start Blob pull,
        queries the registry for the best current-version artifact, and loads it.

        Returns:
            dict keyed by (symbol, expiry_key) -> (model_artifact, ModelRecord).
            Pairs with no available model are absent from the dict.

        Raises:
            PipelineError: If no model found for any symbol/expiry pair.
        """
        manager = ModelManager(storage_dir=self._settings.model_dir)
        model_map: dict[tuple[str, str], Any] = {}

        for symbol in self._settings.pairs:
            for expiry_key in _EXPIRY_KEY_MAP:

                # 4a: Cold-start Blob pull
                pulled = await manager.pull_from_blob(
                    symbol=symbol, expiry_key=expiry_key
                )
                if pulled:
                    logger.info(
                        {
                            "event": "MODEL_PULLED_TO_LOCAL_DISK",
                            "symbol": symbol,
                            "expiry_key": expiry_key,
                            "artifact": pulled,
                        }
                    )

                # 4b: Query registry for best artifact
                record = manager.get_best_model(symbol=symbol, expiry_key=expiry_key)
                if record is None:
                    raise PipelineError(
                        f"No model found for {symbol} {expiry_key}",
                        stage="model_load",
                    )

                # 4c: Skip PyTorch — requires architecture injection
                if record.is_pytorch:
                    # TODO
                    continue

                # 4d: Load artifact into memory
                try:
                    model = manager.load(record.artifact_path)
                    model_map[(symbol, expiry_key)] = (model, record)
                    logger.info(
                        {
                            "event": "MODEL_LOADED",
                            "symbol": symbol,
                            "expiry_key": expiry_key,
                            "model_name": record.model_name,
                            "auc": record.auc,
                        }
                    )

                except Exception as exc:
                    raise PipelineError(
                        f"Failed to load model artifact for {symbol} {expiry_key}: {exc}",
                        stage="model_load",
                    ) from exc

        logger.info(
            {
                "event": "STAGE_COMPLETE",
                "stage": "model_load",
                "models_loaded": len(model_map),
            }
        )
        return model_map

    # ── Stage 5: Ignition ─────────────────────────────────────────────────────

    async def _stage_ignition(
        self,
        model_map: dict[tuple[str, str], Any],
    ) -> list[LiveEngine]:
        """
        Construct LiveEngines and inject pre-loaded model artifacts.

        Derives the expiry_key from settings.expiry_seconds via
        _SECONDS_TO_EXPIRY_KEY. Constructs one LiveEngine per symbol.
        If a model was loaded in Stage 4, injects it directly via
        engine._signal_gen.inject_model() so the first tick fires
        with the pipeline-resolved artifact rather than whatever
        SignalGenerator.reload() found independently.

        Args:
            model_map: Output of _stage_model_load().

        Returns:
            list[LiveEngine]: All constructed engines. Also stored in
                self._engines for stop() and the retrain scheduler.

        Raises:
            PipelineError: If expiry_seconds does not map to a known key,
                            or if any LiveEngine construction fails.
        """
        expiry_seconds: int = self._settings.expiry_seconds
        expiry_key: str | None = _SECONDS_TO_EXPIRY_KEY.get(expiry_seconds)

        if expiry_key is None:
            raise PipelineError(
                f"EXPIRY_SECONDS={expiry_seconds} has no expiry_key mapping. "
                f"Valid values: {sorted(_SECONDS_TO_EXPIRY_KEY.keys())}",
                stage="ignition",
            )

        for symbol in self._settings.pairs:
            try:
                engine = await LiveEngine.create(symbol, expiry_key)

            except LiveEngineError as exc:
                raise PipelineError(str(exc), stage="ignition") from exc

            # Inject the pipeline-resolved model if available.
            # This overwrites whatever SignalGenerator.reload() loaded in its
            # own __init__ call, ensuring the freshest artifact is always used.
            if (symbol, expiry_key) in model_map:
                model, record = model_map[(symbol, expiry_key)]
                engine._signal_gen.inject_model(model, record)
                logger.info(
                    {
                        "event": "MODEL_INJECTED",
                        "symbol": symbol,
                        "expiry_key": expiry_key,
                        "model_name": record.model_name,
                    }
                )

            self._engines.append(engine)

        logger.info(
            {
                "event": "STAGE_COMPLETE",
                "stage": "ignition",
                "engines": len(self._engines),
                "symbols": [e.symbol for e in self._engines],
                "expiry": expiry_key,
            }
        )
        return self._engines

    # ── Task Group ────────────────────────────────────────────────────────────

    async def _run_task_group(self) -> None:
        """
        Launch all background asyncio tasks and await their completion.

        Creates one task per LiveEngine, one retrain scheduler task, and
        an optional TelegramBot polling task if credentials are configured.
        All tasks are stored in self._tasks so stop() can cancel them.

        Uses asyncio.gather(return_exceptions=True) so a single task crash
        does not cancel other tasks. Individual engine crashes are handled
        by _safe_engine_run() which logs but does not re-raise.
        """
        # ── Seed static config fields into the dashboard StatusStore ─────────
        # These never change at runtime so one push at task-group start is
        # sufficient. Dynamic fields (threshold, streak, session) are pushed
        # by live.py on each tick.

        status_store.update(
            {
                "practice_mode": self._settings.practice_mode,
                "base_confidence_threshold": self._settings.confidence_threshold,
                "confidence_threshold": self._settings.confidence_threshold,
                "martingale_max_streak": self._settings.martingale_max_streak,
                "session": {
                    "is_active": True,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "net_profit": 0.0,
                    "signals_fired": 0,
                    "elapsed_minutes": 0,
                },
            }
        )

        # ── Dashboard HTTP server ──────────────────────────────────────────

        self._tasks.append(
            asyncio.create_task(
                run_dashboard(self._settings.dashboard_port),
                name="dashboard",
            )
        )
        logger.info(
            {
                "event": "DASHBOARD_STARTED",
                "port": self._settings.dashboard_port,
            }
        )

        # Engine tasks
        for engine in self._engines:
            self._tasks.append(
                asyncio.create_task(
                    self._safe_engine_run(engine),
                    name=f"engine_{engine.symbol}_{engine.expiry_key}",
                )
            )

        # Retrain scheduler — receives the shared Storage instance
        assert self._storage is not None, (
            "Pipeline._storage must be initialised before _run_task_group(). "
            "Ensure _stage_storage() completed successfully."
        )
        self._tasks.append(
            asyncio.create_task(
                self._retrain_scheduler(self._storage),
                name="retrain_scheduler",
            )
        )

        # ── Reporter + Telegram bot ────────────────────────────────────────
        # Initialised here (not __init__) so engines exist before the bot
        # starts responding to /status commands.
        self._reporter = self._init_reporter()

        if self._reporter is not None:
            self._tasks.append(
                asyncio.create_task(
                    self._reporter.start_telegram_polling(),
                    name="telegram_bot",
                )
            )
            logger.info({"event": "TELEGRAM_BOT_STARTED"})

        # ── Background notification tasks ──────────────────────────────────
        self._tasks.append(
            asyncio.create_task(
                self._quotex_status_loop(),
                name="quotex_status",
            )
        )
        self._tasks.append(
            asyncio.create_task(
                self._daily_report_loop(),
                name="daily_report",
            )
        )

        logger.info(
            {
                "event": "TASK_GROUP_STARTED",
                "task_count": len(self._tasks),
            }
        )

        # Boot notification — fire-and-forget before entering gather.
        # Not a task: we want it sent synchronously before the main loop
        # blocks, but failure must never prevent the loop from starting.
        await self._send_boot_notification()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ── Reporter initialisation ───────────────────────────────────────────────

    def _init_reporter(self) -> Any:
        """
        Construct the pipeline-level Reporter and start Telegram polling.

        Pipeline acts as the orchestrator: get_status() reads status_store,
        stop() schedules graceful shutdown, resume/start_session are no-ops
        (a restart is required to start a new session).

        Returns:
            Reporter instance, or None if both channels are unconfigured.
        """
        try:

            # Thin adapter so Pipeline satisfies OrchestratorProtocol without
            # conflicting with the existing async Pipeline.stop() method.
            pipeline_ref = self

            class _Orchestrator:
                def get_status(self) -> dict[str, Any]:
                    return status_store.get()

                def stop(self) -> str:
                    asyncio.get_running_loop().create_task(pipeline_ref.stop())
                    return "Shutdown signal sent."

                def resume(self) -> str:
                    return "Restart the container to start a new session."

                def start_session(self) -> str:
                    return "Session is already running."

            return Reporter(
                telegram_token=self._settings.telegram_token,
                telegram_chat_id=self._settings.telegram_chat_id,
                discord_webhook_url=self._settings.discord_webhook_url,
                orchestrator=_Orchestrator(),
            )

        except Exception as exc:
            raise PipelineError(
                f"Reporter initialization failed: {exc}", stage="reporter_init"
            ) from exc

    # ── Boot notification ─────────────────────────────────────────────────────

    async def _send_boot_notification(self) -> None:
        """
        Send a 'system ready' alert to Discord and Telegram on every deploy.

        Called once after all tasks are created but before asyncio.gather()
        blocks.

        Raises:
            NotificationError: If Telegram or Discord notification fails.
        """
        if self._reporter is None:
            raise NotificationError(
                "Reporter not available - cannot send boot notification"
            )

        symbols = ", ".join(f"`{e.symbol}`" for e in self._engines)
        mode = "GAMING" if self._settings.practice_mode else "MONEY UP"
        mode_icon = "🎮" if self._settings.practice_mode else "🔥"
        message = (
            f"🚀 <b>Trader AI — System Ready</b>\n"
            f"{mode_icon} Mode: <code>{mode}</code>\n"
            f"💢 Symbols: {symbols}\n"
            f"⏱ Expiry: <code>{self._settings.expiry_seconds}s</code>\n"
            f"🌯 Base threshold: <code>{self._settings.confidence_threshold:.0%}</code>\n"
            f"🛡 Max streak: <code>{self._settings.martingale_max_streak}</code>\n"
            f"📡 Stream: <code>{'Quotex' if self._settings.use_quotex_streaming else 'TwelveData'}</code>"
        )

        # Telegram expects HTML; Discord gets a plain-text variant via
        # send_alert_async which handles its own formatting.
        if hasattr(self._reporter, "_telegram") and self._reporter._telegram:
            await self._reporter._telegram.send_message(message, parse_mode="HTML")
        if hasattr(self._reporter, "_discord") and self._reporter._discord:
            plain = (
                message.replace("<b>", "**")
                .replace("</b>", "**")
                .replace("<code>", "`")
                .replace("</code>", "`")
                .replace("<br>", "\n")
            )
            await self._reporter._discord.send_alert_async(plain, level="info")

    # ── Quotex status polling loop ────────────────────────────────────────────

    async def _quotex_status_loop(self) -> None:
        """
        Poll each engine's QuotexReader every 30 s and push live account
        data to the dashboard StatusStore.

        Pushes: connection status, account balance, session win/loss/draw
        counts (derived from get_history()), and pending trade count.

        Only runs when USE_QUOTEX_STREAMING=True and at least one engine
        has a connected QuotexReader stream. Silently skips non-Quotex
        streams and disconnected clients.
        """
        connection_lost_logged: bool = False
        connection_restored_logged: bool = False

        if not self._settings.use_quotex_streaming:
            logger.info({"event": "QUOTEX_STATUS_DISABLED"})
            return

        # Give QuotexReader time to connect before first poll.
        await asyncio.sleep(30)

        while not self._shutdown_requested:
            try:
                # Use the first engine's stream — all engines share the same
                # Quotex account so one poll is sufficient.
                stream = self._engines[0]._stream if self._engines else None
                if stream is None or not hasattr(stream, "_connected"):
                    await asyncio.sleep(30)
                    continue

                connected: bool = bool(getattr(stream, "_connected", False))

                # Log connection state changes to dashboard
                if not connected and not connection_lost_logged:
                    status_store.add_event("Quotex not connected", event_type="kill")
                    connection_lost_logged = True
                    connection_restored_logged = False
                elif connected and not connection_restored_logged:
                    status_store.add_event("Quotex connected", event_type="info")
                    connection_restored_logged = True
                    connection_lost_logged = False

                balance: float = 0.0
                pending: int = 0

                if connected:
                    health = stream.health()
                    balance = health.get("balance", 0.0)
                    pending = health.get("pending_signals", 0)

                snapshot = status_store.get()
                try:
                    started_at = datetime.fromisoformat(snapshot["started_at"])
                    elapsed_minutes = (
                        datetime.now(timezone.utc) - started_at
                    ).total_seconds() / 60
                except Exception as exc:
                    logger.error(
                        {
                            "event": "ELAPSED_TIME_CALC_ERROR",
                            "error": str(exc),
                            "snapshot_keys": list(snapshot.keys()),
                            "function": "_quotex_status_loop",
                        }
                    )
                    elapsed_minutes = 0

                status_store.update(
                    {
                        "quotex": {"connected": connected, "balance": balance},
                        "pending_signals": pending,
                        "session": {
                            **snapshot.get("session", {}),
                            "elapsed_minutes": round(elapsed_minutes, 1),
                        },
                    }
                )
            except Exception as exc:
                logger.error(
                    {
                        "event": "QUOTEX_STATUS_POLL_ERROR",
                        "error": str(exc),
                    }
                )
                status_store.add_event(
                    f"Quotex status poll error: {exc}", event_type="kill"
                )

            await asyncio.sleep(30)

    # ── Daily report loop ─────────────────────────────────────────────────────

    async def _daily_report_loop(self) -> None:
        """
        Send a session summary to Discord and Telegram every day at 23:50 UTC.

        Calculates seconds until the next 23:50 UTC, sleeps that long, fires
        the report, then repeats. Fires regardless of whether the session is
        active so you always get an end-of-day snapshot.

        _DAILY_REPORT_OFFSET_S controls how many seconds before midnight the
        report fires (default 600 = 23:50 UTC).
        """
        if self._reporter is None:
            logger.info(
                {
                    "event": "DAILY_REPORT_DISABLED",
                    "reason": "no reporter configured, exiting",
                }
            )
            return

        while not self._shutdown_requested:
            # ── Calculate seconds until next fire time (23:50 UTC) ────────
            now = datetime.now(timezone.utc)
            seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
            fire_at = 86400 - _DAILY_REPORT_OFFSET_S  # 85800 = 23:50:00 UTC
            sleep_for = fire_at - seconds_since_midnight
            if sleep_for <= 0:
                # Already past today's window — wait until tomorrow's
                sleep_for += 86400

            logger.info(
                {
                    "event": "DAILY_REPORT_SCHEDULED",
                    "next_report_minutes": round(sleep_for / 60, 1),
                }
            )

            # Sleep in 60-second chunks so shutdown is responsive.
            while sleep_for > 0 and not self._shutdown_requested:
                chunk = min(sleep_for, 60)
                await asyncio.sleep(chunk)
                sleep_for -= chunk

            if self._shutdown_requested:
                break

            # ── Build and send report ──────────────────────────────────────
            try:
                snapshot = status_store.get()
                session = snapshot.get("session", {})

                # Enrich session with config targets for the report builder.
                session_summary = {
                    **session,
                    "daily_trade_target": self._settings.daily_trade_target,
                    "daily_net_profit_target": self._settings.daily_net_profit_target,
                    "is_active": not self._shutdown_requested,
                    "target_reached": (
                        session.get("wins", 0) >= self._settings.daily_trade_target
                        or (
                            self._settings.daily_net_profit_target is not None
                            and session.get("net_profit", 0.0)
                            >= self._settings.daily_net_profit_target
                        )
                    ),
                }

                await self._reporter.send_session_report(
                    session_summary=session_summary,
                    status=snapshot,
                )
                logger.info({"event": "DAILY_REPORT_SENT"})
                status_store.add_event("Daily report sent", event_type="info")

            except Exception as exc:
                logger.warning({"event": "DAILY_REPORT_FAILED", "error": str(exc)})
                status_store.add_event(f"Daily report failed: {exc}", event_type="kill")

    async def _safe_engine_run(self, engine: LiveEngine) -> None:
        """
        Run a LiveEngine with crash isolation.

        Wraps engine.run() so that a single engine crash does not cancel
        other engines in the asyncio.gather() task group. The crash is
        logged as critical but does not propagate.

        Args:
            engine: The LiveEngine instance to run.
        """
        try:
            await engine.run()

        except LiveEngineError as exc:
            status_store.add_event(
                f"Engine crashed: {engine.symbol} - {exc}", event_type="kill"
            )
            logger.error(
                {
                    "event": "ENGINE_CRASHED",
                    "symbol": engine.symbol,
                    "expiry_key": engine.expiry_key,
                    "error": str(exc),
                    "error_type": "LiveEngineError",
                }
            )

        except Exception as exc:
            status_store.add_event(
                f"Engine crashed (unexpected): {engine.symbol} - {exc}",
                event_type="kill",
            )
            logger.error(
                {
                    "event": "ENGINE_CRASHED_UNEXPECTED",
                    "symbol": engine.symbol,
                    "expiry_key": engine.expiry_key,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    # ── Retrain Scheduler ─────────────────────────────────────────────────────

    async def _retrain_scheduler(self, storage: Storage) -> None:
        """
        Background loop that triggers periodic model retraining.

        Wakes every _RETRAIN_CHECK_INTERVAL_S seconds and checks whether
        model_retrain_interval has elapsed since the last retrain. If so,
        runs a full training cycle for each symbol, saves the
        artifact, and hot-reloads each matching LiveEngine.

        Uses the Stage 1 Storage instance — does not construct a second one.

        Individual symbol failures are caught and logged as warnings. The
        scheduler continues to the next symbol and does not abort.

        Args:
            storage: Stage 1 Storage instance shared from run().
        """
        manager = ModelManager(storage_dir=self._settings.model_dir)
        engineer = get_feature_engineer()
        inv_map: dict[int, str] = _SECONDS_TO_EXPIRY_KEY
        is_first_run: bool = True

        while not self._shutdown_requested:
            await asyncio.sleep(_RETRAIN_CHECK_INTERVAL_S)

            if self._shutdown_requested:
                break

            now: datetime = datetime.now(timezone.utc)
            elapsed: float = (now - self._last_retrain_time).total_seconds()

            if not is_first_run and elapsed < self._settings.model_retrain_interval:
                continue

            first_boot = is_first_run  # Snapshot before clearing

            if is_first_run:
                logger.info(
                    {
                        "event": "RETRAIN_FIRST_BOOT",
                        "message": "Checking for missing models",
                    }
                )
                is_first_run = False
            else:
                logger.info(
                    {
                        "event": "RETRAIN_CYCLE_STARTED",
                        "elapsed_seconds": round(elapsed, 1),
                    }
                )
                status_store.add_event("Retrain cycle started", event_type="info")

            expiry_key: str | None = inv_map.get(self._settings.expiry_seconds)
            if expiry_key is None:
                logger.warning(
                    {
                        "event": "RETRAIN_SKIPPED",
                        "reason": f"expiry_seconds={self._settings.expiry_seconds} has no key mapping",
                    }
                )
                self._last_retrain_time = now
                continue

            for symbol in self._settings.pairs:
                try:
                    # ── 1. Load bar data ──────────────────────────────────
                    max_rows: int | None = (
                        None
                        if self._settings.train_on_full_history
                        else self._settings.max_rf_rows
                    )
                    bars_df = storage.get_bars(
                        symbol, timeframe="M1", max_rows=max_rows
                    )
                    if bars_df is None or bars_df.empty:
                        logger.warning(
                            {
                                "event": "RETRAIN_NO_DATA",
                                "symbol": symbol,
                                "message": "No bar data found, skipping",
                            }
                        )
                        continue

                    # ── 2. Build feature matrix and labels ────────────────
                    feature_matrix = engineer.build_matrix(bars_df, symbol)
                    labeler = Labeler(expiry_key=expiry_key)
                    labels = labeler.compute_labels(bars_df)

                    # ── 3. Split and train ────────────────────────────────
                    shaper = DataShaper()
                    split = shaper.split(feature_matrix, labels, expiry_key)

                    # List all models you want to train
                    models_to_try = [
                        ("XGBoost", XGBoostTrainer(expiry_key=expiry_key)),
                        ("RandomForest", RandomForestTrainer(expiry_key=expiry_key)),
                        ("LightGBM", LightGBMTrainer(expiry_key=expiry_key)),
                        ("CatBoost", CatBoostTrainer(expiry_key=expiry_key)),
                        # These need to be fixed to work with AUC
                        """ (
                            "LSTM",
                            LSTMTrainer(expiry_key=expiry_key),
                        ),
                        (
                            "GRU",
                            GRUTrainer(expiry_key=expiry_key),
                        ),
                        (
                            "TCN",
                            TCNTrainer(expiry_key=expiry_key),
                        ), """,
                    ]

                    for model_name, trainer in models_to_try:
                        try:
                            # First boot only: skip this model if a blob artifact
                            # already exists. Scheduled retrains always run all models.

                            if first_boot:
                                existing = await manager.pull_from_blob(
                                    symbol=symbol,
                                    expiry_key=expiry_key,
                                    model_name=f"{model_name}Trainer",
                                )
                                if existing:
                                    logger.info(
                                        {
                                            "event": "RETRAIN_MODEL_EXISTS",
                                            "symbol": symbol,
                                            "expiry_key": expiry_key,
                                            "model_name": model_name,
                                        }
                                    )
                                    continue

                            logger.info(
                                {
                                    "event": "RETRAIN_TRAINING_START",
                                    "symbol": symbol,
                                    "model_name": model_name,
                                }
                            )
                            result = trainer.train(split)
                            auc = result.metrics.get("auc", 0)
                            logger.info(
                                {
                                    "event": "RETRAIN_TRAINING_COMPLETE",
                                    "symbol": symbol,
                                    "model_name": model_name,
                                    "auc": round(auc, 4),
                                }
                            )

                            # ── 4. Save artifact and push to Blob ─────────────────
                            artifact_path = Path(manager.save(result))
                            metadata_path = artifact_path.with_suffix(".json")
                            storage.save_model(artifact_path, metadata_path)

                        except Exception as e:
                            logger.warning(
                                {
                                    "event": "RETRAIN_MODEL_FAILED",
                                    "symbol": symbol,
                                    "model_name": model_name,
                                    "error": str(e),
                                }
                            )

                    # ── 4. Delegate best model selection to registry ───────
                    record = manager.get_best_model(
                        symbol=symbol, expiry_key=expiry_key
                    )
                    if record is None:
                        logger.warning(
                            {
                                "event": "RETRAIN_NO_BEST_MODEL",
                                "symbol": symbol,
                                "expiry_key": expiry_key,
                            }
                        )
                        continue

                    logger.info(
                        {
                            "event": "RETRAIN_COMPLETE",
                            "symbol": symbol,
                            "expiry_key": expiry_key,
                            "best_model": record.model_name,
                            "auc": round(record.auc, 4),
                        }
                    )
                    status_store.add_event(
                        f"Retrain complete: {symbol} - {record.model_name} (AUC: {record.auc:.4f})",
                        event_type="info",
                    )

                    # ── 5. Hot-reload matching engines ────────────────────
                    for engine in self._engines:
                        if engine.symbol == symbol and engine.expiry_key == expiry_key:
                            reloaded = await engine.reload_model()
                            if reloaded:
                                logger.info(
                                    {
                                        "event": "ENGINE_RELOADED",
                                        "symbol": symbol,
                                        "expiry_key": expiry_key,
                                    }
                                )
                                status_store.add_event(
                                    f"Engine reloaded: {symbol}", event_type="info"
                                )
                            else:
                                logger.warning(
                                    {
                                        "event": "ENGINE_RELOAD_FAILED",
                                        "symbol": symbol,
                                        "expiry_key": expiry_key,
                                    }
                                )
                                status_store.add_event(
                                    f"Engine reload failed: {symbol}", event_type="kill"
                                )

                except Exception as exc:
                    logger.warning(
                        {
                            "event": "RETRAIN_SYMBOL_FAILED",
                            "symbol": symbol,
                            "expiry_key": expiry_key,
                            "error": str(exc),
                        }
                    )
                    status_store.add_event(
                        f"Retrain failed for {symbol}: {exc}", event_type="kill"
                    )

            self._last_retrain_time = now
            logger.info(
                {
                    "event": "RETRAIN_CYCLE_COMPLETE",
                    "timestamp": now.isoformat(),
                    "symbols": self._settings.pairs,
                    "next_interval_seconds": self._settings.model_retrain_interval,
                }
            )
