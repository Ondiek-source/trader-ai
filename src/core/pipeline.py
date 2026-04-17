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

from core.config import get_settings
from data.historian import Historian, HistorianError, get_historian
from data.storage import Storage, StorageError
from ml_engine.features import FeatureEngineerError, get_feature_engineer
from ml_engine.model_manager import ModelManager
from ml_engine.trainer import DataShaper, XGBoostTrainer
from ml_engine.labeler import Labeler
from engine.live import LiveEngine, LiveEngineError

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


# ── Custom Exception ──────────────────────────────────────────────────────────


class PipelineError(Exception):
    """
    Raised when a critical stage of the boot sequence fails.

    Propagates to main.py which logs it and exits. Detailed diagnostic
    logging is always emitted by the stage method before raising so the
    operator sees exactly which stage and component failed.

    Attributes:
        stage: The boot stage that failed, e.g. "storage", "historian_sync".
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return f"PipelineError(stage={self.stage!r}, message={str(self)!r})"


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
            self._storage = await self._stage_storage()
            await self._stage_historian_sync(self._storage)
            await self._stage_feature_warmup(self._storage)
            model_map = await self._stage_model_load()
            await self._stage_ignition(model_map)
            await self._run_task_group()

        except PipelineError:
            raise  # Detailed logging already emitted by the stage method.

        except Exception as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"PIPELINE UNHANDLED CRASH\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: An unexpected exception bypassed all stage-level\n"
                f"guards. This is likely a bug in a module imported by the\n"
                f"pipeline rather than in the boot sequence itself.\n"
                f"\nFIX: Review the full stack trace above. Verify that all\n"
                f"module imports resolve correctly and no __init__ methods\n"
                f"raise outside the expected exception hierarchy.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise

        finally:
            logger.info(
                "[^] Pipeline.run() exiting — symbol=%s",
                self._settings.pairs,
            )

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
        logger.info("[^] Pipeline.stop(): shutdown initiated.")

        for engine in self._engines:
            engine.stop()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Push stopped state to dashboard
        try:
            from core.dashboard import status_store

            snapshot = status_store.get()
            status_store.update(
                {
                    "stopped": True,
                    "session": {**snapshot.get("session", {}), "is_active": False},
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

        logger.info("[^] Pipeline.stop(): all engines stopped, all tasks cancelled.")

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
            storage = Storage()
            info_block = (
                f"\n{'+' * 60}\n"
                f"STAGE 1 COMPLETE: STORAGE LINKED\n"
                f"Data mode : {self._settings.data_mode}\n"
                f"Data dir  : {self._settings.data_dir}\n"
                f"{'+' * 60}"
            )
            logger.info(info_block)
            return storage

        except StorageError as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"PIPELINE STAGE 1 FAILURE: STORAGE LINK\n"
                f"Error     : {exc}\n"
                f"Data mode : {self._settings.data_mode}\n"
                f"Data dir  : {self._settings.data_dir}\n\n"
                f"CONTEXT: Storage is the foundation of every downstream component.\n"
                f"Without it no bar data can be read, no features computed, and\n"
                f"no model loaded. This failure is unrecoverable.\n"
                f"\nFIX: Check disk permissions and available disk space.\n"
                f"If DATA_MODE=CLOUD, verify AZURE_STORAGE_CONN in .env and\n"
                f"confirm the container '{self._settings.container_name}' exists.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
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
                    "[^] Stage 2 sync: symbol=%s bars_committed=%d",
                    symbol,
                    count,
                )

            except HistorianError as exc:
                error_block = (
                    f"\n{'!' * 60}\n"
                    f"PIPELINE STAGE 2 FAILURE: HISTORIAN SYNC\n"
                    f"Symbol : {symbol}\n"
                    f"Error  : {exc}\n\n"
                    f"CONTEXT: The gap backfill failed for this symbol. A time-gap\n"
                    f"in M1 bars silently corrupts every rolling indicator (RSI 14,\n"
                    f"MACD 26, ATR 14, BB 20) for the next 26 bars after the gap.\n"
                    f"The model will generate confident but wrong signals with no\n"
                    f"error raised. Aborting is safer than trading on corrupted\n"
                    f"features.\n"
                    f"\nFIX: Check TWELVEDATA_API_KEY in .env. Verify network\n"
                    f"connectivity to api.twelvedata.com. If the provider is down,\n"
                    f"restart the pipeline when connectivity is restored. Do NOT\n"
                    f"set ALLOW_STALE_MODELS=True — that solves a different problem.\n"
                    f"{'!' * 60}"
                )
                logger.critical(error_block)
                raise PipelineError(str(exc), stage="historian_sync") from exc

        info_block = (
            f"\n{'+' * 60}\n"
            f"STAGE 2 COMPLETE: HISTORIAN SYNC\n"
            f"Results: {results}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
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
                error_block = (
                    f"\n{'!' * 60}\n"
                    f"PIPELINE STAGE 3 FAILURE: INSUFFICIENT BARS FOR WARMUP\n"
                    f"Symbol    : {symbol}\n"
                    f"Available : {bar_count}\n"
                    f"Required  : 30\n\n"
                    f"CONTEXT: The FeatureEngineer requires a minimum of 30 M1 bars\n"
                    f"to compute rolling indicators (MACD slow span = 26 bars) without\n"
                    f"producing NaN values. Starting the LiveEngine with fewer bars\n"
                    f"causes FeatureEngineerErrors on every tick for the first 30\n"
                    f"minutes of live data.\n"
                    f"\nFIX: Stage 2 (historian sync) should have filled this gap.\n"
                    f"If this error appears after a successful Stage 2, the bar data\n"
                    f"for {symbol} may be corrupted. Delete the Parquet file at\n"
                    f"data/processed/{symbol}_M1.parquet and restart to trigger a\n"
                    f"full backfill.\n"
                    f"{'!' * 60}"
                )
                logger.critical(error_block)
                raise PipelineError(
                    f"Insufficient bars for {symbol}: {bar_count}/30",
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
                error_block = (
                    f"\n{'!' * 60}\n"
                    f"PIPELINE STAGE 3 FAILURE: FEATURE TRANSFORMATION ERROR\n"
                    f"Symbol : {symbol}\n"
                    f"Bars   : {bar_count}\n"
                    f"Error  : {exc}\n\n"
                    f"CONTEXT: The smoke-test transform() call failed on the warmup\n"
                    f"bars. This indicates a schema mismatch between the stored bar\n"
                    f"data and the FeatureEngineer's expected column layout, or a\n"
                    f"feature flag configuration that produces an incomplete schema.\n"
                    f"\nFIX: Verify that all FEAT_*_ENABLED flags in .env are True.\n"
                    f"Check the bar data schema via storage.get_bars() in a REPL.\n"
                    f"If the Parquet file is corrupted, delete and re-run backfill.\n"
                    f"{'!' * 60}"
                )
                logger.critical(error_block)
                raise PipelineError(str(exc), stage="feature_warmup") from exc

        info_block = (
            f"\n{'+' * 60}\n"
            f"STAGE 3 COMPLETE: FEATURE WARMUP VERIFIED\n"
            f"Results: {results}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
        return results

    # ── Stage 4: Model Load ───────────────────────────────────────────────────

    async def _stage_model_load(self) -> dict[tuple[str, str], Any]:
        """
        Pull model artifacts from Blob and load them into memory.

        For each (symbol, expiry_key) pair: attempts a cold-start Blob pull,
        queries the registry for the best current-version artifact, and loads
        it. PyTorch artifacts are skipped with a warning — they require
        architecture injection which pipeline.py defers to a future iteration.
        Missing models put the engine in SKIP-only mode rather than aborting.

        Returns:
            dict keyed by (symbol, expiry_key) -> (model_artifact, ModelRecord).
            Pairs with no available model are absent from the dict.

        Raises:
            Nothing — Stage 4 never aborts. All failures degrade to SKIP mode.
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
                        "[^] Stage 4 cold-start: restored %s %s -> %s",
                        symbol,
                        expiry_key,
                        pulled,
                    )

                # 4b: Query registry for best artifact
                record = manager.get_best_model(symbol=symbol, expiry_key=expiry_key)
                if record is None:
                    warning_block = (
                        f"\n{'%' * 60}\n"
                        f"PIPELINE STAGE 4 WARNING: NO MODEL FOUND\n"
                        f"Symbol     : {symbol}\n"
                        f"Expiry key : {expiry_key}\n\n"
                        f"CONTEXT: No trained model artifact was found for this\n"
                        f"symbol/expiry combination. The LiveEngine for this pair\n"
                        f"will boot in SKIP-only mode — all signals will be SKIP\n"
                        f"until a model is trained and loaded via the retrain\n"
                        f"scheduler or a manual training pass.\n"
                        f"\nFIX: Run a training pass via the retrain scheduler or\n"
                        f"manually trigger XGBoostTrainer for {symbol} {expiry_key}.\n"
                        f"{'%' * 60}"
                    )
                    logger.warning(warning_block)
                    continue

                # 4c: Skip PyTorch — requires architecture injection
                if record.is_pytorch:
                    warning_block = (
                        f"\n{'%' * 60}\n"
                        f"PIPELINE STAGE 4 WARNING: PYTORCH INJECTION DEFERRED\n"
                        f"Symbol     : {symbol}\n"
                        f"Expiry key : {expiry_key}\n"
                        f"Model      : {record.model_name}\n\n"
                        f"CONTEXT: PyTorch model artifacts require a pre-instantiated\n"
                        f"nn.Module architecture to be provided at load time.\n"
                        f"Pipeline-level PyTorch injection is not yet implemented.\n"
                        f"The engine will use SignalGenerator.reload() which handles\n"
                        f"Classical ML and SB3 models automatically.\n"
                        f"\nFIX: Implement PyTorch architecture resolution in\n"
                        f"pipeline._stage_model_load() when required.\n"
                        f"{'%' * 60}"
                    )
                    logger.warning(warning_block)
                    continue

                # 4d: Load artifact into memory
                try:
                    model = manager.load(record.artifact_path)
                    model_map[(symbol, expiry_key)] = (model, record)
                    logger.info(
                        "[^] Stage 4 loaded: symbol=%s expiry=%s model=%s auc=%.4f",
                        symbol,
                        expiry_key,
                        record.model_name,
                        record.auc,
                    )

                except Exception as exc:
                    warning_block = (
                        f"\n{'%' * 60}\n"
                        f"PIPELINE STAGE 4 WARNING: ARTIFACT LOAD FAILED\n"
                        f"Symbol     : {symbol}\n"
                        f"Expiry key : {expiry_key}\n"
                        f"Artifact   : {record.artifact_path}\n"
                        f"Error      : {exc}\n\n"
                        f"CONTEXT: The artifact exists in the registry but could not\n"
                        f"be deserialised. The engine will boot in SKIP-only mode\n"
                        f"for this pair.\n"
                        f"\nFIX: Verify the artifact file is not corrupted. Check\n"
                        f"model_manager.validate_metadata() for the sidecar. Retrain\n"
                        f"if the artifact is unrecoverable.\n"
                        f"{'%' * 60}"
                    )
                    logger.warning(warning_block)

        info_block = (
            f"\n{'+' * 60}\n"
            f"STAGE 4 COMPLETE: MODELS LOADED\n"
            f"Loaded : {len(model_map)} artifact(s)\n"
            f"Pairs  : {list(model_map.keys())}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
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
            error_block = (
                f"\n{'!' * 60}\n"
                f"PIPELINE STAGE 5 FAILURE: INVALID EXPIRY SECONDS\n"
                f"EXPIRY_SECONDS : {expiry_seconds}\n"
                f"Valid values   : {sorted(_SECONDS_TO_EXPIRY_KEY.keys())}\n\n"
                f"CONTEXT: The configured EXPIRY_SECONDS value does not map to a\n"
                f"known expiry key. This should have been caught by config\n"
                f"validation at boot. Check that VALID_EXPIRIES in config.py\n"
                f"matches _EXPIRY_KEY_MAP in pipeline.py.\n"
                f"\nFIX: Set EXPIRY_SECONDS to 60, 300, or 900 in .env.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise PipelineError(
                f"EXPIRY_SECONDS={expiry_seconds} has no expiry_key mapping.",
                stage="ignition",
            )

        for symbol in self._settings.pairs:
            try:
                engine = await LiveEngine.create(symbol, expiry_key)

            except LiveEngineError as exc:
                error_block = (
                    f"\n{'!' * 60}\n"
                    f"PIPELINE STAGE 5 FAILURE: ENGINE CONSTRUCTION\n"
                    f"Symbol     : {symbol}\n"
                    f"Expiry key : {expiry_key}\n"
                    f"Error      : {exc}\n\n"
                    f"CONTEXT: LiveEngine could not be constructed for this symbol.\n"
                    f"The most likely cause is a stream construction failure —\n"
                    f"either Quotex credentials are invalid or the TwelveData\n"
                    f"API key is missing.\n"
                    f"\nFIX: Verify QUOTEX_EMAIL, QUOTEX_PASSWORD, and\n"
                    f"TWELVEDATA_API_KEY in .env. Check network connectivity.\n"
                    f"{'!' * 60}"
                )
                logger.critical(error_block)
                raise PipelineError(str(exc), stage="ignition") from exc

            # Inject the pipeline-resolved model if available.
            # This overwrites whatever SignalGenerator.reload() loaded in its
            # own __init__ call, ensuring the freshest artifact is always used.
            if (symbol, expiry_key) in model_map:
                model, record = model_map[(symbol, expiry_key)]
                engine._signal_gen.inject_model(model, record)
                logger.info(
                    "[^] Stage 5 injected: symbol=%s expiry=%s model=%s",
                    symbol,
                    expiry_key,
                    record.model_name,
                )

            self._engines.append(engine)

        info_block = (
            f"\n{'+' * 60}\n"
            f"STAGE 5 COMPLETE: ENGINES IGNITED\n"
            f"Engines : {len(self._engines)}\n"
            f"Symbols : {[e.symbol for e in self._engines]}\n"
            f"Expiry  : {expiry_key}\n"
            f"Streams : {[type(e._stream).__name__ for e in self._engines]}\n"
            f"{'+' * 60}"
        )
        logger.info(info_block)
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
        from core.dashboard import status_store

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
        from core.dashboard import run_dashboard

        self._tasks.append(
            asyncio.create_task(
                run_dashboard(self._settings.dashboard_port),
                name="dashboard",
            )
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
            "[^] Pipeline task group: %d tasks launched.",
            len(self._tasks),
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
            from trading.reporter import Reporter

            # Thin adapter so Pipeline satisfies OrchestratorProtocol without
            # conflicting with the existing async Pipeline.stop() method.
            pipeline_ref = self

            class _Orchestrator:
                def get_status(self) -> dict[str, Any]:
                    from core.dashboard import status_store

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
            logger.warning(
                "[%%] Pipeline: Reporter init failed — notifications disabled. %s", exc
            )
            return None

    # ── Boot notification ─────────────────────────────────────────────────────

    async def _send_boot_notification(self) -> None:
        """
        Send a 'system ready' alert to Discord and Telegram on every deploy.

        Called once after all tasks are created but before asyncio.gather()
        blocks. Failure is silently swallowed — a missed boot alert must
        never prevent the trading loop from starting.
        """
        if self._reporter is None:
            return

        symbols = ", ".join(f"`{e.symbol}`" for e in self._engines)
        mode = "PRACTICE" if self._settings.practice_mode else "LIVE"
        mode_icon = "🟡" if self._settings.practice_mode else "🔴"
        message = (
            f"🚀 <b>Trader AI — System Ready</b>\n"
            f"{mode_icon} Mode: <code>{mode}</code>\n"
            f"📊 Symbols: {symbols}\n"
            f"⏱ Expiry: <code>{self._settings.expiry_seconds}s</code>\n"
            f"🎯 Base threshold: <code>{self._settings.confidence_threshold:.0%}</code>\n"
            f"🛡 Max streak: <code>{self._settings.martingale_max_streak}</code>\n"
            f"📡 Stream: <code>{'Quotex' if self._settings.use_quotex_streaming else 'TwelveData'}</code>"
        )

        try:
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
        except Exception as exc:
            logger.warning("[%%] Boot notification failed (non-fatal): %s", exc)

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
        from core.dashboard import status_store

        if not self._settings.use_quotex_streaming:
            logger.info("[^] Quotex status loop: not using Quotex streaming — exiting.")
            return

        # Give QuotexReader time to connect before first poll.
        await asyncio.sleep(15)

        while not self._shutdown_requested:
            try:
                # Use the first engine's stream — all engines share the same
                # Quotex account so one poll is sufficient.
                stream = self._engines[0]._stream if self._engines else None
                if stream is None or not hasattr(stream, "_connected"):
                    await asyncio.sleep(30)
                    continue

                connected: bool = bool(getattr(stream, "_connected", False))
                balance: float = 0.0
                wins: int = 0
                losses: int = 0
                draws: int = 0
                net_profit: float = 0.0
                pending: int = len(getattr(stream, "_pending", {}))

                if connected:
                    try:
                        balance = await asyncio.wait_for(
                            stream.get_balance(), timeout=5.0
                        )
                    except Exception:
                        pass

                    try:
                        history = await asyncio.wait_for(
                            stream.get_history(), timeout=8.0
                        )
                        for trade in history:
                            profit = trade.get("profitAmount", 0)
                            if profit > 0:
                                wins += 1
                            elif profit < 0:
                                losses += 1
                            else:
                                draws += 1
                            net_profit += profit
                    except Exception:
                        pass

                # Compute elapsed minutes from the started_at stamp in StatusStore
                snapshot = status_store.get()
                try:
                    started_at = datetime.fromisoformat(snapshot["started_at"])
                    elapsed_minutes = (
                        datetime.now(timezone.utc) - started_at
                    ).total_seconds() / 60
                except Exception:
                    elapsed_minutes = 0

                status_store.update(
                    {
                        "quotex": {"connected": connected, "balance": balance},
                        "pending_signals": pending,
                        "session": {
                            **snapshot.get("session", {}),
                            "wins": wins,
                            "losses": losses,
                            "draws": draws,
                            "net_profit": net_profit,
                            "elapsed_minutes": round(elapsed_minutes, 1),
                        },
                    }
                )

            except Exception as exc:
                logger.warning({"event": "quotex_status_poll_error", "error": str(exc)})

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
            logger.info("[^] Daily report loop: no reporter configured — exiting.")
            return

        from core.dashboard import status_store

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
                "[^] Daily report loop: next report in %.0f minutes.",
                sleep_for / 60,
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
                logger.info({"event": "daily_report_sent"})

            except Exception as exc:
                logger.warning({"event": "daily_report_failed", "error": str(exc)})

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
            error_block = (
                f"\n{'!' * 60}\n"
                f"ENGINE CRASHED: LIVE ENGINE ERROR\n"
                f"Symbol     : {engine.symbol}\n"
                f"Expiry key : {engine.expiry_key}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: The LiveEngine for this symbol crashed after boot.\n"
                f"Other engines continue running. This engine will not restart\n"
                f"automatically — the session continues without this symbol.\n"
                f"\nFIX: Review the engine log above. Restart the pipeline after\n"
                f"resolving the root cause (stream disconnect, model failure, etc).\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)

        except Exception as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"ENGINE CRASHED: UNEXPECTED EXCEPTION\n"
                f"Symbol     : {engine.symbol}\n"
                f"Expiry key : {engine.expiry_key}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: An unexpected exception crashed this engine. Other\n"
                f"engines continue running. This is likely a bug in the inference\n"
                f"pipeline or a stream-level failure not covered by LiveEngineError.\n"
                f"\nFIX: Review the full stack trace. Restart the pipeline after\n"
                f"resolving the root cause.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)

    # ── Retrain Scheduler ─────────────────────────────────────────────────────

    async def _retrain_scheduler(self, storage: Storage) -> None:
        """
        Background loop that triggers periodic model retraining.

        Wakes every _RETRAIN_CHECK_INTERVAL_S seconds and checks whether
        model_retrain_interval has elapsed since the last retrain. If so,
        runs a full XGBoost training cycle for each symbol, saves the
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

        while not self._shutdown_requested:
            await asyncio.sleep(_RETRAIN_CHECK_INTERVAL_S)

            if self._shutdown_requested:
                break

            now: datetime = datetime.now(timezone.utc)
            elapsed: float = (now - self._last_retrain_time).total_seconds()

            if elapsed < self._settings.model_retrain_interval:
                continue

            logger.info(
                "[^] Retrain scheduler: interval elapsed (%.0fs). Starting cycle.",
                elapsed,
            )

            expiry_key: str | None = inv_map.get(self._settings.expiry_seconds)
            if expiry_key is None:
                logger.warning(
                    "[%%] Retrain scheduler: expiry_seconds=%d has no key mapping. "
                    "Skipping cycle.",
                    self._settings.expiry_seconds,
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
                        warning_block = (
                            f"\n{'%' * 60}\n"
                            f"RETRAIN SCHEDULER WARNING: NO BAR DATA\n"
                            f"Symbol : {symbol}\n\n"
                            f"CONTEXT: No bar data found in storage for this symbol.\n"
                            f"Cannot train without data. Skipping this symbol.\n"
                            f"\nFIX: Verify historian sync ran successfully for\n"
                            f"{symbol}. Check storage for the Parquet file.\n"
                            f"{'%' * 60}"
                        )
                        logger.warning(warning_block)
                        continue

                    # ── 2. Build feature matrix and labels ────────────────
                    feature_matrix = engineer.build_matrix(bars_df, symbol)
                    labeler = Labeler(expiry_key=expiry_key)
                    labels = labeler.compute_labels(bars_df)

                    # ── 3. Split and train ────────────────────────────────
                    shaper = DataShaper()
                    split = shaper.split(feature_matrix, labels, expiry_key)

                    trainer = XGBoostTrainer(expiry_key=expiry_key)
                    result = trainer.train(split)

                    # ── 4. Save artifact and push to Blob ─────────────────
                    artifact_path_str: str = manager.save(result)
                    artifact_path = Path(artifact_path_str)
                    metadata_path = artifact_path.with_suffix(".json")
                    storage.save_model(artifact_path, metadata_path)

                    logger.info(
                        "[^] Retrain complete: symbol=%s expiry=%s auc=%.4f",
                        symbol,
                        expiry_key,
                        result.metrics.get("auc", 0.0),
                    )

                    # ── 5. Hot-reload matching engines ────────────────────
                    for engine in self._engines:
                        if engine.symbol == symbol and engine.expiry_key == expiry_key:
                            reloaded = await engine.reload_model()
                            if reloaded:
                                logger.info(
                                    "[^] Engine reloaded: symbol=%s expiry=%s",
                                    symbol,
                                    expiry_key,
                                )
                            else:
                                warning_block = (
                                    f"\n{'%' * 60}\n"
                                    f"RETRAIN SCHEDULER WARNING: ENGINE RELOAD FAILED\n"
                                    f"Symbol     : {symbol}\n"
                                    f"Expiry key : {expiry_key}\n\n"
                                    f"CONTEXT: The retrain completed and the artifact was\n"
                                    f"saved, but the engine could not load the new model.\n"
                                    f"The engine continues with the previously loaded model.\n"
                                    f"\nFIX: Check model_manager logs. The artifact may\n"
                                    f"have been saved but the registry query returned None.\n"
                                    f"{'%' * 60}"
                                )
                                logger.warning(warning_block)

                except Exception as exc:
                    warning_block = (
                        f"\n{'%' * 60}\n"
                        f"RETRAIN SCHEDULER WARNING: RETRAIN FAILED\n"
                        f"Symbol     : {symbol}\n"
                        f"Expiry key : {expiry_key}\n"
                        f"Error      : {exc}\n\n"
                        f"CONTEXT: The scheduled retrain failed for this symbol.\n"
                        f"The engine continues using the previously loaded model.\n"
                        f"This is non-fatal — the scheduler will retry on the\n"
                        f"next interval.\n"
                        f"\nFIX: Check trainer logs and bar data quality in storage.\n"
                        f"Verify model_dir permissions for artifact saving.\n"
                        f"{'%' * 60}"
                    )
                    logger.warning(warning_block)

            self._last_retrain_time = now
            info_block = (
                f"\n{'+' * 60}\n"
                f"RETRAIN CYCLE COMPLETE\n"
                f"Timestamp : {now.isoformat()}\n"
                f"Symbols   : {self._settings.pairs}\n"
                f"Next in   : {self._settings.model_retrain_interval}s\n"
                f"{'+' * 60}"
            )
            logger.info(info_block)
