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
import sys
import time
import signal
import asyncio
import logging


from typing import Any
from datetime import datetime, timedelta, timezone
from trading.reporter import Reporter
from core.config import get_settings
from core.exceptions import PipelineError, NotificationError
from core.dashboard import status_store, run_dashboard
from data.historian import Historian, HistorianError, get_historian
from data.storage import Storage, StorageError, get_storage
from engine.live import LiveEngine, LiveEngineError
from data.forex_calendar import is_forex_closed
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

# All expiry keys train on M1 bars — the Labeler's lookahead_bars is always
# measured in M1 minutes (expiry_seconds // 60), so M5/M15 bars would give
# wrong labels (5 M5 bars = 25 min, not 5 min for a 5_MIN trade).
_TIMEFRAME_MAP = {"1_MIN": "M1", "5_MIN": "M1", "15_MIN": "M1"}


logger = logging.getLogger(__name__)

# ── Module Constants ──────────────────────────────────────────────────────────

# Bars pre-fetched in Stage 3 warmup verification.
# Must be >= _MIN_BARS_REQUIRED in live.py (30). 100 gives safe headroom.
_WARMUP_BARS: int = 100

# How often the retrain scheduler wakes to check if the interval has elapsed.
# Actual retraining only fires when model_retrain_interval seconds have passed.
# First-boot retrain is immediate regardless of this interval (is_first_run guard).
_RETRAIN_CHECK_INTERVAL_S: int = 3600

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

# Active trainers pulled from Blob and used during retraining.
# RandomForest excluded: produces multi-GiB artifacts that break container boot.
_ACTIVE_TRAINERS: list[str] = [
    "XGBoostTrainer",
    "LightGBMTrainer",
    "CatBoostTrainer",
]


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
        self._manager: ModelManager | None = None
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
        if sys.platform != "win32":
            # add_signal_handler is Unix-only — not available on Windows
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
            await self._stage_historian_sync()
            self._stage_feature_warmup(self._storage)
            self._manager = ModelManager(storage_dir=self._settings.model_dir)
            model_map = await self._stage_model_load(self._manager)
            await self._stage_ignition(model_map, self._manager)
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
            if hasattr(engine._stream, "disconnect"):
                await engine._stream.disconnect()
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
        establishes the Azure Blob connection — including a live network probe
        via get_container_properties(). That probe is a blocking synchronous
        call, so we run Storage construction in a thread to avoid stalling the
        event loop during the TLS handshake.

        Returns:
            Storage: The initialised, verified Storage instance. Passed to
                        all downstream stages that require it.

        Raises:
            PipelineError: If Storage construction fails.
        """
        try:
            storage = await asyncio.to_thread(get_storage)
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

    async def _stage_historian_sync(self) -> dict[str, int]:
        """
        Fill the time gap since the last session for every backfill pair.

        Calls historian.backfill(symbol) directly per pair so that
        HistorianError is caught and treated as a fatal abort. This is
        intentionally different from historian.backfill_all() which catches
        HistorianError internally and returns 0 — making failure indistinct
        from an up-to-date result.

        A return value of 0 from backfill() means the data was already
        current (start_dt >= now_utc). This is not a failure.

        Returns:
            dict[str, int]: Mapping of {symbol: bars_committed}.

        Raises:
            PipelineError: If backfill fails for any pair.
        """
        historian: Historian = get_historian()
        results: dict[str, int] = {}

        # ── Phase 1: Normal forward-fill backfill ───────────────────────
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

        # ── Phase 2: Internal gap repair ────────────────────────────────
        for symbol in self._settings.backfill_pairs:
            try:
                repaired: int = await self._repair_internal_gaps(symbol, historian)
                if repaired > 0:
                    results[symbol] = results.get(symbol, 0) + repaired
            except Exception as exc:
                logger.warning(
                    {
                        "event": "GAP_REPAIR_FAILED",
                        "symbol": symbol,
                        "error": str(exc),
                        "message": "Continuing — indicators may be degraded",
                    }
                )

        logger.info(
            {
                "event": "STAGE_COMPLETE",
                "stage": "historian_sync",
                "results": results,
            }
        )
        return results

    async def _repair_internal_gaps(self, symbol: str, historian: Historian) -> int:
        """
        Scan the full M1 series for internal data gaps and re-fetch missing bars.

        The Historian only forward-fills from the last bar. Internal gaps
        from past API failures persist indefinitely and corrupt indicators.
        This method detects those gaps and delegates to Historian.fetch_range()
        for the actual API work (chunking, rate-limiting, retries).
        """
        timestamps = self._load_m1_timestamps(symbol)
        if timestamps is None or len(timestamps) < 2:
            return 0

        gaps = self._find_internal_gaps(timestamps)
        if not gaps:
            return 0

        # Only repair gaps from the last N days — older gaps are likely
        # unfillable (free tier limitations) and harmless to live indicators.
        max_gap_age_days = 90
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_gap_age_days)
        cutoff_naive = cutoff.replace(tzinfo=None)
        gaps = [(s, e, m) for s, e, m in gaps if s >= cutoff_naive]
        unexpected = self._filter_unexpected_gaps(gaps, symbol)
        if not unexpected:
            return 0

        return await self._fetch_gap_bars(symbol, unexpected, historian)

    def _load_m1_timestamps(self, symbol: str) -> Any:
        """
        Load M1 bars from storage and return sorted timestamps within
        the backfill window.  Returns None if no data exists.
        """
        storage = self._storage
        if storage is None:
            return None

        bars_df = storage.get_bars(symbol, timeframe="M1")
        if bars_df is None or bars_df.empty:
            return None

        timestamps = bars_df.index.sort_values()

        now_utc: datetime = datetime.now(timezone.utc)
        years: int = self._settings.backfill_years
        try:
            oldest_allowed: datetime = now_utc.replace(
                year=now_utc.year - years,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        except ValueError:
            oldest_allowed = now_utc.replace(
                year=now_utc.year - years,
                month=2,
                day=28,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        # Parquet index is tz-naive datetime64[ns]; oldest_allowed is
        # tz-aware UTC.  Strip tzinfo for naive-vs-naive comparison.
        oldest_naive = oldest_allowed.replace(tzinfo=None)
        return timestamps[timestamps >= oldest_naive]

    @staticmethod
    def _find_internal_gaps(
        timestamps: Any,
    ) -> list[tuple[datetime, datetime, int]]:
        """
        Scan timestamps for gaps larger than 1 minute.

        Returns a list of (gap_start, gap_end, gap_size_minutes) tuples.
        """
        gaps: list[tuple[datetime, datetime, int]] = []
        for i in range(1, len(timestamps)):
            prev = timestamps[i - 1]
            curr = timestamps[i]
            if hasattr(prev, "to_pydatetime"):
                prev = prev.to_pydatetime()
            if hasattr(curr, "to_pydatetime"):
                curr = curr.to_pydatetime()
            diff_min = int((curr - prev).total_seconds() // 60)
            if diff_min > 1:
                gaps.append((prev, curr, diff_min))
        return gaps

    def _filter_unexpected_gaps(
        self,
        gaps: list[tuple[datetime, datetime, int]],
        symbol: str,
    ) -> list[tuple[datetime, datetime, int]]:
        """
        Remove gaps that are tiny (<=5 bars) or coincide with
        weekend/holiday market closures.
        """
        from data.forex_calendar import is_forex_closed

        _MAX_FFILL_BARS: int = 5

        def _is_market_closed(dt: datetime) -> bool:
            dow = dt.weekday()
            hour = dt.hour
            if dow == 5:
                return True
            if dow == 6 and hour < 21:
                return True
            if dow == 4 and hour >= 21:
                return True
            return is_forex_closed(dt)

        unexpected: list[tuple[datetime, datetime, int]] = []
        for gap_start, gap_end, gap_size in gaps:
            if gap_size <= _MAX_FFILL_BARS + 1:
                continue
            mid_point = gap_start + (gap_end - gap_start) / 2
            if _is_market_closed(mid_point):
                continue
            unexpected.append((gap_start, gap_end, gap_size))

        if not unexpected:
            logger.info(
                {
                    "event": "GAP_SCAN_ALL_NORMAL",
                    "symbol": symbol,
                    "total_gaps_checked": len(gaps),
                }
            )
        return unexpected

    async def _fetch_gap_bars(
        self,
        symbol: str,
        gaps: list[tuple[datetime, datetime, int]],
        historian: Historian,
    ) -> int:
        """
        Delegate each gap to Historian.fetch_range() and return total bars
        repaired.
        """
        total_gap_bars = sum(g[2] for g in gaps)
        logger.info(
            {
                "event": "GAP_REPAIR_START",
                "symbol": symbol,
                "gaps_found": len(gaps),
                "missing_bars": total_gap_bars,
            }
        )

        total_repaired: int = 0
        for idx, (gap_start, gap_end, _gap_size) in enumerate(gaps):
            fetch_start: datetime = gap_start + timedelta(minutes=1)
            fetch_end: datetime = gap_end - timedelta(minutes=1)
            if fetch_start >= fetch_end:
                continue

            try:
                repaired: int = await historian.fetch_range(
                    symbol, fetch_start, fetch_end
                )
                total_repaired += repaired
                logger.info(
                    {
                        "event": "GAP_REPAIR_PROGRESS",
                        "symbol": symbol,
                        "gaps_done": idx + 1,
                        "gaps_total": len(gaps),
                        "bars_repaired": total_repaired,
                    }
                )
            except Exception as exc:
                logger.warning(
                    {
                        "event": "GAP_REPAIR_CHUNK_ERROR",
                        "symbol": symbol,
                        "error": str(exc),
                    }
                )

        logger.info(
            {
                "event": "GAP_REPAIR_COMPLETE",
                "symbol": symbol,
                "bars_repaired": total_repaired,
                "gaps_filled": len(gaps),
            }
        )
        return total_repaired

    # ── Stage 3: Feature Warmup ───────────────────────────────────────────────

    def _stage_feature_warmup(self, storage: Storage) -> dict[str, int]:
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

    async def _pull_model_artifacts(
        self, symbol: str, expiry_key: str, manager: ModelManager
    ) -> None:
        for trainer_name in _ACTIVE_TRAINERS:
            try:
                pulled = await manager.pull_from_blob(
                    symbol=symbol,
                    expiry_key=expiry_key,
                    model_name=trainer_name,
                )
                if pulled:
                    logger.info(
                        {
                            "event": "MODEL_PULLED_TO_LOCAL_DISK",
                            "symbol": symbol,
                            "expiry_key": expiry_key,
                            "trainer": trainer_name,
                            "artifact": pulled,
                        }
                    )
            except Exception as exc:
                logger.warning(
                    {
                        "event": "MODEL_PULL_FAILED",
                        "symbol": symbol,
                        "expiry_key": expiry_key,
                        "trainer": trainer_name,
                        "error": str(exc),
                    }
                )

    def _load_best_model(
        self, symbol: str, expiry_key: str, manager: ModelManager
    ) -> tuple[Any, Any] | None:
        record = manager.get_best_model(symbol=symbol, expiry_key=expiry_key)
        if record is None:
            return None
        try:
            model = manager.load(record.artifact_path)
            return model, record
        except Exception as exc:
            logger.warning(
                {
                    "event": "MODEL_LOAD_FAILED",
                    "symbol": symbol,
                    "expiry_key": expiry_key,
                    "error": str(exc),
                    "message": "Skipping — retrain scheduler will recover on first boot",
                }
            )
            return None

    async def _stage_model_load(self, manager: ModelManager) -> dict[tuple[str, str], Any]:
        """
        Pull model artifacts from Blob and load them into memory.

        For each (symbol, expiry_key) pair: pulls each active trainer artifact
        individually so a single large or corrupt artifact cannot block others.
        Queries the registry for the best current-version artifact and loads it.
        Load failures are demoted to warnings — the retrain scheduler recovers
        on first boot.

        Args:
            manager: Shared ModelManager instance owned by the pipeline.

        Returns:
            dict keyed by (symbol, expiry_key) -> (model_artifact, ModelRecord).
            Pairs with no available model are absent from the dict.
        """
        model_map: dict[tuple[str, str], Any] = {}

        for symbol in self._settings.pairs:
            for expiry_key in _EXPIRY_KEY_MAP:
                await self._pull_model_artifacts(symbol, expiry_key, manager)
                result = self._load_best_model(symbol, expiry_key, manager)
                if result is not None:
                    model_map[(symbol, expiry_key)] = result

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
        manager: ModelManager,
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
                engine = await LiveEngine.create(symbol, expiry_key, model_manager=manager)

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

        # Retrain scheduler — receives the shared Storage and ModelManager instances
        assert self._storage is not None, (
            "Pipeline._storage must be initialised before _run_task_group(). "
            "Ensure _stage_storage() completed successfully."
        )
        assert self._manager is not None, (
            "Pipeline._manager must be initialised before _run_task_group(). "
            "Ensure _stage_model_load() completed successfully."
        )
        self._tasks.append(
            asyncio.create_task(
                self._retrain_scheduler(self._storage, self._manager),
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

    def _update_connection_flags(
        self,
        connected: bool,
        lost_logged: bool,
        restored_logged: bool,
    ) -> tuple[bool, bool]:
        if not connected and not lost_logged:
            status_store.add_event("Quotex not connected", event_type="kill")
            return True, False
        if connected and not restored_logged:
            status_store.add_event("Quotex connected", event_type="info")
            return False, True
        return lost_logged, restored_logged

    def _calc_elapsed_minutes(self, snapshot: dict) -> float:
        try:
            started_at = datetime.fromisoformat(snapshot["started_at"])
            return (datetime.now(timezone.utc) - started_at).total_seconds() / 60
        except Exception as exc:
            logger.error(
                {
                    "event": "ELAPSED_TIME_CALC_ERROR",
                    "error": str(exc),
                    "snapshot_keys": list(snapshot.keys()),
                    "function": "_quotex_status_loop",
                }
            )
            return 0.0

    async def _quotex_status_loop(self) -> None:
        """
        Poll each engine's QuotexDataStream every 30 s and push live account
        data to the dashboard StatusStore.

        Pushes: connection status, account balance, session win/loss/draw
        counts (derived from get_history()), and pending trade count.

        Only runs when USE_QUOTEX_STREAMING=True and at least one engine
        has a connected QuotexDataStream stream. Silently skips non-Quotex
        streams and disconnected clients.
        """
        connection_lost_logged: bool = False
        connection_restored_logged: bool = False

        if not self._settings.use_quotex_streaming:
            logger.info({"event": "QUOTEX_STATUS_DISABLED"})
            return

        # Give QuotexDataStream time to connect before first poll.
        await asyncio.sleep(30)

        while not self._shutdown_requested:
            try:
                stream = self._engines[0]._stream if self._engines else None
                if stream is None or not hasattr(stream, "_connected"):
                    await asyncio.sleep(30)
                    continue

                connected: bool = bool(getattr(stream, "_connected", False))
                connection_lost_logged, connection_restored_logged = (
                    self._update_connection_flags(
                        connected, connection_lost_logged, connection_restored_logged
                    )
                )

                balance: float = 0.0
                pending: int = 0
                if connected:
                    health = stream.health()
                    balance = health.get("balance", 0.0)
                    pending = health.get("pending_signals", 0)

                snapshot = status_store.get()
                elapsed_minutes = self._calc_elapsed_minutes(snapshot)

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
                logger.error({"event": "QUOTEX_STATUS_POLL_ERROR", "error": str(exc)})
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

    async def _train_model(
        self,
        model_name: str,
        trainer: Any,
        split: Any,
        symbol: str,
        expiry_key: str,
        manager: ModelManager,
        storage: Storage,
        first_boot: bool,
    ) -> None:
        try:
            if first_boot:
                # Check local registry first — works in both LOCAL and CLOUD mode.
                # pull_from_blob() returns None in LOCAL mode so would always
                # fall through and trigger an unnecessary retrain.
                if manager.get_best_model(symbol=symbol, expiry_key=expiry_key) is not None:
                    logger.info(
                        {
                            "event": "RETRAIN_MODEL_EXISTS",
                            "symbol": symbol,
                            "expiry_key": expiry_key,
                            "model_name": model_name,
                            "source": "local",
                        }
                    )
                    return
                # In CLOUD mode also check blob — local registry may be empty
                # on a fresh container that hasn't pulled artifacts yet.
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
                            "source": "blob",
                        }
                    )
                    return

            logger.info(
                {
                    "event": "RETRAIN_TRAINING_START",
                    "symbol": symbol,
                    "model_name": model_name,
                }
            )
            result = trainer.train(split, is_incremental=not first_boot)
            auc = result.metrics.get("auc", 0)
            logger.info(
                {
                    "event": "RETRAIN_TRAINING_COMPLETE",
                    "symbol": symbol,
                    "model_name": model_name,
                    "auc": round(auc, 4),
                }
            )
            manager.save(result)

        except Exception as e:
            logger.warning(
                {
                    "event": "RETRAIN_MODEL_FAILED",
                    "symbol": symbol,
                    "model_name": model_name,
                    "error": str(e),
                }
            )

    async def _reload_engines_for_symbol(self, symbol: str, expiry_key: str) -> None:
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

    async def _retrain_symbol(
        self,
        symbol: str,
        expiry_key: str,
        storage: Storage,
        manager: ModelManager,
        engineer: Any,
        first_boot: bool,
    ) -> None:
        try:
            max_rows: int | None = (
                None
                if (first_boot and self._settings.train_on_full_history)
                else self._settings.max_rf_rows
            )
            bars_df = storage.get_bars(
                symbol,
                timeframe=_TIMEFRAME_MAP.get(expiry_key, "M1"),
                max_rows=max_rows,
            )
            if bars_df is None or bars_df.empty:
                logger.warning(
                    {
                        "event": "RETRAIN_NO_DATA",
                        "symbol": symbol,
                        "expiry_key": expiry_key,
                        "timeframe": _TIMEFRAME_MAP.get(expiry_key, "M1"),
                        "message": "No bar data found, skipping",
                    }
                )
                return

            feature_matrix = engineer.build_matrix(
                bars_df, symbol, timeframe=_TIMEFRAME_MAP.get(expiry_key, "M1")
            )
            labels = Labeler(expiry_key=expiry_key).compute_labels(bars_df)
            split = DataShaper().split(feature_matrix, labels, expiry_key)

            models_to_try = [
                # ("XGBoost", XGBoostTrainer(expiry_key=expiry_key)),
                # ("RandomForest", RandomForestTrainer(expiry_key=expiry_key)),  # excluded: 4.67 GiB artifact breaks container boot
                # ("LightGBM", LightGBMTrainer(expiry_key=expiry_key)),
                ("CatBoost", CatBoostTrainer(expiry_key=expiry_key)),
                # TODO: LSTM/GRU/TCN need AUC-compatible training loop before re-enabling
            ]

            for model_name, trainer in models_to_try:
                await self._train_model(
                    model_name,
                    trainer,
                    split,
                    symbol,
                    expiry_key,
                    manager,
                    storage,
                    first_boot,
                )

            record = manager.get_best_model(symbol=symbol, expiry_key=expiry_key)
            if record is None:
                logger.warning(
                    {
                        "event": "RETRAIN_NO_BEST_MODEL",
                        "symbol": symbol,
                        "expiry_key": expiry_key,
                    }
                )
                return

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
            await self._reload_engines_for_symbol(symbol, expiry_key)

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

    async def _retrain_scheduler(self, storage: Storage, manager: ModelManager) -> None:
        """
        Background loop that triggers periodic model retraining.

        Wakes every _RETRAIN_CHECK_INTERVAL_S seconds and checks whether
        model_retrain_interval has elapsed since the last retrain. If so,
        runs a full training cycle for each symbol, saves the
        artifact, and hot-reloads each matching LiveEngine.

        Uses the shared Stage 1 Storage and pipeline ModelManager — does not
        construct additional instances.

        Individual symbol failures are caught and logged as warnings. The
        scheduler continues to the next symbol and does not abort.

        Args:
            storage: Stage 1 Storage instance shared from run().
            manager: Shared ModelManager instance owned by the pipeline.
        """
        engineer = get_feature_engineer()
        inv_map: dict[int, str] = _SECONDS_TO_EXPIRY_KEY
        is_first_run: bool = True

        while not self._shutdown_requested:
            # First boot: run immediately without sleeping so missing models
            # are trained as soon as engines are up. Subsequent cycles sleep
            # for _RETRAIN_CHECK_INTERVAL_S (1 h) before re-checking.
            if not is_first_run:
                await asyncio.sleep(_RETRAIN_CHECK_INTERVAL_S)

            if self._shutdown_requested:
                break

            now: datetime = datetime.now(timezone.utc)
            elapsed: float = (now - self._last_retrain_time).total_seconds()

            if not is_first_run and elapsed < self._settings.model_retrain_interval:
                continue

            first_boot = is_first_run
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

            for expiry_key in _EXPIRY_KEY_MAP:
                for symbol in self._settings.pairs:
                    await self._retrain_symbol(
                        symbol, expiry_key, storage, manager, engineer, first_boot
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
