"""
src/engine/live.py — The Conductor.

Role: Orchestrate the real-time trading pipeline for a single symbol
and expiry combination. Pulls live ticks from the correct stream,
engineers features, generates a signal every bar, journals every result,
and fires the webhook for executable signals.

Architecture
------------
LiveEngine is single-symbol, single-expiry. pipeline.py instantiates
one LiveEngine per active (symbol, expiry_key) pair and runs each as
a separate asyncio task. Multiple instances share the same Storage
singleton (which owns the threading lock) and the same FeatureEngineer
singleton (stateless, thread-safe). Each instance owns its own
SignalGenerator, WebhookSender, Reporter, and Journal.

Stream routing
--------------
Live tick source is determined once at construction from config:

    USE_QUOTEX_STREAMING=True  -> QuotexDataStream  (OTC symbols via Quotex)
    USE_QUOTEX_STREAMING=False -> TwelveTicksStream (standard forex pairs)

TwelveData is ALWAYS used by historian.py for historical backfill
regardless of this setting. This routing only governs LIVE ticks.
The OTC symbol translation (EUR_USD -> EURUSD_otc) is done by
config.quotex_symbols — live.py never constructs symbol names.

Cold-start
----------
On container restart /models/ is empty. __init__ calls
ModelManager.pull_from_blob() before constructing SignalGenerator so
the inference loop always starts with a model if one has been trained
and uploaded. If pull_from_blob() finds nothing, SignalGenerator runs
in SKIP-only mode and logs a prominent warning.

Fail-fast pattern
-----------------
__init__ fails hard on any infrastructure failure:
    - Storage init failure (disk, permissions)         -> StorageError  -> crash
    - Stream construction failure (bad credentials)    -> Exception     -> crash
    - ModelManager init failure                        -> Exception     -> crash
    - Journal/WebhookSender/Reporter failures at init  -> Fatal         -> crash

The rationale: infrastructure must be verified at boot. A partially
initialised engine that silently skips storage or the stream is more
dangerous than a crash that surfaces the problem immediately.

OOM protection
--------------
_BAR_WINDOW caps the bars loaded per tick at 100 rows * ~50 float64
columns = ~40KB per call. This is negligible. The risk is transform()
being called on a large DataFrame unnecessarily — we cap at _BAR_WINDOW
and transform() runs on that capped frame, not the full history.

Feature DataFrames are local to _process_tick() and are garbage
collected after each bar. No feature data accumulates in memory across
ticks.

Thread safety
-------------
Storage owns its own threading.Lock and serialises all Parquet I/O.
FeatureEngineer is stateless and thread-safe by design (documented in
features.py). SignalGenerator holds mutable state (_model, _record) and
must not be called concurrently — the asyncio single-threaded event loop
guarantees this since _process_tick() is awaited sequentially.

reload_model() is the one unsafe operation: it mutates SignalGenerator
state while the loop may be between ticks. pipeline.py must call it
only when the loop is between _process_tick() calls, which is guaranteed
if it is called from another asyncio task (the event loop serialises
coroutines). Do NOT call reload_model() from a separate OS thread.

Error handling
--------------
Per-bar recoverable errors  -> warning_block + logger.warning + return
Fatal infrastructure errors -> error_block + logger.critical + raise

Public API
----------
    LiveEngine(symbol, expiry_key)
    LiveEngine.run()          -> coroutine, blocks until stop() or crash
    LiveEngine.stop()         -> graceful shutdown after current tick
    LiveEngine.reload_model() -> hot-swap model artifact mid-session
"""

from __future__ import annotations

import asyncio
import logging
import pandas as pd


from typing import Any
from datetime import datetime, timedelta, timezone
from core.config import get_settings
from data.journal import Journal, SignalEntry, TradeEntry
from data.storage import (
    Storage,
    get_storage,
    StorageError,
)
from ml_engine.features import (
    BINARY_EXPIRY_RULES,
    FeatureEngineer,
    FeatureEngineerError,
    get_feature_engineer,
)
from engine.bar_builder import LiveBarBuilder
from ml_engine.model import Bar, Tick as ModelTick
from ml_engine.model_manager import ModelManager
from engine.twelveticks_stream import TwelveDataStream
from engine.quotex_stream import QuotexDataStream
from trading.signals import SignalGenerator, SignalGeneratorError, TradeSignal
from trading.threshold_manager import ThresholdManager
from core.exceptions import (
    LiveEngineError,
    LiveEngineValueError,
)
from trading.webhook import WebhookSender
from trading.reporter import Reporter

logger = logging.getLogger(__name__)

# ── Module Constants ──────────────────────────────────────────────────────────

# Minimum M1 bars before inference starts.
# MACD slow span = 26 bars. _MIN_BARS_REQUIRED adds a safety margin
# so the first get_latest() call never hits the empty-DataFrame guard.
_MIN_BARS_REQUIRED: int = 30

# Maximum bars loaded from storage per tick.
# 100 M1 bars * ~50 float64 columns * 8 bytes = ~40KB per call.
# Capped to prevent OOM when storage holds years of history.
_BAR_WINDOW: int = 100

# Maximum consecutive per-bar errors before the engine self-terminates.
# Protects against a silent infinite loop of recoverable-looking errors
# that are actually a symptom of a deeper unrecoverable failure.
_MAX_CONSECUTIVE_ERRORS: int = 20

_TIMEFRAME_MAP = {"1_MIN": "M1", "5_MIN": "M5", "15_MIN": "M15"}


class LiveEngine:
    """
    The Conductor: real-time stream -> features -> signal -> execution.

    One instance per (symbol, expiry_key) pair. Constructed and managed
    by pipeline.py. Runs as an asyncio coroutine until stop() is called
    or a fatal error occurs.

    Attributes:
        symbol:     Currency pair this engine trades, e.g. "EUR_USD".
        expiry_key: Expiry window, e.g. "1_MIN".
        is_running: True while the inference loop is active.
    """

    def __init__(
        self,
        symbol: str,
        expiry_key: str = "1_MIN",
    ) -> None:
        """
        Initialise the LiveEngine and all downstream components.

        Follows the Dictator pattern from config.py and storage.py:
        either every component initialises successfully or the engine
        raises immediately. There is no partially-initialised state
        that silently degrades.

        Component init order matters:
            1. Config + validation  (fail-hard)
            2. Storage              (fail-hard — no storage = cannot run)
            3. FeatureEngineer      (fail-hard — singleton, stateless)
            4. Stream               (fail-hard — no stream = no data)
            5. ModelManager cold-start (warning-only — SKIP mode if absent)
            6. SignalGenerator      (fail-hard — invalid expiry_key)
            7. Journal              (warning-only — degrade gracefully)
            8. WebhookSender        (warning-only — degrade gracefully)
            9. Reporter             (warning-only — degrade gracefully)

        Args:
            symbol:     Currency pair to trade, e.g. "EUR_USD".
            expiry_key: Expiry window. Must be one of the keys in
                        BINARY_EXPIRY_RULES from features.py.

        Raises:
            ValueError:      If symbol is empty or expiry_key is invalid.
            StorageError:    If Storage cannot provision its directories.
            LiveEngineError: If the stream cannot be constructed.
        """
        # ── Fail-fast validation ──────────────────────────────────────────
        if not symbol or not symbol.strip():
            raise LiveEngineValueError(
                message="LiveEngine requires a non-empty symbol", field_name="symbol"
            )

        if expiry_key not in BINARY_EXPIRY_RULES:
            raise LiveEngineValueError(
                message=f"Invalid expiry_key: '{expiry_key}'. Must be one of {list(BINARY_EXPIRY_RULES.keys())}",
                field_name="expiry_key",
            )

        self._settings = get_settings()
        self.symbol: str = symbol.strip()
        self.expiry_key: str = expiry_key
        self.is_running: bool = False

        # Tracks consecutive per-bar errors for the self-termination guard.
        self._consecutive_errors: int = 0

        # Running counters pushed to the dashboard on every tick.
        self._ticks_processed: int = 0
        self._signals_fired: int = 0

        # M1 bar-close gate: only run inference when the UTC minute advances.
        # -1 ensures the very first tick of a session is always processed.
        self._last_processed_minute: int = -1

        # Live bar builder: accumulates ticks into a completed M1 OHLCV bar so
        # every inference cycle runs on fresh price data rather than static Parquet.
        self._bar_builder: LiveBarBuilder = LiveBarBuilder(symbol=self.symbol)

        # Reload lock: prevents reload_model() from mutating SignalGenerator
        # state while _process_tick() is mid-inference. Both the asyncio
        # event loop and any pipeline.py coroutine use this.
        self._reload_lock: asyncio.Lock = asyncio.Lock()
        # Inter-trade cooldown: minimum seconds between executed signals.
        # Prevents firing a second trade before the previous one settles.
        self._min_trade_gap_seconds: int = 80
        self._last_execution_time: datetime | None = None

        # ── 1. Storage (fail-hard) ────────────────────────────────────────
        # StorageError propagates immediately — no engine without storage.
        self._storage: Storage = get_storage()

        # ── 2. Feature Engineering (fail-hard) ────────────────────────────
        # get_feature_engineer() returns the module singleton.
        # Stateless and thread-safe — shared across all LiveEngine instances.
        self._engineer: FeatureEngineer = get_feature_engineer()

        # ── 3. Stream (fail-hard) ─────────────────────────────────────────
        self._stream: Any = self._init_stream()

        # ── 4. Model Manager + cold-start ─────────────────────────────────
        # Instantiated once here and reused.
        self._model_manager: ModelManager = ModelManager(
            storage_dir=self._settings.model_dir
        )

        # ── 5. Signal Generator (fail-hard) ───────────────────────────────
        self._signal_gen: SignalGenerator = SignalGenerator(
            symbol=self.symbol,
            expiry_key=self.expiry_key,
        )

        # ── 5a. Threshold Manager ─────────────────────────────────────────
        # Constructed from config; injected into SignalGenerator so that
        # generate() uses the dynamic threshold rather than the static value.
        self._threshold_mgr: ThresholdManager = ThresholdManager(
            base_threshold=self._settings.confidence_threshold,
            step=self._settings.martingale_step,
            max_streak=self._settings.martingale_max_streak,
        )

        # ── 5b. Threshold Manager ─────────────────────────────────────────
        # Assign the threshold manager to the current signal generator
        self._signal_gen.set_threshold_manager(self._threshold_mgr)

        # ── 6. Journal ────────────────────────────────────────────────────
        self._journal_client: Journal = self._init_journal()

        # ── 7. WebhookSender  ─────────────────────────────────────────────
        self._webhook: WebhookSender = self._init_webhook()

        # ── 8. Reporter ───────────────────────────────────────────────────
        self._reporter: Reporter = self._init_reporter()
        self._reporter.set_journal(self._journal_client)

        logger.info(
            {
                "event": "LIVE_ENGINE_READY",
                "symbol": self.symbol,
                "expiry_key": self.expiry_key,
                "stream": type(self._stream).__name__,
                "model": (
                    "LOADED" if self._signal_gen._model is not None else "SKIP_MODE"
                ),
                "journal": "OK",
                "webhook": "OK",
                "reporter": "OK",
            }
        )

    # ── Initialisation Helpers ────────────────────────────────────────────────
    @classmethod
    async def create(
        cls,
        symbol: str,
        expiry_key: str = "1_MIN",
    ) -> "LiveEngine":
        """
        Async factory method to create and initialize a LiveEngine.

        This is the recommended way to instantiate LiveEngine since it
        properly awaits the async _cold_start() method.

        Args:
            symbol: Currency pair to trade, e.g. "EUR_USD".
            expiry_key: Expiry window, e.g. "1_MIN".

        Returns:
            LiveEngine: Fully initialized and ready to run.
        """
        # Create instance (__init__ does NOT call _cold_start)
        engine = cls(symbol, expiry_key)

        # Connect the stream if it's QuotexDataStream
        if hasattr(engine._stream, "connect"):
            connected = await engine._stream.connect()
            if not connected:
                raise LiveEngineError(
                    f"Failed initial connection to Quotex for {symbol}"
                )

        # Now await the async cold start to pull the model artifact before the inference loop starts.
        await engine._cold_start()

        # Reload with lock for safety, the model into SignalGenerator now that artifact is on disk
        async with engine._reload_lock:
            if engine._signal_gen._model is None:
                engine._signal_gen.reload()

        return engine

    def _init_stream(self) -> QuotexDataStream | TwelveDataStream:
        """
        Construct the correct live tick stream from config.

        Quotex is used for OTC live ticks when USE_QUOTEX_STREAMING=True.
        TwelveData is used for standard forex live ticks otherwise.
        TwelveData is ALWAYS used by historian.py for backfill — this
        method only governs the live tick source.

        Returns:
            Stream instance with a subscribe() async generator method.

        Raises:
            LiveEngineError: If the stream cannot be constructed. This is
                fatal — the engine cannot run without a tick source.
        """
        try:
            if self._settings.use_quotex_streaming:
                otc_symbol: str = self._settings.quotex_symbols[self.symbol]
                logger.info(
                    {
                        "event": "STREAM_INITIALIZED",
                        "symbol": self.symbol,
                        "type": "QuotexDataStream",
                        "otc_symbol": otc_symbol,
                    }
                )
                return QuotexDataStream(
                    email=self._settings.quotex_email,
                    password=self._settings.quotex_password,
                    practice_mode=self._settings.practice_mode,
                    symbol=otc_symbol,
                )
            else:
                logger.info(
                    {
                        "event": "STREAM_INITIALIZED",
                        "symbol": self.symbol,
                        "type": "TwelveDataStream",
                    }
                )
                return TwelveDataStream(
                    api_key=self._settings.twelvedata_api_key,
                    pairs=self._settings.pairs,
                    storage=self._storage,
                    flush_size=self._settings.tick_flush_size,
                )

        except Exception as exc:
            raise LiveEngineError(
                f"Stream construction failed for {self.symbol}: {exc}",
                stage="init_stream",
            ) from exc

    async def _cold_start(self) -> None:
        """
        Restore the latest model artifact from Azure Blob before inference.

        Must be called BEFORE SignalGenerator is constructed so that
        get_best_model() finds the artifact on disk. Uses the shared
        self._model_manager instance to avoid a second ModelManager
        construction.

        In LOCAL mode the blob pull is skipped entirely — the pipeline
        loads the model from disk and injects it via inject_model().
        In CLOUD mode, a missing artifact logs a warning-only no-op and
        SignalGenerator will run in SKIP-only mode.
        """
        if self._settings.data_mode == "LOCAL":
            logger.debug(
                {
                    "event": "COLD_START_SKIPPED",
                    "reason": "DATA_MODE=LOCAL",
                }
            )
            return

        try:
            result = await self._model_manager.pull_from_blob(
                symbol=self.symbol,
                expiry_key=self.expiry_key,
            )
        except Exception as exc:
            logger.warning(
                {
                    "event": "COLD_START_PULL_FAILED",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                    "error": str(exc),
                }
            )
            return

        if result:
            logger.info(
                {
                    "event": "COLD_START_RESTORED",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                    "artifact": result,
                }
            )
        else:
            logger.warning(
                {
                    "event": "COLD_START_NO_MODEL",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                }
            )

    def _init_journal(self) -> Journal:
        """
        Attempt to construct a Journal instance.
        Failure is fatal.

        Returns:
            Journal instance.
        """
        try:
            logger.info({"event": "JOURNAL_INITIALIZING"})
            return Journal()
        except Exception as exc:
            raise LiveEngineError(
                f"Journal initialization failed: {exc}", stage="init_journal"
            ) from exc

    def _init_webhook(self) -> WebhookSender:
        """
        Attempt to construct a WebhookSender instance.
        Failure is fatal.

        Returns:
            WebhookSender instance.
        """
        try:
            logger.info({"event": "WEBHOOK_INITIALIZING"})
            w = WebhookSender(
                url=self._settings.webhook_url,
                secret=self._settings.webhook_secret,
            )
            return w
        except Exception as exc:
            raise LiveEngineError(
                f"Webhook initialization failed: {exc}", stage="init_webhook"
            ) from exc

    def _init_reporter(self) -> Reporter:
        """
        Attempt to construct a Reporter instance.
        Failure is fatal.

        Returns:
            Reporter instance.
        """
        try:
            logger.info({"event": "REPORTER_INITIALIZING"})
            r = Reporter(
                telegram_token=self._settings.telegram_token,
                telegram_chat_id=self._settings.telegram_chat_id,
                discord_webhook_url=self._settings.discord_webhook_url,
                orchestrator=None,  # wired by pipeline.py after full init
            )
            return r
        except Exception as exc:
            raise LiveEngineError(
                f"Reporter initialization failed: {exc}", stage="init_reporter"
            ) from exc

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start the infinite trading loop.

        Subscribes to the tick stream and processes each tick through the
        full pipeline. Runs until stop() is called, _MAX_CONSECUTIVE_ERRORS
        is exceeded, or a fatal StorageError propagates from _process_tick().

        Per-bar recoverable errors are caught inside _process_tick() and
        increment self._consecutive_errors. If _MAX_CONSECUTIVE_ERRORS is
        reached, the engine self-terminates to prevent a silent infinite
        loop of failures.

        Fatal StorageError and unhandled stream exceptions propagate to
        the caller (pipeline.py) which decides whether to restart or abort.

        Raises:
            StorageError:    If storage becomes unavailable mid-run.
            LiveEngineError: If _MAX_CONSECUTIVE_ERRORS is exceeded.
            Exception:       Any unhandled stream or infrastructure failure.
        """
        logger.info(
            {
                "event": "LIVE_ENGINE_STARTING",
                "symbol": self.symbol,
                "expiry_key": self.expiry_key,
                "stream": type(self._stream).__name__,
            }
        )
        self.is_running = True
        self._consecutive_errors = 0

        # ── Reset dashboard session clock to actual engine start time ─────
        # started_at is seeded at StatusStore.__init__ (module import time).
        # Resetting it here means elapsed_minutes counts from when the engine
        # actually begins trading, not from container boot / backfill time.
        self._reporter.push_dashboard(self._build_context("session_reset"))

        # Start result consumer task (only for Quotex)
        result_task = None
        if hasattr(self._stream, "get_result"):
            result_task = asyncio.create_task(self._consume_results())
            logger.info(
                {
                    "event": "LIVE_ENGINE_CONSUMER_STARTED",
                }
            )
        # Start result producer task (only for Quotex) — resolves pending signals
        result_producer_task = None
        if hasattr(self._stream, "poll_results"):
            result_producer_task = asyncio.create_task(self._stream.poll_results())
            logger.info(
                {
                    "event": "LIVE_ENGINE_RESULT_PRODUCER_STARTED",
                }
            )

        try:
            async for tick in self._stream.subscribe():
                if not self.is_running:
                    break

                await self._process_tick(tick)

                # Self-termination guard: if too many consecutive errors
                # have accumulated, something is structurally wrong.
                if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    self._reporter.push_dashboard(self._build_context("fatal_error"))
                    self.is_running = False
                    raise LiveEngineError(
                        f"Consecutive error limit ({_MAX_CONSECUTIVE_ERRORS}) "
                        f"exceeded for {self.symbol} {self.expiry_key}.",
                        stage="run",
                    )

        except (StorageError, LiveEngineError):
            self.is_running = False
            raise  # Propagate fatal errors to pipeline.py

        except Exception as exc:
            logger.critical(
                {
                    "event": "LIVE_ENGINE_UNHANDLED_EXCEPTION",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            self._reporter.push_dashboard(self._build_context("engine_crash"))
            self.is_running = False
            raise  # Propagate fatal errors to pipeline.py

        finally:
            # Clean up result consumer
            if result_task:
                result_task.cancel()
                await result_task
            if result_producer_task:
                result_producer_task.cancel()
                await result_producer_task
            logger.info(
                {
                    "event": "LIVE_ENGINE_STOPPED",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                }
            )

    async def _process_tick(self, tick: Any) -> None:
        """
        Process a single tick through the full inference pipeline.

        All per-bar recoverable errors are caught here, logged with a
        warning_block, and the method returns early. The caller (run())
        continues on the next tick. Fatal StorageError propagates.

        OOM note: bars_df and fe_df are local to this method and are
        garbage collected after each call. No feature data accumulates
        across ticks.

        Thread safety note: _signal_gen.generate() is called sequentially
        in the asyncio event loop. The asyncio.Lock in reload_model()
        prevents reload_model() from mutating _signal_gen state while
        this method is between awaits.

        Args:
            tick: Tick object from the stream. Type varies by stream.
        """
        self._ticks_processed += 1

        # ── Accumulate tick into the live bar builder ─────────────────────
        # Every tick updates the in-progress M1 OHLCV bar so that when the
        # minute gate fires, the bar is built from real price data rather than
        # being absent entirely.  Ticks that arrive before a valid ModelTick
        # can be constructed (e.g. from streams that pass raw dicts) are skipped.
        if isinstance(tick, ModelTick):
            self._bar_builder.add_tick(tick)

        # ── M1 bar-close gate ─────────────────────────────────────────────
        # QuotexDataStream polls every ~1 s. Without this guard the full feature
        # engineering + model inference pipeline fires ~60× per minute.
        # We run inference exactly once per UTC minute (i.e. once per M1 bar).
        _now_minute = datetime.now(timezone.utc).minute
        if _now_minute == self._last_processed_minute:
            return
        self._last_processed_minute = _now_minute
        self._reporter.push_dashboard(self._build_context("tick"))

        # ── 1. Load recent bars (capped at _BAR_WINDOW - 1) ──────────────
        # We reserve one slot for the live bar built from this minute's ticks.
        # StorageError propagates to run() as a fatal failure.
        bars_df: pd.DataFrame | None = self._storage.get_bars(
            symbol=self.symbol,
            timeframe=_TIMEFRAME_MAP.get(self.expiry_key, "M1"),
            max_rows=_BAR_WINDOW - 1,
        )

        if bars_df is None or bars_df.empty:
            logger.debug(
                {
                    "event": "TICK_PROCESSING_NO_BARS",
                    "symbol": self.symbol,
                    "message": f"No M1 bars for {self.symbol} - awaiting first bar",
                }
            )
            self._bar_builder.close_and_reset(pd.DatetimeIndex([]))
            return

        # Narrow from DataFrame | None to DataFrame for the rest of this scope.
        hist_bars: pd.DataFrame = bars_df

        # ── 2. Close the live bar and append it to the historical window ──
        # close_and_reset() finalises the bar built from this minute's ticks and
        # returns the ticks DataFrame for micro-structure feature engineering.
        live_bar_df, ticks_df = self._bar_builder.close_and_reset(
            pd.DatetimeIndex(hist_bars.index)
        )
        if live_bar_df is not None:
            merged: pd.DataFrame = pd.concat([hist_bars, live_bar_df])
            # Drop duplicate timestamps if historian already wrote this bar.
            deduped: pd.DataFrame = merged.loc[~merged.index.duplicated(keep="last")]
            hist_bars = deduped.sort_index()
            # Write to disk so the next minute's get_bars() returns a rolling window.
            try:
                _row = live_bar_df.iloc[0]
                _bar_ts: pd.Timestamp = pd.DatetimeIndex(live_bar_df.index)[0]  # type: ignore[assignment]
                self._storage.save_bar(
                    Bar(
                        timestamp=_bar_ts,
                        symbol=self.symbol,
                        open=float(_row["open"]),
                        high=float(_row["high"]),
                        low=float(_row["low"]),
                        close=float(_row["close"]),
                        volume=float(_row["volume"]),
                    )
                )
            except Exception as _exc:
                logger.debug(
                    {
                        "event": "LIVE_BAR_PERSIST_FAILED",
                        "symbol": self.symbol,
                        "error": str(_exc),
                    }
                )

        # ── 3. Minimum bars warmup guard ──────────────────────────────────
        if len(hist_bars) < _MIN_BARS_REQUIRED:
            logger.debug(
                {
                    "event": "MINIMUM_BARS_WARMUP",
                    "bars_loaded": len(hist_bars),
                    "bars_required": _MIN_BARS_REQUIRED,
                    "symbol": self.symbol,
                    "message": f"Warmup {len(hist_bars)}/{_MIN_BARS_REQUIRED} bars for {self.symbol}",
                }
            )
            return

        # ── 4. Feature engineering ────────────────────────────────────────
        # transform() and get_latest() receive the live ticks so that
        # TICK_VELOCITY, SPREAD_NORMALIZED, TICK_DELTA_CUMULATIVE and
        # ORDER_FLOW_IMBALANCE are computed from real data instead of
        # being zero-filled.
        live_ticks: pd.DataFrame | None = ticks_df if not ticks_df.empty else None
        try:
            fe_df: pd.DataFrame = self._engineer.transform(
                hist_bars,
                live_ticks,
                timeframe=_TIMEFRAME_MAP.get(self.expiry_key, "M1"),
            )

            fv = self._engineer.get_latest(
                hist_bars,
                live_ticks,
                timeframe=_TIMEFRAME_MAP.get(self.expiry_key, "M1"),
            )
        except FeatureEngineerError as exc:
            logger.warning(
                {
                    "event": "FEATURE_ENGINEERING_FAILED",
                    "symbol": self.symbol,
                    "bars_loaded": len(bars_df),
                    "error": str(exc),
                }
            )
            self._consecutive_errors += 1
            return

        # ── 4. Signal generation (gate check + model inference) ───────────
        try:
            async with self._reload_lock:
                signal: TradeSignal = self._signal_gen.generate(
                    fv=fv,
                    fe_df=fe_df,
                )
        except SignalGeneratorError as exc:
            logger.warning(
                {
                    "event": "SIGNAL_GENERATION_FAILED",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                    "error": str(exc),
                }
            )
            self._consecutive_errors += 1
            return

        # Reset consecutive error counter on a successful inference.
        self._consecutive_errors = 0
        if signal.direction != "SKIP":
            self._signals_fired += 1
            self._reporter.push_dashboard(self._build_context("signal", signal=signal))
        else:
            self._reporter.push_dashboard(self._build_context("skip", signal=signal))

        # ── 5. Journal every signal (including SKIP) ──────────────────────
        self._write_journal(signal)

        # ── 6. Kill switch gate ───────────────────────────────────────────
        if self._threshold_mgr.is_halted():
            self._on_kill_switch_activated()
            return

        # ── 7. Execute only if signal is actionable ───────────────────────
        # is_executable() checks direction != SKIP only — the model's
        # confidence threshold is the sole filter; no redundant gate check.
        # Inter-trade cooldown prevents firing before the previous trade settles.
        if signal.is_executable():
            now = datetime.now(timezone.utc)
            if (
                self._last_execution_time is not None
                and (now - self._last_execution_time).total_seconds()
                < self._min_trade_gap_seconds
            ):
                logger.debug(
                    {
                        "event": "TRADE_COOLDOWN_ACTIVE",
                        "symbol": self.symbol,
                        "direction": signal.direction,
                        "confidence": signal.confidence,
                        "seconds_since_last": round(
                            (now - self._last_execution_time).total_seconds(), 1
                        ),
                        "cooldown_required": self._min_trade_gap_seconds,
                    }
                )
                self._reporter.push_dashboard(
                    self._build_context("cooldown", signal=signal)
                )
            else:
                await self._execute(signal)
                self._last_execution_time = datetime.now(timezone.utc)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute(self, signal: TradeSignal) -> None:
        """
        Fire the webhook and notify the reporter for an executable signal.

        Only called when signal.is_executable() is True. Passes the full
        TradeSignal to both components — they extract the fields they need.
        Failures in webhook or reporter are fatal and will crash
        the engine. The signal is already journaled before this is called.

        Args:
            signal: Executable TradeSignal (CALL or PUT).
        """
        logger.info(
            {
                "event": "ATTEMPTING_TO_EXECUTE_SIGNAL",
                "symbol": signal.symbol,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "expiry_key": signal.expiry_key,
                "model_name": signal.model_name,
            }
        )

        try:
            await self._webhook.fire(signal)
            logger.info(
                {
                    "event": "WEBHOOK_SENT_SUCCESS",
                    "symbol": self.symbol,
                }
            )
        except Exception as exc:
            logger.critical(
                {
                    "event": "WEBHOOK_FIRE_FAILED",
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "error": str(exc),
                }
            )
            self._reporter.push_dashboard(
                self._build_context("webhook_failed", signal=signal)
            )
            raise LiveEngineError(
                f"Webhook fire failed for {signal.symbol} {signal.direction}: {exc}",
                stage="webhook_fire",
            ) from exc

        # Register signal for result tracking via Quotex history matching
        if hasattr(self._stream, "register_pending"):
            _EXPIRY_SECONDS_MAP = {"1_MIN": 60, "5_MIN": 300, "15_MIN": 900}
            expiry_seconds = _EXPIRY_SECONDS_MAP.get(signal.expiry_key, 60)
            expiry_time = datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds)
            self._stream.register_pending(
                signal_id=f"{signal.symbol}_{int(datetime.now(timezone.utc).timestamp())}",
                signal={
                    "pair": signal.symbol,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "expiry_key": signal.expiry_key,
                },
                expiry_time=expiry_time,
            )

        try:
            await self._reporter.notify(signal)
        except Exception as exc:
            logger.warning(
                {
                    "event": "REPORTER_NOTIFICATION_FAILED",
                    "symbol": signal.symbol,
                    "error": str(exc),
                }
            )

    def _write_journal(self, signal: TradeSignal) -> None:
        """
        Write a signal to the trade journal.

        Called for every signal including SKIP. The journal_client is instantiated
        once in __init__ and reused, matching the singleton pattern of
        all other components.

        Args:
            signal: TradeSignal to record. to_dict() produces the
                    JSON-serialisable dict the journal consumes.
        """
        try:
            entry = SignalEntry(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                confidence=signal.confidence,
                direction=signal.direction,
                model_version=signal.model_name,
                metadata=getattr(signal, "metadata", "{}"),
            )
            self._journal_client.record_signal(entry)
        except Exception as exc:
            logger.critical(
                {
                    "event": "JOURNAL_WRITE_FAILED",
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "error": str(exc),
                }
            )
            self._reporter.push_dashboard(
                self._build_context("journal_failed", signal=signal)
            )
            raise LiveEngineError(
                f"Journal write failed for {signal.symbol}: {exc}",
                stage="journal_write",
            ) from exc

    def _on_kill_switch_activated(self) -> None:
        """
        Handle kill switch activation — log prominently and stop the engine.

        Called when ThresholdManager.is_halted() returns True, either
        because consume_result() just crossed the max_streak threshold or
        because _process_tick() detected the halted state after a tick.
        """
        logger.critical(
            {
                "event": "KILL_SWITCH_ACTIVATED",
                "symbol": self.symbol,
                "streak": self._threshold_mgr.streak,
                "max_streak": self._threshold_mgr.max_streak,
                "effective_threshold": self._threshold_mgr.get_threshold(),
            }
        )
        self._reporter.push_dashboard(self._build_context("kill_switch"))

    def _build_context(
        self,
        event_type: str = "tick",
        signal: TradeSignal | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a complete context dict for Reporter.push_dashboard().

        Gathers all metrics in one place so Reporter doesn't need to
        reach back into LiveEngine state.

        Args:
            event_type: "tick", "signal", "cooldown", "result", "kill_switch".
            signal:     TradeSignal when event_type is signal/cooldown.
            result:     Result dict when event_type is result.

        Returns:
            Flat dict with all fields Reporter.push_dashboard() needs.
        """
        connected = bool(getattr(self._stream, "_connected", True))
        balance = float(getattr(self._stream, "_balance", 0.0))

        pending_count = len(getattr(self._stream, "_pending", {}))

        return {
            "event_type": event_type,
            "symbol": self.symbol,
            "expiry_key": self.expiry_key,
            "is_running": self.is_running,
            "ticks_processed": self._ticks_processed,
            "signals_fired": self._signals_fired,
            "connected": connected,
            "balance": balance,
            "threshold_state": self._threshold_mgr.get_state(),
            "signal": signal,
            "result": result,
            "session": {},
            "started_at": None,
            "pending_signals": pending_count,
            "practice_mode": self._settings.practice_mode,
        }

    def _write_trade_entry(self, result: dict[str, Any]) -> None:
        """Write a TradeEntry to the journal for Recent Trades display."""
        try:
            entry_price = float(result.get("open_price", 0) or 0)
            exit_price = float(result.get("close_price", 0) or 0)
            if entry_price == 0 or exit_price == 0:
                return
            outcome = result.get("result", "")
            pnl = (
                result.get("payout", 0.0)
                if outcome == "win"
                else -result.get("stake", 0.0)
            )
            entry = TradeEntry(
                timestamp=datetime.now(timezone.utc),
                symbol=result.get("pair", self.symbol),
                side=result.get("direction", "CALL"),
                entry_price=entry_price,
                exit_price=exit_price,
                result=float(pnl),
                duration_seconds=int(result.get("duration_seconds", 60)),
                signal_id=result.get("signal_id", ""),
            )
            self._journal_client.record_trade(entry)
        except Exception as exc:
            logger.warning({"event": "TRADE_ENTRY_WRITE_FAILED", "error": str(exc)})

    def _process_result(self, result: dict[str, Any]) -> None:
        """Apply a resolved trade result: update thresholds, check kill switch, log."""
        outcome = result["result"]
        if outcome == "win":
            self._threshold_mgr.on_win()
        elif outcome == "loss":
            self._threshold_mgr.on_loss()
        # draw: streak and threshold unchanged

        # Write TradeEntry for Recent Trades
        self._write_trade_entry(result)

        # Push result event to dashboard (includes session counters, recent trades)
        self._reporter.push_dashboard(self._build_context("result", result=result))

        # Kill switch after dashboard push so "result" appears first
        if self._threshold_mgr.is_halted():
            self._on_kill_switch_activated()

        logger.info(
            {
                "event": "TRADE_RESULT_PROCESSED",
                "signal_id": result.get("signal_id"),
                "result": outcome,
                "payout": result.get("payout"),
                "stake": result.get("stake"),
                "streak": self._threshold_mgr.streak,
                "threshold": self._threshold_mgr.get_threshold(),
            }
        )

    async def _consume_results(self) -> None:
        """Background task to consume trade results from QuotexDataStream."""
        while self.is_running:
            try:
                if not hasattr(self._stream, "get_result"):
                    await asyncio.sleep(1)
                    continue

                try:
                    async with asyncio.timeout(0.5):
                        result: dict[str, Any] | None = await self._stream.get_result()
                except TimeoutError:
                    result = None
                if result:
                    self._process_result(result)
                    await self._reporter.notify_result(result)
            except Exception as exc:
                logger.debug({"event": "RESULT_CONSUMER_ERROR", "error": str(exc)})

            await asyncio.sleep(0.5)

    def get_threshold_state(self) -> dict[str, object]:
        """
        Return the current ThresholdManager state snapshot.

        Convenience accessor for pipeline.py and tests. Delegates directly
        to ThresholdManager.get_state().

        Returns:
            dict with keys: streak, effective_threshold, base_threshold,
            step, max_streak, halted, history_len.
        """
        return self._threshold_mgr.get_state()

    # ── Control ───────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """
        Signal the inference loop to stop after the current tick completes.

        Sets is_running=False which is checked at the top of the async for
        loop in run(). The loop exits cleanly after the current
        _process_tick() call returns. Does not cancel the asyncio task —
        call task.cancel() from pipeline.py for immediate termination.
        """
        logger.info(
            {
                "event": "LIVE_ENGINE_STOP_REQUESTED",
                "symbol": self.symbol,
                "expiry_key": self.expiry_key,
            }
        )
        self.is_running = False

    async def reload_model(self) -> bool:
        """
        Hot-swap the model artifact without restarting the engine.

        Called by pipeline.py after a retraining run completes. Acquires
        the asyncio reload lock before mutating SignalGenerator state to
        prevent a race with _process_tick() which also holds the lock
        during generate(). Safe to call from any asyncio coroutine.

        DO NOT call this from a separate OS thread — use asyncio.run_coroutine_threadsafe()
        if you must call it from a thread context.

        Returns:
            bool: True if the new model was loaded successfully.
        """
        logger.info(
            {
                "event": "RELOAD_MODEL_START",
                "symbol": self.symbol,
                "expiry_key": self.expiry_key,
            }
        )
        async with self._reload_lock:
            success = self._signal_gen.reload()

        if success:
            logger.info(
                {
                    "event": "RELOAD_MODEL_SUCCESS",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                }
            )
        else:
            logger.warning(
                {
                    "event": "RELOAD_MODEL_FAILED",
                    "symbol": self.symbol,
                    "expiry_key": self.expiry_key,
                }
            )

        return success

    def __repr__(self) -> str:
        return (
            f"LiveEngine("
            f"symbol={self.symbol!r}, "
            f"expiry={self.expiry_key!r}, "
            f"running={self.is_running}, "
            f"stream={type(self._stream).__name__!r}, "
            f"errors={self._consecutive_errors})"
        )
