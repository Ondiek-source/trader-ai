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

  USE_QUOTEX_STREAMING=True  -> QuotexReader  (OTC symbols via Quotex)
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
  - Journal/WebhookSender/Reporter failures at init  -> WARNING only  -> degrade

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
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.config import get_settings
from data.storage import Storage, StorageError
from ml_engine.features import (
    BINARY_EXPIRY_RULES,
    FeatureEngineer,
    FeatureEngineerError,
    get_feature_engineer,
)
from ml_engine.model_manager import ModelManager
from trading.signals import SignalGenerator, SignalGeneratorError, TradeSignal
from trading.threshold_manager import ThresholdManager

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


class LiveEngineError(Exception):
    """
    Raised when LiveEngine cannot initialise or encounters a fatal
    mid-run failure that is not covered by StorageError.

    Attributes:
        stage: The initialisation or runtime stage that failed.
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return f"LiveEngineError(stage={self.stage!r}, message={str(self)!r})"


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
            raise ValueError(
                "[!] LiveEngine requires a non-empty symbol. " "Got an empty string."
            )
        if expiry_key not in BINARY_EXPIRY_RULES:
            raise ValueError(
                f"[!] Invalid expiry_key: '{expiry_key}'. Must be one of {list(BINARY_EXPIRY_RULES.keys())}."
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

        # Reload lock: prevents reload_model() from mutating SignalGenerator
        # state while _process_tick() is mid-inference. Both the asyncio
        # event loop and any pipeline.py coroutine use this.
        self._reload_lock: asyncio.Lock = asyncio.Lock()

        # ── 1. Storage (fail-hard) ────────────────────────────────────────
        # StorageError propagates immediately — no engine without storage.
        self._storage: Storage = Storage()

        # ── 2. Feature Engineering (fail-hard) ────────────────────────────
        # get_feature_engineer() returns the module singleton.
        # Stateless and thread-safe — shared across all LiveEngine instances.
        self._engineer: FeatureEngineer = get_feature_engineer()

        # ── 3. Stream (fail-hard) ─────────────────────────────────────────
        self._stream: Any = self._init_stream()

        # ── 4. Model Manager + cold-start (warning-only) ──────────────────
        # Instantiated once here and reused. Avoids double-instantiation
        # that was present in the previous version (_cold_start created
        # its own ModelManager separate from SignalGenerator's).
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
        self._signal_gen.set_threshold_manager(self._threshold_mgr)

        # ── 6. Journal (warning-only) ─────────────────────────────────────
        self._journal_client: Any = self._init_journal()

        # ── 7. WebhookSender (warning-only) ───────────────────────────────
        self._webhook: Any = self._init_webhook()

        # ── 8. Reporter (warning-only) ────────────────────────────────────
        self._reporter: Any = self._init_reporter()

        logger.info(
            "[^] LiveEngine ready: symbol=%s expiry=%s stream=%s "
            "model=%s webhook=%s journal=%s",
            self.symbol,
            self.expiry_key,
            type(self._stream).__name__,
            "LOADED" if self._signal_gen._model is not None else "SKIP-MODE",
            "OK" if self._webhook is not None else "DISABLED",
            "OK" if self._journal_client is not None else "DISABLED",
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

        # Connect the stream if it's QuotexReader
        if hasattr(engine._stream, "connect"):
            connected = await engine._stream.connect()
            if not connected:
                raise LiveEngineError(f"Failed to connect to Quotex for {symbol}")

        # Now await the async cold start to pull the model artifact before the inference loop starts.
        await engine._cold_start()

        # Reload with lock for safety, the model into SignalGenerator now that artifact is on disk
        async with engine._reload_lock:
            if engine._signal_gen._model is None:
                engine._signal_gen.reload()

        return engine

    def _init_stream(self) -> Any:
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
                from engine.quotex_reader import QuotexReader

                otc_symbol: str = self._settings.quotex_symbols[self.symbol]
                logger.info(
                    "[^] LiveEngine stream: QuotexReader %s -> %s (OTC)",
                    self.symbol,
                    otc_symbol,
                )
                return QuotexReader(
                    email=self._settings.quotex_email,
                    password=self._settings.quotex_password,
                    practice_mode=self._settings.practice_mode,
                    symbol=otc_symbol,
                )
            else:
                from engine.twelveticks_stream import TwelveDataStream

                logger.info(
                    "[^] LiveEngine stream: TwelveDataStream %s",
                    self.symbol,
                )
                return TwelveDataStream(
                    api_key=self._settings.twelvedata_api_key,
                    pairs=self._settings.pairs,
                    storage=self._storage,
                    flush_size=self._settings.tick_flush_size,
                )

        except Exception as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"LIVE ENGINE INIT FAILURE: STREAM CONSTRUCTION\n"
                f"Symbol            : {self.symbol}\n"
                f"use_quotex_streaming: {self._settings.use_quotex_streaming}\n"
                f"Error             : {exc}\n\n"
                f"CONTEXT: The live tick stream could not be constructed.\n"
                f"Without a tick source the engine has no data to process.\n"
                f"This is a fatal boot failure.\n"
                f"\nFIX: If using Quotex, verify QUOTEX_EMAIL and QUOTEX_PASSWORD\n"
                f"in .env. If using TwelveData, verify TWELVEDATA_API_KEY.\n"
                f"Check network connectivity to the broker/data provider.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
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

        In LOCAL mode or when no artifact exists this is a warning-only
        no-op — SignalGenerator will run in SKIP-only mode.
        """
        try:
            result = await self._model_manager.pull_from_blob(
                symbol=self.symbol,
                expiry_key=self.expiry_key,
            )
        except Exception as exc:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: COLD-START BLOB PULL FAILED\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: An error occurred while pulling the model artifact\n"
                f"from Azure Blob Storage. The engine will start in SKIP-only\n"
                f"mode. All signals will be SKIP until a model is loaded.\n"
                f"\nFIX: Check AZURE_STORAGE_CONN and network connectivity.\n"
                f"Run a training pass via pipeline.py if no artifact exists.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
            return

        if result:
            logger.info("[^] LiveEngine cold-start: restored artifact -> %s", result)
        else:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: NO MODEL FOUND AT COLD-START\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n\n"
                f"CONTEXT: No model artifact was found in Azure Blob Storage\n"
                f"for this symbol/expiry combination. The engine will start\n"
                f"in SKIP-only mode — all signals will be SKIP until a model\n"
                f"is trained, uploaded, and loaded via reload_model().\n"
                f"\nFIX: Run a training pass via pipeline.py to produce and\n"
                f"upload a model artifact, then call engine.reload_model().\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)

    def _init_journal(self) -> Any:
        """
        Attempt to construct a Journal instance.

        Failure is warning-only — the engine degrades gracefully without
        journaling rather than refusing to start.

        Returns:
            Journal instance or None if construction fails.
        """
        try:
            from data.journal import Journal

            j = Journal()
            logger.info("[^] LiveEngine: Journal initialised.")
            return j
        except Exception as exc:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: JOURNAL UNAVAILABLE\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: Journal could not be initialised. Signals will\n"
                f"be executed but not recorded in the trade journal.\n"
                f"This is a data loss risk for post-analysis.\n"
                f"\nFIX: Verify journal.py is implemented and its dependencies\n"
                f"are installed. Check for import errors in journal.py.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
            return None

    def _init_webhook(self) -> Any:
        """
        Attempt to construct a WebhookSender instance.

        Failure is warning-only — the engine runs in journal-only mode
        without executing trades.

        Returns:
            WebhookSender instance or None if construction fails.
        """
        try:
            from trading.webhook import WebhookSender

            w = WebhookSender(
                url=self._settings.webhook_url,
                secret=self._settings.webhook_secret,
            )
            logger.info("[^] LiveEngine: WebhookSender initialised.")
            return w
        except Exception as exc:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: WEBHOOK UNAVAILABLE\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: WebhookSender could not be initialised. Signals\n"
                f"will be generated and journaled but not executed at Quotex.\n"
                f"The engine is running in observation-only mode.\n"
                f"\nFIX: Verify webhook.py is implemented and WEBHOOK_URL,\n"
                f"WEBHOOK_KEY are set correctly in .env.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
            return None

    def _init_reporter(self) -> Any:
        """
        Attempt to construct a Reporter instance.

        Failure is warning-only — trade notifications are disabled but
        the engine continues operating.

        Returns:
            Reporter instance or None if construction fails.
        """
        try:
            from trading.reporter import Reporter

            r = Reporter(
                telegram_token=self._settings.telegram_token,
                telegram_chat_id=self._settings.telegram_chat_id,
                discord_webhook_url=self._settings.discord_webhook_url,
                orchestrator=None,  # wired by pipeline.py after full init
            )
            logger.info("[^] LiveEngine: Reporter initialised.")
            return r
        except Exception as exc:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: REPORTER UNAVAILABLE\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: Reporter could not be initialised. Trade execution\n"
                f"notifications (Telegram/Discord) are disabled.\n"
                f"\nFIX: Verify reporter.py is implemented and TELEGRAM_TOKEN,\n"
                f"DISCORD_WEBHOOK_URL are set correctly in .env.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
            return None

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
            "[^] LiveEngine.run(): starting — symbol=%s expiry=%s stream=%s",
            self.symbol,
            self.expiry_key,
            type(self._stream).__name__,
        )
        self.is_running = True
        self._consecutive_errors = 0

        # ── Reset dashboard session clock to actual engine start time ─────
        # started_at is seeded at StatusStore.__init__ (module import time).
        # Resetting it here means elapsed_minutes counts from when the engine
        # actually begins trading, not from container boot / backfill time.
        try:
            from datetime import datetime, timezone as _tz
            from core.dashboard import status_store

            status_store.update(
                {
                    "started_at": datetime.now(_tz.utc).isoformat(),
                    "session": {
                        "is_active": True,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "net_profit": 0.0,
                        "signals_fired": 0,
                        "elapsed_minutes": 0.0,
                    },
                }
            )
        except Exception:
            pass

        # Start result consumer task (only for Quotex)
        result_task = None
        if hasattr(self._stream, "get_result"):
            result_task = asyncio.create_task(self._consume_results())
            logger.info("[^] LiveEngine: result consumer task started")

        try:
            async for tick in self._stream.subscribe():
                if not self.is_running:
                    logger.info(
                        "[^] LiveEngine.run(): stop requested — "
                        "exiting loop for %s %s.",
                        self.symbol,
                        self.expiry_key,
                    )
                    break

                await self._process_tick(tick)

                # Self-termination guard: if too many consecutive errors
                # have accumulated, something is structurally wrong.
                if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    error_block = (
                        f"\n{'!' * 60}\n"
                        f"LIVE ENGINE FATAL: CONSECUTIVE ERROR LIMIT EXCEEDED\n"
                        f"Symbol          : {self.symbol}\n"
                        f"Expiry key      : {self.expiry_key}\n"
                        f"Errors          : {self._consecutive_errors}"
                        f"/{_MAX_CONSECUTIVE_ERRORS}\n\n"
                        f"CONTEXT: {_MAX_CONSECUTIVE_ERRORS} consecutive per-bar\n"
                        f"errors were caught without a single successful bar.\n"
                        f"This indicates a structural failure (corrupted model,\n"
                        f"broken feature pipeline, or bad bar data) rather than\n"
                        f"transient per-bar noise. Self-terminating to surface\n"
                        f"the problem rather than silently looping forever.\n"
                        f"\nFIX: Check logs above for the repeating warning block.\n"
                        f"Resolve the root cause, then restart the engine.\n"
                        f"{'!' * 60}"
                    )
                    logger.critical(error_block)
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
            error_block = (
                f"\n{'!' * 60}\n"
                f"LIVE ENGINE FATAL: UNHANDLED STREAM EXCEPTION\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: An unexpected exception escaped the inference loop.\n"
                f"This is likely a stream disconnection, network interruption,\n"
                f"or an uncaught edge case. The engine is shutting down.\n"
                f"\nFIX: Review the full stack trace. If the stream disconnected,\n"
                f"pipeline.py should restart the engine after a backoff delay.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            self.is_running = False
            raise

        finally:
            # Clean up result consumer
            if result_task:
                result_task.cancel()
                try:
                    await result_task
                except asyncio.CancelledError:
                    raise
            logger.info(
                "[^] LiveEngine.run(): stopped — symbol=%s expiry=%s",
                self.symbol,
                self.expiry_key,
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
        # ── Push tick counter to dashboard immediately, before any early returns ──
        # This ensures elapsed_minutes and ticks always update even when bars
        # are missing, warmup is incomplete, or feature engineering fails.
        self._ticks_processed += 1
        self._push_stream_status_tick_only()

        # ── M1 bar-close gate ─────────────────────────────────────────────
        # QuotexReader polls every ~1 s. Without this guard the full feature
        # engineering + model inference pipeline fires ~60× per minute.
        # We run inference exactly once per UTC minute (i.e. once per M1 bar).
        _now_minute = datetime.now(timezone.utc).minute
        if _now_minute == self._last_processed_minute:
            return
        self._last_processed_minute = _now_minute

        # ── 1. Load recent bars (capped at _BAR_WINDOW) ───────────────────
        # StorageError propagates to run() as a fatal failure.
        bars_df: pd.DataFrame | None = self._storage.get_bars(
            symbol=self.symbol,
            timeframe="M1",
            max_rows=_BAR_WINDOW,
        )

        if bars_df is None or bars_df.empty:
            logger.debug(
                "[^] _process_tick: no M1 bars for %s — awaiting first bar.",
                self.symbol,
            )
            return

        # ── 2. Minimum bars warmup guard ──────────────────────────────────
        if len(bars_df) < _MIN_BARS_REQUIRED:
            logger.debug(
                "[^] _process_tick: warmup %d/%d bars for %s.",
                len(bars_df),
                _MIN_BARS_REQUIRED,
                self.symbol,
            )
            return

        # ── 3. Feature engineering ────────────────────────────────────────
        # transform() produces the full feature DataFrame needed by gates.
        # get_latest() extracts the final-row FeatureVector for inference.
        # Both operate on the same capped bars_df — no redundant I/O.
        try:
            fe_df: pd.DataFrame = self._engineer.transform(bars_df)
            fv = self._engineer.get_latest(bars_df)
        except FeatureEngineerError as exc:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: FEATURE ENGINEERING FAILED\n"
                f"Symbol     : {self.symbol}\n"
                f"Bars loaded: {len(bars_df)}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: The feature pipeline could not produce a valid\n"
                f"FeatureVector for this bar. This tick is skipped.\n"
                f"Persistent failures indicate corrupted bar data in storage.\n"
                f"\nFIX: Check storage for NaN-heavy bars or gaps exceeding\n"
                f"_MAX_FFILL_BARS in features.py. Re-run the backfill if\n"
                f"the data is corrupted.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
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
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: SIGNAL GENERATION FAILED\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: The signal generator could not produce a valid\n"
                f"TradeSignal for this bar. This tick is skipped.\n"
                f"Persistent failures indicate a corrupted model artifact\n"
                f"or a feature/model schema mismatch.\n"
                f"\nFIX: Call reload_model() to refresh the artifact. If\n"
                f"failures persist, retrain and re-upload the model.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
            self._consecutive_errors += 1
            return

        # Reset consecutive error counter on a successful inference.
        self._consecutive_errors = 0
        if signal.direction != "SKIP":
            self._signals_fired += 1

        # ── Push stream + session counters to dashboard ───────────────────
        self._push_stream_status(signal)  # type: ignore[attr-defined]

        # ── 5. Journal every signal (including SKIP) ──────────────────────
        self._write_journal(signal)

        # ── 6. Kill switch gate ───────────────────────────────────────────
        if self._threshold_mgr.is_halted():
            self._on_kill_switch_activated()
            return

        # ── 7. Execute only if signal is actionable ───────────────────────
        # is_executable() requires direction != SKIP AND gate_passed = True.
        # A CALL/PUT with gate_passed=False is journaled but not executed.
        if signal.is_executable():
            await self._execute(signal)
        else:
            logger.debug(
                "[^] Signal not executable: symbol=%s direction=%s "
                "gate=%s — journaled only.",
                signal.symbol,
                signal.direction,
                signal.gate_passed,
            )

        # ── 8. Push updated threshold state to dashboard ──────────────────
        self._push_threshold_to_dashboard()

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute(self, signal: TradeSignal) -> None:
        """
        Fire the webhook and notify the reporter for an executable signal.

        Only called when signal.is_executable() is True. Passes the full
        TradeSignal to both components — they extract the fields they need.
        Failures in webhook or reporter are warning-only and do not crash
        the engine. The signal is already journaled before this is called.

        Args:
            signal: Executable TradeSignal (CALL or PUT, gate_passed=True).
        """
        logger.info(
            "[^] EXECUTING: symbol=%s direction=%s confidence=%.4f "
            "expiry=%s model=%s",
            signal.symbol,
            signal.direction,
            signal.confidence,
            signal.expiry_key,
            signal.model_name,
        )

        if self._webhook is not None:
            try:
                await self._webhook.fire(signal)
            except Exception as exc:
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"LIVE ENGINE WARNING: WEBHOOK FIRE FAILED\n"
                    f"Symbol    : {signal.symbol}\n"
                    f"Direction : {signal.direction}\n"
                    f"Confidence: {signal.confidence:.4f}\n"
                    f"Error     : {exc}\n\n"
                    f"CONTEXT: The trade signal could not be forwarded to\n"
                    f"the Quotex webhook. The signal has been journaled.\n"
                    f"The trade was NOT placed at the broker.\n"
                    f"\nFIX: Check WEBHOOK_URL and WEBHOOK_KEY in .env.\n"
                    f"Verify network connectivity to the webhook endpoint.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)
        else:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: WEBHOOK NOT AVAILABLE\n"
                f"Symbol    : {signal.symbol}\n"
                f"Direction : {signal.direction}\n\n"
                f"CONTEXT: WebhookSender was not initialised at boot.\n"
                f"The signal has been journaled but the trade was NOT placed.\n"
                f"\nFIX: Verify webhook.py is implemented and restart the engine.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)

        if self._reporter is not None:
            try:
                await self._reporter.notify(signal)
            except Exception as exc:
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"LIVE ENGINE WARNING: REPORTER NOTIFICATION FAILED\n"
                    f"Symbol : {signal.symbol}\n"
                    f"Error  : {exc}\n\n"
                    f"CONTEXT: The reporter could not send a trade notification.\n"
                    f"The trade was placed. Only the notification failed.\n"
                    f"\nFIX: Check TELEGRAM_TOKEN and DISCORD_WEBHOOK_URL in .env.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)

    def _write_journal(self, signal: TradeSignal) -> None:
        """
        Write a signal to the trade journal.

        Called for every signal including SKIP. Journal failures are
        warning-only — a missed journal entry is preferable to a crashed
        engine. The journal_client is instantiated once in __init__ and
        reused, matching the singleton pattern of all other components.

        Args:
            signal: TradeSignal to record. to_dict() produces the
                    JSON-serialisable dict the journal consumes.
        """
        if self._journal_client is None:
            return

        try:
            if isinstance(signal, dict):
                self._journal_client.record_signal(signal)
            else:
                self._journal_client.record_signal(signal.to_dict())
        except Exception as exc:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: JOURNAL WRITE FAILED\n"
                f"Symbol    : {signal.symbol}\n"
                f"Direction : {signal.direction}\n"
                f"Timestamp : {signal.timestamp.isoformat()}\n"
                f"Error     : {exc}\n\n"
                f"CONTEXT: The signal could not be written to the trade journal.\n"
                f"The inference loop continues. This is a data loss risk\n"
                f"for post-analysis and model evaluation.\n"
                f"\nFIX: Check journal.py and its storage dependencies.\n"
                f"Inspect disk space and file permissions.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
            traceback.print_stack()

    # ── Threshold / Martingale helpers ────────────────────────────────────────

    def record_result(self, win: bool) -> None:
        """
        Report the outcome of the most recent executed trade.

        Called by an external result checker (e.g. pipeline.py polling the
        Quotex API for settled contracts) once the binary option has expired
        and the outcome is known.

        Args:
            win: True if the trade was profitable, False if it was a loss.
        """
        if win:
            self._threshold_mgr.on_win()
            logger.info(
                "[^] ThresholdManager: WIN recorded — streak reset, "
                "threshold=%.3f symbol=%s",
                self._threshold_mgr.get_threshold(),
                self.symbol,
            )
        else:
            self._threshold_mgr.on_loss()
            logger.info(
                "[^] ThresholdManager: LOSS recorded — streak=%d "
                "threshold=%.3f symbol=%s",
                self._threshold_mgr.streak,
                self._threshold_mgr.get_threshold(),
                self.symbol,
            )
            if self._threshold_mgr.is_halted():
                self._on_kill_switch_activated()

        self._push_threshold_to_dashboard()

    def _on_kill_switch_activated(self) -> None:
        """
        Handle kill switch activation — log prominently and stop the engine.

        Called when ThresholdManager.is_halted() returns True, either
        because record_result() just crossed the max_streak threshold or
        because _process_tick() detected the halted state after a tick.
        """
        warning_block = (
            f"\n{'!'*60}\n"
            f"MARTINGALE KILL SWITCH ACTIVATED\n"
            f"Symbol              : {self.symbol}\n"
            f"Consecutive losses  : {self._threshold_mgr.streak}\n"
            f"Max streak          : {self._threshold_mgr.max_streak}\n"
            f"Effective threshold : {self._threshold_mgr.get_threshold():.3f}\n\n"
            f"CONTEXT: The engine has reached the maximum consecutive loss\n"
            f"streak. All trade execution is suspended for this session to\n"
            f"prevent further losses. The inference loop continues to run\n"
            f"and journal signals, but no trades will be placed until the\n"
            f"threshold manager is reset (e.g. on next session start).\n"
            f"\nFIX: Call engine.record_result(win=True) after a manual\n"
            f"recovery trade, or restart the engine to reset the session.\n"
            f"{'!'*60}"
        )
        logger.critical(warning_block)

        # Push kill-switch state to dashboard — session is no longer active
        try:
            from core.dashboard import status_store

            snapshot = status_store.get()
            status_store.update(
                {
                    "kill_switch_active": True,
                    "session": {**snapshot.get("session", {}), "is_active": False},
                }
            )
        except Exception:
            pass

    def _push_stream_status(self, signal: TradeSignal) -> None:
        """
        Push stream connection status, tick count, signals fired, and the
        most recent signal event to the dashboard StatusStore.

        Called on every successful inference pass. Failure is silently
        swallowed — dashboard pushes must never block the inference loop.
        """
        try:
            from core.dashboard import status_store

            # Get Quotex balance if available
            quotex_balance = 0.0

            # Detect stream type and read its live properties.
            # QuotexReader exposes ._connected (bool).
            # TwelveDataStream exposes ._thread (threading.Thread).
            # Fallback: a tick was just received, so treat as connected.
            if hasattr(self._stream, "_connected"):
                connected: bool = bool(self._stream._connected)
                quotex_balance = self._stream._balance
            elif hasattr(self._stream, "_thread") and self._stream._thread is not None:
                connected = self._stream._thread.is_alive()
            else:
                connected = True  # tick received — stream must be up

            ticks_received: int = self._ticks_processed

            # For TwelveDataStream expose the stream-level counter if available.
            if hasattr(self._stream, "ticks_received"):
                ticks_received = int(self._stream.ticks_received)

            direction = signal.direction
            if direction != "SKIP":
                gate_icon = "✓" if signal.gate_passed else "✗"
                event_text = f"{self.symbol} {direction} @ {signal.confidence:.1%} [{signal.expiry_key}] gate={gate_icon}"
            else:
                event_text = f"{self.symbol} SKIP ({signal.confidence:.1%})"

            snapshot = status_store.get()

            # Compute elapsed minutes from the started_at stamp seeded at boot.
            # This keeps the dashboard accurate for both Quotex and TwelveData users.
            elapsed_minutes: float = 0.0
            try:
                from datetime import datetime, timezone as _tz

                started_at_raw = snapshot.get("started_at")
                if started_at_raw:
                    started_at = datetime.fromisoformat(started_at_raw)
                    elapsed_minutes = (
                        datetime.now(_tz.utc) - started_at
                    ).total_seconds() / 60.0
            except Exception:
                pass

            status_store.update(
                {
                    "stream": {
                        "connected": connected,
                        "ticks_received": ticks_received,
                    },
                    "quotex": {
                        "connected": connected,
                        "balance": quotex_balance,
                    },
                    "last_event": event_text,
                    "session": {
                        **snapshot.get("session", {}),
                        "signals_fired": self._signals_fired,
                        "elapsed_minutes": round(elapsed_minutes, 1),
                    },
                }
            )
        except Exception:
            pass  # Dashboard is optional — never block on it

    def _push_stream_status_tick_only(self) -> None:
        """
        Minimal dashboard push on every tick — updates elapsed time, tick
        count, and connection state regardless of whether inference succeeded.
        Called before any early returns in _process_tick().
        """
        try:
            from datetime import datetime, timezone as _tz
            from core.dashboard import status_store

            connected = bool(getattr(self._stream, "_connected", True))
            balance = float(getattr(self._stream, "_balance", 0.0))

            snapshot = status_store.get()
            elapsed_minutes = 0.0
            started_at_raw = snapshot.get("started_at")
            if started_at_raw:
                started_at = datetime.fromisoformat(started_at_raw)
                elapsed_minutes = (
                    datetime.now(_tz.utc) - started_at
                ).total_seconds() / 60.0

            status_store.update(
                {
                    "stream": {
                        "connected": connected,
                        "ticks_received": self._ticks_processed,
                    },
                    "quotex": {"connected": connected, "balance": balance},
                    "session": {
                        **snapshot.get("session", {}),
                        "elapsed_minutes": round(elapsed_minutes, 1),
                        "is_active": self.is_running,
                    },
                }
            )
        except Exception:
            pass

    def _push_threshold_to_dashboard(self) -> None:
        """
        Push the current ThresholdManager state to the StatusStore.

        Failure is silently swallowed — dashboard unavailability must
        never block the inference loop.
        """
        try:
            from core.dashboard import status_store

            state = self._threshold_mgr.get_state()
            status_store.update(
                {
                    "martingale_streak": state["streak"],
                    "confidence_threshold": state["effective_threshold"],
                    "kill_switch_active": state["halted"],
                }
            )
        except Exception:
            pass  # Dashboard is optional — never block on it

    def _push_trade_result_to_dashboard(self, result: dict[str, Any]) -> None:
        """Push trade outcome to dashboard session stats."""
        try:
            from core.dashboard import status_store

            snapshot = status_store.get()
            session = snapshot.get("session", {})

            outcome = result["result"]

            if outcome == "win":
                session["wins"] = session.get("wins", 0) + 1
                session["net_profit"] = session.get("net_profit", 0) + result["payout"]
            elif outcome == "loss":
                session["losses"] = session.get("losses", 0) + 1
                session["net_profit"] = session.get("net_profit", 0) - result["stake"]
            elif outcome == "draw":
                session["draws"] = session.get("draws", 0) + 1
                # Draw: no profit change, no streak change
            else:
                # No trade placed (result is None) - do nothing
                return

            status_store.update({"session": session})

        except Exception:
            pass

    async def _consume_results(self) -> None:
        """
        Background task to consume trade results from QuotexReader.
        """
        while self.is_running:
            try:
                if not hasattr(self._stream, "get_result"):
                    await asyncio.sleep(1)
                    continue

                result = await self._stream.get_result(timeout=0.5)
                if result:
                    outcome = result["result"]  # 'win', 'loss', or 'draw'

                    # Update threshold manager ONLY for win/loss (not draw)
                    if outcome == "win":
                        self._threshold_mgr.on_win()
                    elif outcome == "loss":
                        self._threshold_mgr.on_loss()
                    # draw: do nothing - streak unchanged, threshold unchanged

                    # Update dashboard
                    self._push_trade_result_to_dashboard(result)

                    logger.info(
                        {
                            "event": "trade_result_processed",
                            "signal_id": result.get("signal_id"),
                            "result": outcome,
                            "payout": result.get("payout"),
                            "stake": result.get("stake"),
                            "streak": self._threshold_mgr.streak,
                            "threshold": self._threshold_mgr.get_threshold(),
                        }
                    )

            except Exception as exc:
                logger.debug(f"Result consumer error: {exc}")

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
            "[^] LiveEngine.stop(): shutdown requested — symbol=%s expiry=%s",
            self.symbol,
            self.expiry_key,
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
            "[^] LiveEngine.reload_model(): acquiring lock for %s %s",
            self.symbol,
            self.expiry_key,
        )
        async with self._reload_lock:
            success = self._signal_gen.reload()

        if success:
            logger.info(
                "[^] LiveEngine.reload_model(): model refreshed for %s %s",
                self.symbol,
                self.expiry_key,
            )
        else:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"LIVE ENGINE WARNING: MODEL RELOAD FAILED\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n\n"
                f"CONTEXT: reload_model() could not find or load a new\n"
                f"artifact. The engine continues with the previously loaded\n"
                f"model (or in SKIP-only mode if no model was ever loaded).\n"
                f"\nFIX: Verify a training pass has completed and the artifact\n"
                f"was uploaded to Azure Blob before calling reload_model().\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)

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
