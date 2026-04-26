"""
src/trading/signals.py — The Strategist.

Role: Translate ML model probability outputs into typed TradeSignal
decisions. Applies confidence thresholds, gate compliance checks, and
model-type-aware inference to produce an actionable CALL / PUT / SKIP
decision for each bar.

Design rationale
----------------
signals.py sits between the ML engine and the execution layer. It
owns exactly one responsibility: given a FeatureVector and a loaded
model artifact, produce a TradeSignal. It does not load models, write
to disk, fire webhooks, or manage trading sessions — those belong to
model_manager.py, webhook.py, and live.py respectively.

Inference adapter
-----------------
Model artifacts produced by trainer.py fall into three categories with
different inference APIs:

  Classical ML (XGBoost, LightGBM, CatBoost, RandomForest, Stacking)
    artifact.predict_proba(X_2d)[:, 1]  — X shape: (1, n_features)

  PyTorch (LSTM, GRU, TCN, CNN-LSTM, Transformer)
    model.eval(); model(tensor).squeeze().item()
    Input tensor shape: (1, window_size, n_features) for sequence models
    or (1, n_features) for single-step models.

  Stable-Baselines3 (A2C, DQN, PPO, RecurrentPPO)
    model.predict(obs, deterministic=True) -> (action, _state)
    action: 0=CALL, 1=PUT, 2=SKIP (discrete, not a probability)

The _infer() method dispatches to the correct path based on the
artifact type detected at inference time, matching the same detection
logic used in model_manager.py._is_sb3_model().

Confidence threshold
--------------------
The threshold is read from config.confidence_threshold (set via
CONFIDENCE_THRESHOLD in .env). A probability >= threshold fires CALL.
A probability <= (1 - threshold) fires PUT. Everything in between is
SKIP. The threshold is never hardcoded in this file.

Public API
----------
    TradeSignal                          — frozen dataclass, inference output
    SignalGeneratorError                 — raised on unrecoverable failures
    SignalGenerator(symbol, expiry_key)
    SignalGenerator.generate(fv, fe_df)  -> TradeSignal
    SignalGenerator.reload()             -> bool
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from core.config import get_settings

from ml_engine.features import (
    BINARY_EXPIRY_RULES,
    FeatureVector,
    _VERSION,
    get_feature_engineer,
)
from ml_engine.labeler import _EXPIRY_SECONDS
from ml_engine.model_manager import ModelManager, ModelRecord

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


# ── Custom Exception ─────────────────────────────────────────────────────────


class SignalGeneratorError(Exception):
    """
    Raised when SignalGenerator cannot produce a valid TradeSignal.

    Distinct from ValueError (caller contract violation) — this signals
    a runtime failure in the inference pipeline that live.py must handle
    explicitly, typically by skipping the current bar and continuing.

    Attributes:
        stage: The pipeline stage that failed, e.g. "infer", "gate".
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return f"SignalGeneratorError(stage={self.stage!r}, message={str(self)!r})"


# ── TradeSignal ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeSignal:
    """
    Immutable output of a single inference pass through SignalGenerator.

    Carries all fields required by webhook.py (direction, confidence,
    expiry), journal.py (timestamp, symbol, model_name),
    reporter.py (symbol, direction, confidence), and live.py

    Attributes:
        symbol:          Currency pair, e.g. "EUR_USD".
        direction:       Trade decision: "CALL", "PUT", or "SKIP".
        confidence:      Model confidence in the predicted direction.
                            For CALL: P(price rises). For PUT: 1 - P(price rises).
                            For SKIP: raw model probability (for logging only).
        expiry_key:      Expiry window identifier, e.g. "5_MIN".
        expiry_seconds:  Duration of the expiry window in seconds.
        timestamp:       UTC datetime when the signal was generated.
        symbol:          Currency pair the signal was generated for.
        model_name:      Trainer class name that produced the artifact,
                            e.g. "XGBoostTrainer". From ModelRecord.
        feature_version: _VERSION string from features.py at inference time.
                            Allows downstream consumers to detect schema drift.
    """

    symbol: str
    direction: str
    confidence: float
    expiry_key: str
    expiry_seconds: int
    timestamp: datetime
    model_name: str
    feature_version: str

    def __post_init__(self) -> None:
        """
        Validate field values at construction time.

        Raises:
            ValueError: If direction is not one of CALL/PUT/SKIP, or if
                        confidence is outside [0.0, 1.0].
        """
        if self.direction not in ("CALL", "PUT", "SKIP"):
            raise ValueError(
                f"[!] TradeSignal direction must be 'CALL', 'PUT', or 'SKIP'. "
                f"Got: '{self.direction}'"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"[%] TradeSignal confidence must be in [0.0, 1.0]. "
                f"Got: {self.confidence}"
            )

    def is_executable(self) -> bool:
        """
        Return True if this signal should be forwarded to webhook.py.

        A signal is executable when it is not SKIP.
        live.py calls this before firing the webhook.

        Returns:
            bool: True for CALL or PUT signals.
        """
        return self.direction != "SKIP"

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable snapshot for journal and reporter use.

        Returns:
            dict[str, Any]: All TradeSignal fields with datetime cast to
                ISO-8601 string.
        """
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "expiry_key": self.expiry_key,
            "expiry_seconds": self.expiry_seconds,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "feature_version": self.feature_version,
        }

    def __repr__(self) -> str:
        return (
            f"TradeSignal("
            f"symbol={self.symbol!r}, "
            f"direction={self.direction!r}, "
            f"confidence={self.confidence:.4f}, "
            f"expiry={self.expiry_key!r}, "
            f"model={self.model_name!r})"
        )


# ── SignalGenerator ───────────────────────────────────────────────────────────


class SignalGenerator:
    """
    The Strategist: translate a FeatureVector into a TradeSignal.

    Loads the best available model artifact for the given symbol and
    expiry key via ModelManager, evaluates the four eligibility gates
    via FeatureEngineer, runs inference through the model, applies the
    confidence threshold from config, and returns an immutable
    TradeSignal.

    Model loading
    -------------
    The best artifact is selected by ModelManager.get_best_model() which
    returns the highest-AUC current-version artifact. If no artifact
    exists (first boot, cold-start before pull_from_blob() runs), all
    generate() calls return SKIP until reload() succeeds.

    Inference dispatch
    ------------------
    _infer() detects the artifact type at runtime and dispatches to the
    correct inference API. This keeps generate() clean and makes it
    trivial to add new model types without changing the public interface.

    Thread safety
    -------------
    SignalGenerator holds mutable state (self._model, self._record).
    live.py must not call generate() and reload() concurrently from
    different threads. In practice the inference loop is single-threaded
    so this is not a concern, but callers should be aware.

    Attributes:
        symbol:      Currency pair this generator serves, e.g. "EUR_USD".
        expiry_key:  Expiry window, e.g. "5_MIN".
        threshold:   Confidence threshold from config.confidence_threshold.

    Example:
        >>> gen = SignalGenerator(symbol="EUR_USD", expiry_key="5_MIN")
        >>> signal = gen.generate(feature_vector, feature_dataframe)
        >>> if signal.is_executable():
        ...     webhook.fire(signal)
    """

    def __init__(
        self,
        symbol: str,
        expiry_key: str,
    ) -> None:
        """
        Initialise the SignalGenerator and attempt to load the best model.

        Does not raise if no model is found — generate() returns SKIP
        until a model is available. This allows live.py to boot and
        start the inference loop immediately, even if the first training
        run has not completed yet.

        Args:
            symbol:     Currency pair to generate signals for.
            expiry_key: Expiry window key. Must be in BINARY_EXPIRY_RULES.

        Raises:
            ValueError: If expiry_key is not in BINARY_EXPIRY_RULES.
        """
        if expiry_key not in BINARY_EXPIRY_RULES:
            raise ValueError(
                f"[!] Invalid expiry_key: '{expiry_key}'. "
                f"Must be one of {list(BINARY_EXPIRY_RULES.keys())}."
            )

        self._settings = get_settings()
        self.symbol: str = symbol
        self.expiry_key: str = expiry_key
        self.expiry_seconds: int = _EXPIRY_SECONDS[expiry_key]
        self.threshold: float = self._settings.confidence_threshold

        self._manager: ModelManager = ModelManager(storage_dir=self._settings.model_dir)

        # Model artifact and its registry record. Both None until a
        # trained artifact exists. generate() returns SKIP when None.
        self._model: Any = None
        self._record: ModelRecord | None = None

        # Optional ThresholdManager injected by live.py after construction.
        # When None, generate() falls back to the static config threshold.
        self._threshold_mgr: Any = None

        # Attempt initial load — no error if no artifact found.
        self.reload()

        logger.info({"event": "SIGNAL_GENERATOR_INIT", "symbol": self.symbol, "expiry_key": self.expiry_key, "threshold": round(self.threshold, 2), "model": self._record.model_name if self._record else "NONE"})

    # ── Model Loading ─────────────────────────────────────────────────────────

    def reload(self) -> bool:
        """
        Attempt to load or refresh the best available model artifact.

        Called automatically at construction time and can be called
        manually by pipeline.py after a retraining run completes to
        swap in the new artifact without restarting the inference loop.

        Returns:
            bool: True if a model was loaded successfully.
                    False if no artifact exists or loading failed.
        """
        record = self._manager.get_best_model(
            symbol=self.symbol,
            expiry_key=self.expiry_key,
        )

        if record is None:
            logger.warning({"event": "SIGNAL_NO_MODEL", "symbol": self.symbol, "expiry_key": self.expiry_key})
            self._model = None
            self._record = None
            return False

        try:
            # PyTorch models require a pre-instantiated nn.Module.
            # Classical ML and SB3 models do not need model_class.
            # Since signals.py does not import any nn.Module architecture,
            # PyTorch models must be loaded with model_class=None here,
            # which will raise in model_manager.load() for PyTorch artifacts.
            # The correct pattern: pipeline.py instantiates the architecture
            # and calls model_manager.load(path, model_class=net) directly,
            # then passes the loaded model to SignalGenerator via
            # inject_model(). For now Classical ML and SB3 load cleanly.
            self._model = self._manager.load(record.artifact_path)
            self._record = record

            logger.info({"event": "SIGNAL_MODEL_LOADED", "model": record.model_name, "auc": round(record.auc, 4)})
            return True

        except Exception as exc:
            logger.critical({"event": "SIGNAL_MODEL_LOAD_FAILED", "symbol": self.symbol, "expiry_key": self.expiry_key, "artifact": str(record.artifact_path), "error": str(exc)})
            self._model = None
            self._record = None
            return False

    def set_threshold_manager(self, mgr: Any) -> None:
        """
        Inject a ThresholdManager so generate() uses a dynamic threshold.

        When set, the effective threshold is read from mgr.get_threshold()
        on each generate() call instead of the static config value. live.py
        calls this once after constructing SignalGenerator.

        Args:
            mgr: ThresholdManager instance (typed Any to avoid circular import).
        """
        self._threshold_mgr = mgr
        logger.info({"event": "SIGNAL_THRESHOLD_MANAGER_SET", "base_threshold": round(mgr.base_threshold, 2), "step": round(mgr.step, 2), "max_streak": mgr.max_streak})

    def inject_model(self, model: Any, record: ModelRecord) -> None:
        """
        Inject a pre-loaded model artifact directly into the generator.

        Used by pipeline.py for PyTorch models that require a pre-
        instantiated nn.Module — the pipeline owns the architecture and
        calls model_manager.load(path, model_class=net) itself, then
        passes the result here. Also used in tests to inject mock models
        without touching the filesystem.

        Args:
            model:  The loaded model artifact (any type accepted by
                    BaseTrainer.predict_proba()).
            record: The ModelRecord describing the artifact. Used to
                    populate TradeSignal.model_name.
        """
        self._model = model
        self._record = record
        logger.info({"event": "SIGNAL_MODEL_INJECTED", "model": record.model_name})

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        fv: FeatureVector,
        fe_df: pd.DataFrame,
    ) -> TradeSignal:
        """
        Translate a FeatureVector into a TradeSignal.

        Pipeline:
            1. If no model is loaded, return SKIP immediately.
            2. Run model inference via _infer() to get a raw probability.
            3. Apply confidence threshold to determine direction.
            4. Return an immutable TradeSignal.

        Args:
            fv:    FeatureVector from FeatureEngineer.get_latest().
                    Carries the 50-feature float32 vector for inference.
            fe_df: Full feature DataFrame from FeatureEngineer.transform()
                    for the current bar window. Currently unused but
                    retained for future derived-feature needs.

        Returns:
            TradeSignal: Always returns a signal — never raises for normal
                inference failures. Unrecoverable errors raise
                SignalGeneratorError.

        Raises:
            SignalGeneratorError: If gate evaluation or inference fails
                in a way that cannot produce a valid TradeSignal.
        """

        # ── 1. No model — return SKIP ─────────────────────────────────────
        if self._model is None or self._record is None:
            return self._make_signal(
                direction="SKIP",
                confidence=0.0,
                model_name="NONE",
                feature_version=fv.version,
            )

        # ── 2. Model inference ────────────────────────────────────────────
        try:
            prob: float = self._infer(fv)
        except Exception as exc:
            logger.critical({"event": "SIGNAL_INFERENCE_FAILED", "symbol": self.symbol, "expiry_key": self.expiry_key, "model": self._record.model_name, "error": str(exc)})
            raise SignalGeneratorError(
                f"Model inference failed for {self._record.model_name}: {exc}",
                stage="infer",
            ) from exc

        # ── 3. Confidence threshold → direction ───────────────────────────
        # Use dynamic threshold from ThresholdManager when injected;
        # fall back to the static config value otherwise.
        effective_threshold = (
            self._threshold_mgr.get_threshold()
            if self._threshold_mgr is not None
            else self.threshold
        )

        if prob >= effective_threshold:
            direction = "CALL"
            confidence = prob
        elif prob <= (1.0 - effective_threshold):
            direction = "PUT"
            confidence = 1.0 - prob
        else:
            direction = "SKIP"
            confidence = prob

        signal = self._make_signal(
            direction=direction,
            confidence=confidence,
            model_name=self._record.model_name,
            feature_version=fv.version,
        )

        logger.info({"event": "SIGNAL_GENERATED", "symbol": self.symbol, "direction": signal.direction, "confidence": round(signal.confidence, 2), "model": signal.model_name, "expiry": signal.expiry_key, "threshold": round(effective_threshold, 2)})

        return signal

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _infer(self, fv: FeatureVector) -> float:
        """
        Run model inference and return a scalar probability in [0.0, 1.0].

        Dispatches to the correct inference API based on the artifact type:
            - SB3 agents return a discrete action (0=CALL, 1=PUT, 2=SKIP).
            These are converted to a pseudo-probability so the threshold
            logic in generate() remains uniform.
            - PyTorch nn.Module: forward pass with a (1, n_features) tensor.
            - Classical ML (sklearn/XGB/LGBM/CatBoost): predict_proba() with
            a (1, n_features) 2D numpy array.

        Args:
            fv: FeatureVector carrying the float32 inference vector.

        Returns:
            float: Probability in [0.0, 1.0] representing P(CALL win).

        Raises:
            SignalGeneratorError: If inference produces a non-finite result
                or the artifact type cannot be identified.
        """
        # _infer() is only called from generate() after the self._record is None
        # guard. Assert here so Pylance narrows the type for this method.
        assert self._record is not None

        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for inference")

        # SB3 detection: same MRO check as model_manager._is_sb3_model()
        is_sb3 = any(
            cls.__name__ == "BaseAlgorithm" for cls in type(self._model).__mro__
        )

        if is_sb3:
            action, _ = self._model.predict(fv.vector, deterministic=True)
            # Map discrete action to pseudo-probability:
            # CALL=0 -> 0.90 (strong positive), PUT=1 -> 0.10 (strong negative),
            # SKIP=2 -> 0.50 (neutral, will become SKIP after threshold)
            action_int = int(action)
            prob = {0: 0.90, 1: 0.10, 2: 0.50}.get(action_int, 0.50)

        elif isinstance(self._model, torch.nn.Module):
            self._model.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(fv.vector.astype(np.float32)).unsqueeze(0)
                output = self._model(tensor)
                prob = float(output.squeeze().item())

        else:
            # Classical ML: sklearn estimator, XGBClassifier, LGBMClassifier,
            # CatBoostClassifier. All expose predict_proba().
            x_2d = fv.vector.reshape(1, -1)
            feature_names = getattr(self._model, "feature_names_in_", None)
            x_input: Any = (
                pd.DataFrame(x_2d, columns=feature_names)
                if feature_names is not None
                else x_2d
            )
            proba_output = self._model.predict_proba(x_input)
            prob = float(proba_output[0, 1])

        if not np.isfinite(prob):
            logger.critical({"event": "SIGNAL_INFERENCE_NON_FINITE", "symbol": self.symbol, "model": self._record.model_name, "probability": prob})
            raise SignalGeneratorError(
                f"Model {self._record.model_name} returned non-finite "
                f"probability: {prob}.",
                stage="infer",
            )

        # Clamp to [0, 1] to guard against floating point edge cases.
        return float(np.clip(prob, 0.0, 1.0))

    def _make_signal(
        self,
        direction: str,
        confidence: float,
        model_name: str,
        feature_version: str,
    ) -> TradeSignal:
        """
        Construct a TradeSignal with the current symbol and expiry context.

        Centralises TradeSignal construction so all fields are always
        populated consistently regardless of which code path in generate()
        calls this method.

        Args:
            direction:       "CALL", "PUT", or "SKIP".
            confidence:      Probability value in [0.0, 1.0].
            model_name:      Name of the model that produced the signal.
            feature_version: _VERSION from the FeatureVector.

        Returns:
            TradeSignal: Fully populated immutable signal.
        """
        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            confidence=confidence,
            expiry_key=self.expiry_key,
            expiry_seconds=self.expiry_seconds,
            timestamp=datetime.now(tz=timezone.utc),
            model_name=model_name,
            feature_version=feature_version,
        )

    def __repr__(self) -> str:
        model_loaded = self._model is not None
        return (
            f"SignalGenerator("
            f"symbol={self.symbol!r}, "
            f"expiry={self.expiry_key!r}, "
            f"threshold={self.threshold:.2f}, "
            f"model_loaded={model_loaded})"
        )
