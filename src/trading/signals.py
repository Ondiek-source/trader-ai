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

Gate compliance
---------------
evaluate_gates() from FeatureEngineer is called before model inference.
If all four gates pass (BB_WIDTH, ATR, RVOL, SPREAD), gate_passed=True
on the TradeSignal. The gate result does not block inference — the model
runs regardless — but live.py uses gate_passed to decide whether to
forward the signal to webhook.py. This allows the journal to record
gate-failing signals for post-analysis without executing them.

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
    FeatureEngineer,
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
        return f"SignalGeneratorError(stage={self.stage!r}, " f"message={str(self)!r})"


# ── TradeSignal ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeSignal:
    """
    Immutable output of a single inference pass through SignalGenerator.

    Carries all fields required by webhook.py (direction, confidence,
    expiry), journal.py (timestamp, symbol, model_name, gate_passed),
    reporter.py (symbol, direction, confidence), and live.py
    (direction for SKIP filtering, gate_passed for execution gating).

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
        gate_passed:     True if all four TradeEligibility gates passed.
                         False does not prevent signal generation but
                         live.py uses it to decide whether to execute.
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
    gate_passed: bool
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

        A signal is executable when it is not SKIP and the gate passed.
        live.py calls this before firing the webhook — it does not
        inspect direction or gate_passed separately.

        Returns:
            bool: True for CALL or PUT signals where gate_passed is True.
        """
        return self.direction != "SKIP" and self.gate_passed

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
            "gate_passed": self.gate_passed,
            "feature_version": self.feature_version,
        }

    def __repr__(self) -> str:
        return (
            f"TradeSignal("
            f"symbol={self.symbol!r}, "
            f"direction={self.direction!r}, "
            f"confidence={self.confidence:.4f}, "
            f"expiry={self.expiry_key!r}, "
            f"gate={self.gate_passed}, "
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
        self._engineer: FeatureEngineer = get_feature_engineer()

        # Model artifact and its registry record. Both None until a
        # trained artifact exists. generate() returns SKIP when None.
        self._model: Any = None
        self._record: ModelRecord | None = None

        # Optional ThresholdManager injected by live.py after construction.
        # When None, generate() falls back to the static config threshold.
        self._threshold_mgr: Any = None

        # Attempt initial load — no error if no artifact found.
        self.reload()

        logger.info(
            "[^] SignalGenerator initialised: symbol=%s expiry=%s "
            "threshold=%.2f model=%s",
            self.symbol,
            self.expiry_key,
            self.threshold,
            self._record.model_name if self._record else "NONE",
        )

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
            warning_block = (
                f"\n{'%' * 60}\n"
                f"SIGNAL GENERATOR WARNING: NO MODEL AVAILABLE\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n\n"
                f"CONTEXT: No trained model artifact was found for this\n"
                f"symbol/expiry combination. All generate() calls will\n"
                f"return SKIP until a model is trained and loaded.\n"
                f"\nFIX: Run a training pass for symbol={self.symbol}\n"
                f"expiry={self.expiry_key} then call reload().\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
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

            logger.info(
                "[^] SignalGenerator.reload(): loaded model=%s auc=%.4f",
                record.model_name,
                record.auc,
            )
            return True

        except Exception as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"SIGNAL GENERATOR LOAD FAILURE\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n"
                f"Artifact   : {record.artifact_path}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: The model artifact exists but could not be\n"
                f"deserialised. All generate() calls will return SKIP\n"
                f"until this is resolved.\n"
                f"\nFIX: For PyTorch models use inject_model() instead of\n"
                f"reload(). For other failures check the artifact file.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
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
        logger.info(
            "[^] SignalGenerator.set_threshold_manager(): dynamic threshold active "
            "base=%.2f step=%.3f max_streak=%d",
            mgr.base_threshold,
            mgr.step,
            mgr.max_streak,
        )

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
        logger.info(
            "[^] SignalGenerator.inject_model(): injected model=%s",
            record.model_name,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        fv: FeatureVector,
        fe_df: pd.DataFrame,
    ) -> TradeSignal:
        """
        Translate a FeatureVector into a TradeSignal.

        Pipeline:
            1. Evaluate trade eligibility gates via FeatureEngineer.
            2. If no model is loaded, return SKIP immediately.
            3. Run model inference via _infer() to get a raw probability.
            4. Apply confidence threshold to determine direction.
            5. Return an immutable TradeSignal.

        The gate result does not block inference — the model runs
        regardless of gate state. This means CALL/PUT signals can be
        generated even when gates fail. live.py checks
        signal.is_executable() (which requires gate_passed=True) before
        forwarding to webhook.py, allowing gate-failing signals to be
        journaled without being executed.

        Args:
            fv:    FeatureVector from FeatureEngineer.get_latest().
                    Carries the 50-feature float32 vector for inference.
            fe_df: Full feature DataFrame from FeatureEngineer.transform()
                    for the current bar window. Required for evaluate_gates()
                    which reads BB_WIDTH, ATR, RVOL, SPREAD columns.

        Returns:
            TradeSignal: Always returns a signal — never raises for normal
                inference failures. Unrecoverable errors raise
                SignalGeneratorError.

        Raises:
            SignalGeneratorError: If gate evaluation or inference fails
                in a way that cannot produce a valid TradeSignal.
        """
        # ── 1. Gate evaluation ────────────────────────────────────────────
        try:
            eligibility = self._engineer.evaluate_gates(fe_df)
            gate_passed: bool = eligibility.is_eligible
        except Exception as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"SIGNAL GENERATOR GATE EVALUATION FAILURE\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: evaluate_gates() could not read the required gate\n"
                f"columns from the feature DataFrame (BB_WIDTH, ATR_14, RVOL,\n"
                f"SPREAD_NORMALIZED). This usually means the feature DataFrame\n"
                f"passed to generate() is missing derived columns.\n"
                f"\nFIX: Ensure fe_df is the full output of FeatureEngineer.transform()\n"
                f"including the DERIVED pass, not a sliced or partial DataFrame.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise SignalGeneratorError(
                f"evaluate_gates() failed: {exc}", stage="gate"
            ) from exc

        # ── 2. No model — return SKIP ─────────────────────────────────────
        if self._model is None or self._record is None:
            return self._make_signal(
                direction="SKIP",
                confidence=0.0,
                gate_passed=gate_passed,
                model_name="NONE",
                feature_version=fv.version,
            )

        # ── 3. Model inference ────────────────────────────────────────────
        try:
            prob: float = self._infer(fv)
        except Exception as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"SIGNAL GENERATOR INFERENCE FAILURE\n"
                f"Symbol     : {self.symbol}\n"
                f"Expiry key : {self.expiry_key}\n"
                f"Model      : {self._record.model_name}\n"
                f"Error      : {exc}\n\n"
                f"CONTEXT: The model artifact failed during the forward pass.\n"
                f"This may indicate a corrupted artifact, a shape mismatch\n"
                f"between the FeatureVector and the model's expected input,\n"
                f"or a PyTorch model that requires a sequence tensor.\n"
                f"\nFIX: Verify the artifact with model_manager.validate_metadata().\n"
                f"For PyTorch sequence models ensure inject_model() was used\n"
                f"with the correct architecture rather than reload().\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise SignalGeneratorError(
                f"Model inference failed for {self._record.model_name}: {exc}",
                stage="infer",
            ) from exc

        # ── 4. Confidence threshold → direction ───────────────────────────
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
            gate_passed=gate_passed,
            model_name=self._record.model_name,
            feature_version=fv.version,
        )

        logger.info(
            "[^] Signal: symbol=%s direction=%s confidence=%.4f "
            "gate=%s model=%s expiry=%s threshold=%.3f",
            self.symbol,
            signal.direction,
            signal.confidence,
            signal.gate_passed,
            signal.model_name,
            signal.expiry_key,
            effective_threshold,
        )

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
            proba_output = self._model.predict_proba(x_2d)
            prob = float(proba_output[0, 1])

        if not np.isfinite(prob):
            error_block = (
                f"\n{'!' * 60}\n"
                f"SIGNAL GENERATOR INFERENCE FAILURE: NON-FINITE OUTPUT\n"
                f"Symbol     : {self.symbol}\n"
                f"Model      : {self._record.model_name}\n"
                f"Probability: {prob}\n\n"
                f"CONTEXT: The model returned NaN or Inf instead of a valid\n"
                f"probability. This usually means the model received a feature\n"
                f"vector containing NaN or Inf values, or the model artifact\n"
                f"is corrupted (weights contain NaN from a failed training run).\n"
                f"\nFIX: Check FeatureEngineer.get_latest() logs for non-finite\n"
                f"feature warnings. If the artifact is corrupt, retrain and\n"
                f"re-upload via model_manager.save().\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
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
        gate_passed: bool,
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
            gate_passed:     Gate eligibility result.
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
            gate_passed=gate_passed,
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
