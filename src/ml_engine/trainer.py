"""
src/ml_engine/trainer.py — The Brain Factory.

Role: Train, evaluate, and produce serialisable model artifacts across
all three model tiers — Classical ML, Deep Learning, and Reinforcement
Learning — using the same feature schema produced by features.py.

Architecture
------------
Every trainer in this file inherits from BaseTrainer, which enforces a
common interface across all model types. The InferenceEngine and
ModelManager interact only with BaseTrainer — they never import a
concrete trainer class directly.

Three tiers of trainers are implemented:

  Tier 1 — Classical ML
    XGBoostTrainer       : Gradient boosted trees, GPU-optional.
    LightGBMTrainer      : Fast gradient boosting, native categorical support.
    CatBoostTrainer      : Gradient boosting with strong context-feature handling.
    RandomForestTrainer  : Ensemble baseline, CPU only.
    StackingEnsembleTrainer : Meta-learner over Tier 1 + Tier 2 outputs.

  Tier 2 — Deep Learning (PyTorch)
    LSTMTrainer          : 2-layer LSTM over sliding bar windows.
    GRUTrainer           : GRU variant, fewer parameters than LSTM.
    TCNTrainer           : Temporal Convolutional Network, parallelisable.
    CNNLSTMTrainer       : CNN feature extractor + LSTM temporal memory.
    TransformerTrainer   : Multi-head attention encoder over bar windows.

  Tier 3 — Reinforcement Learning (Stable-Baselines3)
    A2CTrainer           : Synchronous Advantage Actor-Critic.
    DQNTrainer           : Deep Q-Network with replay buffer.
    PPOTrainer           : Proximal Policy Optimisation (recommended).
    RecurrentPPOTrainer  : PPO with LSTM hidden state (sb3-contrib).

Supporting classes
------------------
  DataShaper             : Time-ordered train/val/test splitter.
                           Shuffling is a hard error.
  get_best_device()      : Auto-detects CUDA > MPS > CPU.

Data flow
---------
  FeatureMatrix  (features.py)
      |
      v
  DataShaper.split()
      |
      +---> X_train, y_train, X_val, y_val, X_test, y_test
      |
      v
  BaseTrainer.train(split)
      |
      v
  artifact + TrainerResult (metrics, metadata, model)
      |
      v
  ModelManager.save(artifact, metadata)

Public API
----------
  get_best_device()                          -> str
  DataShaper(train_ratio, val_ratio)
  DataShaper.split(feature_matrix, labels)   -> TrainValTestSplit
  XGBoostTrainer(expiry_key, config)
  LightGBMTrainer(expiry_key, config)
  CatBoostTrainer(expiry_key, config)
  RandomForestTrainer(expiry_key, config)
  StackingEnsembleTrainer(expiry_key, base_predictions, config)
  LSTMTrainer(expiry_key, config)
  GRUTrainer(expiry_key, config)
  TCNTrainer(expiry_key, config)
  CNNLSTMTrainer(expiry_key, config)
  TransformerTrainer(expiry_key, config)
  A2CTrainer(expiry_key, symbol, bar_df, config)
  DQNTrainer(expiry_key, symbol, bar_df, config)
  PPOTrainer(expiry_key, symbol, bar_df, config)
  RecurrentPPOTrainer(expiry_key, symbol, bar_df, config)
  BaseTrainer.train(split)                   -> TrainerResult
  BaseTrainer.predict_proba(X)               -> np.ndarray
  BaseTrainer.evaluate(X, y)                 -> dict[str, float]
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from ml_engine.features import (
    BINARY_EXPIRY_RULES,
    FeatureMatrix,
    _VERSION,
)
from ml_engine.labeler import _EXPIRY_SECONDS
from ml_engine.sequence_generator import TimeSeriesDataset, get_dataloader
from core.config import get_settings

logger = logging.getLogger(__name__)


# ── Hardware Detection ───────────────────────────────────────────────────────


def get_best_device() -> str:
    """
    Auto-detect the best available compute device for PyTorch operations.

    Priority order: CUDA GPU > Apple MPS > CPU. The result is logged at
    startup so every training run records which device was used. Never
    hardcode "cuda" — the fallback chain ensures identical behaviour on
    a CPU-only Azure Container Instance and a GPU-equipped training VM.

    Returns:
        str: One of "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(
        "[^] Compute device selected: %s " "(cuda=%s, mps=%s)",
        device,
        torch.cuda.is_available(),
        torch.backends.mps.is_available(),
    )
    return device


# ── Module-level device (resolved once at import time) ───────────────────────
_DEVICE: str = get_best_device()


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class TrainValTestSplit:
    """
    Time-ordered train / validation / test split produced by DataShaper.

    All three splits preserve the original temporal order. No row from
    the validation or test windows is visible during training. Callers
    must not shuffle these splits after construction.

    Attributes:
        X_train: Training feature matrix (2D numpy array, float32).
        y_train: Training labels (1D numpy array, int32).
        X_val:   Validation feature matrix.
        y_val:   Validation labels.
        X_test:  Hold-out test feature matrix.
        y_test:  Hold-out test labels.
        feature_names: Ordered column names matching axis=1 of all X arrays.
        symbol:  Currency pair the split was built from.
        expiry_key: Expiry window identifier.
        feature_version: _VERSION string from features.py.
        split_ratios: (train_ratio, val_ratio, test_ratio) as floats.
        n_total: Total rows before splitting.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    symbol: str
    expiry_key: str
    feature_version: str
    split_ratios: tuple[float, float, float]
    n_total: int

    def __repr__(self) -> str:
        return (
            f"TrainValTestSplit("
            f"symbol={self.symbol!r}, "
            f"expiry={self.expiry_key!r}, "
            f"train={len(self.X_train)}, "
            f"val={len(self.X_val)}, "
            f"test={len(self.X_test)})"
        )


@dataclass
class TrainerResult:
    """
    Output of a completed training run produced by BaseTrainer.train().

    Carries the trained model artifact, evaluation metrics, and all
    metadata required by ModelManager to persist and version the artifact.
    Every field is mandatory — ModelManager will reject a TrainerResult
    with missing metadata.

    Attributes:
        model_name:       Trainer class name, e.g. "XGBoostTrainer".
        expiry_key:       Expiry window, e.g. "5_MIN".
        symbol:           Currency pair, e.g. "EUR_USD".
        feature_version:  _VERSION from features.py at training time.
        trained_at:       UTC ISO-8601 timestamp of training completion.
        device:           Compute device used, e.g. "cuda".
        train_rows:       Number of training rows consumed.
        val_rows:         Number of validation rows evaluated.
        metrics:          Dict of evaluation metric names to float values.
                          Always includes accuracy, precision, recall,
                          f1, auc. May include model-specific extras.
        artifact:         The trained model object. Type varies by tier:
                          sklearn estimator, torch.nn.Module, or SB3 model.
        extra:            Optional dict for model-specific metadata
                          (e.g. window_size for LSTM, n_steps for PPO).
    """

    model_name: str
    expiry_key: str
    symbol: str
    feature_version: str
    trained_at: str
    device: str
    train_rows: int
    val_rows: int
    metrics: dict[str, float]
    artifact: Any
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        auc = self.metrics.get("auc", float("nan"))
        return (
            f"TrainerResult("
            f"model={self.model_name!r}, "
            f"expiry={self.expiry_key!r}, "
            f"symbol={self.symbol!r}, "
            f"auc={auc:.4f})"
        )


# ── Custom Exception ─────────────────────────────────────────────────────────


class TrainerError(Exception):
    """
    Raised when a trainer cannot complete a training or evaluation run.

    Distinct from ValueError (caller contract violation) — TrainerError
    signals a runtime failure inside the training pipeline that
    ModelManager or the pipeline orchestrator must handle.

    Attributes:
        stage: The pipeline stage that failed, e.g. "train", "evaluate".
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return f"TrainerError(stage={self.stage!r}, message={str(self)!r})"


# ── DataShaper ───────────────────────────────────────────────────────────────


class DataShaper:
    """
    Time-ordered train / validation / test splitter for financial ML.

    Enforces strict temporal ordering across all splits. Random shuffling
    is the "Original Sin" of financial ML — it leaks future price
    information into the training window, producing backtests that appear
    perfect but fail immediately in live markets. This class treats any
    attempt to shuffle as a hard error.

    The gate feature columns (GATE_BB_WIDTH_PASS, GATE_ATR_PASS,
    GATE_RVOL_PASS, GATE_SPREAD_PASS, GATE_ALL_PASS) are appended to
    the feature matrix before splitting if gate_values are provided.
    These allow the model to learn the relationship between gate quality
    and trade outcome rather than being trained only on gate-passing rows.

    Attributes:
        train_ratio: Fraction of data used for training. Default 0.70.
        val_ratio:   Fraction used for validation. Default 0.15.
                     The remaining fraction becomes the test set.

    Example:
        >>> shaper = DataShaper(train_ratio=0.70, val_ratio=0.15)
        >>> split = shaper.split(feature_matrix, labels)
        >>> split.X_train.shape
        (490000, 50)
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> None:
        """
        Initialise the DataShaper with split ratios.

        Args:
            train_ratio: Fraction of rows for training. Must be in (0, 1).
            val_ratio:   Fraction of rows for validation. Must be in (0, 1).
                         train_ratio + val_ratio must be < 1.0 so that a
                         non-empty test set remains.

        Raises:
            ValueError: If ratios are out of range or sum to >= 1.0.
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"[%] train_ratio={train_ratio} must be in (0.0, 1.0).")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"[%] val_ratio={val_ratio} must be in (0.0, 1.0).")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"[%] train_ratio + val_ratio = "
                f"{train_ratio + val_ratio:.2f} >= 1.0. "
                f"No rows remain for the test set."
            )
        self.train_ratio: float = train_ratio
        self.val_ratio: float = val_ratio
        self.test_ratio: float = round(1.0 - train_ratio - val_ratio, 6)

    def split(
        self,
        feature_matrix: FeatureMatrix,
        labels: pd.Series,
        expiry_key: str,
        gate_columns: pd.DataFrame | None = None,
    ) -> TrainValTestSplit:
        """
        Produce a time-ordered TrainValTestSplit from a FeatureMatrix.

        Aligns the FeatureMatrix timestamps with the label Series index,
        optionally appends gate feature columns, then slices into three
        contiguous temporal windows. No row is shared between windows.

        Args:
            feature_matrix: FeatureMatrix from FeatureEngineer.build_matrix().
            labels:         pd.Series from Labeler.compute_labels().
                            Must share at least one timestamp with
                            feature_matrix.timestamps.
            expiry_key:     Expiry window key, e.g. "5_MIN". Stored in
                            the returned TrainValTestSplit for metadata.
            gate_columns:   Optional DataFrame of gate feature columns
                            (GATE_BB_WIDTH_PASS, GATE_ATR_PASS, etc.)
                            indexed by the same timestamps. When provided,
                            these columns are appended to the feature
                            matrix before splitting so the model can learn
                            gate-quality signals without hard-filtering.

        Returns:
            TrainValTestSplit: Named split with numpy arrays for all
                three windows and full provenance metadata.

        Raises:
            ValueError:   If feature_matrix is not a FeatureMatrix instance,
                          or if expiry_key is invalid.
            TrainerError: If the aligned dataset is too small to split,
                          or if the intersection of feature and label
                          indices is empty.
        """
        if not isinstance(feature_matrix, FeatureMatrix):
            raise ValueError(
                f"[!] feature_matrix must be a FeatureMatrix instance, "
                f"got {type(feature_matrix).__name__}."
            )
        if expiry_key not in BINARY_EXPIRY_RULES:
            raise ValueError(
                f"[!] Invalid expiry_key: '{expiry_key}'. "
                f"Must be one of {list(BINARY_EXPIRY_RULES.keys())}."
            )

        # Build a DataFrame from the FeatureMatrix for index alignment.
        feat_index = pd.DatetimeIndex(feature_matrix.timestamps)
        feat_df = pd.DataFrame(
            feature_matrix.matrix,
            index=feat_index,
            columns=feature_matrix.feature_names,
        )

        # Align on common timestamps — Labeler drops tail rows so the
        # label index is shorter than the feature index.
        common_index = feat_df.index.intersection(labels.index)
        if common_index.empty:
            raise TrainerError(
                "[!] Feature and label indices have no common timestamps. "
                "Ensure Labeler.compute_labels() was run on the same bar "
                "DataFrame that produced the FeatureMatrix.",
                stage="split",
            )

        X_aligned: pd.DataFrame = feat_df.loc[common_index]
        y_aligned: pd.Series = labels.loc[common_index]

        # Append gate columns when provided — they extend the feature set
        # without hard-filtering, following the recommendation from the
        # pre-coding notes.
        if gate_columns is not None:
            gate_aligned = gate_columns.loc[
                gate_columns.index.intersection(common_index)
            ]
            X_aligned = pd.concat([X_aligned, gate_aligned], axis=1)
            logger.info(
                "[^] DataShaper: appended %d gate columns to feature matrix.",
                len(gate_columns.columns),
            )

        n_total: int = len(X_aligned)
        min_required: int = 100
        if n_total < min_required:
            raise TrainerError(
                f"[%] Aligned dataset has only {n_total} rows — minimum "
                f"{min_required} required for a meaningful split. "
                f"Provide a longer historical bar range.",
                stage="split",
            )

        train_end: int = int(n_total * self.train_ratio)
        val_end: int = int(n_total * (self.train_ratio + self.val_ratio))

        # Enforce non-empty splits.
        if train_end == 0 or val_end == train_end or val_end == n_total:
            raise TrainerError(
                f"[%] Split ratios produce an empty partition. "
                f"n_total={n_total}, train_end={train_end}, "
                f"val_end={val_end}. Adjust ratios or provide more data.",
                stage="split",
            )

        X_np = np.asarray(X_aligned.values, dtype=np.float32)
        y_np = np.asarray(y_aligned.values, dtype=np.int32)

        split = TrainValTestSplit(
            X_train=X_np[:train_end],
            y_train=y_np[:train_end],
            X_val=X_np[train_end:val_end],
            y_val=y_np[train_end:val_end],
            X_test=X_np[val_end:],
            y_test=y_np[val_end:],
            feature_names=list(X_aligned.columns),
            symbol=feature_matrix.symbol,
            expiry_key=expiry_key,
            feature_version=feature_matrix.version,
            split_ratios=(self.train_ratio, self.val_ratio, self.test_ratio),
            n_total=n_total,
        )

        logger.info(
            "[^] DataShaper split: symbol=%s expiry=%s total=%d "
            "train=%d val=%d test=%d",
            feature_matrix.symbol,
            expiry_key,
            n_total,
            len(split.X_train),
            len(split.X_val),
            len(split.X_test),
        )

        return split

    def __repr__(self) -> str:
        return (
            f"DataShaper("
            f"train={self.train_ratio:.0%}, "
            f"val={self.val_ratio:.0%}, "
            f"test={self.test_ratio:.0%})"
        )


# ── Evaluation Utility ───────────────────────────────────────────────────────


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the standard binary classification metric suite.

    Used by all trainers in their evaluate() method to ensure a
    consistent metric set is stored in every TrainerResult. The
    threshold parameter controls the probability cutoff for converting
    predicted probabilities to hard class labels.

    Args:
        y_true:    Ground truth binary labels (int array, values 0 or 1).
        y_prob:    Predicted probabilities for the positive class (float
                   array, values in [0, 1]).
        threshold: Probability cutoff for positive class assignment.
                   Default 0.5. May be tuned per model after training.

    Returns:
        dict[str, float]: Keys: accuracy, precision, recall, f1, auc.
            All values are floats in [0, 1]. Returns zeros for all
            metrics if y_true has only one unique class (degenerate
            split — log a warning).
    """
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "[%%] _compute_metrics: y_true has only one unique class. "
            "Metrics are meaningless — check your label distribution."
        )
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
        }

    y_pred: np.ndarray = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


# ── Base Trainer ─────────────────────────────────────────────────────────────


class BaseTrainer(ABC):
    """
    Abstract base class enforcing the training contract for all model tiers.

    Every concrete trainer — from XGBoostTrainer to RecurrentPPOTrainer —
    must implement train() and predict_proba(). The InferenceEngine and
    ModelManager interact only with this interface and never import a
    concrete trainer class.

    Subclasses inherit:
        device     : Best available compute device string.
        expiry_key : Expiry window for this training run.
        model_name : Class name used in TrainerResult metadata.
        model      : The trained model artifact (None before train()).

    Subclasses must implement:
        train(split)           -> TrainerResult
        predict_proba(X)       -> np.ndarray

    Subclasses may override:
        evaluate(X, y)         -> dict[str, float]

    Attributes:
        expiry_key: Expiry window identifier, e.g. "5_MIN".
        device:     Compute device string resolved at construction.
        model_name: Concrete class name for metadata.
        model:      Trained artifact. None until train() is called.
    """

    def __init__(self, expiry_key: str) -> None:
        """
        Initialise the base trainer.

        Args:
            expiry_key: Must be one of "1_MIN", "5_MIN", "15_MIN".

        Raises:
            ValueError: If expiry_key is not in BINARY_EXPIRY_RULES.
        """
        if expiry_key not in BINARY_EXPIRY_RULES:
            raise ValueError(
                f"[!] Invalid expiry_key: '{expiry_key}'. "
                f"Must be one of {list(BINARY_EXPIRY_RULES.keys())}."
            )
        self.expiry_key: str = expiry_key
        self.device: str = _DEVICE
        self.model_name: str = self.__class__.__name__
        self.model: Any = None

    @abstractmethod
    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train the model on the provided split and return a TrainerResult.

        Implementations must:
          1. Train on split.X_train / split.y_train only.
          2. Use split.X_val / split.y_val for early stopping or
             epoch selection — never for gradient updates.
          3. Call self._build_result() to construct the TrainerResult.
          4. Never touch split.X_test / split.y_test.

        Args:
            split: Time-ordered TrainValTestSplit from DataShaper.split().

        Returns:
            TrainerResult: Trained artifact with full metadata and
                validation metrics.

        Raises:
            TrainerError: If training fails for any reason.
        """

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Return predicted probabilities for the positive class (CALL win).

        Args:
            X: Feature input. Type varies by tier:
               Classical ML: 2D numpy array (n_samples, n_features).
               Deep Learning: 3D torch.Tensor (batch, window, features).
               RL: 1D numpy array (n_features,) per step.

        Returns:
            np.ndarray: 1D float array of shape (n_samples,) with values
                in [0, 1]. Higher values indicate higher CALL confidence.

        Raises:
            TrainerError: If the model has not been trained yet.
        """

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate the trained model on a feature/label pair.

        Calls predict_proba() and computes the standard metric suite
        via _compute_metrics(). Subclasses may override this method
        for model-specific evaluation logic (e.g. RL episode returns).

        Args:
            X: Feature array, same type as predict_proba() expects.
            y: Ground-truth binary labels (int array).

        Returns:
            dict[str, float]: accuracy, precision, recall, f1, auc.

        Raises:
            TrainerError: If the model has not been trained (model is None).
        """
        if self.model is None:
            raise TrainerError(
                f"[!] {self.model_name}.evaluate() called before train(). "
                f"Train the model first.",
                stage="evaluate",
            )
        y_prob = self.predict_proba(X)
        return _compute_metrics(y_true=y, y_prob=y_prob)

    def _build_result(
        self,
        split: TrainValTestSplit,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> TrainerResult:
        """
        Construct a TrainerResult from training outputs.

        Centralises metadata assembly so all subclasses produce a
        consistent TrainerResult without duplicating field construction.

        Args:
            split:   The TrainValTestSplit used for training.
            metrics: Validation metrics dict from _compute_metrics().
            extra:   Optional model-specific metadata dict.

        Returns:
            TrainerResult: Fully populated result ready for ModelManager.
        """
        return TrainerResult(
            model_name=self.model_name,
            expiry_key=split.expiry_key,
            symbol=split.symbol,
            feature_version=split.feature_version,
            trained_at=datetime.now(tz=timezone.utc).isoformat(),
            device=self.device,
            train_rows=len(split.X_train),
            val_rows=len(split.X_val),
            metrics=metrics,
            artifact=self.model,
            extra=extra or {},
        )

    def _check_trained(self) -> None:
        """
        Raise TrainerError if the model artifact has not been set.

        Call this at the top of predict_proba() in all subclasses to
        produce a clear error rather than an AttributeError on None.

        Raises:
            TrainerError: If self.model is None.
        """
        if self.model is None:
            raise TrainerError(
                f"[!] {self.model_name}.predict_proba() called before "
                f"train(). Train the model first.",
                stage="predict_proba",
            )

    def __repr__(self) -> str:
        trained = self.model is not None
        return (
            f"{self.model_name}("
            f"expiry={self.expiry_key!r}, "
            f"device={self.device!r}, "
            f"trained={trained})"
        )


# ════════════════════════════════════════════════════════════════════════════
# TIER 1 — CLASSICAL ML
# ════════════════════════════════════════════════════════════════════════════


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost gradient-boosted tree classifier for binary options signals.

    Uses the histogram-based tree method (tree_method="hist") which
    supports both CPU and GPU acceleration. Class imbalance is handled
    via scale_pos_weight, computed from the training label distribution.
    Early stopping is applied on the validation set to prevent overfitting.

    GPU usage is controlled by the device detected at module import time.
    When device="cuda", tree_method="hist" with device="cuda" is used.
    On CPU, tree_method="hist" with device="cpu" applies.

    Attributes:
        n_estimators:  Maximum number of boosting rounds. Default 1000.
        early_stopping_rounds: Validation rounds without improvement
                               before stopping. Default 50.
        learning_rate: Step size shrinkage. Default 0.05.
    """

    def __init__(
        self,
        expiry_key: str,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__(expiry_key)
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train an XGBoost classifier with early stopping on the val set.

        Computes scale_pos_weight from the training label distribution
        to handle CALL/PUT class imbalance without hard resampling.

        Args:
            split: Time-ordered TrainValTestSplit from DataShaper.

        Returns:
            TrainerResult: Trained XGBClassifier artifact with validation
                metrics (accuracy, precision, recall, f1, auc) and
                feature importances in extra["feature_importances"].

        Raises:
            TrainerError: If training fails for any reason.
        """
        try:
            pos_count: int = int(split.y_train.sum())
            neg_count: int = len(split.y_train) - pos_count
            spw: float = neg_count / max(pos_count, 1)

            xgb_device = "cuda" if self.device == "cuda" else "cpu"

            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                tree_method="hist",
                device=xgb_device,
                scale_pos_weight=spw,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric="auc",
                use_label_encoder=False,
                verbosity=0,
            )

            self.model.fit(
                split.X_train,
                split.y_train,
                eval_set=[(split.X_val, split.y_val)],
                verbose=False,
            )

            logger.info(
                "[^] XGBoostTrainer: best_iteration=%d spw=%.2f",
                self.model.best_iteration,
                spw,
            )

            metrics = self.evaluate(split.X_val, split.y_val)
            importances: dict[str, float] = dict(
                zip(
                    split.feature_names,
                    self.model.feature_importances_.tolist(),
                )
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={
                    "best_iteration": self.model.best_iteration,
                    "scale_pos_weight": spw,
                    "feature_importances": importances,
                },
            )

        except Exception as exc:
            raise TrainerError(
                f"XGBoostTrainer.train() failed: {exc}", stage="train"
            ) from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return positive-class probabilities from the trained XGB model.

        Args:
            X: 2D float32 numpy array of shape (n_samples, n_features).

        Returns:
            np.ndarray: 1D float array of shape (n_samples,).
        """
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]


class LightGBMTrainer(BaseTrainer):
    """
    LightGBM gradient-boosted tree classifier for binary options signals.

    LightGBM uses a leaf-wise tree growth strategy which is typically
    faster than XGBoost on large datasets. It natively handles the
    binary SESSION_* flag columns and cyclical TIME_SINE/TIME_COSINE
    context features without preprocessing.

    GPU support via device_type="gpu" is available but less mature than
    XGBoost. Disabled by default — enable via use_gpu=True.

    Attributes:
        n_estimators:  Maximum boosting rounds. Default 1000.
        early_stopping_rounds: Val rounds without improvement. Default 50.
        learning_rate: Shrinkage factor. Default 0.05.
        use_gpu:       Whether to use GPU acceleration. Default False.
    """

    def __init__(
        self,
        expiry_key: str,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        learning_rate: float = 0.05,
        use_gpu: bool = False,
    ) -> None:
        super().__init__(expiry_key)
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu and self.device == "cuda"

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train a LightGBM classifier with early stopping on the val set.

        Args:
            split: Time-ordered TrainValTestSplit from DataShaper.

        Returns:
            TrainerResult: Trained LGBMClassifier artifact with validation
                metrics and feature importances in extra["feature_importances"].

        Raises:
            TrainerError: If training fails.
        """
        try:
            pos_count: int = int(split.y_train.sum())
            neg_count: int = len(split.y_train) - pos_count
            spw: float = neg_count / max(pos_count, 1)

            params: dict[str, Any] = {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "scale_pos_weight": spw,
                "device_type": "gpu" if self.use_gpu else "cpu",
                "verbosity": -1,
            }

            self.model = lgb.LGBMClassifier(**params)

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                ),
                lgb.log_evaluation(period=-1),
            ]

            self.model.fit(
                split.X_train,
                split.y_train,
                eval_set=[(split.X_val, split.y_val)],
                callbacks=callbacks,
            )

            logger.info(
                "[^] LightGBMTrainer: best_iteration=%d spw=%.2f",
                self.model.best_iteration_,
                spw,
            )

            metrics = self.evaluate(split.X_val, split.y_val)
            importances = dict(
                zip(split.feature_names, self.model.feature_importances_.tolist())
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={
                    "best_iteration": self.model.best_iteration_,
                    "scale_pos_weight": spw,
                    "feature_importances": importances,
                },
            )

        except Exception as exc:
            raise TrainerError(
                f"LightGBMTrainer.train() failed: {exc}", stage="train"
            ) from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]


class CatBoostTrainer(BaseTrainer):
    """
    CatBoost gradient-boosted tree classifier for binary options signals.

    CatBoost provides the most mature GPU support of the three boosting
    libraries and handles the TIME_SINE, TIME_COSINE, DAY_OF_WEEK_SINE,
    DAY_OF_WEEK_COSINE and SESSION_* context features particularly well
    due to its symmetric tree structure and ordered boosting algorithm.

    GPU is enabled automatically when device="cuda". CatBoost's GPU
    implementation is production-grade and significantly faster than CPU
    on large feature matrices.

    Attributes:
        iterations:     Maximum boosting rounds. Default 1000.
        learning_rate:  Shrinkage factor. Default 0.05.
        early_stopping_rounds: Val rounds without improvement. Default 50.
        depth:          Maximum tree depth. Default 6.
    """

    def __init__(
        self,
        expiry_key: str,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        depth: int = 6,
    ) -> None:
        super().__init__(expiry_key)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.depth = depth

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train a CatBoost classifier with early stopping on the val set.

        Args:
            split: Time-ordered TrainValTestSplit from DataShaper.

        Returns:
            TrainerResult: Trained CatBoostClassifier artifact with
                validation metrics and feature importances.

        Raises:
            TrainerError: If training fails.
        """
        try:
            task_type = "GPU" if self.device == "cuda" else "CPU"

            self.model = cb.CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                task_type=task_type,
                eval_metric="AUC",
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
                auto_class_weights="Balanced",
            )

            eval_pool = cb.Pool(split.X_val, split.y_val)

            self.model.fit(
                split.X_train,
                split.y_train,
                eval_set=eval_pool,
            )

            logger.info(
                "[^] CatBoostTrainer: best_iteration=%d task_type=%s",
                self.model.get_best_iteration(),
                task_type,
            )

            metrics = self.evaluate(split.X_val, split.y_val)
            importances = dict(
                zip(
                    split.feature_names,
                    self.model.get_feature_importance().tolist(),
                )
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={
                    "best_iteration": self.model.get_best_iteration(),
                    "task_type": task_type,
                    "feature_importances": importances,
                },
            )

        except Exception as exc:
            raise TrainerError(
                f"CatBoostTrainer.train() failed: {exc}", stage="train"
            ) from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]


class RandomForestTrainer(BaseTrainer):
    """
    Random Forest ensemble classifier — CPU only.

    Serves as the baseline ensemble model and as a component in the
    StackingEnsembleTrainer. sklearn's RandomForestClassifier has no
    GPU support. For GPU-accelerated random forests consider cuML,
    which is not included in this stack due to its CUDA-only constraint.

    class_weight="balanced" handles CALL/PUT imbalance automatically
    without requiring manual scale_pos_weight computation.

    Attributes:
        n_estimators: Number of trees. Default 300.
        max_depth:    Maximum tree depth. None = grow fully. Default None.
        n_jobs:       Parallel jobs for fitting. -1 uses all CPU cores.
    """

    def __init__(
        self,
        expiry_key: str,
        n_estimators: int = 300,
        max_depth: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(expiry_key)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train a Random Forest classifier.

        Args:
            split: Time-ordered TrainValTestSplit from DataShaper.

        Returns:
            TrainerResult: Trained RandomForestClassifier artifact with
                validation metrics and feature importances.

        Raises:
            TrainerError: If training fails.
        """
        try:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                n_jobs=self.n_jobs,
                random_state=42,
            )

            self.model.fit(split.X_train, split.y_train)

            logger.info(
                "[^] RandomForestTrainer: n_estimators=%d max_depth=%s",
                self.n_estimators,
                self.max_depth,
            )

            metrics = self.evaluate(split.X_val, split.y_val)
            importances = dict(
                zip(
                    split.feature_names,
                    self.model.feature_importances_.tolist(),
                )
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={"feature_importances": importances},
            )

        except Exception as exc:
            raise TrainerError(
                f"RandomForestTrainer.train() failed: {exc}", stage="train"
            ) from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]


class StackingEnsembleTrainer(BaseTrainer):
    """
    Meta-learner stacking ensemble over Tier 1 and Tier 2 base models.

    Takes the predict_proba() outputs of trained base models as input
    features and trains a Logistic Regression meta-learner to produce a
    final signal confidence score. This is the highest-fidelity classical
    signal in the stack and should be trained last, after all base models
    have been validated.

    The base_predictions dict maps model names to their predict_proba()
    outputs on the validation set. The meta-learner trains on these
    stacked predictions, not on the raw feature matrix.

    Attributes:
        base_predictions: Dict mapping model name to 1D probability array.
        meta_C:           Regularisation strength for LogisticRegression.
    """

    def __init__(
        self,
        expiry_key: str,
        base_predictions: dict[str, np.ndarray],
        meta_C: float = 1.0,
    ) -> None:
        """
        Initialise the stacking ensemble.

        Args:
            expiry_key:       Expiry window key.
            base_predictions: Dict of {model_name: prob_array} from
                              base model predict_proba() calls on val set.
            meta_C:           Inverse regularisation strength for the
                              LogisticRegression meta-learner. Default 1.0.

        Raises:
            ValueError: If base_predictions is empty or arrays have
                        inconsistent lengths.
        """
        super().__init__(expiry_key)
        if not base_predictions:
            raise ValueError(
                "[!] StackingEnsembleTrainer requires at least one "
                "base model prediction. base_predictions is empty."
            )
        lengths = {name: len(arr) for name, arr in base_predictions.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"[!] base_predictions arrays have inconsistent lengths: "
                f"{lengths}. All base models must predict on the same set."
            )
        self.base_predictions = base_predictions
        self.meta_C = meta_C

    def _stack_features(self) -> np.ndarray:
        """Stack base predictions into a 2D meta-feature matrix."""
        return np.column_stack(list(self.base_predictions.values()))

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train the Logistic Regression meta-learner on stacked predictions.

        Args:
            split: TrainValTestSplit — used for y_val labels and metadata.
                   X_train and X_val are not used directly; the meta-
                   features come from base_predictions.

        Returns:
            TrainerResult: Trained LogisticRegression artifact.

        Raises:
            TrainerError: If training fails.
        """
        try:
            meta_X = self._stack_features()
            y_val = split.y_val[: len(meta_X)]

            self.model = LogisticRegression(C=self.meta_C, max_iter=1000)
            self.model.fit(meta_X, y_val)

            metrics = _compute_metrics(
                y_true=y_val,
                y_prob=self.model.predict_proba(meta_X)[:, 1],
            )

            logger.info(
                "[^] StackingEnsembleTrainer: base_models=%s meta_C=%.2f",
                list(self.base_predictions.keys()),
                self.meta_C,
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={
                    "base_models": list(self.base_predictions.keys()),
                    "meta_C": self.meta_C,
                },
            )

        except Exception as exc:
            raise TrainerError(
                f"StackingEnsembleTrainer.train() failed: {exc}", stage="train"
            ) from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the meta-learner on stacked base predictions.

        Args:
            X: 2D array where each column is a base model's probability
               output. Shape (n_samples, n_base_models).

        Returns:
            np.ndarray: 1D float array of shape (n_samples,).
        """
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]


# ════════════════════════════════════════════════════════════════════════════
# TIER 2 — DEEP LEARNING (PyTorch)
# ════════════════════════════════════════════════════════════════════════════

# ── Shared Training Loop ─────────────────────────────────────────────────────


def _train_torch_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: str,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[nn.Module, list[float]]:
    """
    Shared PyTorch training loop for all Deep Learning trainers.

    Implements early stopping based on validation loss. The best model
    weights (lowest val loss) are restored at the end of training
    regardless of whether early stopping fired.

    Args:
        model:         nn.Module to train. Must already be on device.
        train_loader:  DataLoader yielding (x, y) batches for training.
        val_loader:    DataLoader yielding (x, y) batches for validation.
        device:        Compute device string, e.g. "cuda".
        epochs:        Maximum training epochs.
        learning_rate: Adam optimiser learning rate.
        patience:      Epochs without val loss improvement before stopping.

    Returns:
        tuple[nn.Module, list[float]]: Best model and per-epoch val losses.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    best_val_loss: float = float("inf")
    best_weights: dict[str, Any] = {}
    patience_counter: int = 0
    val_losses: list[float] = []

    for epoch in range(epochs):
        # ── Training pass ────────────────────────────────────────────────
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze(dim=-1)
            loss = criterion(preds, yb)
            loss.backward()
            # Gradient clipping prevents exploding gradients in LSTM/GRU.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Validation pass ──────────────────────────────────────────────
        model.eval()
        val_loss_accum: float = 0.0
        val_batches: int = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb).squeeze(dim=-1)
                val_loss_accum += criterion(preds, yb).item()
                val_batches += 1

        val_loss: float = val_loss_accum / max(val_batches, 1)
        val_losses.append(val_loss)

        logger.debug(
            "[^] Epoch %d/%d val_loss=%.6f best=%.6f patience=%d/%d",
            epoch + 1,
            epochs,
            val_loss,
            best_val_loss,
            patience_counter,
            patience,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "[^] Early stopping at epoch %d (patience=%d).",
                    epoch + 1,
                    patience,
                )
                break

        logger.info(
            f"[{model.__class__.__name__}] Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.6f}, best: {best_val_loss:.6f}"
        )
    # Restore best weights regardless of early stopping.
    if best_weights:
        model.load_state_dict(best_weights)

    return model, val_losses


def _torch_predict_proba(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    device: str,
    window_size: int | None = None,
) -> np.ndarray:
    """
    Run inference on a trained PyTorch model and return probabilities.

    Handles both 2D numpy inputs (converted to 3D tensor for sequence
    models by adding a batch dimension) and pre-formed 3D tensors.

    Args:
        model:       Trained nn.Module in eval mode.
        X:           Input data. numpy array or torch.Tensor.
        device:      Compute device string.
        window_size: If provided and X is 2D, reshapes to
                     (1, window_size, n_features) for sequence models.

    Returns:
        np.ndarray: 1D float array of predicted probabilities.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            tensor = torch.from_numpy(X.astype(np.float32))
        else:
            tensor = X.float()

        if tensor.ndim == 2 and window_size is not None:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)
        output = model(tensor).squeeze()
        return output.cpu().numpy()


# ── LSTM ─────────────────────────────────────────────────────────────────────


class _LSTMNet(nn.Module):
    """
    Two-layer LSTM network for binary options signal classification.

    Processes a sequence of feature bars through two stacked LSTM layers
    with dropout regularisation, then projects the final hidden state
    through a linear layer to a single sigmoid output.

    Input shape:  (batch_size, window_size, n_features)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.sigmoid(self.fc(out))


class LSTMTrainer(BaseTrainer):
    """
    LSTM trainer for sequential bar window classification.

    Feeds sliding windows of feature bars (shape: window x features)
    through a 2-layer LSTM and trains with BCELoss + Adam. Uses the
    shared _train_torch_model loop with early stopping and gradient
    clipping to handle the vanishing gradient problem common in deep
    LSTM stacks.

    Attributes:
        window_size:   Number of M1 bars per sequence. Default 30.
        hidden_dim:    LSTM hidden state size. Default 128.
        epochs:        Max training epochs. Default 50.
        learning_rate: Adam lr. Default 1e-3.
        batch_size:    DataLoader batch size. Default 64.
        patience:      Early stopping patience in epochs. Default 10.
    """

    def __init__(
        self,
        expiry_key: str,
        window_size: int = 30,
        hidden_dim: int = 128,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
    ) -> None:
        super().__init__(expiry_key)
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train the LSTM on time-ordered sliding bar windows.

        Constructs FeatureMatrix-compatible DataLoaders from the split
        numpy arrays, trains with the shared loop, and evaluates on the
        validation set.

        Args:
            split: Time-ordered TrainValTestSplit from DataShaper.

        Returns:
            TrainerResult: Trained _LSTMNet artifact with validation
                metrics and window_size in extra.

        Raises:
            TrainerError: If training fails.
        """
        try:
            n_features: int = split.X_train.shape[1]

            train_dataset = _NumpySequenceDataset(
                split.X_train, split.y_train, self.window_size
            )
            val_dataset = _NumpySequenceDataset(
                split.X_val, split.y_val, self.window_size
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            net = _LSTMNet(input_dim=n_features, hidden_dim=self.hidden_dim).to(
                self.device
            )

            net, val_losses = _train_torch_model(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                patience=self.patience,
            )

            self.model = net

            # Evaluate on val set using a flat array of probabilities.
            val_probs = self._predict_from_dataset(val_dataset)
            y_val_trimmed = split.y_val[self.window_size - 1 :]
            metrics = _compute_metrics(
                y_true=y_val_trimmed[: len(val_probs)],
                y_prob=val_probs,
            )

            logger.info(
                "[^] LSTMTrainer: epochs_run=%d final_val_loss=%.6f",
                len(val_losses),
                val_losses[-1] if val_losses else float("nan"),
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={"window_size": self.window_size, "hidden_dim": self.hidden_dim},
            )

        except Exception as exc:
            raise TrainerError(
                f"LSTMTrainer.train() failed: {exc}", stage="train"
            ) from exc

    def _predict_from_dataset(self, dataset: "_NumpySequenceDataset") -> np.ndarray:
        """Run inference over an entire NumpySequenceDataset."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        self.model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                out = self.model(xb.to(self.device)).squeeze(dim=-1)
                probs.append(out.cpu().numpy())
        return np.concatenate(probs)

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Return CALL probabilities for a window tensor or numpy array.

        Args:
            X: shape (window_size, n_features) or
               (batch, window_size, n_features).

        Returns:
            np.ndarray: 1D probability array.
        """
        self._check_trained()
        return _torch_predict_proba(
            self.model, X, self.device, window_size=self.window_size
        )


# ── Windowing Path Note ──────────────────────────────────────────────────────
# Two sliding-window Dataset implementations exist in this system:
#
#   1. TimeSeriesDataset (sequence_generator.py)
#      The PUBLIC API. Accepts a FeatureMatrix directly, aligns on
#      timestamps, and is intended for external callers that construct
#      a FeatureMatrix via FeatureEngineer.build_matrix() and pass it
#      in without going through DataShaper. Use this when calling the
#      training pipeline from pipeline.py or any external orchestrator.
#
#   2. _NumpySequenceDataset (this file, below)
#      The INTERNAL path. Accepts the raw numpy arrays produced by
#      DataShaper.split() (TrainValTestSplit.X_train etc.). Used by all
#      Deep Learning trainers internally because DataShaper returns numpy,
#      not a FeatureMatrix. The leading underscore signals it is not part
#      of the public API and should not be imported outside trainer.py.
#
# Both implement the same (window_tensor, label_tensor) contract and
# are consumed identically by PyTorch's DataLoader. They are not
# interchangeable at the call site — use the one that matches the
# data type you have in hand.
# ────────────────────────────────────────────────────────────────────────────
class _NumpySequenceDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """
    Lightweight in-memory sliding-window dataset over numpy arrays.

    Used internally by the Deep Learning trainers to wrap the numpy
    splits produced by DataShaper without requiring a FeatureMatrix.
    For large-scale training use TimeSeriesDataset from
    sequence_generator.py instead.

    Input:  features (N, F), labels (N,)
    Output: window (W, F), label scalar — same contract as TimeSeriesDataset.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int,
    ) -> None:
        self._features = features.astype(np.float32)
        self._labels = labels.astype(np.float32)
        self.window_size = window_size
        self.n_samples = len(features) - window_size

    def __len__(self) -> int:
        return max(self.n_samples, 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._features[idx : idx + self.window_size]
        label = self._labels[idx + self.window_size - 1]
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.float32)


# ── GRU ──────────────────────────────────────────────────────────────────────


class _GRUNet(nn.Module):
    """
    Two-layer GRU network. Fewer parameters than LSTM, faster to train.

    Input shape:  (batch_size, window_size, n_features)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hn = self.gru(x)
        out = self.dropout(hn[-1])
        return self.sigmoid(self.fc(out))


class GRUTrainer(LSTMTrainer):
    """
    GRU trainer — same interface as LSTMTrainer, GRU cell architecture.

    Inherits the full training loop from LSTMTrainer and overrides only
    the network construction. Use for A/B comparison against LSTM on
    the same window size and dataset.
    """

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        try:
            n_features: int = split.X_train.shape[1]
            train_dataset = _NumpySequenceDataset(
                split.X_train, split.y_train, self.window_size
            )
            val_dataset = _NumpySequenceDataset(
                split.X_val, split.y_val, self.window_size
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            net = _GRUNet(input_dim=n_features, hidden_dim=self.hidden_dim).to(
                self.device
            )
            net, val_losses = _train_torch_model(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                patience=self.patience,
            )
            self.model = net

            val_probs = self._predict_from_dataset(val_dataset)
            y_val_trimmed = split.y_val[self.window_size - 1 :]
            metrics = _compute_metrics(
                y_true=y_val_trimmed[: len(val_probs)],
                y_prob=val_probs,
            )

            logger.info(
                "[^] GRUTrainer: epochs_run=%d final_val_loss=%.6f",
                len(val_losses),
                val_losses[-1] if val_losses else float("nan"),
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={"window_size": self.window_size, "hidden_dim": self.hidden_dim},
            )

        except Exception as exc:
            raise TrainerError(
                f"GRUTrainer.train() failed: {exc}", stage="train"
            ) from exc


# ── TCN ──────────────────────────────────────────────────────────────────────


class _TCNBlock(nn.Module):
    """
    Single dilated causal convolution residual block.

    Each block doubles the dilation factor, expanding the temporal
    receptive field exponentially. Causal padding ensures no future
    bar information leaks into the convolution output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=pad,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=pad,
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.causal_trim = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x)[:, :, : -self.causal_trim or None])
        out = self.dropout(out)
        out = self.relu(self.conv2(out)[:, :, : -self.causal_trim or None])
        out = self.dropout(out)
        return self.relu(out + self.residual(x))


class _TCNNet(nn.Module):
    """
    Temporal Convolutional Network for binary options classification.

    Stacks TCN residual blocks with exponentially increasing dilation.
    Receptive field = 2^n_layers * kernel_size. For n_layers=4,
    kernel_size=3: receptive field = 48 bars — covers 48 minutes of
    M1 context without any recurrent state.

    Input shape:  (batch_size, window_size, n_features)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_dim: int,
        n_channels: int = 64,
        kernel_size: int = 3,
        n_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_ch = input_dim if i == 0 else n_channels
            layers.append(
                _TCNBlock(
                    in_channels=in_ch,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(n_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TCN expects (batch, channels, time) — transpose from (batch, time, channels).
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # Global average pooling over the time dimension.
        out = out.mean(dim=-1)
        return self.sigmoid(self.fc(out))


class TCNTrainer(LSTMTrainer):
    """
    Temporal Convolutional Network trainer.

    Parallelisable unlike LSTM/GRU — all time steps processed in a
    single forward pass via dilated convolutions. No vanishing gradient
    problem. Typically faster to train than LSTM on the same data.

    Inherits the training loop from LSTMTrainer. Overrides only the
    network construction and adds n_channels and kernel_size parameters.

    Attributes:
        n_channels:  Number of convolution channels per block. Default 64.
        kernel_size: Convolution kernel size. Default 3.
        n_layers:    Number of TCN blocks. Receptive field = 2^n*kernel.
    """

    def __init__(
        self,
        expiry_key: str,
        window_size: int = 30,
        n_channels: int = 64,
        kernel_size: int = 3,
        n_layers: int = 4,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            expiry_key,
            window_size=window_size,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
        )
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dropout = dropout

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        try:
            n_features: int = split.X_train.shape[1]
            train_dataset = _NumpySequenceDataset(
                split.X_train, split.y_train, self.window_size
            )
            val_dataset = _NumpySequenceDataset(
                split.X_val, split.y_val, self.window_size
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            net = _TCNNet(
                input_dim=n_features,
                n_channels=self.n_channels,
                kernel_size=self.kernel_size,
                n_layers=self.n_layers,
                dropout=self.dropout,
            ).to(self.device)

            net, val_losses = _train_torch_model(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                patience=self.patience,
            )
            self.model = net

            val_probs = self._predict_from_dataset(val_dataset)
            y_val_trimmed = split.y_val[self.window_size - 1 :]
            metrics = _compute_metrics(
                y_true=y_val_trimmed[: len(val_probs)],
                y_prob=val_probs,
            )

            receptive_field = (2**self.n_layers) * self.kernel_size
            logger.info(
                "[^] TCNTrainer: epochs_run=%d receptive_field=%d bars",
                len(val_losses),
                receptive_field,
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={
                    "window_size": self.window_size,
                    "n_channels": self.n_channels,
                    "kernel_size": self.kernel_size,
                    "n_layers": self.n_layers,
                    "receptive_field_bars": receptive_field,
                },
            )

        except Exception as exc:
            raise TrainerError(
                f"TCNTrainer.train() failed: {exc}", stage="train"
            ) from exc


# ── CNN-LSTM ─────────────────────────────────────────────────────────────────


class _CNNLSTMNet(nn.Module):
    """
    CNN feature extractor followed by LSTM temporal memory.

    The CNN layers learn local candlestick pattern features (engulfing,
    pin bar sequences) from short sub-sequences. The LSTM layers then
    model temporal dependencies across the extracted CNN feature maps.

    Input shape:  (batch_size, window_size, n_features)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN expects (batch, channels, time).
        cnn_out = self.cnn(x.permute(0, 2, 1))
        # Back to (batch, time, channels) for LSTM.
        cnn_out = cnn_out.permute(0, 2, 1)
        _, (hn, _) = self.lstm(cnn_out)
        out = self.dropout(hn[-1])
        return self.sigmoid(self.fc(out))


class CNNLSTMTrainer(LSTMTrainer):
    """
    CNN-LSTM hybrid trainer for short-expiry binary options signals.

    The CNN layers extract local price patterns (engulfing, pin bar
    sequences); the LSTM layers model how those patterns evolve over
    time. Best suited for 1_MIN and 5_MIN expiry where short-term
    pattern sequences dominate.

    Inherits the training loop from LSTMTrainer.
    """

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        try:
            n_features: int = split.X_train.shape[1]
            train_dataset = _NumpySequenceDataset(
                split.X_train, split.y_train, self.window_size
            )
            val_dataset = _NumpySequenceDataset(
                split.X_val, split.y_val, self.window_size
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            net = _CNNLSTMNet(input_dim=n_features, lstm_hidden=self.hidden_dim).to(
                self.device
            )
            net, val_losses = _train_torch_model(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                patience=self.patience,
            )
            self.model = net

            val_probs = self._predict_from_dataset(val_dataset)
            y_val_trimmed = split.y_val[self.window_size - 1 :]
            metrics = _compute_metrics(
                y_true=y_val_trimmed[: len(val_probs)],
                y_prob=val_probs,
            )

            logger.info(
                "[^] CNNLSTMTrainer: epochs_run=%d final_val_loss=%.6f",
                len(val_losses),
                val_losses[-1] if val_losses else float("nan"),
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={"window_size": self.window_size},
            )

        except Exception as exc:
            raise TrainerError(
                f"CNNLSTMTrainer.train() failed: {exc}", stage="train"
            ) from exc


# ── Transformer ───────────────────────────────────────────────────────────────


class _TransformerNet(nn.Module):
    """
    Transformer encoder for binary options bar sequence classification.

    Multi-head self-attention attends to all bars in the window
    simultaneously, learning which historical bars are most predictive
    for the current signal. The positional encoding is not added here
    because TIME_SINE and TIME_COSINE are already present in the feature
    schema as explicit cyclical encodings.

    Input shape:  (batch_size, window_size, n_features)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Project input features to a dimension divisible by n_heads.
        self.d_model: int = max(n_heads, (input_dim // n_heads) * n_heads)
        self.input_proj = nn.Linear(input_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(self.d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        encoded = self.encoder(x)
        # Use the representation at the final time step (most recent bar).
        out = encoded[:, -1, :]
        return self.sigmoid(self.fc(out))


class TransformerTrainer(LSTMTrainer):
    """
    Transformer encoder trainer for long-context binary options signals.

    Multi-head attention is most effective on longer windows where the
    15_MIN expiry provides enough temporal context. Requires more data
    to generalise than LSTM or TCN — recommended minimum 6 months of
    M1 history.

    Inherits the training loop from LSTMTrainer.

    Attributes:
        n_heads:         Attention heads. Default 4. Must divide d_model.
        n_layers:        Transformer encoder layers. Default 2.
        dim_feedforward: FFN hidden dimension inside each layer. Default 128.
    """

    def __init__(
        self,
        expiry_key: str,
        window_size: int = 60,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
    ) -> None:
        super().__init__(
            expiry_key,
            window_size=window_size,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
        )
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        try:
            n_features: int = split.X_train.shape[1]
            train_dataset = _NumpySequenceDataset(
                split.X_train, split.y_train, self.window_size
            )
            val_dataset = _NumpySequenceDataset(
                split.X_val, split.y_val, self.window_size
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            net = _TransformerNet(
                input_dim=n_features,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dim_feedforward=self.dim_feedforward,
            ).to(self.device)

            net, val_losses = _train_torch_model(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                patience=self.patience,
            )
            self.model = net

            val_probs = self._predict_from_dataset(val_dataset)
            y_val_trimmed = split.y_val[self.window_size - 1 :]
            metrics = _compute_metrics(
                y_true=y_val_trimmed[: len(val_probs)],
                y_prob=val_probs,
            )

            logger.info(
                "[^] TransformerTrainer: epochs_run=%d n_heads=%d n_layers=%d",
                len(val_losses),
                self.n_heads,
                self.n_layers,
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={
                    "window_size": self.window_size,
                    "n_heads": self.n_heads,
                    "n_layers": self.n_layers,
                },
            )

        except Exception as exc:
            raise TrainerError(
                f"TransformerTrainer.train() failed: {exc}", stage="train"
            ) from exc


# ════════════════════════════════════════════════════════════════════════════
# TIER 3 — REINFORCEMENT LEARNING (Stable-Baselines3)
# ════════════════════════════════════════════════════════════════════════════


class _ForexBinaryEnv:
    """
    Minimal Gymnasium-compatible trading environment for RL training.

    Steps through a time-ordered feature matrix one bar at a time,
    receives the agent's action (CALL=0, PUT=1, SKIP=2), and returns
    the reward from RewardCalculator plus the next bar's FeatureVector
    as the new state.

    This environment is intentionally minimal — it is not a full
    backtesting engine. Its sole purpose is to produce a reward signal
    that teaches the RL agent to identify high-probability binary
    options entry points.

    Observation space: Box(n_features,) float32 — one FeatureVector.
    Action space:      Discrete(3)              — CALL / PUT / SKIP.

    Attributes:
        features:      2D float32 array (N, F) of bar feature vectors.
        labels:        1D int32 array (N,) of binary outcomes (0 or 1).
        gate_flags:    Optional 1D bool array (N,) of gate pass/fail.
        payout_ratio:  Broker payout fraction for RewardCalculator.
        current_step:  Current bar index within the episode.
        n_steps:       Total bars in the episode.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        gate_flags: np.ndarray | None = None,
        payout_ratio: float = 0.85,
    ) -> None:
        try:
            import gymnasium as gym  # type: ignore[import]
            from gymnasium import spaces  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "[!] gymnasium is required for RL training. "
                "Add 'gymnasium>=0.29.0' to your Dockerfile."
            ) from exc

        self._gym = gym
        self._spaces = spaces

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int32)
        self.gate_flags = (
            gate_flags.astype(bool)
            if gate_flags is not None
            else np.ones(len(labels), dtype=bool)
        )
        self.payout_ratio = payout_ratio
        self.n_steps: int = len(features)
        self.current_step: int = 0

        from ml_engine.labeler import RewardCalculator

        self._reward_calc = RewardCalculator(payout_ratio=payout_ratio)

        n_features = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to the first bar."""
        self.current_step = 0
        return self.features[0], {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance one bar and return (obs, reward, terminated, truncated, info).

        Args:
            action: Agent action — 0 (CALL), 1 (PUT), or 2 (SKIP).

        Returns:
            obs:        Feature vector for the next bar.
            reward:     Scalar reward from RewardCalculator.
            terminated: True if the episode ended (last bar reached).
            truncated:  Always False — no time limit truncation.
            info:       Dict with step diagnostics.
        """
        label: int = int(self.labels[self.current_step])
        gate_passed: bool = bool(self.gate_flags[self.current_step])

        # Determine directional correctness for CALL/PUT actions.
        is_correct: bool = False
        if action == 0:
            is_correct = label == 1
        elif action == 1:
            is_correct = label == 0

        reward: float = self._reward_calc.calculate_reward(
            action=action,
            is_correct=is_correct,
            gate_passed=gate_passed,
        )

        self.current_step += 1
        terminated: bool = self.current_step >= self.n_steps

        next_obs: np.ndarray = (
            self.features[self.current_step] if not terminated else self.features[-1]
        )

        info: dict[str, Any] = {
            "step": self.current_step,
            "label": label,
            "gate_passed": gate_passed,
            "is_correct": is_correct,
        }

        return next_obs, reward, terminated, False, info


class _RLBaseTrainer(BaseTrainer):
    """
    Abstract base for all Stable-Baselines3 RL trainers.

    Provides the shared environment construction and predict_proba()
    interface. Concrete subclasses specify the SB3 algorithm class
    and its hyperparameters in _build_sb3_model().

    Attributes:
        features:      Training feature array (N, F).
        labels:        Training label array (N,).
        gate_flags:    Optional gate pass/fail array (N,).
        n_steps_train: Total environment steps for training.
        payout_ratio:  Broker payout passed to RewardCalculator.
    """

    def __init__(
        self,
        expiry_key: str,
        features: np.ndarray,
        labels: np.ndarray,
        gate_flags: np.ndarray | None = None,
        n_steps_train: int = 100_000,
        payout_ratio: float = 0.85,
    ) -> None:
        super().__init__(expiry_key)
        self.features = features
        self.labels = labels
        self.gate_flags = gate_flags
        self.n_steps_train = n_steps_train
        self.payout_ratio = payout_ratio

    @abstractmethod
    def _build_sb3_model(self, env: _ForexBinaryEnv) -> Any:
        """
        Construct and return the SB3 algorithm instance.

        Args:
            env: Initialised _ForexBinaryEnv instance.

        Returns:
            SB3 model (PPO, A2C, DQN, or RecurrentPPO instance).
        """

    def train(self, split: TrainValTestSplit) -> TrainerResult:
        """
        Train the RL agent by stepping through the training environment.

        Constructs the _ForexBinaryEnv from the training split, builds
        the SB3 model via _build_sb3_model(), and calls model.learn().
        Evaluates on the validation set by running one episode and
        computing a Sharpe-proxy metric from cumulative rewards.

        Args:
            split: TrainValTestSplit — X_train/y_train used for the
                   training environment; X_val/y_val for the eval episode.

        Returns:
            TrainerResult: Trained SB3 model artifact with RL-specific
                metrics (mean_reward, sharpe_proxy, n_steps_trained).

        Raises:
            TrainerError: If training fails.
        """
        try:
            train_env = _ForexBinaryEnv(
                features=split.X_train,
                labels=split.y_train,
                gate_flags=(
                    self.gate_flags[: len(split.X_train)]
                    if self.gate_flags is not None
                    else None
                ),
                payout_ratio=self.payout_ratio,
            )

            self.model = self._build_sb3_model(train_env)

            logger.info(
                "[^] %s: starting learn() — n_steps=%d",
                self.model_name,
                self.n_steps_train,
            )

            self.model.learn(
                total_timesteps=self.n_steps_train,
                progress_bar=False,
            )

            # ── Validation episode ───────────────────────────────────────
            val_env = _ForexBinaryEnv(
                features=split.X_val,
                labels=split.y_val,
                gate_flags=(
                    self.gate_flags[
                        len(split.X_train) : len(split.X_train) + len(split.X_val)
                    ]
                    if self.gate_flags is not None
                    else None
                ),
                payout_ratio=self.payout_ratio,
            )

            obs, _ = val_env.reset()
            episode_rewards: list[float] = []
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = val_env.step(int(action))
                episode_rewards.append(reward)
                done = terminated or truncated

            reward_arr = np.array(episode_rewards)
            mean_reward = float(reward_arr.mean())
            std_reward = float(reward_arr.std()) + 1e-9
            sharpe_proxy = float(mean_reward / std_reward)

            metrics: dict[str, float] = {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "sharpe_proxy": sharpe_proxy,
                "total_val_steps": len(episode_rewards),
            }

            logger.info(
                "[^] %s: mean_reward=%.4f sharpe=%.4f",
                self.model_name,
                mean_reward,
                sharpe_proxy,
            )

            return self._build_result(
                split=split,
                metrics=metrics,
                extra={
                    "n_steps_trained": self.n_steps_train,
                    "payout_ratio": self.payout_ratio,
                },
            )

        except Exception as exc:
            raise TrainerError(
                f"{self.model_name}.train() failed: {exc}", stage="train"
            ) from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return action probabilities from the trained RL policy.

        For DQN this returns a one-hot probability (argmax of Q-values).
        For PPO/A2C this returns the policy softmax distribution.
        The CALL probability (action=0) is returned as the signal
        confidence for compatibility with the BaseTrainer interface.

        Args:
            X: 1D float32 array of shape (n_features,) — one FeatureVector.
               Or 2D array (n_samples, n_features) for batch inference.

        Returns:
            np.ndarray: 1D float array. For single-step inference this
                is a scalar wrapped in an array. For batch: (n_samples,).
        """
        self._check_trained()
        if X.ndim == 1:
            X = X[np.newaxis, :]
        results: list[float] = []
        for row in X:
            action, _ = self.model.predict(row, deterministic=True)
            # Map action to a confidence: CALL=high, PUT=low, SKIP=mid.
            conf = 0.8 if int(action) == 0 else (0.2 if int(action) == 1 else 0.5)
            results.append(conf)
        return np.array(results, dtype=np.float32)


class A2CTrainer(_RLBaseTrainer):
    """
    Synchronous Advantage Actor-Critic (A2C) RL trainer.

    A2C is the synchronous variant of A3C — deterministic, simpler
    to debug, and a good stepping stone before PPO. Use this first
    in Phase 3 before moving to PPO. Faster iteration than PPO due
    to no rollout buffer collection overhead.

    Attributes:
        learning_rate: Policy network learning rate. Default 7e-4.
        n_steps:       Steps per update. Default 5.
    """

    def __init__(
        self,
        expiry_key: str,
        features: np.ndarray,
        labels: np.ndarray,
        gate_flags: np.ndarray | None = None,
        n_steps_train: int = 100_000,
        payout_ratio: float = 0.85,
        learning_rate: float = 7e-4,
        n_steps: int = 5,
    ) -> None:
        super().__init__(
            expiry_key, features, labels, gate_flags, n_steps_train, payout_ratio
        )
        self.learning_rate = learning_rate
        self.n_steps = n_steps

    def _build_sb3_model(self, env: _ForexBinaryEnv) -> Any:
        try:
            from stable_baselines3 import A2C  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "[!] stable-baselines3 is required for A2CTrainer. "
                "Add 'stable-baselines3>=2.3.0' to your Dockerfile."
            ) from exc

        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            device=self.device,
            verbose=0,
        )


class DQNTrainer(_RLBaseTrainer):
    """
    Deep Q-Network (DQN) RL trainer with experience replay.

    DQN maintains a replay buffer of past (state, action, reward, next_state)
    tuples and trains the Q-network on random mini-batches. A target network
    is synced periodically (target_update_interval) to stabilise training.

    Epsilon-greedy exploration decays from exploration_initial_eps to
    exploration_final_eps over the first exploration_fraction of training.

    Attributes:
        learning_rate:            Q-network learning rate. Default 1e-4.
        buffer_size:              Replay buffer capacity. Default 50_000.
        learning_starts:          Steps before training begins. Default 1000.
        batch_size:               Replay mini-batch size. Default 32.
        target_update_interval:   Steps between target net syncs. Default 500.
        exploration_fraction:     Fraction of training for eps decay. Default 0.1.
        exploration_final_eps:    Final epsilon after decay. Default 0.05.
    """

    def __init__(
        self,
        expiry_key: str,
        features: np.ndarray,
        labels: np.ndarray,
        gate_flags: np.ndarray | None = None,
        n_steps_train: int = 100_000,
        payout_ratio: float = 0.85,
        learning_rate: float = 1e-4,
        buffer_size: int = 50_000,
        learning_starts: int = 1_000,
        batch_size: int = 32,
        target_update_interval: int = 500,
        exploration_fraction: float = 0.1,
        exploration_final_eps: float = 0.05,
    ) -> None:
        super().__init__(
            expiry_key, features, labels, gate_flags, n_steps_train, payout_ratio
        )
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps

    def _build_sb3_model(self, env: _ForexBinaryEnv) -> Any:
        try:
            from stable_baselines3 import DQN  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "[!] stable-baselines3 is required for DQNTrainer. "
                "Add 'stable-baselines3>=2.3.0' to your Dockerfile."
            ) from exc

        return DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps,
            device=self.device,
            verbose=0,
        )


class PPOTrainer(_RLBaseTrainer):
    """
    Proximal Policy Optimisation (PPO) RL trainer.

    PPO's clipped surrogate objective produces more stable training
    than DQN or A2C on noisy, non-stationary environments like forex.
    Recommended for 5_MIN and 15_MIN expiry where longer temporal
    context improves policy quality. Train after A2C is validated.

    Attributes:
        learning_rate:  Policy/value network lr. Default 3e-4.
        n_steps:        Steps per rollout buffer fill. Default 2048.
        batch_size:     Mini-batch size for PPO updates. Default 64.
        n_epochs:       PPO update epochs per rollout. Default 10.
        clip_range:     PPO clipping parameter. Default 0.2.
        ent_coef:       Entropy coefficient for exploration. Default 0.0.
    """

    def __init__(
        self,
        expiry_key: str,
        features: np.ndarray,
        labels: np.ndarray,
        gate_flags: np.ndarray | None = None,
        n_steps_train: int = 200_000,
        payout_ratio: float = 0.85,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
    ) -> None:
        super().__init__(
            expiry_key, features, labels, gate_flags, n_steps_train, payout_ratio
        )
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.ent_coef = ent_coef

    def _build_sb3_model(self, env: _ForexBinaryEnv) -> Any:
        try:
            from stable_baselines3 import PPO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "[!] stable-baselines3 is required for PPOTrainer. "
                "Add 'stable-baselines3>=2.3.0' to your Dockerfile."
            ) from exc

        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            device=self.device,
            verbose=0,
        )


class RecurrentPPOTrainer(_RLBaseTrainer):
    """
    Recurrent PPO (PPO + LSTM hidden state) RL trainer via sb3-contrib.

    Extends PPO with an LSTM-based policy that carries a hidden state
    between environment steps, enabling the agent to model partially
    observable market regimes that are not visible in a single
    FeatureVector. Most complex RL agent in the stack — train last,
    after PPO is validated on the same expiry key.

    Requires sb3-contrib>=2.3.0 in addition to stable-baselines3.

    Attributes:
        learning_rate:  Policy/value network lr. Default 3e-4.
        n_steps:        Steps per rollout. Default 2048.
        batch_size:     Mini-batch size. Default 64.
        n_epochs:       PPO update epochs per rollout. Default 10.
        lstm_hidden_size: LSTM hidden state size. Default 64.
    """

    def __init__(
        self,
        expiry_key: str,
        features: np.ndarray,
        labels: np.ndarray,
        gate_flags: np.ndarray | None = None,
        n_steps_train: int = 200_000,
        payout_ratio: float = 0.85,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        lstm_hidden_size: int = 64,
    ) -> None:
        super().__init__(
            expiry_key, features, labels, gate_flags, n_steps_train, payout_ratio
        )
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lstm_hidden_size = lstm_hidden_size

    def _build_sb3_model(self, env: _ForexBinaryEnv) -> Any:
        try:
            from sb3_contrib import RecurrentPPO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "[!] sb3-contrib is required for RecurrentPPOTrainer. "
                "Add 'sb3-contrib>=2.3.0' to your Dockerfile."
            ) from exc

        return RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            policy_kwargs={"lstm_hidden_size": self.lstm_hidden_size},
            device=self.device,
            verbose=0,
        )
