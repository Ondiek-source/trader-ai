"""
model.py — Multi-model binary classifier with walk-forward validation.

Model candidates:
    - LightGBM      (primary, fast, handles tabular well)
    - XGBoost       (ensemble member)
    - RandomForest  (sklearn, baseline)
    - LSTM          (PyTorch, sequence model on last 30 bars)
    - Transformer   (PyTorch, lightweight encoder on last 30 bars)

Per-indicator confidence: each indicator in features.py has its
historical accuracy tracked; the highest-confidence indicator+model combo wins.

Martingale awareness:
    - MartingaleTracker raises the confidence threshold after each loss.
    - ModelManager.predict() checks the dynamic threshold before returning a signal.

Walk-forward validation:
    - Chronological train/test splits (no data leakage).
    - Expanding window, 30-day steps.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    LGBM_AVAILABLE: bool = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE: bool = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE: bool = True
except ImportError:
    TORCH_AVAILABLE = False

from features import get_feature_columns, get_indicator_feature_groups

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN: int = 30  # lookback bars for LSTM / Transformer
EXPIRY_OPTIONS: list[int] = [60, 120, 300]
MIN_TRAIN_BARS: int = 5_000
MAX_RF_ROWS: int = 100_000


# ── Expiry Optimizer ──────────────────────────────────────────────────────────


class ExpiryOptimizer:
    """Determines the best expiry for each pair based on backtesting."""
    def __init__(self):
        self._results = {}
        self._best = {}

    def optimize(self, pair: str, feature_df: pd.DataFrame) -> dict[str, Any]:
        """
        Optimize the expiry for a given pair.

        Args:
            pair: Internal pair name.
            feature_df: DataFrame containing features and labels.

        Returns:
            Dictionary with the best expiry and all results.
        """
        results: dict[int, float] = {}
        feature_cols: list[str] = get_feature_columns()

        for expiry in EXPIRY_OPTIONS:
            try:
                available: list[str] = [
                    c for c in feature_cols if c in feature_df.columns
                ]
                df: pd.DataFrame = feature_df.dropna(subset=["label"] + available)
                if len(df) < 200:
                    continue

                X: np.ndarray = df[available].to_numpy(dtype=np.float64)
                y: np.ndarray = df["label"].to_numpy(dtype=np.float64)

                split: int = int(len(X) * 0.75)
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
                )
                rf.fit(X_tr, y_tr)
                proba: np.ndarray = rf.predict_proba(X_te)[:, 1]

                # Only count "confident" predictions to simulate real signal filter
                confident_mask: np.ndarray = (proba >= 0.60) | (proba <= 0.40)
                if confident_mask.sum() < 20:
                    win_rate: float = float(
                        accuracy_score(y_te, (proba >= 0.5).astype(int))
                    )
                else:
                    win_rate = float(
                        accuracy_score(
                            y_te[confident_mask],
                            (proba[confident_mask] >= 0.5).astype(int),
                        )
                    )

                results[expiry] = round(float(win_rate), 4)
                logger.info(
                    {
                        "event": "expiry_test",
                        "pair": pair,
                        "expiry_seconds": expiry,
                        "win_rate": results[expiry],
                    }
                )

            except Exception as exc:
                logger.warning(
                    {
                        "event": "expiry_test_failed",
                        "pair": pair,
                        "expiry": expiry,
                        "error": str(exc),
                    }
                )

        self._results[pair] = results
        if results:
            best: int = max(results, key=lambda e: results[e])
            self._best[pair] = best
            logger.info(
                {
                    "event": "expiry_optimized",
                    "pair": pair,
                    "best_expiry_seconds": best,
                    "win_rate": results[best],
                    "all_results": results,
                    "recommendation": (
                        f"Set your Quotex bot expiry to {best}s for {pair}"
                    ),
                }
            )
            return {"best_expiry": best, "results": results}
        return {"best_expiry": 60, "results": {}}

    def best_expiry(self, pair: str, default: int = 60) -> int:
        """
        Return the best expiry for *pair*, or *default* if not optimized.

        Args:
            pair: Internal pair name.
            default: Fallback expiry in seconds.

        Returns:
            Optimal expiry in seconds.
        """
        return self._best.get(pair, default)

    def summary(self) -> dict[str, Any]:
        """Return all optimization results."""
        return {
            "best_expiries": self._best,
            "all_results": self._results,
        }


# ── Market Regime Detector ────────────────────────────────────────────────────


class RegimeDetector:
    """
    Classifies current market microstructure into one of four regimes
    and selects which indicator group is most effective in that regime.

    Regimes:
        trending_up     — strong directional momentum, expanding range
        trending_down   — same but downward
        ranging         — oscillating, tight range, mean-reverting
        volatile        — wide spread, chaotic, elevated ATR

    Indicator affinity per regime (from backtesting research):
        trending_*  → EMA cross, MACD, Momentum are most predictive
        ranging     → RSI, Stochastic, Williams %R, Bollinger Bands
        volatile    → ATR-gated signals only; CCI, volume momentum
    """

    REGIME_INDICATOR_AFFINITY: dict[str, list[str]] = {
        "trending_up": ["ema_cross", "macd", "momentum", "volume_momentum"],
        "trending_down": ["ema_cross", "macd", "momentum", "volume_momentum"],
        "ranging": ["rsi", "stochastic", "williams_r", "bollinger"],
        "volatile": ["cci", "volume_momentum", "atr"],
    }

    def detect(self, feature_row: pd.Series) -> str:
        """
        Classify the current bar into a market regime.

        Args:
            feature_row: Series with feature values for one bar.

        Returns:
            One of ``"trending_up"``, ``"trending_down"``, ``"ranging"``,
            or ``"volatile"``.
        """
        try:
            atr_pct: float = float(feature_row.get("atr_pct") or 0.001)
            momentum: float = float(feature_row.get("momentum") or 0.0)
            bb_bw: float = float(feature_row.get("bb_bandwidth") or 0.01)
            ema_cross: float = float(feature_row.get("ema_cross") or 0.0)
            rsi: float = float(feature_row.get("rsi") or 50.0)

            if atr_pct > 0.003:
                return "volatile"

            if abs(ema_cross) > 0.5 and abs(momentum) > 0.0003:
                return "trending_up" if momentum > 0 else "trending_down"

            if bb_bw < 0.005 or (40 <= rsi <= 60):
                return "ranging"

            return "trending_up" if momentum >= 0 else "trending_down"
        except Exception:
            return "ranging"

    def preferred_indicators(self, regime: str) -> list[str]:
        """
        Return indicators most predictive in the given regime.

        Args:
            regime: Regime string from :meth:`detect`.

        Returns:
            List of indicator group names.
        """
        return self.REGIME_INDICATOR_AFFINITY.get(
            regime, list(get_indicator_feature_groups().keys())
        )


# ── Martingale Tracker ────────────────────────────────────────────────────────


class MartingaleTracker:
    """
    Tracks consecutive losses and progressively raises the confidence threshold.

    Threshold formula:
        ``current = base_threshold + (streak × step_size)``, capped at max_threshold.

    A win OR reaching ``max_streak`` resets the streak to 0.
    """

    def __init__(
        self,
        base_threshold: float = 0.65,
        max_streak: int = 4,
        step_size: float = 0.05,
        max_threshold: float = 0.90,
    ) -> None:
        self._base: float = base_threshold
        self._max_streak: int = max_streak
        self._step: float = step_size
        self._max: float = max_threshold
        self._streak: int = 0

    def record_result(self, win: bool) -> None:
        """
        Update the streak based on a trade result.

        Args:
            win: ``True`` if the trade was profitable.
        """
        if win:
            logger.info({"event": "martingale_win", "streak_reset_from": self._streak})
            self._streak = 0
        else:
            self._streak += 1
            if self._streak >= self._max_streak:
                logger.warning(
                    {
                        "event": "martingale_max_streak_reached",
                        "streak": self._streak,
                        "action": "resetting_streak",
                    }
                )
                self._streak = 0
            else:
                logger.info(
                    {
                        "event": "martingale_loss",
                        "streak": self._streak,
                        "new_threshold": self.current_threshold,
                    }
                )

    @property
    def current_threshold(self) -> float:
        """Current confidence threshold (increases with losses)."""
        return min(self._base + self._streak * self._step, self._max)

    @property
    def current_streak(self) -> int:
        """Current consecutive loss count."""
        return self._streak

    def reset(self) -> None:
        """Reset streak to zero."""
        self._streak = 0


# ── PyTorch models ────────────────────────────────────────────────────────────


if TORCH_AVAILABLE:

    class _LSTMModel(nn.Module):
        """Bidirectional-free LSTM binary classifier."""

        def __init__(
            self,
            n_features: int,
            hidden: int = 64,
            layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                n_features,
                hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out, _ = self.lstm(x)
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze(1)

    class _TransformerModel(nn.Module):
        """Lightweight Transformer encoder binary classifier."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            nhead: int = 4,
            layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.fc = nn.Linear(d_model, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.input_proj(x)
            x = self.encoder(x)
            return self.sigmoid(self.fc(x[:, -1, :])).squeeze(1)


def _train_pytorch_model(
    model: Any,
    X_seq: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
) -> Any:
    """
    Train a PyTorch sequence model with early stopping.

    Args:
        model: An ``nn.Module`` instance.
        X_seq: Shape ``(n, seq_len, n_features)``.
        y: Shape ``(n,)`` binary labels.
        epochs: Max training epochs.
        lr: Learning rate.
        patience: Early stopping patience (epochs without improvement).

    Returns:
        The trained model with best weights restored.
    """
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n: int = len(X_seq)
    split: int = int(n * 0.85)
    X_tr = torch.tensor(X_seq[:split], dtype=torch.float32).to(device)
    y_tr = torch.tensor(y[:split], dtype=torch.float32).to(device)
    X_val = torch.tensor(X_seq[split:], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[split:], dtype=torch.float32).to(device)

    best_val_loss: float = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve: int = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tr)
        loss = criterion(preds, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss: float = criterion(val_preds, y_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug(
                    {
                        "event": "early_stopping",
                        "epoch": epoch,
                        "val_loss": round(val_loss, 6),
                    }
                )
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Convert ``(n, n_features)`` to ``(n - seq_len + 1, seq_len, n_features)``.

    Args:
        X: Feature matrix of shape ``(n, n_features)``.
        seq_len: Lookback window length.

    Returns:
        3D array of sequences, or empty array if ``n < seq_len``.
    """
    n: int = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]))
    return np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])


def _predict_sequence_model(
    model: Any,
    feature_sequence_scaled: np.ndarray,
    n_features: int,
    model_name: str,
) -> float | None:
    """
    Run a pre-built sequence through a PyTorch sequence model.

    FIX D: Sequence Inference Correction (The "Tiling" Fix).
    ─────────────────────────────────────────────────────────
    OLD: np.tile(feature_row, (30, 1)) — fed 30 identical copies of "Now".
         LSTMs look for patterns of CHANGE.  Tiling makes the input look like
         a flat line, which never happens in training, forcing 0.5 output.
    NEW: Accepts a 3D numpy array of actual historical feature rows from
         main.py's rolling history buffer.  The LSTM / Transformer now sees
         the real 30-minute context leading up to the trade.

    Args:
        model: Trained LSTM or Transformer (``nn.Module``).
        feature_sequence_scaled: Pre-scaled array of shape
            ``(1, SEQ_LEN, n_features)`` containing the last SEQ_LEN bars
            of feature history.
        n_features: Number of input features.
        model_name: ``"lstm"`` or ``"transformer"`` (for logging).

    Returns:
        Probability of class 1, or ``None`` on error.
    """
    try:
        seq: np.ndarray = np.asarray(feature_sequence_scaled, dtype=np.float64)

        # Validate expected shape: (1, SEQ_LEN, n_features) or (SEQ_LEN, n_features)
        if seq.ndim == 2:
            if seq.shape != (SEQ_LEN, n_features):
                logger.debug(
                    {
                        "event": "sequence_shape_mismatch",
                        "model": model_name,
                        "expected": f"({SEQ_LEN}, {n_features})",
                        "got": seq.shape,
                    }
                )
                return None
            seq = seq[np.newaxis, ...]  # add batch dim → (1, SEQ_LEN, n_features)
        elif seq.ndim == 3:
            if seq.shape[1] != SEQ_LEN or seq.shape[2] != n_features:
                logger.debug(
                    {
                        "event": "sequence_shape_mismatch",
                        "model": model_name,
                        "expected": f"(?, {SEQ_LEN}, {n_features})",
                        "got": seq.shape,
                    }
                )
                return None
        else:
            logger.debug(
                {
                    "event": "sequence_ndim_error",
                    "model": model_name,
                    "ndim": seq.ndim,
                }
            )
            return None

        seq_tensor = torch.tensor(seq, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            prob: float = float(model(seq_tensor).item())
        return prob
    except Exception as exc:
        logger.debug(
            {
                "event": "sequence_predict_error",
                "model": model_name,
                "error": str(exc),
            }
        )
        return None


# ── Walk-forward validation ───────────────────────────────────────────────────


def walk_forward_evaluate(
    feature_df: pd.DataFrame,
    model_name: str,
    build_model_fn: Callable[[], Any],
    feature_cols: list[str],
    step_days: int = 30,
    min_train_rows: int = 500,
) -> dict[str, float]:
    """
    Walk-forward validation with expanding training window.

    Only works with sklearn-compatible models (those with ``fit`` and
    ``predict_proba``).

    Args:
        feature_df: Full feature DataFrame sorted by time.  Must contain
            a ``timestamp`` column and a ``label`` column.
        model_name: Name for logging.
        build_model_fn: Callable returning an unfitted model instance.
        feature_cols: Feature column names to use.
        step_days: Days per test fold (converted to bar count at 1 bar/min).
        min_train_rows: Minimum training rows before first fold.

    Returns:
        Mean metrics across folds (``accuracy``, ``precision``, ``recall``,
        ``f1``, ``brier``), or empty dict on failure.
    """
    df: pd.DataFrame = feature_df.dropna(subset=["label"] + feature_cols).copy()

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    total: int = len(df)
    initial_train: int = int(total * 0.7)
    # Convert step_days to bar count (1 bar per minute)
    step_bars: int = step_days * 24 * 60

    fold_metrics: list[dict[str, float]] = []
    train_end: int = initial_train

    while train_end + min_train_rows // 5 < total:
        train_df: pd.DataFrame = df.iloc[:train_end]
        test_end: int = min(train_end + step_bars, total)
        test_df: pd.DataFrame = df.iloc[train_end:test_end]

        if len(train_df) < min_train_rows or len(test_df) < 20:
            break

        X_train: np.ndarray = train_df[feature_cols].to_numpy(dtype=np.float64)
        y_train: np.ndarray = train_df["label"].to_numpy(dtype=np.float64)
        X_test: np.ndarray = test_df[feature_cols].to_numpy(dtype=np.float64)
        y_test: np.ndarray = test_df["label"].to_numpy(dtype=np.float64)

        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            mdl: Any = build_model_fn()
            mdl.fit(X_train, y_train)
            proba: np.ndarray = mdl.predict_proba(X_test)[:, 1]
            preds: np.ndarray = (proba >= 0.5).astype(int)

            fold_metrics.append(
                {
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "precision": float(precision_score(y_test, preds, zero_division=0)),
                    "recall": float(recall_score(y_test, preds, zero_division=0)),
                    "f1": float(f1_score(y_test, preds, zero_division=0)),
                    "brier": float(brier_score_loss(y_test, proba)),
                }
            )

        except Exception as exc:
            logger.warning(
                {
                    "event": "wf_fold_failed",
                    "model": model_name,
                    "error": str(exc),
                }
            )

        train_end = test_end

    if not fold_metrics:
        return {}

    means: dict[str, float] = {
        k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]
    }
    logger.info(
        {
            "event": "walk_forward_complete",
            "model": model_name,
            "folds": len(fold_metrics),
            "metrics": means,
        }
    )
    return means


# ── Per-indicator accuracy tracker ───────────────────────────────────────────


class IndicatorAccuracyTracker:
    """
    Tracks historical prediction accuracy per indicator.

    Uses a decaying moving average to weight recent results more heavily
    than stale ones.
    """

    def __init__(self, indicator_names: list[str]) -> None:
        # FIX E: Accuracy Tracker Calibration (The "Responsiveness" Fix).
        # OLD: Initial weight 0.55, Alpha 0.05.
        #   Too optimistic at start, too slow to change.  If a model failed
        #   5 times it took 40+ trades to reflect that in the ensemble weight.
        # NEW: Initial weight 0.50 (neutral — no bias), Alpha 0.10.
        #   Responds to live results in ~20 trades rather than ~100, which
        #   matters for a system that retrains every 500 results and trades
        #   ~10/day.
        self._accuracy: dict[str, float] = {name: 0.50 for name in indicator_names}
        self._alpha: float = 0.10

    def update(self, indicator: str, correct: bool) -> None:
        """
        Update the accuracy weight for *indicator*.

        Args:
            indicator: Indicator name (e.g. ``"rsi"``).
            correct: Whether the indicator's signal was correct.
        """
        current: float = self._accuracy.get(indicator, 0.50)
        self._accuracy[indicator] = (
            current * (1 - self._alpha) + float(correct) * self._alpha
        )

    def get_confidence_weight(self, indicator: str) -> float:
        """
        Return the current accuracy weight for *indicator*.

        Args:
            indicator: Indicator name.

        Returns:
            Accuracy weight in [0, 1], default 0.50.
        """
        return self._accuracy.get(indicator, 0.50)

    def all_weights(self) -> dict[str, float]:
        """Return all indicator accuracy weights."""
        return dict(self._accuracy)


# ── Main ModelManager ─────────────────────────────────────────────────────────


class ModelManager:
    """
    Trains, evaluates, and serves predictions from multiple ML models.

    Usage::

        mgr = ModelManager(config, martingale_tracker)
        mgr.train(pair, expiry_seconds, feature_df)
        signal = mgr.predict(pair, expiry_seconds, feature_row)
    """

    def __init__(self, config: Any, martingale_tracker: MartingaleTracker) -> None:
        self._config: Any = config
        self._mt: MartingaleTracker = martingale_tracker
        self._feature_cols: list[str] = get_feature_columns()
        self._indicator_groups: dict[str, list[str]] = get_indicator_feature_groups()
        self._max_sequences: int = getattr(config, "max_sequences", 20000)

        # Nested dict: {pair: {expiry: {model_name: fitted_model}}}
        self._models: dict[str, dict[int, dict[str, Any]]] = {}
        self._scalers: dict[str, dict[int, StandardScaler]] = {}
        self._metrics: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
        self._indicator_tracker: dict[str, IndicatorAccuracyTracker] = {}

        self._regime_detector: RegimeDetector = RegimeDetector()
        self._expiry_optimizer: ExpiryOptimizer = ExpiryOptimizer()

        # Retrain tracking
        self._result_counts: dict[str, int] = {}
        self._last_train_counts: dict[str, int] = {}
        self._retrain_threshold: int = 500

        # Prediction audit log (rolling window per pair)
        self._prediction_log: dict[str, list[dict[str, Any]]] = {}
        self._max_log_size: int = 1000

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        pair: str,
        expiry_seconds: int,
        feature_df: pd.DataFrame,
        feature_df_recent: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Train all model candidates on *feature_df*.  Run walk-forward
        validation for sklearn models.

        Args:
            pair: Internal pair name.
            expiry_seconds: Option expiry in seconds.
            feature_df: Full feature DataFrame for tree models.
            feature_df_recent: Optional recent data for sequence models
                (LSTM/Transformer).  If ``None``, sequence models are skipped.

        Returns:
            Dict of walk-forward metrics per model name.
        """
        df: pd.DataFrame = feature_df.dropna(
            subset=["label"] + self._feature_cols
        ).copy()

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if len(df) < 200:
            logger.warning(
                {
                    "event": "insufficient_data_for_training",
                    "pair": pair,
                    "rows": len(df),
                }
            )
            return {}

        X: np.ndarray = df[self._feature_cols].to_numpy(dtype=np.float64)
        y: np.ndarray = df["label"].to_numpy(dtype=np.float64)

        scaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(X)
        self._scalers.setdefault(pair, {})[expiry_seconds] = scaler

        trained: dict[str, Any] = {}
        wf_metrics: dict[str, dict[str, float]] = {}

        # ── LightGBM ──────────────────────────────────────────────────────
        if LGBM_AVAILABLE:
            try:
                lgbm_model = lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                )
                lgbm_model.fit(X_scaled, y)
                trained["lightgbm"] = lgbm_model
                wf_metrics["lightgbm"] = walk_forward_evaluate(
                    df,
                    "lightgbm",
                    lambda: lgb.LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        verbose=-1,
                        random_state=42,
                    ),
                    self._feature_cols,
                )
            except Exception as exc:
                logger.warning({"event": "lgbm_train_failed", "error": str(exc)})

        # ── XGBoost ───────────────────────────────────────────────────────
        if XGB_AVAILABLE:
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                )
                xgb_model.fit(X_scaled, y)
                trained["xgboost"] = xgb_model
                wf_metrics["xgboost"] = walk_forward_evaluate(
                    df,
                    "xgboost",
                    lambda: xgb.XGBClassifier(
                        n_estimators=200,
                        verbosity=0,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric="logloss",
                    ),
                    self._feature_cols,
                )
            except Exception as exc:
                logger.warning({"event": "xgb_train_failed", "error": str(exc)})

        # ── RandomForest (subset for memory) ──────────────────────────────
        if len(df) > MAX_RF_ROWS:
            df_rf: pd.DataFrame = df.tail(MAX_RF_ROWS).copy()
            logger.info(
                {
                    "event": "randomforest_subset",
                    "original": len(df),
                    "subset": len(df_rf),
                }
            )
        else:
            df_rf = df

        try:
            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=2
            )
            X_rf: np.ndarray = df_rf[self._feature_cols].to_numpy(dtype=np.float64)
            y_rf: np.ndarray = df_rf["label"].to_numpy(dtype=np.float64)

            # Use the SAME scaler fitted on full data — not a new one
            X_rf_scaled: np.ndarray = scaler.transform(X_rf)

            rf_model.fit(X_rf_scaled, y_rf)
            trained["randomforest"] = rf_model
            wf_metrics["randomforest"] = walk_forward_evaluate(
                df_rf,
                "randomforest",
                lambda: RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=42
                ),
                self._feature_cols,
            )
            logger.info(
                {
                    "event": "randomforest_trained",
                    "pair": pair,
                    "rows_used": len(df_rf),
                }
            )
        except Exception as exc:
            logger.warning({"event": "rf_train_failed", "error": str(exc)})

        # ── Sequence models (LSTM + Transformer) ──────────────────────────
        if (
            TORCH_AVAILABLE
            and feature_df_recent is not None
            and not feature_df_recent.empty
        ):
            seq_result: dict[str, Any] = self._train_sequence_models(
                pair=pair,
                feature_df_recent=feature_df_recent,
                scaler=scaler,
            )
            trained.update(seq_result.get("models", {}))
            wf_metrics.update(seq_result.get("metrics", {}))

        # ── Store ─────────────────────────────────────────────────────────
        self._models.setdefault(pair, {})[expiry_seconds] = trained
        self._metrics.setdefault(pair, {})[expiry_seconds] = wf_metrics

        # Initialise indicator tracker
        if pair not in self._indicator_tracker:
            self._indicator_tracker[pair] = IndicatorAccuracyTracker(
                list(self._indicator_groups.keys())
            )

        # Log summary
        for name, metrics in wf_metrics.items():
            logger.info(
                {
                    "event": "model_trained",
                    "pair": pair,
                    "model": name,
                    "expiry": expiry_seconds,
                    "metrics": metrics,
                }
            )

        return wf_metrics

    def _train_sequence_models(
        self,
        pair: str,
        feature_df_recent: pd.DataFrame,
        scaler: StandardScaler,
    ) -> dict[str, Any]:
        """Train LSTM and Transformer on recent data."""
        models: dict[str, Any] = {}
        metrics: dict[str, dict[str, float]] = {}

        df: pd.DataFrame = feature_df_recent.dropna(
            subset=["label"] + self._feature_cols
        ).copy()

        if len(df) < SEQ_LEN + 50:
            logger.warning(
                {
                    "event": "insufficient_data_for_sequence",
                    "pair": pair,
                    "rows": len(df),
                }
            )
            return {"models": {}, "metrics": {}}

        X: np.ndarray = df[self._feature_cols].to_numpy(dtype=np.float64)
        y: np.ndarray = df["label"].to_numpy(dtype=np.float64)

        # Limit to most recent max_sequences rows
        if len(X) > self._max_sequences:
            X = X[-self._max_sequences :]
            y = y[-self._max_sequences :]

        X_scaled: np.ndarray = scaler.transform(X)
        X_seq: np.ndarray = _make_sequences(X_scaled, SEQ_LEN)
        y_seq: np.ndarray = y[SEQ_LEN - 1 :]

        if len(X_seq) == 0:
            return {"models": {}, "metrics": {}}

        n_features: int = X_seq.shape[2]

        # LSTM
        try:
            lstm = _LSTMModel(n_features=n_features)
            lstm = _train_pytorch_model(lstm, X_seq, y_seq)
            models["lstm"] = lstm
            logger.info({"event": "lstm_trained", "pair": pair, "sequences": len(X_seq)})
        except Exception as exc:
            logger.warning({"event": "lstm_train_failed", "error": str(exc)})

        # Transformer
        try:
            transformer = _TransformerModel(n_features=n_features)
            transformer = _train_pytorch_model(transformer, X_seq, y_seq)
            models["transformer"] = transformer
            logger.info(
                {"event": "transformer_trained", "pair": pair, "sequences": len(X_seq)}
            )
        except Exception as exc:
            logger.warning({"event": "transformer_train_failed", "error": str(exc)})

        return {"models": models, "metrics": metrics}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        pair: str,
        expiry_seconds: int,
        feature_row: pd.Series,
        feature_history: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate a prediction from the best available model.

        FIX D + FIX F: Accepts an optional ``feature_history`` — the last
        SEQ_LEN bars of scaled feature data — so that LSTM / Transformer
        receive real sequential context instead of a tiled single row.

        Selection order:
            1. LightGBM   (if available and trained)
            2. XGBoost    (if available and trained)
            3. RandomForest (if available and trained)
            4. LSTM       (if available, trained, and feature_history provided)
            5. Transformer (if available, trained, and feature_history provided)

        The ensemble picks the model whose indicator group has the highest
        accuracy weight in the current market regime.

        Args:
            pair: Internal pair name.
            expiry_seconds: Option expiry in seconds.
            feature_row: Current bar's feature values.
            feature_history: Optional array of shape
                ``(SEQ_LEN, n_features)`` or ``(1, SEQ_LEN, n_features)``
                containing the last SEQ_LEN bars of feature history
                (pre-scaled).  Required for sequence model inference.

        Returns:
            Dict with ``direction``, ``confidence``, ``model``, ``regime``,
            ``indicator``, or ``None`` if below threshold or no model.
        """
        pair_models: dict[int, dict[str, Any]] | None = self._models.get(pair)
        if not pair_models:
            return None

        expiry_models: dict[str, Any] | None = pair_models.get(expiry_seconds)
        if not expiry_models:
            # Fall back to any available expiry
            expiry_models = next(iter(pair_models.values()), None)
            if not expiry_models:
                return None

        scaler: StandardScaler | None = self._scalers.get(pair, {}).get(expiry_seconds)
        if scaler is None:
            return None

        # Scale the single feature row
        feature_values: np.ndarray = feature_row[self._feature_cols].to_numpy(
            dtype=np.float64
        ).reshape(1, -1)
        X_scaled: np.ndarray = scaler.transform(feature_values)

        # Detect market regime
        regime: str = self._regime_detector.detect(feature_row)
        preferred: list[str] = self._regime_detector.preferred_indicators(regime)

        # Determine indicator tracker
        tracker: IndicatorAccuracyTracker | None = self._indicator_tracker.get(pair)
        if tracker is None:
            tracker = IndicatorAccuracyTracker(list(self._indicator_groups.keys()))
            self._indicator_tracker[pair] = tracker

        best_prediction: dict[str, Any] | None = None
        best_weighted_conf: float = -1.0

        for model_name, model in expiry_models.items():
            # Skip sequence models when no history is available
            if model_name in ("lstm", "transformer") and feature_history is None:
                continue

            try:
                # ── Inference ──────────────────────────────────────────────
                if model_name in ("lstm", "transformer"):
                    # FIX D: Pass actual 3D history instead of tiling one row.
                    # Scale the feature history using the same scaler.
                    if feature_history is not None:
                        n_rows: int = feature_history.shape[0]
                        hist_2d: np.ndarray = feature_history.reshape(
                            n_rows, -1
                        )
                        hist_scaled: np.ndarray = scaler.transform(hist_2d)
                        # Take the last SEQ_LEN rows and reshape to (1, SEQ_LEN, n_features)
                        seq_input: np.ndarray = hist_scaled[-SEQ_LEN:].reshape(
                            1, SEQ_LEN, -1
                        )
                        prob: float | None = _predict_sequence_model(
                            model,
                            seq_input,
                            len(self._feature_cols),
                            model_name,
                        )
                    else:
                        prob = None
                else:
                    # sklearn model — use predict_proba directly
                    prob = float(model.predict_proba(X_scaled)[0, 1])

                if prob is None:
                    continue

                # Direction and raw confidence
                direction: str = "UP" if prob >= 0.5 else "DOWN"
                raw_conf: float = max(prob, 1.0 - prob)

                # Find which indicator group this model is associated with
                indicator: str = "rsi"  # default
                for ind_name, ind_cols in self._indicator_groups.items():
                    if any(
                        col in self._feature_cols[:10] for col in ind_cols
                    ):  # heuristic
                        indicator = ind_name
                        break

                # Per-indicator confidence weighting
                ind_weight: float = tracker.get_confidence_weight(indicator)
                preferred_bonus: float = 1.1 if indicator in preferred else 0.9
                weighted_conf: float = raw_conf * ind_weight * preferred_bonus

                if weighted_conf > best_weighted_conf:
                    best_weighted_conf = weighted_conf
                    best_prediction = {
                        "direction": direction,
                        "confidence": round(weighted_conf, 4),
                        "raw_confidence": round(raw_conf, 4),
                        "model": model_name,
                        "regime": regime,
                        "indicator": indicator,
                        "indicator_weight": round(ind_weight, 4),
                        "pair": pair,
                        "expiry_seconds": expiry_seconds,
                    }

            except Exception as exc:
                logger.debug(
                    {
                        "event": "prediction_error",
                        "model": model_name,
                        "pair": pair,
                        "error": str(exc),
                    }
                )

        if best_prediction is None:
            return None

        # Martingale threshold check
        threshold: float = self._mt.current_threshold
        if best_prediction["confidence"] < threshold:
            logger.debug(
                {
                    "event": "below_threshold",
                    "pair": pair,
                    "confidence": best_prediction["confidence"],
                    "threshold": threshold,
                    "streak": self._mt.current_streak,
                }
            )
            return None

        # Audit log
        self._prediction_log.setdefault(pair, []).append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **best_prediction,
                "threshold": threshold,
            }
        )
        if len(self._prediction_log[pair]) > self._max_log_size:
            self._prediction_log[pair] = self._prediction_log[pair][
                -self._max_log_size :
            ]

        return best_prediction

    # ── Retrain tracking ──────────────────────────────────────────────────────

    def record_result(self, pair: str) -> None:
        """
        Record that a trade result was received for *pair*.

        Used by :meth:`should_retrain` to decide when to retrain.

        Args:
            pair: Internal pair name.
        """
        self._result_counts[pair] = self._result_counts.get(pair, 0) + 1

    def should_retrain(self, pair: str) -> bool:
        """
        Check if enough new results have accumulated to warrant retraining.

        Args:
            pair: Internal pair name.

        Returns:
            ``True`` if results since last training exceed
            :attr:`_retrain_threshold`.
        """
        current: int = self._result_counts.get(pair, 0)
        last: int = self._last_train_counts.get(pair, 0)
        if current - last >= self._retrain_threshold:
            self._last_train_counts[pair] = current
            return True
        return False

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str, storage: Any | None = None) -> None:
        """
        Save all models, scalers, and metadata to disk (and optionally Azure).

        Args:
            path: Local directory path for model files.
            storage: Optional :class:`StorageManager` for Azure blob upload.
        """
        os.makedirs(path, exist_ok=True)

        data: dict[str, Any] = {
            "models": {},
            "scalers": {},
            "metrics": self._metrics,
            "indicator_tracker": {
                pair: tracker.all_weights()
                for pair, tracker in self._indicator_tracker.items()
            },
            "result_counts": self._result_counts,
            "last_train_counts": self._last_train_counts,
        }

        for pair, expiry_dict in self._models.items():
            data["models"][pair] = {}
            for expiry, models in expiry_dict.items():
                data["models"][pair][expiry] = {}
                for name, model in models.items():
                    if name in ("lstm", "transformer") and TORCH_AVAILABLE:
                        # Save PyTorch state dict
                        model_path: str = os.path.join(
                            path, f"{pair}_{expiry}_{name}.pt"
                        )
                        torch.save(model.state_dict(), model_path)
                        data["models"][pair][expiry][name] = f"file:{model_path}"
                    else:
                        data["models"][pair][expiry][name] = model

        for pair, expiry_dict in self._scalers.items():
            data["scalers"][pair] = {}
            for expiry, scaler in expiry_dict.items():
                data["scalers"][pair][expiry] = scaler

        save_path: str = os.path.join(path, "model_manager.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info({"event": "models_saved", "path": path})

        # Upload to Azure if storage available
        if storage is not None:
            try:
                with open(save_path, "rb") as f:
                    storage.upload_blob("models/model_manager.pkl", f.read())
                # Upload PyTorch model files
                for pair, expiry_dict in self._models.items():
                    for expiry, models in expiry_dict.items():
                        for name in models:
                            if name in ("lstm", "transformer"):
                                model_path = os.path.join(
                                    path, f"{pair}_{expiry}_{name}.pt"
                                )
                                if os.path.exists(model_path):
                                    with open(model_path, "rb") as mf:
                                        storage.upload_blob(
                                            f"models/{pair}_{expiry}_{name}.pt",
                                            mf.read(),
                                        )
                logger.info({"event": "models_uploaded_to_azure"})
            except Exception as exc:
                logger.warning(
                    {"event": "model_upload_failed", "error": str(exc)}
                )

    def load(self, path: str, storage: Any | None = None) -> None:
        """
        Load models, scalers, and metadata from disk (or Azure as fallback).

        Args:
            path: Local directory path for model files.
            storage: Optional :class:`StorageManager` for Azure blob download.
        """
        save_path: str = os.path.join(path, "model_manager.pkl")

        # Try local first, then Azure
        if not os.path.exists(save_path) and storage is not None:
            try:
                blob_data: bytes = storage.download_blob("models/model_manager.pkl")
                os.makedirs(path, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(blob_data)
                logger.info({"event": "models_downloaded_from_azure"})
            except Exception as exc:
                logger.info(
                    {"event": "no_saved_models", "error": str(exc)}
                )
                return

        if not os.path.exists(save_path):
            logger.info({"event": "no_saved_models_found"})
            return

        try:
            with open(save_path, "rb") as f:
                data: dict[str, Any] = pickle.load(f)

            self._models = {}
            for pair, expiry_dict in data.get("models", {}).items():
                self._models[pair] = {}
                for expiry, models in expiry_dict.items():
                    self._models[pair][int(expiry)] = {}
                    for name, model in models.items():
                        if isinstance(model, str) and model.startswith("file:"):
                            # Load PyTorch state dict
                            model_file: str = model.replace("file:", "")
                            if os.path.exists(model_file) and TORCH_AVAILABLE:
                                if name == "lstm":
                                    loaded = _LSTMModel(n_features=len(self._feature_cols))
                                elif name == "transformer":
                                    loaded = _TransformerModel(
                                        n_features=len(self._feature_cols)
                                    )
                                else:
                                    continue
                                loaded.load_state_dict(
                                    torch.load(model_file, map_location="cpu")
                                )
                                loaded.eval()
                                self._models[pair][int(expiry)][name] = loaded
                        else:
                            self._models[pair][int(expiry)][name] = model

            self._scalers = {}
            for pair, expiry_dict in data.get("scalers", {}).items():
                self._scalers[pair] = {}
                for expiry, scaler in expiry_dict.items():
                    self._scalers[pair][int(expiry)] = scaler

            self._metrics = data.get("metrics", {})
            self._result_counts = data.get("result_counts", {})
            self._last_train_counts = data.get("last_train_counts", {})

            # Restore indicator tracker
            for pair, weights in data.get("indicator_tracker", {}).items():
                tracker = IndicatorAccuracyTracker(list(weights.keys()))
                tracker._accuracy = dict(weights)
                self._indicator_tracker[pair] = tracker

            total_models: int = sum(
                len(models)
                for expiry_dict in self._models.values()
                for models in expiry_dict.values()
            )
            logger.info(
                {
                    "event": "models_loaded",
                    "pairs": list(self._models.keys()),
                    "total_models": total_models,
                }
            )

        except Exception as exc:
            logger.error({"event": "model_load_failed", "error": str(exc)})

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_prediction_log(
        self, pair: str, last_n: int = 50
    ) -> list[dict[str, Any]]:
        """
        Return the last *n* predictions for *pair*.

        Args:
            pair: Internal pair name.
            last_n: Number of recent predictions to return.

        Returns:
            List of prediction dicts, newest last.
        """
        log: list[dict[str, Any]] = self._prediction_log.get(pair, [])
        return log[-last_n:]

    def get_metrics(
        self, pair: str, expiry_seconds: int
    ) -> dict[str, dict[str, float]]:
        """
        Return walk-forward metrics for all models of a pair/expiry.

        Args:
            pair: Internal pair name.
            expiry_seconds: Option expiry.

        Returns:
            Dict of ``{model_name: {metric: value}}``.
        """
        return self._metrics.get(pair, {}).get(expiry_seconds, {})

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of all trained models."""
        result: dict[str, Any] = {}
        for pair, expiry_dict in self._models.items():
            result[pair] = {}
            for expiry, models in expiry_dict.items():
                result[pair][f"{expiry}s"] = {
                    "models": list(models.keys()),
                    "metrics": self._metrics.get(pair, {}).get(expiry, {}),
                }
        return result
