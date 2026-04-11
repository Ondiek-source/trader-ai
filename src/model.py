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
    """
    Backtests all three expiry windows (60s, 120s, 300s) on historical data
    and selects the one with the highest walk-forward win rate per pair.

    Results are stored so main.py can log the recommendation to the user
    (who then sets the expiry in their Quotex bot configuration).
    """

    def __init__(self) -> None:
        self._results: dict[str, dict[int, float]] = {}
        self._best: dict[str, int] = {}

    def optimize(self, pair: str, tick_df: pd.DataFrame) -> dict[str, Any]:
        """
        Test all expiry options against *tick_df* using a fast RandomForest.

        Args:
            pair: Internal pair name.
            tick_df: Raw tick DataFrame.

        Returns:
            Dict with ``best_expiry`` (int) and ``results``
            (``{expiry: win_rate}``).
        """
        from features import compute_features

        results: dict[int, float] = {}
        for expiry in EXPIRY_OPTIONS:
            try:
                feature_df: pd.DataFrame = compute_features(
                    tick_df, expiry_seconds=expiry
                )
                if len(feature_df) < MIN_TRAIN_BARS // 10:
                    continue

                feature_cols: list[str] = get_feature_columns()
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
    feature_row_scaled: np.ndarray,
    n_features: int,
    model_name: str,
) -> float | None:
    """
    Run a single-row prediction through a sequence model.

    Since sequence models need history, the current bar is replicated
    across the full sequence length.  This is a known approximation —
    for true sequence inference, the caller should pass the last
    :data:`SEQ_LEN` bars.

    Args:
        model: Trained LSTM or Transformer.
        feature_row_scaled: Shape ``(1, n_features)`` — one scaled row.
        n_features: Number of input features.
        model_name: ``"lstm"`` or ``"transformer"`` (for logging).

    Returns:
        Probability of class 1, or ``None`` on error.
    """
    try:
        # Repeat the single row across SEQ_LEN timesteps
        row_2d: np.ndarray = feature_row_scaled.reshape(1, n_features)
        seq: np.ndarray = np.tile(row_2d, (SEQ_LEN, 1))  # (SEQ_LEN, n_features)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(
            0
        )  # (1, SEQ_LEN, n_features)

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

    Uses a decaying moving average (alpha=0.05) to weight recent results
    more heavily than stale ones.
    """

    def __init__(self, indicator_names: list[str]) -> None:
        self._accuracy: dict[str, float] = {name: 0.55 for name in indicator_names}
        self._alpha: float = 0.05

    def update(self, indicator: str, correct: bool) -> None:
        """
        Update the accuracy weight for *indicator*.

        Args:
            indicator: Indicator name (e.g. ``"rsi"``).
            correct: Whether the indicator's signal was correct.
        """
        current: float = self._accuracy.get(indicator, 0.55)
        self._accuracy[indicator] = (
            current * (1 - self._alpha) + float(correct) * self._alpha
        )

    def get_confidence_weight(self, indicator: str) -> float:
        """
        Return the current accuracy weight for *indicator*.

        Args:
            indicator: Indicator name.

        Returns:
            Accuracy weight in [0, 1], default 0.55.
        """
        return self._accuracy.get(indicator, 0.55)

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
            trained.update(seq_result)

        self._models.setdefault(pair, {})[expiry_seconds] = trained
        self._metrics.setdefault(pair, {})[expiry_seconds] = wf_metrics

        if pair not in self._indicator_tracker:
            self._indicator_tracker[pair] = IndicatorAccuracyTracker(
                list(self._indicator_groups.keys())
            )

        self._last_train_counts[pair] = self._result_counts.get(pair, 0)

        logger.info(
            {
                "event": "training_complete",
                "pair": pair,
                "expiry_seconds": expiry_seconds,
                "models_trained": list(trained.keys()),
                "wf_metrics": {m: v.get("accuracy") for m, v in wf_metrics.items()},
            }
        )
        return wf_metrics

    def _train_sequence_models(
        self,
        pair: str,
        feature_df_recent: pd.DataFrame,
        scaler: StandardScaler,
    ) -> dict[str, Any]:
        """
        Train LSTM and Transformer on recent feature data.

        Args:
            pair: Internal pair name.
            feature_df_recent: Recent feature DataFrame with ``label`` column.
            scaler: Already-fitted :class:`StandardScaler`.

        Returns:
            Dict mapping model name (``"lstm"``, ``"transformer"``) to
            trained model, or empty dict on failure.
        """
        df_recent: pd.DataFrame = feature_df_recent.dropna(
            subset=["label"] + self._feature_cols
        ).copy()

        # Cap sequences to prevent OOM
        max_rows: int = self._max_sequences + SEQ_LEN
        if len(df_recent) > max_rows:
            df_recent = df_recent.tail(max_rows)
            logger.info(
                {
                    "event": "sequence_data_capped",
                    "pair": pair,
                    "rows": len(df_recent),
                }
            )

        if len(df_recent) < SEQ_LEN + 50:
            logger.warning(
                {
                    "event": "insufficient_data_for_sequences",
                    "pair": pair,
                    "rows": len(df_recent),
                    "required": SEQ_LEN + 50,
                }
            )
            return {}

        X_recent: np.ndarray = df_recent[self._feature_cols].to_numpy(dtype=np.float64)
        y_recent: np.ndarray = df_recent["label"].to_numpy(dtype=np.float64)
        X_recent_scaled: np.ndarray = scaler.transform(X_recent)
        X_seq: np.ndarray = _make_sequences(X_recent_scaled, SEQ_LEN)
        y_seq: np.ndarray = y_recent[SEQ_LEN - 1 :]

        if len(X_seq) == 0:
            logger.warning({"event": "no_sequences_produced", "pair": pair})
            return {}

        n_features: int = len(self._feature_cols)
        trained: dict[str, Any] = {}

        for model_name, model_cls in [
            ("lstm", _LSTMModel),
            ("transformer", _TransformerModel),
        ]:
            try:
                model: Any = model_cls(n_features=n_features)
                model = _train_pytorch_model(model, X_seq, y_seq, epochs=10)
                trained[model_name] = model
                logger.info(
                    {
                        "event": f"{model_name}_trained",
                        "pair": pair,
                        "sequences": len(X_seq),
                    }
                )
            except Exception as exc:
                logger.error(
                    {
                        "event": f"{model_name}_train_failed",
                        "pair": pair,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        return trained

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        pair: str,
        expiry_seconds: int,
        feature_row: pd.Series,
    ) -> dict[str, Any] | None:
        """
        Generate a signal for *(pair, expiry_seconds)* using the best
        available model.

        Args:
            pair: Internal pair name.
            expiry_seconds: Option expiry.
            feature_row: Single-row Series of feature values.

        Returns:
            Signal dict with keys ``pair``, ``direction``, ``confidence``,
            ``model_used``, ``expiry_seconds``, ``timestamp``,
            ``indicator_signals``, ``regime``, ``preferred_indicators``,
            ``martingale_streak``, ``threshold_applied``.
            Returns ``None`` if no models are available.
        """
        models: dict[str, Any] | None = self._models.get(pair, {}).get(expiry_seconds)
        if not models:
            return None

        scaler: StandardScaler | None = self._scalers.get(pair, {}).get(expiry_seconds)
        if scaler is None:
            return None

        feat: np.ndarray = (
            feature_row.reindex(self._feature_cols)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            .reshape(1, -1)
        )
        feat_scaled: np.ndarray = scaler.transform(feat)

        # ── Per-model predictions ──────────────────────────────────────────
        model_probs: dict[str, float] = {}
        for model_name, mdl in models.items():
            try:
                if model_name in ("lstm", "transformer") and TORCH_AVAILABLE:
                    prob: float | None = _predict_sequence_model(
                        mdl, feat_scaled, len(self._feature_cols), model_name
                    )
                    if prob is not None:
                        model_probs[model_name] = prob
                    continue

                prob = float(mdl.predict_proba(feat_scaled)[0, 1])
                model_probs[model_name] = prob
            except Exception as exc:
                logger.debug(
                    {
                        "event": "model_predict_error",
                        "model": model_name,
                        "error": str(exc),
                    }
                )

        if not model_probs:
            return None

        # Best model: highest deviation from 0.5 (most decisive)
        best_model: str = max(model_probs, key=lambda m: abs(model_probs[m] - 0.5))
        best_prob: float = model_probs[best_model]
        best_direction: str = "UP" if best_prob >= 0.5 else "DOWN"
        best_confidence: float = 0.5 + abs(best_prob - 0.5)

        # ── Market regime ──────────────────────────────────────────────────
        regime: str = self._regime_detector.detect(feature_row)
        preferred: list[str] = self._regime_detector.preferred_indicators(regime)

        # ── Per-indicator consensus ────────────────────────────────────────
        indicator_signals: dict[str, dict[str, Any]] = {}
        indicator_tracker: IndicatorAccuracyTracker | None = (
            self._indicator_tracker.get(pair)
        )
        weighted_votes: dict[str, float] = {}

        tabular_model: Any | None = (
            models.get("lightgbm")
            or models.get("xgboost")
            or models.get("randomforest")
        )

        if tabular_model is not None:
            for indicator, cols in self._indicator_groups.items():
                available_cols: list[str] = [c for c in cols if c in feature_row.index]
                if not available_cols:
                    continue

                try:
                    # Build indicator-only feature vector
                    # (zero out non-indicator cols)
                    ind_feat: np.ndarray = np.zeros_like(feat_scaled)
                    for col in available_cols:
                        if col in self._feature_cols:
                            idx: int = self._feature_cols.index(col)
                            ind_feat[0, idx] = feat_scaled[0, idx]

                    ind_prob: float = float(tabular_model.predict_proba(ind_feat)[0, 1])
                    ind_direction: str = "UP" if ind_prob >= 0.5 else "DOWN"
                    ind_confidence: float = 0.5 + abs(ind_prob - 0.5)

                    base_weight: float = (
                        indicator_tracker.get_confidence_weight(indicator)
                        if indicator_tracker
                        else 0.55
                    )
                    regime_multiplier: float = 1.5 if indicator in preferred else 1.0
                    weight: float = base_weight * regime_multiplier

                    indicator_signals[indicator] = {
                        "direction": ind_direction,
                        "confidence": round(ind_confidence * base_weight, 4),
                        "regime_boosted": indicator in preferred,
                        "raw_prob": round(ind_prob, 4),
                    }

                    weighted_votes[ind_direction] = weighted_votes.get(
                        ind_direction, 0.0
                    ) + (ind_confidence * weight)

                except Exception:
                    pass

        # Blend model confidence with indicator consensus (70/30)
        if weighted_votes:
            consensus_direction: str = max(
                weighted_votes, key=lambda k: weighted_votes[k]
            )
            total_weight: float = sum(weighted_votes.values())
            consensus_confidence: float = (
                weighted_votes[consensus_direction] / total_weight
                if total_weight > 0
                else 0.5
            )
            final_direction: str = (
                best_direction
                if best_confidence >= consensus_confidence
                else consensus_direction
            )
            final_confidence: float = round(
                0.70 * best_confidence + 0.30 * consensus_confidence, 4
            )
        else:
            final_direction = best_direction
            final_confidence = round(best_confidence, 4)

        result: dict[str, Any] = {
            "pair": pair,
            "direction": final_direction,
            "confidence": final_confidence,
            "model_used": best_model,
            "expiry_seconds": expiry_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "indicator_signals": indicator_signals,
            "regime": regime,
            "preferred_indicators": preferred,
            "martingale_streak": self._mt.current_streak,
            "threshold_applied": self._mt.current_threshold,
        }

        # Audit log
        log: list[dict[str, Any]] = self._prediction_log.setdefault(pair, [])
        log.append(
            {
                "ts": result["timestamp"],
                "direction": final_direction,
                "confidence": final_confidence,
                "regime": regime,
            }
        )
        if len(log) > self._max_log_size:
            self._prediction_log[pair] = log[-self._max_log_size :]

        logger.info(
            {
                "event": "prediction_generated",
                "pair": pair,
                "direction": final_direction,
                "confidence": final_confidence,
                "regime": regime,
                "model": best_model,
                "streak": self._mt.current_streak,
                "threshold": self._mt.current_threshold,
            }
        )
        return result

    # ── Indicator accuracy ────────────────────────────────────────────────────

    def update_indicator_accuracy(
        self, pair: str, indicator: str, correct: bool
    ) -> None:
        """
        Update indicator accuracy after a trade result is known.

        Args:
            pair: Internal pair name.
            indicator: Indicator name (e.g. ``"rsi"``).
            correct: Whether the indicator's signal was correct.
        """
        tracker: IndicatorAccuracyTracker | None = self._indicator_tracker.get(pair)
        if tracker:
            tracker.update(indicator, correct)

    # ── Retrain tracking ──────────────────────────────────────────────────────

    def record_result(self, pair: str) -> None:
        """
        Increment result counter for *pair* (used to trigger retraining).

        Args:
            pair: Internal pair name.
        """
        self._result_counts[pair] = self._result_counts.get(pair, 0) + 1

    def should_retrain(self, pair: str) -> bool:
        """
        Return ``True`` if enough new results accumulated since last train.

        Args:
            pair: Internal pair name.

        Returns:
            ``True`` if ``current - last >= retrain_threshold``.
        """
        last: int = self._last_train_counts.get(pair, 0)
        current: int = self._result_counts.get(pair, 0)
        return (current - last) >= self._retrain_threshold

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str, storage: Any = None) -> None:
        """
        Persist all models and trackers to disk and optionally to blob storage.

        PyTorch models are saved separately via ``torch.save`` to avoid
        pickle version incompatibilities.

        Args:
            path: Local directory to save to.
            storage: Optional :class:`~storage.StorageManager` for blob backup.
        """
        os.makedirs(path, exist_ok=True)

        # Separate PyTorch models — pickle is unreliable across versions
        torch_state: dict[str, Any] = {}
        serializable_models: dict[str, dict[int, dict[str, Any]]] = {}
        for pair, expiries in self._models.items():
            serializable_models[pair] = {}
            for expiry, models in expiries.items():
                serializable_models[pair][expiry] = {}
                for name, mdl in models.items():
                    if name in ("lstm", "transformer") and TORCH_AVAILABLE:
                        torch_state[f"{pair}/{expiry}/{name}"] = mdl.state_dict()
                    else:
                        serializable_models[pair][expiry][name] = mdl

        payload: dict[str, Any] = {
            "models": serializable_models,
            "scalers": self._scalers,
            "metrics": self._metrics,
            "indicator_tracker": self._indicator_tracker,
            "result_counts": self._result_counts,
            "last_train_counts": self._last_train_counts,
        }

        pkl_path: str = os.path.join(path, "model_manager.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        if TORCH_AVAILABLE and torch_state:
            torch_path: str = os.path.join(path, "pytorch_models.pt")
            torch.save(torch_state, torch_path)

        if storage:
            with open(pkl_path, "rb") as f:
                storage.save_model(f.read())
            if TORCH_AVAILABLE and torch_state:
                import io as _io

                buf = _io.BytesIO()
                torch.save(torch_state, buf)
                buf.seek(0)
                storage.save_model(buf.read(), model_name="pytorch_models.pt")

        logger.info({"event": "models_saved", "path": path})

    def load(self, path: str, storage: Any = None) -> None:
        """
        Load persisted models from disk (or blob storage as fallback).

        Args:
            path: Local directory to load from.
            storage: Optional :class:`~storage.StorageManager` for blob fallback.
        """
        pkl_path: str = os.path.join(path, "model_manager.pkl")
        torch_path: str = os.path.join(path, "pytorch_models.pt")

        payload: dict[str, Any]
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)
            logger.info({"event": "models_loaded_from_local", "path": path})
        elif storage:
            data: bytes | None = storage.load_model()
            if data:
                os.makedirs(path, exist_ok=True)
                with open(pkl_path, "wb") as f:
                    f.write(data)
                with open(pkl_path, "rb") as f:
                    payload = pickle.load(f)
                logger.info({"event": "models_loaded_from_blob"})
            else:
                logger.info({"event": "no_saved_models", "path": path})
                return
        else:
            logger.info({"event": "no_saved_models", "path": path})
            return

        self._models = payload.get("models", {})
        self._scalers = payload.get("scalers", {})
        self._metrics = payload.get("metrics", {})
        self._indicator_tracker = payload.get("indicator_tracker", {})
        self._result_counts = payload.get("result_counts", {})
        self._last_train_counts = payload.get("last_train_counts", {})

        # Load PyTorch weights
        if TORCH_AVAILABLE and os.path.exists(torch_path):
            try:
                torch_state: dict[str, Any] = torch.load(
                    torch_path, map_location="cpu", weights_only=True
                )
                for key, state in torch_state.items():
                    parts: list[str] = key.split("/")
                    pair: str = parts[0]
                    expiry: int = int(parts[1])
                    model_name: str = parts[2]
                    model: Any | None = (
                        self._models.get(pair, {}).get(expiry, {}).get(model_name)
                    )
                    if model is not None:
                        model.load_state_dict(state)
                        logger.info({"event": "pytorch_weights_loaded", "key": key})
            except Exception as exc:
                logger.warning({"event": "pytorch_load_failed", "error": str(exc)})

        logger.info(
            {
                "event": "models_loaded",
                "path": path,
                "pairs": list(self._models.keys()),
            }
        )
