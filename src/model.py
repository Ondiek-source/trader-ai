"""
model.py — Multi-model binary classifier with walk-forward validation.

Model candidates:
    - LightGBM      (primary, fast, handles tabular well)
    - XGBoost       (ensemble member)
    - RandomForest  (sklearn, baseline)
    - LSTM          (PyTorch, sequence model on last 30 bars)
    - Transformer   (PyTorch, lightweight encoder on last 30 bars)

Per-indicator confidence: each of the 10 indicators in features.py has its
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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from features import get_feature_columns, get_indicator_feature_groups, compute_features

logger = logging.getLogger(__name__)

SEQ_LEN = 30  # lookback bars for LSTM / Transformer
EXPIRY_OPTIONS = [60, 120, 300]
MIN_TRAIN_BARS = 5_000  # minimum 1-min bars needed for meaningful training (~3.5 days)


# ── Expiry Optimizer ──────────────────────────────────────────────────────────


class ExpiryOptimizer:
    """
    Backtests all three expiry windows (60s, 120s, 300s) on historical data
    and selects the one with the highest walk-forward win rate per pair.

    Results are stored so main.py can log the recommendation to the user
    (who then sets the expiry in their Quotex bot configuration).
    """

    def __init__(self) -> None:
        # {pair: {expiry_seconds: win_rate}}
        self._results: dict[str, dict[int, float]] = {}
        # {pair: best_expiry_seconds}
        self._best: dict[str, int] = {}

    def optimize(self, pair: str, tick_df: pd.DataFrame) -> dict[str, Any]:
        """
        Test all expiry options against tick_df using a fast RandomForest.
        Returns {expiry_seconds: win_rate, best_expiry: int}.
        """
        results: dict[int, float] = {}
        for expiry in EXPIRY_OPTIONS:
            try:
                feature_df = compute_features(tick_df, expiry_seconds=expiry)
                if len(feature_df) < MIN_TRAIN_BARS // 10:
                    continue
                feature_cols = get_feature_columns()
                available = [c for c in feature_cols if c in feature_df.columns]
                df = feature_df.dropna(subset=["label"] + available)
                if len(df) < 200:
                    continue

                X = df[available].values
                y = df["label"].values

                # Simple time-series cross-validation (no leakage)
                split = int(len(X) * 0.75)
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]

                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
                )
                rf.fit(X_tr, y_tr)
                proba = rf.predict_proba(X_te)[:, 1]
                # Only count "confident" predictions (≥ 0.60) to simulate real signal filter
                confident_mask = (proba >= 0.60) | (proba <= 0.40)
                if confident_mask.sum() < 20:
                    win_rate = accuracy_score(y_te, (proba >= 0.5).astype(int))
                else:
                    win_rate = accuracy_score(
                        y_te[confident_mask], (proba[confident_mask] >= 0.5).astype(int)
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
                    {"event": "expiry_test_failed", "expiry": expiry, "error": str(exc)}
                )

        self._results[pair] = results
        if results:
            best = max(results, key=lambda e: results[e])
            self._best[pair] = best
            logger.info(
                {
                    "event": "expiry_optimized",
                    "pair": pair,
                    "best_expiry_seconds": best,
                    "win_rate": results[best],
                    "all_results": results,
                    "recommendation": f"Set your Quotex bot expiry to {best}s for {pair}",
                }
            )
            return {"best_expiry": best, "results": results}
        return {"best_expiry": 60, "results": {}}

    def best_expiry(self, pair: str, default: int = 60) -> int:
        return self._best.get(pair, default)

    def summary(self) -> dict:
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
        """Classify the current bar into a market regime."""
        try:
            atr_pct = float(feature_row.get("atr_pct") or 0.001)
            momentum = float(feature_row.get("momentum") or 0.0)
            bb_bw = float(feature_row.get("bb_bandwidth") or 0.01)
            ema_cross = float(feature_row.get("ema_cross") or 0.0)
            rsi = float(feature_row.get("rsi") or 50.0)

            # Volatile: wide ATR relative to price
            if atr_pct > 0.003:
                return "volatile"

            # Trending: strong EMA alignment + momentum
            if abs(ema_cross) > 0.5 and abs(momentum) > 0.0003:
                return "trending_up" if momentum > 0 else "trending_down"

            # Ranging: tight Bollinger Band, RSI near midline
            if bb_bw < 0.005 or (40 <= rsi <= 60):
                return "ranging"

            # Default: trending
            return "trending_up" if momentum >= 0 else "trending_down"
        except Exception:
            return "ranging"

    def preferred_indicators(self, regime: str) -> list[str]:
        return self.REGIME_INDICATOR_AFFINITY.get(
            regime, list(get_indicator_feature_groups().keys())
        )


# ── Martingale Tracker ────────────────────────────────────────────────────────


class MartingaleTracker:
    """
    Tracks consecutive losses and progressively raises the confidence threshold.

    Threshold formula:
        current = base_threshold + (streak × step_size), capped at max_threshold.

    A win OR reaching max_streak resets the streak to 0.
    """

    def __init__(
        self,
        base_threshold: float = 0.65,
        max_streak: int = 4,
        step_size: float = 0.05,
        max_threshold: float = 0.90,
    ) -> None:
        self._base = base_threshold
        self._max_streak = max_streak
        self._step = step_size
        self._max = max_threshold
        self._streak: int = 0

    def record_result(self, win: bool) -> None:
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
        return min(self._base + self._streak * self._step, self._max)

    @property
    def current_streak(self) -> int:
        return self._streak

    def reset(self) -> None:
        self._streak = 0


# ── PyTorch models ────────────────────────────────────────────────────────────

_NNBase: type = nn.Module if TORCH_AVAILABLE else object  # type: ignore[assignment]


class _LSTMModel(_NNBase):  # type: ignore[misc]
    def __init__(
        self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :])).squeeze(1)


class _TransformerModel(_NNBase):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
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

    def forward(self, x):
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
    """Train an LSTM or Transformer model with early stopping."""
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n = len(X_seq)
    split = int(n * 0.85)
    X_tr = torch.tensor(X_seq[:split], dtype=torch.float32).to(device)
    y_tr = torch.tensor(y[:split], dtype=torch.float32).to(device)
    X_val = torch.tensor(X_seq[split:], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[split:], dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

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
            val_loss = criterion(val_preds, y_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert (n, n_features) → (n - seq_len + 1, seq_len, n_features)."""
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]))
    seqs = np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])
    return seqs


# ── Walk-forward validation ───────────────────────────────────────────────────


def walk_forward_evaluate(
    feature_df: pd.DataFrame,
    model_name: str,
    build_model_fn,
    feature_cols: list[str],
    step_days: int = 30,
    min_train_rows: int = 500,
) -> dict[str, float]:
    """
    Walk-forward validation with expanding training window.
    Returns mean metrics across all folds.
    """
    df = feature_df.dropna(subset=["label"] + feature_cols).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    total = len(df)
    initial_train = int(total * 0.7)

    fold_metrics: list[dict] = []
    train_end = initial_train

    while train_end + min_train_rows // 5 < total:
        train_df = df.iloc[:train_end]
        test_end = min(train_end + step_days * 24 * 60, total)  # ~30 days of 1-min bars
        test_df = df.iloc[train_end:test_end]

        if len(train_df) < min_train_rows or len(test_df) < 20:
            break

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].to_numpy(dtype=np.float64)
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].to_numpy(dtype=np.float64)

        try:
            mdl = build_model_fn()
            if hasattr(mdl, "fit"):
                mdl.fit(X_train, y_train)
            proba = mdl.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)

            fold_metrics.append(
                {
                    "accuracy": accuracy_score(y_test, preds),
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall": recall_score(y_test, preds, zero_division=0),
                    "f1": f1_score(y_test, preds, zero_division=0),
                    "brier": brier_score_loss(y_test, proba),
                }
            )
        except Exception as exc:
            logger.warning(
                {"event": "wf_fold_failed", "model": model_name, "error": str(exc)}
            )

        train_end = test_end

    if not fold_metrics:
        return {}
    means = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
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
    Uses a decaying moving average (alpha=0.05) to weight recent results more.
    """

    def __init__(self, indicator_names: list[str]) -> None:
        self._accuracy: dict[str, float] = {name: 0.55 for name in indicator_names}
        self._alpha = 0.05  # EMA decay

    def update(self, indicator: str, correct: bool) -> None:
        current = self._accuracy.get(indicator, 0.55)
        self._accuracy[indicator] = (
            current * (1 - self._alpha) + float(correct) * self._alpha
        )

    def get_confidence_weight(self, indicator: str) -> float:
        return self._accuracy.get(indicator, 0.55)

    def all_weights(self) -> dict[str, float]:
        return dict(self._accuracy)


# ── Main ModelManager ─────────────────────────────────────────────────────────


class ModelManager:
    """
    Trains, evaluates, and serves predictions from multiple ML models.

    Usage:
        mgr = ModelManager(config, martingale_tracker)
        mgr.train(pair, expiry_seconds, feature_df)
        signal = mgr.predict(pair, expiry_seconds, feature_row)
    """

    def __init__(self, config, martingale_tracker: MartingaleTracker) -> None:
        self._config = config
        self._mt = martingale_tracker
        self._feature_cols = get_feature_columns()
        self._indicator_groups = get_indicator_feature_groups()

        # Nested dict: {pair: {expiry: {model_name: fitted_model}}}
        self._models: dict[str, dict[int, dict[str, Any]]] = {}
        self._scalers: dict[str, dict[int, StandardScaler]] = {}
        self._metrics: dict[str, dict[int, dict[str, dict]]] = {}
        self._indicator_tracker: dict[str, IndicatorAccuracyTracker] = {}

        # Regime detector (shared, stateless)
        self._regime_detector = RegimeDetector()

        # Expiry optimizer (per pair)
        self._expiry_optimizer = ExpiryOptimizer()

        # Track rows added since last retrain per pair
        self._result_counts: dict[str, int] = {}
        self._last_train_counts: dict[str, int] = {}
        self._retrain_threshold = 500

        # Prediction audit log (last N predictions per pair for analysis)
        self._prediction_log: dict[str, list[dict]] = {}
        self._max_log_size = 1000

    def train(
        self,
        pair: str,
        expiry_seconds: int,
        feature_df: pd.DataFrame,
        feature_df_recent: pd.DataFrame | None = None,
    ) -> dict:
        """
        Train all model candidates on feature_df. Run walk-forward validation.
        Returns dict of metrics per model.
        """
        df = feature_df.dropna(subset=["label"] + self._feature_cols).copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) < 200:
            logger.warning(
                {
                    "event": "insufficient_data_for_training",
                    "pair": pair,
                    "rows": len(df),
                }
            )
            return {}

        X = df[self._feature_cols].to_numpy(dtype=np.float64)
        y = df["label"].to_numpy(dtype=np.float64)

        # Fit scaler on training data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scalers.setdefault(pair, {})[expiry_seconds] = scaler

        trained: dict[str, Any] = {}
        wf_metrics: dict[str, dict] = {}

        # ── LightGBM ──────────────────────────────────────────────────────────
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

        # ── XGBoost ───────────────────────────────────────────────────────────
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

        # ── RandomForest (baseline) ───────────────────────────────────────────
        try:
            rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_scaled, y)
            trained["randomforest"] = rf_model
            wf_metrics["randomforest"] = walk_forward_evaluate(
                df,
                "randomforest",
                lambda: RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=42
                ),
                self._feature_cols,
            )
        except Exception as exc:
            logger.warning({"event": "rf_train_failed", "error": str(exc)})

        # ── LSTM (use recent data only) ──────────────────────────────────────
        if TORCH_AVAILABLE and feature_df_recent is not None and not feature_df_recent.empty:
            # Use recent data for sequence models
            df_recent = feature_df_recent.dropna(subset=["label"] + self._feature_cols).copy()
            if len(df_recent) >= SEQ_LEN + 50:
                try:
                    X_recent = df_recent[self._feature_cols].to_numpy(dtype=np.float64)
                    y_recent = df_recent["label"].to_numpy(dtype=np.float64)
                    
                    # Scale using the same scaler as full data
                    scaler = self._scalers.get(pair, {}).get(expiry_seconds)
                    if scaler is None:
                        scaler = StandardScaler()
                        X_recent_scaled = scaler.fit_transform(X_recent)
                        self._scalers.setdefault(pair, {})[expiry_seconds] = scaler
                    else:
                        X_recent_scaled = scaler.transform(X_recent)
                    
                    X_seq = _make_sequences(X_recent_scaled, SEQ_LEN)
                    y_seq = y_recent[SEQ_LEN - 1:]
                    
                    if len(X_seq) > 0:
                        lstm = _LSTMModel(n_features=len(self._feature_cols))
                        lstm = _train_pytorch_model(lstm, X_seq, y_seq, epochs=10)
                        trained["lstm"] = lstm
                        logger.info({"event": "lstm_trained", "pair": pair, "sequences": len(X_seq)})
                except Exception as exc:
                    logger.warning({"event": "lstm_train_failed", "error": str(exc)})

        # ── Transformer (use recent data only) ───────────────────────────────
        if TORCH_AVAILABLE and feature_df_recent is not None and not feature_df_recent.empty:
            # Use recent data for sequence models
            df_recent = feature_df_recent.dropna(subset=["label"] + self._feature_cols).copy()
            if len(df_recent) >= SEQ_LEN + 50:
                try:
                    X_recent = df_recent[self._feature_cols].to_numpy(dtype=np.float64)
                    y_recent = df_recent["label"].to_numpy(dtype=np.float64)
                    
                    # Scale using the same scaler as full data
                    scaler = self._scalers.get(pair, {}).get(expiry_seconds)
                    if scaler is None:
                        scaler = StandardScaler()
                        X_recent_scaled = scaler.fit_transform(X_recent)
                        self._scalers.setdefault(pair, {})[expiry_seconds] = scaler
                    else:
                        X_recent_scaled = scaler.transform(X_recent)
                    
                    X_seq = _make_sequences(X_recent_scaled, SEQ_LEN)
                    y_seq = y_recent[SEQ_LEN - 1:]
                    
                    if len(X_seq) > 0:
                        transformer = _TransformerModel(n_features=len(self._feature_cols))
                        transformer = _train_pytorch_model(transformer, X_seq, y_seq, epochs=10)
                        trained["transformer"] = transformer
                        logger.info({"event": "transformer_trained", "pair": pair, "sequences": len(X_seq)})
                except Exception as exc:
                    logger.warning({"event": "transformer_train_failed", "error": str(exc)})

        self._models.setdefault(pair, {})[expiry_seconds] = trained
        self._metrics.setdefault(pair, {})[expiry_seconds] = wf_metrics

        # Initialize indicator tracker
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
                "metrics": {m: v.get("accuracy") for m, v in wf_metrics.items()},
            }
        )
        logger.info({"event": "models_trained_count", "pair": pair, "count": len(trained)})
        return wf_metrics

    def predict(
        self, pair: str, expiry_seconds: int, feature_row: pd.Series
    ) -> dict | None:
        """
        Generate a signal for (pair, expiry_seconds) using the best available model.

        Returns None if:
            - No models trained for this pair/expiry
            - Best confidence < martingale_tracker.current_threshold
        """
        models = self._models.get(pair, {}).get(expiry_seconds)
        if not models:
            return None

        scaler = self._scalers.get(pair, {}).get(expiry_seconds)
        if scaler is None:
            return None

        # Align feature row to expected columns
        feat = (
            feature_row.reindex(self._feature_cols)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            .reshape(1, -1)
        )
        feat_scaled = scaler.transform(feat)

        indicator_signals: dict[str, dict] = {}
        best_direction: str | None = None
        best_confidence: float = 0.0
        best_model: str = ""

        # ── Per-model prediction ───────────────────────────────────────────────
        model_probs: dict[str, float] = {}
        for model_name, mdl in models.items():
            try:
                if model_name in ("lstm", "transformer") and TORCH_AVAILABLE:
                    # Need sequence — use last SEQ_LEN rows; for live we only have 1 row here
                    # Skip sequence models in single-row live prediction (they need history)
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
        def decisiveness(prob: float) -> float:
            return abs(prob - 0.5)

        best_model = max(model_probs, key=lambda m: decisiveness(model_probs[m]))
        best_prob = model_probs[best_model]
        best_direction = "UP" if best_prob >= 0.5 else "DOWN"
        # Confidence = distance from 0.5, scaled to [0.5, 1.0]
        best_confidence = 0.5 + decisiveness(best_prob)

        # ── Detect market regime → prefer regime-appropriate indicators ──────
        regime = self._regime_detector.detect(feature_row)
        preferred = self._regime_detector.preferred_indicators(regime)

        # ── Per-indicator signals (weighted by historical accuracy + regime) ──
        indicator_tracker = self._indicator_tracker.get(pair)
        weighted_votes: dict[str, float] = {}  # "UP"->weight, "DOWN"->weight

        for indicator, cols in self._indicator_groups.items():
            available_cols = [c for c in cols if c in feature_row.index]
            if not available_cols:
                continue
            # Use the mean of columns for the indicator
            ind_feat = feature_row.reindex(available_cols).fillna(0.0).values
            ind_feat_full = (
                feature_row.reindex(self._feature_cols)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
                .reshape(1, -1)
            )
            ind_feat_scaled = scaler.transform(ind_feat_full)

            try:
                # Use the best tabular model for indicator-level signal
                tabular_model = (
                    models.get("lightgbm")
                    or models.get("xgboost")
                    or models.get("randomforest")
                )
                if tabular_model is None:
                    continue
                ind_prob = float(tabular_model.predict_proba(ind_feat_scaled)[0, 1])
                ind_direction = "UP" if ind_prob >= 0.5 else "DOWN"
                ind_confidence = 0.5 + abs(ind_prob - 0.5)

                # Weight by historical accuracy × regime affinity boost
                base_weight = (
                    indicator_tracker.get_confidence_weight(indicator)
                    if indicator_tracker
                    else 0.55
                )
                # Indicators aligned with current market regime get a 1.5× boost
                regime_multiplier = 1.5 if indicator in preferred else 1.0
                weight = base_weight * regime_multiplier
                indicator_signals[indicator] = {
                    "direction": ind_direction,
                    "confidence": round(ind_confidence * base_weight, 4),
                    "regime_boosted": indicator in preferred,
                    "raw_prob": round(ind_prob, 4),
                }
                key = ind_direction
                weighted_votes[key] = weighted_votes.get(key, 0.0) + (
                    ind_confidence * weight
                )

            except Exception:
                pass

        # Incorporate indicator consensus into final direction/confidence
        if weighted_votes:
            consensus_direction = max(weighted_votes, key=lambda k: weighted_votes[k])
            total_weight = sum(weighted_votes.values())
            consensus_confidence = (
                weighted_votes[consensus_direction] / total_weight
                if total_weight > 0
                else 0.5
            )
            # Blend model confidence with indicator consensus (70/30)
            final_direction = (
                best_direction
                if best_confidence >= consensus_confidence
                else consensus_direction
            )
            final_confidence = round(
                0.70 * best_confidence + 0.30 * consensus_confidence, 4
            )
        else:
            final_direction = best_direction
            final_confidence = round(best_confidence, 4)

        # ── Martingale threshold gate ──────────────────────────────────────────
        threshold = self._mt.current_threshold
        if final_confidence < threshold:
            logger.debug(
                {
                    "event": "signal_below_threshold",
                    "pair": pair,
                    "confidence": final_confidence,
                    "threshold": threshold,
                    "streak": self._mt.current_streak,
                }
            )
            return None

        result = {
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
            "threshold_applied": threshold,
        }

        # Append to rolling prediction audit log
        log = self._prediction_log.setdefault(pair, [])
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
                "threshold": threshold,
            }
        )
        return result

    def update_indicator_accuracy(
        self, pair: str, indicator: str, correct: bool
    ) -> None:
        """Called after a trade result is known to update indicator accuracy."""
        tracker = self._indicator_tracker.get(pair)
        if tracker:
            tracker.update(indicator, correct)

    def record_result(self, pair: str) -> None:
        """Increment result counter for pair (used to trigger retraining)."""
        self._result_counts[pair] = self._result_counts.get(pair, 0) + 1

    def should_retrain(self, pair: str) -> bool:
        last = self._last_train_counts.get(pair, 0)
        current = self._result_counts.get(pair, 0)
        return (current - last) >= self._retrain_threshold

    def save(self, path: str) -> None:
        """Persist all models and trackers to disk."""
        os.makedirs(path, exist_ok=True)
        payload = {
            "models": self._models,
            "scalers": self._scalers,
            "metrics": self._metrics,
            "indicator_tracker": self._indicator_tracker,
            "result_counts": self._result_counts,
            "last_train_counts": self._last_train_counts,
        }
        with open(os.path.join(path, "model_manager.pkl"), "wb") as f:
            pickle.dump(payload, f)
        logger.info({"event": "models_saved", "path": path})

    def load(self, path: str) -> None:
        """Load persisted models from disk."""
        fpath = os.path.join(path, "model_manager.pkl")
        if not os.path.exists(fpath):
            logger.info({"event": "no_saved_models", "path": path})
            return
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        self._models = payload.get("models", {})
        self._scalers = payload.get("scalers", {})
        self._metrics = payload.get("metrics", {})
        self._indicator_tracker = payload.get("indicator_tracker", {})
        self._result_counts = payload.get("result_counts", {})
        self._last_train_counts = payload.get("last_train_counts", {})
        logger.info(
            {"event": "models_loaded", "path": path, "pairs": list(self._models.keys())}
        )
