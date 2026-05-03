"""
Test AUC with real volume data from Dukascopy.
Run after regenerating EUR_USD_M1.parquet with volume.
"""

import sys

sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from src.ml_engine.features import (
    FeatureEngineer,
    get_feature_engineer,
    FEATURE_SET_BINARY_OPTIONS_AI,
)
from ml_engine.labeler import Labeler

print("=" * 60)
print("AUC TEST WITH REAL VOLUME DATA")
print("=" * 60)

# Load regenerated parquet
df = pd.read_parquet("../data/processed/EUR_USD_M1.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df.index.min()} → {df.index.max()}")

# Check volume statistics
print(f"\nVolume statistics:")
print(f'  Mean: {df["volume"].mean():.2f}')
print(f'  Std: {df["volume"].std():.2f}')
print(f'  Min: {df["volume"].min():.2f}')
print(f'  Max: {df["volume"].max():.2f}')
print(f'  Zeros: {(df["volume"] == 0).sum():,} rows')

# Use last 100,000 rows for test
df_sample = df.iloc[-100000:].copy()
print(f"\nUsing last {len(df_sample):,} rows for test")

# Compute features
eng = get_feature_engineer()
fe = eng.transform(df_sample)

# Get feature columns from schema
feature_cols = []
for group in FEATURE_SET_BINARY_OPTIONS_AI.values():
    feature_cols.extend(group)

available_cols = [c for c in feature_cols if c in fe.columns]
print(f"Feature columns: {len(available_cols)}")

# Create labels
labeler = Labeler(expiry_key="1_MIN")
labels = labeler.compute_labels(df_sample)

# Align
common_idx = fe.index.intersection(labels.index)
X = fe.loc[common_idx, available_cols].values.astype(np.float32)
y = labels.loc[common_idx].values.astype(np.int32)

print(f"Aligned samples: {len(X):,}")
print(f"CALL ratio: {y.mean():.1%}")

# Time-based 80/20 split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# Train CatBoost
print("\nTraining CatBoost...")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    auto_class_weights="Balanced",
    eval_metric="AUC",
    early_stopping_rounds=50,
    verbose=50,
    random_seed=42,
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Evaluate
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
best_iter = model.get_best_iteration()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"AUC: {auc:.4f}")
print(f"Best iteration: {best_iter}")
print(f"Features used: {X.shape[1]}")

if auc >= 0.76:
    print("\n🎯 Volume appears to be helping!")
elif auc >= 0.70:
    print("\n✅ Target hit (>=0.70)")
else:
    print("\n⚠️ Below target - may need label filtering")
