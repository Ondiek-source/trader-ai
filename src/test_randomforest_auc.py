"""
RandomForest test using your actual features and labeler.
Run this locally on your laptop.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from ml_engine.features import (
    FeatureEngineer,
    get_feature_engineer,
    FEATURE_SET_BINARY_OPTIONS_AI,
)
from ml_engine.labeler import Labeler


def main():
    # Set ATR threshold here
    ATR_THRESHOLD = 1.0  # Change to 0.0, 0.3, 0.5, or 1.0

    print("=" * 60)
    print(f"RANDOM FOREST TEST – ATR THRESHOLD = {ATR_THRESHOLD}")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_parquet("../data/processed/EUR_USD_M1.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"   Loaded {len(df):,} rows")

    # Compute features
    print("\n2. Computing features...")
    eng = get_feature_engineer()
    fe = eng.transform(df)
    print(f"   Feature rows: {len(fe):,}")
    print(f"   Feature columns: {len(fe.columns)}")

    # Get feature columns from schema
    feature_cols = []
    for group in FEATURE_SET_BINARY_OPTIONS_AI.values():
        feature_cols.extend(group)
    available_cols = [c for c in feature_cols if c in fe.columns]
    print(f"   Features in schema: {len(available_cols)}")

    # Create labels with ATR threshold
    print(f"\n3. Creating labels (threshold = {ATR_THRESHOLD}×ATR)...")
    labeler = Labeler(expiry_key="1_MIN", atr_threshold=ATR_THRESHOLD)
    labels = labeler.compute_labels(fe)

    # Filter out skipped rows (-1)
    mask = labels >= 0
    X = fe.loc[mask, available_cols].values.astype(np.float32)
    y = labels[mask].values.astype(np.int32)

    print(f"   Valid rows: {len(X):,}")
    print(f"   CALL ratio: {y.mean():.1%}")

    # Clean any NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split (80/20, time-ordered)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train RandomForest
    print("\n4. Training RandomForest (300 trees, no depth limit)...")
    print("   This may take 5-15 minutes...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,  # Use all CPU cores
        class_weight="balanced",
        random_state=42,
        verbose=1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n5. Evaluating...")
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # Save and check size
    joblib.dump(model, f"randomforest_{ATR_THRESHOLD}.joblib")
    size_mb = os.path.getsize(f"randomforest_{ATR_THRESHOLD}.joblib") / (1024 * 1024)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"AUC: {auc:.4f}")
    print(f"Model size: {size_mb:.0f} MB")
    print(f"ATR threshold: {ATR_THRESHOLD}")
    print(f"Features used: {X.shape[1]}")

    # Compare with XGBoost
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"XGBoost (1.0×ATR): 0.9044 AUC, ~50 MB")
    print(f"RandomForest ({ATR_THRESHOLD}×ATR): {auc:.4f} AUC, {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
