"""
Test AUC with different ATR thresholds using XGBoost.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from ml_engine.features import (
    FeatureEngineer,
    get_feature_engineer,
    FEATURE_SET_BINARY_OPTIONS_AI,
)
from ml_engine.labeler import Labeler


def main():
    print("=" * 60)
    print("AUC TEST – ATR THRESHOLD COMPARISON (XGBoost)")
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

    # Test thresholds (matching V2 experiment)
    thresholds = [0.0, 0.3, 0.5, 1.0]
    results = []

    print("\n" + "=" * 60)
    print("3. Training models for each threshold")
    print("=" * 60)

    for thr in thresholds:
        print(f'\n{"-"*50}')
        print(f"Threshold: {thr}×ATR")
        print(f'{"-"*50}')

        # Create labels with this threshold
        labeler = Labeler(expiry_key="1_MIN", atr_threshold=thr)
        labels = labeler.compute_labels(fe)

        # Filter out skipped rows (-1)
        mask = labels >= 0
        X = fe.loc[mask, available_cols].values.astype(np.float32)
        y = labels[mask].values.astype(np.int32)

        print(f"Valid rows: {len(X):,}")
        print(f"CALL ratio: {y.mean():.1%}")

        # Clean NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Time-based 80/20 split (preserve order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Train: {len(X_train):,} rows")
        print(f"Test: {len(X_test):,} rows")

        # Calculate scale_pos_weight for class balance
        pos_count = int(y_train.sum())
        neg_count = len(y_train) - pos_count
        scale_pos = neg_count / max(pos_count, 1)
        print(f"Scale pos weight: {scale_pos:.2f}")

        # Train XGBoost
        print("Training XGBoost...")
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos,
            eval_metric="auc",
            early_stopping_rounds=50,
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        best_iter = model.best_iteration if hasattr(model, "best_iteration") else "N/A"

        print(f"\n  ✅ AUC: {auc:.4f}")
        print(f"  Best iteration: {best_iter}")

        results.append(
            {
                "threshold": thr,
                "auc": auc,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "call_ratio": y.mean(),
                "best_iter": best_iter,
            }
        )

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n{'Threshold':<12} {'AUC':<8} {'Train Rows':<12} {'CALL Ratio':<12}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['threshold']}×ATR{'':<6} {r['auc']:.4f}    {r['train_rows']:,}       {r['call_ratio']:.1%}"
        )

    print("\n" + "=" * 60)
    print("IMPROVEMENT vs BASELINE (0.0×ATR)")
    print("=" * 60)
    baseline = results[0]["auc"]
    for r in results[1:]:
        improvement = r["auc"] - baseline
        print(f"  {r['threshold']}×ATR: +{improvement:.4f} AUC")

    # Compare to V2 experiment expectations
    print("\n" + "=" * 60)
    print("V2 EXPERIMENT REFERENCE")
    print("=" * 60)
    print("  V2 achieved with ensemble (CatBoost + XGB + LGB):")
    print("    0.0×ATR: 0.7715")
    print("    0.3×ATR: 0.8259")
    print("    0.5×ATR: 0.8405")
    print("    1.0×ATR: 0.8659")
    print("\n  Your XGBoost results (single model) may be slightly lower,")
    print("  but should show similar improvement pattern.")


if __name__ == "__main__":
    main()
