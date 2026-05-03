"""
Test Labeler with ATR thresholds.
"""

import sys
import pandas as pd
import numpy as np
from ml_engine.features import (
    FeatureEngineer,
    get_feature_engineer,
    FEATURE_SET_BINARY_OPTIONS_AI,
)
from ml_engine.labeler import Labeler


def main():
    print("=" * 60)
    print("TEST: Labeler with ATR thresholds")
    print("=" * 60)

    # Load data
    df = pd.read_parquet("../data/processed/EUR_USD_M1.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"Loaded {len(df):,} rows")

    # Compute features (need ATR for filtering)
    eng = get_feature_engineer()
    fe = eng.transform(df)
    print(f"Feature rows: {len(fe):,}")

    # Ensure ATR is present
    if "ATR" not in fe.columns:
        print("ERROR: ATR column missing from features")
        sys.exit(1)

    print(
        f'\nATR stats: min={fe["ATR"].min():.6f}, max={fe["ATR"].max():.6f}, mean={fe["ATR"].mean():.6f}'
    )

    # Test different thresholds
    thresholds = [0.0, 0.3, 0.5, 1.0]
    results = []

    print("\n" + "=" * 60)
    print("LABEL STATISTICS BY THRESHOLD")
    print("=" * 60)

    for thr in thresholds:
        labeler = Labeler(expiry_key="1_MIN", atr_threshold=thr)
        labels = labeler.compute_labels(fe)

        valid = labels[labels >= 0]
        skipped = labels[labels == -1]

        call_pct = valid.mean() if len(valid) > 0 else 0

        results.append(
            {
                "threshold": thr,
                "total_rows": len(labels),
                "kept": len(valid),
                "skipped": len(skipped),
                "kept_pct": len(valid) / len(labels) * 100,
                "call_pct": call_pct * 100,
            }
        )

        print(f"\nThreshold: {thr}×ATR")
        print(f"  Total labels: {len(labels):,}")
        print(f"  Kept: {len(valid):,} ({len(valid)/len(labels)*100:.1f}%)")
        print(f"  Skipped: {len(skipped):,} ({len(skipped)/len(labels)*100:.1f}%)")
        print(f"  CALL ratio (kept only): {call_pct:.1f}%")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['threshold']}×ATR: {r['kept_pct']:.1f}% kept, CALL={r['call_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
