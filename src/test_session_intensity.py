"""
Test session intensity features in isolation.
"""

import sys
import pandas as pd
import numpy as np
from src.ml_engine.features import FeatureEngineer, _EPS


def main():
    print("=" * 60)
    print("TEST: _compute_session_intensity in isolation")
    print("=" * 60)

    # Load data
    df = pd.read_parquet("../data/processed/EUR_USD_M1.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"Loaded {len(df):,} rows")

    # Take a small sample
    df_sample = df.iloc[:10000].copy()

    # Create FeatureEngineer
    eng = FeatureEngineer()

    # Resample and ffill (same as transform does)
    target_freq = "1min"
    fe = df_sample.sort_index().resample(target_freq).last()
    fe = fe.ffill(limit=5)
    print(f"After resample/ffill: {len(fe):,} rows")

    # Test just the session intensity method
    print("\nTesting _compute_session_intensity...")
    fe_result = eng._compute_session_intensity(fe.copy())

    # Check NaN counts
    print("\nResults:")
    print(f'  RVOL_LONDON - NaN count: {fe_result["RVOL_LONDON"].isna().sum()}')
    print(f'  RVOL_NEWYORK - NaN count: {fe_result["RVOL_NEWYORK"].isna().sum()}')
    print(f'  RVOL_TOKYO - NaN count: {fe_result["RVOL_TOKYO"].isna().sum()}')

    # Sample values
    print("\nSample values (first 20 rows):")
    print(fe_result[["RVOL_LONDON", "RVOL_NEWYORK", "RVOL_TOKYO"]].head(20).round(4))

    # Value ranges
    print("\nValue ranges:")
    print(
        f'  RVOL_LONDON: min={fe_result["RVOL_LONDON"].min():.2f}, max={fe_result["RVOL_LONDON"].max():.2f}'
    )
    print(
        f'  RVOL_NEWYORK: min={fe_result["RVOL_NEWYORK"].min():.2f}, max={fe_result["RVOL_NEWYORK"].max():.2f}'
    )
    print(
        f'  RVOL_TOKYO: min={fe_result["RVOL_TOKYO"].min():.2f}, max={fe_result["RVOL_TOKYO"].max():.2f}'
    )

    # Final check
    total_nan = (
        fe_result[["RVOL_LONDON", "RVOL_NEWYORK", "RVOL_TOKYO"]].isna().sum().sum()
    )
    if total_nan == 0:
        print("\n✅ SUCCESS: No NaN values in any RVOL column")
    else:
        print(f"\n❌ FAIL: {total_nan} NaN values still present")


if __name__ == "__main__":
    main()
