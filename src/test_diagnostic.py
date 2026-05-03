"""
Diagnostic test to check DataFrame state before each feature computation.
"""

import sys
import pandas as pd
from src.ml_engine.features import FeatureEngineer


def main():
    print("=" * 60)
    print("DIAGNOSTIC: DataFrame state before compute methods")
    print("=" * 60)

    # Load data
    df = pd.read_parquet("../data/processed/EUR_USD_M1.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {list(df.columns)}")
    print(f"First index: {df.index[0]}")
    print(f"Last index: {df.index[-1]}")

    eng = FeatureEngineer()

    # Step 1: Resample and ffill
    target_freq = "1min"
    fe = df.sort_index().resample(target_freq).last()
    fe = fe.ffill(limit=5)
    print(f"\nAfter resample/ffill: {len(fe):,} rows")
    print(f"Columns: {list(fe.columns)}")
    print(f"First 10 close values:")
    print(fe["close"].head(10))

    # Check for NaN in critical columns
    nan_open = fe["open"].isna().sum()
    nan_high = fe["high"].isna().sum()
    nan_low = fe["low"].isna().sum()
    nan_close = fe["close"].isna().sum()
    nan_volume = fe["volume"].isna().sum()

    print(f"\nNaN counts after resample/ffill:")
    print(f"  open: {nan_open}")
    print(f"  high: {nan_high}")
    print(f"  low: {nan_low}")
    print(f"  close: {nan_close}")
    print(f"  volume: {nan_volume}")

    if nan_close > 0:
        print("\nWARNING: close still has NaN after ffill!")
        print("This means there is a gap > 5 minutes in your data.")
        first_nan = fe[fe["close"].isna()].index[0]
        print(f"First NaN index: {first_nan}")
        return

    print("\nSUCCESS: No NaN in close after ffill")

    # Test price_action
    print("\n" + "=" * 60)
    print("TESTING _compute_price_action")
    print("=" * 60)

    required = ["open", "high", "low", "close"]
    has_all = all(c in fe.columns for c in required)
    print(f"Before price_action:")
    print(f"  Columns: {list(fe.columns)}")
    print(f"  Has open/high/low/close: {has_all}")

    fe_pa = eng._compute_price_action(fe.copy())
    print(f"\nAfter price_action:")
    new_cols = [c for c in fe_pa.columns if c not in fe.columns]
    print(f"  New columns added: {new_cols}")
    print(f"  Shape: {fe_pa.shape}")

    print("\nTest complete - stopping here")

    if __name__ == "__main__":
        main()
