# TRADER-AI ML ENGINE: FEATURES.PY TEST HARNESS

VERSION: 3.1.0-full-binary | STATUS: DESIGN COMPLETE | 125 TESTS PASSING

---

## 1. TEST DIRECTORY LAYOUT

```text
tests/
  ml_engine/
    features/
      conftest.py
      test_feature_vector.py
      test_trade_eligibility.py
      test_transform_pipeline.py
      test_price_action.py
      test_momentum.py
      test_volatility.py
      test_micro_structure.py
      test_context.py
      test_derived.py
      test_evaluate_gates.py
      test_get_expiry_features.py
      test_get_latest.py
      test_singleton.py
      test_feature_matrix.py
```

---

## 2. CONFTEST.PY — GOLDEN PROMPT

```python
"""
tests/ml_engine/features/conftest.py

Shared fixtures for the FeatureEngineer test suite.
All fixtures are function-scoped (default) unless noted.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock
from src.ml_engine.features import FeatureEngineer


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_ohlcv(
    n: int = 100,
    start: str = "2024-01-01",
    freq: str = "1min",
    open_: float = 1.1000,
    high_delta: float = 0.0010,
    low_delta: float = 0.0010,
    close_delta: float = 0.0002,
    volume: float = 1000.0,
) -> pd.DataFrame:
    """
    Build a synthetic M1 OHLCV DataFrame with a DatetimeIndex.

    All prices are offset from ``open_`` by fixed deltas. Volume is
    constant so RVOL-based tests are predictable.
    """
    idx = pd.date_range(start=start, periods=n, freq=freq)
    df = pd.DataFrame(
        {
            "open":   open_,
            "high":   open_ + high_delta,
            "low":    open_ - low_delta,
            "close":  open_ + close_delta,
            "volume": volume,
        },
        index=idx,
    )
    return df


def make_ticks(
    n: int = 600,
    start: str = "2024-01-01",
    freq: str = "100ms",
    bid: float = 1.0999,
    ask: float = 1.1001,
) -> pd.DataFrame:
    """Build a synthetic tick DataFrame with bid/ask columns."""
    idx = pd.date_range(start=start, periods=n, freq=freq)
    return pd.DataFrame({"bid": bid, "ask": ask}, index=idx)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    """
    Settings mock with all feature flags enabled and
    gate thresholds matching test expectations.
    """
    s = MagicMock()
    s.feat_price_action_enabled = True
    s.feat_momentum_enabled = True
    s.feat_volatility_enabled = True
    s.feat_micro_enabled = True
    s.feat_context_enabled = True
    s.gate_min_rvol = 1.5
    s.gate_max_spread = 0.0005
    return s


@pytest.fixture
def engineer(mock_settings, monkeypatch):
    """
    FeatureEngineer with mocked settings.
    Monkeypatches get_settings() so the real config file is never read.
    """
    import src.ml_engine.features as feat_mod
    monkeypatch.setattr(feat_mod, "get_settings", lambda: mock_settings)
    return FeatureEngineer()


@pytest.fixture
def bars_100():
    """100-row M1 OHLCV DataFrame — enough for all rolling windows."""
    return make_ohlcv(n=100)


@pytest.fixture
def bars_30():
    """30-row M1 OHLCV DataFrame — marginal for some rolling windows."""
    return make_ohlcv(n=30)


@pytest.fixture
def bars_5():
    """5-row M1 OHLCV DataFrame — too few for any rolling window."""
    return make_ohlcv(n=5)


@pytest.fixture
def ticks_600():
    """600 ticks at 100 ms intervals — ~1 minute of tick data."""
    return make_ticks(n=600)


@pytest.fixture
def empty_bars():
    """Empty OHLCV DataFrame."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def full_feature_df(engineer, bars_100, ticks_600):
    """Full feature matrix from transform() — used across multiple groups."""
    return engineer.transform(bars_100, ticks_600)
```

---

## 3. GROUP 1: `FeatureVector` SCHEMA TESTS

**File**: `test_feature_vector.py`

### Test Cases

| ID  | Test Name                                    | What it proves                                            |
|-----|----------------------------------------------|-----------------------------------------------------------|
| T01 | `test_feature_vector_vector_is_float32`      | `__post_init__` casts vector to float32                   |
| T02 | `test_feature_vector_vector_is_read_only`    | `flags.writeable = False` is applied                      |
| T03 | `test_feature_vector_to_dict_keys`           | `to_dict()` contains timestamp, vector, feature_names, version |
| T04 | `test_feature_vector_to_dict_vector_is_list` | `to_dict()["vector"]` is a Python list, not ndarray       |
| T05 | `test_feature_vector_repr_contains_n_features` | `repr()` shows `n_features=N`                           |
| T06 | `test_feature_vector_frozen_raises_on_setattr` | `fv.version = "x"` raises `FrozenInstanceError`         |
| T07 | `test_feature_vector_int_vector_coerced`     | Integer input vector is coerced to float32               |

```python
# T02 example
def test_feature_vector_vector_is_read_only():
    import pytest
    from src.ml_engine.features import FeatureVector
    import pandas as pd, numpy as np
    fv = FeatureVector(
        timestamp=pd.Timestamp("2024-01-01"),
        vector=np.array([1.0, 2.0, 3.0]),
        feature_names=["A", "B", "C"],
        version="3.1.0",
    )
    with pytest.raises(ValueError, match="read-only"):
        fv.vector[0] = 99.0
```

---

## 4. GROUP 2: `TradeEligibility` SCHEMA TESTS

**File**: `test_trade_eligibility.py`

### Test Cases

| ID  | Test Name                                       | What it proves                                   |
|-----|-------------------------------------------------|--------------------------------------------------|
| T08 | `test_trade_eligibility_to_dict_keys`           | `to_dict()` has is_eligible, gate_results, gate_values |
| T09 | `test_trade_eligibility_repr_shows_gate_count`  | `repr()` shows `gates=N/M` format               |
| T10 | `test_trade_eligibility_frozen_raises_on_setattr` | Mutation raises FrozenInstanceError            |
| T11 | `test_trade_eligibility_eligible_when_all_pass` | `is_eligible=True` when gate_results all True   |
| T12 | `test_trade_eligibility_ineligible_when_one_fails` | One False gate → `is_eligible=False`          |

---

## 5. GROUP 3: `transform()` PIPELINE TESTS

**File**: `test_transform_pipeline.py`

### Test Cases

| ID  | Test Name                                              | What it proves                                         |
|-----|--------------------------------------------------------|--------------------------------------------------------|
| T13 | `test_transform_returns_dataframe`                     | Return type is `pd.DataFrame`                         |
| T14 | `test_transform_empty_bars_returns_empty_df`           | `bars.empty` → returns `pd.DataFrame()` immediately   |
| T15 | `test_transform_missing_columns_raises_value_error`    | Missing `close` etc. raises `ValueError`              |
| T16 | `test_transform_missing_columns_logs_critical`         | Missing columns logs `CRITICAL` before raising        |
| T17 | `test_transform_output_has_expected_columns`           | Primary schema columns present in output              |
| T18 | `test_transform_output_index_is_datetimeindex`         | Output index is `pd.DatetimeIndex`                    |
| T19 | `test_transform_no_nans_after_warmup`                  | No NaN in any row after warmup rows dropped           |
| T20 | `test_transform_price_action_disabled_zeros_columns`   | Disabled group → zero-fill, not KeyError              |
| T21 | `test_transform_micro_disabled_zeros_micro_columns`    | `feat_micro_enabled=False` → zero-fill micro cols     |
| T22 | `test_transform_micro_none_ticks_zeros_micro_columns`  | `ticks=None` → zero-fill micro cols                  |
| T23 | `test_transform_non_datetime_index_coerced`            | Integer index is coerced to DatetimeIndex             |
| T24 | `test_transform_sorts_output_by_index`                 | Output is sorted ascending by timestamp               |
| T25 | `test_transform_100_bars_non_empty`                    | 100 bars → non-empty output (warmup satisfied)        |

```python
# T15 example
def test_transform_missing_columns_raises_value_error(engineer):
    bad = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1min"))
    with pytest.raises(ValueError, match="missing columns"):
        engineer.transform(bad)
```

---

## 6. GROUP 4: PRICE_ACTION FEATURE TESTS

**File**: `test_price_action.py`

### Test Cases

| ID  | Test Name                                      | What it proves                                          |
|-----|------------------------------------------------|---------------------------------------------------------|
| T26 | `test_body_to_range_ratio_bounds`              | `BODY_TO_RANGE_RATIO` ∈ [0, 1]                         |
| T27 | `test_doji_binary_is_zero_or_one`              | Binary column only contains 0 or 1                     |
| T28 | `test_marubozu_binary_is_zero_or_one`          | Binary column only contains 0 or 1                     |
| T29 | `test_engulfing_binary_is_zero_or_one`         | Binary column only contains 0 or 1                     |
| T30 | `test_pinbar_signal_is_zero_or_one`            | Binary column only contains 0 or 1                     |
| T31 | `test_two_bar_reversal_is_zero_or_one`         | Binary column only contains 0 or 1                     |
| T32 | `test_close_position_in_candle_bounds`         | Values ∈ [0, 1]                                        |
| T33 | `test_candle_position_in_day_bounds`           | Values ∈ [0, 1]                                        |
| T34 | `test_three_bar_slope_is_numeric`              | No NaN after warmup; finite values                     |
| T35 | `test_consecutive_bull_bars_non_negative`      | Values ≥ 0                                             |

---

## 7. GROUP 5: MOMENTUM FEATURE TESTS

**File**: `test_momentum.py`

### Test Cases

| ID  | Test Name                                  | What it proves                                        |
|-----|--------------------------------------------|-------------------------------------------------------|
| T36 | `test_rsi_bounds`                          | `RSI_14` ∈ [0, 100]                                  |
| T37 | `test_macd_hist_equals_value_minus_signal` | `MACD_HIST == MACD_VALUE - MACD_SIGNAL` row-by-row   |
| T38 | `test_roc_5_finite`                        | No NaN in `ROC_5` after warmup                       |
| T39 | `test_return_1_is_log_return`              | `RETURN_1 ≈ log(close / prev_close)` for sample row  |
| T40 | `test_momentum_oscillator_finite`          | `MOMENTUM_OSCILLATOR` has no NaN after warmup         |
| T41 | `test_cci_14_finite`                       | `CCI_14` has no NaN after warmup                     |

---

## 8. GROUP 6: VOLATILITY FEATURE TESTS

**File**: `test_volatility.py`

### Test Cases

| ID  | Test Name                                  | What it proves                                        |
|-----|--------------------------------------------|-------------------------------------------------------|
| T42 | `test_atr_14_non_negative`                 | `ATR_14 >= 0`                                        |
| T43 | `test_natr_14_non_negative`                | `NATR_14 >= 0`                                       |
| T44 | `test_bb_width_non_negative`               | `BB_WIDTH >= 0`                                      |
| T45 | `test_bb_percent_b_near_zero_point_five_flat_market` | Flat price → `BB_PERCENT_B ≈ 0.5`         |
| T46 | `test_kc_width_non_negative`               | `KC_WIDTH >= 0`                                      |
| T47 | `test_range_expansion_ratio_near_one_flat` | Flat market → `RANGE_EXPANSION_RATIO ≈ 1.0`         |

---

## 9. GROUP 7: MICRO_STRUCTURE FEATURE TESTS

**File**: `test_micro_structure.py`

### Test Cases

| ID  | Test Name                                          | What it proves                                          |
|-----|----------------------------------------------------|---------------------------------------------------------|
| T48 | `test_tick_velocity_non_negative`                  | `TICK_VELOCITY >= 0`                                   |
| T49 | `test_relative_volume_rvol_positive`               | `RELATIVE_VOLUME_RVOL > 0`                             |
| T50 | `test_spread_normalized_non_negative`              | `SPREAD_NORMALIZED >= 0`                               |
| T51 | `test_order_flow_imbalance_bounds`                 | `ORDER_FLOW_IMBALANCE` ∈ [-0.5, 0.5]                  |
| T52 | `test_micro_columns_zero_when_ticks_none`          | All micro cols == 0.0 when `ticks=None`                |
| T53 | `test_micro_columns_zero_when_flag_disabled`       | All micro cols == 0.0 when `feat_micro_enabled=False`  |

---

## 10. GROUP 8: CONTEXT FEATURE TESTS

**File**: `test_context.py`

### Test Cases

| ID  | Test Name                                        | What it proves                                         |
|-----|--------------------------------------------------|--------------------------------------------------------|
| T54 | `test_session_london_active_at_10_utc`           | Hour 10 → `SESSION_LONDON == 1`                       |
| T55 | `test_session_london_inactive_at_22_utc`         | Hour 22 → `SESSION_LONDON == 0`                       |
| T56 | `test_session_overlap_active_at_14_utc`          | Hour 14 → `SESSION_OVERLAP_LONDON_NY == 1`            |
| T57 | `test_time_sine_cosine_within_bounds`            | `TIME_SINE`, `TIME_COSINE` ∈ [-1, 1]                 |
| T58 | `test_hour_of_day_normalized_bounds`             | `HOUR_OF_DAY_NORMALIZED` ∈ [0, 1]                    |
| T59 | `test_minutes_to_news_is_90`                     | Placeholder value == 90                               |
| T60 | `test_context_zeros_when_non_datetime_index`     | Non-DatetimeIndex → zero-fill all CONTEXT cols        |

---

## 11. GROUP 9: DERIVED FEATURE TESTS

**File**: `test_derived.py`

### Test Cases

| ID  | Test Name                                      | What it proves                                          |
|-----|------------------------------------------------|---------------------------------------------------------|
| T61 | `test_roc5_acceleration_finite_after_warmup`   | `ROC_5_ACCELERATION` has no NaN after warmup           |
| T62 | `test_atr14_ratio_change_finite_after_warmup`  | `ATR_14_RATIO_CHANGE` has no NaN after warmup          |
| T63 | `test_macd_hist_slope_finite`                  | `MACD_HIST_SLOPE` finite after warmup                  |
| T64 | `test_bb_width_ma20_non_negative`              | `BB_WIDTH_MA_20 >= 0`                                  |
| T65 | `test_rvol_alias_equals_relative_volume`       | `RVOL == RELATIVE_VOLUME_RVOL` row-by-row              |
| T66 | `test_session_context_is_zero_or_one`          | `SESSION_CONTEXT` ∈ {0, 1}                             |
| T67 | `test_derived_zero_filled_when_deps_missing`   | If `ROC_5` absent, `ROC_5_ACCELERATION == 0.0`        |

---

## 12. GROUP 10: `evaluate_gates()` TESTS

**File**: `test_evaluate_gates.py`

### Test Cases

| ID  | Test Name                                            | What it proves                                              |
|-----|------------------------------------------------------|-------------------------------------------------------------|
| T68 | `test_evaluate_gates_empty_df_returns_ineligible`    | `fe.empty` → `is_eligible=False`, all gates False          |
| T69 | `test_evaluate_gates_all_pass_returns_eligible`      | All conditions satisfied → `is_eligible=True`              |
| T70 | `test_evaluate_gates_bb_fail_blocks_trade`           | `BB_WIDTH < BB_WIDTH_MA_20` → `is_eligible=False`          |
| T71 | `test_evaluate_gates_atr_fail_blocks_trade`          | `ATR_14 < ATR_14_MA_20` → `is_eligible=False`             |
| T72 | `test_evaluate_gates_rvol_fail_blocks_trade`         | `RVOL < gate_min_rvol` → `is_eligible=False`              |
| T73 | `test_evaluate_gates_spread_fail_blocks_trade`       | `SPREAD_NORMALIZED > gate_max_spread` → `is_eligible=False`|
| T74 | `test_evaluate_gates_gate_values_populated`          | `gate_values` dict non-empty and contains numeric values   |
| T75 | `test_evaluate_gates_uses_settings_thresholds`       | Gate strings match configured `gate_min_rvol` value        |
| T76 | `test_evaluate_gates_returns_trade_eligibility_type` | Return type is `TradeEligibility`                          |

```python
# T69 example
def test_evaluate_gates_all_pass_returns_eligible(engineer):
    fe = pd.DataFrame({
        "BB_WIDTH":         [0.02],
        "BB_WIDTH_MA_20":   [0.01],   # BB_WIDTH > MA → pass
        "ATR_14":           [0.001],
        "ATR_14_MA_20":     [0.0005], # ATR > MA → pass
        "RVOL":             [2.0],    # > gate_min_rvol (1.5) → pass
        "SPREAD_NORMALIZED":[0.0001], # < gate_max_spread (0.0005) → pass
    })
    result = engineer.evaluate_gates(fe)
    assert result.is_eligible is True
    assert all(result.gate_results.values())
```

---

## 13. GROUP 11: `get_expiry_features()` TESTS

**File**: `test_get_expiry_features.py`

### Test Cases

| ID  | Test Name                                              | What it proves                                          |
|-----|--------------------------------------------------------|---------------------------------------------------------|
| T77 | `test_get_expiry_features_1min_returns_correct_cols`   | Returns `TICK_VELOCITY`, `RELATIVE_VOLUME_RVOL`, `BODY_TO_RANGE_RATIO` subset |
| T78 | `test_get_expiry_features_invalid_key_raises`          | Unknown key raises `ValueError` with message           |
| T79 | `test_get_expiry_features_missing_cols_logs_warning`   | Missing columns log `WARNING` (caplog)                 |
| T80 | `test_get_expiry_features_15min_resolves_session_context` | `SESSION_CONTEXT` alias resolved correctly           |
| T81 | `test_get_expiry_features_returns_dataframe_subset`    | Return type is `pd.DataFrame` with only expiry cols    |

---

## 14. GROUP 12: `get_latest()` TESTS

**File**: `test_get_latest.py`

### Test Cases

| ID  | Test Name                                              | What it proves                                          |
|-----|--------------------------------------------------------|---------------------------------------------------------|
| T82 | `test_get_latest_returns_feature_vector`               | Return type is `FeatureVector`                         |
| T83 | `test_get_latest_vector_is_float32`                    | `fv.vector.dtype == np.float32`                        |
| T84 | `test_get_latest_vector_is_read_only`                  | `fv.vector.flags.writeable == False`                   |
| T85 | `test_get_latest_version_matches_module_constant`      | `fv.version == _VERSION`                               |
| T86 | `test_get_latest_feature_names_match_schema`           | `fv.feature_names` is a subset of `FEATURE_SET_BINARY_OPTIONS_AI` flat list |
| T87 | `test_get_latest_empty_bars_raises_feature_engineer_error` | `bars.empty` → raises `FeatureEngineerError`       |
| T88 | `test_get_latest_too_few_bars_raises_feature_engineer_error` | 5-row bars → raises `FeatureEngineerError`        |
| T89 | `test_get_latest_stage_set_on_error`                   | `FeatureEngineerError.stage == "get_latest"` when insufficient data |
| T90 | `test_get_latest_nan_values_replaced_with_zero`        | NaN in output vector → replaced with 0.0, no raise    |
| T91 | `test_get_latest_inf_values_replaced_with_zero`        | Inf in output vector → replaced with 0.0, no raise    |
| T92 | `test_get_latest_transform_exception_wrapped`          | Exception inside transform → raises `FeatureEngineerError(stage="transform")` |

```python
# T87 example
def test_get_latest_empty_bars_raises_feature_engineer_error(engineer, empty_bars):
    from src.ml_engine.features import FeatureEngineerError
    with pytest.raises(FeatureEngineerError):
        engineer.get_latest(empty_bars)

# T92 example
def test_get_latest_transform_exception_wrapped(engineer, bars_100, monkeypatch):
    from src.ml_engine.features import FeatureEngineerError
    monkeypatch.setattr(engineer, "transform", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(FeatureEngineerError) as exc_info:
        engineer.get_latest(bars_100)
    assert exc_info.value.stage == "transform"
```

---

## 15. GROUP 13: SINGLETON TESTS

**File**: `test_singleton.py`

### Test Cases

| ID  | Test Name                                          | What it proves                                         |
|-----|----------------------------------------------------|--------------------------------------------------------|
| T93 | `test_get_feature_engineer_returns_engineer`       | Return type is `FeatureEngineer`                      |
| T94 | `test_get_feature_engineer_returns_same_instance`  | Second call returns identical object (is-check)       |
| T95 | `test_get_feature_engineer_after_reset_creates_new`| After `_engineer = None`, next call creates fresh instance |

```python
# T94 example
def test_get_feature_engineer_returns_same_instance(monkeypatch):
    import src.ml_engine.features as feat_mod
    monkeypatch.setattr(feat_mod, "_engineer", None)
    a = feat_mod.get_feature_engineer()
    b = feat_mod.get_feature_engineer()
    assert a is b
```

---

## 16. IMPLEMENTATION NOTES

### Rolling Window Warmup

The longest rolling window is MACD (26-period EMA). With additional derived
features using `rolling(20)`, tests should use `n=100` bars to ensure all
warmup rows are consumed before assertions. Tests using `n=30` should assert
that output is non-empty but may have fewer rows than input.

### Deterministic Flat-Market Tests

For tests verifying exact mathematical properties (e.g., `BB_PERCENT_B ≈ 0.5`
in a flat market), construct a DataFrame with all OHLC == constant and zero
volume variance. This eliminates floating-point accumulation error from price
drift.

### Mocking `get_settings()`

Always monkeypatch `get_settings` at the module level:
```python
monkeypatch.setattr("src.ml_engine.features.get_settings", lambda: mock_settings)
```
Do not mock it at the `core.config` level — this would affect other modules
loaded in the same session.

### `FeatureEngineerError` vs `ValueError`

- `FeatureEngineerError` — runtime pipeline failure; test with `pytest.raises(FeatureEngineerError)`
- `ValueError` — caller contract violation (missing columns, invalid array shape); test with `pytest.raises(ValueError)`

These must not be swapped in tests — they signal different failure ownership.

### Checking `caplog` for Diagnostic Symbols

```python
def test_transform_empty_bars_logs_warning(engineer, empty_bars, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        engineer.transform(empty_bars)
    assert "[%]" in caplog.text or "EMPTY BARS" in caplog.text
```

### Singleton Reset Between Tests

The module-level `_engineer` persists across tests in the same process.
Always reset it in singleton tests:
```python
monkeypatch.setattr("src.ml_engine.features._engineer", None)
```

### `_make_fm` Helper Pattern

The `_make_fm` factory in `test_feature_matrix.py` uses `dict.update(**overrides)` which
triggers Pylance type inference warnings when calling `FeatureMatrix(**defaults)`. Suppress
with `# type: ignore[arg-type]` on that line — the same pattern applies to `_make_fv` and
`_make_te`.

---

## 14. GROUP 14: `FeatureMatrix` DATACLASS TESTS

**File**: `test_feature_matrix.py`

### Test Cases

| ID   | Test Name                                              | What it proves                                                     |
|------|--------------------------------------------------------|--------------------------------------------------------------------|
| T102 | `test_feature_matrix_matrix_is_float32`                | `__post_init__` coerces matrix to float32                          |
| T103 | `test_feature_matrix_matrix_is_read_only`              | `flags.writeable = False` is applied; write raises ValueError      |
| T104 | `test_feature_matrix_frozen_raises_on_setattr`         | Frozen dataclass — field assignment raises `FrozenInstanceError`   |
| T105 | `test_feature_matrix_int_matrix_coerced`               | Integer (int32) matrix coerced to float32 with values preserved    |
| T106 | `test_feature_matrix_1d_raises_value_error`            | 1D array raises `ValueError` mentioning "2D"                       |
| T107 | `test_feature_matrix_3d_raises_value_error`            | 3D array raises `ValueError` mentioning "2D"                       |
| T108 | `test_feature_matrix_mismatched_columns_raises_value_error` | `shape[1] != len(feature_names)` raises `ValueError`         |
| T109 | `test_feature_matrix_1d_logs_critical`                 | Invalid shape logs CRITICAL "FEATURE MATRIX ERROR" before raising  |
| T110 | `test_feature_matrix_mismatched_columns_logs_critical` | Column/name mismatch logs CRITICAL before raising                  |
| T111 | `test_feature_matrix_len_returns_row_count`            | `len(fm)` returns `matrix.shape[0]`                               |
| T112 | `test_feature_matrix_to_tensor_shape`                  | `to_tensor()` returns shape `(1, N, F)`                           |
| T113 | `test_feature_matrix_to_tensor_is_3d`                  | `to_tensor().ndim == 3`                                           |
| T114 | `test_feature_matrix_to_tensor_data_preserved`         | `to_tensor()[0]` is numerically identical to `matrix`             |
| T115 | `test_feature_matrix_repr_contains_symbol`             | `repr()` includes the symbol field                                |
| T116 | `test_feature_matrix_repr_contains_shape`              | `repr()` includes both matrix dimensions                          |
| T117 | `test_feature_matrix_repr_contains_version`            | `repr()` includes the version string                              |

```python
# T106 example
def test_feature_matrix_1d_raises_value_error():
    with pytest.raises(ValueError, match="2D"):
        _make_fm(matrix=np.ones(4), feature_names=["A", "B", "C", "D"])

# T112 example
def test_feature_matrix_to_tensor_shape():
    fm = _make_fm(matrix=np.ones((3, 4)))
    assert fm.to_tensor().shape == (1, 3, 4)
```

---

## 15. GROUP 15: `FeatureEngineer.build_matrix()` TESTS

**File**: `test_feature_matrix.py` (same file as Group 14)

`build_matrix()` runs the full `transform()` pipeline and packages the result into
an immutable `FeatureMatrix`. It is the entry point for offline training data preparation.

### Test Cases

| ID   | Test Name                                                  | What it proves                                                      |
|------|------------------------------------------------------------|---------------------------------------------------------------------|
| T118 | `test_build_matrix_returns_feature_matrix`                 | Happy-path returns a `FeatureMatrix` instance                       |
| T119 | `test_build_matrix_version_matches_module_constant`        | `version` field equals `_VERSION` module constant                   |
| T120 | `test_build_matrix_symbol_stored`                          | `symbol` argument stored verbatim in the returned matrix            |
| T121 | `test_build_matrix_matrix_shape_matches_feature_set`       | `shape[1]` matches `len(feature_names)`; bounded by schema          |
| T122 | `test_build_matrix_timestamps_length_matches_rows`         | `len(timestamps) == matrix.shape[0]` — every row has a timestamp   |
| T123 | `test_build_matrix_matrix_is_float32`                      | Matrix dtype is float32 regardless of intermediate precision        |
| T124 | `test_build_matrix_empty_bars_raises_feature_engineer_error` | Empty bars raises `FeatureEngineerError`                          |
| T125 | `test_build_matrix_transform_exception_wrapped`            | Exception from `transform()` wrapped as `FeatureEngineerError(stage="build_matrix")` |

```python
# T125 example
def test_build_matrix_transform_exception_wrapped(engineer, bars_100, monkeypatch):
    from src.ml_engine.features import FeatureEngineerError

    def _boom(*a, **kw):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(engineer, "transform", _boom)
    with pytest.raises(FeatureEngineerError) as exc_info:
        engineer.build_matrix(bars_100, symbol="EURUSD")
    assert exc_info.value.stage == "build_matrix"
    assert "pipeline exploded" in str(exc_info.value)
```

---

## 17. COVERAGE NOTES (as of 3.1.0-full-binary + FeatureMatrix)

| Module | Coverage | Uncovered |
|--------|----------|-----------|
| `ml_engine/features.py` | **100%** | — (two pragma: no cover exclusions documented below) |

The 2 pragma-excluded lines are:

- **Lines ~383-384** (index coercion `except` branch): `pd.to_datetime()` raising on a truly
  uncoercible index — storage guarantees DatetimeIndex on all real data. Logs `[!]` CRITICAL
  before re-raising as `ValueError`.
- **Line ~605** (ATR_14_RATIO_CHANGE zero-fill in `_compute_derived`): Only reachable if
  VOLATILITY group is disabled while DERIVED runs — the zero-fill path is a defensive
  contract, not a normal execution path.

---

## 18. SUMMARY TABLE

| Group | File                           | Tests | Key Concerns                                    |
|-------|--------------------------------|-------|-------------------------------------------------|
| 1     | test_feature_vector.py         | 7     | float32 coercion, read-only, frozen, to_dict    |
| 2     | test_trade_eligibility.py      | 5     | to_dict, repr, frozen, gate aggregation         |
| 3     | test_transform_pipeline.py     | 17    | Empty, missing cols, schema completeness, gap detection, index coercion, schema warning |
| 4     | test_price_action.py           | 10    | Bounds, binary flags, numeric validity          |
| 5     | test_momentum.py               | 6     | RSI bounds, MACD identity, log-return formula   |
| 6     | test_volatility.py             | 6     | Non-negative, finite BB_PERCENT_B, ratio=1 flat |
| 7     | test_micro_structure.py        | 6     | Zero-fill when disabled or ticks=None           |
| 8     | test_context.py                | 7     | Session flags, cyclical encoding, non-DI guard  |
| 9     | test_derived.py                | 7     | Dependencies, zero-fill when deps absent        |
| 10    | test_evaluate_gates.py         | 9     | Each gate individually, settings thresholds     |
| 11    | test_get_expiry_features.py    | 5     | Alias resolution, invalid key, missing cols     |
| 12    | test_get_latest.py             | 13    | FeatureVector shape, error wrapping, NaN/Inf guard, FeatureEngineerError attrs |
| 13    | test_singleton.py              | 3     | Identity, lazy init, reset                      |
| 14    | test_feature_matrix.py         | 16    | float32 coercion, read-only, frozen, 2D shape, mismatch, to_tensor, len, repr, CRITICAL log |
| 15    | test_feature_matrix.py         | 8     | build_matrix() contract, symbol/version, error wrapping, shape alignment |
| **Total** |                            | **125** |                                              |

### Tests Added Beyond Original Design

| ID | Test | Reason added |
|----|------|--------------|
| T_NEW_01 | `test_transform_logs_warning_on_resampled_gaps` | Covers new gap-detection logic (`_MAX_FFILL_BARS`) |
| T_NEW_02 | `test_transform_logs_warning_on_non_datetime_index` | Covers new index-coercion warning |
| T_NEW_03 | `test_transform_logs_schema_warning_when_group_disabled` | Covers new output schema validation |
| T_NEW_04 | `test_transform_large_gap_does_not_fabricate_flat_prices` | Verifies gap > limit is not silently fabricated |
| T_NEW_05 | `test_feature_engineer_error_repr` | Covers `FeatureEngineerError.__repr__` |
| T_NEW_06 | `test_feature_engineer_error_stage_attribute` | Covers `FeatureEngineerError.stage` constructor path |
| T102-T117 | `test_feature_matrix.py` (Group 14) | `FeatureMatrix` dataclass added by user |
| T118-T125 | `test_feature_matrix.py` (Group 15) | `build_matrix()` method added by user |
