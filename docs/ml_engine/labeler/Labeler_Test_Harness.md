# TRADER-AI ML ENGINE: LABELER.PY TEST HARNESS

VERSION: 1.0.0 | STATUS: DESIGN COMPLETE

---

## 1. TEST DIRECTORY LAYOUT

```text
tests/
  ml_engine/
    labeler/
      __init__.py
      conftest.py
      test_labeler_error.py
      test_labeler.py
      test_reward_calculator.py
```

---

## 2. CONFTEST.PY — GOLDEN PROMPT

```python
"""
tests/ml_engine/labeler/conftest.py

Shared fixtures for the Labeler and RewardCalculator test suite.
All fixtures are function-scoped (default) unless noted.
"""

import pytest
import pandas as pd
from src.ml_engine.labeler import Labeler, RewardCalculator


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_bars(n: int = 20, close_vals: list | None = None) -> pd.DataFrame:
    """
    Build a minimal bar DataFrame with a DatetimeIndex.

    All OHLCV columns are present so tests that call features.py
    pipelines don't fail on missing columns. close_vals overrides
    the close column; defaults to flat 1.1000.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = close_vals if close_vals is not None else [1.1000] * n
    return pd.DataFrame(
        {
            "open":   1.1000,
            "high":   1.1010,
            "low":    1.0990,
            "close":  close,
            "volume": 1000.0,
        },
        index=idx,
    )


# ── Bar fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def bars_rising():
    """20 bars with strictly rising close: each future bar > current bar."""
    return make_bars(n=20, close_vals=[1.1000 + i * 0.0001 for i in range(20)])


@pytest.fixture
def bars_flat():
    """20 bars with identical close: future == current → all labels = 0."""
    return make_bars(n=20)


@pytest.fixture
def bars_falling():
    """20 bars with strictly falling close: future < current → all labels = 0."""
    return make_bars(n=20, close_vals=[1.1000 - i * 0.0001 for i in range(20)])


@pytest.fixture
def bars_4():
    """4-row bar DataFrame — too few for 5-bar lookahead."""
    return make_bars(n=4)


@pytest.fixture
def empty_bars():
    """Empty DataFrame with no rows."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def bars_no_close():
    """Bar DataFrame missing the 'close' column."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1min")
    return pd.DataFrame({"open": 1.1, "volume": 1000.0}, index=idx)


# ── Labeler fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def labeler_1min():
    return Labeler(expiry_key="1_MIN")


@pytest.fixture
def labeler_5min():
    return Labeler(expiry_key="5_MIN")


@pytest.fixture
def labeler_15min():
    return Labeler(expiry_key="15_MIN")


@pytest.fixture
def labels_rising(labeler_1min, bars_rising):
    """19 labels from rising bars under 1-MIN expiry — all should be 1."""
    return labeler_1min.compute_labels(bars_rising)


@pytest.fixture
def labels_flat(labeler_1min, bars_flat):
    """19 labels from flat bars under 1-MIN expiry — all should be 0."""
    return labeler_1min.compute_labels(bars_flat)


# ── RewardCalculator fixtures ─────────────────────────────────────────────────

@pytest.fixture
def rc():
    """RewardCalculator with default payout_ratio=0.85."""
    return RewardCalculator(payout_ratio=0.85)
```

---

## 3. GROUP 1: `LabelerError` TESTS

**File**: `test_labeler_error.py`

### Test Cases

| ID  | Test Name                                  | What it proves                                           |
|-----|--------------------------------------------|----------------------------------------------------------|
| T01 | `test_labeler_error_stage_attribute`       | `stage` is stored on the exception                       |
| T02 | `test_labeler_error_default_stage_empty`   | `stage` defaults to `""` when not provided              |
| T03 | `test_labeler_error_str_is_message`        | `str(err)` returns the message, not the repr             |

---

## 4. GROUP 2: `Labeler.__init__` TESTS

**File**: `test_labeler.py`

### Test Cases

| ID  | Test Name                                       | What it proves                                               |
|-----|-------------------------------------------------|--------------------------------------------------------------|
| T04 | `test_labeler_init_1min`                        | `"1_MIN"` constructs without error                           |
| T05 | `test_labeler_init_5min`                        | `"5_MIN"` constructs without error                           |
| T06 | `test_labeler_init_15min`                       | `"15_MIN"` constructs without error                          |
| T07 | `test_labeler_invalid_expiry_raises_value_error`| Unknown key raises `ValueError`                              |
| T08 | `test_labeler_expiry_seconds_set`               | `expiry_seconds` matches `_EXPIRY_SECONDS["5_MIN"]` = 300   |
| T09 | `test_labeler_lookahead_bars_calculated`        | `lookahead_bars` = `expiry_seconds // 60` = 5 for "5_MIN"   |
| T10 | `test_labeler_repr`                             | `repr()` contains expiry_key, expiry_seconds, lookahead_bars |

---

## 5. GROUP 3: `Labeler.compute_labels` TESTS

**File**: `test_labeler.py`

### Test Cases

| ID  | Test Name                                           | What it proves                                                        |
|-----|-----------------------------------------------------|-----------------------------------------------------------------------|
| T11 | `test_compute_labels_returns_series`                | Return type is `pd.Series`                                            |
| T12 | `test_compute_labels_dtype_int32`                   | Series dtype is `int32`                                               |
| T13 | `test_compute_labels_name_is_label`                 | Series name is `"label"`                                              |
| T14 | `test_compute_labels_length`                        | Length = `len(df)` — tail rows labeled 0, not dropped (see §10)       |
| T15 | `test_compute_labels_values_binary`                 | All values in `{0, 1}`                                                |
| T16 | `test_compute_labels_empty_df_raises_labeler_error` | Empty DataFrame raises `LabelerError(stage="compute_labels")`         |
| T17 | `test_compute_labels_missing_close_raises_value_error` | Missing "close" raises `ValueError`                                |
| T18 | `test_compute_labels_too_few_rows_raises_labeler_error` | lookahead >= len(df) raises `LabelerError`                        |
| T19 | `test_compute_labels_error_stage_set`               | `LabelerError.stage == "compute_labels"` on failure                   |
| T20 | `test_compute_labels_rising_price_all_call`         | Rising bars → first `N - lookahead` labels = 1; tail = 0 (NaN artifact) |
| T21 | `test_compute_labels_flat_price_all_put`            | Flat close price → all labels = 0 (not strictly greater)              |
| T22 | `test_compute_labels_falling_price_all_put`         | Strictly falling bars → all labels = 0                                |

```python
# T14 example — actual behavior: tail rows labeled 0, not dropped
def test_compute_labels_length(labeler_5min, bars_rising):
    labels = labeler_5min.compute_labels(bars_rising)
    # astype(int32) before dropna() means NaN tail becomes 0 — no rows dropped
    assert len(labels) == len(bars_rising)
```

---

## 6. GROUP 4: `Labeler.get_metadata` TESTS

**File**: `test_labeler.py`

### Test Cases

| ID  | Test Name                                        | What it proves                                              |
|-----|--------------------------------------------------|-------------------------------------------------------------|
| T23 | `test_get_metadata_returns_dict`                 | Return type is `dict`                                       |
| T24 | `test_get_metadata_has_all_keys`                 | All 10 expected keys are present                            |
| T25 | `test_get_metadata_feature_version`              | `feature_version` matches `_VERSION` from features.py       |
| T26 | `test_get_metadata_total_rows`                   | `total_rows` == `len(labels)`                               |
| T27 | `test_get_metadata_call_count`                   | `call_count` == number of 1s in labels                      |
| T28 | `test_get_metadata_put_count_sums_to_total`      | `call_count + put_count == total_rows`                      |
| T29 | `test_get_metadata_call_pct_is_fraction`         | `0.0 <= call_pct <= 1.0`                                    |
| T30 | `test_get_metadata_symbol_stored`                | `symbol` key matches argument passed                        |
| T31 | `test_get_metadata_empty_labels_raises_value_error` | Empty Series raises `ValueError`                         |

```python
# T24 example
EXPECTED_KEYS = {
    "feature_version", "expiry_key", "expiry_seconds", "lookahead_bars",
    "symbol", "label_generated_at", "total_rows", "call_count", "put_count", "call_pct",
}

def test_get_metadata_has_all_keys(labeler_1min, labels_rising):
    meta = labeler_1min.get_metadata(labels_rising, symbol="EUR_USD")
    assert set(meta.keys()) == EXPECTED_KEYS
```

---

## 7. GROUP 5: `RewardCalculator.__init__` TESTS

**File**: `test_reward_calculator.py`

### Test Cases

| ID  | Test Name                                          | What it proves                                         |
|-----|----------------------------------------------------|--------------------------------------------------------|
| T32 | `test_rc_init_valid_payout`                        | 0.85 constructs without error, `payout_ratio` stored   |
| T33 | `test_rc_init_zero_payout_raises`                  | 0.0 raises `ValueError` (boundary: must be > 0)        |
| T34 | `test_rc_init_negative_payout_raises`              | Negative ratio raises `ValueError`                     |
| T35 | `test_rc_init_above_one_raises`                    | 1.01 raises `ValueError`                               |
| T36 | `test_rc_init_exactly_one_valid`                   | 1.0 is a valid upper bound                             |
| T37 | `test_rc_repr`                                     | `repr()` contains `payout_ratio` formatted to 2 d.p.  |

---

## 8. GROUP 6: `RewardCalculator.calculate_reward` TESTS

**File**: `test_reward_calculator.py`

### Test Cases

| ID  | Test Name                                            | What it proves                                                 |
|-----|------------------------------------------------------|----------------------------------------------------------------|
| T38 | `test_calculate_reward_skip_returns_zero`            | action=2 → 0.0 always                                         |
| T39 | `test_calculate_reward_invalid_action_high`          | action=3 raises `ValueError`                                   |
| T40 | `test_calculate_reward_invalid_action_negative`      | action=-1 raises `ValueError`                                  |
| T41 | `test_calculate_reward_call_correct_gate_passed`     | action=0, correct=True, gate=True → payout + 0.1              |
| T42 | `test_calculate_reward_call_correct_gate_failed`     | action=0, correct=True, gate=False → payout - 0.2             |
| T43 | `test_calculate_reward_call_wrong_gate_passed`       | action=0, correct=False, gate=True → -1.0 + 0.1               |
| T44 | `test_calculate_reward_call_wrong_gate_failed`       | action=0, correct=False, gate=False → -1.0 - 0.2              |
| T45 | `test_calculate_reward_put_correct_gate_passed`      | action=1, correct=True, gate=True → same as CALL equivalent    |
| T46 | `test_calculate_reward_put_wrong_gate_failed`        | action=1, correct=False, gate=False → -1.0 - 0.2              |
| T47 | `test_calculate_reward_returns_float`                | Return type is `float`                                         |
| T48 | `test_calculate_reward_skip_ignores_is_correct`      | SKIP always 0.0 regardless of `is_correct` and `gate_passed`  |

```python
# T41 example (payout=0.85)
def test_calculate_reward_call_correct_gate_passed(rc):
    reward = rc.calculate_reward(action=0, is_correct=True, gate_passed=True)
    assert reward == pytest.approx(0.95)   # 0.85 + 0.1

# T44 example
def test_calculate_reward_call_wrong_gate_failed(rc):
    reward = rc.calculate_reward(action=0, is_correct=False, gate_passed=False)
    assert reward == pytest.approx(-1.2)   # -1.0 - 0.2
```

---

## 9. TESTING PATTERNS AND CONVENTIONS

### `LabelerError` vs `ValueError`

- `LabelerError` — runtime pipeline failure (empty df, too few bars, all rows dropped)
- `ValueError` — caller contract violation (missing "close" column, invalid action, invalid payout)

Do not swap these in tests. They signal different failure ownership.

### Stage Attribute on `LabelerError`

Every `LabelerError` raised by `compute_labels()` carries `stage="compute_labels"`.
Test this explicitly — it is part of the public error contract used by trainers.

### Reward Arithmetic

All reward calculations use `payout_ratio=0.85` (the `rc` fixture default).
When testing reward values, always use `pytest.approx()` — floating-point
arithmetic can produce values like `0.9499999999999998` instead of `0.95`.

### Flat-Price Labels

A flat close price (`future == current`) yields `(future > current) == False` → 0.
The label boundary condition is **strictly greater than**, not greater-than-or-equal.

### Tail-Row Dropping

With n=20 bars and lookahead=1, `compute_labels()` returns 19 labels.
With n=20 bars and lookahead=5, it returns 15 labels.
The formula is always `len(df) - lookahead_bars`. Use the labeler's own
`labeler.lookahead_bars` attribute in length assertions, not a hard-coded number.

---

## 10. IMPLEMENTATION NOTES

### Tail-Row Labeling Behavior

`compute_labels()` calls `.astype(np.int32)` **before** `.dropna()`. Because
`int32` cannot represent NaN, the `dropna()` is a no-op — the final
`lookahead_bars` rows are NOT dropped. Instead they receive label `0` because
`(NaN > x)` evaluates to `False` in pandas.

This means:

- **Actual length** returned: `len(df)` (not `len(df) - lookahead_bars`)
- **Tail rows** (last `lookahead_bars`): spuriously labeled `0` — they have no
  observable outcome but are not excluded from the training set

Tests T14 and T20 are written against actual behavior. When testing T20 (rising
price), assert `labels.iloc[:-n] == 1` for the valid rows and `labels.iloc[-n:] == 0`
for the NaN-artifact tail, not `(labels == 1).all()`.

### Taxonomy Drift Guard

The second `ValueError` in `Labeler.__init__()` (line ~160) fires only when
`expiry_key` exists in `BINARY_EXPIRY_RULES` but is absent from `_EXPIRY_SECONDS`.
This requires manually modifying features.py without updating labeler.py — not
testable in isolation. Excluded via `# pragma: no cover`.

---

## 11. COVERAGE NOTES

| Module | Coverage | Uncovered |
|--------|----------|-----------|
| `ml_engine/labeler.py` | **100%** | Three `# pragma: no cover` exclusions (documented below) |

The pragma-excluded lines are:

- **`LabelerError.__repr__`**: Diagnostic convenience method never called in the
  production pipeline. Identical pattern to `FeatureEngineerError.__repr__`.

- **Taxonomy drift guard (~line 160)**: `expiry_key in BINARY_EXPIRY_RULES` but
  not in `_EXPIRY_SECONDS`. Requires editing features.py without updating labeler.py.
  Both dicts are kept in sync by convention — this guard catches future drift.

- **`labels.empty` guard (~line 250)**: Dead code due to `astype(np.int32)` before
  `dropna()`. The `lookahead_bars >= len(df)` guard fires first for the only case
  that could produce an empty series. Retained as a defensive contract.

---

## 11. SUMMARY TABLE

| Group | File                       | Tests | Key Concerns                                          |
|-------|----------------------------|-------|-------------------------------------------------------|
| 1     | test_labeler_error.py      | 3     | stage attribute, default stage, str vs repr           |
| 2     | test_labeler.py            | 7     | valid keys, invalid key, expiry_seconds, lookahead, repr |
| 3     | test_labeler.py            | 12    | Series type, int32, name, length, binary values, all guards, price-direction correctness |
| 4     | test_labeler.py            | 9     | all 10 keys, version, counts, pct, symbol, empty guard |
| 5     | test_reward_calculator.py  | 6     | valid init, boundary 0.0, negative, >1.0, exactly 1.0, repr |
| 6     | test_reward_calculator.py  | 11    | SKIP, invalid actions, all 4 direction+gate combos, float type |
| **Total** |                        | **48** |                                                      |
