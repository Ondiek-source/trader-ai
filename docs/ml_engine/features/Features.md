# TRADER-AI ML ENGINE: FEATURES.PY DESIGN DOCUMENT

VERSION: 3.1.0-full-binary | STATUS: FROZEN / PRODUCTION-READY

ROLE: THE TRANSLATOR | VISION: RAW OHLCV → MODEL-READY VECTORS | STATELESS | THREAD-SAFE

---

## 1. ARCHITECTURAL PHILOSOPHY

The `features.py` module acts as "The Translator" of the ML engine. Where
`storage.py` owns the market-price vault and `journal.py` owns the decision
ledger, the Translator owns the mathematical bridge between them: it converts
raw OHLCV bars and optional tick-level data into the deterministic, versioned
feature vectors that the model layer consumes.

The module operates under three design principles:

**The Statelessness Principle**: `FeatureEngineer` holds no mutable state after
construction. Every intermediate DataFrame is a local variable inside the call
stack. This makes it safe to call `transform()` and `get_latest()` concurrently
from multiple engine threads without any locking.

**The Completeness Principle**: `transform()` always returns a DataFrame with
the full `FEATURE_SET_BINARY_OPTIONS_AI` column set, regardless of which feature
groups are enabled. Disabled groups are zero-filled so the downstream model always
sees the same column schema and can be retrained without pipeline changes.

**The Fail-Loud Principle** (consistent with the Dictator Pattern): `get_latest()`
raises `FeatureEngineerError` rather than silently returning `None`. The engine
layer decides whether to retry, skip the signal, or halt — silent data loss is
never acceptable at the inference gate.

---

## 2. THE DIAGNOSTIC LANGUAGE

Consistent with the Core Diagnostic Registry:

| Symbol | Category | Context / Usage                                                           |
|--------|----------|---------------------------------------------------------------------------|
| `[!]`  | FATAL    | Schema violation — missing required columns (critical + raise).           |
| `[%]`  | LOGIC    | Non-finite feature values detected; empty bars input; gate anomalies.    |
| `[^]`  | WARNING  | Rolling warmup row-drop (debug level — expected, not an error).           |

---

## 3. CORE COMPONENTS

### A. The Custom Exception (`FeatureEngineerError`)

A typed exception distinct from `StorageError` and `JournalError`. Carries
a `stage` attribute identifying which pipeline step failed (`"transform"`,
`"get_latest"`).

Raised in two situations:

1. **Transform failure** — any unexpected exception inside `transform()` is
   caught and re-raised as `FeatureEngineerError(stage="transform")`.
2. **Empty output** — `transform()` returns an empty DataFrame (too few bars
   for rolling-window warmup). Raised as `FeatureEngineerError(stage="get_latest")`.

`ValueError` from missing columns is NOT wrapped — it propagates unchanged
because it represents a caller contract violation.

### B. The Output Schemas (`FeatureVector`, `FeatureMatrix`, `TradeEligibility`)

All three are frozen dataclasses (`frozen=True`). Mutation after construction is
impossible — any correction requires creating a new instance, ensuring that
the inference and training layers always observe a consistent snapshot.

**`FeatureVector`** is the primary output of `get_latest()` — one row, live inference:

| Field          | Type           | Constraint                                     |
|----------------|----------------|------------------------------------------------|
| `timestamp`    | pd.Timestamp   | UTC timestamp of the source bar                |
| `vector`       | np.ndarray     | float32, read-only (flags.writeable = False)   |
| `feature_names`| list[str]      | Ordered; len must equal len(vector)            |
| `version`      | str            | Mirrors `_VERSION` module constant             |

**`FeatureMatrix`** is the batch output of `build_matrix()` — all rows, offline training:

| Field          | Type           | Constraint                                           |
|----------------|----------------|------------------------------------------------------|
| `timestamps`   | list           | UTC timestamps aligned to each row in matrix         |
| `matrix`       | np.ndarray     | float32, shape (N, F), read-only                     |
| `feature_names`| list[str]      | Ordered column names; len must equal matrix.shape[1] |
| `version`      | str            | Mirrors `_VERSION` — must match label alignment      |
| `symbol`       | str            | Currency pair identifier (e.g. `"EURUSD"`)           |

`FeatureMatrix.__post_init__` validates two hard invariants and raises `ValueError`
with a `[!]`-bordered CRITICAL log block before raising if either is violated:

1. `matrix.ndim == 2` — rejects 1D and 3D inputs
2. `matrix.shape[1] == len(feature_names)` — rejects mismatched column counts

Additional methods on `FeatureMatrix`:

- `to_tensor()` — returns `matrix[np.newaxis, :, :]` (shape `(1, N, F)`) for
  SequenceGenerator seeding in deep-learning pipelines
- `__len__()` — returns `matrix.shape[0]` (number of bars)
- `__repr__()` — shows symbol, shape, and version for quick diagnostics

The `__post_init__` pattern on both `FeatureVector` and `FeatureMatrix` coerces the
numpy field to float32 via `object.__setattr__` (required on frozen dataclasses to
bypass `FrozenInstanceError` on field reassignment) then sets `flags.writeable = False`
directly on the array object (a flag mutation, not a field reassignment — permitted
even on frozen dataclasses).

**`TradeEligibility`** is the output of `evaluate_gates()`:

| Field         | Type            | Constraint                                      |
|---------------|-----------------|-------------------------------------------------|
| `is_eligible` | bool            | True only when ALL gates pass                   |
| `gate_results`| dict[str, bool] | Per-gate boolean, keyed by gate expression      |
| `gate_values` | dict[str, float]| Per-gate raw numeric for diagnostic logging     |

All three schemas implement `to_dict()` and `__repr__` consistent with `TradeEntry`
and `SignalEntry` in `journal.py`.

### C. The Feature Schema (`FEATURE_SET_BINARY_OPTIONS_AI`)

A module-level dict organising all 50 primary features into 5 groups:

| Group           | Features | Purpose                                         |
|-----------------|----------|-------------------------------------------------|
| `MOMENTUM`      | 12       | RSI, MACD, ROC, log-returns, oscillator         |
| `VOLATILITY`    | 8        | ATR, Bollinger Bands, Keltner Channels          |
| `PRICE_ACTION`  | 15       | Candlestick geometry, pattern flags, slopes     |
| `MICRO_STRUCTURE` | 5      | Tick velocity, RVOL, spread, order-flow         |
| `CONTEXT`       | 10       | Session flags, cyclical time encoding           |

**Total**: 50 primary features. An additional 8 DERIVED columns are computed
for gate thresholds and eligibility helpers but are not fed to the model.

### D. The Master Pipeline (`transform`)

```text
TRANSFORM PIPELINE

  bars: pd.DataFrame (OHLCV, DatetimeIndex)
  ticks: pd.DataFrame | None (bid, ask)
          │
          ├── Guard: bars.empty → log [%] warning, return empty DataFrame
          │
          ├── Normalize index to M1 DatetimeIndex (resample + ffill)
          │
          ├── Guard: missing required columns → log [!] critical, raise ValueError
          │
          ├── 1. _compute_price_action(fe)     [gated: feat_price_action_enabled]
          │
          ├── 2. _compute_momentum(fe)         [gated: feat_momentum_enabled]
          │
          ├── 3. _compute_volatility(fe)       [gated: feat_volatility_enabled]
          │
          ├── 4. _compute_micro_structure(fe, ticks, freq)
          │       [gated: feat_micro_enabled AND ticks is not None]
          │       → zero-fill if disabled or ticks=None
          │
          ├── 5. _compute_context(fe)          [gated: feat_context_enabled]
          │
          ├── 6. _compute_derived(fe)          [always runs — gate thresholds]
          │
          └── dropna()  → log [^] debug if rows dropped (rolling warmup)
                  │
                  ▼
          pd.DataFrame: full feature matrix (M1 aligned, dropna-clean)
```

### E. The Live Inference Gate (`get_latest`)

```text
GET_LATEST FLOW

  bars, ticks
      │
      ├── transform(bars, ticks)   → raises FeatureEngineerError on failure
      │
      ├── Guard: empty DataFrame  → raises FeatureEngineerError
      │
      ├── Select FEATURE_SET_BINARY_OPTIONS_AI columns (preserving order)
      │
      ├── Take iloc[-1] (latest bar)
      │
      ├── Guard: NaN / Inf in vector → log [%] warning, replace with 0.0
      │
      └── Return FeatureVector(timestamp, vector, feature_names, _VERSION)
```

### E2. The Batch Training Builder (`build_matrix`)

```text
BUILD_MATRIX FLOW

  bars, symbol, ticks=None
      │
      ├── transform(bars, ticks)   → wraps any exception as FeatureEngineerError(stage="build_matrix")
      │
      ├── Guard: empty DataFrame  → raises FeatureEngineerError(stage="build_matrix")
      │
      ├── Select FEATURE_SET_BINARY_OPTIONS_AI columns available in output
      │
      ├── Cast to float32 numpy array of shape (N, F)
      │
      └── Return FeatureMatrix(timestamps, matrix, feature_names, _VERSION, symbol)
```

`build_matrix()` runs the same `transform()` pipeline as `get_latest()` but packages
the **entire** output DataFrame as an immutable `FeatureMatrix` rather than taking
the last row. Use it for offline training data preparation — never for live inference
(use `get_latest()` for that).

The `symbol` argument is stored verbatim in the matrix metadata so that
`trainer.py` can label training batches without needing to re-derive the
instrument from the bar data.

### F. The Eligibility Gate Evaluator (`evaluate_gates`)

Evaluates four scalar conditions against the latest row of a feature DataFrame.
All four gates must pass for `is_eligible=True`:

| Gate                           | Threshold         | Meaning                       |
|--------------------------------|-------------------|-------------------------------|
| `BB_WIDTH > BB_WIDTH_MA_20`    | rolling 20-bar MA | Volatility is expanding       |
| `ATR_14 > ATR_14_MA_20`        | rolling 20-bar MA | Range is expanding            |
| `RVOL > gate_min_rvol`         | config (default 1.5) | Volume surge detected      |
| `SPREAD_NORMALIZED < gate_max_spread` | config (default 0.0005) | Spread is tight  |

Gate thresholds for `RVOL` and `SPREAD_NORMALIZED` are read from
`self._settings` so they can be tuned without code changes.

### G. The Singleton (`get_feature_engineer`)

`get_feature_engineer()` returns the module-level `_engineer` instance,
creating it lazily on first call. The engine layer calls this once at
startup (before spawning worker threads) to pre-warm the singleton and
ensure `get_settings()` runs on the main thread.

---

## 4. GUARDRAIL REGISTRY

### I. Empty Bars Guard

`transform()` checks `bars.empty` before any processing and returns an
empty `pd.DataFrame()` immediately with a `[%]`-bordered warning. Callers
can always test `if df.empty:` without a `None` guard.

### II. Missing Column Guard

After resampling, `transform()` checks for the required `{open, high, low,
close, volume}` columns. A `[!]`-bordered CRITICAL block is logged before
raising `ValueError`. This is a hard caller contract violation — the engine
must not pass malformed DataFrames.

### III. Rolling Warmup Drop Logging

The `dropna()` call after all group computations may drop the first ~26 bars
(MACD uses a 26-period span). This is expected and not an error. Dropped rows
are logged at `DEBUG` with a `[^]` prefix so the INFO stream is not polluted
in production.

### IV. Non-Finite Value Guard

`get_latest()` checks `np.isfinite(values)` before constructing `FeatureVector`.
Any NaN or Inf value is replaced with `0.0` and logged at `WARNING` with a
`[%]` prefix. Replacing rather than raising preserves live-inference continuity
while surfacing the anomaly for investigation.

### V. Fail-Loud `get_latest`

`get_latest()` raises `FeatureEngineerError` (not returns `None`) on all
failure paths. The engine layer is responsible for deciding whether to retry,
skip the signal, or halt. Silent data loss is never acceptable at the inference
gate.

### VI. MICRO_STRUCTURE Zero-Fill

When `feat_micro_enabled=False` or `ticks=None`, all five
`MICRO_STRUCTURE` columns are explicitly set to `0.0` so that the downstream
model always receives the same column schema. This prevents silent schema
drift when ticks are unavailable.

---

## 5. FEATURE GROUP DETAILS

### Group 1: PRICE_ACTION

| Feature                    | Formula                                           | Purpose                        |
|----------------------------|---------------------------------------------------|--------------------------------|
| `BODY_TO_RANGE_RATIO`      | `abs(close-open) / (high-low)`                    | Candle strength                |
| `UPPER_WICK_TO_BODY_RATIO` | `upper_wick / body_abs`                           | Rejection at highs             |
| `LOWER_WICK_TO_BODY_RATIO` | `lower_wick / body_abs`                           | Rejection at lows              |
| `DOJI_BINARY`              | `BODY_TO_RANGE_RATIO < 0.1`                       | Indecision pattern             |
| `MARUBOZU_BINARY`          | `BODY_TO_RANGE_RATIO > 0.9`                       | Strong directional candle      |
| `ENGULFING_BINARY`         | Bullish or bearish body engulfment                | Reversal signal                |
| `PINBAR_SIGNAL`            | Long wick ≥ 2× body, short opposite wick          | Rejection pin bar              |
| `TWO_BAR_REVERSAL`         | Direction flip + close beyond prior open          | Two-bar reversal pattern       |
| `THREE_BAR_SLOPE`          | `close.diff(3) / 3`                               | Short-term price slope         |
| `CONSECUTIVE_BULL_BARS`    | Cumsum within bull run                            | Trend persistence              |
| `CONSECUTIVE_BEAR_BARS`    | Cumsum within bear run                            | Trend persistence              |
| `CLOSE_POSITION_IN_CANDLE` | `(close-low) / (high-low)`                       | Where close sits in range      |
| `PREVIOUS_CANDLE_DIRECTION`| `sign(prev body)`                                 | Prior bar bias                 |
| `HIGH_LOW_RANGE_NORMALIZED`| `(high-low) / rolling_20_mean(high-low)`          | Range expansion                |
| `CANDLE_POSITION_IN_DAY`   | `minutes_since_midnight / 1440`                   | Intraday position              |

### Group 2: MOMENTUM

| Feature              | Formula                                         | Purpose                   |
|----------------------|-------------------------------------------------|---------------------------|
| `RSI_14`             | Wilder-smoothed RSI (EWM alpha=1/14)            | Overbought / oversold     |
| `CCI_14`             | `(TP - TP_mean) / (0.015 × TP_std)`            | Cycle position            |
| `MACD_VALUE`         | `EMA12 - EMA26`                                 | Trend momentum            |
| `MACD_SIGNAL`        | `EMA9(MACD_VALUE)`                              | Signal line               |
| `MACD_HIST`          | `MACD_VALUE - MACD_SIGNAL`                      | Histogram (divergence)    |
| `ROC_5/10/20`        | `pct_change(n)`                                 | Rate of change            |
| `RETURN_1/3/5`       | `log(close / close.shift(n))`                   | Log returns               |
| `MOMENTUM_OSCILLATOR`| `close - close.shift(10)`                       | Classic momentum          |

### Group 3: VOLATILITY

| Feature                | Formula                                       | Purpose                    |
|------------------------|-----------------------------------------------|----------------------------|
| `ATR_14`               | EWM True Range (alpha=1/14)                   | Absolute volatility        |
| `NATR_14`              | `ATR_14 / close × 100`                        | Normalised volatility      |
| `BB_WIDTH`             | `(upper-lower) / SMA20`                       | Band squeeze detector      |
| `BB_UPPER_DIST`        | `(upper_bb - close) / close`                  | Distance to upper band     |
| `BB_LOWER_DIST`        | `(close - lower_bb) / close`                  | Distance to lower band     |
| `BB_PERCENT_B`         | `(close - lower_bb) / (upper_bb - lower_bb)`  | Position within bands      |
| `KC_WIDTH`             | `(KC_upper - KC_lower) / EMA20`               | Keltner channel width      |
| `RANGE_EXPANSION_RATIO`| `current_range / rolling_20_mean(range)`      | Range expansion ratio      |

### Group 4: MICRO_STRUCTURE

| Feature                 | Formula                                          | Purpose               |
|-------------------------|--------------------------------------------------|-----------------------|
| `TICK_VELOCITY`         | Count of ticks per M1 bar                        | Activity burst        |
| `RELATIVE_VOLUME_RVOL`  | `volume / rolling_20_mean(volume)`               | Volume surge          |
| `SPREAD_NORMALIZED`     | `mean_spread_per_bar / close`                    | Cost-of-trade         |
| `TICK_DELTA_CUMULATIVE` | Rolling 20-bar sum of per-tick signed delta       | Cumulative order flow |
| `ORDER_FLOW_IMBALANCE`  | `buy_ticks / total_ticks - 0.5` (per bar)        | Buy/sell imbalance    |

### Group 5: CONTEXT

| Feature                    | Formula                              | Purpose                   |
|----------------------------|--------------------------------------|---------------------------|
| `SESSION_LONDON`           | `7 ≤ hour ≤ 16` UTC                  | London session flag       |
| `SESSION_NEWYORK`          | `12 ≤ hour ≤ 21` UTC                 | New York session flag     |
| `SESSION_TOKYO`            | `0 ≤ hour ≤ 9` UTC                   | Tokyo session flag        |
| `SESSION_OVERLAP_LONDON_NY`| `12 ≤ hour ≤ 16` UTC                 | Overlap (high liquidity)  |
| `TIME_SINE`                | `sin(2π × frac_hour / 24)`          | Cyclical hour encoding    |
| `TIME_COSINE`              | `cos(2π × frac_hour / 24)`          | Cyclical hour encoding    |
| `DAY_OF_WEEK_SINE`         | `sin(2π × day / 7)`                 | Cyclical day encoding     |
| `DAY_OF_WEEK_COSINE`       | `cos(2π × day / 7)`                 | Cyclical day encoding     |
| `MINUTES_TO_NEWS`          | `90` (placeholder)                   | Economic calendar gate    |
| `HOUR_OF_DAY_NORMALIZED`   | `frac_hour / 24`                     | Linear hour encoding      |

### Group 6: DERIVED (internal gate helpers)

| Feature              | Depends On   | Purpose                                     |
|----------------------|-------------|---------------------------------------------|
| `ROC_5_ACCELERATION` | `ROC_5`     | Second derivative of price (ROC diff)       |
| `ATR_14_RATIO_CHANGE`| `ATR_14`    | Volatility regime shift detector            |
| `MACD_HIST_SLOPE`    | `MACD_HIST` | 3-bar slope for 15-min expiry rule          |
| `BB_WIDTH_MA_20`     | `BB_WIDTH`  | 20-bar MA for volatility gate threshold     |
| `ATR_14_MA_20`       | `ATR_14`    | 20-bar MA for range gate threshold          |
| `SESSION_CONTEXT`    | Sessions    | Aggregate session flag for expiry rule      |
| `RVOL`               | `RELATIVE_VOLUME_RVOL` | Gate alias                     |

---

## 6. MODULE CONSTANTS

| Constant                         | Value                | Rationale                                              |
|----------------------------------|----------------------|--------------------------------------------------------|
| `_EPS`                           | `1e-9`               | Division-by-zero guard; orders of magnitude below any valid price delta. |
| `_RESAMPLE_MAP`                  | `{"M1": "1min", …}` | Maps internal timeframe keys to pandas offset aliases. |
| `_VERSION`                       | `"3.1.0-full-binary"`| Schema version string embedded in every FeatureVector. |
| `FEATURE_SET_BINARY_OPTIONS_AI`  | dict of 5 lists      | The authoritative 50-feature schema fed to the model.  |
| `TOP_WEIGHTED_FEATURES`          | list of 10           | High-signal features for inspection and logging.       |
| `TRADE_ELIGIBILITY_GATES`        | list of 4 strings    | Gate expression strings; evaluated by `evaluate_gates`. |
| `BINARY_EXPIRY_RULES`            | dict of 3 keys       | Expiry-to-feature mapping for `get_expiry_features`.   |

---

## 7. TRADE ELIGIBILITY GATES

```text
GATE EVALUATION FLOW

  fe: pd.DataFrame (output of transform())
          │
          ├── Guard: fe.empty → return TradeEligibility(is_eligible=False)
          │
          ├── row = fe.iloc[-1]
          │
          ├── Gate 1: BB_WIDTH > BB_WIDTH_MA_20     (volatility expanding)
          ├── Gate 2: ATR_14 > ATR_14_MA_20         (range expanding)
          ├── Gate 3: RVOL > settings.gate_min_rvol  (volume surge)
          ├── Gate 4: SPREAD_NORMALIZED < settings.gate_max_spread (tight spread)
          │
          └── is_eligible = all(gate_results.values())
                  │
                  ▼
          TradeEligibility(is_eligible, gate_results, gate_values)
```

Gate thresholds for RVOL and SPREAD are read from `self._settings` and can
be changed in `config.py` without modifying this module.

---

## 8. BINARY EXPIRY RULES

```python
BINARY_EXPIRY_RULES = {
    "1_MIN":  ["TICK_VELOCITY", "RVOL", "BODY_TO_RANGE_RATIO"],
    "5_MIN":  ["ROC_5", "BB_WIDTH", "THREE_BAR_SLOPE"],
    "15_MIN": ["MACD_HIST_SLOPE", "ATR_14", "SESSION_CONTEXT"],
}
```

`get_expiry_features(fe, expiry_key)` resolves aliases (e.g. `"RVOL"` →
`"RELATIVE_VOLUME_RVOL"`) and returns only the columns that are present in
`fe`. Missing columns are logged at `WARNING` with a `[%]` prefix.

---

## 9. COMPARISON WITH OTHER ML ENGINE MODULES

| Concern           | `features.py` (Translator)        | `model.py` (Vault)              |
|-------------------|-----------------------------------|---------------------------------|
| **Input**         | Bar + Tick DataFrames             | Tick / Bar objects              |
| **Output**        | FeatureVector (float32 array)     | Tick / Bar (typed dataclasses)  |
| **Statefulness**  | None (stateless after `__init__`) | DataBuffer (stateful queue)     |
| **Thread safety** | Yes (no shared mutable state)     | Yes (threading.Lock on buffer)  |
| **Error handling**| Raises FeatureEngineerError       | Raises ValueError               |
| **Singleton**     | Yes (`get_feature_engineer()`)    | No                              |
| **Schema version**| `_VERSION = "3.1.0-full-binary"`  | `_SCHEMA_VERSION = 1` (storage) |
