# TRADER-AI ML ENGINE: LABELER.PY DESIGN DOCUMENT

VERSION: 1.0.0 | STATUS: FROZEN / PRODUCTION-READY

ROLE: THE GROUND TRUTH | VISION: RAW BARS → TRAINING LABELS + RL REWARDS | STATELESS

---

## 1. ARCHITECTURAL PHILOSOPHY

The `labeler.py` module acts as "The Ground Truth" of the ML engine. Where
`features.py` owns the mathematical bridge from raw bars to feature vectors,
the Ground Truth owns the authoritative classification of what actually happened:
did the price rise or fall at expiry? That binary answer is the Y the models are
trained to predict.

The module is split along a deliberate boundary:

**Labeler** — pre-compute static Y labels over a historical batch. Runs offline
before training. Produces a `pd.Series` aligned to a `FeatureMatrix` for use by
DataShaper in supervised (Classical ML, Deep Learning) pipelines.

**RewardCalculator** — compute a scalar reward at each RL environment step.
Runs dynamically at training time. Wraps the same price-direction logic as
Labeler but is called one step at a time, not pre-computed over a batch.

This split is mandatory. Mixing static labels with dynamic rewards would couple
Phase 1/2 training pipelines to RL abstractions — and vice versa.

The module operates under three design principles:

**The Binary Principle**: Labels are always 0 or 1. SKIP is not a label — it is
an inference-time decision made by the model when its confidence falls below a
threshold. The Labeler only produces the ground-truth binary outcome.

**The Version-Safety Principle**: Every label batch carries a `feature_version`
field (mirroring `_VERSION` from features.py). If features.py is updated and
`_VERSION` bumps, all cached label sets are stale and must be regenerated. The
trainer must validate this field before using any cached labels.

**The Alignment Principle**: The returned `pd.Series` from `compute_labels()` has
the same DatetimeIndex as the input DataFrame minus the final `lookahead_bars`
rows. Callers must align the `FeatureMatrix` timestamps to this truncated index
before passing both to DataShaper.

---

## 2. THE DIAGNOSTIC LANGUAGE

Consistent with the Core Diagnostic Registry:

| Symbol | Category | Context / Usage                                                     |
|--------|----------|---------------------------------------------------------------------|
| `[!]`  | FATAL    | Invalid expiry key, missing "close" column, invalid action value.   |
| `[%]`  | LOGIC    | Too few bars for lookahead window; all rows dropped; bad payout.    |
| `[^]`  | INFO     | Successful construction, labels produced, reward computed.          |

---

## 3. CORE COMPONENTS

### A. The Custom Exception (`LabelerError`)

Distinct from `ValueError` (caller contract violation) and from
`FeatureEngineerError` (feature pipeline failure). `LabelerError` signals a
runtime failure inside the labeling pipeline that the trainer must handle.

Carries a `stage` attribute identifying which pipeline step failed:

| Stage | Trigger |
|-------|---------|
| `"compute_labels"` | Empty DataFrame, too few bars, all rows dropped after NaN removal |

`ValueError` from missing "close" column or invalid arguments propagates unchanged —
it signals a caller contract violation, not an internal pipeline failure.

### B. The Expiry Taxonomy (`_EXPIRY_SECONDS`)

The authoritative mapping from expiry key to duration in seconds:

| Key | Seconds | Lookahead Bars (M1) |
|-----|---------|---------------------|
| `"1_MIN"` | 60 | 1 |
| `"5_MIN"` | 300 | 5 |
| `"15_MIN"` | 900 | 15 |

`BINARY_EXPIRY_RULES` in `features.py` maps the same keys to feature column
lists — it does **not** store durations. These two dicts are complementary. If
a new expiry key is added to `BINARY_EXPIRY_RULES`, it must be added here too.

`lookahead_bars` is derived as `expiry_seconds // _M1_BAR_SECONDS` (integer
division). For M1 bars: 60 // 60 = 1, 300 // 60 = 5, 900 // 60 = 15.

### C. `Labeler` — Static Label Generator

```text
COMPUTE_LABELS PIPELINE

  df: pd.DataFrame (with "close" column, DatetimeIndex)
      │
      ├── Guard: df.empty → raise LabelerError(stage="compute_labels")
      │
      ├── Guard: "close" not in df → raise ValueError
      │
      ├── Guard: lookahead_bars >= len(df) → raise LabelerError
      │
      ├── future_close = df["close"].shift(-lookahead_bars)
      │       shift(-N) aligns the close N bars ahead with the current index
      │
      ├── raw_labels = (future_close > df["close"]).astype(np.int32)
      │       1 = CALL win (price rose strictly)
      │       0 = PUT win or neutral (price fell or was flat)
      │
      ├── labels = raw_labels.dropna().rename("label")
      │       removes the final lookahead_bars rows where future_close is NaN
      │
      ├── Guard: labels.empty → raise LabelerError(stage="compute_labels")
      │
      └── Return pd.Series (int32, name="label", len = len(df) - lookahead_bars)
```

The final `lookahead_bars` rows are always dropped. Training on these rows would
introduce NaN-derived corruption because no observable outcome exists.

**Output contract**:
- dtype: `int32`
- values: `{0, 1}` only
- name: `"label"`
- length: `len(df) - lookahead_bars`
- index: DatetimeIndex matching the first `len(df) - lookahead_bars` rows of `df`

### D. `Labeler.get_metadata` — Versioned Provenance Snapshot

Produces a 10-field dict that must travel alongside every label set:

| Key | Type | Description |
|-----|------|-------------|
| `feature_version` | str | `_VERSION` from features.py — validate before using cached labels |
| `expiry_key` | str | e.g. `"5_MIN"` |
| `expiry_seconds` | int | e.g. 300 |
| `lookahead_bars` | int | e.g. 5 |
| `symbol` | str | e.g. `"EUR_USD"` |
| `label_generated_at` | str | UTC ISO-8601 timestamp of generation |
| `total_rows` | int | Number of labeled bars |
| `call_count` | int | Number of CALL labels (1) |
| `put_count` | int | Number of PUT labels (0) |
| `call_pct` | float | Fraction of CALL labels (0.0–1.0, 4 d.p.) |

Raises `ValueError` if passed an empty label Series — metadata for an empty
label set is meaningless and indicates a pipeline ordering error.

### E. `RewardCalculator` — RL Reward Signal

Called once per RL environment step. Computes a scalar reward from three inputs:

| Input | Values | Meaning |
|-------|--------|---------|
| `action` | 0, 1, 2 | 0=CALL, 1=PUT, 2=SKIP |
| `is_correct` | bool | True if direction matched price movement at expiry |
| `gate_passed` | bool | True if all four TradeEligibility gates passed |

```text
CALCULATE_REWARD FLOW

  action, is_correct, gate_passed
      │
      ├── Guard: action not in {0, 1, 2} → raise ValueError
      │
      ├── action == 2 (SKIP) → return 0.0 immediately
      │       no gate adjustment, no direction reward
      │
      ├── base_reward = payout_ratio if is_correct else -1.0
      │
      ├── gate adjustment (always applied):
      │       gate_passed=True  → reward += 0.1
      │       gate_passed=False → reward -= 0.2
      │
      └── return float(reward)
```

**Reward table** (payout_ratio = 0.85):

| Action | Correct | Gate | Reward | Rationale |
|--------|---------|------|--------|-----------|
| SKIP | — | — | 0.00 | No trade, no consequence |
| CALL/PUT | True | True | 0.95 | Win + quality bonus |
| CALL/PUT | True | False | 0.65 | Win but traded bad setup |
| CALL/PUT | False | True | -0.90 | Loss, at least quality was good |
| CALL/PUT | False | False | -1.20 | Loss + traded bad setup |

The gate compliance adjustment is applied **on top of** the direction reward,
even when the direction was correct. This teaches the agent to avoid trading in
low-quality conditions regardless of lucky outcomes.

`payout_ratio` must be in `(0.0, 1.0]`. The range `(0.70, 0.92)` covers typical
real-world broker payouts for Quotex-style binary options.

---

## 4. GUARDRAIL REGISTRY

### I. Empty DataFrame Guard

`compute_labels()` checks `df.empty` before any processing and raises
`LabelerError(stage="compute_labels")` immediately. An empty DataFrame means
the historical data pipeline failed to deliver bars.

### II. Missing "close" Column Guard

`compute_labels()` checks for the `"close"` column and raises `ValueError` if
absent. This is a caller contract violation — the feature engineering pipeline
guarantees "close" exists in its output.

### III. Lookahead Overflow Guard

If `lookahead_bars >= len(df)`, there is not a single row with an observable
outcome. Raises `LabelerError(stage="compute_labels")` rather than returning an
empty series, to fail loudly at the training data preparation stage.

### IV. All-Rows-Dropped Guard

After `dropna()`, if `labels.empty`, all rows were consumed by the NaN removal.
Raises `LabelerError(stage="compute_labels")`. This guard catches edge cases
where `lookahead_bars == len(df) - 1` (only one row would survive, but the
`shift()` NaN pattern still removes it).

### V. Empty Labels Metadata Guard

`get_metadata()` raises `ValueError` if passed an empty Series. Metadata for an
empty label set is logically incoherent and indicates a pipeline ordering error.

### VI. Invalid Action Guard

`calculate_reward()` raises `ValueError` for any `action` outside `{0, 1, 2}`.
The RL environment's action space is `Discrete(3)` — any value outside this
range is a programming error in the environment definition.

### VII. Payout Ratio Range Guard

`RewardCalculator.__init__()` raises `ValueError` if `payout_ratio` is not in
`(0.0, 1.0]`. Zero or negative payouts are economically meaningless; values
above 1.0 exceed the theoretical maximum binary option payout.

### VIII. Taxonomy Sync Guard

`Labeler.__init__()` validates that `expiry_key` exists in **both**
`BINARY_EXPIRY_RULES` (features.py) and `_EXPIRY_SECONDS` (this file). This
catches any taxonomy drift between files at construction time rather than at
label generation time.

---

## 5. VERSION SAFETY

The `_VERSION` constant from `features.py` is embedded in every metadata dict
via `get_metadata()`. The trainer must execute the following check before using
any cached label set:

```python
if metadata["feature_version"] != _VERSION:
    raise RuntimeError(
        f"Stale label set: generated with features {metadata['feature_version']}, "
        f"current version is {_VERSION}. Regenerate labels."
    )
```

Failing to perform this check can silently couple a label set generated under
one feature schema to a model trained with a different schema — producing
corrupted training data with no runtime error.

---

## 6. COMPARISON WITH OTHER MODULES

| Concern | `features.py` | `labeler.py` | `journal.py` |
|---------|---------------|--------------|--------------|
| **Role** | Translator | Ground Truth | Decision Ledger |
| **Input** | OHLCV bars + ticks | OHLCV bars | Inference results |
| **Output** | FeatureVector / FeatureMatrix | pd.Series (labels) / float (reward) | TradeEntry / SignalEntry |
| **Error type** | FeatureEngineerError | LabelerError | JournalError |
| **State** | Stateless | Stateless | Stateful (file I/O) |
| **Phase** | All phases (1, 2, 3) | Phase 1+2 (Labeler), Phase 3 (RewardCalculator) | All phases |
