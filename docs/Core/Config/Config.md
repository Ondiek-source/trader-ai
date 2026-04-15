# TRADER-AI CORE: CONFIG.PY DESIGN DOCUMENT

VERSION: 1.0.0 | STATUS: FROZEN / PRODUCTION-READY

ROLE: CENTRALIZED ENVIRONMENT GATEKEEPER | VISION: FAIL-FAST | ZERO-FOOTPRINT | 100/100 CLARITY

## 1. ARCHITECTURAL PHILOSOPHY

The `config.py` module acts as the "Dictator" of the system. It handles the
transition from raw environment strings to a validated, immutable Python object
. By enforcing strict validation at boot, we ensure the system never
enters a "zombie state" with invalid parameters.

## 2. THE DIAGNOSTIC LANGUAGE

Failures trigger specific visual blocks to identify the error source instantly:

## Diagnostic Language Registry

| Symbol | Category  | Context / Usage                                                              |
|--------|-----------|------------------------------------------------------------------------------|
| [!]    | FATAL     | Missing required environment variables or infrastructure mismatches.         |
| [#]    | INTEGER   | Type mismatch where a whole number was expected.                             |
| [~]    | FLOAT     | Type mismatch where a decimal/float was expected.                            |
| [=]    | LIST      | Validation errors for currency pairs or asset whitelists.                    |
| [%]    | LOGIC     | Mathematical violations (e.g., Confidence Threshold out of 0.51-0.99 range). |
| [?]    | TEMPORAL  | Expiry or time-frame errors (e.g., Untested Expiry Sets).                    |
| [^]    | NETWORK   | Infrastructure errors such as out-of-range ports for the dashboard.          |

## 3. CORE COMPONENTS

### A. The Factory (`load_config`)

The single entry point for configuration. It aggregates data from
all specialized helpers, casts them to correct types, and returns the
frozen `Config` dataclass.

### B. The Translator (`@property quotex_symbols`)

Automatically maps "pure" currency names to broker-specific symbols.

* Logic: `EUR_USD` -> `EURUSD_otc`.
* Purpose: Keeps `.env` human-readable while meeting API requirements.

### C. The Gatekeepers (`_parse_and_validate_pairs`)

Strictly compares all requested pairs against the `VALID_PAIRS` whitelist
. If a pair is not supported, the system triggers a CRITICAL
log and exits immediately.

## 4. SETTINGS REGISTRY

### I. Infrastructure & Secrets

* **TWELVEDATA_API_KEY**
  * **What**: Primary 32-character authentication string.
  * **Purpose**: Authenticates calls for real-time and historical price data.
  * **Logic**: Required. Must be active to prevent data stream disconnection.
* **AZURE_STORAGE_CONN**
  * **What**: Full connection string for Azure Blob Storage.
  * **Purpose**: Enables cloud persistence for models and CSV logs.
  * **Logic**: If blank, the system defaults to `LOCAL` mode for file I/O.
* **WEBHOOK_URL**
  * **What**: The target URL for trade signal delivery.
  * **Purpose**: Transmits the "Buy/Sell" command to the execution bridge.
  * **Logic**: Required. Must be a valid HTTPS endpoint to avoid packet loss.
* **QUOTEX_EMAIL**
  * **What**: The Quotex account login ID.
  * **Purpose**: Required for live streaming of price data and trade execution.
  * **Logic**: Required. Must be a valid email address associated with the broker account.
* **QUOTEX_PASSWORD**
  * **What**: The Quotex account password.
  * **Purpose**: Required for authentication to access live market data and execute trades.
  * **Logic**: Required. Must meet the broker's password complexity requirements.

### II. Trading Logic & Strategy

* **PAIRS**
  * **What**: A list of currency pairs to monitor (e.g., `EUR_USD`).
  * **Purpose**: Defines the scope of the market analysis.
  * **Logic**: Validated against `VALID_PAIRS`. Case-insensitive but stored as uppercase.
* **CONFIDENCE_THRESHOLD**
  * **What**: A decimal percentage (e.g., `0.53`).
  * **Purpose**: Acts as the quality filter for ML signals.
  * **Logic**: Must be `0.51 < x < 0.99`. Prevents random (0.50) or overfitted (1.00) trades.
* **EXPIRY_SECONDS**
  * **What**: Trade duration in seconds (e.g., `60`).
  * **Purpose**: Aligns the trade execution with the ML model's prediction window.
  * **Logic**: Restricted to `{60, 120, 300}` to ensure model-to-market synchronization.
* **DAILY_NET_PROFIT_TARGET**
  * **What**: A specific USD amount (e.g., `50.0`).
  * **Purpose**: Locks in financial gains for the session.
  * **Logic**: Once the account balance delta reaches this value, the bot ceases trading.
* **DAILY_TRADE_TARGET**
  * **What**: Total count of closed trades (e.g., `30`).
  * **Purpose**: Prevents over-trading and "revenge trading" fatigue.
  * **Logic**: Hard stop logic. Once trade #30 closes, the bot shuts down regardless of PNL.

### III. Operational & Resource Management

* **DATA_MODE**
  * **What**: A string toggle (`LOCAL` or `CLOUD`).
  * **Purpose**: Routes all file operations (Save/Load).
  * **Logic**: Influences the path resolution for the Data and Model directories.
* **MAX_RF_ROWS**
  * **What**: Integer cap on training data rows (e.g., `50000`).
  * **Purpose**: Prevents the Random Forest model from exhausting system RAM.
  * **Logic**: Limits the matrix size during the `.fit()` stage of model training.
* **MEMORY_SAVER_MODE**
  * **What**: Boolean toggle.
  * **Purpose**: Optimizes hardware usage for low-spec environments (VPS).
  * **Logic**: If `True`, forces manual garbage collection and deletes temporary dataframes.

### IV. Risk & Reporting

* **PRACTICE_MODE**
  * **What**: Boolean safety switch.
  * **Purpose**: Determines if signals carry financial risk.
  * **Logic**: If `True`, signals are generated but execution is simulated (Demo).
* **MARTINGALE_MAX_STREAK**
  * **What**: Integer limit on recovery steps (e.g., `4`).
  * **Purpose**: Protects account balance from catastrophic "streak" wipes.
  * **Logic**: After `X` losses, the bot resets to base stake instead of doubling again.
* **TELEGRAM_TOKEN / CHAT_ID**
  * **What**: Bot API credentials.
  * **Purpose**: Delivers real-time status updates and profit reports.
  * **Logic**: Optional. If either is missing, the reporting service is silently disabled.

## 5. IMMUTABILITY & SAFETY

* **Frozen Instance**: `@dataclass(frozen=True)` ensures runtime settings cannot be tampered with.
* **Real Return Calculation**: Financial targets are assessed in USD but converted to KES for the final real-value report.
* **Fail-Fast**: The system is designed to "die at the front door" if the `.env` is invalid.

## 6. ENVIRONMENT CONFIGURATION (.env)

## --- REQUIRED SECRETS (System will crash if missing) ---

TWELVEDATA_API_KEY=your_api_key_here
WEBHOOK_URL=your_webhook_url_here
AZURE_STORAGE_CONN=your_azure_connection_string

## --- QUOTEX CREDENTIALS (Required for Live Streaming) ---

## System will raise a FATAL error if these are missing

QUOTEX_EMAIL=<your_email@example.com>
QUOTEX_PASSWORD=your_secure_password

## --- TRADING LOGIC ---

## Supported: EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, USD_CHF, XAU_USD

PAIRS=EUR_USD,GBP_USD

## Range: 0.51 to 0.99

CONFIDENCE_THRESHOLD=0.53

## Options: 60, 120, 300

EXPIRY_SECONDS=60

## Target in USD (Leave blank to disable)

DAILY_NET_PROFIT_TARGET=100
DAILY_TRADE_TARGET=30
TRADING_WINDOW_HOURS=19

## --- OPERATIONAL MODES ---

PRACTICE_MODE=True
LOG_LEVEL=INFO
DATA_MODE=LOCAL
WEBHOOK_KEY=Ondiek

## --- RESOURCE MANAGEMENT ---

MAX_RF_ROWS=50000
MAX_SEQUENCES=100000
TICK_FLUSH_SIZE=500
MEMORY_SAVER_MODE=False

## --- INFRASTRUCTURE ---

DASHBOARD_PORT=8080
MARTINGALE_MAX_STREAK=4
POLL_INTERVAL=1.0

## --- DATA & TRAINING ---

BACKFILL_YEARS=2
BACKFILL_PAIRS=EUR_USD,GBP_USD
CONTAINER_NAME=traderai

## --- OPTIONAL OVERRIDES ---

## Only use if you need broker symbols different from Standard_otc

OTC_PAIRS=
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=

## 7. VERIFICATION & TEST COVERAGE

STATUS: VERIFIED | TEST RUNNER: pytest | TOTAL TESTS: 54 | COVERAGE TARGET: 100%

All validation logic in `config.py` has been verified against a complete
test harness. Tests are organised into 11 groups, each targeting a specific
layer of the module. The full harness is documented in `config_test_harness.txt`.

### Test Group Summary

| Group | File                    | Scope                                     | Tests |
|-------|-------------------------|-------------------------------------------|-------|
| 1     | test_helpers.py         | `_require()` -- required var enforcement  | 4     |
| 2     | test_helpers.py         | `_parse_float()` -- float parsing + symbol| 5     |
| 3     | test_helpers.py         | `_bool()` and `_int()` -- type helpers    | 5     |
| 4     | test_helpers.py         | Pair list parsing and OTC delimiter edge  | 5     |
| 5     | test_validation.py      | Security: webhook key enforcement         | 3     |
| 6     | test_validation.py      | Data integrity: OTC length + cloud mode   | 5     |
| 7     | test_validation.py      | Trading logic: thresholds, expiry, streak | 7     |
| 8     | test_validation.py      | Infrastructure: port range, log level     | 4     |
| 9     | test_load_config.py     | Full integration: required vars + defaults| 8     |
| 10    | test_settings.py        | Singleton: lazy init, caching, fail-fast  | 4     |
| 11    | test_properties.py      | `quotex_symbols` translation property     | 4     |

### Key Behaviours Verified

* `_require()` fails on missing, empty, and whitespace-only values.
* `_parse_float()` emits the `~` diagnostic symbol (not `#`) on type error.
* `_bool()` correctly handles all six truthy string variants.
* `PAIRS` defaults to `EUR_USD` when absent; never crashes on omission.
* `DAILY_NET_PROFIT_TARGET` correctly produces `None` when absent.
* `WEBHOOK_KEY` has no hardcoded default; system fails if URL is set without it.
* `OTC_PAIRS` length mismatch triggers a hard failure before `quotex_symbols`
  is ever called, preventing silent zip() truncation.
* `DATA_MODE=CLOUD` without `AZURE_STORAGE_CONN` is a hard failure.
* `MARTINGALE_MAX_STREAK > 6` is rejected as a risk control.
* `CONFIDENCE_THRESHOLD` boundary checks are strict: 0.50 and 1.00 both fail.
* `EXPIRY_SECONDS` only accepts {60, 120, 300}.
* `Config` dataclass is frozen; mutation after construction raises `FrozenInstanceError`.
* `get_settings()` is a lazy singleton: two calls return the same object.
* Importing `config` with an invalid environment does NOT crash at import time.
* `get_settings()` exits with code 1 on validation failure (fail-fast preserved).

### Running the Suite

    pip install pytest pytest-cov python-dotenv
    pytest tests/ -v --cov=config --cov-report=term-missing

### Shared Fixture (conftest.py)

All integration tests use a `minimal_valid_env` fixture that provides the
smallest set of environment variables required to pass all validation. Individual
tests override one key at a time to exercise each failure branch in isolation.

──────────────────────────────────────────────────────────────────────────────

## "IN CODE WE TRUST, BUT ONLY IF IT IS VALIDATED."
