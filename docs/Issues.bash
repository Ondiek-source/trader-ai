===============================================================================
ISSUE DOC: ARCHITECTURAL REFACTOR FOR LINEAR ML PIPELINE (V2)
===============================================================================

PROBLEM STATEMENT
-------------------------------------------------------------------------------
The system currently interleaved data-fetching, training, and trading. This 
causes high memory spikes, Cloudflare 403 blocks during backfills, and log 
pollution. The system must move to a "Data-as-Code" model where history is 
pre-synced and committed to the repository.

===============================================================================
ISSUE DOC: ARCHITECTURAL REFACTOR - HYBRID DATA-AS-CODE PIPELINE (V3)
===============================================================================

1. SYSTEM TOPOLOGY
-------------------------------------------------------------------------------
LOCAL WORKSTATION: 
 - Runs 'sync-history.sh' to fetch OHLCV bars from TwelveData.
 - Processes bars into Parquet format.
 - Commits files to Git and triggers 'deploy.sh'.

AZURE CONTAINER REGISTRY (ACR):
 - Stores Docker images containing the app + 2 years of Parquet data.

AZURE COMPUTE (ACI/VPS):
 - Pulls the image from ACR.
 - Phase 1: Loads local Parquet data.
 - Phase 2: Trains ML Ensemble (LGBM, XGB, RF, Torch).
 - Phase 3: Connects to Quotex and begins trading.

2. THE LINEAR INITIALIZATION CHECKLIST
-------------------------------------------------------------------------------
[STAGE 0] BUILD-TIME SYNC (Local Only)
- Script: 'deploy/sync-history.sh'
- Action: Fetch delta from TwelveData -> Update 'data/*.parquet' -> Git Commit.
- Goal: Ensure repo always contains a sliding window of 730 days.

[STAGE 1] DATA VERIFICATION (Azure Runtime)
- Component: 'src/core/pipeline.py'
- Action: Scan '/app/data' for the committed Parquet files.
- Report once: {"event": "data_verified", "days": 730, "source": "build_context"}

[STAGE 2] COMPUTE-INTENSIVE TRAINING (Azure Runtime)
- Component: 'src/ml_engine/trainer.py'
- Action: Execute Walk-Forward validation and training using Azure CPU/RAM.
- Report once: {"event": "training_complete", "accuracy": 0.58, "duration": "180s"}

[STAGE 3] EXECUTION (Azure Runtime)
- Component: 'src/trading/stream.py'
- Action: Established live WebSocket connection.
- Report once: {"event": "system_ready", "status": "MONITORING_FOR_SIGNALS"}

3. REVISED PROJECT STRUCTURE
-------------------------------------------------------------------------------
├── 📁 data                 <-- COMMITTED TO GIT (The 2-year history)
├── 📁 models               <-- EXCLUDED FROM GIT (Generated in Azure)
├── 📁 src
│   ├── 📁 core
│   │   ├── 📄 config.py    <-- SINGLE SOURCE OF TRUTH (All constants here)
│   │   ├── 📄 logger.py    # Standardized JSON logging
│   │   └── 📄 storage.py   # Local/Azure storage abstraction
│   ├── 📁 data_engine
│   │   ├── 📄 sync.py      # Script used locally to update 'data/'
│   │   └── 📄 features.py  # Feature/Indicator calculation
│   ├── 📁 ml_engine
│   │   ├── 📄 trainer.py   # Logic for the Azure training phase
│   │   └── 📄 model.py     # Model architectures and inference
│   ├── 📁 trading
│   │   ├── 📄 stream.py    # WebSocket connectivity
│   │   └── 📄 signals.py   # Signal logic and Martingale
│   └── 📄 main.py          # Orchestrator (Implements the Checklist)
└── 📄 Dockerfile           # Must include: COPY data/ /app/data/

4. CONFIGURATION ISOLATION (src/core/config.py)
-------------------------------------------------------------------------------
To stop constants from scattering, 'config.py' must be the only file that
reads environment variables.

KEY CONSTANTS TO ISOLATE:
- DATA_PATH = "/app/data"
- MODEL_PATH = "/app/models"
- TRAINING_WINDOW_DAYS = 730
- CONFIDENCE_THRESHOLD = 0.53
- MAX_RF_ROWS = 50000 (Adjust based on Azure RAM)

5. LOGGING PROTOCOL
-------------------------------------------------------------------------------
Stop "looping" logs. Switch to "Milestone" logging:
- { "stage": "BOOT", "status": "DATA_LOADED" }
- { "stage": "TRAIN", "status": "MODELS_OPTIMIZED" }
- { "stage": "TRADE", "status": "CONNECTED" }
===============================================================================

Why this solves your current issues:
Saves TwelveData Credits: You only fetch the "gap" between your last commit and today,
rather than the full 2 years every time you deploy.

Solves the 403 Forbidden Error: Since the bulk of the data is inside the 
Docker image, the Azure container doesn't need to make massive API calls at startup.

Resource Allocation: You don't strain your local PC with training, but you 
also don't pay Azure to download/convert data—you only pay Azure for the high-value training time.

Reliability: By verifying data on the local side before committing, 
you ensure the Azure instance never starts with "corrupt" or "missing" history.

Immediate Next Steps to reach 100/100:
Refactor config.py: Make it the "Dictator" of all settings.
Create src/core/pipeline.py: Write the logic that checks the 4 Gates at startup.
Local Sync Script: Modify your backfill.py logic to run locally and save to the data/ folder 
instead of trying to save to Azure Blobs immediately.
Update Dockerfile: Ensure the data/ folder is included in the build so Azure doesn't have to 
download anything.

================================================================================
  NOTE: MODEL FILE STRATEGY -- DECISION PENDING
  Created: 2026-04-12
================================================================================

  DECISION REQUIRED AFTER FIRST TRAINING RUN:

  1. Run training and check model sizes:
       models/*.pkl  -- check size in MB

  2. If files are under 50MB:
       Option: Remove models/ from .gitignore and commit normally.

  3. If files are over 50MB:
       Option: Keep models/ in .gitignore forever.
       Use Git LFS:
         git lfs track "models/*.pkl"
         git lfs track "models/*.joblib"
       GitHub free tier: 1GB LFS storage.

  REGARDLESS OF SIZE -- PRODUCTION DEPLOY FLOW:
  -----------------------------------------------
  Models are NEVER pushed to ACR via git.
  deploy.sh must handle the upload directly:

    Step 1: Upload models from local to Azure Blob
              az storage blob upload-batch \
                --source models/ \
                --destination traderai \
                --connection-string $AZURE_STORAGE_CONN

    Step 2: ACR build / container pull from Azure Blob at startup
              storage.py already handles the download side.

  TODO: Add the az storage blob upload-batch call to deploy.sh
        before the docker build step.

  REMINDER: models/ and .env are in .gitignore -- keep it that way.

================================================================================

THREAD SAFETY & OOM RISK AUDIT SUMMARY

================================================================================

Thread Safety & OOM Risk Audit Report
CRITICAL (4 issues — process crash risk)
#	File	Lines	Issue
C1	main.py	~332-334	recent_ticks list grows to 15K items per pair, trimmed via O(n) slice — creates 2× memory spike on every overflow, no lock between check and trim
C2	main.py	~309, 455-460	feature_history per pair — same slice-trim anti-pattern, grows unbounded between trim checks under fast tick arrival
C3	data/storage.py	~688-691	_atomic_upsert calls pd.read_parquet (no row limit) then pd.concat — both DataFrames live in RAM simultaneously; a 2-year M1 file = ~400 MB × 2 during every write
C4	engine/live.py	~481-495	_buffers[pair] list has no hard upper bound — if a flush fails (Azure timeout, disk full) the buffer is never cleared and grows forever
Fix for C1, C2, C4: Replace every list buffer with collections.deque(maxlen=N). Append becomes O(1) with automatic eviction — no manual trim, no race window.

HIGH (6 issues)
#	File	Lines	Issue
H1	data/storage.py	get_bars()	Reads entire Parquet file into RAM before applying max_rows tail-slice — a 5M-row file is fully loaded even when caller requests 50K rows
H2	engine/live.py	~103, 481	threading.Lock acquired inside an asyncio coroutine — blocks the event loop thread; if another OS thread waits on the same lock while the event loop is blocked, deadlock
H3	ml_engine/sequence_generator.py	~430-437	DataLoader(num_workers>0) on Windows uses multiprocessing spawn — pickles the Dataset; any unpickleable object (file handles, locks) causes a silent worker crash
H4	data/historian.py	backfill loop	Chunk accumulation without a circuit-breaker — a 2-year backfill is ~100 chunks; if saves queue up, 500K+ rows accumulate per symbol (~120 MB)
H5	main.py	~459-460	feature_history trim is not atomic — len() check and list reassignment are two separate operations with no lock, creating a TOCTOU race
H6	core/dashboard.py	StatusStore	Activity log can overflow its 20-item cap under burst events; StatusStore.get() returns a reference to the shared dict (no deep-copy), so callers can mutate shared state while bypassing the lock
Fix for H2: Replace threading.Lock with asyncio.Lock inside the event loop; use threading.Lock only for background OS threads.
Fix for H1, journal.py: Use pq.read_table(...).slice(start, limit) to read only the required rows from Parquet before converting to DataFrame.

MEDIUM (4 issues)
#	File	Issue
M1	ml_engine/model_manager.py	Storage() instantiated on every save() call — in CLOUD mode each call opens a new Azure connection pool and TCP socket; 50 saves = 50 leaked connections
M2	data/storage.py	_lock is global across all symbols — a slow write to EUR_USD blocks a GBP_USD write even though they touch different files
M3	data/journal.py	Same full-read-then-tail pattern as storage.get_bars() — 1M-row journal loaded, 100 rows returned
M4	core/config.py	Error block strings are unbounded — if an env var holds a large value (e.g. a certificate), the error message inherits its full size
LOW (2 issues)
#	File	Issue
L1	ml_engine/features.py	pd.concat called without pre-validating column sets — schema mismatch throws a generic ValueError with no diagnostic context
L2	main.py	Manual gc.collect() call is a symptom marker — indicates the codebase already has recognized GC pressure that the deque fixes above would eliminate
Priority Fix Order

1. deque(maxlen) everywhere (C1, C2, C4, H5)          — one-line fix per buffer
2. Storage singleton / dependency injection (M1)        — prevents connection leak
3. asyncio.Lock in live.py (H2)                        — deadlock prevention
4. Chunked/sliced Parquet reads (C3, H1, M3)           — biggest OOM reduction
5. DataLoader platform guard (H3)                      — Windows deployment safety
6. Historian circuit-breaker (H4)                      — backfill crash prevention
7. StatusStore deep-copy (H6)                          — subtle shared-state bug
8. Per-symbol write locks (M2)                         — throughput improvement
The most impactful single change is items 1 + 4 together: swapping list buffers for deques and chunking the Parquet reads. Those two address all four CRITICAL findings and cut peak RSS by an estimated 60-70% under live trading load.




Golden Prompt - Multiple Symbol Support for QuotexDataStream
text
## Context
I have a trading system where `LiveEngine` creates one `QuotexDataStream` instance per symbol (e.g., EUR_USD, GBP_USD). Currently, each engine creates its own separate Quotex WebSocket connection to the same account.

## Current Behavior
- Each symbol gets its own `QuotexDataStream` instance
- Each instance opens its own WebSocket connection
- Works fine for 1-3 symbols

## Problem
- Multiple WebSocket connections to same account may cause rate limiting or session conflicts
- Resource waste (multiple connections when one could handle all symbols)

## Goal
Refactor `QuotexDataStream` to support multiple symbols with a SINGLE WebSocket connection.

## Requirements

1. **Single connection, multiple subscriptions**
   - One `QuotexDataStream` instance manages all symbols
   - Connect once, subscribe to multiple symbols
   - Price updates for all symbols flow through the same connection

2. **API method to add symbols**
   ```python
   async def add_symbol(self, symbol: str) -> None:
       """Subscribe to price updates for an additional symbol."""
Subscribe generator per symbol

python
async def subscribe(self, symbol: str):
    """Async generator that yields ticks for a specific symbol."""
Symbol mapping

Internal symbol: "EUR_USD"

Quotex OTC symbol: "EURUSD_otc"

Use self._settings.quotex_symbols for conversion

Thread safety

Use asyncio.Queue per symbol to distribute ticks

Or use a single queue with symbol filtering

Expected Usage
python
# In pipeline.py - single QuotexDataStream for all symbols
self._quotex_reader = QuotexDataStream(
    email=settings.quotex_email,
    password=settings.quotex_password,
    practice_mode=settings.practice_mode,
    symbols=settings.pairs  # ['EUR_USD', 'GBP_USD', 'USD_JPY'...]
)

# In LiveEngine - each gets a filtered stream
stream = self._quotex_reader.subscribe(self.symbol)
Current Code Reference
src/engine/quotex_reader.py - Current implementation (single symbol)

src/engine/live.py - _init_stream() method creates per-symbol readers

src/core/config.py - quotex_symbols property handles symbol conversion

Success Criteria
One WebSocket connection for all symbols

Each symbol receives its own price ticks independently

No cross-symbol contamination

Clean shutdown when all symbols unsubscribe

Constraints
Preserve existing subscribe() interface for backward compatibility

Maintain error handling and reconnection logic

Keep logging clear about which symbol each tick belongs to



I want to commit my processed Parquet files as seed data directly into my Git repository so that every container deployment starts with a complete historical dataset without needing to backfill from the API every time.

Please help me:
1. Add `/app/data/processed/*.parquet` to my .gitignore (except a seed folder)
2. Create a `data/seed/` directory in my project root
3. Copy the current processed bar files from my running container to `data/seed/`
4. Write a script `scripts/seed_data.sh` that:
   - Copies seed data to `/app/data/processed/` on container startup
   - Only copies if the processed directory is empty
   - Preserves existing data if present
5. Modify my Dockerfile to copy seed data during build
6. Modify my entrypoint script to restore seed data if no data exists
7. Compress the seed data to reduce repo size (Parquet already compressed, but could gzip)
8. Add a CI check that seed data matches expected schema version

This way:
- New containers start with 404k bars immediately
- No API backfill needed on first boot
- Data is version-controlled with code
- Works even without Azure Blob
- Cheap and fast deployments

## Known Limitations

# Pseudo-code for your orchestrator
if no_active_trade:
    # Check all pairs for signals
    for pair in pairs:
        signal = generator.generate(pair)
        if signal.is_executable():
            # Execute this trade
            await webhook.fire(signal)
            # Wait for result before checking next pair
            result = await reader.get_result(timeout=65)
            # Process result, then continue loop
            break
    # If no trade executed, wait and retry

    
Due to Quotex API constraints:
- No trade ID returned when placing trades
- No asset/symbol in trade history
- No webhook/callback for trade closure

**Impact:** The bot cannot reliably match results to 
signals when trading multiple pairs in parallel. Sequential 
trading (one pair at a time) is required for accurate result detection.