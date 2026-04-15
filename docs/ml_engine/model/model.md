# TRADER-AI CORE: MODELS.PY DESIGN DOCUMENT

VERSION: 1.0.0 | STATUS: FROZEN / PRODUCTION-READY

ROLE: DATA INTEGRITY FIREWALL | VISION: ZERO-POISON | THREAD-SAFE | 100/100 PHYSICALITY

## 1. ARCHITECTURAL PHILOSOPHY

The `models.py` module acts as the "Bouncer" of the system. While `config.py`
validates the environment, `models.py` validates the live reality. It enforces
mathematical physicality on every packet of data. By using __post_init__ hooks,
we ensure that "poisoned" data (negative prices, inverted spreads) is killed
at the doorstep, preventing ML model corruption or financial loss.

## 2. THE DIAGNOSTIC LANGUAGE

Consistent with the Core Registry, models use specific symbols to identify
the nature of data rejection:

| Symbol | Category  | Context / Usage                                              |
|--------|-----------|--------------------------------------------------------------|
| [%]    | LOGIC     | Mathematical violations (Inverted spreads, High < Low).      |
| [!]    | SOURCE    | Data origin errors (Unknown broker ID or bad source string). |

## 3. CORE COMPONENTS

### A. The Atomic Unit (Tick)

The smallest unit of market reality. Represents a single point-in-time snapshot.

* Immutability: frozen=True ensures a tick cannot be altered once captured.
* Validation: Enforces bid > 0, ask > 0, and bid <= ask.
* Standardization: Forces sources into either "TWELVE" or "QUOTEX".

### B. The Aggregated Unit (Bar)

The primary input for Feature Engineering and Technical Analysis.

* Physicality: Enforces that High is the absolute ceiling and Low the floor.
* Integrity: Rejects any bar with negative volume.
* Renaming Logic: Uses open_price to avoid shadowing Python's built-in open().

### C. The Orchestrator (DataBuffer)

A thread-safe gateway between the Live Stream and the Disk/Cloud.

* Batching: Reduces I/O overhead by accumulating ticks before a "flush".
* Concurrency: Uses threading.Lock to support simultaneous streams.
* Memory Safety: Implements a 10x flush_size safety cap to prevent RAM exhaustion.

## 4. GUARDRAIL REGISTRY

### I. Tick Integrity (The Poison Filter)

* Bid/Ask Validity: Both must be > 0. Rejects broker glitches.
* Spread Sincerity: bid <= ask. Prevents impossible mid-price calculations.
* Source Authentication: source must be "TWELVE" or "QUOTEX".

### II. Bar Integrity (The Physicality Filter)

* High/Low Ceiling: high >= low AND high >= {open, close}.
* Volume Non-Negativity: volume >= 0. Protects volume-weighted features.

### III. Buffer Safety (The Resource Filter)

* Positive Threshold: flush_size must be > 0. Prevents infinite flush loops.
* Hardware Protection: len(_data) < (flush_size * 10).
* Logic: If storage hangs, the system crashes to save VPS RAM.

## 5. DATA FLOW & IMMUTABILITY

* Dependency Injection: DataBuffer does not import config.py; it is injected.
* Flat Serialization: Models include to_dict() for Pandas/Parquet compatibility.
* Mid-Price Property: Calculated as (bid + ask) / 2.

## 6. FAIL-FAST SCENARIOS (CRITICAL CRASH)

| Trigger                       | Diagnostic              | Context                     |
|-------------------------------|-------------------------|-----------------------------|
| Broker sends Bid: -1.0        | [%] LOGIC VIOLATION     | Prevents ML Corruption      |
| Unknown Source String         | [!] DATA SOURCE ERROR   | Prevents Filter Failure     |
| Low > High in Candle          | [%] OHLC VIOLATION      | Prevents Indicator Errors   |
| Buffer exceeds 10x capacity   | [%] BUFFER OVERFLOW     | Prevents RAM Exhaustion     |

──────────────────────────────────────────────────────────────────────────────

## "CLEAN DATA IS THE ONLY DATA."
