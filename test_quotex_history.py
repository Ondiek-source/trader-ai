"""
test_quotex_history.py — Integration test for QuotexReader history matching.

Connects using the same QuotexReader.connect() path your bot uses.
Reads live history and runs the match logic against it.

Install:
    pip install git+https://github.com/cleitonleonel/pyquotex.git python-dotenv

Run:
    python test_quotex_history.py

Required env vars (.env or shell export):
    QUOTEX_EMAIL
    QUOTEX_PASSWORD
    QUOTEX_PRACTICE_MODE   (optional, default "true")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# ── Load .env if present ───────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # no dotenv — rely on shell env vars

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("quotex_test")

# ── pyquotex import ────────────────────────────────────────────────────────────
try:
    from pyquotex.stable_api import Quotex as _QuotexClient
except ImportError as e:
    sys.exit(
        f"[FATAL] pyquotex not installed.\n"
        f"  pip install git+https://github.com/cleitonleonel/pyquotex.git\n"
        f"  Original error: {e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — raw history probe
# Prints every field returned by get_history() so you can see exactly what
# Quotex sends. Run once to calibrate field names / value types.
# ══════════════════════════════════════════════════════════════════════════════


async def probe_raw_history(client: Any) -> list[dict]:
    print("\n" + "=" * 60)
    print("SECTION 1 — RAW HISTORY PROBE")
    print("=" * 60)

    try:
        history = await asyncio.wait_for(client.get_history(), timeout=10.0)
    except asyncio.TimeoutError:
        print("  TIMEOUT — get_history() did not respond within 10 s")
        return []

    if not history:
        print("  EMPTY — get_history() returned nothing")
        return []

    if not isinstance(history, list):
        print(f"  UNEXPECTED TYPE: {type(history)}")
        print(f"  repr: {repr(history)[:400]}")
        return []

    print(f"  Total trades in history : {len(history)}")

    # ── All field names across every trade ────────────────────────────────────
    all_keys: set[str] = set()
    for t in history:
        if isinstance(t, dict):
            all_keys.update(t.keys())
    print(f"\n  Fields present (across all trades):")
    for k in sorted(all_keys):
        example = next((str(t.get(k))[:60] for t in history if k in t), "—")
        print(f"    {k:<25}  e.g. {example}")

    # ── First trade — full dump ───────────────────────────────────────────────
    print(f"\n  First trade (full):")
    print(json.dumps(history[0], indent=4, default=str))

    # ── Last 5 trades — summary ───────────────────────────────────────────────
    print(f"\n  Last 5 trades (summary):")
    for t in history[-5:]:
        direction = t.get("direction") or t.get("type") or "?"
        profit = t.get("profitAmount", t.get("profit", "MISSING"))
        print(
            f"    ticket={str(t.get('ticket','?'))[:36]:<38} "
            f"asset={str(t.get('asset','?')):<16} "
            f"dir={str(direction):<5} "
            f"close_time={t.get('close_time','?')}  "
            f"profit={profit}"
        )

    return history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — asset normaliser (mirrors _assets_match in QuotexReader)
# ══════════════════════════════════════════════════════════════════════════════


def _normalise(s: str) -> str:
    return s.upper().replace("_OTC", "").replace("-", "").replace("/", "").strip()


def assets_match(signal_pair: str, trade_asset: str) -> bool:
    if not trade_asset:
        return False
    return _normalise(signal_pair) == _normalise(trade_asset)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — pure match function (mirrors _match_from_history exactly)
# ══════════════════════════════════════════════════════════════════════════════


def match_trade(
    history: list[dict],
    pair: str,
    direction: str,
    expiry_time: datetime,
) -> dict[str, Any]:
    """
    Pure function — no I/O. Returns a result dict always:
      outcome  = "win" | "loss" | "draw" | None
      reason   = why outcome is None (if applicable)
      debug    = diagnostics for every filter stage
    """
    target_time = expiry_time.replace(tzinfo=None)

    closest_trade = None
    smallest_diff = float("inf")
    skipped_asset = 0
    skipped_direction = 0
    skipped_time = 0
    candidates: list = []

    for trade in history:
        # filter 1: asset
        trade_asset = trade.get("asset", "")
        if not assets_match(pair, trade_asset):
            skipped_asset += 1
            continue

        # filter 2: direction
        trade_direction = trade.get("direction", trade.get("type", "")).lower()
        if trade_direction and trade_direction != direction.lower():
            skipped_direction += 1
            continue

        # filter 3: close_time present and parseable
        close_time_str = trade.get("close_time", "")
        if not close_time_str:
            continue
        try:
            trade_time = datetime.strptime(close_time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        time_diff = abs((trade_time - target_time).total_seconds())
        candidates.append(
            {
                "ticket": trade.get("ticket"),
                "asset": trade_asset,
                "direction": trade_direction,
                "close_time": close_time_str,
                "time_diff": round(time_diff, 3),
                "profit_raw": trade.get("profitAmount", trade.get("profit")),
            }
        )

        # filter 4: within 66 s window
        if time_diff < 66 and time_diff < smallest_diff:
            smallest_diff = time_diff
            closest_trade = trade
        else:
            skipped_time += 1

    debug = {
        "total_trades": len(history),
        "skipped_asset_mismatch": skipped_asset,
        "skipped_direction_mismatch": skipped_direction,
        "skipped_outside_window": skipped_time,
        "candidates_after_filters": candidates,
        "matched_ticket": closest_trade.get("ticket") if closest_trade else None,
        "time_diff_seconds": round(smallest_diff, 3) if closest_trade else None,
    }

    if not closest_trade:
        return {"outcome": None, "reason": "NO_TRADE_NEAR_EXPIRY", "debug": debug}

    if smallest_diff > 60:
        return {"outcome": None, "reason": "TRADE_MATCH_OUTSIDE_WINDOW", "debug": debug}

    # safe profit extraction — None check BEFORE float() cast
    profit_raw = closest_trade.get("profitAmount", closest_trade.get("profit"))

    if profit_raw is None:
        return {"outcome": None, "reason": "TRADE_PROFIT_MISSING", "debug": debug}

    profit = float(profit_raw)

    if profit > 0:
        outcome, payout, stake = "win", profit, 0.0
    elif profit < 0:
        outcome, payout, stake = "loss", 0.0, abs(profit)
    else:
        outcome, payout, stake = "draw", 0.0, 0.0

    return {
        "outcome": outcome,
        "payout": payout,
        "stake": stake,
        "profit": profit,
        "ticket": closest_trade.get("ticket"),
        "asset": closest_trade.get("asset"),
        "direction": closest_trade.get("direction") or closest_trade.get("type"),
        "close_time": closest_trade.get("close_time"),
        "debug": debug,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — known trade test cases
# Add your real trades here. The test verifies ticket + outcome for each.
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_TRADES = [
    {
        "label": "USD/PKR OTC — 5 s trade",
        "pair": "USD/PKR",
        "direction": "put",
        # broker returns "2026-04-17 22:44:27" (drops ms), so diff will be ~0.47 s
        "expiry_time": datetime(2026, 4, 17, 22, 44, 27, 474000),
        "expected_ticket": "52a1eccd-45e3-41aa-9dec-26286e361e05",
        # set expected_outcome to "win"/"loss"/"draw" once you know it, or None
        "expected_outcome": None,
    },
    # add more as you observe live trades:
    # {
    #     "label":            "EUR/USD OTC — win",
    #     "pair":             "EUR/USD",
    #     "direction":        "call",
    #     "expiry_time":      datetime(2026, 4, 17, 20, 0, 0),
    #     "expected_ticket":  "your-ticket-uuid",
    #     "expected_outcome": "win",
    # },
]


async def run_match_tests(history: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("SECTION 4 — MATCH TESTS AGAINST KNOWN TRADES")
    print("=" * 60)

    if not history:
        print("  No history available — skipping match tests")
        return

    all_passed = True

    for tc in KNOWN_TRADES:
        print(f"\n  >> {tc['label']}")
        print(
            f"     pair={tc['pair']}  direction={tc['direction']}  "
            f"expiry={tc['expiry_time']}"
        )

        result = match_trade(
            history,
            pair=tc["pair"],
            direction=tc["direction"],
            expiry_time=tc["expiry_time"],
        )

        outcome = result.get("outcome")

        if outcome is None:
            print(f"  FAIL — No match.  reason: {result.get('reason')}")
            d = result["debug"]
            print(f"     total trades       : {d['total_trades']}")
            print(f"     skipped asset      : {d['skipped_asset_mismatch']}")
            print(f"     skipped direction  : {d['skipped_direction_mismatch']}")
            print(f"     skipped time window: {d['skipped_outside_window']}")
            if d["candidates_after_filters"]:
                print(f"     candidates (passed asset+dir filter):")
                for c in d["candidates_after_filters"]:
                    print(f"       {c}")
            all_passed = False
            continue

        # ticket check
        if tc.get("expected_ticket"):
            ticket_ok = str(result.get("ticket")) == str(tc["expected_ticket"])
            label = "PASS" if ticket_ok else "FAIL"
            print(f"     Ticket  [{label}]  got={result.get('ticket')}")
            if not ticket_ok:
                print(f"             expected={tc['expected_ticket']}")
                all_passed = False

        # outcome check
        if tc.get("expected_outcome"):
            outcome_ok = outcome == tc["expected_outcome"]
            label = "PASS" if outcome_ok else "FAIL"
            print(
                f"     Outcome [{label}]  got={outcome}  expected={tc['expected_outcome']}"
            )
            if not outcome_ok:
                all_passed = False
        else:
            print(f"     Outcome [INFO]  {outcome}  (no expected value set)")

        print(f"     Profit  : {result.get('profit')}")
        print(f"     Payout  : {result.get('payout')}")
        print(f"     Stake   : {result.get('stake')}")
        print(f"     Asset   : {result.get('asset')}")
        print(f"     Time D  : {result['debug']['time_diff_seconds']} s")

    print()
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — see output above")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — main: connect the same way QuotexReader.connect() does
# ══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    email = os.environ.get("QUOTEX_EMAIL")
    password = os.environ.get("QUOTEX_PASSWORD")
    practice = os.environ.get("QUOTEX_PRACTICE_MODE", "true").lower() != "false"

    if not email or not password:
        sys.exit(
            "[FATAL] QUOTEX_EMAIL and QUOTEX_PASSWORD must be set.\n"
            "  export QUOTEX_EMAIL=you@example.com\n"
            "  export QUOTEX_PASSWORD=yourpassword\n"
            "  or add them to a .env file in this directory."
        )

    account_type = "PRACTICE" if practice else "REAL"
    print(f"\nConnecting  email={email}  account={account_type} ...")

    # connect exactly as QuotexReader.connect() does
    client = _QuotexClient(email=email, password=password, lang="en")
    ok, reason = await client.connect()

    if not ok:
        sys.exit(f"[FATAL] Connection failed: {reason}")

    print(f"  Connected OK  reason={reason}")

    if hasattr(client, "change_account"):
        result = client.change_account(account_type)
        if asyncio.iscoroutine(result):
            await result

    try:
        balance = await asyncio.wait_for(client.get_balance(), timeout=5.0)
        print(f"  Balance : {balance}")
    except Exception as e:
        print(f"  Balance fetch failed: {e}")

    try:
        history = await probe_raw_history(client)
        await run_match_tests(history)
    finally:
        try:
            result = client.close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass
        print("\nDisconnected.\n")


if __name__ == "__main__":
    asyncio.run(main())
