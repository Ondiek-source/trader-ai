"""Quick script to inspect Quotex trade history field names."""

import asyncio
import json
from pyquotex.stable_api import Quotex

EMAIL = "appke350@gmail.com"  # <-- replace
PASSWORD = "465856xoR"  # <-- replace


async def main():
    client = Quotex(email=EMAIL, password=PASSWORD, lang="en")
    ok, reason = await client.connect()
    if not ok:
        print(f"Connection failed: {reason}")
        return

    print(f"Connected: {reason}")

    # Switch to practice mode
    if hasattr(client, "change_account"):
        result = client.change_account("PRACTICE")
        if asyncio.iscoroutine(result):
            await result

    history = await client.get_history()
    print(f"\nTotal trades returned: {len(history)}")
    print(f"Type: {type(history)}")

    # Print last 3 trades with all fields
    for i, trade in enumerate(history[:3]):
        print(f"\n{'='*60}")
        print(f"Trade {i+1}:")
        print(json.dumps(trade, indent=2, default=str))

    # Also print all unique keys across the first 3 trades
    all_keys = set()
    for trade in history[:3]:
        all_keys.update(trade.keys())
    print(f"\n{'='*60}")
    print(f"All unique keys in first 3 trades: {sorted(all_keys)}")

    if hasattr(client, "close"):
        result = client.close()
        if asyncio.iscoroutine(result):
            await result


asyncio.run(main())
