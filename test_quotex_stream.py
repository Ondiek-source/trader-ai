import asyncio
import os
import sys
import time

sys.path.insert(0, "./src")

from dotenv import load_dotenv

load_dotenv()


async def test():
    from engine.quotex_reader import QuotexReader

    email = os.getenv("QUOTEX_EMAIL")
    password = os.getenv("QUOTEX_PASSWORD")

    if not email or not password:
        raise ValueError("QUOTEX_EMAIL and QUOTEX_PASSWORD must be set in environment")

    reader = QuotexReader(
        email=email,
        password=password,
        practice_mode=True,
        symbol="EURUSD_otc",
    )

    print("Connecting...")
    await reader.connect()

    if not reader._connected:
        print("Failed to connect")
        return

    print("Connected! Testing real-time price stream...\n")

    otc_asset = "EURUSD_otc"
    client = reader._client

    # Test different batch sizes with AWAIT
    for batch_size in [1, 5, 10, 60]:
        print(f"\n{'='*50}")
        print(f"Testing batch_size={batch_size}")
        print(f"{'='*50}")

        result = await client.start_realtime_price(otc_asset, batch_size)
        print(f"Return type: {type(result)}")

        if result and isinstance(result, dict) and otc_asset in result:
            price_points = result[otc_asset]
            print(f"Number of price points returned: {len(price_points)}")
            if price_points:
                print(f"First point: {price_points[0]}")
                print(f"Last point: {price_points[-1]}")

    # Test polling frequency - call multiple times to see if data changes
    print(f"\n{'='*50}")
    print(f"Testing polling frequency (batch_size=1)")
    print(f"{'='*50}")

    for i in range(5):
        result = await client.start_realtime_price(otc_asset, 1)
        if result and isinstance(result, dict) and otc_asset in result:
            price_points = result[otc_asset]
            if price_points:
                print(
                    f"Poll {i+1}: price={price_points[0]['price']}, time={price_points[0]['time']}"
                )
        await asyncio.sleep(0.5)

    # Test if price changes over time with larger batch
    print(f"\n{'='*50}")
    print(f"Testing price changes over time (batch_size=60)")
    print(f"{'='*50}")

    for i in range(3):
        result = await client.start_realtime_price(otc_asset, 60)
        if result and isinstance(result, dict) and otc_asset in result:
            price_points = result[otc_asset]
            if price_points:
                first_price = price_points[0]["price"]
                last_price = price_points[-1]["price"]
                print(
                    f"Poll {i+1}: first={first_price}, last={last_price}, count={len(price_points)}"
                )
        await asyncio.sleep(2)

    await reader.disconnect()


if __name__ == "__main__":
    asyncio.run(test())
