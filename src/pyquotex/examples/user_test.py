import asyncio
from pathlib import Path
from pyquotex.stable_api import Quotex

# USER_AGENT = (
#     "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
# )
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"


class QuotexExchange:

    def __init__(self, **kwargs):
        # Initialize the Quotex client using the new library
        self.client = Quotex(
            email=kwargs.get("email") or "",
            password=kwargs.get("password") or "",
            lang=kwargs.get("lang", "pt"),  # Language default set to Portuguese
        )
        data_dir = Path(__file__).parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        self.client.set_session(user_agent=USER_AGENT)
        self.client.debug_ws_enable = True
        self.connected = False
        self.max_retries = int(kwargs.get("retry", 5))  # Retry attempts for connection
        self.practice = kwargs.get("practice", "yes").lower() == "yes"  # Practice mode
        self.otc = kwargs.get("otc", "yes").lower() == "yes"  # OTC mode

    async def connect(self):
        return await self.client.connect()

    async def disconnect(self):
        await self.client.close()

    async def check_connect(self):
        await asyncio.sleep(0)
        return self.client.check_connect()

    async def get_balance(self):
        return await self.client.get_balance()


async def main():
    params = {
        "email": "appke350@gmail.com",
        "password": "465856xoR",
        "lang": "pt",
    }
    trade = QuotexExchange(**params)
    await trade.connect()
    is_connected = await trade.check_connect()
    if is_connected:
        print(f"Connected: {is_connected}")
        balance = await trade.get_balance()
        print(f"Balance: {balance}")
    print("Closing...")
    await trade.disconnect()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        loop.close()
