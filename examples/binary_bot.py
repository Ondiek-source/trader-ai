# Bot Options Binary — Experiment on Quotex (educational only)
# Uses an unofficial library to communicate with the platform (warning: experimental only)
import asyncio
import time
from pyquotex.stable_api import Quotex

# ----- Settings -----
EMAIL = "your_email@example.com"
PASSWORD = "your_password"
SYMBOL = "EURUSD_otc"  # financial asset — use OTC symbol format
TIMEFRAME = 60  # candle period in seconds (60 = 1 min)
DURATION = 60  # trade duration in seconds
RISK_PERCENT = 1.0  # risk percentage per trade %
MAX_TRADES = 2  # maximum number of open trades at the same time
RSI_PERIOD = 14
SMA_PERIOD = 50

# ----- Platform connection -----
client = Quotex(email=EMAIL, password=PASSWORD, lang="en")


# ----- Analysis functions -----
def calc_rsi(closes, period=RSI_PERIOD):
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_sma(closes, period=SMA_PERIOD):
    if len(closes) < period:
        return sum(closes) / len(closes)
    return sum(closes[-period:]) / period


# ----- Risk management -----
def calc_amount(balance, risk_percent):
    return max(balance * (risk_percent / 100), 1.0)


# ----- Wait for next candle -----
def wait_next_candle(timeframe):
    now = int(time.time())
    wait = timeframe - (now % timeframe)
    time.sleep(wait + 1)


# ----- Main loop -----
async def main():
    connected, msg = await client.connect()
    if not connected:
        print(f"❌ Connection failed: {msg}")
        return

    print("🚀 Bot started...")

    try:
        while True:
            wait_next_candle(TIMEFRAME)

            # Fetch recent candles: asset, end_time, offset_seconds, period
            candles = await client.get_candles(
                SYMBOL,
                time.time(),
                TIMEFRAME * (SMA_PERIOD + RSI_PERIOD + 5),
                TIMEFRAME,
            )

            if not candles or len(candles) < RSI_PERIOD + 1:
                print("⚠️ Not enough candle data")
                continue

            closes = [float(c["close"]) for c in candles]
            rsi = calc_rsi(closes)
            rsi_prev = calc_rsi(closes[:-1])
            sma = calc_sma(closes)
            last_close = closes[-1]

            direction = None
            if rsi_prev < 30 and rsi > 30 and last_close > sma:
                direction = "call"
            elif rsi_prev > 70 and rsi < 70 and last_close < sma:
                direction = "put"

            if direction:
                balance = await client.get_balance()
                amount = calc_amount(balance, RISK_PERCENT)
                success, trade_id = await client.buy(
                    amount, SYMBOL, direction, DURATION
                )
                if success:
                    print(
                        f"✅ Trade opened {direction} | amount: {amount} | id: {trade_id}"
                    )
                else:
                    print(f"❌ Trade failed: {trade_id}")
            else:
                print("No signal...")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
