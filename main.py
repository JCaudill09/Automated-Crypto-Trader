# main.py
import logging
import os
import time

from trader import CryptoTrader
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def main():
    bot = CryptoTrader(
        exchange_id=config.DEFAULT_EXCHANGE,
        api_key=os.environ["COINBASE_API_KEY"],
        api_secret=os.environ["COINBASE_API_SECRET"],
        paper_trading=config.PAPER_TRADING,
    )

    positions = {}  # symbol -> entry_price

    while True:
        for symbol in config.DEFAULT_SYMBOLS:
            try:
                if symbol not in positions:
                    if bot.should_buy(symbol):
                        order = bot.buy(symbol, config.MAX_BUY_ORDER)
                        positions[symbol] = order["price"]
                        logging.info("Opened position: %s @ %s", symbol, order["price"])
                else:
                    result = bot.check_exit(symbol, positions[symbol])
                    if result in ("take_profit", "stop_loss"):
                        # you'd need to track quantity here in a real bot
                        logging.info("Exit signal (%s) for %s", result, symbol)
                        del positions[symbol]
            except Exception as e:
                logging.error("Error on %s: %s", symbol, e)

        time.sleep(300)  # check every 5 minutes

if __name__ == "__main__":
    main()
