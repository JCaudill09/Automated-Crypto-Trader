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

    positions = {}  # symbol -> {"entry_price": float, "quantity": float}

    while True:
        for symbol in config.DEFAULT_SYMBOLS:
            try:
                if symbol not in positions:
                    if bot.should_buy(symbol):
                        order = bot.buy(symbol, config.MAX_BUY_ORDER)
                        positions[symbol] = {
                            "entry_price": order["price"],
                            "quantity": order["amount"],
                        }
                        logging.info("Opened position: %s @ %s qty=%s", symbol, order["price"], order["amount"])
                else:
                    entry_price = positions[symbol]["entry_price"]
                    result = bot.check_exit(symbol, entry_price)
                    if result in ("take_profit", "stop_loss"):
                        quantity = positions[symbol]["quantity"]
                        sell_order = bot.sell(symbol, quantity)
                        logging.info(
                            "Closed position (%s): %s qty=%s @ %s",
                            result, symbol, quantity, sell_order.get("price"),
                        )
                        del positions[symbol]
            except Exception as e:
                logging.error("Error on %s: %s", symbol, e)

        time.sleep(300)  # check every 5 minutes

if __name__ == "__main__":
    main()
