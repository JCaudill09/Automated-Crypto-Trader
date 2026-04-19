# main.py
import logging
import os
import time

import ccxt
from dotenv import load_dotenv

from trader import CryptoTrader
import config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# How often (in seconds) to refresh the list of available USD pairs.
SYMBOL_REFRESH_INTERVAL = config.SYMBOL_REFRESH_INTERVAL


def main():
    api_key = os.environ.get("KRAKEN_API_KEY")
    api_secret = os.environ.get("KRAKEN_API_SECRET")
    if not api_key or not api_secret:
        raise EnvironmentError(
            f"API credentials for '{config.DEFAULT_EXCHANGE}' are not set. "
            "Provide them as environment variables (e.g. KRAKEN_API_KEY / KRAKEN_API_SECRET) "
            "or in your deployment platform's secrets settings."
        )

    bot = CryptoTrader(
        exchange_id=config.DEFAULT_EXCHANGE,
        api_key=api_key,
        api_secret=api_secret,
        paper_trading=config.PAPER_TRADING,
    )

    positions = {}  # symbol -> {"entry_price": float, "quantity": float}

    # Discover all active USD pairs from the exchange.  Fall back to the
    # static list in config if the market-discovery call fails.
    try:
        symbols = bot.get_usd_symbols()
    except Exception as exc:
        logging.warning(
            "Could not load USD pairs from exchange (%s). "
            "Falling back to DEFAULT_SYMBOLS.",
            exc,
        )
        symbols = list(config.DEFAULT_SYMBOLS)

    last_symbol_refresh = time.monotonic()

    while True:
        # Refresh the symbol list once per hour so newly listed pairs are
        # picked up automatically without restarting the bot.
        if time.monotonic() - last_symbol_refresh >= SYMBOL_REFRESH_INTERVAL:
            try:
                symbols = bot.get_usd_symbols()
                last_symbol_refresh = time.monotonic()
            except Exception as exc:
                logging.warning("Symbol refresh failed (%s). Keeping current list.", exc)

        for symbol in symbols:
            try:
                if symbol not in positions:
                    if bot.should_buy(symbol):
                        usd_balance = bot.get_usd_balance()
                        if usd_balance < config.MIN_BUY_ORDER:
                            logging.warning(
                                "Skipping buy for %s — insufficient USD balance "
                                "(%.2f < %.2f min)",
                                symbol,
                                usd_balance,
                                config.MIN_BUY_ORDER,
                            )
                        else:
                            order_size = min(config.MAX_BUY_ORDER, usd_balance)
                            order = bot.buy(symbol, order_size)
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
            except ccxt.InsufficientFunds as e:
                logging.warning("Insufficient funds for %s — skipping: %s", symbol, e)
            except Exception as e:
                logging.error("Error on %s: %s", symbol, e)

        time.sleep(60)  # check every minute

if __name__ == "__main__":
    main()
