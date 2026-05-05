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


def _retry_on_invalid_nonce(fn):
    """Call fn(), retrying once after 1 s if the exchange rejects the nonce.

    If the retry also raises, that exception propagates to the caller.
    """
    try:
        return fn()
    except ccxt.InvalidNonce:
        logging.warning("Invalid nonce — retrying after 1 s")
        time.sleep(1)
        return fn()


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

    positions = {}  # symbol -> {"entry_price": float, "quantity": float, "opened_at": float}

    # Seed positions from existing exchange holdings so the bot can apply
    # take-profit and stop-loss to assets held before this session started.
    # The current market price is used as the entry price, meaning thresholds
    # are measured from the time the bot starts up.
    try:
        holdings = bot.get_holdings()
        for symbol, holding in holdings.items():
            try:
                entry_price = bot.get_current_price(symbol)
                positions[symbol] = {
                    "entry_price": entry_price,
                    "quantity": holding["quantity"],
                    "tp_order_id": None,
                    "sl_order_id": None,
                    "opened_at": time.time(),
                }
                logging.info(
                    "Seeded existing holding: %s qty=%.8f entry_price=%.6f",
                    symbol,
                    holding["quantity"],
                    entry_price,
                )
            except Exception as exc:
                logging.warning(
                    "Could not fetch price for existing holding %s: %s", symbol, exc
                )
    except Exception as exc:
        logging.warning(
            "Could not load holdings from exchange (%s). "
            "Starting with empty positions.",
            exc,
        )

    # Build a reverse lookup: symbol -> bundle_name (first match wins).
    # Only used when config.USE_BUNDLES is True.
    symbol_to_bundle: dict = {}
    if config.USE_BUNDLES:
        for bundle_name, bundle_symbols in config.BUNDLES.items():
            for sym in bundle_symbols:
                symbol_to_bundle.setdefault(sym, bundle_name)

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
                        elif config.USE_BUNDLES and symbol in symbol_to_bundle:
                            bundle_name = symbol_to_bundle[symbol]
                            order_size = min(config.MAX_BUY_ORDER, usd_balance)
                            bundle_orders = bot.buy_bundle(bundle_name, order_size)
                            for sym, order in bundle_orders.items():
                                if sym not in positions:
                                    try:
                                        exit_orders = _retry_on_invalid_nonce(
                                            lambda: bot.place_exit_orders(
                                                sym, order["amount"], order["price"]
                                            )
                                        )
                                    except Exception as exc:
                                        logging.error(
                                            "Could not place exit orders for %s after "
                                            "retry (%s) — position tracked without "
                                            "orders; price-based exit will apply.",
                                            sym,
                                            exc,
                                        )
                                        exit_orders = {
                                            "take_profit_order_id": None,
                                            "stop_loss_order_id": None,
                                        }
                                    positions[sym] = {
                                        "entry_price": order["price"],
                                        "quantity": order["amount"],
                                        "tp_order_id": exit_orders["take_profit_order_id"],
                                        "sl_order_id": exit_orders["stop_loss_order_id"],
                                        "opened_at": time.time(),
                                    }
                                    logging.info(
                                        "Opened bundle position (%s): %s @ %s qty=%s "
                                        "TP_order=%s SL_order=%s",
                                        bundle_name,
                                        sym,
                                        order["price"],
                                        order["amount"],
                                        exit_orders["take_profit_order_id"],
                                        exit_orders["stop_loss_order_id"],
                                    )
                        else:
                            order_size = min(config.MAX_BUY_ORDER, usd_balance)
                            order = _retry_on_invalid_nonce(
                                lambda: bot.buy(symbol, order_size)
                            )
                            try:
                                exit_orders = _retry_on_invalid_nonce(
                                    lambda: bot.place_exit_orders(
                                        symbol, order["amount"], order["price"]
                                    )
                                )
                            except Exception as exc:
                                logging.error(
                                    "Could not place exit orders for %s after retry "
                                    "(%s) — position tracked without orders; "
                                    "price-based exit will apply.",
                                    symbol,
                                    exc,
                                )
                                exit_orders = {
                                    "take_profit_order_id": None,
                                    "stop_loss_order_id": None,
                                }
                            positions[symbol] = {
                                "entry_price": order["price"],
                                "quantity": order["amount"],
                                "tp_order_id": exit_orders["take_profit_order_id"],
                                "sl_order_id": exit_orders["stop_loss_order_id"],
                                "opened_at": time.time(),
                            }
                            logging.info(
                                "Opened position: %s @ %s qty=%s TP_order=%s SL_order=%s",
                                symbol,
                                order["price"],
                                order["amount"],
                                exit_orders["take_profit_order_id"],
                                exit_orders["stop_loss_order_id"],
                            )
                else:
                    entry_price = positions[symbol]["entry_price"]
                    tp_order_id = positions[symbol].get("tp_order_id")
                    sl_order_id = positions[symbol].get("sl_order_id")
                    quantity = positions[symbol]["quantity"]
                    opened_at = positions[symbol].get("opened_at", time.time())
                    # Auto-close if the position has been open for 24 hours
                    if time.time() - opened_at >= config.MAX_POSITION_HOLD_SECONDS:
                        # Cancel any pending TP/SL orders before selling at market
                        for oid in (tp_order_id, sl_order_id):
                            if oid:
                                try:
                                    bot.exchange.cancel_order(oid, symbol)
                                except Exception as cancel_exc:
                                    logging.warning(
                                        "Could not cancel order %s for %s: %s",
                                        oid, symbol, cancel_exc,
                                    )
                        sell_order = bot.sell(symbol, quantity)
                        logging.info(
                            "Closed position (max_hold_time): %s qty=%s @ %s",
                            symbol,
                            quantity,
                            sell_order.get("price"),
                        )
                        del positions[symbol]
                    else:
                        result = bot.check_exit_orders(
                            symbol, tp_order_id, sl_order_id, entry_price
                        )
                        if result in ("take_profit", "stop_loss"):
                            sell_order = bot.sell(symbol, quantity)
                            logging.info(
                                "Closed position (%s): %s qty=%s @ %s",
                                result, symbol, quantity, sell_order.get("price"),
                            )
                            del positions[symbol]
            except ccxt.InvalidNonce as e:
                # Nonce rejected by the exchange (e.g. after a fast restart or
                # a clock correction).  Wait briefly so the microsecond-based
                # nonce generator has time to advance past the server's counter,
                # then retry the symbol on the next iteration — no action is
                # taken this pass to avoid a double-buy.
                logging.warning(
                    "Invalid nonce for %s — will retry next iteration: %s", symbol, e
                )
                time.sleep(1)
            except ccxt.InsufficientFunds as e:
                logging.warning("Insufficient funds for %s — order skipped: %s", symbol, e)
            except Exception as e:
                logging.error("Error on %s: %s", symbol, e)

        time.sleep(60)  # check every minute

if __name__ == "__main__":
    main()
