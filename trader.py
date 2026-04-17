"""
Automated Crypto Trader

Buys and sells cryptocurrency through a ccxt-compatible exchange.

Order constraints
-----------------
- Minimum buy order : $30 USD  (config.MIN_BUY_ORDER)
- Maximum buy order : $50 USD  (config.MAX_BUY_ORDER)

Set PAPER_TRADING = True in config.py (the default) to simulate orders
without spending real money.
"""

import logging
from typing import Optional

import ccxt

import config

logger = logging.getLogger(__name__)


class OrderSizeError(ValueError):
    """Raised when a buy order amount is outside the allowed range."""


class CryptoTrader:
    """
    A simple automated crypto trader.

    Parameters
    ----------
    exchange_id : str
        The ccxt exchange identifier (e.g. ``"coinbase"``, ``"binance"``).
    api_key : str, optional
        Exchange API key.  Not required in paper-trading mode.
    api_secret : str, optional
        Exchange API secret.  Not required in paper-trading mode.
    paper_trading : bool
        When *True* (default from ``config.PAPER_TRADING``) all orders are
        simulated and no real money is spent.
    min_buy_order : float
        Minimum allowed buy-order size in USD.  Defaults to
        ``config.MIN_BUY_ORDER`` ($30).
    max_buy_order : float
        Maximum allowed buy-order size in USD.  Defaults to
        ``config.MAX_BUY_ORDER`` ($50).
    """

    def __init__(
        self,
        exchange_id: str = config.DEFAULT_EXCHANGE,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper_trading: bool = config.PAPER_TRADING,
        min_buy_order: float = config.MIN_BUY_ORDER,
        max_buy_order: float = config.MAX_BUY_ORDER,
    ):
        self.paper_trading = paper_trading
        self.min_buy_order = min_buy_order
        self.max_buy_order = max_buy_order

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
            }
        )

        logger.info(
            "CryptoTrader initialised — exchange=%s paper_trading=%s "
            "min_buy=$%.2f max_buy=$%.2f",
            exchange_id,
            paper_trading,
            min_buy_order,
            max_buy_order,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_buy_amount(self, amount_usd: float) -> None:
        """
        Raise :class:`OrderSizeError` if *amount_usd* is outside the
        configured buy-order limits.
        """
        if amount_usd < self.min_buy_order:
            raise OrderSizeError(
                f"Buy order of ${amount_usd:.2f} is below the minimum "
                f"allowed amount of ${self.min_buy_order:.2f}."
            )
        if amount_usd > self.max_buy_order:
            raise OrderSizeError(
                f"Buy order of ${amount_usd:.2f} exceeds the maximum "
                f"allowed amount of ${self.max_buy_order:.2f}."
            )

    def _get_price(self, symbol: str) -> float:
        """Return the current ask price for *symbol*."""
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker.get("ask") or ticker.get("last")
        if not price:
            raise RuntimeError(
                f"Unable to retrieve a valid price for {symbol}. "
                "Both 'ask' and 'last' fields are missing or zero."
            )
        return float(price)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def buy(self, symbol: str, amount_usd: float) -> dict:
        """
        Place a market buy order for *symbol* worth *amount_usd* USD.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USDT"``.
        amount_usd :
            Order value in USD.  Must be between ``min_buy_order`` ($30)
            and ``max_buy_order`` ($50) inclusive.

        Returns
        -------
        dict
            Order details returned by the exchange, or a simulated order
            dict when *paper_trading* is ``True``.

        Raises
        ------
        OrderSizeError
            If *amount_usd* is outside the configured range.
        """
        self._validate_buy_amount(amount_usd)

        price = self._get_price(symbol)
        quantity = amount_usd / price

        logger.info(
            "BUY  %s — $%.2f @ %.6f ≈ %.8f units (paper=%s)",
            symbol,
            amount_usd,
            price,
            quantity,
            self.paper_trading,
        )

        if self.paper_trading:
            return {
                "symbol": symbol,
                "side": "buy",
                "type": "market",
                "amount": quantity,
                "price": price,
                "cost": amount_usd,
                "status": "closed",
                "paper": True,
            }

        return self.exchange.create_market_buy_order(symbol, quantity)

    def sell(self, symbol: str, quantity: float) -> dict:
        """
        Place a market sell order for *quantity* units of *symbol*.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USDT"``.
        quantity :
            Number of units to sell.

        Returns
        -------
        dict
            Order details returned by the exchange, or a simulated order
            dict when *paper_trading* is ``True``.
        """
        if quantity <= 0:
            raise ValueError(f"Sell quantity must be positive, got {quantity}.")

        price = self._get_price(symbol)

        logger.info(
            "SELL %s — %.8f units @ %.6f (paper=%s)",
            symbol,
            quantity,
            price,
            self.paper_trading,
        )

        if self.paper_trading:
            return {
                "symbol": symbol,
                "side": "sell",
                "type": "market",
                "amount": quantity,
                "price": price,
                "cost": quantity * price,
                "status": "closed",
                "paper": True,
            }

        return self.exchange.create_market_sell_order(symbol, quantity)
