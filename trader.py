"""
Automated Crypto Trader

Buys and sells cryptocurrency through a ccxt-compatible exchange.

Order constraints
-----------------
- Minimum buy order : $30 USD  (config.MIN_BUY_ORDER)
- Maximum buy order : $78 USD  (config.MAX_BUY_ORDER)

Trade signals
-------------
- Buy  : 20-EMA crosses **above** the 50-EMA (bullish crossover) **and**
         RSI is below ``config.RSI_OVERSOLD`` (default 50).
- Exit : take profit when price rises 6.5 % above entry
         (config.TAKE_PROFIT_PCT); stop loss when price falls 1.75 % below
         entry (config.STOP_LOSS_PCT).

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
        The ccxt exchange identifier (e.g. ``"kraken"``, ``"binance"``).
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

        exchange_id = exchange_id.lower()
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

    @staticmethod
    def _validate_usdt_pair(symbol: str) -> None:
        """
        Raise :class:`ValueError` if *symbol* is not a USDT-quoted pair.

        All buys are funded from USDT and all sells return proceeds to USDT,
        so only pairs of the form ``"BASE/USDT"`` are accepted.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USDT"``.

        Raises
        ------
        ValueError
            If *symbol* does not end with ``"/USDT"``.
        """
        if not symbol.upper().endswith("/USDT"):
            raise ValueError(
                f"Only USDT-quoted pairs are supported (e.g. 'BTC/USDT'), "
                f"got '{symbol}'."
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

    def get_usdt_symbols(self) -> list:
        """
        Return all active USDT-quoted trading pairs available on the exchange.

        Calls ``exchange.load_markets()`` to fetch the full market list, then
        filters to pairs whose quote currency is ``USDT`` and which are marked
        active by the exchange.

        Returns
        -------
        list[str]
            Sorted list of symbols, e.g. ``["BTC/USDT", "ETH/USDT", ...]``.

        Raises
        ------
        RuntimeError
            If no active USDT pairs are found.
        """
        markets = self.exchange.load_markets()
        symbols = sorted(
            symbol
            for symbol, market in markets.items()
            if market.get("quote") == "USDT" and market.get("active", True)
        )
        if not symbols:
            raise RuntimeError(
                "No active USDT-quoted markets found on the exchange."
            )
        logger.info("get_usdt_symbols — found %d active USDT pairs", len(symbols))
        return symbols

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
        ValueError
            If *symbol* is not a USDT-quoted pair.
        """
        self._validate_usdt_pair(symbol)
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

        self._validate_usdt_pair(symbol)

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

    def get_usdt_balance(self) -> float:
        """
        Return the free USDT balance available on the exchange.

        Returns
        -------
        float
            Amount of free USDT available to spend on new buy orders.

        Raises
        ------
        RuntimeError
            If the exchange does not return a USDT balance entry.
        """
        balance = self.exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        free = usdt.get("free")
        if free is None:
            raise RuntimeError(
                "Unable to retrieve USDT balance from exchange. "
                "The 'USDT.free' field is missing."
            )
        return float(free)

    def buy_max_orders(self, symbol: str) -> list:
        """
        Place as many buy orders for *symbol* as the current USDT balance
        allows.

        Each order is sized at ``max_buy_order`` USD.  Orders are placed
        until the remaining balance falls below ``min_buy_order``.  All
        proceeds from sells flow back to USDT, so the balance is the single
        source of funds for new buys.

        Parameters
        ----------
        symbol :
            USDT-quoted trading pair, e.g. ``"BTC/USDT"``.

        Returns
        -------
        list[dict]
            A list of order dicts — one entry per order placed.  Returns an
            empty list when the USDT balance is below ``min_buy_order``.

        Raises
        ------
        ValueError
            If *symbol* is not a USDT-quoted pair.
        """
        self._validate_usdt_pair(symbol)

        balance = self.get_usdt_balance()
        orders = []

        logger.info(
            "buy_max_orders %s — USDT balance=%.2f max_per_order=%.2f",
            symbol,
            balance,
            self.max_buy_order,
        )

        while balance >= self.min_buy_order:
            order_size = min(self.max_buy_order, balance)
            order = self.buy(symbol, order_size)
            orders.append(order)
            balance -= order_size
            logger.info(
                "buy_max_orders %s — placed order #%d ($%.2f), remaining balance=%.2f",
                symbol,
                len(orders),
                order_size,
                balance,
            )

        logger.info(
            "buy_max_orders %s — placed %d order(s) total",
            symbol,
            len(orders),
        )
        return orders

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ema(closes: list, period: int) -> float:
        """
        Compute the Exponential Moving Average (EMA) over *period* and
        return the last value.

        The EMA is seeded with the simple average of the first *period*
        closing prices and then updated with the standard multiplier
        ``2 / (period + 1)``.

        Parameters
        ----------
        closes :
            Ordered list of closing prices (oldest first).
        period :
            Look-back window for the EMA.

        Raises
        ------
        ValueError
            If fewer than *period* closes are provided.
        """
        if len(closes) < period:
            raise ValueError(
                f"Need at least {period} closing prices to compute "
                f"EMA-{period}, got {len(closes)}."
            )
        multiplier = 2.0 / (period + 1)
        ema = sum(closes[:period]) / period  # seed with SMA
        for price in closes[period:]:
            ema = price * multiplier + ema * (1.0 - multiplier)
        return ema

    @staticmethod
    def _compute_rsi(closes: list, period: int = 14) -> float:
        """
        Compute the Relative Strength Index (RSI) and return the last
        value.

        Uses Wilder's smoothing (RMA) seeded with the simple average of
        the first *period* up/down moves.

        Parameters
        ----------
        closes :
            Ordered list of closing prices (oldest first).
        period :
            Look-back window for the RSI (default 14).

        Raises
        ------
        ValueError
            If fewer than *period + 1* closes are provided.
        """
        if len(closes) < period + 1:
            raise ValueError(
                f"Need at least {period + 1} closing prices to compute "
                f"RSI-{period}, got {len(closes)}."
            )
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))

        # Seed with simple average of first `period` moves
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder smoothing for remaining moves
        for g, l in zip(gains[period:], losses[period:]):
            avg_gain = (avg_gain * (period - 1) + g) / period
            avg_loss = (avg_loss * (period - 1) + l) / period

        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def get_indicators(self, symbol: str, timeframe: str = "1h") -> dict:
        """
        Fetch OHLCV candles and return the latest indicator values.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USDT"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).

        Returns
        -------
        dict
            ``{"price": float, "ema20": float, "ema50": float,
               "prev_ema20": float, "prev_ema50": float, "rsi": float}``
        """
        limit = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        closes = [candle[4] for candle in ohlcv]  # index 4 = close price

        ema20 = self._compute_ema(closes, config.SIMPLE_ALGO_SHORT_PERIOD)
        ema50 = self._compute_ema(closes, config.SIMPLE_ALGO_LONG_PERIOD)
        prev_ema20 = self._compute_ema(closes[:-1], config.SIMPLE_ALGO_SHORT_PERIOD)
        prev_ema50 = self._compute_ema(closes[:-1], config.SIMPLE_ALGO_LONG_PERIOD)
        rsi = self._compute_rsi(closes, config.RSI_PERIOD)
        current_price = closes[-1]

        logger.debug(
            "Indicators %s — price=%.4f EMA20=%.4f EMA50=%.4f RSI=%.2f",
            symbol,
            current_price,
            ema20,
            ema50,
            rsi,
        )
        return {
            "price": current_price,
            "ema20": ema20,
            "ema50": ema50,
            "prev_ema20": prev_ema20,
            "prev_ema50": prev_ema50,
            "rsi": rsi,
        }

    def should_buy(self, symbol: str, timeframe: str = "1h") -> bool:
        """
        Return ``True`` when the buy signal fires.

        Buy conditions (both must be met):

        1. 20-EMA crosses **above** the 50-EMA (bullish crossover): on the
           previous candle EMA-20 was at or below EMA-50, and on the current
           candle EMA-20 is above EMA-50.
        2. RSI is **below** ``config.RSI_OVERSOLD`` (default 50) →
           momentum has not yet become overbought.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USDT"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).
        """
        indicators = self.get_indicators(symbol, timeframe)
        crossed_above = (
            indicators["prev_ema20"] <= indicators["prev_ema50"]
            and indicators["ema20"] > indicators["ema50"]
        )
        rsi_below = indicators["rsi"] < config.RSI_OVERSOLD

        signal = crossed_above and rsi_below

        logger.info(
            "should_buy %s — ema20=%.4f ema50=%.4f prev_ema20=%.4f prev_ema50=%.4f "
            "rsi=%.2f → crossed_above=%s rsi_below=%s signal=%s",
            symbol,
            indicators["ema20"],
            indicators["ema50"],
            indicators["prev_ema20"],
            indicators["prev_ema50"],
            indicators["rsi"],
            crossed_above,
            rsi_below,
            signal,
        )
        return signal

    def check_exit(self, symbol: str, entry_price: float) -> str:
        """
        Check whether a take-profit or stop-loss condition has been hit.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USDT"``.
        entry_price :
            The price at which the position was opened.

        Returns
        -------
        str
            * ``"take_profit"`` — current price ≥ entry × (1 + 7.5 %)
            * ``"stop_loss"``   — current price ≤ entry × (1 − 2.5 %)
            * ``"hold"``        — neither threshold reached

        Raises
        ------
        ValueError
            If *entry_price* is not positive.
        """
        if entry_price <= 0:
            raise ValueError(
                f"entry_price must be positive, got {entry_price}."
            )

        current_price = self._get_price(symbol)
        take_profit_price = entry_price * (1.0 + config.TAKE_PROFIT_PCT)
        stop_loss_price = entry_price * (1.0 - config.STOP_LOSS_PCT)

        logger.info(
            "check_exit %s — entry=%.6f current=%.6f TP=%.6f SL=%.6f",
            symbol,
            entry_price,
            current_price,
            take_profit_price,
            stop_loss_price,
        )

        if current_price >= take_profit_price:
            return "take_profit"
        if current_price <= stop_loss_price:
            return "stop_loss"
        return "hold"
