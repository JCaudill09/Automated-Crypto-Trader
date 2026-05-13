"""
Automated Crypto Trader

Buys and sells cryptocurrency through a ccxt-compatible exchange.

Order constraints
-----------------
- Minimum buy order : $40 USD  (config.MIN_BUY_ORDER)
- Maximum buy order : $70 USD  (config.MAX_BUY_ORDER)

Trade signals
-------------
- Buy  : EMA-20 is above EMA-200 (uptrend confirmed) **and** current price
         is above EMA-20 **and** RSI > 40 (momentum filter).  The signal
         fires on every qualifying 15-minute candle — not only on the
         crossover candle — so entries are taken throughout a sustained
         uptrend.
- Exit : take profit when price rises 6.5 % above entry
         (config.TAKE_PROFIT_PCT); stop loss when price falls 1.75 % below
         entry (config.STOP_LOSS_PCT).

Set PAPER_TRADING = True in config.py (the default) to simulate orders
without spending real money.

All trading pairs are USD-quoted (e.g. ``"BTC/USD"``).
"""

import logging
import threading
import time
from typing import Optional

import ccxt

import config

# Number of times to retry an order when Kraken rejects it with an invalid-nonce
# error before giving up and propagating the exception.
_NONCE_RETRY_ATTEMPTS = 3
# Seconds to wait between nonce-retry attempts to allow the exchange clock to
# advance past the previously accepted nonce.
_NONCE_RETRY_DELAY = 1.0
# Conversion factor from nanoseconds to milliseconds.
_NANOS_PER_MILLISECOND = 1_000_000

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
        ``config.MIN_BUY_ORDER`` ($40).
    max_buy_order : float
        Maximum allowed buy-order size in USD.  Defaults to
        ``config.MAX_BUY_ORDER`` ($70).
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
        self._last_nonce = 0
        self._nonce_lock = threading.Lock()

        exchange_id = exchange_id.lower()
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        if exchange_id == "kraken":
            if not hasattr(self.exchange, "options") or not isinstance(self.exchange.options, dict):
                self.exchange.options = {}
            self.exchange.options.setdefault("adjustForTimeDifference", True)
            self.exchange.nonce = self._next_nonce

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

    def _next_nonce(self) -> int:
        """
        Return a process-local strictly increasing nonce for Kraken requests.
        """
        nonce_candidate = time.time_ns() // _NANOS_PER_MILLISECOND
        with self._nonce_lock:
            if nonce_candidate <= self._last_nonce:
                nonce_candidate = self._last_nonce + 1
            self._last_nonce = nonce_candidate
            return nonce_candidate

    def _resync_exchange_clock(self) -> bool:
        """
        Ask the exchange to refresh server time offset when supported.
        """
        load_time_difference = getattr(self.exchange, "load_time_difference", None)
        if callable(load_time_difference):
            try:
                load_time_difference()
                return True
            except Exception as exc:
                logger.debug("Failed to refresh exchange time offset: %s", exc)
        return False

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
    def _validate_usd_pair(symbol: str) -> None:
        """
        Raise :class:`ValueError` if *symbol* is not a USD-quoted pair.

        All buys are funded from USD and all sells return proceeds to USD,
        so only pairs of the form ``"BASE/USD"`` are accepted.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.

        Raises
        ------
        ValueError
            If *symbol* does not end with ``"/USD"``.
        """
        if not symbol.upper().endswith("/USD"):
            raise ValueError(
                f"Only USD-quoted pairs are supported (e.g. 'BTC/USD'), "
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

    def _execute_order(self, order_fn, *args, **kwargs) -> dict:
        """
        Call *order_fn* with *args* / *kwargs* and retry up to
        ``_NONCE_RETRY_ATTEMPTS`` times when the exchange rejects the request
        with an invalid-nonce error (``EAPI:Invalid nonce``).

        Kraken requires every authenticated request to carry a strictly-
        increasing nonce.  Rapid sequential requests or momentary network
        jitter can cause nonces to arrive out of order, triggering this
        otherwise-transient error.  A short delay between retries lets the
        exchange clock advance so the next attempt uses a safely higher nonce.

        Parameters
        ----------
        order_fn :
            A callable that places an order (e.g.
            ``self.exchange.create_market_buy_order``).
        *args, **kwargs :
            Positional and keyword arguments forwarded to *order_fn*.

        Returns
        -------
        dict
            The order dict returned by the exchange on success.

        Raises
        ------
        ccxt.InvalidNonce
            If all retry attempts are exhausted.
        """
        last_exc: ccxt.InvalidNonce | None = None
        for attempt in range(1, _NONCE_RETRY_ATTEMPTS + 1):
            try:
                return order_fn(*args, **kwargs)
            except ccxt.InvalidNonce as exc:
                last_exc = exc
                resynced = self._resync_exchange_clock()
                retry_delay = _NONCE_RETRY_DELAY * attempt
                logger.warning(
                    "Invalid nonce on attempt %d/%d — retrying in %.1fs (resynced=%s, %s)",
                    attempt,
                    _NONCE_RETRY_ATTEMPTS,
                    retry_delay,
                    resynced,
                    exc,
                )
                if attempt < _NONCE_RETRY_ATTEMPTS:
                    time.sleep(retry_delay)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_usd_symbols(self) -> list:
        """
        Return all active USD-quoted trading pairs available on the exchange.

        Calls ``exchange.load_markets()`` to fetch the full market list, then
        filters to pairs whose quote currency is ``USD`` and which are marked
        active by the exchange.

        Returns
        -------
        list[str]
            Sorted list of symbols, e.g. ``["BTC/USD", "ETH/USD", ...]``.

        Raises
        ------
        RuntimeError
            If no active USD pairs are found.
        """
        markets = self.exchange.load_markets()
        symbols = sorted(
            symbol
            for symbol, market in markets.items()
            if market.get("quote") == "USD" and market.get("active", True)
        )
        if not symbols:
            raise RuntimeError(
                "No active USD-quoted markets found on the exchange."
            )
        logger.info("get_usd_symbols — found %d active USD pairs", len(symbols))
        return symbols

    def buy(self, symbol: str, amount_usd: float) -> dict:
        """
        Place a market buy order for *symbol* worth *amount_usd* USD.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        amount_usd :
            Order value in USD.  Must be between ``min_buy_order`` ($40)
            and ``max_buy_order`` ($70) inclusive.

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
            If *symbol* is not a USD-quoted pair.
        """
        self._validate_usd_pair(symbol)
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

        return self._execute_order(self.exchange.create_market_buy_order, symbol, quantity)

    def sell(self, symbol: str, quantity: float) -> dict:
        """
        Place a market sell order for *quantity* units of *symbol*.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
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

        self._validate_usd_pair(symbol)

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

        return self._execute_order(self.exchange.create_market_sell_order, symbol, quantity)

    def get_usd_balance(self) -> float:
        """
        Return the free USD balance available on the exchange.

        Returns
        -------
        float
            Amount of free USD available to spend on new buy orders.

        Raises
        ------
        RuntimeError
            If the exchange does not return a USD balance entry.
        """
        balance = self.exchange.fetch_balance()
        usd = balance.get("USD", {})
        free = usd.get("free")
        if free is None:
            raise RuntimeError(
                "Unable to retrieve USD balance from exchange. "
                "The 'USD.free' field is missing."
            )
        return float(free)

    def buy_max_orders(self, symbol: str) -> list:
        """
        Place as many buy orders for *symbol* as the current USD balance
        allows.

        Each order is sized at ``max_buy_order`` USD.  Orders are placed
        until the remaining balance falls below ``min_buy_order``.  All
        proceeds from sells flow back to USD, so the balance is the single
        source of funds for new buys.

        Parameters
        ----------
        symbol :
            USD-quoted trading pair, e.g. ``"BTC/USD"``.

        Returns
        -------
        list[dict]
            A list of order dicts — one entry per order placed.  Returns an
            empty list when the USD balance is below ``min_buy_order``.

        Raises
        ------
        ValueError
            If *symbol* is not a USD-quoted pair.
        """
        self._validate_usd_pair(symbol)

        balance = self.get_usd_balance()
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

    def get_indicators(self, symbol: str, timeframe: str = "15m") -> dict:
        """
        Fetch OHLCV candles and return the latest indicator values.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"15m"``).

        Returns
        -------
        dict
            ``{"price": float, "ema50": float, "ema200": float,
               "prev_ema50": float, "prev_ema200": float, "rsi": float}``
        """
        limit = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        closes = [candle[4] for candle in ohlcv]  # index 4 = close price

        ema50 = self._compute_ema(closes, config.SIMPLE_ALGO_SHORT_PERIOD)
        ema200 = self._compute_ema(closes, config.SIMPLE_ALGO_LONG_PERIOD)
        prev_ema50 = self._compute_ema(closes[:-1], config.SIMPLE_ALGO_SHORT_PERIOD)
        prev_ema200 = self._compute_ema(closes[:-1], config.SIMPLE_ALGO_LONG_PERIOD)
        rsi = self._compute_rsi(closes, config.RSI_PERIOD)
        current_price = closes[-1]

        logger.debug(
            "Indicators %s — price=%.4f EMA%d=%.4f EMA%d=%.4f RSI=%.2f",
            symbol,
            current_price,
            config.SIMPLE_ALGO_SHORT_PERIOD,
            ema50,
            config.SIMPLE_ALGO_LONG_PERIOD,
            ema200,
            rsi,
        )
        return {
            "price": current_price,
            "ema50": ema50,
            "ema200": ema200,
            "prev_ema50": prev_ema50,
            "prev_ema200": prev_ema200,
            "rsi": rsi,
        }

    def should_buy(self, symbol: str, timeframe: str = "15m") -> bool:
        """
        Return ``True`` when the buy signal fires.

        Buy condition (all three must be true):

        1. Short EMA (EMA-20) is **above** the long EMA (EMA-200) — confirming
           an established uptrend.  The signal fires on every candle while the
           trend holds, not only on the crossover candle.
        2. Current price is **above** the short EMA — price is leading the
           trend, not lagging it.
        3. RSI is **above** ``config.RSI_BUY_THRESHOLD`` (40) — confirms at
           least minimal bullish momentum.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"15m"``).
        """
        indicators = self.get_indicators(symbol, timeframe)
        in_uptrend = indicators["ema50"] > indicators["ema200"]
        price_above_short_ema = indicators["price"] > indicators["ema50"]
        rsi_confirmed = indicators["rsi"] > config.RSI_BUY_THRESHOLD
        signal = in_uptrend and price_above_short_ema and rsi_confirmed

        logger.info(
            "should_buy %s — price=%.4f ema_short=%.4f ema_long=%.4f rsi=%.2f "
            "→ in_uptrend=%s price_above_ema=%s rsi_ok=%s signal=%s",
            symbol,
            indicators["price"],
            indicators["ema50"],
            indicators["ema200"],
            indicators["rsi"],
            in_uptrend,
            price_above_short_ema,
            rsi_confirmed,
            signal,
        )
        return signal

    def check_exit(self, symbol: str, entry_price: float) -> str:
        """
        Check whether a take-profit or stop-loss condition has been hit.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
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
