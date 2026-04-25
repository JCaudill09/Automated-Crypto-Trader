"""
Automated Crypto Trader

Buys and sells USD-quoted instruments (cryptocurrencies, tokenized stocks,
and ETFs) through a ccxt-compatible exchange such as Kraken.

Order constraints
-----------------
- Minimum buy order : $30 USD  (config.MIN_BUY_ORDER)
- Maximum buy order : $78 USD  (config.MAX_BUY_ORDER)

Trade signals
-------------
- Volume  : Relative Volume (RVOL) must be at least ``config.RVOL_THRESHOLD``
            (default 5×) — current candle volume is 5× the rolling average,
            confirming a high-participation breakout move.
- Breakout : Price must close **above** the upper Bollinger Band
             (``config.BB_PERIOD``/``config.BB_NUM_STD``) **or** above the
             upper Keltner Channel (``config.KC_PERIOD``/``config.KC_MULTIPLIER``)
             → a confirmed volatility/momentum breakout.
  Both conditions together (plus the 24-hour quote-volume and bid-ask
  spread checks) must pass for a buy signal to fire.
- Execution (Volatility) : stop-loss is placed at
  ``entry_price − config.ATR_STOP_LOSS_MULTIPLIER × ATR``
  (default 1.5 × ATR), adapting risk to current market speed.
- Exit : RSI above ``config.RSI_OVERBOUGHT`` (default 70) → asset is
         overbought / expensive; sell to take profit.  Take-profit target is
         5 % above entry; stop-loss is 1.75 % below entry.

- Volume  : buy orders and buy signals are only issued when the 24-hour
           quote-currency volume is at least ``config.MIN_VOLUME_USD``
           (default $15,000), ensuring there is sufficient liquidity to
           fill and exit positions.
- Spread  : buy orders and buy signals are only issued when the bid-ask
           spread is below ``config.MAX_BID_ASK_SPREAD_PCT`` (default 1 %),
           ensuring the market is tight enough for reliable order fills.

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


class InsufficientVolumeError(RuntimeError):
    """Raised when 24-hour quote volume is below the configured minimum."""


class WideBidAskSpreadError(RuntimeError):
    """Raised when the bid-ask spread exceeds the configured maximum."""


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

    def get_current_price(self, symbol: str) -> float:
        """
        Return the current market price for *symbol*.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.

        Returns
        -------
        float
            Current ask price, falling back to last-traded price when ask
            is unavailable.

        Raises
        ------
        RuntimeError
            If neither ``ask`` nor ``last`` fields are available or non-zero.
        """
        return self._get_price(symbol)

    def _check_volume(self, symbol: str) -> None:
        """
        Raise :class:`InsufficientVolumeError` if the 24-hour quote-currency
        volume for *symbol* is below ``config.MIN_VOLUME_USD``.

        The ``quoteVolume`` field from ``fetch_ticker`` represents the total
        value traded over the last 24 hours expressed in the quote currency
        (e.g. USD for ``BTC/USD``).

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.

        Raises
        ------
        InsufficientVolumeError
            If the reported 24-hour quote volume is below
            ``config.MIN_VOLUME_USD``, or if the exchange does not return a
            volume figure.
        """
        ticker = self.exchange.fetch_ticker(symbol)
        volume = ticker.get("quoteVolume")
        if not volume:
            raise InsufficientVolumeError(
                f"Unable to retrieve 24-hour quote volume for {symbol}. "
                "Cannot verify sufficient liquidity."
            )
        volume = float(volume)

        if volume < config.MIN_VOLUME_USD:
            raise InsufficientVolumeError(
                f"{symbol} 24-hour quote volume ${volume:,.2f} is below "
                f"the minimum ${config.MIN_VOLUME_USD:,.2f}."
            )
        logger.debug(
            "Volume check %s — quoteVolume=$%.2f (min=$%.2f) ✓",
            symbol,
            volume,
            config.MIN_VOLUME_USD,
        )

    def _check_spread(self, symbol: str) -> None:
        """
        Raise :class:`WideBidAskSpreadError` if the bid-ask spread for
        *symbol* is at or above ``config.MAX_BID_ASK_SPREAD_PCT``.

        The spread is computed as ``(ask - bid) / ask`` and compared against
        the configured threshold.  A spread at or above the threshold
        indicates insufficient book depth or excessive market-maker costs.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.

        Raises
        ------
        WideBidAskSpreadError
            If the bid-ask spread is at or above
            ``config.MAX_BID_ASK_SPREAD_PCT``, or if bid/ask prices cannot
            be retrieved from the exchange.
        """
        ticker = self.exchange.fetch_ticker(symbol)
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if not bid or not ask:
            raise WideBidAskSpreadError(
                f"Unable to retrieve bid/ask prices for {symbol}. "
                "Cannot verify bid-ask spread."
            )
        bid = float(bid)
        ask = float(ask)
        spread_pct = (ask - bid) / ask
        if spread_pct >= config.MAX_BID_ASK_SPREAD_PCT:
            raise WideBidAskSpreadError(
                f"{symbol} bid-ask spread {spread_pct:.4%} is at or above the "
                f"maximum allowed {config.MAX_BID_ASK_SPREAD_PCT:.2%} "
                f"(bid=${bid:,.6f}, ask=${ask:,.6f})."
            )
        logger.debug(
            "Spread check %s — spread=%.4f%% (max=%.1f%%) ✓",
            symbol,
            spread_pct * 100,
            config.MAX_BID_ASK_SPREAD_PCT * 100,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_usd_symbols(self) -> list:
        """
        Return all active USD-quoted trading pairs available on the exchange.

        Calls ``exchange.load_markets()`` to fetch the full market list, then
        filters to pairs whose quote currency is ``USD`` and which are marked
        active by the exchange.

        When ``config.ASSET_TYPES`` is a non-empty list only markets whose
        ccxt ``type`` field is contained in that list are included.  This
        lets callers restrict discovery to specific asset classes, e.g.
        ``["spot"]`` for spot markets only.  When ``config.ASSET_TYPES`` is
        ``None`` no type-filtering is applied and all asset types — including
        cryptocurrencies, tokenized stocks, and ETFs — are returned.

        Returns
        -------
        list[str]
            Sorted list of symbols, e.g. ``["AAPL/USD", "BTC/USD", "SPY/USD", ...]``.

        Raises
        ------
        RuntimeError
            If no active USD pairs are found.
        """
        markets = self.exchange.load_markets()
        asset_types = config.ASSET_TYPES  # None → no filter
        symbols = sorted(
            symbol
            for symbol, market in markets.items()
            if market.get("quote") == "USD"
            and market.get("active", True)
            and (
                asset_types is None
                or market.get("type") in asset_types
            )
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
        InsufficientVolumeError
            If the 24-hour quote volume for *symbol* is below 4 % of its
            market capitalisation.
        WideBidAskSpreadError
            If the bid-ask spread for *symbol* is at or above
            ``config.MAX_BID_ASK_SPREAD_PCT``.
        ValueError
            If *symbol* is not a USD-quoted pair.
        """
        self._validate_usd_pair(symbol)
        self._validate_buy_amount(amount_usd)
        self._check_volume(symbol)
        self._check_spread(symbol)

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

        return self.exchange.create_market_sell_order(symbol, quantity)

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
            "buy_max_orders %s — USD balance=%.2f max_per_order=%.2f",
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

    def buy_bundle(self, bundle_name: str, amount_usd_per_symbol: float) -> dict:
        """
        Place a market buy order for every symbol in a named bundle.

        Bundles are defined in ``config.BUNDLES`` as a mapping of
        ``bundle_name → [symbol, ...]``.  Each symbol receives its own
        independent order for *amount_usd_per_symbol* USD.

        Symbols whose orders fail due to insufficient volume, a wide bid-ask
        spread, or an invalid order size are **skipped** with a warning rather
        than aborting the entire bundle, so a single illiquid asset cannot
        block the rest.
        All other exceptions are re-raised immediately.

        Works in both **paper-trading** and **live-trading** modes — the
        underlying :meth:`buy` method handles the distinction transparently.

        Parameters
        ----------
        bundle_name :
            Key in ``config.BUNDLES``, e.g. ``"large_caps"``.
        amount_usd_per_symbol :
            USD value to spend on each symbol in the bundle.  Must satisfy
            the configured ``min_buy_order`` / ``max_buy_order`` limits.

        Returns
        -------
        dict[str, dict]
            Mapping of ``symbol → order`` for every symbol that was
            successfully bought.  Symbols that were skipped are absent.

        Raises
        ------
        KeyError
            If *bundle_name* is not found in ``config.BUNDLES``.
        """
        if bundle_name not in config.BUNDLES:
            raise KeyError(
                f"Bundle '{bundle_name}' is not defined in config.BUNDLES. "
                f"Available bundles: {list(config.BUNDLES.keys())}"
            )

        symbols = config.BUNDLES[bundle_name]
        orders: dict = {}

        logger.info(
            "buy_bundle '%s' — %d symbol(s), $%.2f each (paper=%s)",
            bundle_name,
            len(symbols),
            amount_usd_per_symbol,
            self.paper_trading,
        )

        for symbol in symbols:
            try:
                order = self.buy(symbol, amount_usd_per_symbol)
                orders[symbol] = order
                logger.info(
                    "buy_bundle '%s' — bought %s: qty=%.8f @ %.6f",
                    bundle_name,
                    symbol,
                    order["amount"],
                    order["price"],
                )
            except (InsufficientVolumeError, WideBidAskSpreadError, OrderSizeError) as exc:
                logger.warning(
                    "buy_bundle '%s' — skipping %s: %s",
                    bundle_name,
                    symbol,
                    exc,
                )

        logger.info(
            "buy_bundle '%s' — placed %d / %d orders",
            bundle_name,
            len(orders),
            len(symbols),
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

    @staticmethod
    def _compute_vwap(ohlcv: list) -> float:
        """
        Compute the Volume Weighted Average Price (VWAP).

        VWAP = Σ(typical_price × volume) / Σ(volume)
        where typical_price = (high + low + close) / 3.

        When the current price is above the VWAP the intraday trend is up;
        when it is below, the trend is down.

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles
            ``[timestamp, open, high, low, close, volume]``.

        Returns
        -------
        float
            VWAP value.

        Raises
        ------
        ValueError
            If the total volume across all candles is zero.
        """
        total_pv = 0.0
        total_volume = 0.0
        for candle in ohlcv:
            high, low, close, volume = candle[2], candle[3], candle[4], candle[5]
            typical_price = (high + low + close) / 3.0
            total_pv += typical_price * volume
            total_volume += volume
        if total_volume == 0.0:
            raise ValueError(
                "Cannot compute VWAP: total volume across all candles is zero."
            )
        return total_pv / total_volume

    @staticmethod
    def _compute_volume_profile(ohlcv: list, num_bins: int = 50) -> dict:
        """
        Compute a basic Volume Profile and return the Point of Control (POC).

        The price range spanned by all candles is divided into *num_bins*
        equal-width bins.  Each candle's volume is assigned to the bin that
        contains its typical price ``(high + low + close) / 3``.  The POC is
        the mid-price of the bin that accumulated the most volume — i.e. the
        strongest support/resistance level in the period.

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles
            ``[timestamp, open, high, low, close, volume]``.
        num_bins :
            Number of equal-width price bins
            (default ``config.VOLUME_PROFILE_BINS``, 50).

        Returns
        -------
        dict
            ``{"poc": float}`` — the Point of Control price level.
        """
        min_price = float("inf")
        max_price = float("-inf")
        for candle in ohlcv:
            if candle[3] < min_price:
                min_price = candle[3]
            if candle[2] > max_price:
                max_price = candle[2]

        if max_price <= min_price:
            return {"poc": float(ohlcv[-1][4])}  # fallback: last close

        bin_size = (max_price - min_price) / num_bins
        volume_bins = [0.0] * num_bins

        for candle in ohlcv:
            typical_price = (candle[2] + candle[3] + candle[4]) / 3.0
            bin_idx = min(
                int((typical_price - min_price) / bin_size), num_bins - 1
            )
            volume_bins[bin_idx] += candle[5]

        poc_bin = 0
        poc_volume = volume_bins[0]
        for i in range(1, num_bins):
            if volume_bins[i] > poc_volume:
                poc_volume = volume_bins[i]
                poc_bin = i
        poc_price = min_price + (poc_bin + 0.5) * bin_size
        return {"poc": poc_price}

    @staticmethod
    def _compute_simple_algo_signal(closes: list) -> bool:
        """
        Return a bullish trend signal using an EMA 50 / EMA 200 golden cross.

        The signal is ``True`` (bullish) when the EMA 50
        (``config.SIMPLE_ALGO_SHORT_PERIOD``) is above the EMA 200
        (``config.SIMPLE_ALGO_LONG_PERIOD``), indicating that the market is in
        a confirmed uptrend.

        Parameters
        ----------
        closes :
            Ordered list of closing prices (oldest first).

        Returns
        -------
        bool
            ``True`` when EMA 50 is above EMA 200 (golden cross).
        """
        if len(closes) < config.SIMPLE_ALGO_LONG_PERIOD:
            return False
        ema_short = CryptoTrader._compute_ema(closes, config.SIMPLE_ALGO_SHORT_PERIOD)
        ema_long = CryptoTrader._compute_ema(closes, config.SIMPLE_ALGO_LONG_PERIOD)
        return ema_short > ema_long

    @staticmethod
    def _compute_bollinger_bands(
        closes: list, period: int, num_std: float
    ) -> dict:
        """
        Compute Bollinger Bands and return the upper, middle, and lower bands.

        The middle band is the simple moving average (SMA) of the last
        *period* closes.  The upper and lower bands are placed *num_std*
        population-standard-deviations above and below the middle band.

        Parameters
        ----------
        closes :
            Ordered list of closing prices (oldest first).
        period :
            Look-back window for the SMA and standard deviation.
        num_std :
            Number of standard deviations for the band width.

        Returns
        -------
        dict
            ``{"upper": float, "middle": float, "lower": float}``

        Raises
        ------
        ValueError
            If fewer than *period* closes are provided.
        """
        if len(closes) < period:
            raise ValueError(
                f"Need at least {period} closing prices to compute "
                f"Bollinger Bands, got {len(closes)}."
            )
        window = closes[-period:]
        middle = sum(window) / period
        variance = sum((p - middle) ** 2 for p in window) / period
        std = variance ** 0.5
        return {
            "upper": middle + num_std * std,
            "middle": middle,
            "lower": middle - num_std * std,
        }

    @staticmethod
    def _compute_keltner_channels(
        ohlcv: list, closes: list, period: int, multiplier: float
    ) -> dict:
        """
        Compute Keltner Channels and return the upper, middle, and lower lines.

        The middle line is the EMA of the last *period* closes.  The channel
        width is *multiplier* × ATR(*period*), so the upper channel is
        ``EMA + multiplier × ATR`` and the lower channel is
        ``EMA − multiplier × ATR``.

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles ``[timestamp, open, high, low, close, volume]``.
        closes :
            Ordered list of closing prices (oldest first, same length as
            *ohlcv*).
        period :
            EMA and ATR look-back window.
        multiplier :
            ATR multiplier for the channel width.

        Returns
        -------
        dict
            ``{"upper": float, "middle": float, "lower": float}``

        Raises
        ------
        ValueError
            If fewer than *period* closes or *period + 1* candles are provided.
        """
        if len(closes) < period:
            raise ValueError(
                f"Need at least {period} closing prices to compute "
                f"Keltner Channels, got {len(closes)}."
            )
        if len(ohlcv) < period + 1:
            raise ValueError(
                f"Need at least {period + 1} candles to compute "
                f"Keltner Channel ATR-{period}, got {len(ohlcv)}."
            )
        middle = CryptoTrader._compute_ema(closes, period)
        atr = CryptoTrader._compute_atr(ohlcv, period)
        return {
            "upper": middle + multiplier * atr,
            "middle": middle,
            "lower": middle - multiplier * atr,
        }

    @staticmethod
    def _compute_relative_volume(ohlcv: list, period: int) -> float:
        """
        Compute the Relative Volume (RVOL) for the most recent candle.

        RVOL = current_candle_volume / average_volume_over_prior_period_candles

        A value of 5.0 means the current candle is trading at 5× its normal
        pace — a strong signal that an unusual number of participants are
        active (e.g. a breakout or news-driven move).

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles ``[timestamp, open, high, low, close, volume]``.
            Must contain at least *period + 1* candles.
        period :
            Number of prior candles used to compute the average volume baseline.

        Returns
        -------
        float
            RVOL ratio.  Returns ``0.0`` when the average baseline volume is
            zero (avoids division by zero).

        Raises
        ------
        ValueError
            If fewer than *period + 1* candles are provided.
        """
        if len(ohlcv) < period + 1:
            raise ValueError(
                f"Need at least {period + 1} candles to compute "
                f"RVOL-{period}, got {len(ohlcv)}."
            )
        current_volume = float(ohlcv[-1][5])
        avg_volume = sum(float(c[5]) for c in ohlcv[-(period + 1):-1]) / period
        if avg_volume == 0.0:
            return 0.0
        return current_volume / avg_volume

    @staticmethod
    def _compute_atr(ohlcv: list, period: int = 14) -> float:
        """
        Compute the Average True Range (ATR) and return the last value.

        The True Range for each candle (after the first) is:
        ``max(high − low, |high − prev_close|, |low − prev_close|)``

        Uses Wilder's smoothing (RMA) seeded with the simple average of the
        first *period* true ranges.

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles
            ``[timestamp, open, high, low, close, volume]``.
        period :
            Look-back window for the ATR (default ``config.ATR_PERIOD``, 14).

        Returns
        -------
        float
            ATR value.

        Raises
        ------
        ValueError
            If fewer than *period + 1* candles are provided.
        """
        if len(ohlcv) < period + 1:
            raise ValueError(
                f"Need at least {period + 1} candles to compute "
                f"ATR-{period}, got {len(ohlcv)}."
            )
        true_ranges = []
        for i in range(1, len(ohlcv)):
            high = ohlcv[i][2]
            low = ohlcv[i][3]
            prev_close = ohlcv[i - 1][4]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        # Seed with simple average of first `period` true ranges
        atr = sum(true_ranges[:period]) / period
        # Wilder's smoothing for remaining true ranges
        for tr in true_ranges[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    def get_indicators(self, symbol: str, timeframe: str = "1h") -> dict:
        """
        Fetch OHLCV candles and return the latest indicator values.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).

        Returns
        -------
        dict
            ``{"price": float, "vwap": float, "rsi": float, "atr": float,
            "volume_profile_poc": float, "simple_algo_signal": bool,
            "bb_upper": float, "bb_middle": float, "bb_lower": float,
            "kc_upper": float, "kc_middle": float, "kc_lower": float,
            "rvol": float}``
        """
        limit = max(
            config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10,
            config.BB_PERIOD + config.RVOL_PERIOD + 10,
            config.KC_PERIOD + config.RVOL_PERIOD + 10,
        )
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        closes = [candle[4] for candle in ohlcv]  # index 4 = close price

        vwap = self._compute_vwap(ohlcv)
        rsi = self._compute_rsi(closes, config.RSI_PERIOD)
        atr = self._compute_atr(ohlcv, config.ATR_PERIOD)
        volume_profile = self._compute_volume_profile(ohlcv, config.VOLUME_PROFILE_BINS)
        simple_algo_signal = self._compute_simple_algo_signal(closes)
        bb = self._compute_bollinger_bands(closes, config.BB_PERIOD, config.BB_NUM_STD)
        kc = self._compute_keltner_channels(
            ohlcv, closes, config.KC_PERIOD, config.KC_MULTIPLIER
        )
        rvol = self._compute_relative_volume(ohlcv, config.RVOL_PERIOD)
        current_price = closes[-1]

        logger.debug(
            "Indicators %s — price=%.4f VWAP=%.4f RSI=%.2f ATR=%.6f "
            "VP_POC=%.4f algo_signal=%s BB_upper=%.4f KC_upper=%.4f RVOL=%.2f",
            symbol,
            current_price,
            vwap,
            rsi,
            atr,
            volume_profile["poc"],
            simple_algo_signal,
            bb["upper"],
            kc["upper"],
            rvol,
        )
        return {
            "price": current_price,
            "vwap": vwap,
            "rsi": rsi,
            "atr": atr,
            "volume_profile_poc": volume_profile["poc"],
            "simple_algo_signal": simple_algo_signal,
            "bb_upper": bb["upper"],
            "bb_middle": bb["middle"],
            "bb_lower": bb["lower"],
            "kc_upper": kc["upper"],
            "kc_middle": kc["middle"],
            "kc_lower": kc["lower"],
            "rvol": rvol,
        }

    def should_buy(self, symbol: str, timeframe: str = "1h") -> bool:
        """
        Return ``True`` when the buy signal fires.

        Buy conditions (both must be met):

        1. Relative Volume (RVOL) ≥ ``config.RVOL_THRESHOLD`` (default 5.0) →
           current candle volume is at least 5× the rolling average of the
           prior ``config.RVOL_PERIOD`` candles, confirming unusually high
           participation (the volume surge filter).
        2. Price is **above** the upper Bollinger Band
           (``config.BB_PERIOD``/``config.BB_NUM_STD``) **or** above the upper
           Keltner Channel (``config.KC_PERIOD``/``config.KC_MULTIPLIER``) →
           a confirmed volatility or momentum breakout (the breakout trigger).

        The minimum 24-hour volume check and the bid-ask spread check must
        also pass.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).
        """
        indicators = self.get_indicators(symbol, timeframe)
        price = indicators["price"]

        rvol_high = indicators["rvol"] >= config.RVOL_THRESHOLD
        breakout = price > indicators["bb_upper"] or price > indicators["kc_upper"]

        try:
            self._check_volume(symbol)
            volume_ok = True
        except InsufficientVolumeError as exc:
            logger.warning("should_buy %s — volume check failed: %s", symbol, exc)
            volume_ok = False

        try:
            self._check_spread(symbol)
            spread_ok = True
        except WideBidAskSpreadError as exc:
            logger.warning("should_buy %s — spread check failed: %s", symbol, exc)
            spread_ok = False

        signal = rvol_high and breakout and volume_ok and spread_ok

        logger.info(
            "should_buy %s — rvol=%.2f rvol_high=%s price=%.4f bb_upper=%.4f "
            "kc_upper=%.4f breakout=%s volume_ok=%s spread_ok=%s → signal=%s",
            symbol,
            indicators["rvol"],
            rvol_high,
            price,
            indicators["bb_upper"],
            indicators["kc_upper"],
            breakout,
            volume_ok,
            spread_ok,
            signal,
        )
        return signal

    def should_sell(self, symbol: str, timeframe: str = "1h") -> bool:
        """
        Return ``True`` when the RSI-based take-profit exit signal fires.

        Sell condition:

        * RSI is **above** ``config.RSI_OVERBOUGHT`` (default 70) →
          the asset is overbought / expensive; take profit.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).

        Returns
        -------
        bool
            ``True`` when RSI > ``config.RSI_OVERBOUGHT``.
        """
        indicators = self.get_indicators(symbol, timeframe)
        rsi_overbought = indicators["rsi"] > config.RSI_OVERBOUGHT

        logger.info(
            "should_sell %s — rsi=%.2f → rsi_overbought=%s",
            symbol,
            indicators["rsi"],
            rsi_overbought,
        )
        return rsi_overbought

    def get_holdings(self) -> dict:
        """
        Return current non-USD crypto holdings from the exchange.

        Queries the exchange balance and returns every non-USD asset with a
        positive free quantity, expressed as ``"BASE/USD"`` trading-pair keys.

        Meta-keys returned by ccxt alongside per-currency dicts (``"info"``,
        ``"free"``, ``"used"``, ``"total"``, ``"datetime"``, ``"timestamp"``)
        are automatically ignored.

        Returns
        -------
        dict[str, dict]
            Mapping of ``"BASE/USD"`` symbol → ``{"quantity": float}`` for
            every non-USD asset with a positive free balance.
        """
        balance = self.exchange.fetch_balance()
        _meta_keys = {"info", "free", "used", "total", "datetime", "timestamp"}
        holdings: dict = {}
        for currency, amounts in balance.items():
            if currency in _meta_keys or currency == "USD":
                continue
            if not isinstance(amounts, dict):
                continue
            free = amounts.get("free")
            if free is None or float(free) <= 0:
                continue
            symbol = f"{currency}/USD"
            holdings[symbol] = {"quantity": float(free)}
            logger.debug("get_holdings — %s free=%.8f", symbol, float(free))
        logger.info("get_holdings — found %d non-USD position(s)", len(holdings))
        return holdings

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
            * ``"take_profit"`` — current price ≥ entry × (1 + 5 %)
            * ``"stop_loss"``   — current price ≤ entry × (1 − 1.75 %)
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

    def place_exit_orders(
        self, symbol: str, quantity: float, entry_price: float
    ) -> dict:
        """
        Place take-profit and stop-loss sell orders immediately after a buy.

        A **limit sell** is placed at the take-profit price
        (``entry_price × (1 + config.TAKE_PROFIT_PCT)``) and a
        **stop-market sell** is placed at the ATR-based stop-loss price
        (``entry_price − config.ATR_STOP_LOSS_MULTIPLIER × ATR``), adapting
        risk to current market volatility.

        ATR is computed from the most recent ``config.ATR_PERIOD + 1`` hourly
        candles fetched at call time.

        When the exchange fills one of the two orders the caller should
        cancel the other via :meth:`check_exit_orders`, which handles that
        housekeeping automatically.

        In **paper-trading** mode no real orders are submitted; the method
        returns simulated order dicts with ``"paper": True`` and
        ``"id": None`` so that :meth:`check_exit_orders` falls back to
        price-based polling for those positions.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        quantity :
            Number of units to sell when the TP or SL is triggered.
        entry_price :
            Fill price of the preceding buy order.

        Returns
        -------
        dict
            ``{
              "take_profit_order_id": str | None,
              "stop_loss_order_id":   str | None,
              "take_profit_price":    float,
              "stop_loss_price":      float,
            }``

        Raises
        ------
        ValueError
            If *entry_price* or *quantity* is not positive, or *symbol*
            is not a USD-quoted pair.
        """
        self._validate_usd_pair(symbol)
        if entry_price <= 0:
            raise ValueError(
                f"entry_price must be positive, got {entry_price}."
            )
        if quantity <= 0:
            raise ValueError(
                f"quantity must be positive, got {quantity}."
            )

        # Compute ATR for dynamic stop-loss sizing
        atr_ohlcv = self.exchange.fetch_ohlcv(
            symbol, "1h", limit=config.ATR_PERIOD + 1
        )
        atr = self._compute_atr(atr_ohlcv, config.ATR_PERIOD)

        take_profit_price = entry_price * (1.0 + config.TAKE_PROFIT_PCT)
        stop_loss_price = entry_price - config.ATR_STOP_LOSS_MULTIPLIER * atr

        logger.info(
            "place_exit_orders %s — qty=%.8f entry=%.6f TP=%.6f SL=%.6f (paper=%s)",
            symbol,
            quantity,
            entry_price,
            take_profit_price,
            stop_loss_price,
            self.paper_trading,
        )

        if self.paper_trading:
            return {
                "take_profit_order_id": None,
                "stop_loss_order_id": None,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_loss_price,
            }

        # Place limit sell for take-profit
        tp_order = self.exchange.create_order(
            symbol, "limit", "sell", quantity, take_profit_price
        )
        tp_order_id = tp_order.get("id")
        logger.info(
            "place_exit_orders %s — TP limit-sell placed id=%s @ %.6f",
            symbol,
            tp_order_id,
            take_profit_price,
        )

        # Place stop-market sell for stop-loss
        try:
            sl_order = self.exchange.create_order(
                symbol,
                "stop",
                "sell",
                quantity,
                stop_loss_price,
                {"stopPrice": stop_loss_price},
            )
        except Exception:
            # Some exchanges use a different order-type name for stop-market.
            # Fall back to "stopMarket" which is the ccxt unified alias.
            sl_order = self.exchange.create_order(
                symbol,
                "stopMarket",
                "sell",
                quantity,
                stop_loss_price,
                {"stopPrice": stop_loss_price},
            )
        sl_order_id = sl_order.get("id")
        logger.info(
            "place_exit_orders %s — SL stop-market placed id=%s @ %.6f",
            symbol,
            sl_order_id,
            stop_loss_price,
        )

        return {
            "take_profit_order_id": tp_order_id,
            "stop_loss_order_id": sl_order_id,
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
        }

    def check_exit_orders(
        self,
        symbol: str,
        tp_order_id: Optional[str],
        sl_order_id: Optional[str],
        entry_price: float,
    ) -> str:
        """
        Check whether the exchange-side take-profit or stop-loss order has
        filled, cancel the survivor, and report the outcome.

        When both *tp_order_id* and *sl_order_id* are ``None`` (paper-
        trading or orders that could not be placed) the method falls back to
        the price-based :meth:`check_exit` logic.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        tp_order_id :
            Exchange order ID for the take-profit limit-sell order, or
            ``None`` to skip exchange-order checking.
        sl_order_id :
            Exchange order ID for the stop-loss stop-market order, or
            ``None`` to skip exchange-order checking.
        entry_price :
            Entry price used by the price-based fallback.

        Returns
        -------
        str
            * ``"take_profit"`` — TP order filled (SL order cancelled).
            * ``"stop_loss"``   — SL order filled (TP order cancelled).
            * ``"hold"``        — neither order has filled yet.

        Raises
        ------
        ValueError
            If *entry_price* is not positive (fallback path only).
        """
        # Paper-trading or missing order IDs → use price-based polling
        if tp_order_id is None and sl_order_id is None:
            return self.check_exit(symbol, entry_price)

        def _is_filled(order_id: Optional[str]) -> bool:
            if order_id is None:
                return False
            try:
                order = self.exchange.fetch_order(order_id, symbol)
                return order.get("status") == "closed"
            except Exception as exc:
                logger.warning(
                    "check_exit_orders %s — could not fetch order %s: %s",
                    symbol,
                    order_id,
                    exc,
                )
                return False

        def _cancel_order(order_id: Optional[str]) -> None:
            if order_id is None:
                return
            try:
                self.exchange.cancel_order(order_id, symbol)
                logger.info(
                    "check_exit_orders %s — cancelled order %s",
                    symbol,
                    order_id,
                )
            except Exception as exc:
                logger.warning(
                    "check_exit_orders %s — could not cancel order %s: %s",
                    symbol,
                    order_id,
                    exc,
                )

        if _is_filled(tp_order_id):
            _cancel_order(sl_order_id)
            logger.info(
                "check_exit_orders %s — take-profit order %s filled",
                symbol,
                tp_order_id,
            )
            return "take_profit"

        if _is_filled(sl_order_id):
            _cancel_order(tp_order_id)
            logger.info(
                "check_exit_orders %s — stop-loss order %s filled",
                symbol,
                sl_order_id,
            )
            return "stop_loss"

        return "hold"
