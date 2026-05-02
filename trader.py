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
Both ``should_buy`` and ``should_sell`` use a **comprehensive five-indicator
scoring system**.  Each indicator contributes one point toward a score;
the signal fires when the score meets or exceeds the configured threshold.

Buy signal
~~~~~~~~~~
Scored conditions (3 out of 5 required by default):

1. **RSI** < ``config.RSI_OVERSOLD`` (35) — approaching oversold territory.
2. **WaveTrend** WT1 > WT2 and WT1 < overbought level — bullish momentum.
3. **CCI** crosses above ``config.CCI_OVERSOLD`` (−100) from below — recovering from oversold.
4. **ADX** > ``config.ADX_THRESHOLD`` (20) and +DI > −DI — uptrend strength.
5. **Kernel Filter** — price ≥ kernel-smoothed regression line (upward bias).

Mandatory gates (must all pass regardless of score):

- RVOL ≥ ``config.RVOL_THRESHOLD`` (1.5×) — above-average volume confirmation.
- 24-hour quote volume ≥ ``config.MIN_VOLUME_USD`` ($15 000).
- Bid-ask spread < ``config.MAX_BID_ASK_SPREAD_PCT`` (0.75 %).

Sell signal
~~~~~~~~~~~
Scored conditions (3 out of 5 required by default):

1. **RSI** > ``config.RSI_OVERBOUGHT`` (70) — overbought; take profit.
2. **WaveTrend** WT1 < WT2 **or** WT1 above overbought level — momentum fading.
3. **CCI** > ``config.CCI_OVERBOUGHT`` (+100) — overbought on CCI.
4. **ADX** > ``config.ADX_THRESHOLD`` (20) and −DI > +DI — downtrend strength.
5. **Kernel Filter** — price < kernel-smoothed regression line (downward bias).

- Execution (Volatility) : stop-loss is placed at
  ``entry_price − config.ATR_STOP_LOSS_MULTIPLIER × ATR``
  (default 1.5 × ATR), adapting risk to current market speed.
- Take-profit target is 5.5 % above entry; stop-loss is 1.75 % below entry.

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
        **population** standard deviations above and below the middle band
        (variance divided by *period*, not *period − 1*).

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

    @staticmethod
    def _compute_wavetrend(
        ohlcv: list,
        n1: int = 10,
        n2: int = 21,
        ma_len: int = 4,
    ) -> dict:
        """
        Compute the WaveTrend oscillator and return WT1 and WT2.

        WaveTrend uses the HLC3 (typical price) to build two oscillator lines:

        * ``WT1`` — the smoothed channel indicator (EMA of a normalised
          deviation of HLC3 from its own EMA).
        * ``WT2`` — a simple moving average of WT1 used as the trigger line.

        A bullish crossover occurs when WT1 crosses above WT2; a bearish
        crossover occurs when WT1 crosses below WT2.

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles
            ``[timestamp, open, high, low, close, volume]``.
        n1 :
            Channel period — EMA length used to smooth HLC3 (default 10).
        n2 :
            Average period — EMA length applied to the normalised CI series
            to produce WT1 (default 21).
        ma_len :
            SMA length used to derive WT2 from WT1 (default 4).

        Returns
        -------
        dict
            ``{"wt1": float, "wt2": float}`` — the final WT1 and WT2 values.

        Raises
        ------
        ValueError
            If fewer than ``n1 + n2 + ma_len`` candles are provided.
        """
        min_candles = n1 + n2 + ma_len
        if len(ohlcv) < min_candles:
            raise ValueError(
                f"Need at least {min_candles} candles to compute WaveTrend "
                f"(n1={n1}, n2={n2}, ma_len={ma_len}), got {len(ohlcv)}."
            )

        hlc3 = [(c[2] + c[3] + c[4]) / 3.0 for c in ohlcv]

        # EMA of HLC3 over n1 (esa)
        multiplier_n1 = 2.0 / (n1 + 1)
        esa = sum(hlc3[:n1]) / n1
        esa_series = [esa]
        for price in hlc3[n1:]:
            esa = price * multiplier_n1 + esa * (1.0 - multiplier_n1)
            esa_series.append(esa)

        # Rebuild a full-length esa list (first n1-1 entries are unavailable;
        # pad with the seed value so indices align with hlc3).
        esa_full = [esa_series[0]] * (n1 - 1) + esa_series

        # EMA of |HLC3 - esa| over n1 (d)
        abs_dev = [abs(hlc3[i] - esa_full[i]) for i in range(len(hlc3))]
        d = sum(abs_dev[:n1]) / n1
        d_series = [d]
        for dev in abs_dev[n1:]:
            d = dev * multiplier_n1 + d * (1.0 - multiplier_n1)
            d_series.append(d)
        d_full = [d_series[0]] * (n1 - 1) + d_series

        # Channel Index: ci = (hlc3 - esa) / (0.015 * d)
        ci = []
        for i in range(len(hlc3)):
            denom = 0.015 * d_full[i]
            ci.append((hlc3[i] - esa_full[i]) / denom if denom != 0.0 else 0.0)

        # WT1 = EMA(ci, n2)
        multiplier_n2 = 2.0 / (n2 + 1)
        wt1_val = sum(ci[:n2]) / n2
        wt1_series = [wt1_val]
        for val in ci[n2:]:
            wt1_val = val * multiplier_n2 + wt1_val * (1.0 - multiplier_n2)
            wt1_series.append(wt1_val)

        # WT2 = SMA(WT1, ma_len)  — computed from the last ma_len WT1 values
        wt2_val = sum(wt1_series[-ma_len:]) / ma_len

        return {"wt1": wt1_series[-1], "wt2": wt2_val}

    @staticmethod
    def _compute_cci(ohlcv: list, period: int = 20) -> float:
        """
        Compute the Commodity Channel Index (CCI) and return the latest value.

        CCI = (typical_price − SMA(typical_price, period)) /
              (0.015 × mean_deviation)

        where ``typical_price = (high + low + close) / 3`` and
        ``mean_deviation`` is the mean of ``|typical_price − SMA|`` over the
        same *period*.

        CCI above +100 indicates overbought conditions; below −100 indicates
        oversold conditions.

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles
            ``[timestamp, open, high, low, close, volume]``.
        period :
            Look-back window (default 20).

        Returns
        -------
        float
            CCI value for the most recent candle.

        Raises
        ------
        ValueError
            If fewer than *period* candles are provided.
        """
        if len(ohlcv) < period:
            raise ValueError(
                f"Need at least {period} candles to compute CCI-{period}, "
                f"got {len(ohlcv)}."
            )
        window = ohlcv[-period:]
        typical_prices = [(c[2] + c[3] + c[4]) / 3.0 for c in window]
        sma = sum(typical_prices) / period
        mean_dev = sum(abs(tp - sma) for tp in typical_prices) / period
        if mean_dev == 0.0:
            return 0.0
        return (typical_prices[-1] - sma) / (0.015 * mean_dev)

    @staticmethod
    def _compute_adx(ohlcv: list, period: int = 14) -> dict:
        """
        Compute the Average Directional Index (ADX) and return ADX, +DI, −DI.

        Uses Wilder's smoothing for the True Range, Directional Movement, and
        DX series — the same approach used in the standard ADX calculation.

        * ``adx``      — trend strength (0–100; values above 20–25 indicate a
          trending market).
        * ``plus_di``  — positive directional indicator (+DI); when +DI > −DI
          the trend is upward.
        * ``minus_di`` — negative directional indicator (−DI).

        Parameters
        ----------
        ohlcv :
            List of OHLCV candles
            ``[timestamp, open, high, low, close, volume]``.
        period :
            Wilder smoothing period (default 14).

        Returns
        -------
        dict
            ``{"adx": float, "plus_di": float, "minus_di": float}``

        Raises
        ------
        ValueError
            If fewer than ``2 * period + 1`` candles are provided (the minimum
            needed to seed +DM/-DM smoothing and then compute ADX).
        """
        min_candles = 2 * period + 1
        if len(ohlcv) < min_candles:
            raise ValueError(
                f"Need at least {min_candles} candles to compute ADX-{period}, "
                f"got {len(ohlcv)}."
            )

        tr_list, plus_dm_list, minus_dm_list = [], [], []
        for i in range(1, len(ohlcv)):
            high, low, close = ohlcv[i][2], ohlcv[i][3], ohlcv[i][4]
            prev_high = ohlcv[i - 1][2]
            prev_low  = ohlcv[i - 1][3]
            prev_close = ohlcv[i - 1][4]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            up_move   = high - prev_high
            down_move = prev_low - low
            pdm = up_move   if up_move > down_move and up_move > 0   else 0.0
            mdm = down_move if down_move > up_move and down_move > 0 else 0.0

            tr_list.append(tr)
            plus_dm_list.append(pdm)
            minus_dm_list.append(mdm)

        # Seed Wilder's smoothed values with the simple average of the first
        # `period` values, then apply Wilder's update for the rest.
        atr_w  = sum(tr_list[:period]) / period
        pdm_w  = sum(plus_dm_list[:period]) / period
        mdm_w  = sum(minus_dm_list[:period]) / period

        dx_list = []
        for i in range(period, len(tr_list)):
            atr_w = (atr_w * (period - 1) + tr_list[i])  / period
            pdm_w = (pdm_w * (period - 1) + plus_dm_list[i])  / period
            mdm_w = (mdm_w * (period - 1) + minus_dm_list[i]) / period

            plus_di  = 100.0 * pdm_w / atr_w if atr_w != 0.0 else 0.0
            minus_di = 100.0 * mdm_w / atr_w if atr_w != 0.0 else 0.0
            di_sum = plus_di + minus_di
            dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum != 0.0 else 0.0
            dx_list.append((dx, plus_di, minus_di))

        # ADX = Wilder's smoothing of DX over `period` values
        adx = sum(d[0] for d in dx_list[:period]) / period
        for dx, plus_di_val, minus_di_val in dx_list[period:]:
            adx = (adx * (period - 1) + dx) / period
        # The last computed +DI/-DI represent the final bar
        final_plus_di  = dx_list[-1][1]
        final_minus_di = dx_list[-1][2]

        return {"adx": adx, "plus_di": final_plus_di, "minus_di": final_minus_di}

    @staticmethod
    def _compute_kernel_filter(closes: list, bandwidth: int = 8) -> float:
        """
        Compute a Rational Quadratic kernel-weighted regression estimate.

        The kernel assigns weights to the most recent *bandwidth* closing prices
        using the Rational Quadratic (RQ) kernel function:

            K(i) = (1 + i² / (2 × alpha × bandwidth²))^(−alpha)

        with ``alpha = 1``, which decays smoothly from 1 at distance 0 toward
        0 at large distances.  The result is a noise-reduced estimate of the
        current price level (a non-parametric trend line).

        * Price **above** the kernel line → upward trend bias (buy-friendly).
        * Price **below** the kernel line → downward trend bias (sell-friendly).

        Parameters
        ----------
        closes :
            Ordered list of closing prices (oldest first).
        bandwidth :
            Number of prior bars used as the kernel window (default 8).

        Returns
        -------
        float
            Kernel-smoothed price estimate at the most recent bar.

        Raises
        ------
        ValueError
            If fewer than *bandwidth* closing prices are provided.
        """
        if len(closes) < bandwidth:
            raise ValueError(
                f"Need at least {bandwidth} closing prices for Kernel Filter "
                f"(bandwidth={bandwidth}), got {len(closes)}."
            )
        alpha = 1.0
        window = closes[-bandwidth:]
        weights = [
            (1.0 + (i * i) / (2.0 * alpha * bandwidth * bandwidth)) ** (-alpha)
            for i in range(bandwidth)
        ]
        total_weight = sum(weights)
        kernel_val = sum(w * p for w, p in zip(weights, reversed(window)))
        return kernel_val / total_weight

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
            "rvol": float,
            "wt1": float, "wt2": float,
            "cci": float, "prev_cci": float,
            "adx": float, "plus_di": float, "minus_di": float,
            "kernel": float}``
        """
        wt_min = config.WT_CHANNEL_LENGTH + config.WT_AVERAGE_LENGTH + config.WT_MA_LENGTH
        adx_min = 2 * config.ADX_PERIOD + 1
        limit = max(
            config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10,
            config.BB_PERIOD + config.RVOL_PERIOD + 10,
            config.KC_PERIOD + config.RVOL_PERIOD + 10,
            wt_min + 10,
            adx_min + 10,
            config.CCI_PERIOD + 11,  # +1 for prev_cci (uses ohlcv[:-1])
            config.KERNEL_BANDWIDTH + 10,
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
        wt = self._compute_wavetrend(
            ohlcv, config.WT_CHANNEL_LENGTH, config.WT_AVERAGE_LENGTH, config.WT_MA_LENGTH
        )
        cci = self._compute_cci(ohlcv, config.CCI_PERIOD)
        prev_cci = self._compute_cci(ohlcv[:-1], config.CCI_PERIOD)
        adx_result = self._compute_adx(ohlcv, config.ADX_PERIOD)
        kernel = self._compute_kernel_filter(closes, config.KERNEL_BANDWIDTH)
        current_price = closes[-1]

        logger.debug(
            "Indicators %s — price=%.4f VWAP=%.4f RSI=%.2f ATR=%.6f "
            "VP_POC=%.4f algo_signal=%s BB_upper=%.4f KC_upper=%.4f RVOL=%.2f "
            "WT1=%.2f WT2=%.2f CCI=%.2f ADX=%.2f +DI=%.2f -DI=%.2f kernel=%.4f",
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
            wt["wt1"],
            wt["wt2"],
            cci,
            adx_result["adx"],
            adx_result["plus_di"],
            adx_result["minus_di"],
            kernel,
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
            "wt1": wt["wt1"],
            "wt2": wt["wt2"],
            "cci": cci,
            "prev_cci": prev_cci,
            "adx": adx_result["adx"],
            "plus_di": adx_result["plus_di"],
            "minus_di": adx_result["minus_di"],
            "kernel": kernel,
        }

    def should_buy(self, symbol: str, timeframe: str = "1h") -> bool:
        """
        Return ``True`` when the comprehensive buy signal fires.

        The signal combines five momentum/trend indicators (each contributing
        one point toward a buy score) with mandatory volume and breakout gates:

        **Scored conditions** (each worth 1 point; scored against
        ``config.BUY_SIGNAL_THRESHOLD``, default 3 out of 5):

        1. **RSI** < ``config.RSI_OVERSOLD`` (default 35) — price is approaching
           oversold territory; momentum has declined enough to warrant attention.
        2. **WaveTrend** — WT1 > WT2 (bullish momentum crossover direction)
           and WT1 is below the overbought level (``config.WT_OVERBOUGHT``).
        3. **CCI** crosses above ``config.CCI_OVERSOLD`` (default −100) from
           below — the previous candle was in oversold territory and the
           current candle has recovered above it.
        4. **ADX** > ``config.ADX_THRESHOLD`` (default 20) and +DI > −DI —
           a trending market with upward directional bias.
        5. **Kernel Filter** — current price ≥ kernel-smoothed regression line,
           confirming the trend direction is upward.

        **Mandatory gates** (all must pass regardless of score):

        * RVOL ≥ ``config.RVOL_THRESHOLD`` (default 1.5) — above-average
          volume confirms participation in the move.
        * 24-hour quote volume ≥ ``config.MIN_VOLUME_USD`` — sufficient
          liquidity.
        * Bid-ask spread < ``config.MAX_BID_ASK_SPREAD_PCT`` — tight market.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).

        Returns
        -------
        bool
            ``True`` when buy_score ≥ ``config.BUY_SIGNAL_THRESHOLD`` and all
            mandatory gates pass.
        """
        indicators = self.get_indicators(symbol, timeframe)
        price = indicators["price"]

        # --- Scored indicator conditions (1 point each) ---
        rsi_bullish = indicators["rsi"] < config.RSI_OVERSOLD
        wt_bullish  = (
            indicators["wt1"] > indicators["wt2"]
            and indicators["wt1"] < config.WT_OVERBOUGHT
        )
        cci_bullish  = (
            indicators["prev_cci"] < config.CCI_OVERSOLD
            and indicators["cci"] > config.CCI_OVERSOLD
        )
        adx_bullish  = (
            indicators["adx"] > config.ADX_THRESHOLD
            and indicators["plus_di"] > indicators["minus_di"]
        )
        kernel_bullish = price >= indicators["kernel"]

        buy_score = sum([rsi_bullish, wt_bullish, cci_bullish, adx_bullish, kernel_bullish])
        score_ok = buy_score >= config.BUY_SIGNAL_THRESHOLD

        # --- Mandatory gates ---
        rvol_high = indicators["rvol"] >= config.RVOL_THRESHOLD

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

        signal = score_ok and rvol_high and volume_ok and spread_ok

        logger.info(
            "should_buy %s — rsi=%.2f(bull=%s) wt1=%.2f wt2=%.2f(bull=%s) "
            "prev_cci=%.2f cci=%.2f(bull=%s) adx=%.2f +di=%.2f -di=%.2f(bull=%s) "
            "kernel=%.4f(bull=%s) score=%d/%d score_ok=%s "
            "rvol=%.2f(ok=%s) volume_ok=%s spread_ok=%s → signal=%s",
            symbol,
            indicators["rsi"], rsi_bullish,
            indicators["wt1"], indicators["wt2"], wt_bullish,
            indicators["prev_cci"], indicators["cci"], cci_bullish,
            indicators["adx"], indicators["plus_di"], indicators["minus_di"], adx_bullish,
            indicators["kernel"], kernel_bullish,
            buy_score, config.BUY_SIGNAL_THRESHOLD, score_ok,
            indicators["rvol"], rvol_high,
            volume_ok, spread_ok,
            signal,
        )
        return signal

    def should_sell(self, symbol: str, timeframe: str = "1h") -> bool:
        """
        Return ``True`` when the comprehensive sell signal fires.

        The signal combines five momentum/trend indicators (each contributing
        one point toward a sell score):

        **Scored conditions** (each worth 1 point; scored against
        ``config.SELL_SIGNAL_THRESHOLD``, default 3 out of 5):

        1. **RSI** > ``config.RSI_OVERBOUGHT`` (default 70) — asset is
           overbought; take-profit opportunity.
        2. **WaveTrend** — WT1 < WT2 (bearish direction) **or** WT1 is above
           the overbought level (``config.WT_OVERBOUGHT``), signalling momentum
           exhaustion.
        3. **CCI** > ``config.CCI_OVERBOUGHT`` (default +100) — price is in
           overbought territory on the Commodity Channel Index.
        4. **ADX** > ``config.ADX_THRESHOLD`` (default 20) and −DI > +DI —
           a trending market with downward directional bias.
        5. **Kernel Filter** — current price < kernel-smoothed regression line,
           indicating the trend direction has turned downward.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).

        Returns
        -------
        bool
            ``True`` when sell_score ≥ ``config.SELL_SIGNAL_THRESHOLD``.
        """
        indicators = self.get_indicators(symbol, timeframe)
        price = indicators["price"]

        rsi_bearish    = indicators["rsi"] > config.RSI_OVERBOUGHT
        wt_bearish     = (
            indicators["wt1"] < indicators["wt2"]
            or indicators["wt1"] > config.WT_OVERBOUGHT
        )
        cci_bearish    = indicators["cci"] > config.CCI_OVERBOUGHT
        adx_bearish    = (
            indicators["adx"] > config.ADX_THRESHOLD
            and indicators["minus_di"] > indicators["plus_di"]
        )
        kernel_bearish = price < indicators["kernel"]

        sell_score = sum([rsi_bearish, wt_bearish, cci_bearish, adx_bearish, kernel_bearish])
        signal = sell_score >= config.SELL_SIGNAL_THRESHOLD

        logger.info(
            "should_sell %s — rsi=%.2f(bear=%s) wt1=%.2f wt2=%.2f(bear=%s) "
            "cci=%.2f(bear=%s) adx=%.2f +di=%.2f -di=%.2f(bear=%s) "
            "kernel=%.4f(bear=%s) score=%d/%d → signal=%s",
            symbol,
            indicators["rsi"], rsi_bearish,
            indicators["wt1"], indicators["wt2"], wt_bearish,
            indicators["cci"], cci_bearish,
            indicators["adx"], indicators["plus_di"], indicators["minus_di"], adx_bearish,
            indicators["kernel"], kernel_bearish,
            sell_score, config.SELL_SIGNAL_THRESHOLD,
            signal,
        )
        return signal

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
            * ``"take_profit"`` — current price ≥ entry × (1 + 5.5 %)
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
