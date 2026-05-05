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
Buy signal
~~~~~~~~~~
Both of the following conditions must be true:

1. **200 EMA** — the current closing price is above the 200-period EMA,
   confirming the asset is in a long-term uptrend.
2. **13/48 EMA alignment** — the 13-period EMA is above the 48-period
   EMA, indicating bullish momentum.

Mandatory gates (must pass for a buy signal to fire):

- 24-hour quote volume ≥ ``config.MIN_VOLUME_USD`` ($15 000).
- Bid-ask spread < ``config.MAX_BID_ASK_SPREAD_PCT`` (0.75 %).

Exit
~~~~
Positions are closed automatically by price-level orders:

- Take-profit target : 6.5 % above entry (``config.TAKE_PROFIT_PCT``).
- Stop-loss          : 1.75 % below entry (``config.STOP_LOSS_PCT``).
- Time-based exit    : position is closed after 24 hours regardless.

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
import threading
import time
from typing import Optional

import ccxt

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monotonically increasing nonce
# ---------------------------------------------------------------------------
# Kraken requires each authenticated request to carry a nonce that is strictly
# greater than the previous one.  The ccxt default (millisecond timestamp) can
# produce duplicate nonces when two requests are dispatched within the same
# millisecond, causing "EAPI:Invalid nonce" errors.  The lock-protected counter
# below guarantees uniqueness: it always advances to at least the current
# millisecond, but never goes backward or repeats.
_nonce_lock = threading.Lock()
_nonce_state = [0]  # mutable container so the closure can update it


def _make_nonce() -> int:
    with _nonce_lock:
        now = int(time.time() * 1000)
        _nonce_state[0] = max(now, _nonce_state[0]) + 1
        return _nonce_state[0]


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
                "enableRateLimit": True,
                "nonce": _make_nonce,
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
        Fetch OHLCV candles and return the latest EMA indicator values.

        Fetches enough candles to compute the 200-period EMA plus one extra
        candle so that the previous-bar EMA values for the 13/48 crossover
        can also be derived.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).

        Returns
        -------
        dict
            ``{"price": float,
            "ema_200": float,
            "ema_13": float, "ema_48": float,
            "prev_ema_13": float, "prev_ema_48": float}``
        """
        # Fetch 3× the EMA period so the 200-period EMA has enough warm-up
        # candles to track current price accurately.  A seed of only
        # EMA_PERIOD candles leaves the EMA anchored to older prices because
        # the EMA-200 multiplier (≈0.01) requires hundreds of additional bars
        # to converge; too few warm-up bars cause the EMA to sit well above
        # current prices during a recent pullback, permanently blocking buys.
        limit = config.EMA_PERIOD * 3
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        closes = [candle[4] for candle in ohlcv]  # index 4 = close price

        ema_200 = self._compute_ema(closes, config.EMA_PERIOD)
        ema_13 = self._compute_ema(closes, config.EMA_CROSS_SHORT_PERIOD)
        ema_48 = self._compute_ema(closes, config.EMA_CROSS_LONG_PERIOD)
        prev_closes = closes[:-1]
        prev_ema_13 = self._compute_ema(prev_closes, config.EMA_CROSS_SHORT_PERIOD)
        prev_ema_48 = self._compute_ema(prev_closes, config.EMA_CROSS_LONG_PERIOD)
        current_price = closes[-1]

        logger.debug(
            "Indicators %s — price=%.4f EMA200=%.4f EMA13=%.4f EMA48=%.4f "
            "prev_EMA13=%.4f prev_EMA48=%.4f",
            symbol,
            current_price,
            ema_200,
            ema_13,
            ema_48,
            prev_ema_13,
            prev_ema_48,
        )
        return {
            "price": current_price,
            "ema_200": ema_200,
            "ema_13": ema_13,
            "ema_48": ema_48,
            "prev_ema_13": prev_ema_13,
            "prev_ema_48": prev_ema_48,
        }

    def should_buy(self, symbol: str, timeframe: str = "1h") -> bool:
        """
        Return ``True`` when the EMA-based buy signal fires.

        Both of the following conditions must be true:

        1. **200 EMA** — the current closing price is above the 200-period
           EMA, confirming a long-term uptrend.
        2. **13/48 EMA alignment** — the 13-period EMA is above the
           48-period EMA, indicating bullish momentum.

        Additionally, the following mandatory gates must pass:

        - 24-hour quote volume ≥ ``config.MIN_VOLUME_USD``.
        - Bid-ask spread < ``config.MAX_BID_ASK_SPREAD_PCT``.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"BTC/USD"``.
        timeframe :
            Candle interval accepted by the exchange (default ``"1h"``).

        Returns
        -------
        bool
            ``True`` when both EMA conditions are met and all gates pass.
        """
        indicators = self.get_indicators(symbol, timeframe)
        price = indicators["price"]

        # Condition 1: price is above the 200-period EMA
        ema_200_bullish = price > indicators["ema_200"]

        # Condition 2: 13-period EMA is above the 48-period EMA (bullish alignment)
        ema_cross_bullish = indicators["ema_13"] > indicators["ema_48"]

        signal_ok = ema_200_bullish and ema_cross_bullish

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

        signal = signal_ok and volume_ok and spread_ok

        logger.info(
            "should_buy %s — price=%.4f EMA200=%.4f(bull=%s) "
            "EMA13=%.4f EMA48=%.4f(aligned=%s) "
            "volume_ok=%s spread_ok=%s → signal=%s",
            symbol,
            price,
            indicators["ema_200"], ema_200_bullish,
            indicators["ema_13"], indicators["ema_48"], ema_cross_bullish,
            volume_ok, spread_ok,
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
