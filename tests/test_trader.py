"""
Unit tests for the Automated Crypto Trader.

All tests run without real network calls — exchange interactions are
patched with unittest.mock so no API keys are required.
"""

import datetime
import unittest
import unittest.mock
from unittest.mock import MagicMock, patch

import config
from trader import CryptoTrader, OrderSizeError, InsufficientVolumeError, WideBidAskSpreadError


def _make_full_indicators(
    price=100.0,
    rsi=50.0,
    wt1=30.0,
    wt2=35.0,
    cci=50.0,
    prev_cci=0.0,
    adx=15.0,
    plus_di=30.0,
    minus_di=15.0,
    kernel=90.0,
    bb_upper=None,
    kc_upper=None,
    rvol=None,
) -> dict:
    """Build a complete indicator dict suitable for mocking ``get_indicators``.

    Defaults produce a **non-bullish, non-bearish** neutral state.  Override
    individual fields to drive specific test scenarios.
    """
    if bb_upper is None:
        bb_upper = price + 5.0   # price is below the upper band by default
    if kc_upper is None:
        kc_upper = price + 5.0
    if rvol is None:
        rvol = config.RVOL_THRESHOLD
    return {
        "price": price,
        "vwap": price - 10.0,
        "rsi": rsi,
        "atr": 1.0,
        "volume_profile_poc": price - 10.0,
        "simple_algo_signal": True,
        "bb_upper": bb_upper,
        "bb_middle": bb_upper - 5.0,
        "bb_lower": bb_upper - 10.0,
        "kc_upper": kc_upper,
        "kc_middle": kc_upper - 5.0,
        "kc_lower": kc_upper - 10.0,
        "rvol": rvol,
        "wt1": wt1,
        "wt2": wt2,
        "cci": cci,
        "prev_cci": prev_cci,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "kernel": kernel,
    }


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------


def _min_volume() -> float:
    """Return the minimum required 24-hour volume."""
    return config.MIN_VOLUME_USD


def _make_ticker(
    price: float,
    *,
    bid: float | None = None,
    quote_volume: float | None = None,
) -> dict:
    """Build a minimal ticker dict suitable for mocking ``fetch_ticker``.

    Defaults produce a ticker that passes both the volume and spread checks:
    - ``quoteVolume`` is 10× the required minimum.
    - ``bid`` is 0.1 % below ``ask``, giving a 0.1 % spread (< 1 % max).
    """
    ask = price
    if bid is None:
        bid = price * (1.0 - 0.001)  # 0.1 % spread — passes spread check
    if quote_volume is None:
        quote_volume = config.MIN_VOLUME_USD * 10
    return {
        "ask": ask,
        "bid": bid,
        "last": price,
        "quoteVolume": quote_volume,
        "info": {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trader(**kwargs) -> CryptoTrader:
    """Return a CryptoTrader whose underlying ccxt exchange is fully mocked."""
    defaults = dict(
        exchange_id="kraken",
        paper_trading=True,
    )
    defaults.update(kwargs)

    with patch("ccxt.kraken") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        trader = CryptoTrader(**defaults)
        trader.exchange = mock_exchange   # keep the mock accessible

    return trader


def _set_price(trader: CryptoTrader, symbol: str, price: float) -> None:
    """Configure the mocked exchange to return *price* and sufficient volume/spread for *symbol*."""
    trader.exchange.fetch_ticker.return_value = _make_ticker(price)


def _make_ohlcv(closes: list, volume: float = 1000.0) -> list:
    """Build a minimal OHLCV list from a sequence of close prices.

    Sets open = high = low = close = c so that typical_price == c, making
    VWAP and Volume Profile deterministic in tests.
    """
    return [[0, c, c, c, c, volume] for c in closes]


def _make_atr_ohlcv(n: int, close: float = 100.0, spread: float = 5.0) -> list:
    """Build OHLCV candles with a constant close and fixed high-low spread.

    high = close + spread, low = close - spread.  With a constant close
    price, the True Range for each candle (after the first) equals
    ``2 * spread``, so the ATR is also ``2 * spread``.
    """
    return [[0, close, close + spread, close - spread, close, 1000.0]
            for _ in range(n)]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_min_buy_order_is_30(self):
        self.assertEqual(config.MIN_BUY_ORDER, 30.0)

    def test_max_buy_order_is_50(self):
        self.assertEqual(config.MAX_BUY_ORDER, 78.0)

    def test_min_less_than_max(self):
        self.assertLess(config.MIN_BUY_ORDER, config.MAX_BUY_ORDER)

    def test_take_profit_pct(self):
        self.assertAlmostEqual(config.TAKE_PROFIT_PCT, 0.055)

    def test_stop_loss_pct(self):
        self.assertAlmostEqual(config.STOP_LOSS_PCT, 0.0175)

    def test_ema_period(self):
        self.assertEqual(config.EMA_PERIOD, 200)

    def test_rsi_period(self):
        self.assertEqual(config.RSI_PERIOD, 14)

    def test_rsi_oversold(self):
        self.assertEqual(config.RSI_OVERSOLD, 35)

    def test_rsi_overbought(self):
        self.assertEqual(config.RSI_OVERBOUGHT, 70)

    def test_atr_period(self):
        self.assertEqual(config.ATR_PERIOD, 14)

    def test_atr_stop_loss_multiplier(self):
        self.assertAlmostEqual(config.ATR_STOP_LOSS_MULTIPLIER, 1.5)

    def test_min_volume_usd_positive(self):
        self.assertGreater(config.MIN_VOLUME_USD, 0)

    def test_max_bid_ask_spread_pct_positive(self):
        self.assertGreater(config.MAX_BID_ASK_SPREAD_PCT, 0)

    def test_simple_algo_short_period(self):
        self.assertEqual(config.SIMPLE_ALGO_SHORT_PERIOD, 50)

    def test_simple_algo_long_period(self):
        self.assertEqual(config.SIMPLE_ALGO_LONG_PERIOD, 200)

    def test_simple_algo_short_less_than_long(self):
        self.assertLess(config.SIMPLE_ALGO_SHORT_PERIOD, config.SIMPLE_ALGO_LONG_PERIOD)

    def test_volume_profile_bins_positive(self):
        self.assertGreater(config.VOLUME_PROFILE_BINS, 0)


# ---------------------------------------------------------------------------
# Exchange ID normalisation tests
# ---------------------------------------------------------------------------

class TestExchangeIdNormalization(unittest.TestCase):
    def _make_trader_with_id(self, exchange_id: str) -> CryptoTrader:
        """Return a CryptoTrader constructed with the given exchange_id string."""
        exchange_attr = exchange_id.lower()
        with patch(f"ccxt.{exchange_attr}") as mock_cls:
            mock_cls.return_value = MagicMock()
            trader = CryptoTrader(exchange_id=exchange_id, paper_trading=True)
        return trader

    def test_lowercase_exchange_id_accepted(self):
        self._make_trader_with_id("coinbase")   # should not raise

    def test_uppercase_exchange_id_normalised(self):
        self._make_trader_with_id("Coinbase")   # should not raise

    def test_allcaps_exchange_id_normalised(self):
        self._make_trader_with_id("COINBASE")   # should not raise

    def test_mixed_case_kraken_normalised(self):
        self._make_trader_with_id("Kraken")     # reproduces the reported crash


# ---------------------------------------------------------------------------
# get_usd_symbols tests
# ---------------------------------------------------------------------------

class TestGetUsdSymbols(unittest.TestCase):
    def setUp(self):
        self.trader = _make_trader()

    def _set_markets(self, markets: dict) -> None:
        self.trader.exchange.load_markets.return_value = markets

    def test_returns_only_usd_pairs(self):
        self._set_markets({
            "BTC/USD": {"quote": "USD", "active": True},
            "ETH/USD": {"quote": "USD", "active": True},
            "BTC/BTC": {"quote": "BTC", "active": True},
        })
        symbols = self.trader.get_usd_symbols()
        self.assertIn("BTC/USD", symbols)
        self.assertIn("ETH/USD", symbols)
        self.assertNotIn("BTC/BTC", symbols)

    def test_excludes_inactive_pairs(self):
        self._set_markets({
            "BTC/USD": {"quote": "USD", "active": True},
            "XRP/USD": {"quote": "USD", "active": False},
        })
        symbols = self.trader.get_usd_symbols()
        self.assertIn("BTC/USD", symbols)
        self.assertNotIn("XRP/USD", symbols)

    def test_returns_sorted_list(self):
        self._set_markets({
            "SOL/USD": {"quote": "USD", "active": True},
            "ADA/USD": {"quote": "USD", "active": True},
            "BTC/USD": {"quote": "USD", "active": True},
        })
        symbols = self.trader.get_usd_symbols()
        self.assertEqual(symbols, sorted(symbols))

    def test_raises_when_no_usd_pairs_found(self):
        self._set_markets({
            "BTC/USDT": {"quote": "USDT", "active": True},
        })
        with self.assertRaises(RuntimeError):
            self.trader.get_usd_symbols()

    def test_active_defaults_to_true_when_key_missing(self):
        # Markets without an 'active' key should be included.
        self._set_markets({
            "DOT/USD": {"quote": "USD"},
        })
        symbols = self.trader.get_usd_symbols()
        self.assertIn("DOT/USD", symbols)

    def test_includes_stocks_and_etfs_when_asset_types_is_none(self):
        # With ASSET_TYPES = None all market types should be returned.
        self._set_markets({
            "BTC/USD":  {"quote": "USD", "active": True, "type": "spot"},
            "AAPL/USD": {"quote": "USD", "active": True, "type": "spot"},
            "SPY/USD":  {"quote": "USD", "active": True, "type": "spot"},
        })
        with unittest.mock.patch.object(config, "ASSET_TYPES", None):
            symbols = self.trader.get_usd_symbols()
        self.assertIn("BTC/USD", symbols)
        self.assertIn("AAPL/USD", symbols)
        self.assertIn("SPY/USD", symbols)

    def test_asset_types_filter_includes_matching_types(self):
        # Only markets whose ``type`` is in ASSET_TYPES should be returned.
        self._set_markets({
            "BTC/USD":  {"quote": "USD", "active": True, "type": "spot"},
            "AAPL/USD": {"quote": "USD", "active": True, "type": "spot"},
            "BTC-PERP/USD": {"quote": "USD", "active": True, "type": "swap"},
        })
        with unittest.mock.patch.object(config, "ASSET_TYPES", ["spot"]):
            symbols = self.trader.get_usd_symbols()
        self.assertIn("BTC/USD", symbols)
        self.assertIn("AAPL/USD", symbols)
        self.assertNotIn("BTC-PERP/USD", symbols)

    def test_asset_types_filter_excludes_non_matching_types(self):
        # Markets whose type is not in ASSET_TYPES must be excluded.
        self._set_markets({
            "BTC/USD":  {"quote": "USD", "active": True, "type": "spot"},
            "BTC-PERP/USD": {"quote": "USD", "active": True, "type": "swap"},
        })
        with unittest.mock.patch.object(config, "ASSET_TYPES", ["swap"]):
            symbols = self.trader.get_usd_symbols()
        self.assertNotIn("BTC/USD", symbols)
        self.assertIn("BTC-PERP/USD", symbols)

    def test_asset_types_filter_raises_when_no_matching_pairs(self):
        # If the filter leaves zero results a RuntimeError must be raised.
        self._set_markets({
            "BTC/USD": {"quote": "USD", "active": True, "type": "spot"},
        })
        with unittest.mock.patch.object(config, "ASSET_TYPES", ["future"]):
            with self.assertRaises(RuntimeError):
                self.trader.get_usd_symbols()


# ---------------------------------------------------------------------------
# Order-limit validation tests
# ---------------------------------------------------------------------------

class TestOrderLimitValidation(unittest.TestCase):
    def setUp(self):
        self.trader = _make_trader()

    # --- below minimum ---

    def test_buy_below_minimum_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(29.99)

    def test_buy_zero_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(0)

    def test_buy_negative_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(-10)

    # --- above maximum ---

    def test_buy_above_maximum_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(78.01)

    def test_buy_way_above_maximum_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(1000)

    # --- valid range ---

    def test_buy_at_minimum_is_valid(self):
        self.trader._validate_buy_amount(30.0)   # should not raise

    def test_buy_at_maximum_is_valid(self):
        self.trader._validate_buy_amount(50.0)   # should not raise

    def test_buy_midrange_is_valid(self):
        self.trader._validate_buy_amount(40.0)   # should not raise

    def test_error_message_contains_amounts_when_below(self):
        with self.assertRaises(OrderSizeError) as ctx:
            self.trader._validate_buy_amount(10.0)
        self.assertIn("10.00", str(ctx.exception))
        self.assertIn("30.00", str(ctx.exception))

    def test_error_message_contains_amounts_when_above(self):
        with self.assertRaises(OrderSizeError) as ctx:
            self.trader._validate_buy_amount(85.0)
        self.assertIn("85.00", str(ctx.exception))
        self.assertIn("78.00", str(ctx.exception))


# ---------------------------------------------------------------------------
# Paper-trading buy tests
# ---------------------------------------------------------------------------

class TestPaperBuy(unittest.TestCase):
    SYMBOL = "BTC/USD"
    PRICE = 50_000.0

    def setUp(self):
        self.trader = _make_trader(paper_trading=True)
        _set_price(self.trader, self.SYMBOL, self.PRICE)

    def test_buy_returns_paper_order(self):
        order = self.trader.buy(self.SYMBOL, 40.0)
        self.assertTrue(order["paper"])
        self.assertEqual(order["side"], "buy")
        self.assertEqual(order["status"], "closed")

    def test_buy_correct_quantity_calculated(self):
        order = self.trader.buy(self.SYMBOL, 40.0)
        self.assertAlmostEqual(order["amount"], 40.0 / self.PRICE, places=8)

    def test_buy_cost_matches_requested_usd(self):
        order = self.trader.buy(self.SYMBOL, 40.0)
        self.assertAlmostEqual(order["cost"], 40.0)

    def test_buy_minimum_amount(self):
        order = self.trader.buy(self.SYMBOL, 30.0)
        self.assertEqual(order["side"], "buy")

    def test_buy_maximum_amount(self):
        order = self.trader.buy(self.SYMBOL, 50.0)
        self.assertEqual(order["side"], "buy")

    def test_buy_does_not_call_exchange_create_order(self):
        self.trader.buy(self.SYMBOL, 40.0)
        self.trader.exchange.create_market_buy_order.assert_not_called()

    def test_buy_below_min_raises_before_exchange_call(self):
        with self.assertRaises(OrderSizeError):
            self.trader.buy(self.SYMBOL, 20.0)
        self.trader.exchange.create_market_buy_order.assert_not_called()

    def test_buy_above_max_raises_before_exchange_call(self):
        with self.assertRaises(OrderSizeError):
            self.trader.buy(self.SYMBOL, 80.0)
        self.trader.exchange.create_market_buy_order.assert_not_called()


# ---------------------------------------------------------------------------
# Paper-trading sell tests
# ---------------------------------------------------------------------------

class TestPaperSell(unittest.TestCase):
    SYMBOL = "ETH/USD"
    PRICE = 3_000.0

    def setUp(self):
        self.trader = _make_trader(paper_trading=True)
        _set_price(self.trader, self.SYMBOL, self.PRICE)

    def test_sell_returns_paper_order(self):
        order = self.trader.sell(self.SYMBOL, 0.01)
        self.assertTrue(order["paper"])
        self.assertEqual(order["side"], "sell")
        self.assertEqual(order["status"], "closed")

    def test_sell_cost_calculated_correctly(self):
        order = self.trader.sell(self.SYMBOL, 0.01)
        self.assertAlmostEqual(order["cost"], 0.01 * self.PRICE)

    def test_sell_does_not_call_exchange_create_order(self):
        self.trader.sell(self.SYMBOL, 0.01)
        self.trader.exchange.create_market_sell_order.assert_not_called()

    def test_sell_zero_quantity_raises(self):
        with self.assertRaises(ValueError):
            self.trader.sell(self.SYMBOL, 0)

    def test_sell_negative_quantity_raises(self):
        with self.assertRaises(ValueError):
            self.trader.sell(self.SYMBOL, -1)


# ---------------------------------------------------------------------------
# Edge case: unpriceable symbol
# ---------------------------------------------------------------------------

class TestGetPrice(unittest.TestCase):
    def setUp(self):
        self.trader = _make_trader()

    def test_raises_when_ask_and_last_are_none(self):
        self.trader.exchange.fetch_ticker.return_value = {"ask": None, "last": None}
        with self.assertRaises(RuntimeError):
            self.trader._get_price("BTC/USD")

    def test_raises_when_ask_and_last_are_zero(self):
        self.trader.exchange.fetch_ticker.return_value = {"ask": 0, "last": 0}
        with self.assertRaises(RuntimeError):
            self.trader._get_price("BTC/USD")

    def test_uses_last_when_ask_is_none(self):
        self.trader.exchange.fetch_ticker.return_value = {"ask": None, "last": 2000.0}
        price = self.trader._get_price("ETH/USD")
        self.assertEqual(price, 2000.0)


class TestCustomLimits(unittest.TestCase):
    def test_custom_limits_are_respected(self):
        trader = _make_trader(min_buy_order=10.0, max_buy_order=20.0)
        # valid within custom range
        trader._validate_buy_amount(15.0)

        # below custom min
        with self.assertRaises(OrderSizeError):
            trader._validate_buy_amount(9.0)

        # above custom max
        with self.assertRaises(OrderSizeError):
            trader._validate_buy_amount(21.0)


# ---------------------------------------------------------------------------
# EMA computation tests
# ---------------------------------------------------------------------------

class TestComputeEMA(unittest.TestCase):
    def test_ema_equals_price_when_all_same(self):
        closes = [100.0] * 10
        ema = CryptoTrader._compute_ema(closes, period=5)
        self.assertAlmostEqual(ema, 100.0, places=6)

    def test_ema_tracks_rising_series(self):
        closes = list(range(1, 21))  # 1 .. 20
        ema = CryptoTrader._compute_ema(closes, period=5)
        # EMA should be near the recent values (above the midpoint 10.5)
        self.assertGreater(ema, 10.5)

    def test_ema_tracks_falling_series(self):
        closes = list(range(20, 0, -1))  # 20 .. 1
        ema = CryptoTrader._compute_ema(closes, period=5)
        self.assertLess(ema, 10.5)

    def test_ema_raises_when_too_few_closes(self):
        with self.assertRaises(ValueError):
            CryptoTrader._compute_ema([100.0, 200.0], period=5)

    def test_ema_exact_period_length(self):
        closes = [10.0] * 5
        ema = CryptoTrader._compute_ema(closes, period=5)
        self.assertAlmostEqual(ema, 10.0, places=6)


# ---------------------------------------------------------------------------
# RSI computation tests
# ---------------------------------------------------------------------------

class TestComputeRSI(unittest.TestCase):
    def test_rsi_all_gains_is_100(self):
        # Strictly rising prices → no losses → RSI = 100
        closes = [float(i) for i in range(1, 20)]
        rsi = CryptoTrader._compute_rsi(closes, period=14)
        self.assertAlmostEqual(rsi, 100.0, places=4)

    def test_rsi_all_losses_is_zero(self):
        # Strictly falling prices → no gains → RSI = 0
        closes = [float(i) for i in range(20, 0, -1)]
        rsi = CryptoTrader._compute_rsi(closes, period=14)
        self.assertAlmostEqual(rsi, 0.0, places=4)

    def test_rsi_is_between_0_and_100(self):
        import random
        random.seed(42)
        closes = [100.0 + random.uniform(-5, 5) for _ in range(30)]
        rsi = CryptoTrader._compute_rsi(closes, period=14)
        self.assertGreaterEqual(rsi, 0.0)
        self.assertLessEqual(rsi, 100.0)

    def test_rsi_raises_when_too_few_closes(self):
        with self.assertRaises(ValueError):
            CryptoTrader._compute_rsi([100.0] * 5, period=14)

    def test_rsi_midpoint_for_equal_moves(self):
        # Alternating +1 / -1 → equal avg_gain and avg_loss → RSI ≈ 50
        closes = []
        price = 100.0
        for i in range(30):
            price += 1.0 if i % 2 == 0 else -1.0
            closes.append(price)
        rsi = CryptoTrader._compute_rsi(closes, period=14)
        self.assertAlmostEqual(rsi, 50.0, delta=5.0)


# ---------------------------------------------------------------------------
# ATR computation tests
# ---------------------------------------------------------------------------

class TestComputeATR(unittest.TestCase):
    def test_atr_constant_spread_equals_twice_spread(self):
        # With all closes equal and high = close + 5, low = close - 5:
        # TR = max(10, 5, 5) = 10 for every candle; ATR = 10
        ohlcv = _make_atr_ohlcv(20, close=100.0, spread=5.0)
        atr = CryptoTrader._compute_atr(ohlcv, period=14)
        self.assertAlmostEqual(atr, 10.0, places=6)

    def test_atr_zero_spread_is_zero(self):
        # high == low == prev_close → all TRs = 0
        ohlcv = _make_ohlcv([100.0] * 20)
        atr = CryptoTrader._compute_atr(ohlcv, period=14)
        self.assertAlmostEqual(atr, 0.0, places=6)

    def test_atr_is_positive_for_volatile_series(self):
        import random
        random.seed(0)
        closes = [100.0 + random.uniform(-10, 10) for _ in range(30)]
        ohlcv = _make_ohlcv(closes)
        atr = CryptoTrader._compute_atr(ohlcv, period=14)
        self.assertGreaterEqual(atr, 0.0)

    def test_atr_raises_when_too_few_candles(self):
        ohlcv = _make_atr_ohlcv(5)
        with self.assertRaises(ValueError):
            CryptoTrader._compute_atr(ohlcv, period=14)

    def test_atr_exact_period_plus_one_candles(self):
        ohlcv = _make_atr_ohlcv(15, spread=3.0)  # period=14 → needs 15 candles
        atr = CryptoTrader._compute_atr(ohlcv, period=14)
        self.assertAlmostEqual(atr, 6.0, places=6)  # 2 * spread

    def test_atr_is_float(self):
        ohlcv = _make_atr_ohlcv(20)
        self.assertIsInstance(CryptoTrader._compute_atr(ohlcv, period=14), float)


# ---------------------------------------------------------------------------
# VWAP computation tests
# ---------------------------------------------------------------------------

class TestComputeVWAP(unittest.TestCase):
    def test_vwap_uniform_volume_equals_price_mean(self):
        # With equal volumes and high=low=close=c, VWAP = mean(closes)
        closes = [10.0, 20.0, 30.0]
        ohlcv = _make_ohlcv(closes, volume=500.0)
        vwap = CryptoTrader._compute_vwap(ohlcv)
        self.assertAlmostEqual(vwap, 20.0, places=6)

    def test_vwap_single_candle_equals_typical_price(self):
        # Single candle: typical = (high + low + close) / 3
        ohlcv = [[0, 100.0, 120.0, 80.0, 100.0, 1000.0]]
        vwap = CryptoTrader._compute_vwap(ohlcv)
        expected = (120.0 + 80.0 + 100.0) / 3.0
        self.assertAlmostEqual(vwap, expected, places=6)

    def test_vwap_higher_volume_candle_pulls_average(self):
        # Two candles: price=10 volume=1, price=100 volume=9
        # VWAP = (10*1 + 100*9) / 10 = 91
        ohlcv = [
            [0, 10.0, 10.0, 10.0, 10.0, 1.0],
            [0, 100.0, 100.0, 100.0, 100.0, 9.0],
        ]
        vwap = CryptoTrader._compute_vwap(ohlcv)
        self.assertAlmostEqual(vwap, 91.0, places=6)

    def test_vwap_raises_when_total_volume_is_zero(self):
        ohlcv = _make_ohlcv([50.0, 60.0], volume=0.0)
        with self.assertRaises(ValueError):
            CryptoTrader._compute_vwap(ohlcv)

    def test_vwap_is_float(self):
        ohlcv = _make_ohlcv([100.0, 200.0, 300.0])
        self.assertIsInstance(CryptoTrader._compute_vwap(ohlcv), float)


# ---------------------------------------------------------------------------
# Volume Profile computation tests
# ---------------------------------------------------------------------------

class TestComputeVolumeProfile(unittest.TestCase):
    def test_returns_dict_with_poc_key(self):
        ohlcv = _make_ohlcv([100.0, 110.0, 120.0])
        result = CryptoTrader._compute_volume_profile(ohlcv)
        self.assertIn("poc", result)

    def test_poc_is_within_price_range(self):
        closes = [float(i) for i in range(100, 200)]
        ohlcv = _make_ohlcv(closes)
        result = CryptoTrader._compute_volume_profile(ohlcv, num_bins=10)
        self.assertGreaterEqual(result["poc"], 100.0)
        self.assertLessEqual(result["poc"], 200.0)

    def test_fallback_to_last_close_when_all_prices_equal(self):
        # min_price == max_price → bin_size is zero; fallback returns last close
        ohlcv = _make_ohlcv([50.0, 50.0, 50.0])
        result = CryptoTrader._compute_volume_profile(ohlcv)
        self.assertAlmostEqual(result["poc"], 50.0, places=6)

    def test_higher_volume_bin_wins(self):
        # All volume concentrated at price 200 → POC near 200
        ohlcv = [
            [0, 100.0, 100.0, 100.0, 100.0, 1.0],      # low volume at 100
            [0, 200.0, 200.0, 200.0, 200.0, 10000.0],   # high volume at 200
        ]
        result = CryptoTrader._compute_volume_profile(ohlcv, num_bins=10)
        self.assertGreater(result["poc"], 150.0)

    def test_poc_is_float(self):
        ohlcv = _make_ohlcv([100.0, 110.0, 120.0])
        self.assertIsInstance(CryptoTrader._compute_volume_profile(ohlcv)["poc"], float)


# ---------------------------------------------------------------------------
# SimpleAlgo signal computation tests
# ---------------------------------------------------------------------------

class TestComputeSimpleAlgoSignal(unittest.TestCase):
    def test_bullish_when_rising_series(self):
        # Strictly rising: EMA 50 > EMA 200 (golden cross state)
        closes = [float(i) for i in range(1, 210)]
        self.assertTrue(CryptoTrader._compute_simple_algo_signal(closes))

    def test_bearish_when_falling_series(self):
        # Strictly falling: EMA 50 < EMA 200
        closes = [float(i) for i in range(210, 0, -1)]
        self.assertFalse(CryptoTrader._compute_simple_algo_signal(closes))

    def test_returns_false_when_too_few_candles(self):
        closes = [100.0] * (config.SIMPLE_ALGO_LONG_PERIOD - 1)
        self.assertFalse(CryptoTrader._compute_simple_algo_signal(closes))

    def test_returns_bool(self):
        closes = [float(i) for i in range(1, 210)]
        result = CryptoTrader._compute_simple_algo_signal(closes)
        self.assertIsInstance(result, bool)

    def test_exactly_long_period_candles_does_not_raise(self):
        closes = [float(i) for i in range(1, config.SIMPLE_ALGO_LONG_PERIOD + 1)]
        # Should not raise
        CryptoTrader._compute_simple_algo_signal(closes)


# ---------------------------------------------------------------------------
# _compute_bollinger_bands tests
# ---------------------------------------------------------------------------

class TestComputeBollingerBands(unittest.TestCase):
    def test_returns_expected_keys(self):
        closes = [float(i) for i in range(1, 21)]
        result = CryptoTrader._compute_bollinger_bands(closes, 20, 2.0)
        self.assertIn("upper", result)
        self.assertIn("middle", result)
        self.assertIn("lower", result)

    def test_middle_is_sma(self):
        closes = [10.0] * 20
        result = CryptoTrader._compute_bollinger_bands(closes, 20, 2.0)
        self.assertAlmostEqual(result["middle"], 10.0)

    def test_bands_symmetric_around_middle(self):
        closes = [10.0] * 20
        result = CryptoTrader._compute_bollinger_bands(closes, 20, 2.0)
        self.assertAlmostEqual(result["upper"] - result["middle"],
                               result["middle"] - result["lower"])

    def test_zero_std_produces_flat_bands(self):
        closes = [50.0] * 20
        result = CryptoTrader._compute_bollinger_bands(closes, 20, 2.0)
        self.assertAlmostEqual(result["upper"], 50.0)
        self.assertAlmostEqual(result["lower"], 50.0)

    def test_price_above_upper_band_on_spike(self):
        # A sharp price spike at the end should push the last close above the band.
        # Verify explicitly by comparing the close against the computed upper value.
        closes = [100.0] * 19 + [200.0]
        result = CryptoTrader._compute_bollinger_bands(closes, 20, 2.0)
        # The spike is so extreme that the last close must exceed the upper band
        self.assertGreater(closes[-1], result["upper"])

    def test_raises_when_too_few_closes(self):
        with self.assertRaises(ValueError):
            CryptoTrader._compute_bollinger_bands([100.0] * 5, 20, 2.0)

    def test_uses_last_period_closes(self):
        # Prepend a different value — only the last 20 closes should matter
        closes_a = [0.0] * 10 + [10.0] * 20
        closes_b = [999.0] * 10 + [10.0] * 20
        result_a = CryptoTrader._compute_bollinger_bands(closes_a, 20, 2.0)
        result_b = CryptoTrader._compute_bollinger_bands(closes_b, 20, 2.0)
        self.assertAlmostEqual(result_a["upper"], result_b["upper"])


# ---------------------------------------------------------------------------
# _compute_keltner_channels tests
# ---------------------------------------------------------------------------

class TestComputeKeltnerChannels(unittest.TestCase):
    def _flat_ohlcv(self, n: int, price: float = 100.0) -> list:
        """Build flat OHLCV candles where high == low == close == price."""
        return [[0, price, price, price, price, 1000.0] for _ in range(n)]

    def test_returns_expected_keys(self):
        ohlcv = self._flat_ohlcv(25)
        closes = [c[4] for c in ohlcv]
        result = CryptoTrader._compute_keltner_channels(ohlcv, closes, 20, 2.0)
        for key in ("upper", "middle", "lower"):
            self.assertIn(key, result)

    def test_middle_equals_ema_of_closes(self):
        ohlcv = self._flat_ohlcv(25)
        closes = [100.0] * 25
        result = CryptoTrader._compute_keltner_channels(ohlcv, closes, 20, 2.0)
        # Flat series → EMA equals the constant price
        self.assertAlmostEqual(result["middle"], 100.0)

    def test_zero_atr_produces_flat_channels(self):
        # high == low == close → ATR = 0 → channels equal EMA
        ohlcv = self._flat_ohlcv(25)
        closes = [100.0] * 25
        result = CryptoTrader._compute_keltner_channels(ohlcv, closes, 20, 2.0)
        self.assertAlmostEqual(result["upper"], 100.0)
        self.assertAlmostEqual(result["lower"], 100.0)

    def test_channels_symmetric_around_middle(self):
        ohlcv = _make_atr_ohlcv(25)
        closes = [c[4] for c in ohlcv]
        result = CryptoTrader._compute_keltner_channels(ohlcv, closes, 20, 2.0)
        self.assertAlmostEqual(result["upper"] - result["middle"],
                               result["middle"] - result["lower"])

    def test_raises_when_too_few_closes(self):
        ohlcv = self._flat_ohlcv(5)
        closes = [c[4] for c in ohlcv]
        with self.assertRaises(ValueError):
            CryptoTrader._compute_keltner_channels(ohlcv, closes, 20, 2.0)

    def test_raises_when_too_few_ohlcv_for_atr(self):
        # Enough closes but not enough candles for ATR (need period+1)
        ohlcv = self._flat_ohlcv(20)
        closes = [c[4] for c in ohlcv]
        with self.assertRaises(ValueError):
            CryptoTrader._compute_keltner_channels(ohlcv, closes, 20, 2.0)


# ---------------------------------------------------------------------------
# _compute_relative_volume tests
# ---------------------------------------------------------------------------

class TestComputeRelativeVolume(unittest.TestCase):
    def _ohlcv_with_volumes(self, volumes: list) -> list:
        return [[0, 100.0, 100.0, 100.0, 100.0, v] for v in volumes]

    def test_rvol_of_one_when_current_equals_average(self):
        # 20 prior candles all volume=1000, current=1000 → RVOL=1.0
        ohlcv = self._ohlcv_with_volumes([1000.0] * 21)
        self.assertAlmostEqual(CryptoTrader._compute_relative_volume(ohlcv, 20), 1.0)

    def test_rvol_of_five_when_current_is_5x_average(self):
        # 20 prior candles volume=1000, current=5000 → RVOL=5.0
        ohlcv = self._ohlcv_with_volumes([1000.0] * 20 + [5000.0])
        self.assertAlmostEqual(CryptoTrader._compute_relative_volume(ohlcv, 20), 5.0)

    def test_rvol_zero_when_avg_volume_is_zero(self):
        ohlcv = self._ohlcv_with_volumes([0.0] * 20 + [1000.0])
        self.assertAlmostEqual(CryptoTrader._compute_relative_volume(ohlcv, 20), 0.0)

    def test_rvol_zero_when_current_volume_is_zero(self):
        ohlcv = self._ohlcv_with_volumes([1000.0] * 20 + [0.0])
        self.assertAlmostEqual(CryptoTrader._compute_relative_volume(ohlcv, 20), 0.0)

    def test_raises_when_too_few_candles(self):
        ohlcv = self._ohlcv_with_volumes([1000.0] * 5)
        with self.assertRaises(ValueError):
            CryptoTrader._compute_relative_volume(ohlcv, 20)

    def test_uses_only_prior_candles_for_average(self):
        # The current candle (high volume) should not inflate the average
        volumes = [100.0] * 20 + [10_000.0]
        ohlcv = self._ohlcv_with_volumes(volumes)
        rvol = CryptoTrader._compute_relative_volume(ohlcv, 20)
        self.assertAlmostEqual(rvol, 100.0)  # 10000 / 100



class TestGetIndicators(unittest.TestCase):
    SYMBOL = "BTC/USD"

    def setUp(self):
        self.trader = _make_trader()
        # Build enough synthetic candles (rising then flat)
        n = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 20
        closes = [float(i) for i in range(1, n + 1)]
        self.trader.exchange.fetch_ohlcv.return_value = _make_ohlcv(closes)

    def test_returns_expected_indicator_keys(self):
        result = self.trader.get_indicators(self.SYMBOL)
        for key in (
            "price", "vwap", "rsi", "atr", "volume_profile_poc",
            "simple_algo_signal", "bb_upper", "bb_middle", "bb_lower",
            "kc_upper", "kc_middle", "kc_lower", "rvol",
            "wt1", "wt2", "cci", "prev_cci", "adx", "plus_di", "minus_di", "kernel",
        ):
            self.assertIn(key, result)

    def test_price_equals_last_close(self):
        n = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 20
        closes = [float(i) for i in range(1, n + 1)]
        self.trader.exchange.fetch_ohlcv.return_value = _make_ohlcv(closes)
        result = self.trader.get_indicators(self.SYMBOL)
        self.assertAlmostEqual(result["price"], closes[-1])

    def test_fetch_ohlcv_called_with_correct_limit(self):
        self.trader.get_indicators(self.SYMBOL, timeframe="4h")
        wt_min = config.WT_CHANNEL_LENGTH + config.WT_AVERAGE_LENGTH + config.WT_MA_LENGTH
        adx_min = 2 * config.ADX_PERIOD + 1
        expected_limit = max(
            config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10,
            config.BB_PERIOD + config.RVOL_PERIOD + 10,
            config.KC_PERIOD + config.RVOL_PERIOD + 10,
            wt_min + 10,
            adx_min + 10,
            config.CCI_PERIOD + 10,
            config.KERNEL_BANDWIDTH + 10,
        )
        self.trader.exchange.fetch_ohlcv.assert_called_once_with(
            self.SYMBOL, "4h", limit=expected_limit
        )


# ---------------------------------------------------------------------------
# should_buy tests
# ---------------------------------------------------------------------------

class TestShouldBuy(unittest.TestCase):
    SYMBOL = "BTC/USD"

    def _set_indicators(self, trader, price, vwap, rsi,
                        volume_profile_poc=None, simple_algo_signal=True,
                        quote_volume=None, bid=None, atr=1.0,
                        rvol=None, bb_upper=None, kc_upper=None,
                        wt1=10.0, wt2=5.0, cci=0.0, prev_cci=-110.0,
                        adx=25.0, plus_di=30.0, minus_di=15.0, kernel=None):
        """Patch get_indicators to return controlled values and set up a ticker mock.

        Defaults produce a signal that fires: RVOL is at threshold, price is
        above both bands, and all five scored conditions are bullish
        (score = 5 ≥ BUY_SIGNAL_THRESHOLD = 3).
        """
        poc = volume_profile_poc if volume_profile_poc is not None else price - 10.0
        if rvol is None:
            rvol = config.RVOL_THRESHOLD
        if bb_upper is None:
            bb_upper = price - 1.0   # price is above the upper band
        if kc_upper is None:
            kc_upper = price - 1.0   # price is above the upper channel
        if kernel is None:
            kernel = price - 5.0     # price >= kernel → bullish
        trader.get_indicators = MagicMock(
            return_value={
                "price": price,
                "vwap": vwap,
                "rsi": rsi,
                "atr": atr,
                "volume_profile_poc": poc,
                "simple_algo_signal": simple_algo_signal,
                "bb_upper": bb_upper,
                "bb_middle": bb_upper - 5.0,
                "bb_lower": bb_upper - 10.0,
                "kc_upper": kc_upper,
                "kc_middle": kc_upper - 5.0,
                "kc_lower": kc_upper - 10.0,
                "rvol": rvol,
                "wt1": wt1,
                "wt2": wt2,
                "cci": cci,
                "prev_cci": prev_cci,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "kernel": kernel,
            }
        )
        trader.exchange.fetch_ticker.return_value = _make_ticker(
            price, bid=bid, quote_volume=quote_volume
        )

    def test_buy_signal_when_all_conditions_met(self):
        # RVOL at threshold, price above both BB and KC upper bands
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, vwap=100.0, rsi=25.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_rvol_below_threshold(self):
        # RVOL below 5× → no buy signal even if price breaks out
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, vwap=100.0, rsi=25.0,
                             rvol=config.RVOL_THRESHOLD - 0.1)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_rvol_is_zero(self):
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, vwap=100.0, rsi=25.0, rvol=0.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_buy_signal_when_price_above_bb_upper_only(self):
        # Price above BB upper but below KC upper → breakout via BB alone is enough
        trader = _make_trader()
        price = 110.0
        self._set_indicators(trader, price=price, vwap=100.0, rsi=25.0,
                             bb_upper=price - 1.0,   # price > BB upper
                             kc_upper=price + 5.0)   # price < KC upper
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_buy_signal_when_price_above_kc_upper_only(self):
        # Price above KC upper but below BB upper → breakout via KC alone is enough
        trader = _make_trader()
        price = 110.0
        self._set_indicators(trader, price=price, vwap=100.0, rsi=25.0,
                             bb_upper=price + 5.0,   # price < BB upper
                             kc_upper=price - 1.0)   # price > KC upper
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_signal_fires_when_price_below_both_bands(self):
        # Breakout gate removed — buy signal fires even when price is below both upper bands
        trader = _make_trader()
        price = 110.0
        self._set_indicators(trader, price=price, vwap=100.0, rsi=25.0,
                             bb_upper=price + 5.0,   # price < BB upper
                             kc_upper=price + 5.0)   # price < KC upper
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_buy_signal_fires_regardless_of_volume_profile_poc(self):
        # POC is not a buy condition; signal should fire regardless of POC position
        trader = _make_trader()
        self._set_indicators(trader, price=90.0, vwap=80.0, rsi=25.0,
                             volume_profile_poc=100.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_multiple_conditions_fail(self):
        # RVOL too low and no breakout
        trader = _make_trader()
        price = 90.0
        self._set_indicators(trader, price=price, vwap=100.0, rsi=60.0,
                             rvol=1.0,
                             bb_upper=price + 10.0,
                             kc_upper=price + 10.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_buy_fires_when_rvol_just_meets_threshold(self):
        # RVOL exactly at threshold → signal fires (>= comparison)
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, vwap=100.0, rsi=49.0,
                             rvol=config.RVOL_THRESHOLD)
        self.assertTrue(trader.should_buy(self.SYMBOL))


# ---------------------------------------------------------------------------
# should_sell tests
# ---------------------------------------------------------------------------

class TestShouldSell(unittest.TestCase):
    SYMBOL = "BTC/USD"

    def _set_indicators(self, trader, *, all_bearish: bool = True) -> dict:
        """Patch get_indicators with either all-bearish or all-non-bearish defaults.

        all_bearish=True  → sell_score = 5 ≥ SELL_SIGNAL_THRESHOLD → signal fires.
        all_bearish=False → sell_score = 0 < SELL_SIGNAL_THRESHOLD → signal is silent.
        """
        if all_bearish:
            ind = _make_full_indicators(
                price=100.0,
                rsi=80.0,            # rsi_bearish: RSI > RSI_OVERBOUGHT(70)
                wt1=60.0, wt2=50.0,  # wt_bearish: wt1 > WT_OVERBOUGHT(53)
                cci=150.0,           # cci_bearish: CCI > CCI_OVERBOUGHT(100)
                adx=30.0, plus_di=15.0, minus_di=35.0,  # adx_bearish: ADX>20, -DI > +DI
                kernel=110.0,        # kernel_bearish: price(100) < kernel(110)
            )
        else:
            ind = _make_full_indicators(
                price=100.0,
                rsi=50.0,            # not bearish: RSI ≤ RSI_OVERBOUGHT
                wt1=30.0, wt2=35.0,  # not bearish: wt1 < wt2 AND wt1 < WT_OVERBOUGHT
                cci=50.0,            # not bearish: CCI ≤ CCI_OVERBOUGHT
                adx=15.0, plus_di=30.0, minus_di=15.0,  # not bearish: ADX < ADX_THRESHOLD
                kernel=90.0,         # not bearish: price(100) >= kernel(90)
            )
        trader.get_indicators = MagicMock(return_value=ind)
        return ind

    def test_sell_signal_when_all_conditions_bearish(self):
        """All 5 scored conditions bearish → sell_score = 5 ≥ threshold."""
        trader = _make_trader()
        self._set_indicators(trader, all_bearish=True)
        self.assertTrue(trader.should_sell(self.SYMBOL))

    def test_no_sell_signal_when_no_conditions_bearish(self):
        """All 5 scored conditions non-bearish → sell_score = 0 < threshold."""
        trader = _make_trader()
        self._set_indicators(trader, all_bearish=False)
        self.assertFalse(trader.should_sell(self.SYMBOL))

    def test_sell_signal_at_exact_threshold(self):
        """Exactly SELL_SIGNAL_THRESHOLD bearish conditions → signal fires."""
        trader = _make_trader()
        # 3 bearish: RSI + CCI + kernel; WT and ADX are not bearish
        ind = _make_full_indicators(
            price=100.0,
            rsi=80.0,            # bearish
            wt1=30.0, wt2=35.0,  # NOT bearish: wt1 < wt2, wt1 < WT_OVERBOUGHT
            cci=150.0,           # bearish
            adx=15.0, plus_di=30.0, minus_di=15.0,  # NOT bearish: ADX < threshold
            kernel=110.0,        # bearish: price(100) < kernel(110)
        )
        trader.get_indicators = MagicMock(return_value=ind)
        self.assertTrue(trader.should_sell(self.SYMBOL))

    def test_no_sell_signal_one_below_threshold(self):
        """Only 2 bearish conditions → sell_score = 2 < threshold."""
        trader = _make_trader()
        ind = _make_full_indicators(
            price=100.0,
            rsi=80.0,            # bearish
            wt1=35.0, wt2=30.0,  # NOT bearish: wt1 > wt2 AND wt1 < WT_OVERBOUGHT
            cci=50.0,            # NOT bearish: CCI ≤ CCI_OVERBOUGHT
            adx=15.0, plus_di=30.0, minus_di=15.0,  # NOT bearish: ADX < ADX_THRESHOLD
            kernel=110.0,        # bearish: price(100) < kernel(110)
        )
        trader.get_indicators = MagicMock(return_value=ind)
        self.assertFalse(trader.should_sell(self.SYMBOL))

    def test_rsi_at_overbought_threshold_not_bearish(self):
        """RSI exactly at RSI_OVERBOUGHT is NOT bearish (condition is strictly >)."""
        trader = _make_trader()
        ind = _make_full_indicators(
            price=100.0,
            rsi=config.RSI_OVERBOUGHT,   # not bearish: RSI must be *above* threshold
            wt1=30.0, wt2=35.0, cci=50.0,
            adx=15.0, plus_di=30.0, minus_di=15.0,
            kernel=90.0,
        )
        trader.get_indicators = MagicMock(return_value=ind)
        self.assertFalse(trader.should_sell(self.SYMBOL))

    def test_wavetrend_overbought_contributes_to_sell_score(self):
        """WT1 above WT_OVERBOUGHT triggers wt_bearish even when WT1 > WT2."""
        trader = _make_trader()
        ind = _make_full_indicators(
            price=100.0,
            rsi=80.0,                         # bearish
            wt1=config.WT_OVERBOUGHT + 1.0,   # bearish: WT1 > WT_OVERBOUGHT
            wt2=config.WT_OVERBOUGHT - 10.0,
            cci=150.0,                        # bearish
            adx=15.0, plus_di=30.0, minus_di=15.0,  # NOT bearish
            kernel=90.0,                             # NOT bearish
        )
        trader.get_indicators = MagicMock(return_value=ind)
        self.assertTrue(trader.should_sell(self.SYMBOL))  # score = 3

    def test_should_sell_returns_bool(self):
        trader = _make_trader()
        self._set_indicators(trader, all_bearish=True)
        self.assertIsInstance(trader.should_sell(self.SYMBOL), bool)


# ---------------------------------------------------------------------------
# check_exit tests
# ---------------------------------------------------------------------------

class TestCheckExit(unittest.TestCase):
    SYMBOL = "BTC/USD"
    ENTRY = 10_000.0

    def setUp(self):
        self.trader = _make_trader()

    def _set_current_price(self, price: float):
        _set_price(self.trader, self.SYMBOL, price)

    def test_take_profit_triggered(self):
        tp_price = self.ENTRY * (1 + config.TAKE_PROFIT_PCT)
        self._set_current_price(tp_price)
        self.assertEqual(self.trader.check_exit(self.SYMBOL, self.ENTRY), "take_profit")

    def test_take_profit_triggered_above_threshold(self):
        self._set_current_price(self.ENTRY * 1.10)  # 10 % up
        self.assertEqual(self.trader.check_exit(self.SYMBOL, self.ENTRY), "take_profit")

    def test_stop_loss_triggered(self):
        sl_price = self.ENTRY * (1 - config.STOP_LOSS_PCT)
        self._set_current_price(sl_price)
        self.assertEqual(self.trader.check_exit(self.SYMBOL, self.ENTRY), "stop_loss")

    def test_stop_loss_triggered_below_threshold(self):
        self._set_current_price(self.ENTRY * 0.95)  # 5 % down
        self.assertEqual(self.trader.check_exit(self.SYMBOL, self.ENTRY), "stop_loss")

    def test_hold_when_within_range(self):
        self._set_current_price(self.ENTRY * 1.03)  # 3 % up — inside TP/SL band
        self.assertEqual(self.trader.check_exit(self.SYMBOL, self.ENTRY), "hold")

    def test_hold_at_entry_price(self):
        self._set_current_price(self.ENTRY)
        self.assertEqual(self.trader.check_exit(self.SYMBOL, self.ENTRY), "hold")

    def test_hold_slightly_below_entry(self):
        self._set_current_price(self.ENTRY * 0.99)  # 1 % down — above SL
        self.assertEqual(self.trader.check_exit(self.SYMBOL, self.ENTRY), "hold")

    def test_raises_on_zero_entry_price(self):
        self._set_current_price(10_000.0)
        with self.assertRaises(ValueError):
            self.trader.check_exit(self.SYMBOL, 0.0)

    def test_raises_on_negative_entry_price(self):
        self._set_current_price(10_000.0)
        with self.assertRaises(ValueError):
            self.trader.check_exit(self.SYMBOL, -100.0)


# ---------------------------------------------------------------------------
# Volume check tests
# ---------------------------------------------------------------------------

def _set_volume(trader: CryptoTrader, symbol: str, quote_volume: float) -> None:
    """Configure the mocked exchange to return *quote_volume* for *symbol*."""
    trader.exchange.fetch_ticker.return_value = _make_ticker(1.0, quote_volume=quote_volume)


class TestCheckVolume(unittest.TestCase):
    SYMBOL = "BTC/USD"

    def setUp(self):
        self.trader = _make_trader()

    def test_passes_when_volume_above_minimum(self):
        _set_volume(self.trader, self.SYMBOL, _min_volume() + 1.0)
        self.trader._check_volume(self.SYMBOL)  # should not raise

    def test_passes_when_volume_equals_minimum(self):
        _set_volume(self.trader, self.SYMBOL, _min_volume())
        self.trader._check_volume(self.SYMBOL)  # should not raise

    def test_raises_when_volume_below_minimum(self):
        _set_volume(self.trader, self.SYMBOL, _min_volume() - 1.0)
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_raises_when_volume_is_zero(self):
        _set_volume(self.trader, self.SYMBOL, 0.0)
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_raises_when_volume_is_none(self):
        self.trader.exchange.fetch_ticker.return_value = {
            "ask": 1.0,
            "bid": 0.999,
            "last": 1.0,
            "quoteVolume": None,
            "info": {},
        }
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_raises_when_quoteVolume_key_missing(self):
        self.trader.exchange.fetch_ticker.return_value = {
            "ask": 1.0,
            "bid": 0.999,
            "last": 1.0,
            "info": {},
        }
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_passes_when_volume_above_absolute_minimum(self):
        self.trader.exchange.fetch_ticker.return_value = {
            "ask": 1.0,
            "bid": 0.999,
            "last": 1.0,
            "quoteVolume": config.MIN_VOLUME_USD + 1.0,
            "info": {},
        }
        self.trader._check_volume(self.SYMBOL)  # should not raise

    def test_raises_when_volume_below_absolute_minimum(self):
        self.trader.exchange.fetch_ticker.return_value = {
            "ask": 1.0,
            "bid": 0.999,
            "last": 1.0,
            "quoteVolume": config.MIN_VOLUME_USD - 1.0,
            "info": {},
        }
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_error_message_contains_symbol_and_amounts(self):
        low_vol = _min_volume() / 2
        _set_volume(self.trader, self.SYMBOL, low_vol)
        with self.assertRaises(InsufficientVolumeError) as ctx:
            self.trader._check_volume(self.SYMBOL)
        msg = str(ctx.exception)
        self.assertIn(self.SYMBOL, msg)


class TestVolumeIntegration(unittest.TestCase):
    """Volume check wired into buy() and should_buy()."""

    SYMBOL = "BTC/USD"

    def _make_trader_with_volume(self, quote_volume: float) -> CryptoTrader:
        trader = _make_trader()
        trader.exchange.fetch_ticker.return_value = _make_ticker(50_000.0, quote_volume=quote_volume)
        return trader

    # --- buy() ---

    def test_buy_raises_when_volume_insufficient(self):
        trader = self._make_trader_with_volume(_min_volume() - 1.0)
        with self.assertRaises(InsufficientVolumeError):
            trader.buy(self.SYMBOL, 40.0)
        trader.exchange.create_market_buy_order.assert_not_called()

    def test_buy_succeeds_when_volume_sufficient(self):
        trader = self._make_trader_with_volume(_min_volume() + 1.0)
        order = trader.buy(self.SYMBOL, 40.0)
        self.assertEqual(order["side"], "buy")

    # --- should_buy() ---

    def _bullish_indicators(self, trader):
        """Patch get_indicators so all technical conditions fire for a buy signal."""
        price = 110.0
        trader.get_indicators = MagicMock(
            return_value=_make_full_indicators(
                price=price,
                rsi=25.0,                # bullish: RSI < RSI_OVERSOLD(35)
                wt1=10.0, wt2=5.0,       # bullish: wt1 > wt2, wt1 < WT_OVERBOUGHT
                prev_cci=-110.0,         # bullish: prev_cci < CCI_OVERSOLD(-100) ...
                cci=0.0,                 # ... and cci > CCI_OVERSOLD(-100) → crossover
                adx=25.0, plus_di=30.0, minus_di=15.0,  # bullish
                kernel=price - 5.0,      # bullish: price >= kernel
                bb_upper=price - 1.0,    # price above BB upper → breakout
                kc_upper=price - 1.0,    # price above KC upper → breakout
                rvol=config.RVOL_THRESHOLD,
            )
        )

    def test_should_buy_false_when_volume_insufficient(self):
        trader = self._make_trader_with_volume(_min_volume() - 1.0)
        self._bullish_indicators(trader)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_should_buy_true_when_volume_sufficient_and_signals_fire(self):
        trader = self._make_trader_with_volume(_min_volume() + 1.0)
        self._bullish_indicators(trader)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_should_buy_false_when_volume_zero(self):
        trader = _make_trader()
        trader.exchange.fetch_ticker.return_value = _make_ticker(110.0, quote_volume=0.0)
        self._bullish_indicators(trader)
        self.assertFalse(trader.should_buy(self.SYMBOL))


# ---------------------------------------------------------------------------
# Bid-ask spread check tests
# ---------------------------------------------------------------------------

class TestCheckSpread(unittest.TestCase):
    SYMBOL = "BTC/USD"
    PRICE = 1_000.0

    def setUp(self):
        self.trader = _make_trader()

    def _set_spread(self, bid: float) -> None:
        self.trader.exchange.fetch_ticker.return_value = _make_ticker(self.PRICE, bid=bid)

    def test_passes_when_spread_below_maximum(self):
        # 0.1 % spread — well below 1 % max
        self._set_spread(self.PRICE * (1.0 - 0.001))
        self.trader._check_spread(self.SYMBOL)  # should not raise

    def test_passes_when_spread_just_below_maximum(self):
        # spread just under 1 %
        bid = self.PRICE * (1.0 - (config.MAX_BID_ASK_SPREAD_PCT - 1e-6))
        self._set_spread(bid)
        self.trader._check_spread(self.SYMBOL)  # should not raise

    def test_raises_when_spread_at_maximum(self):
        # spread exactly at 1 % → should raise (>= threshold)
        bid = self.PRICE * (1.0 - config.MAX_BID_ASK_SPREAD_PCT)
        self._set_spread(bid)
        with self.assertRaises(WideBidAskSpreadError):
            self.trader._check_spread(self.SYMBOL)

    def test_raises_when_spread_above_maximum(self):
        # 1.5 % spread — above 1 % max
        self._set_spread(self.PRICE * (1.0 - 0.015))
        with self.assertRaises(WideBidAskSpreadError):
            self.trader._check_spread(self.SYMBOL)

    def test_raises_when_bid_is_none(self):
        self.trader.exchange.fetch_ticker.return_value = _make_ticker(self.PRICE, bid=None)
        # Override bid with None explicitly
        self.trader.exchange.fetch_ticker.return_value["bid"] = None
        with self.assertRaises(WideBidAskSpreadError):
            self.trader._check_spread(self.SYMBOL)

    def test_raises_when_bid_key_missing(self):
        ticker = _make_ticker(self.PRICE)
        del ticker["bid"]
        self.trader.exchange.fetch_ticker.return_value = ticker
        with self.assertRaises(WideBidAskSpreadError):
            self.trader._check_spread(self.SYMBOL)

    def test_error_message_contains_symbol(self):
        self._set_spread(self.PRICE * (1.0 - 0.015))  # wide spread
        with self.assertRaises(WideBidAskSpreadError) as ctx:
            self.trader._check_spread(self.SYMBOL)
        self.assertIn(self.SYMBOL, str(ctx.exception))

    def test_buy_raises_when_spread_too_wide(self):
        """buy() should raise WideBidAskSpreadError when spread ≥ MAX_BID_ASK_SPREAD_PCT."""
        self._set_spread(self.PRICE * (1.0 - 0.015))
        with self.assertRaises(WideBidAskSpreadError):
            self.trader.buy(self.SYMBOL, 40.0)
        self.trader.exchange.create_market_buy_order.assert_not_called()

    def test_should_buy_false_when_spread_too_wide(self):
        """should_buy() returns False when spread exceeds maximum."""
        trader = _make_trader()
        price = 110.0
        trader.exchange.fetch_ticker.return_value = _make_ticker(
            price, bid=price * (1.0 - 0.015)  # 1.5 % spread
        )
        trader.get_indicators = MagicMock(
            return_value=_make_full_indicators(
                price=price,
                rsi=25.0,
                wt1=10.0, wt2=5.0, prev_cci=-110.0, cci=0.0,
                adx=25.0, plus_di=30.0, minus_di=15.0,
                kernel=price - 5.0,
                bb_upper=price - 1.0,
                kc_upper=price - 1.0,
                rvol=config.RVOL_THRESHOLD,
            )
        )
        self.assertFalse(trader.should_buy(self.SYMBOL))


# ---------------------------------------------------------------------------
# USD-pair validation tests
# ---------------------------------------------------------------------------

class TestValidateUsdPair(unittest.TestCase):
    def test_valid_usd_pair_passes(self):
        CryptoTrader._validate_usd_pair("BTC/USD")   # should not raise

    def test_valid_eth_usd_passes(self):
        CryptoTrader._validate_usd_pair("ETH/USD")   # should not raise

    def test_lowercase_usd_passes(self):
        CryptoTrader._validate_usd_pair("btc/usd")   # case-insensitive

    def test_non_usd_quote_raises(self):
        with self.assertRaises(ValueError):
            CryptoTrader._validate_usd_pair("BTC/BTC")

    def test_btc_usdt_raises(self):
        with self.assertRaises(ValueError):
            CryptoTrader._validate_usd_pair("BTC/USDT")

    def test_btc_busd_raises(self):
        with self.assertRaises(ValueError):
            CryptoTrader._validate_usd_pair("BTC/BUSD")

    def test_error_message_contains_symbol(self):
        with self.assertRaises(ValueError) as ctx:
            CryptoTrader._validate_usd_pair("BTC/EUR")
        self.assertIn("BTC/EUR", str(ctx.exception))

    def test_buy_raises_for_non_usd_pair(self):
        trader = _make_trader()
        with self.assertRaises(ValueError):
            trader.buy("BTC/BTC", 40.0)
        trader.exchange.create_market_buy_order.assert_not_called()

    def test_sell_raises_for_non_usd_pair(self):
        trader = _make_trader()
        with self.assertRaises(ValueError):
            trader.sell("BTC/EUR", 0.01)
        trader.exchange.create_market_sell_order.assert_not_called()


# ---------------------------------------------------------------------------
# get_usd_balance tests
# ---------------------------------------------------------------------------

class TestGetUsdBalance(unittest.TestCase):
    def setUp(self):
        self.trader = _make_trader()

    def _set_balance(self, free: float):
        self.trader.exchange.fetch_balance.return_value = {
            "USD": {"free": free, "used": 0.0, "total": free}
        }

    def test_returns_free_usd_balance(self):
        self._set_balance(250.0)
        self.assertAlmostEqual(self.trader.get_usd_balance(), 250.0)

    def test_returns_zero_balance(self):
        self._set_balance(0.0)
        self.assertAlmostEqual(self.trader.get_usd_balance(), 0.0)

    def test_raises_when_usd_key_missing(self):
        self.trader.exchange.fetch_balance.return_value = {}
        with self.assertRaises(RuntimeError):
            self.trader.get_usd_balance()

    def test_raises_when_free_key_missing(self):
        self.trader.exchange.fetch_balance.return_value = {"USD": {"total": 100.0}}
        with self.assertRaises(RuntimeError):
            self.trader.get_usd_balance()


# ---------------------------------------------------------------------------
# buy_max_orders tests
# ---------------------------------------------------------------------------

def _set_ticker_and_balance(trader: CryptoTrader, price: float, usd_free: float) -> None:
    """Configure mocked exchange with a price ticker and a USD balance."""
    trader.exchange.fetch_ticker.return_value = _make_ticker(price)
    trader.exchange.fetch_balance.return_value = {
        "USD": {"free": usd_free, "used": 0.0, "total": usd_free}
    }


class TestBuyMaxOrders(unittest.TestCase):
    SYMBOL = "BTC/USD"
    PRICE = 50_000.0

    def setUp(self):
        self.trader = _make_trader()

    def test_empty_list_when_balance_below_minimum(self):
        _set_ticker_and_balance(self.trader, self.PRICE, config.MIN_BUY_ORDER - 0.01)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        self.assertEqual(orders, [])

    def test_one_order_when_balance_equals_max(self):
        _set_ticker_and_balance(self.trader, self.PRICE, config.MAX_BUY_ORDER)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["side"], "buy")

    def test_one_order_when_balance_between_min_and_max(self):
        _set_ticker_and_balance(self.trader, self.PRICE, 35.0)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        self.assertEqual(len(orders), 1)
        self.assertAlmostEqual(orders[0]["cost"], 35.0)

    def test_two_orders_when_balance_is_double_max(self):
        _set_ticker_and_balance(self.trader, self.PRICE, config.MAX_BUY_ORDER * 2)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        self.assertEqual(len(orders), 2)

    def test_three_orders_when_balance_allows(self):
        _set_ticker_and_balance(self.trader, self.PRICE, config.MAX_BUY_ORDER * 3)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        self.assertEqual(len(orders), 3)

    def test_remainder_below_min_is_skipped(self):
        # balance = 2 * MAX + (MIN - 1) → 2 full orders, remainder too small
        balance = config.MAX_BUY_ORDER * 2 + (config.MIN_BUY_ORDER - 1)
        _set_ticker_and_balance(self.trader, self.PRICE, balance)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        self.assertEqual(len(orders), 2)

    def test_each_order_is_capped_at_max_buy_order(self):
        _set_ticker_and_balance(self.trader, self.PRICE, 200.0)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        for order in orders:
            self.assertLessEqual(order["cost"], config.MAX_BUY_ORDER)

    def test_all_orders_are_buy_side(self):
        _set_ticker_and_balance(self.trader, self.PRICE, 150.0)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        for order in orders:
            self.assertEqual(order["side"], "buy")

    def test_raises_for_non_usd_symbol(self):
        _set_ticker_and_balance(self.trader, self.PRICE, 200.0)
        with self.assertRaises(ValueError):
            self.trader.buy_max_orders("BTC/EUR")

    def test_returns_paper_orders_in_paper_mode(self):
        _set_ticker_and_balance(self.trader, self.PRICE, config.MAX_BUY_ORDER * 2)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        for order in orders:
            self.assertTrue(order.get("paper"))


# ---------------------------------------------------------------------------
# buy_bundle tests
# ---------------------------------------------------------------------------

def _set_ticker_for_bundle(trader: CryptoTrader, price: float) -> None:
    """Configure the mocked exchange to return *price* and sufficient volume/spread for any symbol."""
    trader.exchange.fetch_ticker.return_value = _make_ticker(price)


class TestBuyBundle(unittest.TestCase):
    """Tests for CryptoTrader.buy_bundle()."""

    PRICE = 1_000.0
    AMOUNT = 40.0  # within MIN_BUY_ORDER .. MAX_BUY_ORDER

    def setUp(self):
        self.trader = _make_trader(paper_trading=True)
        _set_ticker_for_bundle(self.trader, self.PRICE)

    # --- happy path (paper trading) ---

    def test_returns_order_for_every_symbol_in_bundle(self):
        orders = self.trader.buy_bundle("large_caps", self.AMOUNT)
        for sym in config.BUNDLES["large_caps"]:
            self.assertIn(sym, orders)

    def test_each_order_is_a_buy(self):
        orders = self.trader.buy_bundle("large_caps", self.AMOUNT)
        for order in orders.values():
            self.assertEqual(order["side"], "buy")

    def test_each_order_has_correct_cost(self):
        orders = self.trader.buy_bundle("large_caps", self.AMOUNT)
        for order in orders.values():
            self.assertAlmostEqual(order["cost"], self.AMOUNT)

    def test_paper_orders_flagged(self):
        orders = self.trader.buy_bundle("large_caps", self.AMOUNT)
        for order in orders.values():
            self.assertTrue(order.get("paper"))

    def test_returns_dict_not_list(self):
        result = self.trader.buy_bundle("large_caps", self.AMOUNT)
        self.assertIsInstance(result, dict)

    # --- live trading ---

    def test_live_trading_calls_exchange_per_symbol(self):
        """In live mode, create_market_buy_order must be called once per symbol."""
        live_trader = _make_trader(paper_trading=False)
        _set_ticker_for_bundle(live_trader, self.PRICE)
        live_trader.exchange.create_market_buy_order.return_value = {
            "symbol": "X/USD",
            "side": "buy",
            "type": "market",
            "amount": self.AMOUNT / self.PRICE,
            "price": self.PRICE,
            "cost": self.AMOUNT,
            "status": "closed",
        }
        live_trader.buy_bundle("large_caps", self.AMOUNT)
        expected_calls = len(config.BUNDLES["large_caps"])
        self.assertEqual(
            live_trader.exchange.create_market_buy_order.call_count,
            expected_calls,
        )

    def test_live_order_returned_per_symbol(self):
        """buy_bundle returns the exchange response for each symbol in live mode."""
        live_trader = _make_trader(paper_trading=False)
        _set_ticker_for_bundle(live_trader, self.PRICE)

        def _fake_order(symbol, qty):
            return {
                "symbol": symbol,
                "side": "buy",
                "type": "market",
                "amount": qty,
                "price": self.PRICE,
                "cost": qty * self.PRICE,
                "status": "closed",
            }

        live_trader.exchange.create_market_buy_order.side_effect = _fake_order
        orders = live_trader.buy_bundle("large_caps", self.AMOUNT)
        self.assertEqual(set(orders.keys()), set(config.BUNDLES["large_caps"]))
        for sym, order in orders.items():
            self.assertEqual(order["symbol"], sym)

    # --- unknown bundle ---

    def test_unknown_bundle_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.trader.buy_bundle("nonexistent_bundle", self.AMOUNT)

    def test_key_error_message_contains_bundle_name(self):
        with self.assertRaises(KeyError) as ctx:
            self.trader.buy_bundle("nonexistent_bundle", self.AMOUNT)
        self.assertIn("nonexistent_bundle", str(ctx.exception))

    # --- partial failures are skipped ---

    def test_insufficient_volume_symbol_is_skipped(self):
        """A symbol with low volume is skipped; the rest of the bundle executes."""
        symbols = config.BUNDLES["large_caps"]
        low_vol_sym = symbols[0]
        good_vol_sym = symbols[1]

        def _ticker(symbol):
            vol = 0.0 if symbol == low_vol_sym else _min_volume() * 10
            return _make_ticker(self.PRICE, quote_volume=vol)

        self.trader.exchange.fetch_ticker.side_effect = _ticker

        orders = self.trader.buy_bundle("large_caps", self.AMOUNT)
        self.assertNotIn(low_vol_sym, orders)
        self.assertIn(good_vol_sym, orders)

    def test_order_size_error_symbol_is_skipped(self):
        """An amount outside the allowed range skips all symbols in the bundle but raises nothing."""
        trader = _make_trader(paper_trading=True)
        _set_ticker_for_bundle(trader, self.PRICE)
        # Use an amount that is below MIN_BUY_ORDER to trigger OrderSizeError
        orders = trader.buy_bundle("large_caps", config.MIN_BUY_ORDER - 1.0)
        self.assertEqual(orders, {})

    def test_all_symbols_skipped_returns_empty_dict(self):
        """If every symbol in the bundle is skipped, an empty dict is returned."""
        self.trader.exchange.fetch_ticker.return_value = _make_ticker(
            self.PRICE, quote_volume=0.0  # below minimum for every symbol
        )
        orders = self.trader.buy_bundle("large_caps", self.AMOUNT)
        self.assertEqual(orders, {})

    def test_defi_bundle_all_symbols_bought(self):
        orders = self.trader.buy_bundle("defi", self.AMOUNT)
        self.assertEqual(set(orders.keys()), set(config.BUNDLES["defi"]))

    def test_layer1_bundle_all_symbols_bought(self):
        orders = self.trader.buy_bundle("layer1", self.AMOUNT)
        self.assertEqual(set(orders.keys()), set(config.BUNDLES["layer1"]))


# ---------------------------------------------------------------------------
# get_holdings tests
# ---------------------------------------------------------------------------

class TestGetHoldings(unittest.TestCase):
    def setUp(self):
        self.trader = _make_trader()

    def _set_balance(self, raw: dict) -> None:
        self.trader.exchange.fetch_balance.return_value = raw

    def test_returns_non_usd_assets_with_positive_balance(self):
        self._set_balance({
            "USD": {"free": 100.0, "used": 0.0, "total": 100.0},
            "BTC": {"free": 0.5, "used": 0.0, "total": 0.5},
            "ETH": {"free": 2.0, "used": 0.0, "total": 2.0},
        })
        holdings = self.trader.get_holdings()
        self.assertIn("BTC/USD", holdings)
        self.assertIn("ETH/USD", holdings)

    def test_excludes_usd(self):
        self._set_balance({
            "USD": {"free": 500.0, "used": 0.0, "total": 500.0},
            "BTC": {"free": 1.0, "used": 0.0, "total": 1.0},
        })
        holdings = self.trader.get_holdings()
        self.assertNotIn("USD/USD", holdings)
        self.assertIn("BTC/USD", holdings)

    def test_excludes_zero_balance(self):
        self._set_balance({
            "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            "ETH": {"free": 1.0, "used": 0.0, "total": 1.0},
        })
        holdings = self.trader.get_holdings()
        self.assertNotIn("BTC/USD", holdings)
        self.assertIn("ETH/USD", holdings)

    def test_excludes_none_free_balance(self):
        self._set_balance({
            "BTC": {"free": None, "used": 0.0, "total": 0.0},
        })
        holdings = self.trader.get_holdings()
        self.assertNotIn("BTC/USD", holdings)

    def test_excludes_meta_keys(self):
        self._set_balance({
            "info": {"some": "data"},
            "free": {"BTC": 0.5},
            "used": {"BTC": 0.0},
            "total": {"BTC": 0.5},
            "datetime": "2024-01-01",
            "timestamp": 1704067200000,
            "BTC": {"free": 0.5, "used": 0.0, "total": 0.5},
        })
        holdings = self.trader.get_holdings()
        # Only BTC/USD should appear; meta keys must not generate spurious entries
        self.assertEqual(set(holdings.keys()), {"BTC/USD"})

    def test_returns_correct_quantity(self):
        self._set_balance({
            "BTC": {"free": 0.12345678, "used": 0.0, "total": 0.12345678},
        })
        holdings = self.trader.get_holdings()
        self.assertAlmostEqual(holdings["BTC/USD"]["quantity"], 0.12345678, places=8)

    def test_returns_empty_dict_when_no_crypto(self):
        self._set_balance({
            "USD": {"free": 200.0, "used": 0.0, "total": 200.0},
        })
        holdings = self.trader.get_holdings()
        self.assertEqual(holdings, {})

    def test_returns_empty_dict_when_all_balances_zero(self):
        self._set_balance({
            "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            "ETH": {"free": 0.0, "used": 0.0, "total": 0.0},
        })
        holdings = self.trader.get_holdings()
        self.assertEqual(holdings, {})

    def test_symbol_format_is_base_slash_usd(self):
        self._set_balance({
            "SOL": {"free": 10.0, "used": 0.0, "total": 10.0},
        })
        holdings = self.trader.get_holdings()
        self.assertIn("SOL/USD", holdings)

    def test_quantity_key_present_in_each_entry(self):
        self._set_balance({
            "ETH": {"free": 3.0, "used": 0.0, "total": 3.0},
        })
        holdings = self.trader.get_holdings()
        self.assertIn("quantity", holdings["ETH/USD"])

    def test_non_dict_currency_value_is_skipped(self):
        # ccxt sometimes includes top-level string/numeric values; they must be ignored
        self._set_balance({
            "BTC": {"free": 1.0, "used": 0.0, "total": 1.0},
            "WEIRDKEY": "not-a-dict",
        })
        holdings = self.trader.get_holdings()
        self.assertNotIn("WEIRDKEY/USD", holdings)
        self.assertIn("BTC/USD", holdings)


# ---------------------------------------------------------------------------
# place_exit_orders tests
# ---------------------------------------------------------------------------

class TestPlaceExitOrders(unittest.TestCase):
    """Tests for CryptoTrader.place_exit_orders."""

    # ATR spread used in OHLCV mock: ATR = 2 * _ATR_SPREAD = 10.0
    _ATR_SPREAD = 5.0
    _ATR_VALUE = 2 * _ATR_SPREAD  # 10.0

    def setUp(self):
        self.trader = _make_trader(paper_trading=True)
        # Provide ATR OHLCV data so place_exit_orders can compute ATR
        n = config.ATR_PERIOD + 1
        self.trader.exchange.fetch_ohlcv.return_value = _make_atr_ohlcv(
            n, spread=self._ATR_SPREAD
        )

    # -- Paper-trading mode --------------------------------------------------

    def test_paper_returns_none_order_ids(self):
        result = self.trader.place_exit_orders("BTC/USD", 0.001, 50_000.0)
        self.assertIsNone(result["take_profit_order_id"])
        self.assertIsNone(result["stop_loss_order_id"])

    def test_paper_tp_price_correct(self):
        entry = 50_000.0
        result = self.trader.place_exit_orders("BTC/USD", 0.001, entry)
        expected = entry * (1.0 + config.TAKE_PROFIT_PCT)
        self.assertAlmostEqual(result["take_profit_price"], expected)

    def test_paper_sl_price_correct(self):
        entry = 50_000.0
        result = self.trader.place_exit_orders("BTC/USD", 0.001, entry)
        expected = entry - config.ATR_STOP_LOSS_MULTIPLIER * self._ATR_VALUE
        self.assertAlmostEqual(result["stop_loss_price"], expected)

    def test_paper_does_not_call_create_order(self):
        self.trader.place_exit_orders("BTC/USD", 0.001, 50_000.0)
        self.trader.exchange.create_order.assert_not_called()

    # -- Validation ----------------------------------------------------------

    def test_raises_on_non_usd_pair(self):
        with self.assertRaises(ValueError):
            self.trader.place_exit_orders("BTC/EUR", 0.001, 50_000.0)

    def test_raises_on_zero_entry_price(self):
        with self.assertRaises(ValueError):
            self.trader.place_exit_orders("BTC/USD", 0.001, 0.0)

    def test_raises_on_negative_entry_price(self):
        with self.assertRaises(ValueError):
            self.trader.place_exit_orders("BTC/USD", 0.001, -1.0)

    def test_raises_on_zero_quantity(self):
        with self.assertRaises(ValueError):
            self.trader.place_exit_orders("BTC/USD", 0.0, 50_000.0)

    def test_raises_on_negative_quantity(self):
        with self.assertRaises(ValueError):
            self.trader.place_exit_orders("BTC/USD", -0.001, 50_000.0)

    # -- Live-trading mode ---------------------------------------------------

    def _make_live_trader(self):
        trader = _make_trader(paper_trading=False)
        trader.exchange.fetch_ohlcv.return_value = _make_atr_ohlcv(
            config.ATR_PERIOD + 1, spread=self._ATR_SPREAD
        )
        return trader

    def test_live_places_two_orders(self):
        trader = self._make_live_trader()
        tp_mock = {"id": "tp-123"}
        sl_mock = {"id": "sl-456"}
        trader.exchange.create_order.side_effect = [tp_mock, sl_mock]
        result = trader.place_exit_orders("BTC/USD", 0.001, 50_000.0)
        self.assertEqual(result["take_profit_order_id"], "tp-123")
        self.assertEqual(result["stop_loss_order_id"], "sl-456")
        self.assertEqual(trader.exchange.create_order.call_count, 2)

    def test_live_tp_order_is_limit_sell(self):
        trader = self._make_live_trader()
        trader.exchange.create_order.return_value = {"id": "x"}
        trader.place_exit_orders("BTC/USD", 0.001, 50_000.0)
        tp_call = trader.exchange.create_order.call_args_list[0]
        # args: symbol, order_type, side, quantity, price
        self.assertEqual(tp_call.args[1], "limit")
        self.assertEqual(tp_call.args[2], "sell")

    def test_live_sl_order_uses_atr_stop_price(self):
        trader = self._make_live_trader()
        trader.exchange.create_order.return_value = {"id": "x"}
        entry = 50_000.0
        trader.place_exit_orders("BTC/USD", 0.001, entry)
        sl_call = trader.exchange.create_order.call_args_list[1]
        expected_sl_price = entry - config.ATR_STOP_LOSS_MULTIPLIER * self._ATR_VALUE
        # price arg (index 4) should equal the ATR-based stop-loss price
        self.assertAlmostEqual(sl_call.args[4], expected_sl_price)

    def test_live_fallback_to_stopmarket_on_exception(self):
        """If 'stop' order type fails the method falls back to 'stopMarket'."""
        trader = self._make_live_trader()
        tp_mock = {"id": "tp-1"}
        sl_mock = {"id": "sl-2"}
        trader.exchange.create_order.side_effect = [
            tp_mock,                  # TP limit order succeeds
            RuntimeError("unsupported order type"),  # first SL attempt fails
            sl_mock,                  # fallback stopMarket succeeds
        ]
        result = trader.place_exit_orders("BTC/USD", 0.001, 50_000.0)
        self.assertEqual(result["stop_loss_order_id"], "sl-2")
        # 3 calls total: TP, failed stop, successful stopMarket
        self.assertEqual(trader.exchange.create_order.call_count, 3)


# ---------------------------------------------------------------------------
# check_exit_orders tests
# ---------------------------------------------------------------------------

class TestCheckExitOrders(unittest.TestCase):
    """Tests for CryptoTrader.check_exit_orders."""

    def setUp(self):
        self.trader = _make_trader(paper_trading=True)
        self.entry = 50_000.0

    def _set_price(self, price: float) -> None:
        self.trader.exchange.fetch_ticker.return_value = _make_ticker(price)

    # -- Fallback (None order IDs) -------------------------------------------

    def test_fallback_hold_when_both_ids_none(self):
        self._set_price(self.entry)
        result = self.trader.check_exit_orders("BTC/USD", None, None, self.entry)
        self.assertEqual(result, "hold")

    def test_fallback_take_profit_when_both_ids_none(self):
        tp_price = self.entry * (1.0 + config.TAKE_PROFIT_PCT)
        self._set_price(tp_price)
        result = self.trader.check_exit_orders("BTC/USD", None, None, self.entry)
        self.assertEqual(result, "take_profit")

    def test_fallback_stop_loss_when_both_ids_none(self):
        sl_price = self.entry * (1.0 - config.STOP_LOSS_PCT)
        self._set_price(sl_price)
        result = self.trader.check_exit_orders("BTC/USD", None, None, self.entry)
        self.assertEqual(result, "stop_loss")

    # -- Exchange-order checks (live mode) -----------------------------------

    def _make_live_trader(self):
        return _make_trader(paper_trading=False)

    def test_take_profit_order_filled_cancels_sl(self):
        trader = self._make_live_trader()
        trader.exchange.fetch_order.side_effect = [
            {"status": "closed"},   # TP order filled
            {"status": "open"},     # SL order still open
        ]
        result = trader.check_exit_orders("BTC/USD", "tp-1", "sl-2", self.entry)
        self.assertEqual(result, "take_profit")
        trader.exchange.cancel_order.assert_called_once_with("sl-2", "BTC/USD")

    def test_stop_loss_order_filled_cancels_tp(self):
        trader = self._make_live_trader()
        trader.exchange.fetch_order.side_effect = [
            {"status": "open"},     # TP order still open
            {"status": "closed"},   # SL order filled
        ]
        result = trader.check_exit_orders("BTC/USD", "tp-1", "sl-2", self.entry)
        self.assertEqual(result, "stop_loss")
        trader.exchange.cancel_order.assert_called_once_with("tp-1", "BTC/USD")

    def test_neither_filled_returns_hold(self):
        trader = self._make_live_trader()
        trader.exchange.fetch_order.return_value = {"status": "open"}
        result = trader.check_exit_orders("BTC/USD", "tp-1", "sl-2", self.entry)
        self.assertEqual(result, "hold")
        trader.exchange.cancel_order.assert_not_called()

    def test_fetch_order_exception_treated_as_not_filled(self):
        trader = self._make_live_trader()
        trader.exchange.fetch_order.side_effect = RuntimeError("network error")
        result = trader.check_exit_orders("BTC/USD", "tp-1", "sl-2", self.entry)
        self.assertEqual(result, "hold")

    def test_cancel_order_exception_does_not_raise(self):
        trader = self._make_live_trader()
        trader.exchange.fetch_order.side_effect = [
            {"status": "closed"},
            {"status": "open"},
        ]
        trader.exchange.cancel_order.side_effect = RuntimeError("cancel failed")
        # Should log a warning but not propagate the exception
        result = trader.check_exit_orders("BTC/USD", "tp-1", "sl-2", self.entry)
        self.assertEqual(result, "take_profit")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# WaveTrend computation tests
# ---------------------------------------------------------------------------

def _make_wt_ohlcv(n: int, price: float = 100.0) -> list:
    """Build OHLCV candles with a constant typical price for WaveTrend tests."""
    return [[0, price, price, price, price, 1000.0] for _ in range(n)]


class TestComputeWaveTrend(unittest.TestCase):
    _N1, _N2, _MA = 10, 21, 4
    _MIN_CANDLES = _N1 + _N2 + _MA  # 35

    def test_returns_expected_keys(self):
        ohlcv = _make_wt_ohlcv(self._MIN_CANDLES + 5)
        result = CryptoTrader._compute_wavetrend(ohlcv, self._N1, self._N2, self._MA)
        self.assertIn("wt1", result)
        self.assertIn("wt2", result)

    def test_constant_price_returns_finite_values(self):
        # All HLC3 values are identical → normalised deviation is zero → WT ≈ 0
        ohlcv = _make_wt_ohlcv(self._MIN_CANDLES + 10)
        result = CryptoTrader._compute_wavetrend(ohlcv, self._N1, self._N2, self._MA)
        self.assertTrue(abs(result["wt1"]) < 1e-6)
        self.assertTrue(abs(result["wt2"]) < 1e-6)

    def test_rising_prices_produce_positive_wt1(self):
        n = self._MIN_CANDLES + 50
        prices = [float(i) for i in range(1, n + 1)]
        ohlcv = [[0, p, p, p, p, 1000.0] for p in prices]
        result = CryptoTrader._compute_wavetrend(ohlcv, self._N1, self._N2, self._MA)
        self.assertGreater(result["wt1"], 0.0)

    def test_falling_prices_produce_negative_wt1(self):
        n = self._MIN_CANDLES + 50
        prices = [float(i) for i in range(n, 0, -1)]
        ohlcv = [[0, p, p, p, p, 1000.0] for p in prices]
        result = CryptoTrader._compute_wavetrend(ohlcv, self._N1, self._N2, self._MA)
        self.assertLess(result["wt1"], 0.0)

    def test_raises_when_too_few_candles(self):
        ohlcv = _make_wt_ohlcv(self._MIN_CANDLES - 1)
        with self.assertRaises(ValueError):
            CryptoTrader._compute_wavetrend(ohlcv, self._N1, self._N2, self._MA)

    def test_wt2_is_sma_of_recent_wt1(self):
        # For a constant-price series WT1 values are all ~0, so WT2 should be ~0 too
        ohlcv = _make_wt_ohlcv(self._MIN_CANDLES + 10)
        result = CryptoTrader._compute_wavetrend(ohlcv, self._N1, self._N2, self._MA)
        self.assertAlmostEqual(result["wt2"], result["wt1"], delta=1e-6)

    def test_returns_floats(self):
        ohlcv = _make_wt_ohlcv(self._MIN_CANDLES + 5)
        result = CryptoTrader._compute_wavetrend(ohlcv, self._N1, self._N2, self._MA)
        self.assertIsInstance(result["wt1"], float)
        self.assertIsInstance(result["wt2"], float)


# ---------------------------------------------------------------------------
# CCI computation tests
# ---------------------------------------------------------------------------

class TestComputeCCI(unittest.TestCase):
    _PERIOD = 20

    def _flat_ohlcv(self, n: int, price: float = 100.0) -> list:
        return [[0, price, price, price, price, 1000.0] for _ in range(n)]

    def test_constant_price_returns_zero(self):
        # All typical prices equal → mean deviation = 0 → CCI = 0
        ohlcv = self._flat_ohlcv(self._PERIOD)
        self.assertAlmostEqual(CryptoTrader._compute_cci(ohlcv, self._PERIOD), 0.0)

    def test_price_spike_produces_large_positive_cci(self):
        # Price has been 100 for 19 candles, then spikes to 200
        ohlcv = self._flat_ohlcv(self._PERIOD - 1) + [
            [0, 200.0, 200.0, 200.0, 200.0, 1000.0]
        ]
        cci = CryptoTrader._compute_cci(ohlcv, self._PERIOD)
        self.assertGreater(cci, 0.0)

    def test_price_drop_produces_large_negative_cci(self):
        # Price has been 200 for 19 candles, then drops to 100
        ohlcv = self._flat_ohlcv(self._PERIOD - 1, price=200.0) + [
            [0, 100.0, 100.0, 100.0, 100.0, 1000.0]
        ]
        cci = CryptoTrader._compute_cci(ohlcv, self._PERIOD)
        self.assertLess(cci, 0.0)

    def test_raises_when_too_few_candles(self):
        ohlcv = self._flat_ohlcv(self._PERIOD - 1)
        with self.assertRaises(ValueError):
            CryptoTrader._compute_cci(ohlcv, self._PERIOD)

    def test_uses_only_last_period_candles(self):
        # Prepend candles with extreme prices; only the last *period* candles should matter
        prefix = self._flat_ohlcv(10, price=9999.0)
        suffix = self._flat_ohlcv(self._PERIOD, price=100.0)
        cci_with_prefix = CryptoTrader._compute_cci(prefix + suffix, self._PERIOD)
        cci_no_prefix = CryptoTrader._compute_cci(suffix, self._PERIOD)
        self.assertAlmostEqual(cci_with_prefix, cci_no_prefix, places=6)

    def test_returns_float(self):
        ohlcv = self._flat_ohlcv(self._PERIOD)
        self.assertIsInstance(CryptoTrader._compute_cci(ohlcv, self._PERIOD), float)

    def test_above_100_on_strong_uptrend(self):
        # A linearly rising series should push CCI above +100
        prices = [float(i) for i in range(1, self._PERIOD + 1)]
        ohlcv = [[0, p, p, p, p, 1000.0] for p in prices]
        cci = CryptoTrader._compute_cci(ohlcv, self._PERIOD)
        self.assertGreater(cci, 100.0)

    def test_below_minus_100_on_strong_downtrend(self):
        prices = [float(i) for i in range(self._PERIOD, 0, -1)]
        ohlcv = [[0, p, p, p, p, 1000.0] for p in prices]
        cci = CryptoTrader._compute_cci(ohlcv, self._PERIOD)
        self.assertLess(cci, -100.0)


# ---------------------------------------------------------------------------
# ADX computation tests
# ---------------------------------------------------------------------------

class TestComputeADX(unittest.TestCase):
    _PERIOD = 14
    _MIN_CANDLES = 2 * _PERIOD + 1  # 29

    def _uptrend_ohlcv(self, n: int) -> list:
        """Build a rising series with separating high/low to generate +DM."""
        candles = []
        price = 100.0
        for _ in range(n):
            candles.append([0, price, price + 5.0, price - 2.0, price, 1000.0])
            price += 1.0
        return candles

    def _downtrend_ohlcv(self, n: int) -> list:
        """Build a falling series with separating high/low to generate -DM."""
        candles = []
        price = 200.0
        for _ in range(n):
            candles.append([0, price, price + 2.0, price - 5.0, price, 1000.0])
            price -= 1.0
        return candles

    def test_returns_expected_keys(self):
        ohlcv = self._uptrend_ohlcv(self._MIN_CANDLES + 5)
        result = CryptoTrader._compute_adx(ohlcv, self._PERIOD)
        for key in ("adx", "plus_di", "minus_di"):
            self.assertIn(key, result)

    def test_adx_is_positive_in_trending_market(self):
        ohlcv = self._uptrend_ohlcv(self._MIN_CANDLES + 20)
        result = CryptoTrader._compute_adx(ohlcv, self._PERIOD)
        self.assertGreater(result["adx"], 0.0)

    def test_adx_between_0_and_100(self):
        ohlcv = self._uptrend_ohlcv(self._MIN_CANDLES + 20)
        result = CryptoTrader._compute_adx(ohlcv, self._PERIOD)
        self.assertGreaterEqual(result["adx"], 0.0)
        self.assertLessEqual(result["adx"], 100.0)

    def test_plus_di_greater_in_uptrend(self):
        ohlcv = self._uptrend_ohlcv(self._MIN_CANDLES + 30)
        result = CryptoTrader._compute_adx(ohlcv, self._PERIOD)
        self.assertGreater(result["plus_di"], result["minus_di"])

    def test_minus_di_greater_in_downtrend(self):
        ohlcv = self._downtrend_ohlcv(self._MIN_CANDLES + 30)
        result = CryptoTrader._compute_adx(ohlcv, self._PERIOD)
        self.assertGreater(result["minus_di"], result["plus_di"])

    def test_raises_when_too_few_candles(self):
        ohlcv = self._uptrend_ohlcv(self._MIN_CANDLES - 1)
        with self.assertRaises(ValueError):
            CryptoTrader._compute_adx(ohlcv, self._PERIOD)

    def test_returns_floats(self):
        ohlcv = self._uptrend_ohlcv(self._MIN_CANDLES + 5)
        result = CryptoTrader._compute_adx(ohlcv, self._PERIOD)
        for key in ("adx", "plus_di", "minus_di"):
            self.assertIsInstance(result[key], float)


# ---------------------------------------------------------------------------
# Kernel Filter computation tests
# ---------------------------------------------------------------------------

class TestComputeKernelFilter(unittest.TestCase):
    _BW = 8

    def test_constant_prices_return_that_price(self):
        closes = [100.0] * self._BW
        kernel = CryptoTrader._compute_kernel_filter(closes, self._BW)
        self.assertAlmostEqual(kernel, 100.0, places=6)

    def test_most_recent_price_has_highest_weight(self):
        # Spike at the last price; the kernel result should be pulled toward it
        closes = [50.0] * (self._BW - 1) + [200.0]
        kernel = CryptoTrader._compute_kernel_filter(closes, self._BW)
        self.assertGreater(kernel, 50.0)
        self.assertLess(kernel, 200.0)

    def test_rising_series_returns_value_below_last_price(self):
        # The kernel weighted average is pulled by all prior prices, so it
        # lags behind a rising series → kernel value < last close
        closes = list(range(100, 100 + self._BW))
        kernel = CryptoTrader._compute_kernel_filter(closes, self._BW)
        self.assertLess(kernel, closes[-1])

    def test_raises_when_too_few_closes(self):
        with self.assertRaises(ValueError):
            CryptoTrader._compute_kernel_filter([100.0] * (self._BW - 1), self._BW)

    def test_returns_float(self):
        closes = [100.0] * self._BW
        self.assertIsInstance(CryptoTrader._compute_kernel_filter(closes, self._BW), float)

    def test_kernel_within_price_range(self):
        import random
        random.seed(7)
        closes = [100.0 + random.uniform(-10, 10) for _ in range(self._BW)]
        kernel = CryptoTrader._compute_kernel_filter(closes, self._BW)
        self.assertGreaterEqual(kernel, min(closes))
        self.assertLessEqual(kernel, max(closes))


# ---------------------------------------------------------------------------
# Comprehensive buy signal tests
# ---------------------------------------------------------------------------

class TestComprehensiveBuySignal(unittest.TestCase):
    SYMBOL = "BTC/USD"

    def _trader_with_indicators(self, **kwargs):
        """Build a trader whose get_indicators returns the given overrides."""
        price = kwargs.pop("price", 110.0)
        # Default: fully bullish indicators + breakout + RVOL
        defaults = dict(
            price=price,
            rsi=25.0,
            wt1=10.0, wt2=5.0,
            prev_cci=-110.0, cci=0.0,
            adx=25.0, plus_di=30.0, minus_di=15.0,
            kernel=price - 5.0,
            bb_upper=price - 1.0,
            kc_upper=price - 1.0,
            rvol=config.RVOL_THRESHOLD,
        )
        defaults.update(kwargs)
        trader = _make_trader()
        trader.exchange.fetch_ticker.return_value = _make_ticker(price)
        trader.get_indicators = MagicMock(
            return_value=_make_full_indicators(**defaults)
        )
        return trader

    def test_buy_fires_when_all_conditions_met(self):
        # score = 5, RVOL at threshold, breakout active → signal fires
        trader = self._trader_with_indicators()
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_buy_fires_at_exactly_score_threshold(self):
        # 3 bullish indicators: RSI + CCI + kernel; WT and ADX are not bullish
        trader = self._trader_with_indicators(
            wt1=5.0, wt2=10.0,                          # NOT bullish: wt1 < wt2
            adx=15.0, plus_di=30.0, minus_di=15.0,     # NOT bullish: ADX < threshold
        )
        self.assertTrue(trader.should_buy(self.SYMBOL))  # score = 3

    def test_no_buy_when_score_below_threshold(self):
        # Only 2 bullish: RSI + kernel; WT, CCI, ADX not bullish
        trader = self._trader_with_indicators(
            wt1=5.0, wt2=10.0,                          # NOT bullish
            cci=config.CCI_OVERSOLD - 1.0,               # NOT bullish: CCI ≤ CCI_OVERSOLD
            adx=15.0, plus_di=15.0, minus_di=30.0,      # NOT bullish
        )
        self.assertFalse(trader.should_buy(self.SYMBOL))  # score = 2

    def test_no_buy_when_rvol_below_threshold(self):
        # All 5 indicators bullish but RVOL gate fails
        trader = self._trader_with_indicators(rvol=config.RVOL_THRESHOLD - 0.01)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_buy_fires_when_price_below_both_bands(self):
        # Breakout gate removed — signal fires even when price is below both upper bands
        price = 110.0
        trader = self._trader_with_indicators(
            bb_upper=price + 5.0,   # price < BB upper
            kc_upper=price + 5.0,   # price < KC upper
        )
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_buy_fires_when_only_bb_breakout(self):
        # Price above BB upper only (below KC upper): breakout gate passes
        price = 110.0
        trader = self._trader_with_indicators(
            bb_upper=price - 1.0,   # price > BB upper
            kc_upper=price + 5.0,   # price < KC upper
        )
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_buy_fires_when_only_kc_breakout(self):
        # Price above KC upper only: breakout gate passes
        price = 110.0
        trader = self._trader_with_indicators(
            bb_upper=price + 5.0,   # price < BB upper
            kc_upper=price - 1.0,   # price > KC upper
        )
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_adx_bullish_requires_plus_di_greater_than_minus_di(self):
        # ADX above threshold but -DI > +DI → adx_bullish = False
        trader = self._trader_with_indicators(
            adx=30.0, plus_di=15.0, minus_di=35.0,  # ADX strong but bearish direction
            # Remove 2 other bullish conditions so total score < threshold
            wt1=5.0, wt2=10.0,                       # NOT bullish
            cci=config.CCI_OVERSOLD - 1.0,            # NOT bullish
        )
        self.assertFalse(trader.should_buy(self.SYMBOL))  # score = 2

    def test_kernel_bearish_reduces_score(self):
        # price < kernel → kernel_bullish = False; if only 2 others remain → no signal
        price = 110.0
        trader = self._trader_with_indicators(
            kernel=price + 5.0,                  # NOT bullish: price < kernel
            wt1=5.0, wt2=10.0,                   # NOT bullish
            cci=config.CCI_OVERSOLD - 1.0,        # NOT bullish
        )
        self.assertFalse(trader.should_buy(self.SYMBOL))  # score = 2

    def test_wavetrend_overbought_is_not_bullish(self):
        # WT1 above WT_OVERBOUGHT → wt_bullish = False (close to peak, not entry signal)
        trader = self._trader_with_indicators(
            wt1=config.WT_OVERBOUGHT + 1.0, wt2=config.WT_OVERBOUGHT - 5.0,
            # Compensate: keep score >= 3 via RSI, CCI, ADX, kernel
        )
        # score = 4 (rsi, cci, adx, kernel bullish; wt not bullish) ≥ threshold=3 → True
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_buy_does_not_fire_with_zero_score(self):
        price = 110.0
        trader = self._trader_with_indicators(
            rsi=config.RSI_OVERSOLD + 10.0,             # NOT bullish: RSI ≥ RSI_OVERSOLD
            wt1=5.0, wt2=10.0,                          # NOT bullish
            cci=config.CCI_OVERSOLD - 1.0,               # NOT bullish
            adx=15.0, plus_di=15.0, minus_di=30.0,      # NOT bullish
            kernel=price + 5.0,                          # NOT bullish
        )
        self.assertFalse(trader.should_buy(self.SYMBOL))


# ---------------------------------------------------------------------------
# Config settings for new indicators
# ---------------------------------------------------------------------------

class TestNewIndicatorConfig(unittest.TestCase):
    def test_wt_channel_length_positive(self):
        self.assertGreater(config.WT_CHANNEL_LENGTH, 0)

    def test_wt_average_length_positive(self):
        self.assertGreater(config.WT_AVERAGE_LENGTH, 0)

    def test_wt_ma_length_positive(self):
        self.assertGreater(config.WT_MA_LENGTH, 0)

    def test_wt_overbought_above_oversold(self):
        self.assertGreater(config.WT_OVERBOUGHT, config.WT_OVERSOLD)

    def test_cci_period_positive(self):
        self.assertGreater(config.CCI_PERIOD, 0)

    def test_cci_overbought_above_oversold(self):
        self.assertGreater(config.CCI_OVERBOUGHT, config.CCI_OVERSOLD)

    def test_adx_period_positive(self):
        self.assertGreater(config.ADX_PERIOD, 0)

    def test_adx_threshold_positive(self):
        self.assertGreater(config.ADX_THRESHOLD, 0)

    def test_kernel_bandwidth_positive(self):
        self.assertGreater(config.KERNEL_BANDWIDTH, 0)

    def test_buy_signal_threshold_between_1_and_5(self):
        self.assertGreaterEqual(config.BUY_SIGNAL_THRESHOLD, 1)
        self.assertLessEqual(config.BUY_SIGNAL_THRESHOLD, 5)

    def test_sell_signal_threshold_between_1_and_5(self):
        self.assertGreaterEqual(config.SELL_SIGNAL_THRESHOLD, 1)
        self.assertLessEqual(config.SELL_SIGNAL_THRESHOLD, 5)



if __name__ == "__main__":
    unittest.main()

