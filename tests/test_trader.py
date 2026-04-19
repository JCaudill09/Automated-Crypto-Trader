"""
Unit tests for the Automated Crypto Trader.

All tests run without real network calls — exchange interactions are
patched with unittest.mock so no API keys are required.
"""

import unittest
from unittest.mock import MagicMock, patch

import config
from trader import CryptoTrader, OrderSizeError, InsufficientVolumeError


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
    """Configure the mocked exchange to return *price* and sufficient volume for *symbol*."""
    trader.exchange.fetch_ticker.return_value = {
        "ask": price,
        "last": price,
        "quoteVolume": config.MIN_VOLUME_USD * 10,
    }


def _make_ohlcv(closes: list, volume: float = 1000.0) -> list:
    """Build a minimal OHLCV list from a sequence of close prices.

    Sets open = high = low = close = c so that typical_price == c, making
    VWAP and Volume Profile deterministic in tests.
    """
    return [[0, c, c, c, c, volume] for c in closes]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_min_buy_order_is_30(self):
        self.assertEqual(config.MIN_BUY_ORDER, 30.0)

    def test_max_buy_order_is_50(self):
        self.assertEqual(config.MAX_BUY_ORDER, 50.0)

    def test_min_less_than_max(self):
        self.assertLess(config.MIN_BUY_ORDER, config.MAX_BUY_ORDER)

    def test_take_profit_pct(self):
        self.assertAlmostEqual(config.TAKE_PROFIT_PCT, 0.075)

    def test_stop_loss_pct(self):
        self.assertAlmostEqual(config.STOP_LOSS_PCT, 0.025)

    def test_ema_period(self):
        self.assertEqual(config.EMA_PERIOD, 200)

    def test_rsi_period(self):
        self.assertEqual(config.RSI_PERIOD, 14)

    def test_rsi_oversold(self):
        self.assertEqual(config.RSI_OVERSOLD, 30)

    def test_rsi_overbought(self):
        self.assertEqual(config.RSI_OVERBOUGHT, 70)

    def test_min_volume_usd_positive(self):
        self.assertGreater(config.MIN_VOLUME_USD, 0)

    def test_simple_algo_short_period(self):
        self.assertEqual(config.SIMPLE_ALGO_SHORT_PERIOD, 9)

    def test_simple_algo_long_period(self):
        self.assertEqual(config.SIMPLE_ALGO_LONG_PERIOD, 21)

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
            self.trader._validate_buy_amount(50.01)

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
            self.trader._validate_buy_amount(75.0)
        self.assertIn("75.00", str(ctx.exception))
        self.assertIn("50.00", str(ctx.exception))


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
            self.trader.buy(self.SYMBOL, 60.0)
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
        # Strictly rising: short EMA > long EMA
        closes = [float(i) for i in range(1, 30)]
        self.assertTrue(CryptoTrader._compute_simple_algo_signal(closes))

    def test_bearish_when_falling_series(self):
        # Strictly falling: short EMA < long EMA
        closes = [float(i) for i in range(30, 0, -1)]
        self.assertFalse(CryptoTrader._compute_simple_algo_signal(closes))

    def test_returns_false_when_too_few_candles(self):
        closes = [100.0] * (config.SIMPLE_ALGO_LONG_PERIOD - 1)
        self.assertFalse(CryptoTrader._compute_simple_algo_signal(closes))

    def test_returns_bool(self):
        closes = [float(i) for i in range(1, 30)]
        result = CryptoTrader._compute_simple_algo_signal(closes)
        self.assertIsInstance(result, bool)

    def test_exactly_long_period_candles_does_not_raise(self):
        closes = [float(i) for i in range(1, config.SIMPLE_ALGO_LONG_PERIOD + 1)]
        # Should not raise
        CryptoTrader._compute_simple_algo_signal(closes)


# ---------------------------------------------------------------------------
# get_indicators tests
# ---------------------------------------------------------------------------

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
        self.assertIn("price", result)
        self.assertIn("vwap", result)
        self.assertIn("rsi", result)
        self.assertIn("volume_profile_poc", result)
        self.assertIn("simple_algo_signal", result)

    def test_price_equals_last_close(self):
        n = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 20
        closes = [float(i) for i in range(1, n + 1)]
        self.trader.exchange.fetch_ohlcv.return_value = _make_ohlcv(closes)
        result = self.trader.get_indicators(self.SYMBOL)
        self.assertAlmostEqual(result["price"], closes[-1])

    def test_fetch_ohlcv_called_with_correct_limit(self):
        self.trader.get_indicators(self.SYMBOL, timeframe="4h")
        expected_limit = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10
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
                        quote_volume=None):
        """Patch get_indicators to return controlled values and set up a ticker mock."""
        poc = volume_profile_poc if volume_profile_poc is not None else price - 10.0
        trader.get_indicators = MagicMock(
            return_value={
                "price": price,
                "vwap": vwap,
                "rsi": rsi,
                "volume_profile_poc": poc,
                "simple_algo_signal": simple_algo_signal,
            }
        )
        vol = quote_volume if quote_volume is not None else config.MIN_VOLUME_USD * 10
        trader.exchange.fetch_ticker.return_value = {
            "ask": price,
            "last": price,
            "quoteVolume": vol,
        }

    def test_buy_signal_when_all_conditions_met(self):
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, vwap=100.0, rsi=25.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_price_below_vwap(self):
        trader = _make_trader()
        self._set_indicators(trader, price=90.0, vwap=100.0, rsi=25.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_price_below_poc(self):
        trader = _make_trader()
        self._set_indicators(trader, price=90.0, vwap=80.0, rsi=25.0,
                             volume_profile_poc=100.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_algo_signal_bearish(self):
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, vwap=100.0, rsi=25.0,
                             simple_algo_signal=False)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_rsi_not_oversold(self):
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, vwap=100.0, rsi=50.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_rsi_at_exact_oversold_threshold(self):
        # RSI must be *below* the threshold, not equal
        trader = _make_trader()
        self._set_indicators(
            trader, price=110.0, vwap=100.0, rsi=config.RSI_OVERSOLD
        )
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_multiple_conditions_fail(self):
        trader = _make_trader()
        self._set_indicators(trader, price=90.0, vwap=100.0, rsi=60.0,
                             simple_algo_signal=False)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_buy_fires_when_price_equals_poc(self):
        # price >= poc (not strictly greater), so price == poc should still fire
        trader = _make_trader()
        self._set_indicators(trader, price=100.0, vwap=90.0, rsi=25.0,
                             volume_profile_poc=100.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))


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
    trader.exchange.fetch_ticker.return_value = {
        "ask": 1.0,
        "last": 1.0,
        "quoteVolume": quote_volume,
    }


class TestCheckVolume(unittest.TestCase):
    SYMBOL = "BTC/USD"

    def setUp(self):
        self.trader = _make_trader()

    def test_passes_when_volume_above_minimum(self):
        _set_volume(self.trader, self.SYMBOL, config.MIN_VOLUME_USD + 1.0)
        self.trader._check_volume(self.SYMBOL)  # should not raise

    def test_passes_when_volume_equals_minimum(self):
        _set_volume(self.trader, self.SYMBOL, config.MIN_VOLUME_USD)
        self.trader._check_volume(self.SYMBOL)  # should not raise

    def test_raises_when_volume_below_minimum(self):
        _set_volume(self.trader, self.SYMBOL, config.MIN_VOLUME_USD - 1.0)
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_raises_when_volume_is_zero(self):
        _set_volume(self.trader, self.SYMBOL, 0.0)
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_raises_when_volume_is_none(self):
        self.trader.exchange.fetch_ticker.return_value = {
            "ask": 1.0,
            "last": 1.0,
            "quoteVolume": None,
        }
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_raises_when_quoteVolume_key_missing(self):
        self.trader.exchange.fetch_ticker.return_value = {"ask": 1.0, "last": 1.0}
        with self.assertRaises(InsufficientVolumeError):
            self.trader._check_volume(self.SYMBOL)

    def test_error_message_contains_symbol_and_amounts(self):
        low_vol = config.MIN_VOLUME_USD / 2
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
        trader.exchange.fetch_ticker.return_value = {
            "ask": 50_000.0,
            "last": 50_000.0,
            "quoteVolume": quote_volume,
        }
        return trader

    # --- buy() ---

    def test_buy_raises_when_volume_insufficient(self):
        trader = self._make_trader_with_volume(config.MIN_VOLUME_USD - 1.0)
        with self.assertRaises(InsufficientVolumeError):
            trader.buy(self.SYMBOL, 40.0)
        trader.exchange.create_market_buy_order.assert_not_called()

    def test_buy_succeeds_when_volume_sufficient(self):
        trader = self._make_trader_with_volume(config.MIN_VOLUME_USD + 1.0)
        order = trader.buy(self.SYMBOL, 40.0)
        self.assertEqual(order["side"], "buy")

    # --- should_buy() ---

    def _bullish_indicators(self, trader):
        """Patch get_indicators so the technical conditions fire."""
        trader.get_indicators = MagicMock(
            return_value={
                "price": 110.0,
                "vwap": 100.0,
                "volume_profile_poc": 100.0,
                "simple_algo_signal": True,
                "rsi": 25.0,
            }
        )

    def test_should_buy_false_when_volume_insufficient(self):
        trader = self._make_trader_with_volume(config.MIN_VOLUME_USD - 1.0)
        self._bullish_indicators(trader)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_should_buy_true_when_volume_sufficient_and_signals_fire(self):
        trader = self._make_trader_with_volume(config.MIN_VOLUME_USD + 1.0)
        self._bullish_indicators(trader)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_should_buy_false_when_volume_zero(self):
        trader = _make_trader()
        trader.exchange.fetch_ticker.return_value = {
            "ask": 110.0,
            "last": 110.0,
            "quoteVolume": 0.0,
        }
        self._bullish_indicators(trader)
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
    trader.exchange.fetch_ticker.return_value = {
        "ask": price,
        "last": price,
        "quoteVolume": config.MIN_VOLUME_USD * 10,
    }
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
    """Configure the mocked exchange to return *price* and sufficient volume for any symbol."""
    trader.exchange.fetch_ticker.return_value = {
        "ask": price,
        "last": price,
        "quoteVolume": config.MIN_VOLUME_USD * 10,
    }


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
            vol = 0.0 if symbol == low_vol_sym else config.MIN_VOLUME_USD * 10
            return {"ask": self.PRICE, "last": self.PRICE, "quoteVolume": vol}

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
        self.trader.exchange.fetch_ticker.return_value = {
            "ask": self.PRICE,
            "last": self.PRICE,
            "quoteVolume": 0.0,  # below minimum for every symbol
        }
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


if __name__ == "__main__":
    unittest.main()

