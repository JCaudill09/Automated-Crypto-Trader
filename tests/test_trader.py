"""
Unit tests for the Automated Crypto Trader.

All tests run without real network calls — exchange interactions are
patched with unittest.mock so no API keys are required.
"""

import unittest
from unittest.mock import MagicMock, patch

import ccxt

import config
from trader import CryptoTrader, OrderSizeError


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
    """Configure the mocked exchange to return *price* for *symbol*."""
    trader.exchange.fetch_ticker.return_value = {
        "ask": price,
        "last": price,
    }


def _make_ohlcv(closes: list) -> list:
    """Build a minimal OHLCV list from a sequence of close prices."""
    return [[0, 0, 0, 0, c, 0] for c in closes]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_min_buy_order_is_30(self):
        self.assertEqual(config.MIN_BUY_ORDER, 40.0)

    def test_max_buy_order_is_78(self):
        self.assertEqual(config.MAX_BUY_ORDER, 70.0)

    def test_min_less_than_max(self):
        self.assertLess(config.MIN_BUY_ORDER, config.MAX_BUY_ORDER)

    def test_take_profit_pct(self):
        self.assertAlmostEqual(config.TAKE_PROFIT_PCT, 0.065)

    def test_stop_loss_pct(self):
        self.assertAlmostEqual(config.STOP_LOSS_PCT, 0.0175)

    def test_ema_period(self):
        self.assertEqual(config.EMA_PERIOD, 200)

    def test_rsi_period(self):
        self.assertEqual(config.RSI_PERIOD, 14)

    def test_rsi_oversold(self):
        self.assertEqual(config.RSI_OVERSOLD, 50)

    def test_rsi_overbought(self):
        self.assertEqual(config.RSI_OVERBOUGHT, 70)

    def test_rsi_buy_threshold(self):
        self.assertEqual(config.RSI_BUY_THRESHOLD, 40)


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
            "BTC/EUR": {"quote": "EUR", "active": True},
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
            self.trader._validate_buy_amount(39.99)

    def test_buy_zero_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(0)

    def test_buy_negative_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(-10)

    # --- above maximum ---

    def test_buy_above_maximum_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(70.01)

    def test_buy_way_above_maximum_raises(self):
        with self.assertRaises(OrderSizeError):
            self.trader._validate_buy_amount(1000)

    # --- valid range ---

    def test_buy_at_minimum_is_valid(self):
        self.trader._validate_buy_amount(40.0)   # should not raise

    def test_buy_at_maximum_is_valid(self):
        self.trader._validate_buy_amount(70.0)   # should not raise

    def test_buy_midrange_is_valid(self):
        self.trader._validate_buy_amount(50.0)   # should not raise

    def test_error_message_contains_amounts_when_below(self):
        with self.assertRaises(OrderSizeError) as ctx:
            self.trader._validate_buy_amount(10.0)
        self.assertIn("10.00", str(ctx.exception))
        self.assertIn("40.00", str(ctx.exception))

    def test_error_message_contains_amounts_when_above(self):
        with self.assertRaises(OrderSizeError) as ctx:
            self.trader._validate_buy_amount(100.0)
        self.assertIn("100.00", str(ctx.exception))
        self.assertIn("70.00", str(ctx.exception))


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
        order = self.trader.buy(self.SYMBOL, 40.0)
        self.assertEqual(order["side"], "buy")

    def test_buy_maximum_amount(self):
        order = self.trader.buy(self.SYMBOL, 70.0)
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
# get_indicators tests
# ---------------------------------------------------------------------------

class TestGetIndicators(unittest.TestCase):
    SYMBOL = "BTC/USD"

    def setUp(self):
        self.trader = _make_trader()
        # Build enough synthetic closes for EMA-20, EMA-50, and RSI
        n = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10
        closes = [float(i) for i in range(1, n + 1)]
        self.trader.exchange.fetch_ohlcv.return_value = _make_ohlcv(closes)

    def test_returns_price_ema20_ema50_rsi_keys(self):
        result = self.trader.get_indicators(self.SYMBOL)
        self.assertIn("price", result)
        self.assertIn("ema50", result)
        self.assertIn("ema200", result)
        self.assertIn("prev_ema50", result)
        self.assertIn("prev_ema200", result)
        self.assertIn("rsi", result)

    def test_price_equals_last_close(self):
        n = config.SIMPLE_ALGO_LONG_PERIOD + config.RSI_PERIOD + 10
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

    def _set_indicators(self, trader, price, ema50, ema200, prev_ema50, prev_ema200, rsi=50.0):
        """Patch get_indicators to return controlled indicator values."""
        trader.get_indicators = MagicMock(
            return_value={
                "price": price,
                "ema50": ema50,
                "ema200": ema200,
                "prev_ema50": prev_ema50,
                "prev_ema200": prev_ema200,
                "rsi": rsi,
            }
        )

    def test_buy_signal_on_golden_cross(self):
        # prev: ema50 below ema200; current: ema50 above ema200 → signal
        trader = _make_trader()
        self._set_indicators(trader, price=110.0,
                             ema50=105.0, ema200=100.0,
                             prev_ema50=95.0, prev_ema200=100.0,
                             rsi=50.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_buy_signal_when_prev_ema50_exactly_equals_prev_ema200(self):
        # prev_ema50 == prev_ema200 (touching), current ema50 > ema200 → signal
        trader = _make_trader()
        self._set_indicators(trader, price=110.0,
                             ema50=105.0, ema200=100.0,
                             prev_ema50=100.0, prev_ema200=100.0,
                             rsi=50.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_signal_fires_when_in_established_uptrend(self):
        # ema50 already above ema200 AND price above ema50 → signal fires
        # (trend-confirmation fires on every qualifying candle, not just crossover)
        trader = _make_trader()
        self._set_indicators(trader, price=110.0,
                             ema50=108.0, ema200=100.0,
                             prev_ema50=105.0, prev_ema200=100.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_ema50_below_ema200(self):
        # Downtrend: ema50 below ema200 on both candles
        trader = _make_trader()
        self._set_indicators(trader, price=90.0,
                             ema50=95.0, ema200=100.0,
                             prev_ema50=93.0, prev_ema200=100.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_on_death_cross(self):
        # prev: ema50 above ema200; current: ema50 below ema200 → death cross
        trader = _make_trader()
        self._set_indicators(trader, price=90.0,
                             ema50=95.0, ema200=100.0,
                             prev_ema50=105.0, prev_ema200=100.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_price_below_short_ema(self):
        # ema50 above ema200 (uptrend) but price < ema50 → price lagging, no signal
        trader = _make_trader()
        self._set_indicators(trader, price=100.0,
                             ema50=105.0, ema200=95.0,
                             prev_ema50=103.0, prev_ema200=95.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_rsi_below_threshold(self):
        # All EMAs bullish but RSI too low → no signal
        trader = _make_trader()
        self._set_indicators(trader, price=110.0,
                             ema50=105.0, ema200=100.0,
                             prev_ema50=95.0, prev_ema200=100.0,
                             rsi=39.9)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_signal_fires_with_rsi_just_above_threshold(self):
        # RSI exactly one tick above threshold → signal fires
        trader = _make_trader()
        self._set_indicators(trader, price=110.0,
                             ema50=105.0, ema200=100.0,
                             prev_ema50=95.0, prev_ema200=100.0,
                             rsi=40.1)
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
            trader.buy("BTC/EUR", 40.0)
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
        _set_ticker_and_balance(self.trader, self.PRICE, 50.0)
        orders = self.trader.buy_max_orders(self.SYMBOL)
        self.assertEqual(len(orders), 1)
        self.assertAlmostEqual(orders[0]["cost"], 50.0)

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
# _execute_order retry tests
# ---------------------------------------------------------------------------

class TestExecuteOrderRetry(unittest.TestCase):
    """Tests for the InvalidNonce retry logic in CryptoTrader._execute_order."""

    def setUp(self):
        self.trader = _make_trader(paper_trading=False)

    @patch("trader.time.sleep")
    def test_succeeds_on_first_attempt_without_retry(self, mock_sleep):
        """A successful first call should not sleep or retry."""
        expected = {"id": "order-1"}
        order_fn = MagicMock(return_value=expected)
        result = self.trader._execute_order(order_fn, "BTC/USD", 0.001)
        self.assertEqual(result, expected)
        order_fn.assert_called_once_with("BTC/USD", 0.001)
        mock_sleep.assert_not_called()

    @patch("trader.time.sleep")
    def test_retries_on_invalid_nonce_and_succeeds(self, mock_sleep):
        """Should retry after an InvalidNonce and return the successful result."""
        expected = {"id": "order-2"}
        order_fn = MagicMock(
            side_effect=[ccxt.InvalidNonce("kraken", "EAPI:Invalid nonce"), expected]
        )
        result = self.trader._execute_order(order_fn, "BTC/USD", 0.001)
        self.assertEqual(result, expected)
        self.assertEqual(order_fn.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("trader.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        """Should raise InvalidNonce once all retry attempts are used up."""
        import trader as trader_module
        nonce_exc = ccxt.InvalidNonce("kraken", "EAPI:Invalid nonce")
        order_fn = MagicMock(side_effect=nonce_exc)
        with self.assertRaises(ccxt.InvalidNonce):
            self.trader._execute_order(order_fn, "BTC/USD", 0.001)
        self.assertEqual(order_fn.call_count, trader_module._NONCE_RETRY_ATTEMPTS)
        self.assertEqual(mock_sleep.call_count, trader_module._NONCE_RETRY_ATTEMPTS - 1)

    @patch("trader.time.sleep")
    def test_non_nonce_exception_is_not_retried(self, mock_sleep):
        """Other exchange errors must propagate immediately without retry."""
        order_fn = MagicMock(
            side_effect=ccxt.NetworkError("kraken", "connection reset")
        )
        with self.assertRaises(ccxt.NetworkError):
            self.trader._execute_order(order_fn, "BTC/USD", 0.001)
        order_fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("trader.time.sleep")
    def test_buy_live_retries_on_invalid_nonce(self, mock_sleep):
        """CryptoTrader.buy must retry create_market_buy_order on InvalidNonce."""
        _set_price(self.trader, "BTC/USD", 50_000.0)
        expected = {"id": "buy-order", "price": 50_000.0, "amount": 0.001}
        self.trader.exchange.create_market_buy_order.side_effect = [
            ccxt.InvalidNonce("kraken", "EAPI:Invalid nonce"),
            expected,
        ]
        result = self.trader.buy("BTC/USD", 50.0)
        self.assertEqual(result, expected)
        self.assertEqual(self.trader.exchange.create_market_buy_order.call_count, 2)

    @patch("trader.time.sleep")
    def test_sell_live_retries_on_invalid_nonce(self, mock_sleep):
        """CryptoTrader.sell must retry create_market_sell_order on InvalidNonce."""
        _set_price(self.trader, "ETH/USD", 3_000.0)
        expected = {"id": "sell-order", "price": 3_000.0, "amount": 0.01}
        self.trader.exchange.create_market_sell_order.side_effect = [
            ccxt.InvalidNonce("kraken", "EAPI:Invalid nonce"),
            expected,
        ]
        result = self.trader.sell("ETH/USD", 0.01)
        self.assertEqual(result, expected)
        self.assertEqual(self.trader.exchange.create_market_sell_order.call_count, 2)


if __name__ == "__main__":
    unittest.main()

