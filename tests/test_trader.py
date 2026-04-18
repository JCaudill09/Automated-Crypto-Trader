"""
Unit tests for the Automated Crypto Trader.

All tests run without real network calls — exchange interactions are
patched with unittest.mock so no API keys are required.
"""

import unittest
from unittest.mock import MagicMock, patch

import config
from trader import CryptoTrader, OrderSizeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trader(**kwargs) -> CryptoTrader:
    """Return a CryptoTrader whose underlying ccxt exchange is fully mocked."""
    defaults = dict(
        exchange_id="coinbase",
        paper_trading=True,
    )
    defaults.update(kwargs)

    with patch("ccxt.coinbase") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        trader = CryptoTrader(**defaults)
        trader.exchange = mock_exchange   # keep the mock accessible

    return trader


def _set_price(trader: CryptoTrader, symbol: str, price: float) -> None:
    """Configure the mocked exchange to return *price* for *symbol*."""
    trader.exchange.fetch_ticker.return_value = {"ask": price, "last": price}


def _make_ohlcv(closes: list) -> list:
    """Build a minimal OHLCV list from a sequence of close prices."""
    return [[0, 0, 0, 0, c, 0] for c in closes]


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
    SYMBOL = "BTC/USDT"
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
    SYMBOL = "ETH/USDT"
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
            self.trader._get_price("BTC/USDT")

    def test_raises_when_ask_and_last_are_zero(self):
        self.trader.exchange.fetch_ticker.return_value = {"ask": 0, "last": 0}
        with self.assertRaises(RuntimeError):
            self.trader._get_price("BTC/USDT")

    def test_uses_last_when_ask_is_none(self):
        self.trader.exchange.fetch_ticker.return_value = {"ask": None, "last": 2000.0}
        price = self.trader._get_price("ETH/USDT")
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
    SYMBOL = "BTC/USDT"

    def setUp(self):
        self.trader = _make_trader()
        # Build enough synthetic closes (rising then flat)
        n = config.EMA_PERIOD + config.RSI_PERIOD + 20
        closes = [float(i) for i in range(1, n + 1)]
        self.trader.exchange.fetch_ohlcv.return_value = _make_ohlcv(closes)

    def test_returns_price_ema200_rsi_keys(self):
        result = self.trader.get_indicators(self.SYMBOL)
        self.assertIn("price", result)
        self.assertIn("ema200", result)
        self.assertIn("rsi", result)

    def test_price_equals_last_close(self):
        n = config.EMA_PERIOD + config.RSI_PERIOD + 20
        closes = [float(i) for i in range(1, n + 1)]
        self.trader.exchange.fetch_ohlcv.return_value = _make_ohlcv(closes)
        result = self.trader.get_indicators(self.SYMBOL)
        self.assertAlmostEqual(result["price"], closes[-1])

    def test_fetch_ohlcv_called_with_correct_limit(self):
        self.trader.get_indicators(self.SYMBOL, timeframe="4h")
        expected_limit = config.EMA_PERIOD + config.RSI_PERIOD + 10
        self.trader.exchange.fetch_ohlcv.assert_called_once_with(
            self.SYMBOL, "4h", limit=expected_limit
        )


# ---------------------------------------------------------------------------
# should_buy tests
# ---------------------------------------------------------------------------

class TestShouldBuy(unittest.TestCase):
    SYMBOL = "BTC/USDT"

    def _set_indicators(self, trader, price, ema200, rsi):
        """Patch get_indicators to return controlled values."""
        trader.get_indicators = MagicMock(
            return_value={"price": price, "ema200": ema200, "rsi": rsi}
        )

    def test_buy_signal_when_price_above_ema_and_rsi_oversold(self):
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, ema200=100.0, rsi=25.0)
        self.assertTrue(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_price_below_ema(self):
        trader = _make_trader()
        self._set_indicators(trader, price=90.0, ema200=100.0, rsi=25.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_rsi_not_oversold(self):
        trader = _make_trader()
        self._set_indicators(trader, price=110.0, ema200=100.0, rsi=50.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_rsi_at_exact_oversold_threshold(self):
        # RSI must be *below* the threshold, not equal
        trader = _make_trader()
        self._set_indicators(
            trader, price=110.0, ema200=100.0, rsi=config.RSI_OVERSOLD
        )
        self.assertFalse(trader.should_buy(self.SYMBOL))

    def test_no_signal_when_both_conditions_fail(self):
        trader = _make_trader()
        self._set_indicators(trader, price=90.0, ema200=100.0, rsi=60.0)
        self.assertFalse(trader.should_buy(self.SYMBOL))


# ---------------------------------------------------------------------------
# check_exit tests
# ---------------------------------------------------------------------------

class TestCheckExit(unittest.TestCase):
    SYMBOL = "BTC/USDT"
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


if __name__ == "__main__":
    unittest.main()

