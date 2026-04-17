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


if __name__ == "__main__":
    unittest.main()
