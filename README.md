# Automated-Crypto-Trader

An automated cryptocurrency trading bot built on top of [ccxt](https://github.com/ccxt/ccxt), supporting any exchange that ccxt provides.

## Order limits

| Constraint | Value |
|---|---|
| Minimum buy order | **$30 USD** |
| Maximum buy order | **$50 USD** |

Buy orders outside this range are rejected before any exchange call is made.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt
```

### Paper trading (safe simulation — no real money)

```python
from trader import CryptoTrader

# Paper trading is on by default (config.PAPER_TRADING = True)
bot = CryptoTrader(exchange_id="kraken")

# Buy $40 worth of BTC — simulated, no real order placed
order = bot.buy("BTC/USD", 40.0)
print(order)

# Sell 0.001 BTC — simulated
order = bot.sell("BTC/USD", 0.001)
print(order)
```

### Live trading

```python
from trader import CryptoTrader

bot = CryptoTrader(
    exchange_id="kraken",
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    paper_trading=False,   # enable real orders
)

# Will raise OrderSizeError if amount < $30 or > $50
order = bot.buy("BTC/USD", 45.00)
```

## Configuration (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `MIN_BUY_ORDER` | `30.0` | Minimum buy order size in USD |
| `MAX_BUY_ORDER` | `50.0` | Maximum buy order size in USD |
| `DEFAULT_EXCHANGE` | `"kraken"` | ccxt exchange identifier |
| `DEFAULT_SYMBOLS` | `["BTC/USD", "ETH/USD"]` | Trading pairs to monitor |
| `PAPER_TRADING` | `True` | Simulate orders without spending real money |

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```
