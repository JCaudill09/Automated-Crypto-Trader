"""
Configuration settings for the Automated Crypto Trader.
"""

# Order size limits (in USD)
MIN_BUY_ORDER = 30.0   # Minimum buy order amount in USD
MAX_BUY_ORDER = 50.0   # Maximum buy order amount in USD

# Default exchange to use (must be supported by ccxt)
DEFAULT_EXCHANGE = "kraken"

# Default trading pairs to monitor
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# Paper trading mode — when True, no real orders are placed
PAPER_TRADING = False

# Profit taking and stop loss (as fractions of entry price)
TAKE_PROFIT_PCT = 0.075   # Close position when price rises 7.5 % above entry
STOP_LOSS_PCT   = 0.025   # Close position when price falls 2.5 % below entry

# Technical indicator settings
EMA_PERIOD     = 200   # 200-period Exponential Moving Average
RSI_PERIOD     = 14    # RSI look-back period (Wilder smoothing)
RSI_OVERSOLD   = 30    # RSI below this level → oversold (potential buy)
RSI_OVERBOUGHT = 70    # RSI above this level → overbought (potential sell)

# How often (in seconds) to refresh the list of tradeable pairs from the exchange
SYMBOL_REFRESH_INTERVAL = 3600  # 1 hour

# Minimum 24-hour trading volume in quote currency (USD / USDT) required
# before a buy order is placed or a buy signal is issued.  Markets below
# this threshold are considered too illiquid to reliably fill or exit orders.
MIN_VOLUME_USD = 1_000_000.0
