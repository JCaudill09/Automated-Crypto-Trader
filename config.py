"""
Configuration settings for the Automated Crypto Trader.
"""

# Order size limits (in USD)
MIN_BUY_ORDER = 30.0   # Minimum buy order amount in USD
MAX_BUY_ORDER = 50.0   # Maximum buy order amount in USD

# Default exchange to use (must be supported by ccxt)
DEFAULT_EXCHANGE = "coinbase"

# Default trading pairs to monitor
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# Paper trading mode — when True, no real orders are placed
PAPER_TRADING = True
