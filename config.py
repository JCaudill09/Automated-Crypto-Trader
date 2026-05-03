"""
Configuration settings for the Automated Crypto Trader.

Kraken supports trading of cryptocurrencies via USD-quoted pairs
(e.g. ``"BTC/USD"``, ``"ETH/USD"``).
"""

# Order size limits (in USD)
MIN_BUY_ORDER = 30.0   # Minimum buy order amount in USD
MAX_BUY_ORDER = 78.0   # Maximum buy order amount in USD

# Default exchange to use (must be supported by ccxt)
DEFAULT_EXCHANGE = "kraken"

# Default crypto trading pairs to monitor
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD"]

# Paper trading mode — when True, no real orders are placed
PAPER_TRADING = False

# Maximum time (in seconds) a position may remain open before it is
# automatically closed at market price.  Default is 24 hours.
MAX_POSITION_HOLD_SECONDS = 86400  # 24 hours

# Profit taking and stop loss (as fractions of entry price)
TAKE_PROFIT_PCT = 0.065   # Close position when price rises 6.5 % above entry
STOP_LOSS_PCT   = 0.0175  # Close position when price falls 1.75 % below entry

# 200-period EMA — buy signal when price crosses above this line
EMA_PERIOD = 200   # Long-term trend filter EMA period

# 13/48 EMA crossover — buy signal when EMA 13 crosses above EMA 48
EMA_CROSS_SHORT_PERIOD = 13   # Short-term EMA period for the crossover signal
EMA_CROSS_LONG_PERIOD  = 48   # Long-term EMA period for the crossover signal

# ATR (Average True Range) settings
ATR_PERIOD              = 14   # Look-back period for ATR computation
ATR_STOP_LOSS_MULTIPLIER = 1.5  # Stop-loss distance = ATR_STOP_LOSS_MULTIPLIER × ATR

# Asset types to include when discovering pairs from the exchange.
# ccxt market dicts carry a ``type`` field (e.g. ``"spot"``, ``"swap"``,
# ``"future"``).  Set this to a list of strings to restrict which types are
# returned by ``get_usd_symbols()``.  ``None`` (the default) disables
# type-filtering so every active USD crypto pair is included.
# Examples:
#   ASSET_TYPES = ["spot"]         # spot markets only
#   ASSET_TYPES = None             # no filter — include all market types
ASSET_TYPES: list | None = None

# How often (in seconds) to refresh the list of tradeable pairs from the exchange
SYMBOL_REFRESH_INTERVAL = 3600  # 1 hour

# Minimum 24-hour quote-currency volume (USD) required before a buy order is
# placed or a buy signal is issued.  Any symbol whose 24-hour volume is below
# this threshold is rejected as too illiquid.
MIN_VOLUME_USD = 15_000.0  # $15,000

# Maximum allowed bid-ask spread expressed as a fraction of the ask price.
# A spread above this threshold indicates insufficient liquidity or a
# market-maker-dominated book, and no buy order or buy signal is issued.
MAX_BID_ASK_SPREAD_PCT = 0.0075  # 0.75 %

# ---------------------------------------------------------------------------
# Order bundles
# ---------------------------------------------------------------------------

# Named groups of USD-quoted crypto symbols that can be bought together in one
# call via CryptoTrader.buy_bundle().  Add, remove, or rename bundles freely.
BUNDLES: dict = {
    "large_caps": ["BTC/USD", "ETH/USD"],
    "defi":       ["LINK/USD", "UNI/USD", "AAVE/USD"],
    "layer1":     ["SOL/USD", "ADA/USD", "DOT/USD"],
}

# When True the main trading loop uses bundle-based buying: a buy signal for
# any symbol that belongs to a bundle triggers a buy for the entire bundle.
# When False (default) the loop behaves exactly as before — single symbols.
USE_BUNDLES = False
