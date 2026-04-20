"""
Configuration settings for the Automated Crypto Trader.
"""

# Order size limits (in USD)
MIN_BUY_ORDER = 30.0   # Minimum buy order amount in USD
MAX_BUY_ORDER = 50.0   # Maximum buy order amount in USD

# Default exchange to use (must be supported by ccxt)
DEFAULT_EXCHANGE = "kraken"

# Default trading pairs to monitor
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD"]

# Paper trading mode — when True, no real orders are placed
PAPER_TRADING = False

# Profit taking and stop loss (as fractions of entry price)
TAKE_PROFIT_PCT = 0.075   # Close position when price rises 7.5 % above entry
STOP_LOSS_PCT   = 0.025   # Close position when price falls 2.5 % below entry

# Technical indicator settings
EMA_PERIOD     = 200   # 200-period Exponential Moving Average (utility; not used by buy signal)
RSI_PERIOD     = 14    # RSI look-back period (Wilder smoothing)
RSI_OVERSOLD   = 30    # RSI below this level → oversold (potential buy)
RSI_OVERBOUGHT = 70    # RSI above this level → overbought (potential sell)

# SimpleAlgo signal — EMA crossover proxy (short over long = bullish momentum)
SIMPLE_ALGO_SHORT_PERIOD = 9    # Short-term EMA period
SIMPLE_ALGO_LONG_PERIOD  = 21   # Long-term EMA period

# Volume Profile HD settings
VOLUME_PROFILE_BINS = 50  # Number of equal-width price bins for the volume profile

# How often (in seconds) to refresh the list of tradeable pairs from the exchange
SYMBOL_REFRESH_INTERVAL = 3600  # 1 hour

# Minimum 24-hour trading volume as a fraction of the asset's market
# capitalisation required before a buy order is placed or a buy signal is
# issued.  The exchange must supply a market-cap figure (via the ticker
# ``info`` dict).  Markets whose volume/market-cap ratio is below this
# threshold are considered too illiquid to reliably fill or exit orders.
MIN_VOLUME_MARKET_CAP_PCT = 0.04  # 4% of market cap

# Absolute minimum 24-hour quote-currency volume (USD) used as a fallback
# when the exchange does not supply a market-cap figure.  Any symbol whose
# 24-hour volume is below this threshold is still rejected as too illiquid.
MIN_VOLUME_USD = 500_000.0  # $500,000

# Maximum allowed bid-ask spread expressed as a fraction of the ask price.
# A spread above this threshold indicates insufficient liquidity or a
# market-maker-dominated book, and no buy order or buy signal is issued.
MAX_BID_ASK_SPREAD_PCT = 0.005  # 0.5 %

# ---------------------------------------------------------------------------
# Order bundles
# ---------------------------------------------------------------------------

# Named groups of USD-quoted symbols that can be bought together in one call
# via CryptoTrader.buy_bundle().  Add, remove, or rename bundles freely.
BUNDLES: dict = {
    "large_caps": ["BTC/USD", "ETH/USD"],
    "defi":       ["LINK/USD", "UNI/USD", "AAVE/USD"],
    "layer1":     ["SOL/USD", "ADA/USD", "DOT/USD"],
}

# When True the main trading loop uses bundle-based buying: a buy signal for
# any symbol that belongs to a bundle triggers a buy for the entire bundle.
# When False (default) the loop behaves exactly as before — single symbols.
USE_BUNDLES = False
