"""
Configuration settings for the Automated Crypto Trader.

Kraken supports trading of cryptocurrencies, tokenized stocks, and ETFs.
All three asset classes use USD-quoted pairs (e.g. ``"BTC/USD"``,
``"AAPL/USD"``, ``"SPY/USD"``) and are handled uniformly by the trader.
"""

# Order size limits (in USD)
MIN_BUY_ORDER = 30.0   # Minimum buy order amount in USD
MAX_BUY_ORDER = 78.0   # Maximum buy order amount in USD

# Default exchange to use (must be supported by ccxt)
DEFAULT_EXCHANGE = "kraken"

# Default trading pairs to monitor (crypto, stocks, and ETFs are all supported)
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "AAPL/USD", "SPY/USD"]

# Paper trading mode — when True, no real orders are placed
PAPER_TRADING = False

# Maximum time (in seconds) a position may remain open before it is
# automatically closed at market price.  Default is 24 hours.
MAX_POSITION_HOLD_SECONDS = 86400  # 24 hours

# Profit taking and stop loss (as fractions of entry price)
TAKE_PROFIT_PCT = 0.055   # Close position when price rises 5.5 % above entry
STOP_LOSS_PCT   = 0.0175  # Close position when price falls 1.75 % below entry

# Technical indicator settings
EMA_PERIOD     = 200   # 200-period Exponential Moving Average (utility; not used by buy signal)
RSI_PERIOD     = 14    # RSI look-back period (Wilder smoothing)
RSI_OVERSOLD   = 50    # RSI below this level → buy signal (< 50, momentum not yet overbought)
RSI_OVERBOUGHT = 70    # RSI above this level → overbought (potential sell)

# ATR (Average True Range) settings
ATR_PERIOD              = 14   # Look-back period for ATR computation
ATR_STOP_LOSS_MULTIPLIER = 1.5  # Stop-loss distance = ATR_STOP_LOSS_MULTIPLIER × ATR

# EMA golden-cross signal — EMA 50 crossing above EMA 200 indicates a bullish trend
SIMPLE_ALGO_SHORT_PERIOD = 50   # Short-term EMA period (EMA 50)
SIMPLE_ALGO_LONG_PERIOD  = 200  # Long-term EMA period (EMA 200)

# Bollinger Bands settings
BB_PERIOD  = 20   # Look-back period for Bollinger Bands
BB_NUM_STD = 2.0  # Number of standard deviations for the upper/lower bands

# Keltner Channel settings
KC_PERIOD     = 20   # EMA period and ATR period for Keltner Channels
KC_MULTIPLIER = 2.0  # Upper/lower channel distance = KC_MULTIPLIER × ATR

# Relative Volume (RVOL) settings
# RVOL = current candle volume / average volume over the preceding RVOL_PERIOD candles.
# A high RVOL indicates that the current candle is seeing unusually heavy participation.
RVOL_PERIOD    = 20   # Number of prior candles used to compute the average volume
RVOL_THRESHOLD = 5.0  # Buy signal requires current volume ≥ RVOL_THRESHOLD × average

# Volume Profile HD settings
VOLUME_PROFILE_BINS = 50  # Number of equal-width price bins for the volume profile

# Asset types to include when discovering pairs from the exchange.
# ccxt market dicts carry a ``type`` field (e.g. ``"spot"``, ``"swap"``,
# ``"future"``).  Set this to a list of strings to restrict which types are
# returned by ``get_usd_symbols()``.  ``None`` (the default) disables
# type-filtering so every active USD pair is included — cryptocurrencies,
# tokenized stocks, and ETFs alike.
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
MAX_BID_ASK_SPREAD_PCT = 0.01  # 1 %

# ---------------------------------------------------------------------------
# Order bundles
# ---------------------------------------------------------------------------

# Named groups of USD-quoted symbols that can be bought together in one call
# via CryptoTrader.buy_bundle().  Add, remove, or rename bundles freely.
BUNDLES: dict = {
    "large_caps": ["BTC/USD", "ETH/USD"],
    "defi":       ["LINK/USD", "UNI/USD", "AAVE/USD"],
    "layer1":     ["SOL/USD", "ADA/USD", "DOT/USD"],
    # Tokenized stocks available on Kraken
    "stocks":     ["AAPL/USD", "TSLA/USD", "MSFT/USD", "NVDA/USD", "AMZN/USD"],
    # ETFs available on Kraken
    "etfs":       ["SPY/USD", "QQQ/USD", "VOO/USD"],
}

# When True the main trading loop uses bundle-based buying: a buy signal for
# any symbol that belongs to a bundle triggers a buy for the entire bundle.
# When False (default) the loop behaves exactly as before — single symbols.
USE_BUNDLES = False

# ---------------------------------------------------------------------------
# Trading schedule
# ---------------------------------------------------------------------------

# When True the bot routes symbols based on the time of day:
#   • NYSE/Nasdaq hours (Mon–Fri 09:30–16:00 ET) → trade only STOCK_SYMBOLS
#   • All other times                             → trade only non-STOCK_SYMBOLS (crypto)
# Set to False to disable time-based routing and trade all symbols at all times.
TRADING_SCHEDULE_ENABLED = True

# Symbols treated as stocks or ETFs for schedule-based routing.
# Any symbol NOT in this list is treated as crypto.
STOCK_SYMBOLS: list = [
    # Tokenized stocks available on Kraken
    "AAPL/USD", "TSLA/USD", "MSFT/USD", "NVDA/USD", "AMZN/USD",
    # ETFs available on Kraken
    "SPY/USD", "QQQ/USD", "VOO/USD",
]
