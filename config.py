"""
Configuration file for Risk Parity Portfolio Project
Defines asset universe, parameters, and settings
"""

# Asset Universe - Global Multi-Asset Portfolio
ASSETS = {
    # US Equities (ETFs)
    'SPY': 'US Large Cap Equities',
    'QQQ': 'US Tech/Growth',
    'IWM': 'US Small Cap',
    
    # US Individual Stocks (Large Cap)
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',
    'NVDA': 'NVIDIA',
    'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson',
    'XOM': 'Exxon Mobil',
    
    '''# International Equities (ETFs)
    'EFA': 'Developed Markets ex-US',
    'EEM': 'Emerging Markets',
    'FXI': 'China Large Cap',
    
    # European Stocks (LSE - London Stock Exchange)
    'BP.L': 'BP plc (UK)',
    'HSBA.L': 'HSBC Holdings (UK)',
    'SHEL.L': 'Shell plc (UK)',
    'VOD.L': 'Vodafone (UK)',
    
    # European Stocks (Euronext)
    'MC.PA': 'LVMH (France)',
    'OR.PA': "L'Oréal (France)",
    'SAN.PA': 'Sanofi (France)',
    
    # German Stocks (XETRA)
    'SAP.DE': 'SAP SE (Germany)',
    'SIE.DE': 'Siemens (Germany)',
    'VOW3.DE': 'Volkswagen (Germany)',
    
    # Asian Stocks (Japan)
    '7203.T': 'Toyota Motor (Japan)',
    '6758.T': 'Sony Group (Japan)',
    '9984.T': 'SoftBank Group (Japan)',
    
    # Asian Stocks (Hong Kong)
    '0700.HK': 'Tencent (Hong Kong)',
    '9988.HK': 'Alibaba (Hong Kong)',
    '0941.HK': 'China Mobile (Hong Kong)',
    
    # Asian Stocks (India)
    'RELIANCE.NS': 'Reliance Industries (India)',
    'TCS.NS': 'Tata Consultancy (India)',
    'INFY.NS': 'Infosys (India)',
    
    # Australian Stocks
    'BHP.AX': 'BHP Group (Australia)',
    'CBA.AX': 'Commonwealth Bank (Australia)',
    'CSL.AX': 'CSL Limited (Australia)',
    
    # Canadian Stocks
    'RY.TO': 'Royal Bank of Canada',
    'TD.TO': 'TD Bank',
    'SHOP.TO': 'Shopify',
    
    # Fixed Income
    'TLT': 'US Long-Term Treasury',
    'IEF': 'US 7-10Y Treasury',
    'LQD': 'Investment Grade Corporate',
    'HYG': 'High Yield Corporate',
    'EMB': 'Emerging Market Bonds',
    '''
    # Commodities
    'GLD': 'Gold',
    'SLV': 'Silver',
    'USO': 'Oil',
    'DBC': 'Commodity Index',
    
    # Crypto
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    
    # Real Estate
    'VNQ': 'US REITs',
    'VNQI': 'International REITs'
}

# Market-specific suffixes for Yahoo Finance
# These are automatically handled by yFinance, but good to document:
MARKET_SUFFIXES = {
    'US': '',           # No suffix (AAPL, MSFT)
    'London': '.L',     # London Stock Exchange (BP.L)
    'Paris': '.PA',     # Euronext Paris (MC.PA)
    'Germany': '.DE',   # XETRA (SAP.DE)
    'Tokyo': '.T',      # Tokyo Stock Exchange (7203.T)
    'Hong Kong': '.HK', # Hong Kong Exchange (0700.HK)
    'India NSE': '.NS', # National Stock Exchange India (RELIANCE.NS)
    'India BSE': '.BO', # Bombay Stock Exchange (RELIANCE.BO)
    'Australia': '.AX', # Australian Securities Exchange (BHP.AX)
    'Canada': '.TO',    # Toronto Stock Exchange (RY.TO)
}

# Asset categorization for analysis
ASSET_CATEGORIES = {
    'US_Equities': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'JNJ', 'XOM'],
    'European_Equities': ['BP.L', 'HSBA.L', 'SHEL.L', 'VOD.L', 'MC.PA', 'OR.PA', 'SAN.PA', 'SAP.DE', 'SIE.DE', 'VOW3.DE'],
    'Asian_Equities': ['7203.T', '6758.T', '9984.T', '0700.HK', '9988.HK', '0941.HK', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
    'Pacific_Equities': ['BHP.AX', 'CBA.AX', 'CSL.AX'],
    'Canadian_Equities': ['RY.TO', 'TD.TO', 'SHOP.TO'],
    'International_ETFs': ['EFA', 'EEM', 'FXI'],
    'Fixed_Income': ['TLT', 'IEF', 'LQD', 'HYG', 'EMB'],
    'Commodities': ['GLD', 'SLV', 'USO', 'DBC'],
    'Crypto': ['BTC-USD', 'ETH-USD'],
    'Real_Estate': ['VNQ', 'VNQI']
}

# Backtesting Parameters
START_DATE = '2020-01-01'  # 4 years of data
END_DATE = '2024-12-31'
INITIAL_CAPITAL = 100000  # $100k initial investment

# Rebalancing Settings
REBALANCING_FREQUENCY = 'Q'  # 'M' = Monthly, 'Q' = Quarterly, 'Y' = Yearly
ROLLING_WINDOW = 252  # Trading days for covariance estimation (1 year)
MIN_HISTORY = 252  # Minimum history required before first rebalance

# Optimization Parameters
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRANSACTION_COST_BPS = 5  # 5 basis points (0.05%) per trade
MIN_WEIGHT = 0.01  # Minimum 1% allocation per asset
MAX_WEIGHT = 0.30  # Maximum 30% allocation per asset

# Risk Parity Settings
RISK_PARITY_METHOD = 'ERC'  # Equal Risk Contribution
OPTIMIZATION_TOLERANCE = 1e-8
MAX_ITERATIONS = 1000

# Benchmark
BENCHMARK_TICKER = 'SPY'  # S&P 500 as benchmark
USE_EQUAL_WEIGHT_BENCHMARK = True  # Also compare to equal-weight portfolio

# Output Settings
OUTPUT_DIR = 'output/'
SAVE_PLOTS = True
GENERATE_HTML_REPORT = True
VERBOSE = True