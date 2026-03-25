"""
Configuration file for Risk Parity Portfolio Project
Defines asset universe, parameters, and settings
"""

# Asset Universe - Global Multi-Asset Portfolio
ASSETS = {
    
    'SPY': 'US Large Cap Equities', # US Equities (ETFs)
    'QQQ': 'US Tech/Growth',
    'IWM': 'US Small Cap',
    

    'AAPL': 'Apple Inc', # US Individual Stocks (Large Cap)
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',
    'NVDA': 'NVIDIA',
    'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson',
    'XOM': 'Exxon Mobil',
    
    # --- ADDITIONAL US STOCKS - to test out different combinations of assets ---
    # -- TECHNOLOGY --
    # 'TSLA': 'Tesla',
    # 'META': 'Meta Platforms',
    # 'AVGO': 'Broadcom',
    # 'ORCL': 'Oracle',
    # 'ADBE': 'Adobe',
    # 'CRM': 'Salesforce',
    # 'AMD': 'AMD',
    # 'CSCO': 'Cisco Systems',
    # 'INTC': 'Intel',
    # 'QCOM': 'Qualcomm',
    
    # -- FINANCIALS --
    # 'BAC': 'Bank of America',
    # 'V': 'Visa',
    # 'MA': 'Mastercard',
    # 'WFC': 'Wells Fargo',
    # 'MS': 'Morgan Stanley',
    # 'GS': 'Goldman Sachs',
    # 'SCHW': 'Charles Schwab',
    # 'BLK': 'BlackRock',
    # 'C': 'Citigroup',
    # 'AXP': 'American Express',
    
    # -- HEALTHCARE --
    # 'UNH': 'UnitedHealth Group',
    # 'LLY': 'Eli Lilly',
    # 'PFE': 'Pfizer',
    # 'ABBV': 'AbbVie',
    # 'MRK': 'Merck',
    # 'TMO': 'Thermo Fisher',
    # 'DHR': 'Danaher',
    # 'AMGN': 'Amgen',
    # 'ISRG': 'Intuitive Surgical',
    # 'GILD': 'Gilead Sciences',
    
    # -- CONSUMER / RETAIL --
    # 'WMT': 'Walmart',
    # 'COST': 'Costco',
    # 'HD': 'Home Depot',
    # 'PG': 'Procter & Gamble',
    # 'KO': 'Coca-Cola',
    # 'PEP': 'PepsiCo',
    # 'MCD': 'McDonalds',
    # 'NKE': 'Nike',
    # 'DIS': 'Disney',
    # 'SBUX': 'Starbucks'
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