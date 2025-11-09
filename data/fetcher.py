"""
Data Fetcher Module
Handles downloading price data and currency conversion
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataFetcher:
    """Fetches and processes multi-asset price data with currency conversion"""
    
    def __init__(self, tickers: Dict[str, str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.currencies = None
        
    def fetch_prices(self) -> pd.DataFrame:
        """
        Download adjusted close prices for all assets
        Automatically converts to USD using yfinance's built-in conversion
        """
        print(f"Fetching data for {len(self.tickers)} assets from {self.start_date} to {self.end_date}...")
        
        ticker_list = list(self.tickers.keys())
        
        # Download data in batches to handle international stocks better
        batch_size = 10
        all_prices = []
        
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i+batch_size]
            print(f"  Downloading batch {i//batch_size + 1}/{(len(ticker_list)-1)//batch_size + 1}: {batch[0]}...")
            
            try:
                data = yf.download(
                    batch,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=True  # Enable multi-threading for faster downloads
                )
                
                # Extract adjusted close prices
                if len(batch) == 1:
                    prices_batch = data[['Close']].copy()
                    prices_batch.columns = batch
                else:
                    prices_batch = data['Close'].copy()
                
                all_prices.append(prices_batch)
                
            except Exception as e:
                print(f"  Warning: Failed to download {batch}: {str(e)}")
                continue
        
        # Combine all batches
        if len(all_prices) > 0:
            prices = pd.concat(all_prices, axis=1)
        else:
            raise ValueError("No data could be downloaded")
        
        # Handle missing data
        prices = self._handle_missing_data(prices)
        
        # Validate data
        self._validate_data(prices)
        
        # Convert all prices to USD
        prices = self._ensure_usd_prices(prices)
        
        self.prices = prices
        print(f"Successfully fetched data: {len(prices)} days, {len(prices.columns)} assets")
        return prices
    
    def _handle_missing_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using forward fill then backward fill"""
        # Check for missing data
        missing_pct = prices.isnull().sum() / len(prices) * 100
        
        if missing_pct.any():
            print("\nMissing data detected:")
            for ticker, pct in missing_pct[missing_pct > 0].items():
                print(f"  {ticker}: {pct:.2f}% missing")
        
        # Forward fill then backward fill
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        # Drop columns with all NaN
        prices = prices.dropna(axis=1, how='all')
        
        return prices
    
    def _validate_data(self, prices: pd.DataFrame):
        """Validate data quality"""
        if prices.empty:
            raise ValueError("No data retrieved")
        
        # Check for remaining NaN
        if prices.isnull().any().any():
            print("\nWarning: Some NaN values remain after cleaning")
            print(prices.isnull().sum()[prices.isnull().sum() > 0])
        
        # Check for zero or negative prices
        if (prices <= 0).any().any():
            print("\nWarning: Zero or negative prices detected")
            for col in prices.columns:
                if (prices[col] <= 0).any():
                    print(f"  {col}: {(prices[col] <= 0).sum()} invalid prices")
        
        # Check data length
        if len(prices) < 252:  # Less than 1 year of data
            print(f"\nWarning: Only {len(prices)} days of data available (less than 1 year)")
    
    def _ensure_usd_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all non-USD prices to USD using historical forex rates.
        For assets already in USD, returns unchanged.
        """
        print("\nChecking currency conversion requirements...")
        
        # Get currency info for each ticker
        currencies_needed = {}
        for ticker in prices.columns:
            try:
                info = yf.Ticker(ticker).info
                currency = info.get('currency', 'USD')
                if currency != 'USD':
                    currencies_needed[ticker] = currency
            except Exception as e:
                print(f"  Warning: Could not get currency info for {ticker}, assuming USD")
                continue
        
        if not currencies_needed:
            print("  All assets already in USD")
            return prices
        
        print(f"  Converting {len(currencies_needed)} assets to USD: {list(currencies_needed.keys())}")
        
        # Download forex rates for non-USD currencies
        unique_currencies = set(currencies_needed.values())
        forex_pairs = {curr: f"{curr}USD=X" for curr in unique_currencies}
        
        converted_prices = prices.copy()
        
        for ticker, currency in currencies_needed.items():
            try:
                # Download forex rate
                forex_ticker = forex_pairs[currency]
                print(f"    Converting {ticker} ({currency} → USD) using {forex_ticker}")
                
                forex_data = yf.download(
                    forex_ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )
                
                if forex_data.empty:
                    print(f"      Warning: No forex data for {forex_ticker}, skipping conversion")
                    continue
                
                # Get close prices and align with asset prices
                if 'Close' in forex_data.columns:
                    forex_rate = forex_data['Close']
                else:
                    forex_rate = forex_data
                
                # Reindex to match asset dates and forward fill missing forex rates
                forex_rate = forex_rate.reindex(prices.index, method='ffill')
                
                # Convert to USD
                converted_prices[ticker] = prices[ticker] * forex_rate
                
                print(f"      ✓ Converted {ticker} to USD")
                
            except Exception as e:
                print(f"      Warning: Failed to convert {ticker}: {str(e)}")
                print(f"      Keeping original prices for {ticker}")
                continue
        
        # Validate conversion
        if (converted_prices <= 0).any().any():
            print("\n  Warning: Some converted prices are zero or negative")
        
        return converted_prices
    
    def get_currency_info(self) -> pd.DataFrame:
        """Get currency information for each asset"""
        currency_info = []
        
        for ticker in self.tickers.keys():
            try:
                info = yf.Ticker(ticker).info
                currency = info.get('currency', 'USD')
                currency_info.append({
                    'ticker': ticker,
                    'currency': currency,
                    'name': self.tickers[ticker]
                })
            except:
                currency_info.append({
                    'ticker': ticker,
                    'currency': 'USD',
                    'name': self.tickers[ticker]
                })
        
        return pd.DataFrame(currency_info)
    
    def get_valid_tickers(self) -> List[str]:
        """Return list of tickers with valid data"""
        if self.prices is None:
            self.fetch_prices()
        return list(self.prices.columns)