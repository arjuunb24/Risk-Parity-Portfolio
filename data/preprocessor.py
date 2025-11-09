"""
Data Preprocessor Module
Calculates returns and prepares data for portfolio optimization
"""

import pandas as pd
import numpy as np
from typing import Tuple


class DataPreprocessor:
    """Processes price data into returns for portfolio analysis"""
    
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.returns = None
        self.log_returns = None
        
    def calculate_returns(self, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate asset returns
        
        Parameters:
        -----------
        method : str
            'simple' for arithmetic returns or 'log' for logarithmic returns
        """
        if method == 'simple':
            self.returns = self.prices.pct_change().dropna()
        elif method == 'log':
            self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        # Store both for flexibility
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        print(f"Calculated {method} returns: {len(self.returns)} periods")
        return self.returns
    
    def get_rolling_statistics(self, window: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate rolling mean and volatility
        
        Parameters:
        -----------
        window : int
            Rolling window size (default 252 = 1 year of trading days)
        """
        if self.returns is None:
            self.calculate_returns()
        
        rolling_mean = self.returns.rolling(window=window).mean()
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        return rolling_mean, rolling_vol
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame = None, 
                                   annualize: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix of returns
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Returns data (uses self.returns if None)
        annualize : bool
            Whether to annualize the covariance matrix
        """
        if returns is None:
            if self.returns is None:
                self.calculate_returns()
            returns = self.returns
        
        cov_matrix = returns.cov()
        
        if annualize:
            cov_matrix = cov_matrix * 252  # Annualize
        
        return cov_matrix
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate correlation matrix of returns"""
        if returns is None:
            if self.returns is None:
                self.calculate_returns()
            returns = self.returns
        
        return returns.corr()
    
    def get_asset_statistics(self) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each asset
        """
        if self.returns is None:
            self.calculate_returns()
        
        stats = pd.DataFrame({
            'Mean Return (Annual)': self.returns.mean() * 252,
            'Volatility (Annual)': self.returns.std() * np.sqrt(252),
            'Sharpe Ratio': (self.returns.mean() * 252) / (self.returns.std() * np.sqrt(252)),
            'Min Return': self.returns.min(),
            'Max Return': self.returns.max(),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis()
        })
        
        return stats.round(4)
    
    def align_data(self, start_date: str = None, end_date: str = None) -> 'DataPreprocessor':
        """Align data to specific date range"""
        if start_date or end_date:
            self.prices = self.prices.loc[start_date:end_date]
            if self.returns is not None:
                self.returns = self.returns.loc[start_date:end_date]
        
        return self
    
    def remove_outliers(self, n_std: float = 5.0) -> pd.DataFrame:
        """
        Remove extreme outliers from returns
        
        Parameters:
        -----------
        n_std : float
            Number of standard deviations to use as threshold
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Calculate z-scores
        z_scores = np.abs((self.returns - self.returns.mean()) / self.returns.std())
        
        # Mask outliers
        cleaned_returns = self.returns.copy()
        cleaned_returns[z_scores > n_std] = np.nan
        
        # Forward fill to handle NaN
        cleaned_returns = cleaned_returns.fillna(method='ffill')
        
        outliers_removed = (z_scores > n_std).sum().sum()
        if outliers_removed > 0:
            print(f"Removed {outliers_removed} outliers ({n_std} std threshold)")
        
        return cleaned_returns
    
    def get_data_summary(self) -> dict:
        """Get summary of the dataset"""
        summary = {
            'n_assets': len(self.prices.columns),
            'n_periods': len(self.prices),
            'start_date': self.prices.index[0].strftime('%Y-%m-%d'),
            'end_date': self.prices.index[-1].strftime('%Y-%m-%d'),
            'assets': list(self.prices.columns)
        }
        
        if self.returns is not None:
            summary['n_return_periods'] = len(self.returns)
        
        return summary