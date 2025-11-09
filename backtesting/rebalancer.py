"""
Dynamic Rebalancer Module
Handles periodic portfolio rebalancing with rolling window optimization
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import sys
sys.path.append('..')


class DynamicRebalancer:
    """
    Manages dynamic portfolio rebalancing with rolling window updates
    """
    
    def __init__(self,
                 returns: pd.DataFrame,
                 rolling_window: int = 252,
                 rebalance_freq: str = 'Q',
                 min_weight: float = 0.01,
                 max_weight: float = 0.30):
        """
        Initialize rebalancer
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns
        rolling_window : int
            Number of periods for rolling covariance estimation
        rebalance_freq : str
            Rebalancing frequency ('M', 'Q', 'Y')
        min_weight : float
            Minimum weight constraint
        max_weight : float
            Maximum weight constraint
        """
        self.returns = returns
        self.rolling_window = rolling_window
        self.rebalance_freq = rebalance_freq
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Import optimizer here to avoid circular imports
        from portfolio.optimizer import RiskParityOptimizer
        self.optimizer_class = RiskParityOptimizer
        
    def generate_rebalance_schedule(self) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
        """
        Generate complete rebalance schedule with optimized weights
        
        Returns:
        --------
        weights_schedule : pd.DataFrame
            Weights at each rebalance date
        rebalance_dates : list
            List of rebalance dates
        """
        # Generate rebalance dates
        rebalance_dates = self._get_rebalance_dates()
        
        weights_schedule = []
        
        print(f"Generating rebalance schedule: {len(rebalance_dates)} rebalances")
        
        for i, date in enumerate(rebalance_dates):
            # Get historical data up to this date
            hist_returns = self._get_historical_returns(date)
            
            if len(hist_returns) < self.rolling_window:
                print(f"  Skipping {date}: insufficient history ({len(hist_returns)} < {self.rolling_window})")
                continue
            
            # Calculate covariance matrix for this period
            cov_matrix = self._calculate_rolling_covariance(hist_returns)
            
            # Optimize weights
            optimal_weights = self._optimize_weights(cov_matrix)
            
            weights_schedule.append({
                'date': date,
                **optimal_weights.to_dict()
            })
            
            if (i + 1) % 5 == 0 or (i + 1) == len(rebalance_dates):
                print(f"  Completed {i + 1}/{len(rebalance_dates)} rebalances")
        
        # Convert to DataFrame
        weights_df = pd.DataFrame(weights_schedule).set_index('date')
        valid_dates = list(weights_df.index)
        
        print(f"Rebalance schedule complete: {len(valid_dates)} periods")
        
        return weights_df, valid_dates
    
    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Generate rebalancing dates based on frequency"""
        start_date = self.returns.index[self.rolling_window]  # Start after rolling window
        end_date = self.returns.index[-1]
        
        # Generate dates based on frequency
        if self.rebalance_freq == 'M':  # Monthly
            dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        elif self.rebalance_freq == 'Q':  # Quarterly
            dates = pd.date_range(start=start_date, end=end_date, freq='QS')
        elif self.rebalance_freq == 'Y':  # Yearly
            dates = pd.date_range(start=start_date, end=end_date, freq='YS')
        elif self.rebalance_freq == 'W':  # Weekly
            dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        else:
            raise ValueError(f"Invalid rebalance frequency: {self.rebalance_freq}")
        
        # Filter to actual trading dates in our data
        valid_dates = []
        for date in dates:
            # Find the nearest available date
            nearest_date = self.returns.index[self.returns.index >= date][0]
            valid_dates.append(nearest_date)
        
        return valid_dates
    
    def _get_historical_returns(self, date: pd.Timestamp) -> pd.DataFrame:
        """Get historical returns up to (but not including) the rebalance date"""
        # Get all returns before this date
        hist_returns = self.returns.loc[:date].iloc[:-1]  # Exclude rebalance date itself
        
        # Use only the rolling window
        if len(hist_returns) > self.rolling_window:
            hist_returns = hist_returns.iloc[-self.rolling_window:]
        
        return hist_returns
    
    def _calculate_rolling_covariance(self, hist_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate annualized covariance matrix from historical returns"""
        cov_matrix = hist_returns.cov() * 252  # Annualize
        return cov_matrix
    
    def _optimize_weights(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """Optimize weights using risk parity"""
        optimizer = self.optimizer_class(
            cov_matrix=cov_matrix,
            min_weight=self.min_weight,
            max_weight=self.max_weight
        )
        
        # Use SciPy optimization
        optimal_weights = optimizer.optimize_scipy(method='SLSQP')
        
        return optimal_weights
    
    def analyze_weight_stability(self, weights_schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze stability of weights over time
        """
        # Calculate weight changes
        weight_changes = weights_schedule.diff().abs()
        
        stability_metrics = pd.DataFrame({
            'Mean Weight': weights_schedule.mean(),
            'Std Weight': weights_schedule.std(),
            'Min Weight': weights_schedule.min(),
            'Max Weight': weights_schedule.max(),
            'Mean Change': weight_changes.mean(),
            'Max Change': weight_changes.max()
        })
        
        return stability_metrics
    
    def get_turnover_statistics(self, weights_schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio turnover at each rebalance
        """
        # Calculate turnover (sum of absolute weight changes)
        weight_changes = weights_schedule.diff().abs()
        turnover = weight_changes.sum(axis=1)
        
        turnover_stats = pd.DataFrame({
            'date': turnover.index,
            'turnover': turnover.values,
            'turnover_pct': turnover.values * 100
        }).set_index('date')
        
        return turnover_stats
    
    def simulate_adaptive_rebalancing(self,
                                     volatility_threshold: float = 0.05) -> pd.DataFrame:
        """
        Simulate adaptive rebalancing based on volatility changes
        Rebalances more frequently when volatility increases
        
        Parameters:
        -----------
        volatility_threshold : float
            Rebalance if portfolio volatility changes by more than this amount
        """
        rebalance_signals = []
        
        # Calculate rolling volatility
        rolling_vol = self.returns.rolling(window=20).std() * np.sqrt(252)
        
        last_rebalance_vol = None
        
        for date in rolling_vol.index[self.rolling_window:]:
            current_vol = rolling_vol.loc[date].mean()
            
            should_rebalance = False
            
            if last_rebalance_vol is None:
                should_rebalance = True
            elif abs(current_vol - last_rebalance_vol) / last_rebalance_vol > volatility_threshold:
                should_rebalance = True
            
            if should_rebalance:
                rebalance_signals.append({
                    'date': date,
                    'volatility': current_vol,
                    'vol_change': 0 if last_rebalance_vol is None else (current_vol - last_rebalance_vol)
                })
                last_rebalance_vol = current_vol
        
        return pd.DataFrame(rebalance_signals)