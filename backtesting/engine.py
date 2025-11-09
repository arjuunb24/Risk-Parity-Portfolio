"""
Backtesting Engine Module
Simulates portfolio performance over historical period
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class BacktestEngine:
    """
    Backtesting framework for portfolio strategies
    Handles transaction costs and realistic execution
    """
    
    def __init__(self,
                 prices: pd.DataFrame,
                 returns: pd.DataFrame,
                 initial_capital: float = 100000,
                 transaction_cost_bps: float = 5):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Historical prices
        returns : pd.DataFrame
            Historical returns
        initial_capital : float
            Starting capital
        transaction_cost_bps : float
            Transaction cost in basis points (5 bps = 0.05%)
        """
        self.prices = prices
        self.returns = returns
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps / 10000  # Convert to decimal
        
        # Results storage
        self.portfolio_values = []
        self.portfolio_returns = []
        self.portfolio_weights_history = []
        self.rebalance_dates = []
        self.transaction_costs = []
        
    def run_backtest(self,
                    weights_schedule: pd.DataFrame,
                    rebalance_dates: List[str]) -> Dict:
        """
        Run backtest with predetermined weights at rebalance dates
        
        Parameters:
        -----------
        weights_schedule : pd.DataFrame
            DataFrame with rebalance dates as index and assets as columns
        rebalance_dates : list
            List of rebalance dates
        
        Returns:
        --------
        dict : Backtest results including portfolio values, returns, weights history
        """
        portfolio_value = self.initial_capital
        current_weights = None
        
        # Initialize storage
        portfolio_values = []
        portfolio_returns = []
        weights_history = []
        transaction_costs_history = []
        
        # Get all dates
        all_dates = self.returns.index
        
        # Track holdings in shares
        holdings_shares = pd.Series(0.0, index=self.returns.columns)
        
        for i, date in enumerate(all_dates):
            # Check if rebalance date
            is_rebalance = date in rebalance_dates
            
            if is_rebalance:
                # Get new target weights
                target_weights = weights_schedule.loc[date]
                
                # Calculate transaction costs
                if current_weights is not None:
                    # Turnover: sum of absolute weight changes
                    turnover = np.abs(target_weights - current_weights).sum()
                    transaction_cost = portfolio_value * turnover * self.transaction_cost_bps
                else:
                    # First rebalance: cost to establish positions
                    transaction_cost = portfolio_value * self.transaction_cost_bps
                
                # Deduct transaction costs
                portfolio_value -= transaction_cost
                transaction_costs_history.append({
                    'date': date,
                    'cost': transaction_cost,
                    'portfolio_value': portfolio_value
                })
                
                # Calculate new holdings (shares)
                current_prices = self.prices.loc[date, target_weights.index]
                position_values = portfolio_value * target_weights
                holdings_shares = position_values / current_prices
                
                # Update current weights
                current_weights = target_weights.copy()
                
                # Store weights
                weights_history.append({
                    'date': date,
                    'weights': current_weights.to_dict(),
                    'is_rebalance': True
                })
            
            # Calculate daily return
            if i > 0 and current_weights is not None:
                # Get returns for this period
                period_returns = self.returns.loc[date, current_weights.index]
                
                # Calculate portfolio return
                portfolio_return = (current_weights * period_returns).sum()
                
                # Update portfolio value
                portfolio_value = portfolio_value * (1 + portfolio_return)
                
                # Update weights due to price changes (drift)
                current_prices = self.prices.loc[date, current_weights.index]
                position_values = holdings_shares * current_prices
                current_weights = position_values / position_values.sum()
                
                # Store results
                portfolio_returns.append({
                    'date': date,
                    'return': portfolio_return,
                    'portfolio_value': portfolio_value
                })
                
                if not is_rebalance:
                    weights_history.append({
                        'date': date,
                        'weights': current_weights.to_dict(),
                        'is_rebalance': False
                    })
            
            portfolio_values.append({
                'date': date,
                'value': portfolio_value
            })
        
        # Convert to DataFrames
        self.portfolio_values = pd.DataFrame(portfolio_values).set_index('date')
        self.portfolio_returns = pd.DataFrame(portfolio_returns).set_index('date')
        self.portfolio_weights_history = pd.DataFrame(weights_history)
        self.transaction_costs = pd.DataFrame(transaction_costs_history)
        
        # Calculate cumulative returns
        self.portfolio_values['cumulative_return'] = (
            self.portfolio_values['value'] / self.initial_capital - 1
        )
        
        # Package results
        results = {
            'portfolio_values': self.portfolio_values,
            'portfolio_returns': self.portfolio_returns,
            'weights_history': self.portfolio_weights_history,
            'transaction_costs': self.transaction_costs,
            'final_value': portfolio_value,
            'total_return': (portfolio_value / self.initial_capital) - 1,
            'total_transaction_costs': self.transaction_costs['cost'].sum() if len(self.transaction_costs) > 0 else 0
        }
        
        return results
    
    def run_benchmark_backtest(self, 
                               benchmark_weights: pd.Series,
                               rebalance_freq: str = 'Q') -> Dict:
        """
        Run backtest for benchmark portfolio (e.g., equal-weight)
        
        Parameters:
        -----------
        benchmark_weights : pd.Series
            Static benchmark weights
        rebalance_freq : str
            Rebalancing frequency ('M', 'Q', 'Y')
        """
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(rebalance_freq)
        
        # Create weights schedule (constant weights at each rebalance)
        weights_schedule = pd.DataFrame(
            index=rebalance_dates,
            columns=benchmark_weights.index
        )
        
        for date in rebalance_dates:
            weights_schedule.loc[date] = benchmark_weights
        
        # Run backtest
        results = self.run_backtest(weights_schedule, rebalance_dates)
        
        return results
    
    def _generate_rebalance_dates(self, freq: str) -> List[pd.Timestamp]:
        """Generate rebalancing dates based on frequency"""
        # Get date range
        start_date = self.returns.index[0]
        end_date = self.returns.index[-1]
        
        # Generate rebalance dates
        if freq == 'M':  # Monthly
            dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        elif freq == 'Q':  # Quarterly
            dates = pd.date_range(start=start_date, end=end_date, freq='QS')
        elif freq == 'Y':  # Yearly
            dates = pd.date_range(start=start_date, end=end_date, freq='YS')
        else:
            raise ValueError(f"Invalid frequency: {freq}")
        
        # Filter to dates in our data
        valid_dates = [d for d in dates if d in self.returns.index]
        
        return valid_dates
    
    def compare_strategies(self,
                          strategy_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Parameters:
        -----------
        strategy_results : dict
            Dictionary of strategy names to backtest results
        """
        comparison = []
        
        for name, results in strategy_results.items():
            portfolio_values = results['portfolio_values']
            portfolio_returns = results['portfolio_returns']
            
            # Calculate metrics
            total_return = results['total_return']
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns['return'].std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            # Max drawdown
            cumulative = (1 + portfolio_returns['return']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            comparison.append({
                'Strategy': name,
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_drawdown,
                'Final Value': results['final_value'],
                'Total Costs': results['total_transaction_costs']
            })
        
        comparison_df = pd.DataFrame(comparison).set_index('Strategy')
        
        return comparison_df