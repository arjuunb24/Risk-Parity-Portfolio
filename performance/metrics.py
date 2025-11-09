"""
Performance Metrics Module
Calculates comprehensive portfolio performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class PerformanceMetrics:
    """Calculate portfolio performance and risk metrics"""
    
    @staticmethod
    def calculate_total_return(portfolio_values: pd.DataFrame,
                              initial_capital: float) -> float:
        """Calculate total return over backtest period"""
        final_value = portfolio_values['value'].iloc[-1]
        total_return = (final_value / initial_capital) - 1
        return total_return
    
    @staticmethod
    def calculate_annualized_return(total_return: float,
                                   n_periods: int,
                                   periods_per_year: int = 252) -> float:
        """Calculate annualized return"""
        n_years = n_periods / periods_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        return annualized_return
    
    @staticmethod
    def calculate_volatility(returns: pd.Series,
                            annualize: bool = True,
                            periods_per_year: int = 252) -> float:
        """Calculate volatility (standard deviation of returns)"""
        volatility = returns.std()
        
        if annualize:
            volatility = volatility * np.sqrt(periods_per_year)
        
        return volatility
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series,
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio
        
        Sharpe = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
        """
        annual_return = returns.mean() * periods_per_year
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        if volatility == 0:
            return 0
        
        sharpe_ratio = (annual_return - risk_free_rate) / volatility
        
        return sharpe_ratio
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total volatility)
        """
        annual_return = returns.mean() * periods_per_year
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
        
        if downside_vol == 0:
            return 0
        
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol
        
        return sortino_ratio
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: pd.DataFrame) -> Dict:
        """
        Calculate maximum drawdown and related metrics
        
        Returns:
        --------
        dict : Contains max_drawdown, peak_date, trough_date, recovery_date, duration
        """
        values = portfolio_values['value']
        
        # Calculate running maximum
        running_max = values.expanding().max()
        
        # Calculate drawdown
        drawdown = (values - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        
        # Find peak date (last peak before trough)
        peak_date = running_max.loc[:trough_date].idxmax()
        
        # Find recovery date (first date back to peak value after trough)
        peak_value = values.loc[peak_date]
        post_trough = values.loc[trough_date:]
        recovery_dates = post_trough[post_trough >= peak_value]
        
        if len(recovery_dates) > 0:
            recovery_date = recovery_dates.index[0]
            duration_days = (recovery_date - peak_date).days
        else:
            recovery_date = None
            duration_days = (values.index[-1] - peak_date).days
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'peak_date': peak_date,
            'trough_date': trough_date,
            'recovery_date': recovery_date,
            'duration_days': duration_days,
            'drawdown_series': drawdown
        }
    
    @staticmethod
    def calculate_calmar_ratio(annual_return: float,
                              max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio
        
        Calmar = Annualized Return / |Maximum Drawdown|
        """
        if max_drawdown == 0:
            return 0
        
        calmar_ratio = annual_return / abs(max_drawdown)
        
        return calmar_ratio
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Calculate percentage of positive return periods"""
        win_rate = (returns > 0).sum() / len(returns)
        return win_rate
    
    @staticmethod
    def calculate_value_at_risk(returns: pd.Series,
                               confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at given confidence level
        """
        var = returns.quantile(1 - confidence)
        return var
    
    @staticmethod
    def calculate_conditional_var(returns: pd.Series,
                                 confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        Average of returns below VaR threshold
        """
        var = PerformanceMetrics.calculate_value_at_risk(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio
        
        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        """
        excess_returns = portfolio_returns - benchmark_returns
        
        if excess_returns.std() == 0:
            return 0
        
        ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return ir
    
    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta relative to benchmark"""
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        
        if benchmark_variance == 0:
            return 0
        
        beta = covariance / benchmark_variance
        
        return beta
    
    @staticmethod
    def calculate_alpha(portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       risk_free_rate: float = 0.02,
                       periods_per_year: int = 252) -> float:
        """
        Calculate Jensen's Alpha
        
        Alpha = Portfolio Return - (Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate))
        """
        portfolio_annual_return = portfolio_returns.mean() * periods_per_year
        benchmark_annual_return = benchmark_returns.mean() * periods_per_year
        
        beta = PerformanceMetrics.calculate_beta(portfolio_returns, benchmark_returns)
        
        expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        alpha = portfolio_annual_return - expected_return
        
        return alpha
    
    @staticmethod
    def generate_performance_report(portfolio_returns: pd.DataFrame,
                                   portfolio_values: pd.DataFrame,
                                   initial_capital: float,
                                   benchmark_returns: Optional[pd.Series] = None,
                                   risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate comprehensive performance report
        """
        returns = portfolio_returns['return']
        
        # Calculate all metrics
        total_return = PerformanceMetrics.calculate_total_return(portfolio_values, initial_capital)
        annual_return = PerformanceMetrics.calculate_annualized_return(
            total_return, len(returns)
        )
        volatility = PerformanceMetrics.calculate_volatility(returns)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate)
        
        dd_metrics = PerformanceMetrics.calculate_max_drawdown(portfolio_values)
        calmar = PerformanceMetrics.calculate_calmar_ratio(annual_return, dd_metrics['max_drawdown'])
        
        win_rate = PerformanceMetrics.calculate_win_rate(returns)
        var_95 = PerformanceMetrics.calculate_value_at_risk(returns, 0.95)
        cvar_95 = PerformanceMetrics.calculate_conditional_var(returns, 0.95)
        
        # Build report
        report = {
            'Total Return': total_return,
            'Annualized Return': annual_return,
            'Volatility (Annual)': volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': dd_metrics['max_drawdown'],
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Best Day': returns.max(),
            'Worst Day': returns.min()
        }
        
        # Add benchmark comparison if available
        if benchmark_returns is not None:
            beta = PerformanceMetrics.calculate_beta(returns, benchmark_returns)
            alpha = PerformanceMetrics.calculate_alpha(returns, benchmark_returns, risk_free_rate)
            ir = PerformanceMetrics.calculate_information_ratio(returns, benchmark_returns)
            
            report['Beta'] = beta
            report['Alpha'] = alpha
            report['Information Ratio'] = ir
        
        report_df = pd.DataFrame(report, index=['Value']).T
        
        return report_df