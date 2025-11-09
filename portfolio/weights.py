"""
Portfolio Weighting Module
Implements volatility-based initial weighting
"""

import pandas as pd
import numpy as np
from typing import Dict


class PortfolioWeights:
    """Calculate initial portfolio weights based on volatility"""
    
    @staticmethod
    def inverse_volatility_weights(returns: pd.DataFrame, 
                                   min_weight: float = 0.0,
                                   max_weight: float = 1.0) -> pd.Series:
        """
        Calculate weights inversely proportional to volatility
        Lower volatility assets get higher weights
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        min_weight : float
            Minimum weight constraint
        max_weight : float
            Maximum weight constraint
        
        Returns:
        --------
        pd.Series : Normalized weights
        """
        # Calculate annualized volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Inverse volatility
        inv_vol = 1 / volatility
        
        # Normalize to sum to 1
        weights = inv_vol / inv_vol.sum()
        
        # Apply constraints
        weights = weights.clip(lower=min_weight, upper=max_weight)
        
        # Re-normalize after clipping
        weights = weights / weights.sum()
        
        return weights
    
    @staticmethod
    def equal_weights(returns: pd.DataFrame) -> pd.Series:
        """Calculate equal weights (1/N)"""
        n_assets = len(returns.columns)
        weights = pd.Series(1/n_assets, index=returns.columns)
        return weights
    
    @staticmethod
    def minimum_variance_weights(cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate minimum variance portfolio weights
        
        Parameters:
        -----------
        cov_matrix : pd.DataFrame
            Covariance matrix of returns
        """
        # Inverse of covariance matrix
        inv_cov = np.linalg.inv(cov_matrix)
        
        # Ones vector
        ones = np.ones(len(cov_matrix))
        
        # Minimum variance weights: (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        weights = inv_cov @ ones
        weights = weights / (ones @ weights)
        
        weights = pd.Series(weights, index=cov_matrix.index)
        
        # Ensure no negative weights
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()
        
        return weights
    
    @staticmethod
    def risk_budget_weights(returns: pd.DataFrame, 
                           risk_budgets: Dict[str, float] = None) -> pd.Series:
        """
        Calculate weights based on risk budgets
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        risk_budgets : dict
            Custom risk budget for each asset (must sum to 1)
        """
        if risk_budgets is None:
            # Equal risk budget
            n_assets = len(returns.columns)
            risk_budgets = {asset: 1/n_assets for asset in returns.columns}
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Weights proportional to risk budget / volatility
        weights = pd.Series(risk_budgets) / volatility
        weights = weights / weights.sum()
        
        return weights
    
    @staticmethod
    def get_portfolio_volatility(weights: pd.Series, 
                                 cov_matrix: pd.DataFrame) -> float:
        """
        Calculate portfolio volatility
        σ_p = sqrt(w^T * Σ * w)
        
        Parameters:
        -----------
        weights : pd.Series
            Asset weights
        cov_matrix : pd.DataFrame
            Covariance matrix (annualized)
        """
        w = weights.values
        cov = cov_matrix.loc[weights.index, weights.index].values
        
        portfolio_variance = w.T @ cov @ w
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_volatility
    
    @staticmethod
    def get_portfolio_return(weights: pd.Series, 
                            returns: pd.DataFrame) -> float:
        """
        Calculate expected portfolio return
        
        Parameters:
        -----------
        weights : pd.Series
            Asset weights
        returns : pd.DataFrame
            Asset returns
        """
        mean_returns = returns.mean() * 252  # Annualized
        portfolio_return = (weights * mean_returns).sum()
        
        return portfolio_return
    
    @staticmethod
    def display_weights(weights: pd.Series, asset_names: Dict[str, str] = None):
        """Display weights in a formatted table"""
        weights_df = pd.DataFrame({
            'Weight': weights,
            'Weight (%)': weights * 100
        }).sort_values('Weight', ascending=False)
        
        if asset_names:
            weights_df['Asset Name'] = weights_df.index.map(asset_names)
            weights_df = weights_df[['Asset Name', 'Weight', 'Weight (%)']]
        
        print("\nPortfolio Weights:")
        print("=" * 60)
        print(weights_df.to_string())
        print("=" * 60)
        print(f"Total Weight: {weights.sum():.6f}")
        
        return weights_df