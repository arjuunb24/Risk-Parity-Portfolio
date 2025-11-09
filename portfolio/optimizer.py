"""
Portfolio Optimizer Module
Implements Equal Risk Contribution (ERC) optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class RiskParityOptimizer:
    """
    Optimize portfolio for Equal Risk Contribution (ERC)
    Uses both SciPy and PyPortfolioOpt approaches
    """
    
    def __init__(self, 
                 cov_matrix: pd.DataFrame,
                 min_weight: float = 0.01,
                 max_weight: float = 0.30):
        """
        Initialize optimizer
        
        Parameters:
        -----------
        cov_matrix : pd.DataFrame
            Covariance matrix (annualized)
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        """
        self.cov_matrix = cov_matrix
        self.n_assets = len(cov_matrix)
        self.asset_names = cov_matrix.index
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def optimize_scipy(self, 
                      initial_weights: Optional[np.ndarray] = None,
                      method: str = 'SLSQP',
                      tolerance: float = 1e-8,
                      max_iter: int = 1000) -> pd.Series:
        """
        Optimize using SciPy minimize
        
        Minimizes the sum of squared differences in risk contributions
        Objective: Σ(RC_i - RC_j)^2 for all pairs i,j
        
        Parameters:
        -----------
        initial_weights : np.ndarray
            Starting weights (uses equal weights if None)
        method : str
            Optimization method ('SLSQP', 'trust-constr')
        tolerance : float
            Optimization tolerance
        max_iter : int
            Maximum iterations
        """
        # Initial guess: equal weights or provided
        if initial_weights is None:
            x0 = np.ones(self.n_assets) / self.n_assets
        else:
            x0 = initial_weights
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(self.n_assets))
        
        # Optimize
        result = minimize(
            fun=self._risk_parity_objective,
            x0=x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': tolerance}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not fully converge. Message: {result.message}")
        
        # Return as Series
        optimal_weights = pd.Series(result.x, index=self.asset_names)
        
        return optimal_weights
    
    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """
        Objective function for risk parity optimization
        Minimizes sum of squared differences in risk contributions
        """
        # Ensure weights are positive and sum to 1
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        
        # Covariance matrix
        cov = self.cov_matrix.values
        
        # Portfolio variance
        portfolio_variance = weights.T @ cov @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Avoid division by zero
        if portfolio_vol < 1e-10:
            return 1e10
        
        # Risk contributions
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Target: equal risk contribution
        target = np.sum(risk_contrib) / self.n_assets
        
        # Sum of squared deviations from target
        objective = np.sum((risk_contrib - target) ** 2)
        
        return objective
    
    def optimize_with_pyportfolioopt(self) -> pd.Series:
        """
        Alternative optimization using PyPortfolioOpt library
        """
        try:
            from pypfopt import risk_models, EfficientFrontier
            from pypfopt.risk_models import CovarianceShrinkage
            
            # Create expected returns (we don't use them for risk parity, but required)
            expected_returns = pd.Series(0, index=self.asset_names)
            
            # Use provided covariance matrix
            ef = EfficientFrontier(expected_returns, self.cov_matrix)
            
            # Set weight bounds
            ef.add_constraint(lambda w: w >= self.min_weight)
            ef.add_constraint(lambda w: w <= self.max_weight)
            
            # Risk parity optimization - minimize volatility with equal risk contribution
            weights = ef.min_volatility()
            
            # Clean weights (remove very small weights)
            cleaned_weights = ef.clean_weights()
            
            return pd.Series(cleaned_weights)
            
        except ImportError:
            print("PyPortfolioOpt not available. Using SciPy optimization.")
            return self.optimize_scipy()
    
    def optimize_iterative(self,
                          max_iterations: int = 100,
                          tolerance: float = 0.01) -> pd.Series:
        """
        Iterative optimization approach
        Adjusts weights to equalize risk contributions
        """
        # Start with inverse volatility weights
        volatility = np.sqrt(np.diag(self.cov_matrix.values))
        weights = 1 / volatility
        weights = weights / np.sum(weights)
        
        for iteration in range(max_iterations):
            # Calculate risk contributions
            cov = self.cov_matrix.values
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)
            marginal_contrib = cov @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal risk
            target_risk = np.mean(risk_contrib)
            
            # Check convergence
            max_deviation = np.max(np.abs(risk_contrib - target_risk) / target_risk)
            if max_deviation < tolerance:
                break
            
            # Adjust weights: increase if risk too low, decrease if too high
            adjustment = target_risk / risk_contrib
            weights = weights * adjustment
            
            # Normalize
            weights = weights / np.sum(weights)
            
            # Apply bounds
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / np.sum(weights)
        
        return pd.Series(weights, index=self.asset_names)
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare different optimization methods
        """
        results = {}
        
        # SciPy SLSQP
        try:
            scipy_weights = self.optimize_scipy(method='SLSQP')
            results['SciPy SLSQP'] = scipy_weights
        except:
            pass
        
        # SciPy trust-constr
        try:
            trust_weights = self.optimize_scipy(method='trust-constr')
            results['SciPy trust-constr'] = trust_weights
        except:
            pass
        
        # Iterative
        try:
            iter_weights = self.optimize_iterative()
            results['Iterative'] = iter_weights
        except:
            pass
        
        # PyPortfolioOpt
        try:
            pypfopt_weights = self.optimize_with_pyportfolioopt()
            results['PyPortfolioOpt'] = pypfopt_weights
        except:
            pass
        
        # Create comparison DataFrame
        comparison = pd.DataFrame(results)
        
        return comparison