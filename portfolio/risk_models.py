"""
Risk Models Module
Calculates risk contributions and portfolio risk decomposition
"""

import pandas as pd
import numpy as np
from typing import Tuple


class RiskModels:
    """Portfolio risk analysis and decomposition"""
    
    @staticmethod
    def calculate_risk_contribution(weights: pd.Series, 
                                    cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate each asset's contribution to portfolio risk
        
        Risk Contribution_i = w_i * (Σw)_i / σ_p
        
        Parameters:
        -----------
        weights : pd.Series
            Asset weights
        cov_matrix : pd.DataFrame
            Covariance matrix (annualized)
        
        Returns:
        --------
        pd.Series : Risk contribution for each asset
        """
        # Ensure alignment
        w = weights.values.reshape(-1, 1)
        cov = cov_matrix.loc[weights.index, weights.index].values
        
        # Portfolio variance: w^T * Σ * w
        portfolio_variance = (w.T @ cov @ w)[0, 0]
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Marginal contribution: Σ * w
        marginal_contrib = cov @ w
        
        # Risk contribution: w_i * (Σw)_i / σ_p
        risk_contrib = (weights.values * marginal_contrib.flatten()) / portfolio_vol
        
        risk_contrib = pd.Series(risk_contrib, index=weights.index)
        
        return risk_contrib
    
    @staticmethod
    def calculate_marginal_risk_contribution(weights: pd.Series,
                                            cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate marginal risk contribution (MRC)
        MRC_i = (Σw)_i / σ_p
        
        This represents the change in portfolio risk from a small change in weight
        """
        w = weights.values.reshape(-1, 1)
        cov = cov_matrix.loc[weights.index, weights.index].values
        
        portfolio_variance = (w.T @ cov @ w)[0, 0]
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Marginal contribution
        marginal_contrib = (cov @ w).flatten() / portfolio_vol
        
        mrc = pd.Series(marginal_contrib, index=weights.index)
        
        return mrc
    
    @staticmethod
    def calculate_percentage_risk_contribution(weights: pd.Series,
                                              cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate percentage risk contribution
        Shows what % of total portfolio risk comes from each asset
        """
        risk_contrib = RiskModels.calculate_risk_contribution(weights, cov_matrix)
        
        # Normalize to percentages
        pct_risk_contrib = risk_contrib / risk_contrib.sum() * 100
        
        return pct_risk_contrib
    
    @staticmethod
    def check_risk_parity(weights: pd.Series,
                         cov_matrix: pd.DataFrame,
                         tolerance: float = 0.01) -> Tuple[bool, pd.Series]:
        """
        Check if portfolio satisfies risk parity condition
        
        Parameters:
        -----------
        weights : pd.Series
            Asset weights
        cov_matrix : pd.DataFrame
            Covariance matrix
        tolerance : float
            Acceptable deviation from equal risk (default 1%)
        
        Returns:
        --------
        bool : Whether risk parity is satisfied
        pd.Series : Risk contribution deviations
        """
        risk_contrib = RiskModels.calculate_risk_contribution(weights, cov_matrix)
        
        # Target: equal risk contribution
        target_contrib = risk_contrib.sum() / len(weights)
        
        # Calculate deviations
        deviations = np.abs(risk_contrib - target_contrib) / target_contrib * 100
        
        # Check if all deviations within tolerance
        is_risk_parity = (deviations <= tolerance * 100).all()
        
        return is_risk_parity, deviations
    
    @staticmethod
    def diversification_ratio(weights: pd.Series,
                             returns: pd.DataFrame,
                             cov_matrix: pd.DataFrame) -> float:
        """
        Calculate portfolio diversification ratio
        DR = (Σ w_i * σ_i) / σ_p
        
        Higher values indicate better diversification
        """
        # Individual volatilities (annualized)
        individual_vols = returns.std() * np.sqrt(252)
        
        # Weighted average volatility
        weighted_vol = (weights * individual_vols).sum()
        
        # Portfolio volatility
        w = weights.values
        cov = cov_matrix.loc[weights.index, weights.index].values
        portfolio_vol = np.sqrt(w.T @ cov @ w)
        
        div_ratio = weighted_vol / portfolio_vol
        
        return div_ratio
    
    @staticmethod
    def risk_decomposition_table(weights: pd.Series,
                                 cov_matrix: pd.DataFrame,
                                 returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive risk decomposition table
        """
        # Calculate various risk metrics
        risk_contrib = RiskModels.calculate_risk_contribution(weights, cov_matrix)
        pct_risk_contrib = RiskModels.calculate_percentage_risk_contribution(weights, cov_matrix)
        mrc = RiskModels.calculate_marginal_risk_contribution(weights, cov_matrix)
        
        # Build table
        decomp_table = pd.DataFrame({
            'Weight': weights,
            'Weight (%)': weights * 100,
            'Risk Contribution': risk_contrib,
            'Risk Contrib (%)': pct_risk_contrib,
            'Marginal Risk Contrib': mrc
        })
        
        # Add individual volatility if returns provided
        if returns is not None:
            decomp_table['Volatility (Annual)'] = returns[weights.index].std() * np.sqrt(252)
        
        # Sort by risk contribution
        decomp_table = decomp_table.sort_values('Risk Contrib (%)', ascending=False)
        
        return decomp_table
    
    @staticmethod
    def display_risk_decomposition(weights: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  returns: pd.DataFrame = None):
        """Display risk decomposition in formatted table"""
        decomp_table = RiskModels.risk_decomposition_table(weights, cov_matrix, returns)
        
        print("\nRisk Decomposition Analysis:")
        print("=" * 100)
        print(decomp_table.round(4).to_string())
        print("=" * 100)
        
        # Calculate portfolio statistics
        w = weights.values
        cov = cov_matrix.loc[weights.index, weights.index].values
        portfolio_vol = np.sqrt(w.T @ cov @ w)
        
        print(f"\nPortfolio Volatility: {portfolio_vol:.4f} ({portfolio_vol*100:.2f}%)")
        
        # Check risk parity
        is_rp, deviations = RiskModels.check_risk_parity(weights, cov_matrix)
        print(f"Risk Parity Achieved: {is_rp}")
        print(f"Max Deviation from Equal Risk: {deviations.max():.2f}%")
        
        if returns is not None:
            div_ratio = RiskModels.diversification_ratio(weights, returns, cov_matrix)
            print(f"Diversification Ratio: {div_ratio:.4f}")
        
        return decomp_table