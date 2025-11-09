"""
Geographic Analysis Module
Analyzes portfolio composition and performance across different markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class GeographicAnalyzer:
    """Analyze portfolio from geographic perspective"""
    
    def __init__(self, asset_categories: Dict[str, List[str]]):
        """
        Initialize with asset categories
        
        Parameters:
        -----------
        asset_categories : dict
            Dictionary mapping categories to list of tickers
        """
        self.asset_categories = asset_categories
        
    def categorize_weights(self, weights: pd.Series) -> pd.DataFrame:
        """
        Categorize portfolio weights by geography/asset class
        
        Parameters:
        -----------
        weights : pd.Series
            Portfolio weights
        """
        category_weights = {}
        
        for category, tickers in self.asset_categories.items():
            # Filter for tickers that exist in weights
            valid_tickers = [t for t in tickers if t in weights.index]
            
            if valid_tickers:
                category_weights[category] = weights[valid_tickers].sum()
            else:
                category_weights[category] = 0.0
        
        return pd.Series(category_weights).sort_values(ascending=False)
    
    def analyze_geographic_exposure(self, weights: pd.Series) -> pd.DataFrame:
        """
        Analyze exposure to different geographic regions
        """
        geographic_exposure = {
            'North America': 0.0,
            'Europe': 0.0,
            'Asia Pacific': 0.0,
            'Global': 0.0,
            'Other': 0.0
        }
        
        # Map categories to regions
        region_mapping = {
            'US_Equities': 'North America',
            'Canadian_Equities': 'North America',
            'European_Equities': 'Europe',
            'Asian_Equities': 'Asia Pacific',
            'Pacific_Equities': 'Asia Pacific',
            'International_ETFs': 'Global',
            'Fixed_Income': 'Global',
            'Commodities': 'Global',
            'Crypto': 'Global',
            'Real_Estate': 'Global'
        }
        
        for category, tickers in self.asset_categories.items():
            valid_tickers = [t for t in tickers if t in weights.index]
            category_weight = weights[valid_tickers].sum()
            
            region = region_mapping.get(category, 'Other')
            geographic_exposure[region] += category_weight
        
        exposure_df = pd.DataFrame({
            'Region': list(geographic_exposure.keys()),
            'Weight': list(geographic_exposure.values()),
            'Weight (%)': [v * 100 for v in geographic_exposure.values()]
        })
        
        return exposure_df.sort_values('Weight', ascending=False)
    
    def calculate_category_performance(self,
                                      returns: pd.DataFrame,
                                      weights: pd.Series,
                                      periods_per_year: int = 252) -> pd.DataFrame:
        """
        Calculate performance metrics for each category
        """
        category_metrics = []
        
        for category, tickers in self.asset_categories.items():
            valid_tickers = [t for t in tickers if t in weights.index and t in returns.columns]
            
            if not valid_tickers:
                continue
            
            # Get category weights (normalized within category)
            cat_weights = weights[valid_tickers]
            cat_weights_norm = cat_weights / cat_weights.sum()
            
            # Calculate category returns
            cat_returns = returns[valid_tickers]
            portfolio_returns = (cat_returns * cat_weights_norm).sum(axis=1)
            
            # Calculate metrics
            annual_return = portfolio_returns.mean() * periods_per_year
            volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            category_metrics.append({
                'Category': category,
                'Weight': cat_weights.sum(),
                'Annual Return': annual_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe,
                'N Assets': len(valid_tickers)
            })
        
        metrics_df = pd.DataFrame(category_metrics)
        metrics_df['Weight (%)'] = metrics_df['Weight'] * 100
        
        return metrics_df.sort_values('Weight', ascending=False)
    
    def display_category_analysis(self, 
                                  weights: pd.Series,
                                  returns: pd.DataFrame = None):
        """Display comprehensive category analysis"""
        print("\n" + "="*80)
        print("GEOGRAPHIC & CATEGORY ANALYSIS")
        print("="*80)
        
        # 1. Category weights
        category_weights = self.categorize_weights(weights)
        print("\nPortfolio Composition by Category:")
        print("-" * 60)
        for cat, weight in category_weights.items():
            print(f"  {cat:30s}: {weight:6.2%}")
        print("-" * 60)
        print(f"  {'Total':30s}: {category_weights.sum():6.2%}")
        
        # 2. Geographic exposure
        print("\nGeographic Exposure:")
        print("-" * 60)
        geographic_exposure = self.analyze_geographic_exposure(weights)
        for _, row in geographic_exposure.iterrows():
            if row['Weight'] > 0.001:  # Show only significant exposures
                print(f"  {row['Region']:20s}: {row['Weight (%)']:6.2f}%")
        
        # 3. Category performance (if returns provided)
        if returns is not None:
            print("\nCategory Performance Metrics:")
            print("-" * 80)
            cat_perf = self.calculate_category_performance(returns, weights)
            print(cat_perf.to_string(index=False))
        
        return category_weights, geographic_exposure
    
    def calculate_home_bias(self, weights: pd.Series, home_region: str = 'North America') -> float:
        """
        Calculate home bias - overweight vs global market cap weights
        
        Parameters:
        -----------
        weights : pd.Series
            Portfolio weights
        home_region : str
            Home region of the investor
        """
        geographic_exposure = self.analyze_geographic_exposure(weights)
        
        home_weight = geographic_exposure[
            geographic_exposure['Region'] == home_region
        ]['Weight'].sum()
        
        # Typical global market cap weights (approximate)
        global_weights = {
            'North America': 0.60,  # US + Canada
            'Europe': 0.15,
            'Asia Pacific': 0.20,
            'Global': 0.05
        }
        
        expected_weight = global_weights.get(home_region, 0.5)
        home_bias = home_weight - expected_weight
        
        print(f"\nHome Bias Analysis ({home_region}):")
        print(f"  Portfolio Weight: {home_weight:.2%}")
        print(f"  Global Market Weight: {expected_weight:.2%}")
        print(f"  Home Bias: {home_bias:+.2%}")
        
        return home_bias
    
    def calculate_diversification_score(self, weights: pd.Series) -> Dict[str, float]:
        """
        Calculate diversification scores across multiple dimensions
        """
        # 1. Category diversification (Herfindahl index)
        category_weights = self.categorize_weights(weights)
        category_hhi = (category_weights ** 2).sum()
        category_div_score = 1 - category_hhi
        
        # 2. Geographic diversification
        geographic_exposure = self.analyze_geographic_exposure(weights)
        geo_weights = geographic_exposure['Weight']
        geo_hhi = (geo_weights ** 2).sum()
        geo_div_score = 1 - geo_hhi
        
        # 3. Asset-level diversification
        asset_hhi = (weights ** 2).sum()
        asset_div_score = 1 - asset_hhi
        
        scores = {
            'Category Diversification': category_div_score,
            'Geographic Diversification': geo_div_score,
            'Asset Diversification': asset_div_score,
            'Overall Score': (category_div_score + geo_div_score + asset_div_score) / 3
        }
        
        print("\nDiversification Scores (0 = concentrated, 1 = highly diversified):")
        print("-" * 60)
        for metric, score in scores.items():
            print(f"  {metric:30s}: {score:.4f}")
        
        return scores