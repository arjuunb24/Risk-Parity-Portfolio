"""
Main Execution Script for Risk Parity Portfolio Project
Orchestrates the entire workflow from data acquisition to visualization
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import *
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from portfolio.weights import PortfolioWeights
from portfolio.risk_models import RiskModels
from portfolio.optimizer import RiskParityOptimizer
from backtesting.engine import BacktestEngine
from backtesting.rebalancer import DynamicRebalancer
from performance.metrics import PerformanceMetrics
from performance.geographic_analysis import GeographicAnalyzer
from visualization.plots import PortfolioVisualizer


def main():
    """Main execution function"""
    
    print("="*80)
    print("RISK PARITY PORTFOLIO OPTIMIZATION PROJECT")
    print("="*80)
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # ========================================================================
    # STEP 1: DATA ACQUISITION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA ACQUISITION")
    print("="*80)
    
    fetcher = DataFetcher(ASSETS, START_DATE, END_DATE)
    prices = fetcher.fetch_prices()
    
    # Get valid tickers
    valid_tickers = fetcher.get_valid_tickers()
    print(f"\nValid assets: {len(valid_tickers)}")
    
    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = DataPreprocessor(prices)
    returns = preprocessor.calculate_returns(method='simple')
    
    # Display data summary
    summary = preprocessor.get_data_summary()
    print(f"\nData Summary:")
    print(f"  Assets: {summary['n_assets']}")
    print(f"  Periods: {summary['n_periods']}")
    print(f"  Date Range: {summary['start_date']} to {summary['end_date']}")
    
    # Calculate statistics
    asset_stats = preprocessor.get_asset_statistics()
    print("\nAsset Statistics:")
    print(asset_stats)
    
    # Calculate covariance and correlation matrices
    cov_matrix = preprocessor.calculate_covariance_matrix()
    corr_matrix = preprocessor.calculate_correlation_matrix()
    
    # ========================================================================
    # STEP 3: INITIAL PORTFOLIO WEIGHTING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: INITIAL PORTFOLIO WEIGHTING")
    print("="*80)
    
    # Calculate inverse volatility weights
    inv_vol_weights = PortfolioWeights.inverse_volatility_weights(
        returns, min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT
    )
    
    print("\nInverse Volatility Weights:")
    PortfolioWeights.display_weights(inv_vol_weights, ASSETS)
    
    # ========================================================================
    # STEP 4: RISK PARITY OPTIMIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: RISK PARITY OPTIMIZATION")
    print("="*80)
    
    optimizer = RiskParityOptimizer(
        cov_matrix=cov_matrix,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT
    )
    
    # Optimize using SciPy
    optimal_weights = optimizer.optimize_scipy(
        initial_weights=inv_vol_weights.values,
        method='SLSQP',
        tolerance=OPTIMIZATION_TOLERANCE,
        max_iter=MAX_ITERATIONS
    )
    
    print("\nOptimal Risk Parity Weights:")
    PortfolioWeights.display_weights(optimal_weights, ASSETS)
    
    # Calculate portfolio metrics
    portfolio_vol = PortfolioWeights.get_portfolio_volatility(optimal_weights, cov_matrix)
    portfolio_ret = PortfolioWeights.get_portfolio_return(optimal_weights, returns)
    
    print(f"\nPortfolio Statistics:")
    print(f"  Expected Annual Return: {portfolio_ret*100:.2f}%")
    print(f"  Annual Volatility: {portfolio_vol*100:.2f}%")
    print(f"  Sharpe Ratio: {portfolio_ret/portfolio_vol:.3f}")
    
    # ========================================================================
    # STEP 5: RISK DECOMPOSITION ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: RISK DECOMPOSITION ANALYSIS")
    print("="*80)
    
    risk_decomp = RiskModels.display_risk_decomposition(
        optimal_weights, cov_matrix, returns
    )
    
    # Check risk parity
    is_rp, deviations = RiskModels.check_risk_parity(optimal_weights, cov_matrix)
    print(f"\nRisk Parity Achieved: {is_rp}")
    
    # ========================================================================
    # STEP 6: DYNAMIC REBALANCING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: DYNAMIC REBALANCING")
    print("="*80)
    
    rebalancer = DynamicRebalancer(
        returns=returns,
        rolling_window=ROLLING_WINDOW,
        rebalance_freq=REBALANCING_FREQUENCY,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT
    )
    
    # Generate rebalance schedule
    weights_schedule, rebalance_dates = rebalancer.generate_rebalance_schedule()
    
    print(f"\nRebalancing Schedule:")
    print(f"  Total Rebalances: {len(rebalance_dates)}")
    print(f"  First Rebalance: {rebalance_dates[0]}")
    print(f"  Last Rebalance: {rebalance_dates[-1]}")
    
    # Analyze weight stability
    stability = rebalancer.analyze_weight_stability(weights_schedule)
    print("\nWeight Stability Analysis:")
    print(stability)
    
    # ========================================================================
    # STEP 7: BACKTESTING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: BACKTESTING")
    print("="*80)
    
    backtest_engine = BacktestEngine(
        prices=prices,
        returns=returns,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=TRANSACTION_COST_BPS
    )
    
    # Run Risk Parity backtest
    print("\nRunning Risk Parity Portfolio backtest...")
    rp_results = backtest_engine.run_backtest(weights_schedule, rebalance_dates)
    
    # Run Equal Weight benchmark
    print("Running Equal Weight benchmark...")
    equal_weights = PortfolioWeights.equal_weights(returns)
    ew_results = backtest_engine.run_benchmark_backtest(
        equal_weights, 
        rebalance_freq=REBALANCING_FREQUENCY
    )
    
    # Run benchmark (S&P 500)
    if BENCHMARK_TICKER in returns.columns:
        print(f"Running {BENCHMARK_TICKER} benchmark...")
        benchmark_weights = pd.Series(0, index=returns.columns)
        benchmark_weights[BENCHMARK_TICKER] = 1.0
        benchmark_results = backtest_engine.run_benchmark_backtest(
            benchmark_weights,
            rebalance_freq='Y'  # Annual rebalance (or hold)
        )
    else:
        benchmark_results = None
    
    # ========================================================================
    # STEP 8: PERFORMANCE EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: PERFORMANCE EVALUATION")
    print("="*80)
    
    # Generate performance reports
    strategies = {
        'Risk Parity': rp_results,
        'Equal Weight': ew_results
    }
    
    if benchmark_results:
        strategies[BENCHMARK_TICKER] = benchmark_results
    
    # Calculate metrics for each strategy
    for name, results in strategies.items():
        print(f"\n{name} Portfolio Performance:")
        print("-" * 60)
        
        report = PerformanceMetrics.generate_performance_report(
            portfolio_returns=results['portfolio_returns'],
            portfolio_values=results['portfolio_values'],
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate=RISK_FREE_RATE
        )
        
        print(report)
    
    # Compare strategies
    comparison = backtest_engine.compare_strategies(strategies)
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(comparison.round(4))
    
    # ========================================================================
    # STEP 9: VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 9: VISUALIZATION")
    print("="*80)
    
    visualizer = PortfolioVisualizer(save_plots=SAVE_PLOTS, output_dir=OUTPUT_DIR)
    
    # 1. Portfolio Weights Over Time
    print("\nGenerating weights over time plot...")
    visualizer.plot_weights_over_time(
        rp_results['weights_history'],
        title="Risk Parity Portfolio - Weights Over Time"
    )
    
    # 2. Cumulative Returns Comparison
    print("Generating cumulative returns plot...")
    visualizer.plot_cumulative_returns(
        strategies,
        title="Cumulative Returns: Risk Parity vs Benchmarks"
    )
    
    # 3. Risk Contribution
    print("Generating risk contribution plot...")
    visualizer.plot_risk_contribution(
        risk_decomp,
        title="Risk Contribution by Asset (Equal Risk Contribution)"
    )
    
    # 4. Correlation Heatmap
    print("Generating correlation heatmap...")
    visualizer.plot_correlation_heatmap(
        corr_matrix,
        title="Asset Correlation Matrix"
    )
    
    # 5. Rolling Volatility
    print("Generating rolling volatility plot...")
    visualizer.plot_rolling_volatility(
        strategies,
        window=60,
        title="Rolling 60-Day Volatility"
    )
    
    # 6. Drawdown Analysis
    print("Generating drawdown plot...")
    visualizer.plot_drawdown(
        strategies,
        title="Portfolio Drawdown Analysis"
    )
    
    # 7. Weight Distribution (Final Weights)
    print("Generating weight distribution plot...")
    final_weights = weights_schedule.iloc[-1]
    visualizer.plot_weight_distribution(
        final_weights,
        title="Final Portfolio Weight Distribution"
    )
    
    # 8. Interactive Dashboard
    if GENERATE_HTML_REPORT:
        print("Generating interactive dashboard...")
        visualizer.create_interactive_dashboard(
            strategies,
            rp_results['weights_history'],
            risk_decomp
        )
    
    # ========================================================================
    # STEP 10: QUANTSTATS REPORT (Optional)
    # ========================================================================
    try:
        import quantstats as qs
        
        print("\n" + "="*80)
        print("STEP 10: QUANTSTATS REPORT")
        print("="*80)
        
        # Prepare returns
        rp_returns = rp_results['portfolio_returns']['return']
        
        # Generate full report
        print("\nGenerating QuantStats HTML report...")
        qs.reports.html(
            rp_returns,
            output=f'{OUTPUT_DIR}quantstats_report.html',
            title='Risk Parity Portfolio'
        )
        print(f"Report saved to {OUTPUT_DIR}quantstats_report.html")
        
    except ImportError:
        print("\nQuantStats not installed. Skipping detailed report generation.")
        print("Install with: pip install quantstats")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PROJECT COMPLETE!")
    print("="*80)
    
    print(f"\nFinal Results:")
    print(f"  Risk Parity Total Return: {rp_results['total_return']*100:.2f}%")
    print(f"  Equal Weight Total Return: {ew_results['total_return']*100:.2f}%")
    if benchmark_results:
        print(f"  {BENCHMARK_TICKER} Total Return: {benchmark_results['total_return']*100:.2f}%")
    
    print(f"\n  Total Transaction Costs (RP): ${rp_results['total_transaction_costs']:,.2f}")
    print(f"  Final Portfolio Value (RP): ${rp_results['final_value']:,.2f}")
    
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    
    return {
        'rp_results': rp_results,
        'ew_results': ew_results,
        'benchmark_results': benchmark_results,
        'weights_schedule': weights_schedule,
        'comparison': comparison
    }


if __name__ == "__main__":
    results = main()