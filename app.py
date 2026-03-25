import streamlit as st
import os
import pandas as pd
from main import main
from config import OUTPUT_DIR, INITIAL_CAPITAL

# --- COMPATIBILITY HELPERS ---
def st_rerun():
    """Fallback for older Streamlit versions that use experimental_rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def sanitize_df(df):
    """Fix 'LargeUtf8' error by casting object columns to standard strings."""
    if df is None: return None
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    return df

# --- QUANTSTATS COMPATIBILITY PATCH (for Pandas 2.x) ---
try:
    import pandas as pd
    # Force 'M' to be interpreted correctly or handle the resample freq change.
    # This specifically fixes the ValueError: to_offset(freq) in QuantStats.
    if pd.__version__ >= "2.0.0":
        import pandas.tseries.offsets as offsets
        if not hasattr(offsets, 'M'):
            offsets.M = offsets.MonthEnd
except Exception:
    pass

# Page configuration
st.set_page_config(
    page_title="Risk Parity Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FULLSCREEN IMAGE VIEWER ROUTING (session-state based, no page reload) ---
if st.session_state.get('selected_image'):
    img_file = st.session_state['selected_image']
    img_path = os.path.join(OUTPUT_DIR, img_file)

    # Disable page scroll entirely while in viewer
    st.markdown("""
    <style>
        [data-testid="stSidebar"], header, footer { display: none !important; }
        .block-container { padding-top: 1rem !important; max-width: 100% !important; overflow: hidden !important; }
        html, body, [data-testid="stAppViewContainer"] { overflow: hidden !important; height: 100vh !important; }
    </style>
    """, unsafe_allow_html=True)

    if os.path.exists(img_path):
        import base64
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div style="display:flex; justify-content:center; align-items:center; width:100%;">
            <img src="data:image/png;base64,{encoded}"
                 style="max-height:72vh; max-width:95vw; border-radius:8px; object-fit:contain;">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Image not found.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("⬅️ Back to Portfolio Visualizations", type="primary", use_container_width=True):
            st.session_state['selected_image'] = None
            st_rerun()
    st.stop()


# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 14px;
        color: #fafafa;
    }
    .section-header {
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    
    /* Force high-contrast text for st.text output logs */
    div[data-testid="stText"] {
        color: #fafafa !important;
        font-family: monospace !important;
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Risk Parity Portfolio Dashboard")
st.markdown("Automated asset allocation and backtesting using Equal Risk Contribution principles.")

st.sidebar.header("Controls")

# Initialization of session state for results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

if st.sidebar.button("Run Portfolio Generation", type="primary"):
    with st.spinner("Fetching data, optimizing weights, and running backtests... This may take a moment."):
        # Run the main pipeline (with run_streamlit=True to avoid opening browser tabs automatically)
        st.session_state.results = main(run_streamlit=True)

if st.session_state.results is not None:
    results = st.session_state.results
    
    st.sidebar.markdown("""
<a href="#portfolio-visualizations" target="_self" style="text-decoration: none;">
    <div style="background-color: #262730; padding: 10px; border-radius: 5px; text-align: center; color: #00d4ff; font-weight: bold; border: 1px solid #00d4ff; margin-top: 15px;">
        ⏬ Jump to Visualizations
    </div>
</a>
""", unsafe_allow_html=True)
    
    # Extract results
    rp_results = results['rp_results']
    ew_results = results['ew_results']
    benchmark_results_data = results.get('benchmark_results')
    weights_schedule = results['weights_schedule']
    comparison = results['comparison']
    
    # Additional info added in our modification of main.py
    asset_stats = results.get('asset_stats', pd.DataFrame())
    valid_tickers = results.get('valid_tickers', [])
    data_summary = results.get('data_summary', {})
    
    # Layout: Step 1 & 2
    st.markdown("<h2 class='section-header'>Data Acquisition & Preprocessing</h2>", unsafe_allow_html=True)
    st.text(f"Data Summary:\n  Assets: {data_summary.get('n_assets')}\n  Date Range: {data_summary.get('start_date')} to {data_summary.get('end_date')}\n  Trading Days: {data_summary.get('n_periods')}")
    st.text("Asset Statistics:")
    st.dataframe(sanitize_df(asset_stats), use_container_width=True)
    
    # Step 3: Initial Weights
    st.markdown("<h2 class='section-header'>Initial Portfolio Weighting</h2>", unsafe_allow_html=True)
    st.text("Inverse Volatility Weights:")
    inv_weights = results.get('inv_vol_weights')
    if inv_weights is not None:
        st.dataframe(sanitize_df(pd.DataFrame(inv_weights).T), use_container_width=True)
        
    # Step 4: Optimization
    st.markdown("<h2 class='section-header'>Risk Parity Optimization</h2>", unsafe_allow_html=True)
    opt_weights = results.get('optimal_weights')
    if opt_weights is not None:
        st.text("Optimal Risk Parity Weights:")
        st.dataframe(sanitize_df(pd.DataFrame(opt_weights).T), use_container_width=True)
        
    pref = results.get('portfolio_ret', 0) * 100
    pvol = results.get('portfolio_vol', 0) * 100
    psharpe = results.get('portfolio_ret', 0) / max(results.get('portfolio_vol', 0.0001), 0.0001)
    st.text(f"Portfolio Statistics:\n  Expected Annual Return: {pref:.2f}%\n  Annual Volatility: {pvol:.2f}%\n  Sharpe Ratio: {psharpe:.3f}")
        
    # Step 5: Risk Decomposition
    st.markdown("<h2 class='section-header'>Risk Decomposition Analysis</h2>", unsafe_allow_html=True)
    risk_decomp_df = results.get('risk_decomp')
    if risk_decomp_df is not None:
        st.text("Risk Decomposition Analysis:")
        st.dataframe(sanitize_df(risk_decomp_df), use_container_width=True)
        
    devs = results.get('deviations', pd.Series([0])).max()
    st.text(f"Portfolio Volatility: {results.get('portfolio_vol', 0):.4f} ({pvol:.2f}%)\nRisk Parity Achieved: {results.get('is_rp', 'Unknown')}\nMax Deviation from Equal Risk: {devs:.2f}%\nDiversification Ratio: {results.get('div_ratio', 0):.4f}")
        
    # Step 6: Dynamic Rebalancing
    st.markdown("<h2 class='section-header'>Dynamic Rebalancing</h2>", unsafe_allow_html=True)
    rebalance_dates = weights_schedule.index.unique() if weights_schedule is not None else []
    if len(rebalance_dates) > 0:
        st.text(f"Rebalancing Schedule:\n  Total Rebalances: {len(rebalance_dates)}\n  First Rebalance: {rebalance_dates[0]}\n  Last Rebalance: {rebalance_dates[-1]}")
    st.text("Weight Stability Analysis:")
    st.dataframe(sanitize_df(results.get('weight_stability')), use_container_width=True)
    
    # Step 7 & 8: Performance & Comparison
    st.markdown("<h2 class='section-header'>Performance Evaluation</h2>", unsafe_allow_html=True)
    from performance.metrics import PerformanceMetrics
    from config import RISK_FREE_RATE
    
    strategies = {
        'Risk Parity': rp_results,
        'Equal Weight': ew_results
    }
    if benchmark_results_data:
        from config import BENCHMARK_TICKER
        strategies[BENCHMARK_TICKER] = benchmark_results_data
        
    for name, res in strategies.items():
        st.text(f"\n{name} Portfolio Performance:\n")
        report = PerformanceMetrics.generate_performance_report(
            portfolio_returns=res['portfolio_returns'],
            portfolio_values=res['portfolio_values'],
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate=RISK_FREE_RATE
        )
        st.dataframe(sanitize_df(report), use_container_width=True)
    
    st.markdown("<h2 class='section-header'>Strategy Comparison</h2>", unsafe_allow_html=True)
    st.text(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    st.dataframe(sanitize_df(comparison.round(4)), use_container_width=True)
    
    st.text(f"\nFinal Results:\n  Risk Parity Total Return: {rp_results.get('total_return', 0)*100:.2f}%\n  Equal Weight Total Return: {ew_results.get('total_return', 0)*100:.2f}%")
    if benchmark_results_data:
        from config import BENCHMARK_TICKER
        st.text(f"  {BENCHMARK_TICKER} Total Return: {benchmark_results_data.get('total_return', 0)*100:.2f}%")
    
    st.text(f"\n  Total Transaction Costs (RP): ${rp_results.get('total_transaction_costs', 0):,.2f}\n  Final Portfolio Value (RP): ${rp_results.get('final_value', 0):,.2f}")
    
    # Visualize Generated Plots
    st.markdown("<h2 id='portfolio-visualizations' class='section-header'>Portfolio Visualizations</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reordered image files as requested
    image_files = [
        ("weights_over_time.png", "Weights Over Time"),
        ("cumulative_returns.png", "Cumulative Returns"),
        ("risk_contribution.png", "Risk Contribution"),
        ("rolling_volatility.png", "Rolling 60-Day Volatility"),
        ("drawdown.png", "Portfolio Drawdown"),
        ("return_distribution.png", "Return Distribution"),
        ("correlation_heatmap.png", "Asset Correlation Matrix"),
        ("weight_distribution.png", "Final Weight Distribution")
    ]
    
    import base64
    
    def render_clickable_image(col, img_filename, caption):
        """Display image as a clickable tile using session_state (no page reload)."""
        img_path = os.path.join(OUTPUT_DIR, img_filename)
        if not os.path.exists(img_path):
            return
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # Show the image via HTML (display only, not interactive)
        col.markdown(f'''
        <img src="data:image/png;base64,{encoded_string}" style="width: 100%; border-radius: 5px;">
        <div style="text-align: center; color: #a0a0a0; font-size: 14px; margin-top: 4px;">{caption}</div>
        ''', unsafe_allow_html=True)
        # Invisible Streamlit button overlaid below image to capture click
        if col.button(f"\U0001f50d View: {caption}", key=img_filename, use_container_width=True):
            st.session_state['selected_image'] = img_filename
            st_rerun()
    
    for i in range(0, len(image_files), 2):
        icol1, icol2 = st.columns(2)
        
        f1, t1 = image_files[i]
        if os.path.exists(os.path.join(OUTPUT_DIR, f1)):
            render_clickable_image(icol1, f1, t1)
            
        if i + 1 < len(image_files):
            f2, t2 = image_files[i+1]
            if os.path.exists(os.path.join(OUTPUT_DIR, f2)):
                render_clickable_image(icol2, f2, t2)
                
    # Optional Links
    st.markdown("<h2 class='section-header'>Detailed Reports</h2>", unsafe_allow_html=True)
    
    # Add a button to natively render QuantStats within Streamlit
    qs_report = os.path.realpath(os.path.join(OUTPUT_DIR, "quantstats_report.html"))
    if os.path.exists(qs_report):
        with st.expander("📊 View Detailed QuantStats HTML Report", expanded=True):
            import streamlit.components.v1 as components
            with open(qs_report, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            # Inject custom CSS to fix light-theme text color clashes and reduce image size
            dark_mode_css = """
            <style>
                body {
                    background-color: #0e1117 !important;
                    color: #fafafa !important;
                }
                /* Only target text color - do NOT modify layout/width/display of elements */
                div, span, p, h1, h2, h3, h4, h5, h6, th, td, li, a {
                    color: #fafafa !important;
                }
                /* Scale images slightly smaller but keep inline so side-by-side works */
                img {
                    max-width: 90% !important;
                    height: auto !important;
                    background-color: #f0f0f0;
                    padding: 6px;
                    border-radius: 6px;
                }
                /* Fix table cell backgrounds without touching widths or layout */
                table {
                    background-color: #1e2030 !important;
                }
                th {
                    background-color: #1e1e1e !important;
                    color: #00d4ff !important;
                    padding: 10px;
                }
                td {
                    padding: 10px;
                    border-bottom: 1px solid #444 !important;
                }
            </style>
            """
            if "</head>" in html_content:
                html_content = html_content.replace('</head>', f'{dark_mode_css}</head>')
            else:
                html_content = dark_mode_css + html_content
                
            # Embed the HTML natively in an iframe
            components.html(html_content, height=1000, scrolling=True)
    else:
        st.info("QuantStats report not found. Make sure QuantStats is installed and ran successfully.")

else:
    st.info("👈 Click **Run Portfolio Generation** in the sidebar to start.")
    
    st.markdown("<br><h3 style='color: #00d4ff; text-align: center;'>Current Portfolio Assets</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0a0; margin-bottom: 30px;'>(The universe of assets can be modified in the config.py file)</p>", unsafe_allow_html=True)
    
    from config import ASSETS
    
    # Create a 3 column grid to display current assets spanning the whole screen breadth
    cols = st.columns(3)
    
    for i, (ticker, name) in enumerate(ASSETS.items()):
        col = cols[i % 3]
        with col:
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 8px; margin-bottom: 15px; text-align: center; border: 1px solid #333;">
                <div style="font-size: 20px; font-weight: bold; color: #fafafa;">{ticker}</div>
                <div style="font-size: 14px; color: #00d4ff; margin-top: 5px;">{name}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Project Features Section
    st.markdown("<hr style='border: 1px dashed #333; opacity: 0.5; margin-top: 5px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00d4ff; text-align: center; margin-top: 0; margin-bottom: 25px;'>Features</h3>", unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 25px; border-radius: 12px; border: 1px solid #333; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h4 style="color: #00d4ff; margin-top: 0; text-align: center; border-bottom: 1px solid #00d4ff; padding-bottom: 10px;">Techniques</h4>
            <ul style="color: #fafafa; padding-left: 15px; margin-top: 15px; font-size: 14px; line-height: 1.6;">
                <li>Equal Risk Contribution Optimization</li>
                <li>Robust Covariance Matrix Estimation</li>
                <li>Quadratic Programming Framework</li>
                <li>Inverse Volatility Scaling</li>
                <li>Transaction Cost Modeling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with feat_col2:
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 25px; border-radius: 12px; border: 1px solid #333; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h4 style="color: #00d4ff; margin-top: 0; text-align: center; border-bottom: 1px solid #00d4ff; padding-bottom: 10px;">Strategies</h4>
            <ul style="color: #fafafa; padding-left: 15px; margin-top: 15px; font-size: 14px; line-height: 1.6;">
                <li>Risk Parity Allocation</li>
                <li>Equal Weight Baseline Comparison</li>
                <li>S&P 500 Benchmark Tracking</li>
                <li>Dynamic Quarterly Rebalancing</li>
                <li>Rolling Window Backtesting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with feat_col3:
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 25px; border-radius: 12px; border: 1px solid #333; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h4 style="color: #00d4ff; margin-top: 0; text-align: center; border-bottom: 1px solid #00d4ff; padding-bottom: 10px;">Visualizations</h4>
            <ul style="color: #fafafa; padding-left: 15px; margin-top: 15px; font-size: 14px; line-height: 1.6;">
                <li>Cumulative Return Analysis</li>
                <li>Dynamic Asset Weighting Charts</li>
                <li>Risk Contribution Breakdown</li>
                <li>Drawdown and Volatility Profiles</li>
                <li>Correlation and Return Heatmaps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

