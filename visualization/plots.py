"""
Visualization Module
Creates comprehensive portfolio visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class PortfolioVisualizer:
    """Create visualizations for portfolio analysis"""
    
    def __init__(self, save_plots: bool = False, output_dir: str = 'output/'):
        self.save_plots = save_plots
        self.output_dir = output_dir
        
    def plot_weights_over_time(self, weights_history: pd.DataFrame,
                               title: str = "Portfolio Weights Over Time"):
        """Plot stacked area chart of portfolio weights"""
        # Extract weights from history
        weights_data = []
        for _, row in weights_history.iterrows():
            weights_dict = row['weights']
            weights_dict['date'] = row['date']
            weights_data.append(weights_dict)
        
        weights_df = pd.DataFrame(weights_data).set_index('date')
        
        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(14, 8))
        weights_df.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        ax.set_ylim([0, 1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}weights_over_time.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_cumulative_returns(self, 
                               strategies_results: Dict[str, Dict],
                               title: str = "Cumulative Returns Comparison"):
        """Plot cumulative returns for multiple strategies"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for name, results in strategies_results.items():
            portfolio_values = results['portfolio_values']
            cumulative_returns = portfolio_values['cumulative_return'] * 100
            
            ax.plot(cumulative_returns.index, cumulative_returns, 
                   label=name, linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}cumulative_returns.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_risk_contribution(self, risk_decomp: pd.DataFrame,
                              title: str = "Risk Contribution by Asset"):
        """Plot risk contribution bar chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        risk_pct = risk_decomp['Risk Contrib (%)'].sort_values(ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(risk_pct)))
        risk_pct.plot(kind='barh', ax=ax, color=colors)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Risk Contribution (%)', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(risk_pct.values):
            ax.text(v + 0.2, i, f'{v:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}risk_contribution.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame,
                                title: str = "Asset Correlation Matrix"):
        """Plot correlation heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}correlation_heatmap.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_rolling_volatility(self, strategies_results: Dict[str, Dict],
                               window: int = 60,
                               title: str = "Rolling Volatility"):
        """Plot rolling volatility over time"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for name, results in strategies_results.items():
            returns = results['portfolio_returns']['return']
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            
            ax.plot(rolling_vol.index, rolling_vol, label=name, linewidth=2, alpha=0.7)
        
        ax.set_title(f'{title} ({window}-day window)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}rolling_volatility.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_drawdown(self, strategies_results: Dict[str, Dict],
                     title: str = "Drawdown Analysis"):
        """Plot drawdown over time"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for name, results in strategies_results.items():
            portfolio_values = results['portfolio_values']
            values = portfolio_values['value']
            
            # Calculate drawdown
            running_max = values.expanding().max()
            drawdown = (values - running_max) / running_max * 100
            
            ax.plot(drawdown.index, drawdown, label=name, linewidth=2, alpha=0.7)
        
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.2)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}drawdown.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_weight_distribution(self, weights: pd.Series,
                                title: str = "Portfolio Weight Distribution"):
        """Plot pie chart of portfolio weights"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort weights
        weights_sorted = weights.sort_values(ascending=False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights_sorted)))
        
        wedges, texts, autotexts = ax.pie(weights_sorted, labels=weights_sorted.index,
                                          autopct='%1.1f%%', colors=colors,
                                          startangle=90, pctdistance=0.85)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}weight_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_geographic_exposure(self, geographic_exposure: pd.DataFrame,
                                title: str = "Geographic Exposure"):
        """Plot geographic distribution as pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter out zero weights
        data = geographic_exposure[geographic_exposure['Weight'] > 0.001]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        
        wedges, texts, autotexts = ax.pie(data['Weight'], 
                                          labels=data['Region'],
                                          autopct='%1.1f%%', 
                                          colors=colors,
                                          startangle=90, 
                                          pctdistance=0.85)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}geographic_exposure.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_category_weights(self, category_weights: pd.Series,
                             title: str = "Portfolio Composition by Category"):
        """Plot category weights as horizontal bar chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort and plot
        cat_sorted = category_weights.sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(cat_sorted)))
        
        bars = ax.barh(range(len(cat_sorted)), cat_sorted * 100, color=colors)
        ax.set_yticks(range(len(cat_sorted)))
        ax.set_yticklabels(cat_sorted.index)
        
        ax.set_xlabel('Weight (%)', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, val) in enumerate(cat_sorted.items()):
            ax.text(val * 100 + 0.5, i, f'{val*100:.1f}%', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}category_weights.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_category_performance(self, category_perf: pd.DataFrame,
                                 title: str = "Category Performance Comparison"):
        """Plot scatter of return vs risk for each category"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        scatter = ax.scatter(category_perf['Volatility'] * 100,
                           category_perf['Annual Return'] * 100,
                           s=category_perf['Weight (%)'] * 20,  # Size by weight
                           c=category_perf['Sharpe Ratio'],
                           cmap='RdYlGn',
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=1.5)
        
        # Add labels
        for _, row in category_perf.iterrows():
            ax.annotate(row['Category'], 
                       xy=(row['Volatility'] * 100, row['Annual Return'] * 100),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9,
                       alpha=0.8)
        
        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Annual Return (%)', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=11)
        
        # Legend for size
        legend_sizes = [1, 5, 10]
        legend_handles = [plt.scatter([], [], s=size*20, c='gray', alpha=0.6, edgecolors='black') 
                         for size in legend_sizes]
        legend_labels = [f'{size}%' for size in legend_sizes]
        ax.legend(legend_handles, legend_labels, 
                 title='Portfolio Weight', 
                 loc='upper left',
                 framealpha=0.9)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}category_performance.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, 
                                    strategies_results: Dict[str, Dict],
                                    weights_history: pd.DataFrame,
                                    risk_decomp: pd.DataFrame):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cumulative Returns', 'Portfolio Weights Over Time',
                          'Risk Contribution', 'Rolling Volatility',
                          'Drawdown', 'Return Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                  [{"type": "bar"}, {"type": "scatter"}],
                  [{"type": "scatter"}, {"type": "histogram"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set2
        
        # 1. Cumulative Returns
        for i, (name, results) in enumerate(strategies_results.items()):
            portfolio_values = results['portfolio_values']
            cumulative_returns = portfolio_values['cumulative_return'] * 100
            
            fig.add_trace(
                go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                          name=name, line=dict(color=colors[i % len(colors)], width=2)),
                row=1, col=1
            )
        
        # 2. Portfolio Weights Over Time (stacked area)
        weights_data = []
        for _, row in weights_history.iterrows():
            weights_dict = row['weights']
            weights_dict['date'] = row['date']
            weights_data.append(weights_dict)
        
        weights_df = pd.DataFrame(weights_data).set_index('date')
        
        for i, col in enumerate(weights_df.columns):
            fig.add_trace(
                go.Scatter(x=weights_df.index, y=weights_df[col] * 100,
                          name=col, stackgroup='one',
                          line=dict(color=colors[i % len(colors)])),
                row=1, col=2
            )
        
        # 3. Risk Contribution
        risk_pct = risk_decomp['Risk Contrib (%)'].sort_values()
        fig.add_trace(
            go.Bar(y=risk_pct.index, x=risk_pct.values,
                  orientation='h', marker_color=colors[0]),
            row=2, col=1
        )
        
        # 4. Rolling Volatility
        for i, (name, results) in enumerate(strategies_results.items()):
            returns = results['portfolio_returns']['return']
            rolling_vol = returns.rolling(window=60).std() * np.sqrt(252) * 100
            
            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol,
                          name=name, line=dict(color=colors[i % len(colors)], width=2)),
                row=2, col=2
            )
        
        # 5. Drawdown
        for i, (name, results) in enumerate(strategies_results.items()):
            portfolio_values = results['portfolio_values']
            values = portfolio_values['value']
            running_max = values.expanding().max()
            drawdown = (values - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown,
                          name=name, fill='tozeroy',
                          line=dict(color=colors[i % len(colors)])),
                row=3, col=1
            )
        
        # 6. Return Distribution
        for i, (name, results) in enumerate(strategies_results.items()):
            returns = results['portfolio_returns']['return'] * 100
            
            fig.add_trace(
                go.Histogram(x=returns, name=name,
                            marker_color=colors[i % len(colors)], opacity=0.7),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1400,
            showlegend=True,
            title_text="Risk Parity Portfolio - Interactive Dashboard",
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Weight (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Risk Contribution (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Return (%)", row=3, col=2)
        fig.update_yaxes(title_text="Frequency", row=3, col=2)
        
        if self.save_plots:
            fig.write_html(f'{self.output_dir}interactive_dashboard.html')
        
        
        return fig