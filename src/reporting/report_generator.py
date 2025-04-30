"""
Backtest Report Generator for the Intraday Statistical Arbitrage System.

This module provides a BacktestReportGenerator class that creates comprehensive
HTML reports from backtest results with interactive visualizations and metrics.
"""

import os
import jinja2
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import scipy.stats as stats
from typing import Dict, List, Union, Optional, Any, Tuple

from .metrics import calculate_performance_metrics



# Helper function to convert Plotly figures to JSON
def _plotly_fig_to_json(fig):
    '''Convert a Plotly figure to JSON-compatible dict.'''
    if hasattr(fig, 'to_dict'):
        return fig.to_dict()
    elif hasattr(fig, 'to_plotly_json'):
        return fig.to_plotly_json()
    else:
        # For basic data structures, attempt direct conversion
        return {
            'data': [trace for trace in fig.data],
            'layout': fig.layout
        }

class BacktestReportGenerator:
    """
    Backtest Report Generator for the Intraday Statistical Arbitrage System.
    
    This class creates comprehensive HTML reports from backtest results with
    interactive visualizations and metrics, making it easy to analyze and
    share backtest results.
    """
    
    def __init__(self, 
                template_dir: str = None,
                output_dir: str = None):
        """
        Initialize the backtest report generator.
        
        Parameters:
        -----------
        template_dir : str, optional
            Directory containing the HTML templates
        output_dir : str, optional
            Directory to save the generated reports
        """
        # Set template directory
        if template_dir is None:
            # Use the templates directory in the current module
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        
        self.template_dir = template_dir
        
        # Set output directory
        if output_dir is None:
            # Use the 'reports' directory by default
            output_dir = os.path.join(os.getcwd(), 'reports')
        
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up Jinja2 template environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_report(self,
                       title: str,
                       strategy_description: str,
                       equity_curve: pd.Series,
                       trades: Optional[pd.DataFrame] = None,
                       benchmark: Optional[pd.Series] = None,
                       risk_free_rate: float = 0.0,
                       output_filename: Optional[str] = None,
                       additional_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive HTML report from backtest results.
        
        Parameters:
        -----------
        title : str
            Title of the report
        strategy_description : str
            Description of the strategy
        equity_curve : pd.Series
            Time series of portfolio equity values
        trades : pd.DataFrame, optional
            DataFrame containing individual trade information
        benchmark : pd.Series, optional
            Time series of benchmark values for comparison
        risk_free_rate : float
            Annualized risk-free rate
        output_filename : str, optional
            Filename for the HTML report
        additional_metrics : dict, optional
            Additional metrics to include in the report
        
        Returns:
        --------
        str
            Path to the generated HTML report
        """
        # Calculate performance metrics
        metrics = calculate_performance_metrics(
            equity_curve=equity_curve,
            trades=trades,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate
        )
        
        # Add additional metrics if provided
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Generate visualizations
        equity_data = self._generate_equity_chart(equity_curve, benchmark)
        drawdown_data = self._generate_drawdown_chart(metrics['underwater'])
        
        # Generate monthly returns data
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        monthly_returns_data = self._generate_monthly_returns_chart(monthly_returns)
        monthly_returns_table = self._generate_monthly_returns_table(monthly_returns)
        
        # Generate trade analysis data if trades are provided
        trade_analysis = trades is not None and len(trades) > 0
        
        trade_outcomes_data = None
        trade_pnl_data = None
        pnl_by_side_data = None
        top_trades_table = None
        
        if trade_analysis:
            trade_outcomes_data = self._generate_trade_outcomes_chart(trades)
            trade_pnl_data = self._generate_trade_pnl_chart(trades)
            pnl_by_side_data = self._generate_pnl_by_side_chart(trades)
            top_trades_table = self._generate_top_trades_table(trades)
        
        # Generate benchmark comparison data if benchmark is provided
        benchmark_comparison = benchmark is not None and len(benchmark) > 0
        benchmark_data = None
        
        if benchmark_comparison:
            benchmark_data = self._generate_benchmark_chart(equity_curve, benchmark)
        
        # Prepare template context
        context = {
            'title': title,
            'strategy_description': strategy_description,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': equity_curve.index[0].strftime('%Y-%m-%d'),
            'end_date': equity_curve.index[-1].strftime('%Y-%m-%d'),
            
            # Add all metrics to the context
            **metrics,
            
            # Add default values for metrics that might not be in the metrics dict
            'recovery_factor': metrics.get('recovery_factor', 0.0),
            'skewness': stats.skew(metrics.get('daily_returns', pd.Series([0]))),
            'kurtosis': stats.kurtosis(metrics.get('daily_returns', pd.Series([0]))),
            'best_day': metrics.get('daily_returns', pd.Series([0])).max(),
            'worst_day': metrics.get('daily_returns', pd.Series([0])).min(),
            'positive_days': (metrics.get('daily_returns', pd.Series([0])) > 0).mean(),
            'avg_up_day': metrics.get('avg_positive_return', 0.0),
            'avg_down_day': metrics.get('avg_negative_return', 0.0),
            'strategy_rating': metrics.get('summary', {}).get('strategy_rating', 'C'),
            
            # Add chart data
            'equity_data': json.dumps(equity_data, default=lambda obj: _plotly_fig_to_json(obj) if hasattr(obj, "to_plotly_json") or hasattr(obj, "to_dict") else str(obj)),
            'drawdown_data': json.dumps(drawdown_data, default=lambda obj: _plotly_fig_to_json(obj) if hasattr(obj, "to_plotly_json") or hasattr(obj, "to_dict") else str(obj)),
            'monthly_returns_data': json.dumps(monthly_returns_data, default=lambda obj: _plotly_fig_to_json(obj) if hasattr(obj, "to_plotly_json") or hasattr(obj, "to_dict") else str(obj)),
            'monthly_returns_table': monthly_returns_table,
            
            # Add trade analysis data
            'trade_analysis': trade_analysis,
            'benchmark_comparison': benchmark_comparison,
        }
        
        # Add trade-specific data if available
        if trade_analysis:
            context.update({
                'trade_outcomes_data': json.dumps(trade_outcomes_data, default=lambda obj: _plotly_fig_to_json(obj) if hasattr(obj, "to_plotly_json") or hasattr(obj, "to_dict") else str(obj)),
                'trade_pnl_data': json.dumps(trade_pnl_data, default=lambda obj: _plotly_fig_to_json(obj) if hasattr(obj, "to_plotly_json") or hasattr(obj, "to_dict") else str(obj)),
                'pnl_by_side_data': json.dumps(pnl_by_side_data, default=lambda obj: _plotly_fig_to_json(obj) if hasattr(obj, "to_plotly_json") or hasattr(obj, "to_dict") else str(obj)),
                'top_trades_table': top_trades_table
            })
        
        # Add benchmark-specific data if available
        if benchmark_comparison:
            context.update({
                'benchmark_data': json.dumps(benchmark_data, default=lambda obj: _plotly_fig_to_json(obj) if hasattr(obj, "to_plotly_json") or hasattr(obj, "to_dict") else str(obj))
            })
        
        # Render the template
        template = self.env.get_template('report_template.html')
        html_content = template.render(**context)
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Add .html extension if not present
        if not output_filename.endswith('.html'):
            output_filename += '.html'
        
        # Write HTML to file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_equity_chart(self, 
                             equity_curve: pd.Series, 
                             benchmark: Optional[pd.Series] = None) -> Dict:
        """
        Generate equity curve chart data.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Time series of portfolio equity values
        benchmark : pd.Series, optional
            Time series of benchmark values for comparison
        
        Returns:
        --------
        dict
            Plotly chart data and layout
        """
        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add benchmark if provided
        if benchmark is not None:
            # Align benchmark to equity curve dates if needed
            if not benchmark.index.equals(equity_curve.index):
                benchmark = benchmark.reindex(equity_curve.index, method='ffill')
            
            # Normalize benchmark to start at the same value as equity curve
            benchmark_normalized = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
        
        # Create layout
        layout = dict(
            title='Equity Curve',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Equity ($)'),
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
            template='plotly_white'
        )
        
        # Set figure layout
        fig.update_layout(layout)
        
        # Return figure data and layout
        return {'data': fig.data, 'layout': fig.layout}
    
    def _generate_drawdown_chart(self, underwater: pd.Series) -> Dict:
        """
        Generate drawdown chart data.
        
        Parameters:
        -----------
        underwater : pd.Series
            Time series of drawdown percentages
        
        Returns:
        --------
        dict
            Plotly chart data and layout
        """
        # Create figure
        fig = go.Figure()
        
        # Add underwater curve
        fig.add_trace(go.Scatter(
            x=underwater.index,
            y=underwater.values * 100,  # Convert to percentage
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#d62728', width=1),
            fillcolor='rgba(214, 39, 40, 0.3)'
        ))
        
        # Create layout
        layout = dict(
            title='Drawdown',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Drawdown (%)', range=[min(underwater.values * 100) * 1.1, 0]),
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Set figure layout
        fig.update_layout(layout)
        
        # Return figure data and layout
        return {'data': fig.data, 'layout': fig.layout}
    
    def _generate_benchmark_chart(self, 
                                equity_curve: pd.Series, 
                                benchmark: pd.Series) -> Dict:
        """
        Generate benchmark comparison chart data.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Time series of portfolio equity values
        benchmark : pd.Series
            Time series of benchmark values for comparison
        
        Returns:
        --------
        dict
            Plotly chart data and layout
        """
        # Create figure
        fig = go.Figure()
        
        # Calculate cumulative returns
        strategy_returns = equity_curve.pct_change().fillna(0)
        strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
        
        # Align benchmark to equity curve dates
        if not benchmark.index.equals(equity_curve.index):
            benchmark = benchmark.reindex(equity_curve.index, method='ffill')
        
        benchmark_returns = benchmark.pct_change().fillna(0)
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        
        # Add strategy curve
        fig.add_trace(go.Scatter(
            x=strategy_cum_returns.index,
            y=strategy_cum_returns.values * 100,  # Convert to percentage
            mode='lines',
            name='Strategy',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add benchmark curve
        fig.add_trace(go.Scatter(
            x=benchmark_cum_returns.index,
            y=benchmark_cum_returns.values * 100,  # Convert to percentage
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Create layout
        layout = dict(
            title='Cumulative Returns Comparison',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Cumulative Return (%)'),
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
            template='plotly_white'
        )
        
        # Set figure layout
        fig.update_layout(layout)
        
        # Return figure data and layout
        return {'data': fig.data, 'layout': fig.layout}
    
    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Calculate monthly returns from equity curve.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Time series of portfolio equity values
        
        Returns:
        --------
        pd.DataFrame
            DataFrame of monthly returns
        """
        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)
        
        # Resample to monthly returns (last day of month)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table with years as rows and months as columns
        monthly_pivot = monthly_returns.reset_index()
        monthly_pivot['Year'] = monthly_pivot['index'].dt.year
        monthly_pivot['Month'] = monthly_pivot['index'].dt.month
        
        # Pivot the data
        monthly_pivot = monthly_pivot.pivot(index='Year', columns='Month', values=0)
        
        # Calculate YTD returns
        monthly_pivot['YTD'] = ((1 + monthly_pivot.fillna(0)).prod(axis=1) - 1)
        
        return monthly_pivot
    
    def _generate_monthly_returns_chart(self, monthly_returns: pd.DataFrame) -> Dict:
        """
        Generate monthly returns chart data.
        
        Parameters:
        -----------
        monthly_returns : pd.DataFrame
            DataFrame of monthly returns
        
        Returns:
        --------
        dict
            Plotly chart data and layout
        """
        # Drop YTD column for the heatmap
        monthly_data = monthly_returns.drop(columns=['YTD'], errors='ignore')
        
        # Create a copy and replace column numbers with month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        # Convert to percentage values
        monthly_pct = monthly_data * 100
        
        # Create figure using make_subplots
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=monthly_pct.values,
            x=[month_names[m] for m in monthly_pct.columns],
            y=monthly_pct.index,
            colorscale='RdBu',
            zmid=0,
            text=[[f"{val:.2f}%" for val in row] for row in monthly_pct.values],
            hoverinfo='text+x+y',
            colorbar=dict(title='Return (%)')
        ))
        
        # Create layout
        layout = dict(
            title='Monthly Returns (%)',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Year', dtick=1),
            template='plotly_white'
        )
        
        # Set figure layout
        fig.update_layout(layout)
        
        # Return figure data and layout
        return {'data': fig.data, 'layout': fig.layout}
    
    def _generate_monthly_returns_table(self, monthly_returns: pd.DataFrame) -> str:
        """
        Generate monthly returns table HTML.
        
        Parameters:
        -----------
        monthly_returns : pd.DataFrame
            DataFrame of monthly returns
        
        Returns:
        --------
        str
            HTML for the monthly returns table
        """
        # Create a copy with proper month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        # Format the DataFrame for display
        formatted_returns = monthly_returns.copy()
        
        # Convert values to strings with formatting
        for col in formatted_returns.columns:
            formatted_returns[col] = formatted_returns[col].apply(
                lambda x: f"{x*100:.2f}%" if not pd.isna(x) else ""
            )
        
        # Rename columns
        formatted_returns.columns = [month_names.get(col, col) for col in formatted_returns.columns]
        
        # Convert to HTML table
        table_html = formatted_returns.reset_index().to_html(
            index=False, 
            classes='table table-sm table-striped',
            border=0,
            escape=False
        )
        
        return table_html
    
    def _generate_trade_outcomes_chart(self, trades: pd.DataFrame) -> Dict:
        """
        Generate trade outcomes chart data.
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame containing individual trade information
        
        Returns:
        --------
        dict
            Plotly chart data and layout
        """
        # Count winning and losing trades
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        breakeven_trades = len(trades[trades['pnl'] == 0])
        
        # Create figure
        fig = go.Figure()
        
        # Add pie chart
        fig.add_trace(go.Pie(
            labels=['Winning', 'Losing', 'Breakeven'],
            values=[winning_trades, losing_trades, breakeven_trades],
            marker=dict(colors=['#28a745', '#dc3545', '#6c757d']),
            textinfo='value+percent',
            hole=0.4
        ))
        
        # Create layout
        layout = dict(
            title='Trade Outcomes',
            template='plotly_white',
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)')
        )
        
        # Set figure layout
        fig.update_layout(layout)
        
        # Return figure data and layout
        return {'data': fig.data, 'layout': fig.layout}
    
    def _generate_trade_pnl_chart(self, trades: pd.DataFrame) -> Dict:
        """
        Generate trade P&L distribution chart data.
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame containing individual trade information
        
        Returns:
        --------
        dict
            Plotly chart data and layout
        """
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=trades['pnl'],
            nbinsx=20,
            marker=dict(
                color=trades['pnl'],
                colorscale=[[0, '#dc3545'], [0.5, '#6c757d'], [1, '#28a745']],
                cmin=trades['pnl'].min(),
                cmax=trades['pnl'].max()
            ),
            name='P&L Distribution'
        ))
        
        # Add mean line
        fig.add_vline(
            x=trades['pnl'].mean(),
            line_dash='dash',
            line_color='black',
            annotation_text=f"Mean: {trades['pnl'].mean():.2f}",
            annotation_position='top right'
        )
        
        # Create layout
        layout = dict(
            title='Trade P&L Distribution',
            xaxis=dict(title='P&L'),
            yaxis=dict(title='Count'),
            template='plotly_white'
        )
        
        # Set figure layout
        fig.update_layout(layout)
        
        # Return figure data and layout
        return {'data': fig.data, 'layout': fig.layout}
    
    def _generate_pnl_by_side_chart(self, trades: pd.DataFrame) -> Dict:
        """
        Generate cumulative P&L by side chart data.
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame containing individual trade information
        
        Returns:
        --------
        dict
            Plotly chart data and layout
        """
        # Check if we have the 'side' column
        if 'side' not in trades.columns:
            # Create a default chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 0],
                mode='lines',
                name='No side data'
            ))
            layout = dict(
                title='Cumulative P&L by Side',
                xaxis=dict(title='Trade #'),
                yaxis=dict(title='Cumulative P&L'),
                template='plotly_white',
                showlegend=False
            )
            fig.update_layout(layout)
            return {'data': fig.data, 'layout': fig.layout}
        
        # Calculate cumulative P&L by side
        long_trades = trades[trades['side'] == 'long'].sort_values('entry_time')
        short_trades = trades[trades['side'] == 'short'].sort_values('entry_time')
        
        long_cum_pnl = long_trades['pnl'].cumsum()
        short_cum_pnl = short_trades['pnl'].cumsum()
        all_cum_pnl = trades.sort_values('entry_time')['pnl'].cumsum()
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        if len(long_cum_pnl) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(long_cum_pnl) + 1)),
                y=long_cum_pnl.values,
                mode='lines',
                name='Long',
                line=dict(color='#28a745', width=2)
            ))
        
        if len(short_cum_pnl) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(short_cum_pnl) + 1)),
                y=short_cum_pnl.values,
                mode='lines',
                name='Short',
                line=dict(color='#dc3545', width=2)
            ))
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(all_cum_pnl) + 1)),
            y=all_cum_pnl.values,
            mode='lines',
            name='All',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Create layout
        layout = dict(
            title='Cumulative P&L by Side',
            xaxis=dict(title='Trade #'),
            yaxis=dict(title='Cumulative P&L'),
            template='plotly_white',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)')
        )
        
        # Set figure layout
        fig.update_layout(layout)
        
        # Return figure data and layout
        return {'data': fig.data, 'layout': fig.layout}
    
    def _generate_top_trades_table(self, trades: pd.DataFrame) -> str:
        """
        Generate top 10 trades table HTML.
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame containing individual trade information
        
        Returns:
        --------
        str
            HTML for the top 10 trades table
        """
        # Check for required columns
        required_columns = ['entry_time', 'exit_time', 'pnl']
        missing_columns = [col for col in required_columns if col not in trades.columns]
        
        if missing_columns:
            return f"<tr><td colspan='8'>Missing columns: {', '.join(missing_columns)}</td></tr>"
        
        # Sort by P&L descending
        top_trades = trades.sort_values('pnl', ascending=False).head(10)
        
        # Create a formatted DataFrame for display
        display_df = pd.DataFrame()
        
        # Convert timestamps to strings
        display_df['Entry Time'] = top_trades['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['Exit Time'] = top_trades['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add side if available
        if 'side' in top_trades.columns:
            display_df['Side'] = top_trades['side'].str.capitalize()
        else:
            display_df['Side'] = 'Unknown'
        
        # Calculate duration in minutes
        duration_minutes = (top_trades['exit_time'] - top_trades['entry_time']).dt.total_seconds() / 60
        display_df['Duration'] = duration_minutes.map(lambda x: f"{x:.1f} min")
        
        # Format P&L
        display_df['P&L'] = top_trades['pnl'].map(lambda x: f"{x:.2f}")
        
        # Add symbol if available
        if 'symbol' in top_trades.columns:
            display_df['Symbol'] = top_trades['symbol']
        else:
            display_df['Symbol'] = 'Unknown'
        
        # Add entry/exit prices if available
        if 'entry_price' in top_trades.columns:
            display_df['Entry Price'] = top_trades['entry_price'].map(lambda x: f"{x:.2f}")
        else:
            display_df['Entry Price'] = 'N/A'
        
        if 'exit_price' in top_trades.columns:
            display_df['Exit Price'] = top_trades['exit_price'].map(lambda x: f"{x:.2f}")
        else:
            display_df['Exit Price'] = 'N/A'
        
        # Convert to HTML table
        table_html = display_df.to_html(
            index=False, 
            classes='table table-sm table-striped table-trades',
            border=0,
            escape=False
        )
        
        # Extract just the tbody content
        tbody_start = table_html.find('<tbody>')
        tbody_end = table_html.find('</tbody>') + 8
        
        return table_html[tbody_start:tbody_end] 