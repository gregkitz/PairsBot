"""
Visualization module for intraday backtest performance analysis.

This module provides functions for visualizing intraday backtest results
with a focus on time-of-day patterns, transaction costs, and regime-aware
performance attribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, time, timedelta
import logging
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

def plot_equity_curve(equity_curve: pd.DataFrame, title: str = "Equity Curve", figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the equity curve from a backtest.
    
    Parameters:
    -----------
    equity_curve : pandas.DataFrame
        DataFrame containing 'equity' column with equity values over time
    title : str
        Title for the plot
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve['equity'], label='Equity', linewidth=2)
    
    # Add drawdown if available
    if 'drawdown' in equity_curve.columns:
        ax_dd = ax.twinx()
        ax_dd.fill_between(
            equity_curve.index, 
            0, 
            -100 * equity_curve['drawdown'], 
            alpha=0.3, 
            color='red', 
            label='Drawdown (%)'
        )
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.set_ylim(bottom=0)
    
    # Format x-axis to show dates nicely
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if len(equity_curve) > 20:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
    
    # Add labels and grid
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    if 'drawdown' in equity_curve.columns:
        lines2, labels2 = ax_dd.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend(loc='upper left')
    
    return fig

def plot_intraday_patterns(equity_curve: pd.DataFrame, title: str = "Intraday Return Patterns", 
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot intraday return patterns by time of day.
    
    Parameters:
    -----------
    equity_curve : pandas.DataFrame
        DataFrame containing 'equity' column with equity values over time
    title : str
        Title for the plot
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        logger.warning("Equity curve index is not DatetimeIndex. Cannot plot intraday patterns.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Cannot plot intraday patterns: Equity curve does not have datetime index", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Calculate returns
    returns = equity_curve['equity'].pct_change().dropna()
    
    # Group by hour and minute
    returns_by_time = pd.DataFrame({
        'returns': returns,
        'hour': returns.index.hour,
        'minute': returns.index.minute
    })
    
    # Create time buckets (30-minute intervals)
    def create_time_bucket(row):
        hour = row['hour']
        minute = row['minute']
        bucket_minute = (minute // 30) * 30
        return f"{hour:02d}:{bucket_minute:02d}"
    
    returns_by_time['time_bucket'] = returns_by_time.apply(create_time_bucket, axis=1)
    
    # Calculate statistics by time bucket
    time_stats = returns_by_time.groupby('time_bucket')['returns'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
        ('sum', 'sum')
    ]).reset_index()
    
    # Calculate Sharpe ratio by time bucket (using mean / std)
    time_stats['sharpe'] = time_stats['mean'] / time_stats['std'].replace(0, np.nan)
    time_stats = time_stats.sort_values('time_bucket')
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Mean returns by time bucket
    axes[0, 0].bar(time_stats['time_bucket'], time_stats['mean'] * 100, color='blue', alpha=0.7)
    axes[0, 0].set_title('Mean Returns by Time of Day (%)')
    axes[0, 0].set_xlabel('Time Bucket')
    axes[0, 0].set_ylabel('Mean Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    for label in axes[0, 0].get_xticklabels():
        label.set_rotation(45)
    
    # Plot 2: Total returns by time bucket
    axes[0, 1].bar(time_stats['time_bucket'], time_stats['sum'] * 100, color='green', alpha=0.7)
    axes[0, 1].set_title('Total Returns by Time of Day (%)')
    axes[0, 1].set_xlabel('Time Bucket')
    axes[0, 1].set_ylabel('Total Return (%)')
    axes[0, 1].grid(True, alpha=0.3)
    for label in axes[0, 1].get_xticklabels():
        label.set_rotation(45)
    
    # Plot 3: Sharpe ratio by time bucket
    axes[1, 0].bar(time_stats['time_bucket'], time_stats['sharpe'], color='purple', alpha=0.7)
    axes[1, 0].set_title('Sharpe Ratio by Time of Day')
    axes[1, 0].set_xlabel('Time Bucket')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    for label in axes[1, 0].get_xticklabels():
        label.set_rotation(45)
    
    # Plot 4: Trade count by time bucket
    axes[1, 1].bar(time_stats['time_bucket'], time_stats['count'], color='orange', alpha=0.7)
    axes[1, 1].set_title('Number of Trades by Time of Day')
    axes[1, 1].set_xlabel('Time Bucket')
    axes[1, 1].set_ylabel('Trade Count')
    axes[1, 1].grid(True, alpha=0.3)
    for label in axes[1, 1].get_xticklabels():
        label.set_rotation(45)
    
    # Adjust layout and add main title
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    return fig

def plot_transaction_costs(transaction_costs: Dict, trades: List[Dict], 
                          title: str = "Transaction Cost Analysis", figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot transaction cost analysis.
    
    Parameters:
    -----------
    transaction_costs : dict
        Dictionary with transaction cost data (commission and slippage lists)
    trades : list
        List of trade dictionaries with timestamps
    title : str
        Title for the plot
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Extract commission and slippage from transaction costs
    commission = transaction_costs.get('commission', [])
    slippage = transaction_costs.get('slippage', [])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Commission vs Slippage as pie chart
    total_commission = sum(commission)
    total_slippage = sum(slippage)
    axes[0, 0].pie(
        [total_commission, total_slippage],
        labels=['Commission', 'Slippage'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['lightblue', 'lightgreen']
    )
    axes[0, 0].set_title('Commission vs Slippage Breakdown')
    
    # Plot 2: Commission and Slippage by trade (histogram)
    axes[0, 1].hist(commission, bins=20, alpha=0.7, label='Commission', color='blue')
    axes[0, 1].hist(slippage, bins=20, alpha=0.7, label='Slippage', color='green')
    axes[0, 1].set_title('Distribution of Transaction Costs')
    axes[0, 1].set_xlabel('Cost')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Plot 3: Transaction costs over time
    if trades and all('timestamp' in trade for trade in trades):
        trade_timestamps = [trade['timestamp'] for trade in trades]
        total_costs = [c + s for c, s in zip(commission, slippage)]
        
        axes[1, 0].plot(trade_timestamps, commission, label='Commission', color='blue', marker='o', linestyle='-', alpha=0.7)
        axes[1, 0].plot(trade_timestamps, slippage, label='Slippage', color='green', marker='o', linestyle='-', alpha=0.7)
        axes[1, 0].plot(trade_timestamps, total_costs, label='Total Cost', color='red', marker='o', linestyle='-', alpha=0.7)
        axes[1, 0].set_title('Transaction Costs Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()
    else:
        axes[1, 0].text(0.5, 0.5, "No timestamp data available for trades", 
                     horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
    
    # Plot 4: Transaction cost impact on returns
    total_costs = sum(commission) + sum(slippage)
    labels = ['Gross Return', 'Commission', 'Slippage', 'Net Return']
    if trades and all('pnl' in trade for trade in trades):
        gross_pnl = sum(trade['pnl'] for trade in trades) + total_costs
        net_pnl = gross_pnl - total_costs
        values = [gross_pnl, -sum(commission), -sum(slippage), net_pnl]
        
        # Create waterfall chart
        bottom = 0
        for i, value in enumerate(values):
            if i == 0:  # First bar (Gross Return)
                axes[1, 1].bar(labels[i], value, bottom=0, color='green', alpha=0.7)
                bottom = value
            elif i == len(values) - 1:  # Last bar (Net Return)
                axes[1, 1].bar(labels[i], value, bottom=0, color='blue', alpha=0.7)
            else:  # Middle bars (costs)
                axes[1, 1].bar(labels[i], value, bottom=bottom, color='red', alpha=0.7)
                bottom += value
        
        # Add dashed line connecting gross and net
        axes[1, 1].plot([0, 3], [gross_pnl, net_pnl], 'k--', alpha=0.3)
        
        axes[1, 1].set_title('Transaction Cost Impact on Returns')
        axes[1, 1].set_ylabel('PnL')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No PnL data available for trades", 
                     horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
    
    # Adjust layout and add main title
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    return fig

def plot_regime_performance(equity_curve: pd.DataFrame, regime_data: pd.DataFrame, 
                          title: str = "Performance by Market Regime", figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot performance broken down by market regime.
    
    Parameters:
    -----------
    equity_curve : pandas.DataFrame
        DataFrame containing 'equity' column with equity values over time
    regime_data : pandas.DataFrame
        DataFrame with regime labels over time
    title : str
        Title for the plot
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if not isinstance(equity_curve.index, pd.DatetimeIndex) or not isinstance(regime_data.index, pd.DatetimeIndex):
        logger.warning("Equity curve or regime data does not have DatetimeIndex. Cannot plot regime performance.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Cannot plot regime performance: Data does not have datetime index", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Calculate returns
    returns = equity_curve['equity'].pct_change().dropna()
    
    # Align regime data with returns (forward fill regime labels)
    aligned_regime = regime_data.reindex(returns.index, method='ffill')
    
    # Create DataFrame with returns and regime
    regime_returns = pd.DataFrame({
        'returns': returns,
        'regime': aligned_regime.iloc[:, 0] if aligned_regime.shape[1] > 0 else aligned_regime
    })
    
    # Calculate statistics by regime
    regime_stats = regime_returns.groupby('regime')['returns'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
        ('sum', 'sum')
    ]).reset_index()
    
    # Calculate Sharpe ratio by regime (using mean / std)
    regime_stats['sharpe'] = regime_stats['mean'] / regime_stats['std'].replace(0, np.nan)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Cumulative returns by regime
    for regime in regime_returns['regime'].unique():
        # Get returns for this regime
        regime_mask = regime_returns['regime'] == regime
        regime_cum_returns = (1 + regime_returns.loc[regime_mask, 'returns']).cumprod() - 1
        
        # Find contiguous periods for this regime
        periods = []
        current_period = []
        
        for i, (idx, is_regime) in enumerate(regime_mask.items()):
            if is_regime:
                current_period.append(idx)
            elif current_period:
                periods.append(current_period)
                current_period = []
        
        if current_period:
            periods.append(current_period)
        
        # Plot each contiguous period
        for period in periods:
            if len(period) > 1:  # Need at least 2 points to plot a line
                period_returns = (1 + regime_returns.loc[period, 'returns']).cumprod() - 1
                axes[0, 0].plot(period, period_returns * 100, label=f'Regime {regime}' if period == periods[0] else "")
    
    axes[0, 0].set_title('Cumulative Returns by Regime (%)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Cumulative Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Mean daily returns by regime
    regime_colors = sns.color_palette("viridis", len(regime_stats))
    axes[0, 1].bar(regime_stats['regime'].astype(str), regime_stats['mean'] * 100, color=regime_colors, alpha=0.7)
    axes[0, 1].set_title('Mean Returns by Regime (%)')
    axes[0, 1].set_xlabel('Regime')
    axes[0, 1].set_ylabel('Mean Return (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sharpe ratio by regime
    axes[1, 0].bar(regime_stats['regime'].astype(str), regime_stats['sharpe'], color=regime_colors, alpha=0.7)
    axes[1, 0].set_title('Sharpe Ratio by Regime')
    axes[1, 0].set_xlabel('Regime')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Regime distribution
    axes[1, 1].pie(
        regime_stats['count'],
        labels=regime_stats['regime'].astype(str),
        autopct='%1.1f%%',
        startangle=90,
        colors=regime_colors
    )
    axes[1, 1].set_title('Distribution of Regimes')
    
    # Adjust layout and add main title
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    return fig

def plot_intraday_constraints(intraday_metrics: Dict, title: str = "Intraday Trading Constraints Analysis", 
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot analysis of intraday trading constraints.
    
    Parameters:
    -----------
    intraday_metrics : dict
        Dictionary with intraday metrics including time violations and forced exits
    title : str
        Title for the plot
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data from intraday metrics
    time_violations = intraday_metrics.get('time_violations', [])
    forced_exits = intraday_metrics.get('forced_exits', [])
    missed_trades = intraday_metrics.get('missed_trades', [])
    
    # Plot 1: Time violations by reason
    time_violation_reasons = {}
    for violation in time_violations:
        reason = violation.get('reason', 'unknown')
        time_violation_reasons[reason] = time_violation_reasons.get(reason, 0) + 1
    
    if time_violation_reasons:
        axes[0, 0].bar(time_violation_reasons.keys(), time_violation_reasons.values(), color='orange', alpha=0.7)
        axes[0, 0].set_title('Time Violations by Reason')
        axes[0, 0].set_xlabel('Reason')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "No time violations recorded", 
                     horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)
    
    # Plot 2: Forced exits by reason
    forced_exit_reasons = {}
    for exit in forced_exits:
        reason = exit.get('reason', 'unknown')
        forced_exit_reasons[reason] = forced_exit_reasons.get(reason, 0) + 1
    
    if forced_exit_reasons:
        axes[0, 1].bar(forced_exit_reasons.keys(), forced_exit_reasons.values(), color='red', alpha=0.7)
        axes[0, 1].set_title('Forced Exits by Reason')
        axes[0, 1].set_xlabel('Reason')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "No forced exits recorded", 
                     horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)
    
    # Plot 3: Holding time distribution for forced exits
    holding_times = [exit.get('holding_minutes', 0) for exit in forced_exits if 'holding_minutes' in exit]
    
    if holding_times:
        axes[1, 0].hist(holding_times, bins=20, color='purple', alpha=0.7)
        axes[1, 0].set_title('Holding Time Distribution for Forced Exits')
        axes[1, 0].set_xlabel('Holding Time (minutes)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "No holding time data available", 
                     horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
    
    # Plot 4: Time of day for violations and exits
    if isinstance(time_violations, list) and isinstance(forced_exits, list) and \
       (len(time_violations) > 0 or len(forced_exits) > 0):
        
        # Group time violations by hour
        violation_hours = {}
        for violation in time_violations:
            if 'timestamp' in violation and isinstance(violation['timestamp'], (datetime, pd.Timestamp)):
                hour = violation['timestamp'].hour
                violation_hours[hour] = violation_hours.get(hour, 0) + 1
        
        # Group forced exits by hour
        exit_hours = {}
        for exit in forced_exits:
            if 'timestamp' in exit and isinstance(exit['timestamp'], (datetime, pd.Timestamp)):
                hour = exit['timestamp'].hour
                exit_hours[hour] = exit_hours.get(hour, 0) + 1
        
        # Plot on the same axes
        hours = sorted(set(list(violation_hours.keys()) + list(exit_hours.keys())))
        violation_counts = [violation_hours.get(hour, 0) for hour in hours]
        exit_counts = [exit_hours.get(hour, 0) for hour in hours]
        
        if hours:
            width = 0.35
            axes[1, 1].bar(np.array(hours) - width/2, violation_counts, width, label='Time Violations', color='orange', alpha=0.7)
            axes[1, 1].bar(np.array(hours) + width/2, exit_counts, width, label='Forced Exits', color='red', alpha=0.7)
            axes[1, 1].set_title('Constraints by Hour of Day')
            axes[1, 1].set_xlabel('Hour')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xticks(hours)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No time-of-day data available", 
                         horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, "No time-of-day data available", 
                     horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
    
    # Adjust layout and add main title
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    return fig

def create_intraday_performance_dashboard(backtest_results: Dict, regime_data: Optional[pd.DataFrame] = None, 
                                        figsize: Tuple[int, int] = (12, 8)) -> List[plt.Figure]:
    """
    Create a comprehensive dashboard of intraday backtest performance analysis.
    
    Parameters:
    -----------
    backtest_results : dict
        Dictionary with backtest results including equity curve, trades, and intraday metrics
    regime_data : pandas.DataFrame, optional
        DataFrame with regime labels over time
    figsize : tuple
        Base figure size as (width, height)
        
    Returns:
    --------
    list
        List of matplotlib.figure.Figure objects
    """
    figures = []
    
    # Extract data from backtest results
    equity_curve = backtest_results.get('equity_curve')
    trades = backtest_results.get('trades', [])
    intraday_metrics = backtest_results.get('intraday_metrics', {})
    transaction_costs = intraday_metrics.get('transaction_costs', {})
    
    # 1. Plot equity curve
    if equity_curve is not None and isinstance(equity_curve, pd.DataFrame) and 'equity' in equity_curve.columns:
        fig_equity = plot_equity_curve(equity_curve, title="Intraday Trading Equity Curve", figsize=figsize)
        figures.append(fig_equity)
    
    # 2. Plot intraday patterns
    if equity_curve is not None and isinstance(equity_curve, pd.DataFrame) and 'equity' in equity_curve.columns:
        fig_patterns = plot_intraday_patterns(equity_curve, title="Intraday Return Patterns", figsize=figsize)
        figures.append(fig_patterns)
    
    # 3. Plot transaction costs
    if transaction_costs and trades:
        fig_costs = plot_transaction_costs(transaction_costs, trades, title="Transaction Cost Analysis", figsize=figsize)
        figures.append(fig_costs)
    
    # 4. Plot regime performance
    if regime_data is not None and equity_curve is not None and isinstance(equity_curve, pd.DataFrame) and 'equity' in equity_curve.columns:
        fig_regime = plot_regime_performance(equity_curve, regime_data, title="Performance by Market Regime", figsize=figsize)
        figures.append(fig_regime)
    
    # 5. Plot intraday constraints
    if intraday_metrics:
        fig_constraints = plot_intraday_constraints(intraday_metrics, title="Intraday Trading Constraints Analysis", figsize=figsize)
        figures.append(fig_constraints)
    
    return figures

def save_performance_dashboard(figures: List[plt.Figure], output_dir: str, prefix: str = "intraday_performance"):
    """
    Save all figures in the performance dashboard to files.
    
    Parameters:
    -----------
    figures : list
        List of matplotlib.figure.Figure objects
    output_dir : str
        Directory to save figures to
    prefix : str
        Prefix for filenames
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each figure
    for i, fig in enumerate(figures):
        filename = f"{prefix}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close(fig)
    
    logger.info(f"Saved {len(figures)} performance dashboard figures to {output_dir}") 