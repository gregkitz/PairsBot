import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os
import seaborn as sns


def plot_pair_prices(ticker1_data, ticker2_data, ticker1_name=None, ticker2_name=None, 
                    normalize=True, figsize=(14, 7), save_path=None):
    """
    Plot price series for a pair.
    
    Parameters:
    -----------
    ticker1_data : pandas.Series
        Price data for first ticker
    ticker2_data : pandas.Series
        Price data for second ticker
    ticker1_name : str, optional
        Name for first ticker
    ticker2_name : str, optional
        Name for second ticker
    normalize : bool
        Whether to normalize prices to start at 1
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects
    """
    if ticker1_name is None:
        ticker1_name = getattr(ticker1_data, 'name', 'Ticker 1')
    if ticker2_name is None:
        ticker2_name = getattr(ticker2_data, 'name', 'Ticker 2')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        ticker1_norm = ticker1_data / ticker1_data.iloc[0]
        ticker2_norm = ticker2_data / ticker2_data.iloc[0]
        
        ax.plot(ticker1_norm.index, ticker1_norm, label=f'{ticker1_name} (normalized)')
        ax.plot(ticker2_norm.index, ticker2_norm, label=f'{ticker2_name} (normalized)')
        ax.set_ylabel('Normalized Price')
    else:
        ax.plot(ticker1_data.index, ticker1_data, label=ticker1_name)
        ax.plot(ticker2_data.index, ticker2_data, label=ticker2_name)
        ax.set_ylabel('Price')
    
    ax.set_title(f'{ticker1_name} vs {ticker2_name} Price Comparison')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax


def plot_spread_zscore(spread, zscore, spread_name=None, figsize=(14, 10), save_path=None):
    """
    Plot spread and its z-score.
    
    Parameters:
    -----------
    spread : pandas.Series
        Spread series
    zscore : pandas.Series
        Z-score series
    spread_name : str, optional
        Name for the spread
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    tuple
        (fig, axes) matplotlib objects
    """
    if spread_name is None:
        spread_name = getattr(spread, 'name', 'Pair')
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot spread
    axes[0].plot(spread.index, spread, label='Spread')
    axes[0].set_title(f'{spread_name} Spread')
    axes[0].set_ylabel('Spread Value')
    axes[0].axhline(y=spread.mean(), color='r', linestyle='--', alpha=0.5, label='Mean')
    axes[0].axhline(y=spread.mean() + spread.std(), color='g', linestyle='--', alpha=0.5, label='+1 Std Dev')
    axes[0].axhline(y=spread.mean() - spread.std(), color='g', linestyle='--', alpha=0.5, label='-1 Std Dev')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot z-score
    axes[1].plot(zscore.index, zscore, label='Z-Score', color='orange')
    axes[1].set_title(f'{spread_name} Z-Score')
    axes[1].set_ylabel('Z-Score')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(y=2, color='g', linestyle='--', alpha=0.5, label='+2')
    axes[1].axhline(y=-2, color='g', linestyle='--', alpha=0.5, label='-2')
    axes[1].legend()
    axes[1].grid(True)
    
    # Format x-axis
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig, axes


def plot_signals_positions(signals, spread=None, zscore=None, pair_name=None, 
                         figsize=(14, 12), save_path=None):
    """
    Plot trading signals and positions.
    
    Parameters:
    -----------
    signals : pandas.DataFrame
        DataFrame with signal columns
    spread : pandas.Series, optional
        Spread series
    zscore : pandas.Series, optional
        Z-score series
    pair_name : str, optional
        Name for the pair
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    tuple
        (fig, axes) matplotlib objects
    """
    if pair_name is None:
        pair_name = 'Pair'
    
    # Determine how many subplots to create
    n_plots = 1
    if spread is not None:
        n_plots += 1
    if zscore is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    # If only one subplot, convert to array for consistent indexing
    if n_plots == 1:
        axes = np.array([axes])
    
    plot_idx = 0
    
    # Plot spread if provided
    if spread is not None:
        axes[plot_idx].plot(spread.index, spread, label='Spread')
        axes[plot_idx].set_title(f'{pair_name} Spread')
        axes[plot_idx].set_ylabel('Spread Value')
        axes[plot_idx].axhline(y=spread.mean(), color='r', linestyle='--', alpha=0.5, label='Mean')
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
        plot_idx += 1
    
    # Plot z-score if provided
    if zscore is not None:
        axes[plot_idx].plot(zscore.index, zscore, label='Z-Score', color='orange')
        axes[plot_idx].set_title(f'{pair_name} Z-Score')
        axes[plot_idx].set_ylabel('Z-Score')
        axes[plot_idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[plot_idx].axhline(y=2, color='g', linestyle='--', alpha=0.5, label='+2')
        axes[plot_idx].axhline(y=-2, color='g', linestyle='--', alpha=0.5, label='-2')
        
        # Overlay entry signals if available
        if 'entry_long' in signals.columns and 'entry_short' in signals.columns:
            long_entries = signals.index[signals['entry_long'] == 1]
            short_entries = signals.index[signals['entry_short'] == 1]
            
            if len(long_entries) > 0:
                long_vals = [zscore.loc[idx] for idx in long_entries if idx in zscore.index]
                axes[plot_idx].scatter(long_entries, long_vals, color='green', marker='^', 
                                      s=100, label='Long Entry')
            
            if len(short_entries) > 0:
                short_vals = [zscore.loc[idx] for idx in short_entries if idx in zscore.index]
                axes[plot_idx].scatter(short_entries, short_vals, color='red', marker='v', 
                                      s=100, label='Short Entry')
        
        # Overlay exit signals if available
        if 'exit_long' in signals.columns and 'exit_short' in signals.columns:
            long_exits = signals.index[(signals['exit_long'] == 1) & (signals['position_long'].shift(1) == 1)]
            short_exits = signals.index[(signals['exit_short'] == 1) & (signals['position_short'].shift(1) == 1)]
            
            if len(long_exits) > 0:
                long_exit_vals = [zscore.loc[idx] for idx in long_exits if idx in zscore.index]
                axes[plot_idx].scatter(long_exits, long_exit_vals, color='blue', marker='o', 
                                      s=80, label='Long Exit')
            
            if len(short_exits) > 0:
                short_exit_vals = [zscore.loc[idx] for idx in short_exits if idx in zscore.index]
                axes[plot_idx].scatter(short_exits, short_exit_vals, color='purple', marker='o', 
                                      s=80, label='Short Exit')
        
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
        plot_idx += 1
    
    # Plot positions
    if 'position' in signals.columns:
        axes[plot_idx].plot(signals.index, signals['position'], label='Position', 
                          drawstyle='steps-post', color='blue')
        axes[plot_idx].set_title(f'{pair_name} Positions')
        axes[plot_idx].set_ylabel('Position (-1 = Short, 0 = Flat, 1 = Long)')
        axes[plot_idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig, axes


def plot_equity_curve(equity_curve, title='Equity Curve', figsize=(14, 7), save_path=None):
    """
    Plot equity curve from backtest results.
    
    Parameters:
    -----------
    equity_curve : pandas.Series
        Equity curve series
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, label='Equity')
    ax.set_title(title)
    ax.set_ylabel('Equity Value')
    ax.grid(True)
    
    # Calculate and plot drawdowns
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100  # as percentage
    
    # Create twin axis for drawdown
    ax2 = ax.twinx()
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Set up legend for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig, (ax, ax2)


def plot_backtest_summary(results, figsize=(14, 14), save_path=None):
    """
    Plot comprehensive backtest summary.
    
    Parameters:
    -----------
    results : dict
        Backtest results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    tuple
        (fig, axes) matplotlib objects
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, figure=fig)
    
    # Create axes
    ax_equity = fig.add_subplot(gs[0, :])
    ax_drawdown = fig.add_subplot(gs[1, :], sharex=ax_equity)
    ax_returns = fig.add_subplot(gs[2, 0])
    ax_monthly = fig.add_subplot(gs[2, 1])
    ax_trades = fig.add_subplot(gs[3, 0])
    ax_metrics = fig.add_subplot(gs[3, 1])
    
    # 1. Equity Curve
    equity_curve = results['equity_curve']
    ax_equity.plot(equity_curve.index, equity_curve.values)
    ax_equity.set_title('Equity Curve')
    ax_equity.set_ylabel('Equity Value')
    ax_equity.grid(True)
    
    # 2. Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100  # as percentage
    ax_drawdown.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax_drawdown.set_title('Drawdown (%)')
    ax_drawdown.set_ylabel('Drawdown (%)')
    ax_drawdown.grid(True)
    
    # 3. Returns Distribution
    daily_returns = results['daily_returns']
    sns.histplot(daily_returns * 100, kde=True, ax=ax_returns)
    ax_returns.set_title('Daily Returns Distribution (%)')
    ax_returns.set_xlabel('Daily Return (%)')
    
    # 4. Monthly Returns Heatmap
    if len(daily_returns) > 30:  # Only if we have enough data
        monthly_returns = daily_returns.resample('M').sum() * 100
        monthly_pivot = monthly_returns.to_frame('returns')
        monthly_pivot.index = pd.to_datetime(monthly_pivot.index)
        monthly_pivot['year'] = monthly_pivot.index.year
        monthly_pivot['month'] = monthly_pivot.index.month
        
        try:
            # Convert to pivot table
            pivot_table = monthly_pivot.pivot_table(index='year', columns='month', values='returns')
            
            # Plot heatmap
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax_monthly)
            ax_monthly.set_title('Monthly Returns (%)')
            ax_monthly.set_xlabel('Month')
            ax_monthly.set_ylabel('Year')
        except:
            # Fall back to bar chart if pivot fails
            monthly_returns.plot(kind='bar', ax=ax_monthly)
            ax_monthly.set_title('Monthly Returns (%)')
            ax_monthly.set_xlabel('Month')
            ax_monthly.set_ylabel('Return (%)')
    else:
        ax_monthly.text(0.5, 0.5, 'Not enough data for monthly returns',
                      horizontalalignment='center', verticalalignment='center')
        ax_monthly.set_title('Monthly Returns (%)')
    
    # 5. Trade Analysis
    if 'trade_history' in results and results['trade_history']:
        trade_df = pd.DataFrame(results['trade_history'])
        
        if not trade_df.empty:
            # Trade PnL
            trade_df['pnl'].plot(kind='bar', ax=ax_trades)
            ax_trades.set_title('Trade P&L')
            ax_trades.set_xlabel('Trade Number')
            ax_trades.set_ylabel('P&L')
            ax_trades.grid(True)
    else:
        ax_trades.text(0.5, 0.5, 'No trade history available',
                     horizontalalignment='center', verticalalignment='center')
        ax_trades.set_title('Trade P&L')
    
    # 6. Performance Metrics
    if 'metrics' in results:
        metrics = results['metrics']
        metrics_text = "\n".join([
            f"Total Return: {metrics.get('total_return', 0):.2%}",
            f"Annual Return: {metrics.get('annual_return', 0):.2%}",
            f"Annual Volatility: {metrics.get('annual_volatility', 0):.2%}",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"Win Rate: {metrics.get('win_rate', 0):.2%}",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Winning Trades: {metrics.get('winning_trades', 0)}",
            f"Losing Trades: {metrics.get('losing_trades', 0)}",
            f"Avg Win: {metrics.get('average_win', 0):.2f}",
            f"Avg Loss: {metrics.get('average_loss', 0):.2f}"
        ])
        
        # Hide axes and show text
        ax_metrics.axis('off')
        ax_metrics.text(0.05, 0.95, metrics_text, horizontalalignment='left',
                      verticalalignment='top', transform=ax_metrics.transAxes,
                      fontsize=12)
    else:
        ax_metrics.text(0.5, 0.5, 'No metrics available',
                      horizontalalignment='center', verticalalignment='center')
    
    ax_metrics.set_title('Performance Metrics')
    
    # Format date axis
    for ax in [ax_equity, ax_drawdown]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig, [ax_equity, ax_drawdown, ax_returns, ax_monthly, ax_trades, ax_metrics] 