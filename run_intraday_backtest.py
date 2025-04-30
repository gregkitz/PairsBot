"""
Script to run an intraday backtest with ML-enhanced signals.

This script demonstrates how to use the intraday configuration with
ML signal enhancements for pairs trading.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import logging

from src.ml_enhancements.intraday_signals import IntradaySignalProcessor
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_intraday_config(config_file):
    """
    Load intraday configuration from a JSON file.
    
    Parameters:
    -----------
    config_file : str
        Path to the intraday configuration file
        
    Returns:
    --------
    dict
        Intraday configuration
    """
    logger.info(f"Loading intraday configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration with {len(config['pairs'])} pairs")
        return config
    except Exception as e:
        logger.error(f"Error loading intraday configuration: {e}")
        return None

def load_intraday_data(symbols, start_date, end_date, timeframe="5min", data_dir="data/processed"):
    """
    Load intraday price data for a list of symbols.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to load data for
    start_date : str
        Start date for data (YYYY-MM-DD)
    end_date : str
        End date for data (YYYY-MM-DD)
    timeframe : str
        Timeframe to resample data to
    data_dir : str
        Directory containing processed data files
        
    Returns:
    --------
    dict
        Dictionary with DataFrames for each symbol
    """
    logger.info(f"Loading intraday data for {len(symbols)} symbols")
    
    data = {}
    
    for symbol in symbols:
        try:
            file_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
            
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                continue
            
            # Load data
            df = pd.read_parquet(file_path)
            
            # Ensure data is sorted by timestamp
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    logger.warning(f"No timestamp column found for {symbol}")
                    continue
            
            df.sort_index(inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Resample to desired timeframe
            if timeframe != "1min":
                # For OHLCV data
                ohlcv_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                
                # Extract available columns
                resample_dict = {col: ohlcv_dict[col] for col in ohlcv_dict.keys() if col in df.columns}
                
                # Resample
                df = df.resample(timeframe).agg(resample_dict)
            
            # Keep only market hours (9:30 AM - 4:00 PM Eastern)
            df = df.between_time('09:30', '16:00')
            
            logger.info(f"Loaded data for {symbol}: {len(df)} rows")
            data[symbol] = df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    return data

def calculate_spread(data, pair_config):
    """
    Calculate spread for a pair based on configuration.
    
    Parameters:
    -----------
    data : dict
        Dictionary with DataFrames for each symbol
    pair_config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with spread and z-score
    """
    symbol1 = pair_config["symbol1"]
    symbol2 = pair_config["symbol2"]
    
    if symbol1 not in data or symbol2 not in data:
        logger.error(f"Missing data for pair {symbol1}-{symbol2}")
        return None
    
    # Extract price data
    df1 = data[symbol1].copy()
    df2 = data[symbol2].copy()
    
    # Ensure we have 'close' column in both DataFrames
    if 'close' not in df1.columns or 'close' not in df2.columns:
        logger.error(f"Missing 'close' column for pair {symbol1}-{symbol2}")
        return None
    
    # Reindex to ensure both have the same dates
    # Use the intersection of both indexes to ensure alignment
    common_index = df1.index.intersection(df2.index)
    
    if len(common_index) == 0:
        logger.error(f"No common timestamps for pair {symbol1}-{symbol2}")
        return None
    
    # Filter to only include common timestamps
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    # Create the merged DataFrame
    merged = pd.DataFrame({
        symbol1: df1['close'],
        symbol2: df2['close']
    }, index=common_index)
    
    # Drop rows with missing values
    merged = merged.dropna()
    
    if len(merged) == 0:
        logger.error(f"No valid data after merging for pair {symbol1}-{symbol2}")
        return None
    
    # Calculate spread
    lookback = pair_config["config"]["lookback"]
    
    if pair_config["config"].get("use_rolling_regression", False):
        # Use rolling regression to calculate hedge ratio
        regression_window = pair_config["config"].get("regression_window", 60)
        
        # Initialize spread dataframe
        spread_df = pd.DataFrame(index=merged.index)
        
        # Rolling regression
        symbol1_returns = merged[symbol1].pct_change()
        symbol2_returns = merged[symbol2].pct_change()
        
        # Calculate rolling hedge ratio (with error handling)
        try:
            rolling_cov = symbol1_returns.rolling(regression_window).cov(symbol2_returns)
            rolling_var = symbol2_returns.rolling(regression_window).var()
            
            # Avoid division by zero
            rolling_var = rolling_var.replace(0, np.nan)
            
            rolling_coef = rolling_cov / rolling_var
            
            # Calculate spread using rolling hedge ratio
            spread_df['hedge_ratio'] = rolling_coef
            spread_df['spread'] = merged[symbol1] - (spread_df['hedge_ratio'] * merged[symbol2])
            
            # Calculate z-score
            spread_df['mean'] = spread_df['spread'].rolling(lookback).mean()
            spread_df['std'] = spread_df['spread'].rolling(lookback).std()
            
            # Avoid division by zero
            spread_df['std'] = spread_df['std'].replace(0, np.nan)
            
            spread_df['zscore'] = (spread_df['spread'] - spread_df['mean']) / spread_df['std']
        except Exception as e:
            logger.error(f"Error calculating rolling regression for {symbol1}-{symbol2}: {e}")
            return None
    else:
        # Use fixed hedge ratio from configuration
        hedge_ratio = pair_config.get("hedge_ratio", 1.0)
        
        # Calculate spread
        spread = merged[symbol1] - (hedge_ratio * merged[symbol2])
        
        # Calculate z-score
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        
        # Avoid division by zero
        spread_std = spread_std.replace(0, np.nan)
        
        zscore = (spread - spread_mean) / spread_std
        
        # Create spread dataframe
        spread_df = pd.DataFrame({
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'mean': spread_mean,
            'std': spread_std,
            'zscore': zscore
        })
    
    # Drop rows with NaN values
    spread_df = spread_df.dropna()
    
    return spread_df

def generate_signals(spread_df, pair_config):
    """
    Generate trading signals based on z-score and configuration.
    
    Parameters:
    -----------
    spread_df : pd.DataFrame
        DataFrame with spread and z-score
    pair_config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.Series
        Series with trading signals (1 for long, -1 for short, 0 for no position)
    """
    entry_zscore = pair_config["config"]["entry_zscore"]
    exit_zscore = pair_config["config"]["exit_zscore"]
    stop_loss_std = pair_config["config"]["stop_loss_std"]
    
    zscore = spread_df['zscore']
    
    # Initialize signals
    signals = pd.Series(0, index=zscore.index)
    position = 0
    
    # Generate signals
    for i in range(1, len(zscore)):
        # Current position
        prev_position = position
        
        # Check for stop loss
        if position != 0 and abs(zscore.iloc[i]) > stop_loss_std:
            position = 0  # Close position on stop loss
        # Check for exit
        elif position > 0 and zscore.iloc[i] <= exit_zscore:
            position = 0  # Close long position
        elif position < 0 and zscore.iloc[i] >= -exit_zscore:
            position = 0  # Close short position
        # Check for entry
        elif position == 0 and zscore.iloc[i] <= -entry_zscore:
            position = 1  # Long signal (spread is negative)
        elif position == 0 and zscore.iloc[i] >= entry_zscore:
            position = -1  # Short signal (spread is positive)
        
        # Set signal
        signals.iloc[i] = position
    
    return signals

def apply_intraday_constraints(signals, spread_df, pair_config):
    """
    Apply intraday-specific constraints to trading signals.
    
    Parameters:
    -----------
    signals : pd.Series
        Series with trading signals
    spread_df : pd.DataFrame
        DataFrame with spread and z-score
    pair_config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.Series
        Series with constrained signals
    """
    constrained_signals = signals.copy()
    
    # Check if intraday parameters are defined
    if "intraday_params" not in pair_config["config"]:
        return constrained_signals
    
    intraday_params = pair_config["config"]["intraday_params"]
    time_filters = pair_config["config"].get("time_filters", {})
    
    # Apply time filters if timestamp index is available
    if isinstance(signals.index, pd.DatetimeIndex):
        # 1. Apply maximum holding period
        max_holding_period = intraday_params.get("max_holding_period", 180)  # in minutes
        
        # Track entry timestamps
        entry_times = {}
        position = 0
        
        for i in range(len(signals)):
            timestamp = signals.index[i]
            current_signal = signals.iloc[i]
            
            # Check for new position
            if position == 0 and current_signal != 0:
                position = current_signal
                entry_times[position] = timestamp
            # Check for closed position
            elif position != 0 and current_signal == 0:
                position = 0
                entry_times = {}
            
            # Close position if holding period exceeded
            if position != 0 and entry_times.get(position) is not None:
                holding_time = timestamp - entry_times[position]
                if holding_time.total_seconds() / 60 > max_holding_period:
                    constrained_signals.iloc[i] = 0
                    position = 0
                    entry_times = {}
        
        # 2. Apply time-of-day filters
        
        # a. Avoid first 15 minutes
        if time_filters.get("avoid_first_15min", False):
            morning_mask = (signals.index.time >= time(9, 30)) & (signals.index.time < time(9, 45))
            # Don't enter new positions
            for i in range(len(signals)):
                if morning_mask[i] and signals.iloc[i-1] == 0 and signals.iloc[i] != 0:
                    constrained_signals.iloc[i] = 0
        
        # b. Avoid lunch hour
        if time_filters.get("avoid_lunch_hour", False):
            lunch_mask = (signals.index.time >= time(12, 0)) & (signals.index.time < time(13, 0))
            # Don't enter new positions
            for i in range(len(signals)):
                if lunch_mask[i] and signals.iloc[i-1] == 0 and signals.iloc[i] != 0:
                    constrained_signals.iloc[i] = 0
        
        # c. Force close before market close
        exit_buffer = intraday_params.get("exit_buffer_minutes", 15)
        close_hour = 16
        close_minute = 0
        
        # Calculate the closing time by subtracting minutes
        if exit_buffer >= 60:
            # If buffer is an hour or more
            close_hour -= exit_buffer // 60
            close_minute = 60 - (exit_buffer % 60)
            if close_minute == 60:
                close_minute = 0
            else:
                close_hour -= 1
        else:
            # If buffer is less than an hour
            if exit_buffer <= close_minute:
                close_minute -= exit_buffer
            else:
                close_hour -= 1
                close_minute = 60 - (exit_buffer - close_minute)
        
        # Ensure valid time values
        if close_hour < 0:
            close_hour = 0
        if close_minute < 0:
            close_minute = 0
        if close_minute >= 60:
            close_minute = 59
        
        close_mask = (signals.index.time >= time(close_hour, close_minute))
        constrained_signals[close_mask] = 0
        
        # 3. Apply liquidity window trading
        high_liquidity_windows = time_filters.get("high_liquidity_windows", [])
        
        if high_liquidity_windows:
            # Initialize mask for high liquidity periods
            in_liquidity_window = pd.Series(False, index=signals.index)
            
            for window in high_liquidity_windows:
                start_str = window.get("start", "09:30")
                end_str = window.get("end", "16:00")
                
                start_hour, start_minute = map(int, start_str.split(":"))
                end_hour, end_minute = map(int, end_str.split(":"))
                
                start_time = time(start_hour, start_minute)
                end_time = time(end_hour, end_minute)
                
                # Update mask for this window
                window_mask = (signals.index.time >= start_time) & (signals.index.time < end_time)
                in_liquidity_window = in_liquidity_window | window_mask
            
            # Only allow entries during high liquidity windows
            for i in range(1, len(signals)):
                if not in_liquidity_window.iloc[i] and signals.iloc[i-1] == 0 and signals.iloc[i] != 0:
                    constrained_signals.iloc[i] = 0
    
    return constrained_signals

def calculate_pnl(signals, spread_df, pair_config):
    """
    Calculate profit and loss for trading signals.
    
    Parameters:
    -----------
    signals : pd.Series
        Series with trading signals
    spread_df : pd.DataFrame
        DataFrame with spread and z-score
    pair_config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with trade performance
    """
    # Initialize PnL dataframe
    pnl_df = pd.DataFrame(index=signals.index)
    pnl_df['signal'] = signals
    pnl_df['zscore'] = spread_df['zscore']
    pnl_df['spread'] = spread_df['spread']
    
    # Calculate spread changes
    pnl_df['spread_change'] = pnl_df['spread'].diff()
    
    # Calculate trade returns (positive when long and spread narrows or short and spread widens)
    pnl_df['trade_return'] = -pnl_df['signal'].shift(1) * pnl_df['spread_change']
    
    # Calculate cumulative returns
    pnl_df['cumulative_return'] = pnl_df['trade_return'].cumsum()
    
    # Identify trades
    pnl_df['trade_start'] = (pnl_df['signal'] != 0) & (pnl_df['signal'].shift(1) == 0)
    pnl_df['trade_end'] = (pnl_df['signal'] == 0) & (pnl_df['signal'].shift(1) != 0)
    
    # Calculate trade-level metrics
    trades = []
    current_trade = None
    
    for i in range(1, len(pnl_df)):
        # Start of new trade
        if pnl_df['trade_start'].iloc[i]:
            current_trade = {
                'entry_time': pnl_df.index[i],
                'entry_price': pnl_df['spread'].iloc[i],
                'direction': 'long' if pnl_df['signal'].iloc[i] > 0 else 'short',
                'entry_zscore': pnl_df['zscore'].iloc[i]
            }
        
        # End of trade
        if pnl_df['trade_end'].iloc[i] and current_trade is not None:
            current_trade['exit_time'] = pnl_df.index[i]
            current_trade['exit_price'] = pnl_df['spread'].iloc[i]
            current_trade['exit_zscore'] = pnl_df['zscore'].iloc[i]
            
            # Calculate trade PnL
            price_change = current_trade['exit_price'] - current_trade['entry_price']
            if current_trade['direction'] == 'long':
                current_trade['pnl'] = -price_change  # Negative when spread narrows
            else:
                current_trade['pnl'] = price_change  # Positive when spread widens
            
            # Calculate holding period
            holding_period = current_trade['exit_time'] - current_trade['entry_time']
            current_trade['holding_minutes'] = holding_period.total_seconds() / 60
            
            trades.append(current_trade)
            current_trade = None
    
    # Create trades DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Calculate win rate
        trades_df['is_win'] = trades_df['pnl'] > 0
        win_rate = trades_df['is_win'].mean()
        
        # Calculate profit factor
        gross_profit = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum()
        gross_loss = abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Add to PnL dataframe
        pnl_df.loc[pnl_df.index[0], 'win_rate'] = win_rate
        pnl_df.loc[pnl_df.index[0], 'profit_factor'] = profit_factor
        pnl_df.loc[pnl_df.index[0], 'trade_count'] = len(trades_df)
        pnl_df.loc[pnl_df.index[0], 'avg_trade_pnl'] = trades_df['pnl'].mean()
        pnl_df.loc[pnl_df.index[0], 'avg_win_pnl'] = trades_df.loc[trades_df['is_win'], 'pnl'].mean()
        pnl_df.loc[pnl_df.index[0], 'avg_loss_pnl'] = trades_df.loc[~trades_df['is_win'], 'pnl'].mean()
        pnl_df.loc[pnl_df.index[0], 'avg_holding_minutes'] = trades_df['holding_minutes'].mean()
    
    return pnl_df, trades

def apply_ml_enhancements(original_signals, spread_df, price_dfs, volumes_dfs=None):
    """
    Apply ML signal enhancements to original signals.
    
    Parameters:
    -----------
    original_signals : pd.Series
        Series with original trading signals
    spread_df : pd.DataFrame
        DataFrame with spread and z-score
    price_dfs : dict
        Dictionary with price DataFrames
    volumes_dfs : dict, optional
        Dictionary with volume DataFrames
        
    Returns:
    --------
    pd.Series
        Series with enhanced signals
    pd.DataFrame
        DataFrame with enhancement metrics
    """
    # Create price DataFrame from dict
    prices_df = pd.DataFrame({symbol: df['close'] for symbol, df in price_dfs.items()})
    
    # Create volume DataFrame if available
    volumes_df = None
    if volumes_dfs is not None:
        volumes_df = pd.DataFrame({symbol: df['volume'] for symbol, df in volumes_dfs.items()})
    
    # Create signal processor
    signal_processor = IntradaySignalProcessor()
    
    # Process signals
    enhanced_signals, metrics = signal_processor.process_intraday_signals(
        prices_df, spread_df, original_signals, volumes_df
    )
    
    return enhanced_signals, metrics

def plot_intraday_backtest(spread_df, original_signals, enhanced_signals, pnl_df, metrics_df=None, pair_id=None):
    """
    Plot intraday backtest results.
    
    Parameters:
    -----------
    spread_df : pd.DataFrame
        DataFrame with spread and z-score
    original_signals : pd.Series
        Series with original trading signals
    enhanced_signals : pd.Series
        Series with enhanced signals
    pnl_df : pd.DataFrame
        DataFrame with PnL information
    metrics_df : pd.DataFrame, optional
        DataFrame with enhancement metrics
    pair_id : str, optional
        Pair identifier for the plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the plot
    """
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot spread and z-score
    ax1 = axs[0]
    ax1.plot(spread_df.index, spread_df['zscore'], label='Z-Score')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
    
    # Highlight long and short positions
    for i in range(1, len(enhanced_signals)):
        if enhanced_signals.iloc[i] > 0:  # Long position
            ax1.fill_between([enhanced_signals.index[i-1], enhanced_signals.index[i]], -4, 4, color='green', alpha=0.1)
        elif enhanced_signals.iloc[i] < 0:  # Short position
            ax1.fill_between([enhanced_signals.index[i-1], enhanced_signals.index[i]], -4, 4, color='red', alpha=0.1)
    
    # Plot original vs enhanced signals
    ax2 = axs[1]
    ax2.plot(original_signals.index, original_signals, label='Original Signals', marker='o', markersize=3, alpha=0.5)
    ax2.plot(enhanced_signals.index, enhanced_signals, label='Enhanced Signals', marker='x', markersize=4)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    ax2.legend()
    
    # Plot cumulative PnL
    ax3 = axs[2]
    ax3.plot(pnl_df.index, pnl_df['cumulative_return'], label='Cumulative PnL')
    
    # Set titles and labels
    if pair_id:
        fig.suptitle(f"Intraday Backtest Results - {pair_id}", fontsize=16)
    else:
        fig.suptitle("Intraday Backtest Results", fontsize=16)
    
    ax1.set_title("Z-Score and Positions")
    ax1.set_ylabel("Z-Score")
    ax1.legend()
    
    ax2.set_title("Trading Signals")
    ax2.set_ylabel("Signal")
    
    ax3.set_title("Cumulative PnL")
    ax3.set_ylabel("Return")
    ax3.set_xlabel("Time")
    
    # Add performance metrics as text
    if 'win_rate' in pnl_df.columns:
        win_rate = pnl_df.loc[pnl_df.index[0], 'win_rate']
        profit_factor = pnl_df.loc[pnl_df.index[0], 'profit_factor']
        trade_count = pnl_df.loc[pnl_df.index[0], 'trade_count']
        avg_holding = pnl_df.loc[pnl_df.index[0], 'avg_holding_minutes']
        
        metrics_text = (
            f"Win Rate: {win_rate:.2%}\n"
            f"Profit Factor: {profit_factor:.2f}\n"
            f"Trades: {int(trade_count)}\n"
            f"Avg Holding: {avg_holding:.1f} min"
        )
        
        ax3.annotate(metrics_text, xy=(0.02, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def main(args=None):
    """
    Main function to run intraday backtest.
    
    Parameters:
    -----------
    args : argparse.Namespace, optional
        Command line arguments
    """
    # Use command line arguments if provided
    if args and hasattr(args, 'config') and args.config:
        config_file = args.config
        logger.info(f"Using provided config: {config_file}")
    else:
        # Find most recent intraday config file
        config_dir = "data/configs"
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json') and f.startswith('intraday_config_')]
        
        if not config_files:
            logger.error("No intraday configuration files found")
            return
        
        # Sort by timestamp in filename (most recent first)
        config_files.sort(reverse=True)
        config_file = os.path.join(config_dir, config_files[0])
        
        logger.info(f"Using latest config: {config_file}")
    
    # Load intraday configuration
    intraday_config = load_intraday_config(config_file)
    
    if intraday_config is None:
        return
    
    # Get backtest parameters
    backtest_params = intraday_config["backtest_config"]
    
    # Get test period (use most recent month if not specified)
    start_date = backtest_params["data_settings"].get("start_date", "2023-12-01")
    end_date = backtest_params["data_settings"].get("end_date", "2023-12-31")
    timeframe = backtest_params["data_settings"].get("timeframe", "5min")
    
    # Use command line arguments for output dir if provided
    if args and hasattr(args, 'output_dir') and args.output_dir:
        output_dir = args.output_dir
    else:
        # Create output directory for results
        output_dir = "data/results/intraday"
    
    # Backtest each pair
    for pair in intraday_config["pairs"]:
        pair_id = pair["pair_id"]
        symbol1 = pair["symbol1"]
        symbol2 = pair["symbol2"]
        
        logger.info(f"Backtesting pair: {pair_id}")
        
        # Load intraday data
        data = load_intraday_data([symbol1, symbol2], start_date, end_date, timeframe)
        
        if not data or symbol1 not in data or symbol2 not in data:
            logger.error(f"Could not load data for pair {pair_id}")
            continue
        
        # Calculate spread
        spread_df = calculate_spread(data, pair)
        
        if spread_df is None or len(spread_df) == 0:
            logger.error(f"Could not calculate spread for pair {pair_id}")
            continue
        
        # Generate original signals
        original_signals = generate_signals(spread_df, pair)
        
        # Apply intraday constraints
        constrained_signals = apply_intraday_constraints(original_signals, spread_df, pair)
        
        # Apply ML enhancements
        enhanced_signals, metrics = apply_ml_enhancements(
            constrained_signals, spread_df, {symbol1: data[symbol1], symbol2: data[symbol2]}
        )
        
        # Calculate PnL
        pnl_df, trades = calculate_pnl(enhanced_signals, spread_df, pair)
        
        # Plot results
        fig = plot_intraday_backtest(spread_df, constrained_signals, enhanced_signals, pnl_df, metrics, pair_id)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save plot
        plot_file = os.path.join(output_dir, f"intraday_backtest_{pair_id}_{timestamp}.png")
        fig.savefig(plot_file)
        plt.close(fig)
        
        # Save results to CSV
        results_file = os.path.join(output_dir, f"intraday_results_{pair_id}_{timestamp}.csv")
        pnl_df.to_csv(results_file)
        
        # Print summary
        total_return = pnl_df['cumulative_return'].iloc[-1]
        win_rate = pnl_df.loc[pnl_df.index[0], 'win_rate'] if 'win_rate' in pnl_df.columns else 0
        profit_factor = pnl_df.loc[pnl_df.index[0], 'profit_factor'] if 'profit_factor' in pnl_df.columns else 0
        trade_count = pnl_df.loc[pnl_df.index[0], 'trade_count'] if 'trade_count' in pnl_df.columns else 0
        
        print(f"\nIntraday Backtest Results for {pair_id}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Timeframe: {timeframe}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Number of Trades: {int(trade_count)}")
        print(f"Results saved to {results_file}")
    
    print("\nIntraday backtesting completed successfully")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run intraday backtest")
    parser.add_argument("--config", help="Path to intraday configuration file")
    parser.add_argument("--output_dir", help="Output directory for results")
    args = parser.parse_args()
    main(args) 