"""
Modified script to run an intraday backtest with ML-enhanced signals.

This version handles different feature names for different pairs.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import logging
import joblib
import shutil

from src.ml_enhancements.intraday_signals import IntradaySignalProcessor, IntradaySignalEnhancer
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
        hedge_ratio = pair_config["config"].get("hedge_ratio", 1.0)
        
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
    
    logger.info(f"Calculated spread for pair {symbol1}-{symbol2}: {len(spread_df)} rows")
    
    return spread_df

def generate_signals(spread_df, pair_config):
    """
    Generate trading signals based on z-score.
    
    Parameters:
    -----------
    spread_df : pd.DataFrame
        DataFrame with spread data
    pair_config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with signals
    """
    logger.info("Generating trading signals")
    
    # Get parameters
    entry_zscore = pair_config["config"].get("entry_zscore", 2.0)
    exit_zscore = pair_config["config"].get("exit_zscore", 0.0)
    stop_loss_std = pair_config["config"].get("stop_loss_std", 3.0)
    
    zscore = spread_df['zscore']
    
    # Initialize signals DataFrame
    signals_df = pd.DataFrame(index=zscore.index)
    signals_df['zscore'] = zscore
    signals_df['signal'] = 0
    
    # Initialize position tracking
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
        signals_df.iloc[i, signals_df.columns.get_loc('signal')] = position
    
    # Add signal changes
    signals_df['signal_change'] = signals_df['signal'].diff() != 0
    signals_df['entry'] = (signals_df['signal'] != 0) & (signals_df['signal'].shift(1) == 0)
    signals_df['exit'] = (signals_df['signal'] == 0) & (signals_df['signal'].shift(1) != 0)
    
    logger.info(f"Generated trading signals: {len(signals_df)} rows")
    
    return signals_df

def apply_intraday_constraints(signals, spread_df, pair_config):
    """
    Apply intraday trading constraints to the signals.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        DataFrame with signals
    spread_df : pd.DataFrame
        DataFrame with spread data
    pair_config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with constrained signals
    """
    logger.info("Applying intraday constraints")
    
    constrained_signals = signals.copy()
    
    # Apply max holding period constraint
    max_holding_minutes = pair_config.get("max_holding_period_minutes", 180)
    
    if max_holding_minutes > 0:
        # Track position entry time
        entry_time = None
        position = 0
        
        for i in range(len(constrained_signals)):
            curr_signal = constrained_signals.iloc[i, constrained_signals.columns.get_loc('signal')]
            
            # Position entry
            if position == 0 and curr_signal != 0:
                position = curr_signal
                entry_time = constrained_signals.index[i]
            
            # Position exit
            elif position != 0 and curr_signal == 0:
                position = 0
                entry_time = None
            
            # Check if max holding period exceeded
            if position != 0 and entry_time is not None:
                current_time = constrained_signals.index[i]
                holding_minutes = (current_time - entry_time).total_seconds() / 60
                
                if holding_minutes > max_holding_minutes:
                    # Force exit
                    constrained_signals.iloc[i, constrained_signals.columns.get_loc('signal')] = 0
                    position = 0
                    entry_time = None
    
    # Apply time of day constraints
    if "time_filters" in pair_config:
        time_filters = pair_config["time_filters"]
        
        # Entry time filter
        if "entry_allowed" in time_filters:
            entry_start = time_filters["entry_allowed"].get("start", "09:30")
            entry_end = time_filters["entry_allowed"].get("end", "15:30")
            
            entry_start_time = datetime.strptime(entry_start, "%H:%M").time()
            entry_end_time = datetime.strptime(entry_end, "%H:%M").time()
            
            for i in range(1, len(constrained_signals)):
                current_time = constrained_signals.index[i].time()
                prev_signal = constrained_signals.iloc[i-1, constrained_signals.columns.get_loc('signal')]
                curr_signal = constrained_signals.iloc[i, constrained_signals.columns.get_loc('signal')]
                
                # Check if this is an entry
                is_entry = prev_signal == 0 and curr_signal != 0
                
                # Only allow entries during allowed times
                if is_entry and (current_time < entry_start_time or current_time > entry_end_time):
                    constrained_signals.iloc[i, constrained_signals.columns.get_loc('signal')] = 0
        
        # Exit time filter (force exit)
        if "exit_allowed" in time_filters:
            exit_end = time_filters["exit_allowed"].get("end", "15:59")
            exit_end_time = datetime.strptime(exit_end, "%H:%M").time()
            
            for i in range(1, len(constrained_signals)):
                current_time = constrained_signals.index[i].time()
                curr_signal = constrained_signals.iloc[i, constrained_signals.columns.get_loc('signal')]
                
                # Force exit at market close
                if current_time >= exit_end_time and curr_signal != 0:
                    constrained_signals.iloc[i, constrained_signals.columns.get_loc('signal')] = 0
    
    # Recalculate signal changes
    constrained_signals['signal_change'] = constrained_signals['signal'].diff() != 0
    constrained_signals['entry'] = (constrained_signals['signal'] != 0) & (constrained_signals['signal'].shift(1) == 0)
    constrained_signals['exit'] = (constrained_signals['signal'] == 0) & (constrained_signals['signal'].shift(1) != 0)
    
    logger.info(f"Applied intraday constraints: {len(constrained_signals)} rows")
    
    return constrained_signals

def calculate_pnl(signals, spread_df, pair_config):
    """
    Calculate P&L for the trading signals.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        DataFrame with trading signals
    spread_df : pd.DataFrame
        DataFrame with spread data
    pair_config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with P&L metrics
    pd.DataFrame
        DataFrame with individual trades
    """
    logger.info("Calculating P&L metrics")
    
    # Calculate spread changes
    signals_df = signals.copy()
    signals_df['spread'] = spread_df['spread']
    signals_df['spread_change'] = signals_df['spread'].diff()
    
    # Calculate trade returns
    signals_df['trade_return'] = -signals_df['signal'].shift(1) * signals_df['spread_change']
    signals_df['cumulative_return'] = signals_df['trade_return'].cumsum()
    
    # Identify trades
    signals_df['trade_id'] = 0
    trade_id = 0
    in_trade = False
    
    for i in range(1, len(signals_df)):
        if signals_df['entry'].iloc[i]:
            trade_id += 1
            in_trade = True
        
        if in_trade:
            signals_df.iloc[i, signals_df.columns.get_loc('trade_id')] = trade_id
        
        if signals_df['exit'].iloc[i]:
            in_trade = False
    
    # Calculate trade outcomes
    trades = []
    
    for tid in signals_df['trade_id'].unique():
        if tid == 0:
            continue
        
        trade_df = signals_df[signals_df['trade_id'] == tid]
        
        if len(trade_df) < 2:
            continue
        
        entry_idx = trade_df.index[0]
        exit_idx = trade_df.index[-1]
        
        # Get trade details
        direction = trade_df['signal'].iloc[0]
        entry_price = trade_df['spread'].iloc[0]
        exit_price = trade_df['spread'].iloc[-1]
        
        # Calculate PnL
        if direction > 0:  # Long position
            pnl = -(exit_price - entry_price)
        else:  # Short position
            pnl = exit_price - entry_price
        
        # Create trade record
        trade = {
            'trade_id': tid,
            'entry_time': entry_idx,
            'exit_time': exit_idx,
            'direction': 'long' if direction > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'is_profitable': pnl > 0,
            'holding_minutes': (exit_idx - entry_idx).total_seconds() / 60
        }
        
        trades.append(trade)
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    if len(trades_df) > 0:
        # Calculate metrics
        win_rate = trades_df['is_profitable'].mean()
        profit_factor = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum() / abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) if abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) > 0 else float('inf')
        avg_profit = trades_df.loc[trades_df['is_profitable'], 'pnl'].mean() if len(trades_df.loc[trades_df['is_profitable']]) > 0 else 0
        avg_loss = trades_df.loc[~trades_df['is_profitable'], 'pnl'].mean() if len(trades_df.loc[~trades_df['is_profitable']]) > 0 else 0
        avg_holding = trades_df['holding_minutes'].mean()
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'win_rate': [win_rate],
            'profit_factor': [profit_factor],
            'avg_profit': [avg_profit],
            'avg_loss': [avg_loss],
            'total_return': [signals_df['cumulative_return'].iloc[-1]],
            'trade_count': [len(trades_df)],
            'winning_trades': [trades_df['is_profitable'].sum()],
            'losing_trades': [(~trades_df['is_profitable']).sum()],
            'avg_holding_minutes': [avg_holding]
        })
        
        # Merge metrics with signals
        pnl_df = pd.concat([signals_df, pd.DataFrame({
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'trade_count': len(trades_df),
            'avg_holding_minutes': avg_holding
        }, index=signals_df.index)], axis=1)
    else:
        # Create empty metrics DataFrame
        metrics_df = pd.DataFrame({
            'win_rate': [0],
            'profit_factor': [0],
            'avg_profit': [0],
            'avg_loss': [0],
            'total_return': [0],
            'trade_count': [0],
            'winning_trades': [0],
            'losing_trades': [0],
            'avg_holding_minutes': [0]
        })
        
        # Add empty metrics to signals
        pnl_df = pd.concat([signals_df, pd.DataFrame({
            'win_rate': 0,
            'profit_factor': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'trade_count': 0,
            'avg_holding_minutes': 0
        }, index=signals_df.index)], axis=1)
    
    logger.info(f"Calculated P&L: {len(trades_df)} trades")
    
    return pnl_df, trades_df

def apply_ml_enhancements(original_signals, spread_df, price_dfs, pair_id, volumes_dfs=None):
    """
    Apply ML enhancements to the original signals.
    
    Parameters:
    -----------
    original_signals : pd.DataFrame
        DataFrame with original trading signals
    spread_df : pd.DataFrame
        DataFrame with spread data
    price_dfs : dict
        Dictionary with price DataFrames for each symbol
    pair_id : str
        Pair identifier (e.g., "ES_NQ")
    volumes_dfs : dict, optional
        Dictionary with volume DataFrames for each symbol
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with enhanced signals
    pd.DataFrame
        DataFrame with signal metrics
    """
    # Create a models directory specific to this pair
    pair_model_dir = f"models/intraday_{pair_id}"
    base_model_dir = "models/intraday"
    
    # If the pair-specific directory doesn't exist, copy the base models to it
    if not os.path.exists(pair_model_dir):
        os.makedirs(pair_model_dir, exist_ok=True)
        
        # Copy all model files from the base directory to the pair-specific directory
        for model_file in os.listdir(base_model_dir):
            source_path = os.path.join(base_model_dir, model_file)
            dest_path = os.path.join(pair_model_dir, model_file)
            
            if os.path.isfile(source_path):
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied model file {model_file} to {pair_model_dir}")
    
    # Create a signal processor with the pair-specific model directory
    signal_processor = IntradaySignalProcessor({
        "models_dir": pair_model_dir,
        "feature_lookback": 20,
        "min_training_samples": 500,
        "prediction_threshold": 0.6,
        "retrain_frequency": "weekly",
        "use_rsi_filter": True,
        "use_volume_filter": True,
        "use_volatility_filter": True,
        "enable_ml_filtering": True,
        "enable_ml_timing": True,
        "enable_ml_adaptation": True
    })
    
    # Load models
    signal_processor.signal_enhancer.load_models()
    
    # Convert price_dfs dictionary to DataFrame
    prices_df = pd.DataFrame({symbol: df['close'] for symbol, df in price_dfs.items()})
    
    # Convert volumes_dfs dictionary to DataFrame if provided
    if volumes_dfs:
        volumes_df = pd.DataFrame({symbol: df['volume'] for symbol, df in volumes_dfs.items() if 'volume' in df.columns})
    else:
        volumes_df = None
    
    # Process signals
    enhanced_signals, metrics = signal_processor.process_intraday_signals(
        prices_df, spread_df, original_signals, volumes_df
    )
    
    logger.info(f"Applied ML enhancements with models from {pair_model_dir}")
    
    return enhanced_signals, metrics

def plot_intraday_backtest(spread_df, original_signals, enhanced_signals, pnl_df, metrics_df=None, pair_id=None):
    """
    Plot the results of an intraday backtest.
    
    Parameters:
    -----------
    spread_df : pd.DataFrame
        DataFrame with spread data
    original_signals : pd.DataFrame
        DataFrame with original trading signals
    enhanced_signals : pd.DataFrame or pd.Series
        DataFrame or Series with ML-enhanced trading signals
    pnl_df : pd.DataFrame
        DataFrame with P&L metrics
    metrics_df : pd.DataFrame, optional
        DataFrame with signal metrics
    pair_id : str, optional
        Pair identifier
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with plots
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot z-score and signals
    ax1.plot(spread_df.index, spread_df['zscore'], label='Z-Score', alpha=0.7)
    
    # Plot entry/exit points for original signals
    if 'entry' in original_signals.columns:
        entries_long = original_signals.index[original_signals['entry'] & (original_signals['signal'] > 0)]
        entries_short = original_signals.index[original_signals['entry'] & (original_signals['signal'] < 0)]
        exits = original_signals.index[original_signals['exit']]
    else:
        # Detect entries/exits by signal changes
        entries_long = []
        entries_short = []
        exits = []
        
        # Process each signal change
        for i in range(1, len(original_signals)):
            prev_signal = original_signals.iloc[i-1]
            curr_signal = original_signals.iloc[i]
            
            # Detect entries
            if prev_signal == 0 and curr_signal > 0:
                entries_long.append(original_signals.index[i])
            elif prev_signal == 0 and curr_signal < 0:
                entries_short.append(original_signals.index[i])
            # Detect exits
            elif prev_signal != 0 and curr_signal == 0:
                exits.append(original_signals.index[i])
    
    if len(entries_long) > 0:
        ax1.scatter(entries_long, spread_df.loc[entries_long, 'zscore'], marker='^', color='green', s=100, label='Long Entry (Original)', alpha=0.7)
    if len(entries_short) > 0:
        ax1.scatter(entries_short, spread_df.loc[entries_short, 'zscore'], marker='v', color='red', s=100, label='Short Entry (Original)', alpha=0.7)
    if len(exits) > 0:
        ax1.scatter(exits, spread_df.loc[exits, 'zscore'], marker='X', color='black', s=80, label='Exit (Original)', alpha=0.7)
    
    # Plot entry/exit points for enhanced signals
    if enhanced_signals is not None:
        # Similar processing for enhanced signals
        if isinstance(enhanced_signals, pd.DataFrame) and 'entry' in enhanced_signals.columns:
            entries_long = enhanced_signals.index[enhanced_signals['entry'] & (enhanced_signals['signal'] > 0)]
            entries_short = enhanced_signals.index[enhanced_signals['entry'] & (enhanced_signals['signal'] < 0)]
            exits = enhanced_signals.index[enhanced_signals['exit']]
        else:
            # Detect entries/exits by signal changes
            entries_long = []
            entries_short = []
            exits = []
            
            # Process each signal change
            for i in range(1, len(enhanced_signals)):
                prev_signal = enhanced_signals.iloc[i-1]
                curr_signal = enhanced_signals.iloc[i]
                
                # Detect entries
                if prev_signal == 0 and curr_signal > 0:
                    entries_long.append(enhanced_signals.index[i])
                elif prev_signal == 0 and curr_signal < 0:
                    entries_short.append(enhanced_signals.index[i])
                # Detect exits
                elif prev_signal != 0 and curr_signal == 0:
                    exits.append(enhanced_signals.index[i])
        
        if len(entries_long) > 0:
            ax1.scatter(entries_long, spread_df.loc[entries_long, 'zscore'], marker='^', color='blue', s=80, label='Long Entry (ML)', alpha=0.8)
        if len(entries_short) > 0:
            ax1.scatter(entries_short, spread_df.loc[entries_short, 'zscore'], marker='v', color='purple', s=80, label='Short Entry (ML)', alpha=0.8)
        if len(exits) > 0:
            ax1.scatter(exits, spread_df.loc[exits, 'zscore'], marker='x', color='orange', s=60, label='Exit (ML)', alpha=0.8)
    
    # Add threshold lines
    entry_threshold = 2.0  # Default value
    exit_threshold = 0.5   # Default value
    
    ax1.axhline(y=entry_threshold, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=-entry_threshold, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=exit_threshold, color='black', linestyle=':', alpha=0.5)
    ax1.axhline(y=-exit_threshold, color='black', linestyle=':', alpha=0.5)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax1.set_title(f'Z-Score and Trading Signals for {pair_id}' if pair_id else 'Z-Score and Trading Signals')
    ax1.set_ylabel('Z-Score')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot spread
    ax2.plot(spread_df.index, spread_df['spread'], label='Spread', color='blue')
    
    # Plot spread mean and std bands
    ax2.plot(spread_df.index, spread_df['mean'], label='Mean', color='orange', alpha=0.7)
    ax2.fill_between(spread_df.index, 
                    spread_df['mean'] - 2*spread_df['std'], 
                    spread_df['mean'] + 2*spread_df['std'], 
                    color='orange', alpha=0.2, label='±2σ')
    
    # Plot entry/exit points for enhanced signals on spread
    if enhanced_signals is not None and len(entries_long) + len(entries_short) + len(exits) > 0:
        if len(entries_long) > 0:
            ax2.scatter(entries_long, spread_df.loc[entries_long, 'spread'], marker='^', color='green', s=80, label='Long Entry')
        if len(entries_short) > 0:
            ax2.scatter(entries_short, spread_df.loc[entries_short, 'spread'], marker='v', color='red', s=80, label='Short Entry')
        if len(exits) > 0:
            ax2.scatter(exits, spread_df.loc[exits, 'spread'], marker='x', color='black', s=60, label='Exit')
    
    ax2.set_title('Spread with Mean and Bands')
    ax2.set_ylabel('Spread')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot cumulative returns for enhanced signals
    if 'cumulative_return' in pnl_df.columns:
        ax3.plot(pnl_df.index, pnl_df['cumulative_return'], label='Cumulative Return', color='green')
        ax3.set_title('Cumulative Returns')
        ax3.set_ylabel('Return')
        ax3.grid(True, alpha=0.3)
        
        # Add metrics as annotation
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

def main():
    """Main function to run intraday backtest."""
    # Find most recent intraday config file
    config_dir = "data/configs"
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json') and f.startswith('intraday_config_')]
    
    if not config_files:
        logger.error("No intraday configuration files found")
        return
    
    # Sort by timestamp in filename (most recent first)
    config_files.sort(reverse=True)
    latest_config = os.path.join(config_dir, config_files[0])
    
    logger.info(f"Using latest config: {latest_config}")
    
    # Load intraday configuration
    intraday_config = load_intraday_config(latest_config)
    
    if intraday_config is None:
        return
    
    # Get backtest parameters
    backtest_params = intraday_config["backtest_config"]
    
    # Get test period (use most recent month if not specified)
    start_date = backtest_params["data_settings"].get("start_date", "2023-12-01")
    end_date = backtest_params["data_settings"].get("end_date", "2023-12-31")
    timeframe = backtest_params["data_settings"].get("timeframe", "5min")
    
    # Create output directory for results
    output_dir = "data/results/intraday"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the specific pairs to test
    test_pairs = []
    
    # Add ES_NQ pair
    es_nq_pair = {
        "pair_id": "ES_NQ",
        "symbol1": "ES",
        "symbol2": "NQ",
        "config": {
            "lookback": 50,
            "entry_zscore": 2.0,
            "exit_zscore": 0.0,
            "stop_loss_std": 3.0,
            "hedge_ratio": 1.0,
            "use_rolling_regression": True,
            "regression_window": 100
        }
    }
    test_pairs.append(es_nq_pair)
    
    # Add GC_SI pair
    gc_si_pair = {
        "pair_id": "GC_SI",
        "symbol1": "GC",
        "symbol2": "SI",
        "config": {
            "lookback": 50,
            "entry_zscore": 2.0,
            "exit_zscore": 0.0,
            "stop_loss_std": 3.0,
            "hedge_ratio": 1.0,
            "use_rolling_regression": True,
            "regression_window": 100
        }
    }
    test_pairs.append(gc_si_pair)
    
    # Backtest each pair
    for pair in test_pairs:
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
        
        logger.info(f"Generated trading signals: {len(original_signals)} rows")
        
        # Apply intraday constraints
        constrained_signals = apply_intraday_constraints(original_signals, spread_df, pair)
        
        logger.info(f"Applied intraday constraints: {len(constrained_signals)} rows")
        
        # Apply ML enhancements using pair-specific approach
        enhanced_signals, metrics = apply_ml_enhancements(
            constrained_signals, spread_df, {symbol1: data[symbol1], symbol2: data[symbol2]}, pair_id
        )
        
        # Calculate PnL
        pnl_df, trades = calculate_pnl(enhanced_signals, spread_df, pair)
        
        logger.info(f"Calculated P&L: {len(trades)} trades")
        
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
    main() 