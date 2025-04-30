"""
Script to train machine learning models for intraday signal enhancement.

This script loads historical data, calculates features, and trains ML models
to enhance trading signals for intraday pairs trading.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import joblib
from tqdm import tqdm

from src.ml_enhancements.intraday_signals import IntradaySignalProcessor
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_pair_data(pair_id, start_date, end_date, timeframe="5min", data_dir="data/processed"):
    """
    Load intraday data for a pair.
    
    Parameters:
    -----------
    pair_id : str
        Pair identifier (e.g. "BFX_ZN")
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
        Dictionary with data for each symbol and the spread
    """
    logger.info(f"Loading data for pair {pair_id} from {start_date} to {end_date}")
    
    # Split pair_id to get symbols
    try:
        symbol1, symbol2 = pair_id.split('_')
    except ValueError:
        logger.error(f"Invalid pair_id format: {pair_id}")
        return None
    
    # Load data for each symbol
    data = {}
    
    for symbol in [symbol1, symbol2]:
        try:
            file_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
            
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                return None
            
            # Load data
            df = pd.read_parquet(file_path)
            
            # Ensure data is sorted by timestamp
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    logger.warning(f"No timestamp column found for {symbol}")
                    return None
            
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
            return None
    
    # Check if we have data for both symbols
    if symbol1 not in data or symbol2 not in data:
        logger.error(f"Missing data for one or both symbols in pair {pair_id}")
        return None
    
    # Create price DataFrame
    prices_df = pd.DataFrame({
        symbol1: data[symbol1]['close'],
        symbol2: data[symbol2]['close']
    })
    
    # Create volume DataFrame if available
    volumes_df = None
    if 'volume' in data[symbol1].columns and 'volume' in data[symbol2].columns:
        volumes_df = pd.DataFrame({
            symbol1: data[symbol1]['volume'],
            symbol2: data[symbol2]['volume']
        })
    
    return {
        'prices': prices_df,
        'volumes': volumes_df,
        'data': data
    }

def calculate_spread(prices_df, symbol1, symbol2, config):
    """
    Calculate spread for a pair.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data
    symbol1 : str
        First symbol in the pair
    symbol2 : str
        Second symbol in the pair
    config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with spread data
    """
    logger.info(f"Calculating spread for pair {symbol1}_{symbol2}")
    
    # Ensure we have data for both symbols
    if symbol1 not in prices_df.columns or symbol2 not in prices_df.columns:
        logger.error(f"Missing price data for one or both symbols in pair {symbol1}_{symbol2}")
        return None
    
    # Extract price data
    symbol1_prices = prices_df[symbol1]
    symbol2_prices = prices_df[symbol2]
    
    # Get lookback parameter
    lookback = config.get("lookback", 50)
    
    # Calculate spread based on configuration
    if config.get("use_rolling_regression", False):
        # Use rolling regression to calculate hedge ratio
        regression_window = config.get("regression_window", 60)
        
        # Initialize spread dataframe
        spread_df = pd.DataFrame(index=prices_df.index)
        
        # Calculate returns
        symbol1_returns = symbol1_prices.pct_change()
        symbol2_returns = symbol2_prices.pct_change()
        
        # Calculate rolling hedge ratio
        try:
            rolling_cov = symbol1_returns.rolling(regression_window).cov(symbol2_returns)
            rolling_var = symbol2_returns.rolling(regression_window).var()
            
            # Avoid division by zero
            rolling_var = rolling_var.replace(0, np.nan)
            
            rolling_coef = rolling_cov / rolling_var
            
            # Calculate spread using rolling hedge ratio
            spread_df['hedge_ratio'] = rolling_coef
            spread_df['spread'] = symbol1_prices - (spread_df['hedge_ratio'] * symbol2_prices)
            
            # Calculate z-score
            spread_df['mean'] = spread_df['spread'].rolling(lookback).mean()
            spread_df['std'] = spread_df['spread'].rolling(lookback).std()
            
            # Avoid division by zero
            spread_df['std'] = spread_df['std'].replace(0, np.nan)
            
            spread_df['zscore'] = (spread_df['spread'] - spread_df['mean']) / spread_df['std']
        except Exception as e:
            logger.error(f"Error calculating spread for {symbol1}_{symbol2}: {e}")
            return None
    else:
        # Use fixed hedge ratio from configuration
        hedge_ratio = config.get("hedge_ratio", 1.0)
        
        # Calculate spread
        spread = symbol1_prices - (hedge_ratio * symbol2_prices)
        
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
    
    logger.info(f"Calculated spread for pair {symbol1}_{symbol2}: {len(spread_df)} rows")
    
    return spread_df

def generate_signals(spread_df, config):
    """
    Generate trading signals based on z-score.
    
    Parameters:
    -----------
    spread_df : pd.DataFrame
        DataFrame with spread data
    config : dict
        Configuration for the pair
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with signals
    """
    logger.info("Generating trading signals")
    
    # Get parameters
    entry_zscore = config.get("entry_zscore", 2.0)
    exit_zscore = config.get("exit_zscore", 0.0)
    stop_loss_std = config.get("stop_loss_std", 3.0)
    
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
    
    # Calculate PnL
    signals_df['spread_change'] = spread_df['spread'].diff()
    signals_df['trade_return'] = -signals_df['signal'].shift(1) * signals_df['spread_change']
    signals_df['cumulative_return'] = signals_df['trade_return'].cumsum()
    
    # Identify trade outcomes
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
        entry_price = spread_df.loc[entry_idx, 'spread']
        exit_price = spread_df.loc[exit_idx, 'spread']
        
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
    
    logger.info(f"Generated {len(trades_df)} trades")
    
    return signals_df, trades_df 

def train_models_for_pair(pair_id, pair_config, start_date, end_date, timeframe="5min"):
    """
    Train ML models for a specific pair.
    
    Parameters:
    -----------
    pair_id : str
        Pair identifier (e.g. "BFX_ZN")
    pair_config : dict
        Configuration for the pair
    start_date : str
        Start date for training data (YYYY-MM-DD)
    end_date : str
        End date for training data (YYYY-MM-DD)
    timeframe : str
        Timeframe to use for training
        
    Returns:
    --------
    dict
        Training results and metrics
    """
    logger.info(f"Training ML models for pair {pair_id}")
    
    # Load data
    pair_data = load_pair_data(pair_id, start_date, end_date, timeframe)
    
    if pair_data is None:
        logger.error(f"Failed to load data for pair {pair_id}")
        return None
    
    # Extract symbols from pair_id
    symbol1, symbol2 = pair_id.split('_')
    
    # Calculate spread
    spread_df = calculate_spread(
        pair_data['prices'], 
        symbol1, 
        symbol2, 
        pair_config['config']
    )
    
    if spread_df is None or len(spread_df) == 0:
        logger.error(f"Failed to calculate spread for pair {pair_id}")
        return None
    
    # Generate signals
    signals_df, trades_df = generate_signals(spread_df, pair_config['config'])
    
    if len(signals_df) == 0:
        logger.error(f"No signals generated for pair {pair_id}")
        return None
    
    if len(trades_df) == 0:
        logger.warning(f"No trades generated for pair {pair_id}")
    
    # Create performance DataFrame
    performance_df = pd.DataFrame(index=signals_df.index)
    
    # For each trade, mark the entry and exit points
    for _, trade in trades_df.iterrows():
        trade_id = trade['trade_id']
        is_profitable = trade['is_profitable']
        
        # Mark trade PnL
        mask = signals_df['trade_id'] == trade_id
        performance_df.loc[mask, 'trade_id'] = trade_id
        performance_df.loc[mask, 'trade_pnl'] = trade['pnl']
        performance_df.loc[mask, 'is_profitable'] = is_profitable
    
    # Create signal processor
    processor = IntradaySignalProcessor()
    
    # Train ML models
    training_results = processor.train_models(
        pair_data['prices'],
        spread_df,
        signals_df,
        performance_df,
        pair_data['volumes']
    )
    
    # Calculate training metrics
    metrics = {}
    
    if len(trades_df) > 0:
        win_rate = trades_df['is_profitable'].mean()
        profit_factor = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum() / abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) if abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) > 0 else float('inf')
        avg_profit = trades_df.loc[trades_df['is_profitable'], 'pnl'].mean() if len(trades_df.loc[trades_df['is_profitable']]) > 0 else 0
        avg_loss = trades_df.loc[~trades_df['is_profitable'], 'pnl'].mean() if len(trades_df.loc[~trades_df['is_profitable']]) > 0 else 0
        
        metrics = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'num_trades': len(trades_df),
            'num_winning_trades': trades_df['is_profitable'].sum(),
            'num_losing_trades': (~trades_df['is_profitable']).sum()
        }
    
    # Combine results
    results = {
        'pair_id': pair_id,
        'training_period': {
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe
        },
        'training_results': training_results,
        'metrics': metrics,
        'model_metrics': processor.signal_enhancer.model_metrics
    }
    
    return results

def create_train_test_periods(start_date, end_date, test_ratio=0.3):
    """
    Create train and test periods for model training and evaluation.
    
    Parameters:
    -----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    test_ratio : float
        Ratio of data to use for testing (0.0-1.0)
        
    Returns:
    --------
    dict
        Dictionary with train and test periods
    """
    # Convert to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Calculate total days
    total_days = (end_dt - start_dt).days
    
    # Calculate test period
    test_days = int(total_days * test_ratio)
    
    # Calculate split date
    split_dt = end_dt - timedelta(days=test_days)
    
    return {
        'train': {
            'start': start_dt.strftime('%Y-%m-%d'),
            'end': split_dt.strftime('%Y-%m-%d')
        },
        'test': {
            'start': split_dt.strftime('%Y-%m-%d'),
            'end': end_dt.strftime('%Y-%m-%d')
        }
    }

def load_portfolio_config(config_file):
    """
    Load portfolio configuration from a JSON file.
    
    Parameters:
    -----------
    config_file : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Portfolio configuration
    """
    logger.info(f"Loading portfolio configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration with {len(config['pairs'])} pairs")
        return config
    except Exception as e:
        logger.error(f"Error loading portfolio configuration: {e}")
        return None

def save_training_results(results, output_dir="data/results/models"):
    """
    Save training results to a JSON file.
    
    Parameters:
    -----------
    results : dict
        Training results
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    str
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"training_results_{timestamp}.json")
    
    # Convert non-serializable objects
    cleaned_results = {}
    
    for pair_id, pair_results in results.items():
        cleaned_pair = {}
        
        # Process each key in pair_results
        for key, value in pair_results.items():
            if isinstance(value, dict):
                # Recursively clean nested dictionaries
                cleaned_value = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        # Handle nested-nested dictionaries
                        cleaned_nested = {}
                        for nk, nv in v.items():
                            cleaned_nested[nk] = convert_to_serializable(nv)
                        cleaned_value[k] = cleaned_nested
                    else:
                        cleaned_value[k] = convert_to_serializable(v)
                cleaned_pair[key] = cleaned_value
            else:
                cleaned_pair[key] = convert_to_serializable(value)
        
        cleaned_results[pair_id] = cleaned_pair
    
    with open(output_file, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    logger.info(f"Saved training results to {output_file}")
    
    return output_file

def convert_to_serializable(obj):
    """Convert an object to a JSON serializable type."""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (float, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        try:
            # Check if it's JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # If not, convert to string
            return str(obj)

def plot_training_results(results, output_dir="data/results/models"):
    """
    Create visualizations of training results.
    
    Parameters:
    -----------
    results : dict
        Training results
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    list
        List of saved plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_files = []
    
    # Create model metrics comparison plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract metrics
    pairs = list(results.keys())
    
    win_rates = [results[pair]['metrics'].get('win_rate', 0) for pair in pairs]
    profit_factors = [results[pair]['metrics'].get('profit_factor', 0) for pair in pairs]
    num_trades = [results[pair]['metrics'].get('num_trades', 0) for pair in pairs]
    
    # Plotting
    axs[0, 0].bar(pairs, win_rates)
    axs[0, 0].set_title('Win Rate')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_ylabel('Win Rate')
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].bar(pairs, profit_factors)
    axs[0, 1].set_title('Profit Factor')
    axs[0, 1].set_ylabel('Profit Factor')
    axs[0, 1].grid(True, alpha=0.3)
    
    axs[1, 0].bar(pairs, num_trades)
    axs[1, 0].set_title('Number of Trades')
    axs[1, 0].set_ylabel('Number of Trades')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Model performance metrics
    model_types = []
    accuracies = []
    precisions = []
    recalls = []
    
    for pair in pairs:
        if 'model_metrics' in results[pair]:
            for model, metrics in results[pair]['model_metrics'].items():
                if 'accuracy' in metrics:
                    model_types.append(f"{pair} - {model}")
                    accuracies.append(metrics.get('accuracy', 0))
                    precisions.append(metrics.get('precision', 0))
                    recalls.append(metrics.get('recall', 0))
    
    if model_types:
        axs[1, 1].bar(range(len(model_types)), accuracies, label='Accuracy')
        axs[1, 1].bar(range(len(model_types)), precisions, label='Precision', alpha=0.7)
        axs[1, 1].bar(range(len(model_types)), recalls, label='Recall', alpha=0.4)
        axs[1, 1].set_xticks(range(len(model_types)))
        axs[1, 1].set_xticklabels(model_types, rotation=90)
        axs[1, 1].set_title('Model Performance')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    metrics_file = os.path.join(output_dir, f"training_metrics_{timestamp}.png")
    plt.savefig(metrics_file)
    plt.close()
    
    plot_files.append(metrics_file)
    
    return plot_files

def main():
    """Main function to train ML models for pairs trading."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ML models for intraday pairs trading")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="data/configs/intraday_config_latest.json",
        help="Path to intraday configuration file"
    )
    
    parser.add_argument(
        "--start_date", 
        type=str, 
        default="2023-01-01",
        help="Start date for training data (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end_date", 
        type=str, 
        default="2023-12-31",
        help="End date for training data (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--timeframe", 
        type=str, 
        default="5min",
        help="Timeframe to use for training"
    )
    
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.3,
        help="Ratio of data to use for testing (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Find the configuration file
    config_file = args.config
    
    if config_file == "data/configs/intraday_config_latest.json":
        # Find the latest configuration file
        config_dir = "data/configs"
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json') and f.startswith('intraday_config_')]
        
        if not config_files:
            logger.error("No intraday configuration files found")
            return
        
        # Sort by timestamp in filename (most recent first)
        config_files.sort(reverse=True)
        config_file = os.path.join(config_dir, config_files[0])
    
    logger.info(f"Using configuration file: {config_file}")
    
    # Load configuration
    config = load_portfolio_config(config_file)
    
    if config is None:
        return
    
    # Create train/test periods
    periods = create_train_test_periods(args.start_date, args.end_date, args.test_ratio)
    
    logger.info(f"Training period: {periods['train']['start']} to {periods['train']['end']}")
    logger.info(f"Testing period: {periods['test']['start']} to {periods['test']['end']}")
    
    # Train models for each pair
    results = {}
    
    for pair in tqdm(config['pairs'], desc="Training models for pairs"):
        # Create pair_id from leg1 and leg2 if not present
        if 'pair_id' not in pair:
            if 'leg1' in pair and 'leg2' in pair:
                pair_id = f"{pair['leg1']}_{pair['leg2']}"
                pair['pair_id'] = pair_id
            else:
                logger.error(f"Pair configuration missing required fields: {pair}")
                continue
        else:
            pair_id = pair['pair_id']
        
        # Train models
        pair_results = train_models_for_pair(
            pair_id,
            pair,
            periods['train']['start'],
            periods['train']['end'],
            args.timeframe
        )
        
        if pair_results:
            results[pair_id] = pair_results
            logger.info(f"Completed training for pair {pair_id}")
        else:
            logger.warning(f"Failed to train models for pair {pair_id}")
    
    # Save results
    if results:
        save_training_results(results)
        plot_training_results(results)
        
        # Print summary
        print("\nTraining Summary:")
        print(f"Configuration: {os.path.basename(config_file)}")
        print(f"Period: {periods['train']['start']} to {periods['train']['end']} (training)")
        print(f"Timeframe: {args.timeframe}")
        print(f"Pairs trained: {len(results)}/{len(config['pairs'])}")
        
        for pair_id, pair_results in results.items():
            metrics = pair_results.get('metrics', {})
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            num_trades = metrics.get('num_trades', 0)
            
            print(f"\n{pair_id}:")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Trades: {num_trades}")
            
            if 'model_metrics' in pair_results:
                print("  Model Metrics:")
                for model, model_metrics in pair_results['model_metrics'].items():
                    if 'accuracy' in model_metrics:
                        print(f"    {model}: Accuracy={model_metrics.get('accuracy', 0):.4f}, Precision={model_metrics.get('precision', 0):.4f}, Recall={model_metrics.get('recall', 0):.4f}")
                    elif 'rmse' in model_metrics:
                        print(f"    {model}: RMSE={model_metrics.get('rmse', 0):.4f}")
    else:
        logger.warning("No models were successfully trained")

if __name__ == "__main__":
    main() 