"""
Script to evaluate trained machine learning models for intraday signal enhancement.

This script loads trained ML models, applies them to test data, and evaluates
the performance of the enhanced signals compared to the original signals.
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
import argparse

from src.ml_enhancements.intraday_signals import IntradaySignalProcessor
from train_intraday_models import (
    load_pair_data,
    calculate_spread,
    generate_signals,
    load_portfolio_config,
    create_train_test_periods
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_pair(pair_id, pair_config, start_date, end_date, timeframe="5min"):
    """
    Evaluate ML models for a specific pair on test data.
    
    Parameters:
    -----------
    pair_id : str
        Pair identifier (e.g. "BFX_ZN")
    pair_config : dict
        Configuration for the pair
    start_date : str
        Start date for test data (YYYY-MM-DD)
    end_date : str
        End date for test data (YYYY-MM-DD)
    timeframe : str
        Timeframe to use for evaluation
        
    Returns:
    --------
    dict
        Evaluation results and metrics
    """
    logger.info(f"Evaluating ML models for pair {pair_id}")
    
    # Load test data
    pair_data = load_pair_data(pair_id, start_date, end_date, timeframe)
    
    if pair_data is None:
        logger.error(f"Failed to load test data for pair {pair_id}")
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
    
    # Generate original signals
    signals_df, trades_df = generate_signals(spread_df, pair_config['config'])
    
    if len(signals_df) == 0:
        logger.error(f"No signals generated for pair {pair_id}")
        return None
    
    if len(trades_df) == 0:
        logger.warning(f"No trades generated for pair {pair_id}")
    
    # Calculate original performance metrics
    original_metrics = calculate_performance_metrics(signals_df, trades_df)
    
    # Create signal processor
    processor = IntradaySignalProcessor()
    
    # Load pre-trained models
    if not processor.signal_enhancer.load_models():
        logger.warning(f"Failed to load pre-trained models for pair {pair_id}")
    
    # Apply ML enhancements
    enhanced_signals, metrics = processor.process_intraday_signals(
        pair_data['prices'],
        spread_df,
        signals_df['signal'],
        pair_data['volumes']
    )
    
    # Reprocess signals to calculate PnL and trades
    enhanced_signals_df = signals_df.copy()
    enhanced_signals_df['signal'] = enhanced_signals
    
    # Calculate PnL
    enhanced_signals_df['trade_return'] = -enhanced_signals_df['signal'].shift(1) * enhanced_signals_df['spread_change']
    enhanced_signals_df['cumulative_return'] = enhanced_signals_df['trade_return'].cumsum()
    
    # Identify trade outcomes
    enhanced_signals_df['trade_id'] = 0
    enhanced_signals_df['entry'] = (enhanced_signals_df['signal'] != 0) & (enhanced_signals_df['signal'].shift(1) == 0)
    enhanced_signals_df['exit'] = (enhanced_signals_df['signal'] == 0) & (enhanced_signals_df['signal'].shift(1) != 0)
    
    # Compute enhanced trades
    enhanced_trades = extract_trades(enhanced_signals_df, spread_df)
    enhanced_metrics = calculate_performance_metrics(enhanced_signals_df, enhanced_trades)
    
    # Compare original vs enhanced
    comparison = {
        'original': original_metrics,
        'enhanced': enhanced_metrics,
        'improvement': {
            'win_rate': enhanced_metrics.get('win_rate', 0) - original_metrics.get('win_rate', 0),
            'profit_factor': enhanced_metrics.get('profit_factor', 0) - original_metrics.get('profit_factor', 0),
            'total_return': enhanced_metrics.get('total_return', 0) - original_metrics.get('total_return', 0),
            'max_drawdown': enhanced_metrics.get('max_drawdown', 0) - original_metrics.get('max_drawdown', 0),
            'sharpe_ratio': enhanced_metrics.get('sharpe_ratio', 0) - original_metrics.get('sharpe_ratio', 0)
        }
    }
    
    # Combine results
    results = {
        'pair_id': pair_id,
        'test_period': {
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe
        },
        'comparison': comparison,
        'signal_metrics': {
            'signal_changes': (enhanced_signals != signals_df['signal']).sum(),
            'total_signals': len(signals_df),
            'change_percentage': (enhanced_signals != signals_df['signal']).sum() / len(signals_df) if len(signals_df) > 0 else 0
        }
    }
    
    return results, signals_df, enhanced_signals_df, spread_df

def extract_trades(signals_df, spread_df):
    """
    Extract trades from signals.
    
    Parameters:
    -----------
    signals_df : pd.DataFrame
        DataFrame with signals
    spread_df : pd.DataFrame
        DataFrame with spread data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with trades
    """
    # Identify trade outcomes
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
    
    return trades_df

def calculate_performance_metrics(signals_df, trades_df):
    """
    Calculate performance metrics for a trading strategy.
    
    Parameters:
    -----------
    signals_df : pd.DataFrame
        DataFrame with signals
    trades_df : pd.DataFrame
        DataFrame with trades
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    metrics = {}
    
    # Trade-based metrics
    if len(trades_df) > 0:
        win_rate = trades_df['is_profitable'].mean()
        profit_factor = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum() / abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) if abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) > 0 else float('inf')
        avg_profit = trades_df.loc[trades_df['is_profitable'], 'pnl'].mean() if len(trades_df.loc[trades_df['is_profitable']]) > 0 else 0
        avg_loss = trades_df.loc[~trades_df['is_profitable'], 'pnl'].mean() if len(trades_df.loc[~trades_df['is_profitable']]) > 0 else 0
        
        metrics.update({
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'num_trades': len(trades_df),
            'num_winning_trades': trades_df['is_profitable'].sum(),
            'num_losing_trades': (~trades_df['is_profitable']).sum()
        })
    
    # PnL-based metrics
    if 'cumulative_return' in signals_df.columns and len(signals_df) > 0:
        total_return = signals_df['cumulative_return'].iloc[-1]
        
        # Calculate daily returns
        if isinstance(signals_df.index, pd.DatetimeIndex):
            daily_returns = signals_df['trade_return'].resample('D').sum()
        else:
            daily_returns = signals_df['trade_return']
        
        # Calculate Sharpe ratio
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = signals_df['cumulative_return']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics.update({
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': total_return / ((signals_df.index[-1] - signals_df.index[0]).days / 365) if isinstance(signals_df.index, pd.DatetimeIndex) else 0
        })
    
    return metrics 

def plot_pair_evaluation(pair_id, signals_df, enhanced_signals_df, spread_df, results, output_dir="data/results/evaluation"):
    """
    Create visualizations for pair evaluation.
    
    Parameters:
    -----------
    pair_id : str
        Pair identifier
    signals_df : pd.DataFrame
        DataFrame with original signals
    enhanced_signals_df : pd.DataFrame
        DataFrame with enhanced signals
    spread_df : pd.DataFrame
        DataFrame with spread data
    results : dict
        Evaluation results
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    str
        Path to the saved plot file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # 1. Plot spread and z-score
    axs[0].plot(spread_df.index, spread_df['zscore'], label='Z-Score')
    axs[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axs[0].axhline(y=2, color='red', linestyle='--', alpha=0.5)
    axs[0].axhline(y=-2, color='green', linestyle='--', alpha=0.5)
    
    # Add signals as shaded regions
    for i in range(1, len(enhanced_signals_df)):
        # Original signal background
        if signals_df['signal'].iloc[i] > 0:  # Long
            axs[0].axvspan(signals_df.index[i-1], signals_df.index[i], 
                          alpha=0.1, color='green')
        elif signals_df['signal'].iloc[i] < 0:  # Short
            axs[0].axvspan(signals_df.index[i-1], signals_df.index[i], 
                          alpha=0.1, color='red')
        
        # Enhanced signal as colored points
        if enhanced_signals_df['signal'].iloc[i] != signals_df['signal'].iloc[i]:
            if enhanced_signals_df['signal'].iloc[i] > 0:  # Enhanced long
                axs[0].plot(enhanced_signals_df.index[i], 
                           spread_df.loc[enhanced_signals_df.index[i], 'zscore'],
                           'g^', markersize=8)
            elif enhanced_signals_df['signal'].iloc[i] < 0:  # Enhanced short
                axs[0].plot(enhanced_signals_df.index[i], 
                           spread_df.loc[enhanced_signals_df.index[i], 'zscore'],
                           'rv', markersize=8)
            elif enhanced_signals_df['signal'].iloc[i] == 0:  # Enhanced exit
                axs[0].plot(enhanced_signals_df.index[i], 
                           spread_df.loc[enhanced_signals_df.index[i], 'zscore'],
                           'ko', markersize=6)
    
    axs[0].set_title(f'Z-Score and Signals - {pair_id}')
    axs[0].set_ylabel('Z-Score')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # 2. Plot signals
    axs[1].plot(signals_df.index, signals_df['signal'], label='Original Signals')
    axs[1].plot(enhanced_signals_df.index, enhanced_signals_df['signal'], 
               label='Enhanced Signals', linestyle='--')
    axs[1].set_title('Signal Comparison')
    axs[1].set_ylabel('Position')
    axs[1].set_yticks([-1, 0, 1])
    axs[1].set_yticklabels(['Short', 'Flat', 'Long'])
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # 3. Plot cumulative returns
    axs[2].plot(signals_df.index, signals_df['cumulative_return'], 
               label='Original Strategy')
    axs[2].plot(enhanced_signals_df.index, enhanced_signals_df['cumulative_return'], 
               label='Enhanced Strategy')
    axs[2].set_title('Cumulative Returns')
    axs[2].set_ylabel('Return')
    axs[2].set_xlabel('Date')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # Add metrics as text box
    comparison = results['comparison']
    original = comparison['original']
    enhanced = comparison['enhanced']
    improvement = comparison['improvement']
    
    metrics_text = (
        f"Original Strategy:\n"
        f"  Win Rate: {original.get('win_rate', 0):.2%}\n"
        f"  Profit Factor: {original.get('profit_factor', 0):.2f}\n"
        f"  Total Return: {original.get('total_return', 0):.2%}\n"
        f"  Sharpe Ratio: {original.get('sharpe_ratio', 0):.2f}\n"
        f"  Max Drawdown: {original.get('max_drawdown', 0):.2%}\n\n"
        f"Enhanced Strategy:\n"
        f"  Win Rate: {enhanced.get('win_rate', 0):.2%}\n"
        f"  Profit Factor: {enhanced.get('profit_factor', 0):.2f}\n"
        f"  Total Return: {enhanced.get('total_return', 0):.2%}\n"
        f"  Sharpe Ratio: {enhanced.get('sharpe_ratio', 0):.2f}\n"
        f"  Max Drawdown: {enhanced.get('max_drawdown', 0):.2%}\n\n"
        f"Improvement:\n"
        f"  Win Rate: {improvement.get('win_rate', 0):.2%}\n"
        f"  Profit Factor: {improvement.get('profit_factor', 0):.2f}\n"
        f"  Total Return: {improvement.get('total_return', 0):.2%}\n"
        f"  Sharpe Ratio: {improvement.get('sharpe_ratio', 0):.2f}\n"
        f"  Max Drawdown: {improvement.get('max_drawdown', 0):.2%}"
    )
    
    plt.figtext(0.02, 0.01, metrics_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"evaluation_{pair_id}_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()
    
    return output_file

def plot_evaluation_summary(all_results, output_dir="data/results/evaluation"):
    """
    Create summary visualizations for all pairs.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with evaluation results for all pairs
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    str
        Path to the saved plot file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    pairs = list(all_results.keys())
    
    if not pairs:
        logger.warning("No results to plot")
        return None
    
    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # Prepare data
    win_rates_orig = [all_results[pair]['comparison']['original'].get('win_rate', 0) for pair in pairs]
    win_rates_enh = [all_results[pair]['comparison']['enhanced'].get('win_rate', 0) for pair in pairs]
    profit_factors_orig = [all_results[pair]['comparison']['original'].get('profit_factor', 0) for pair in pairs]
    profit_factors_enh = [all_results[pair]['comparison']['enhanced'].get('profit_factor', 0) for pair in pairs]
    returns_orig = [all_results[pair]['comparison']['original'].get('total_return', 0) for pair in pairs]
    returns_enh = [all_results[pair]['comparison']['enhanced'].get('total_return', 0) for pair in pairs]
    sharpes_orig = [all_results[pair]['comparison']['original'].get('sharpe_ratio', 0) for pair in pairs]
    sharpes_enh = [all_results[pair]['comparison']['enhanced'].get('sharpe_ratio', 0) for pair in pairs]
    drawdowns_orig = [all_results[pair]['comparison']['original'].get('max_drawdown', 0) for pair in pairs]
    drawdowns_enh = [all_results[pair]['comparison']['enhanced'].get('max_drawdown', 0) for pair in pairs]
    
    # Signal changes
    signal_changes = [all_results[pair]['signal_metrics'].get('change_percentage', 0) for pair in pairs]
    
    # Plot win rates
    bar_width = 0.35
    x = np.arange(len(pairs))
    
    axs[0, 0].bar(x - bar_width/2, win_rates_orig, bar_width, label='Original')
    axs[0, 0].bar(x + bar_width/2, win_rates_enh, bar_width, label='Enhanced')
    axs[0, 0].set_title('Win Rate')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(pairs, rotation=45)
    axs[0, 0].set_ylabel('Win Rate')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot profit factors
    axs[0, 1].bar(x - bar_width/2, profit_factors_orig, bar_width, label='Original')
    axs[0, 1].bar(x + bar_width/2, profit_factors_enh, bar_width, label='Enhanced')
    axs[0, 1].set_title('Profit Factor')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(pairs, rotation=45)
    axs[0, 1].set_ylabel('Profit Factor')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot returns
    axs[1, 0].bar(x - bar_width/2, returns_orig, bar_width, label='Original')
    axs[1, 0].bar(x + bar_width/2, returns_enh, bar_width, label='Enhanced')
    axs[1, 0].set_title('Total Return')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(pairs, rotation=45)
    axs[1, 0].set_ylabel('Return')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot Sharpe ratios
    axs[1, 1].bar(x - bar_width/2, sharpes_orig, bar_width, label='Original')
    axs[1, 1].bar(x + bar_width/2, sharpes_enh, bar_width, label='Enhanced')
    axs[1, 1].set_title('Sharpe Ratio')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(pairs, rotation=45)
    axs[1, 1].set_ylabel('Sharpe Ratio')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    # Plot drawdowns
    axs[2, 0].bar(x - bar_width/2, drawdowns_orig, bar_width, label='Original')
    axs[2, 0].bar(x + bar_width/2, drawdowns_enh, bar_width, label='Enhanced')
    axs[2, 0].set_title('Max Drawdown')
    axs[2, 0].set_xticks(x)
    axs[2, 0].set_xticklabels(pairs, rotation=45)
    axs[2, 0].set_ylabel('Drawdown')
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)
    
    # Plot signal changes
    axs[2, 1].bar(x, signal_changes)
    axs[2, 1].set_title('Signal Changes')
    axs[2, 1].set_xticks(x)
    axs[2, 1].set_xticklabels(pairs, rotation=45)
    axs[2, 1].set_ylabel('Percentage of Modified Signals')
    axs[2, 1].set_ylim(0, 1)
    axs[2, 1].grid(True, alpha=0.3)
    
    # Add average improvement text
    avg_win_rate_imp = np.mean([all_results[pair]['comparison']['improvement'].get('win_rate', 0) for pair in pairs])
    avg_profit_factor_imp = np.mean([all_results[pair]['comparison']['improvement'].get('profit_factor', 0) for pair in pairs])
    avg_return_imp = np.mean([all_results[pair]['comparison']['improvement'].get('total_return', 0) for pair in pairs])
    avg_sharpe_imp = np.mean([all_results[pair]['comparison']['improvement'].get('sharpe_ratio', 0) for pair in pairs])
    avg_drawdown_imp = np.mean([all_results[pair]['comparison']['improvement'].get('max_drawdown', 0) for pair in pairs])
    
    summary_text = (
        f"Average Improvements:\n"
        f"  Win Rate: {avg_win_rate_imp:.2%}\n"
        f"  Profit Factor: {avg_profit_factor_imp:.2f}\n"
        f"  Total Return: {avg_return_imp:.2%}\n"
        f"  Sharpe Ratio: {avg_sharpe_imp:.2f}\n"
        f"  Max Drawdown: {avg_drawdown_imp:.2%}\n"
        f"Average Signal Changes: {np.mean(signal_changes):.2%}"
    )
    
    plt.figtext(0.5, 0.01, summary_text, fontsize=12, ha='center',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()
    
    return output_file

def save_evaluation_results(all_results, output_dir="data/results/evaluation"):
    """
    Save evaluation results to a JSON file.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with evaluation results for all pairs
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    str
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    
    # Convert non-serializable objects
    cleaned_results = {}
    
    for pair_id, pair_results in all_results.items():
        cleaned_pair = {}
        
        for key, value in pair_results.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                continue
            elif isinstance(value, dict):
                cleaned_dict = {}
                for k, v in value.items():
                    if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        cleaned_dict[k] = str(v)
                    else:
                        cleaned_dict[k] = v
                cleaned_pair[key] = cleaned_dict
            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                cleaned_pair[key] = str(value)
            else:
                cleaned_pair[key] = value
        
        cleaned_results[pair_id] = cleaned_pair
    
    with open(output_file, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    logger.info(f"Saved evaluation results to {output_file}")
    
    return output_file

def main():
    """Main function to evaluate ML models for pairs trading."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate ML models for intraday pairs trading")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="data/configs/intraday_config_latest.json",
        help="Path to intraday configuration file"
    )
    
    parser.add_argument(
        "--start_date", 
        type=str, 
        default=None,
        help="Start date for test data (YYYY-MM-DD). If None, uses the test period from training."
    )
    
    parser.add_argument(
        "--end_date", 
        type=str, 
        default=None,
        help="End date for test data (YYYY-MM-DD). If None, uses the test period from training."
    )
    
    parser.add_argument(
        "--timeframe", 
        type=str, 
        default="5min",
        help="Timeframe to use for evaluation"
    )
    
    parser.add_argument(
        "--training_period", 
        type=str, 
        default="2023-01-01,2023-12-31",
        help="Training period (YYYY-MM-DD,YYYY-MM-DD) to derive test period if not specified"
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
    
    # Set test period
    if args.start_date is None or args.end_date is None:
        # Use training period to derive test period
        train_start, train_end = args.training_period.split(',')
        periods = create_train_test_periods(train_start, train_end, args.test_ratio)
        test_start = periods['test']['start']
        test_end = periods['test']['end']
    else:
        test_start = args.start_date
        test_end = args.end_date
    
    logger.info(f"Test period: {test_start} to {test_end}")
    
    # Evaluate models for each pair
    all_results = {}
    
    for pair in tqdm(config['pairs'], desc="Evaluating models for pairs"):
        pair_id = pair['pair_id']
        
        # Evaluate
        evaluation = evaluate_pair(
            pair_id,
            pair,
            test_start,
            test_end,
            args.timeframe
        )
        
        if evaluation:
            results, signals_df, enhanced_signals_df, spread_df = evaluation
            all_results[pair_id] = results
            
            # Plot individual pair results
            plot_pair_evaluation(pair_id, signals_df, enhanced_signals_df, spread_df, results)
            
            logger.info(f"Completed evaluation for pair {pair_id}")
        else:
            logger.warning(f"Failed to evaluate models for pair {pair_id}")
    
    # Save and plot summary results
    if all_results:
        save_evaluation_results(all_results)
        summary_plot = plot_evaluation_summary(all_results)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Configuration: {os.path.basename(config_file)}")
        print(f"Period: {test_start} to {test_end}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Pairs evaluated: {len(all_results)}/{len(config['pairs'])}")
        
        # Calculate average improvement
        avg_win_rate_imp = np.mean([all_results[pair]['comparison']['improvement'].get('win_rate', 0) for pair in all_results])
        avg_profit_factor_imp = np.mean([all_results[pair]['comparison']['improvement'].get('profit_factor', 0) for pair in all_results])
        avg_return_imp = np.mean([all_results[pair]['comparison']['improvement'].get('total_return', 0) for pair in all_results])
        avg_sharpe_imp = np.mean([all_results[pair]['comparison']['improvement'].get('sharpe_ratio', 0) for pair in all_results])
        
        print("\nAverage Improvements:")
        print(f"  Win Rate: {avg_win_rate_imp:.2%}")
        print(f"  Profit Factor: {avg_profit_factor_imp:.2f}")
        print(f"  Total Return: {avg_return_imp:.2%}")
        print(f"  Sharpe Ratio: {avg_sharpe_imp:.2f}")
        
        print("\nPair Details:")
        for pair_id, results in all_results.items():
            comp = results['comparison']
            print(f"\n{pair_id}:")
            print(f"  Original Win Rate: {comp['original'].get('win_rate', 0):.2%} → Enhanced: {comp['enhanced'].get('win_rate', 0):.2%}")
            print(f"  Original Return: {comp['original'].get('total_return', 0):.2%} → Enhanced: {comp['enhanced'].get('total_return', 0):.2%}")
            print(f"  Signal Changes: {results['signal_metrics'].get('change_percentage', 0):.2%} of signals")
        
        if summary_plot:
            print(f"\nSummary plot saved to: {summary_plot}")
    else:
        logger.warning("No pairs were successfully evaluated")

if __name__ == "__main__":
    main() 