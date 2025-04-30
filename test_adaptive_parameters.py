#!/usr/bin/env python3
"""
Test script for adaptive parameter system.

This script tests the adaptive parameter framework for intraday trading by:
1. Loading sample price data
2. Running parameter optimization for different regimes
3. Testing the parameter adaptation in real-time
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import argparse

from src.optimization.intraday_parameter_optimizer import IntradayParameterOptimizer
from src.optimization.adaptive_parameter_manager import AdaptiveParameterManager
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(data_file: str) -> pd.DataFrame:
    """
    Load test data from a CSV or parquet file.
    
    Parameters:
    -----------
    data_file : str
        Path to data file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with price data
    """
    logger.info(f"Loading test data from {data_file}")
    
    try:
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
            
            # Convert date column to datetime if exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
        elif data_file.endswith('.parquet'):
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def create_sample_pair_config() -> dict:
    """
    Create a sample pair configuration for testing.
    
    Returns:
    --------
    dict
        Sample pair configuration
    """
    return {
        "ticker1": "GC",
        "ticker2": "SI",
        "hedge_ratio": 0.085,
        "config": {
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss_std": 2.5,
            "max_risk_per_trade": 0.01,
            "max_allocation": 0.1,
            "max_holding_period": 120,
            "use_kalman": True,
            "z_score_window": 40
        }
    }


def test_optimization(data_file: str, output_dir: str):
    """
    Test parameter optimization with regime detection.
    
    Parameters:
    -----------
    data_file : str
        Path to data file
    output_dir : str
        Output directory for results
    """
    # Load test data
    prices_df = load_test_data(data_file)
    
    if prices_df is None:
        return
    
    # Create sample pairs
    pairs = [('GC', 'SI', 0.085), ('ZN', 'ZB', 0.95)]
    
    # Create optimizer
    optimizer = IntradayParameterOptimizer(
        n_regimes=3,
        n_jobs=1,
        verbose=True
    )
    
    # Detect regimes
    classifier, regimes = optimizer.detect_regimes(prices_df)
    
    # Plot regimes
    plt.figure(figsize=(12, 8))
    
    # Plot prices
    ax1 = plt.subplot(2, 1, 1)
    for col in prices_df.columns:
        # Normalize prices for better visualization
        normalized = prices_df[col] / prices_df[col].iloc[0]
        ax1.plot(normalized, label=col)
    
    ax1.set_title('Price Data')
    ax1.set_ylabel('Normalized Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot regimes
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(regimes.index, regimes, drawstyle='steps-post')
    ax2.set_title('Market Regimes')
    ax2.set_ylabel('Regime')
    ax2.set_yticks(list(range(optimizer.n_regimes)))
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'detected_regimes.png'))
    plt.close()
    
    # Run optimization with only a subset of data for faster testing
    test_start_date = str(prices_df.index[0].date())
    test_end_date = str(prices_df.index[min(500, len(prices_df)-1)].date())
    
    logger.info(f"Running optimization on data from {test_start_date} to {test_end_date}")
    
    # Run optimization
    regime_results = optimizer.optimize_by_regime(
        pairs=pairs,
        prices_df=prices_df.iloc[:500],
        start_date=test_start_date,
        end_date=test_end_date,
        timeframe='1hour',
        commission=2.0,
        slippage=1.0,
        account_size=25000,
        lookback_window=60
    )
    
    # Create adaptive parameter configuration
    adaptive_config = optimizer.create_adaptive_parameter_config()
    
    # Save adaptive parameter configuration
    config_file = os.path.join(output_dir, 'test_adaptive_parameters.json')
    with open(config_file, 'w') as f:
        json.dump(adaptive_config, f, indent=2)
    
    logger.info(f"Saved adaptive parameter configuration to {config_file}")
    
    # Plot regime comparison
    optimizer.plot_regime_comparison(output_dir)
    
    return config_file, prices_df


def test_parameter_adaptation(config_file: str, prices_df: pd.DataFrame, output_dir: str):
    """
    Test parameter adaptation with regime detection.
    
    Parameters:
    -----------
    config_file : str
        Path to adaptive parameter configuration file
    prices_df : pd.DataFrame
        DataFrame with price data
    output_dir : str
        Output directory for results
    """
    # Create adaptive parameter manager
    manager = AdaptiveParameterManager(config_file=config_file)
    
    # Create sample pair configuration
    pair_config = create_sample_pair_config()
    
    # Split data into chunks to simulate real-time updates
    chunk_size = min(50, len(prices_df) // 5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store adapted configurations for each chunk
    adapted_configs = []
    
    # Process each chunk
    for i in range(0, len(prices_df), chunk_size):
        # Get chunk of data
        chunk = prices_df.iloc[i:i+chunk_size]
        
        if len(chunk) == 0:
            continue
        
        logger.info(f"Processing chunk {i//chunk_size+1}/{(len(prices_df)+chunk_size-1)//chunk_size}")
        
        # Adapt parameters
        adapted_config = manager.adapt_pair_config(pair_config, chunk)
        
        # Store adapted configuration with timestamp
        adapted_configs.append({
            'timestamp': str(chunk.index[-1]),
            'regime': manager.current_regime[1] if manager.current_regime else None,
            'config': adapted_config
        })
        
        # Print adapted parameters
        logger.info(f"Adapted parameters for regime {manager.current_regime if manager.current_regime else 'unknown'}:")
        logger.info(f"  entry_threshold: {adapted_config['config']['entry_threshold']:.2f}")
        logger.info(f"  exit_threshold: {adapted_config['config']['exit_threshold']:.2f}")
        logger.info(f"  stop_loss_std: {adapted_config['config']['stop_loss_std']:.2f}")
        logger.info(f"  adjusted_max_allocation: {adapted_config['config']['adjusted_max_allocation']:.4f}")
    
    # Save adapted configurations
    config_file = os.path.join(output_dir, 'adapted_configurations.json')
    with open(config_file, 'w') as f:
        json.dump(adapted_configs, f, indent=2)
    
    logger.info(f"Saved adapted configurations to {config_file}")
    
    # Plot regime history
    manager.plot_regime_history(os.path.join(output_dir, 'regime_history.png'))
    
    # Plot parameter changes over time
    param_history = []
    for config in adapted_configs:
        param_history.append({
            'timestamp': pd.to_datetime(config['timestamp']),
            'entry_threshold': config['config']['config']['entry_threshold'],
            'exit_threshold': config['config']['config']['exit_threshold'],
            'stop_loss_std': config['config']['config']['stop_loss_std'],
            'adjusted_max_allocation': config['config']['config']['adjusted_max_allocation'],
            'regime': config['regime']
        })
    
    param_df = pd.DataFrame(param_history)
    param_df.set_index('timestamp', inplace=True)
    
    plt.figure(figsize=(12, 10))
    
    # Plot entry threshold
    ax1 = plt.subplot(4, 1, 1)
    for regime in param_df['regime'].unique():
        if pd.isna(regime):
            continue
        regime_df = param_df[param_df['regime'] == regime]
        ax1.plot(regime_df.index, regime_df['entry_threshold'], 'o-', label=regime)
    
    ax1.set_title('Entry Threshold')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot exit threshold
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    for regime in param_df['regime'].unique():
        if pd.isna(regime):
            continue
        regime_df = param_df[param_df['regime'] == regime]
        ax2.plot(regime_df.index, regime_df['exit_threshold'], 'o-', label=regime)
    
    ax2.set_title('Exit Threshold')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    # Plot stop loss
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    for regime in param_df['regime'].unique():
        if pd.isna(regime):
            continue
        regime_df = param_df[param_df['regime'] == regime]
        ax3.plot(regime_df.index, regime_df['stop_loss_std'], 'o-', label=regime)
    
    ax3.set_title('Stop Loss (std)')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True)
    
    # Plot allocation
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    for regime in param_df['regime'].unique():
        if pd.isna(regime):
            continue
        regime_df = param_df[param_df['regime'] == regime]
        ax4.plot(regime_df.index, regime_df['adjusted_max_allocation'], 'o-', label=regime)
    
    ax4.set_title('Adjusted Max Allocation')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_changes.png'))
    plt.close()
    
    logger.info(f"Saved parameter changes plot to {os.path.join(output_dir, 'parameter_changes.png')}")


def main():
    """Main function to test adaptive parameter system."""
    parser = argparse.ArgumentParser(description='Test adaptive parameter system')
    
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to data file (CSV or parquet)')
    parser.add_argument('--output_dir', type=str, default='output/test_adaptive',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Test optimization
    config_file, prices_df = test_optimization(args.data_file, args.output_dir)
    
    if config_file is None or prices_df is None:
        logger.error("Optimization test failed")
        return
    
    # Test parameter adaptation
    test_parameter_adaptation(config_file, prices_df, args.output_dir)
    
    logger.info("Adaptive parameter system tests completed!")


if __name__ == "__main__":
    main() 