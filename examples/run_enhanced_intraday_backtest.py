#!/usr/bin/env python
"""
Example script demonstrating how to use the IntradayBacktestEngine with performance visualization.

This script shows how to:
1. Set up an intraday backtest with realistic constraints
2. Run the backtest with transaction cost modeling
3. Visualize the results with detailed performance analysis
4. Incorporate market regime information into the analysis
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import logging

from src.backtest.intraday_backtest_engine import IntradayBacktestEngine
from src.backtest.intraday_performance_visualization import (
    create_intraday_performance_dashboard,
    save_performance_dashboard
)
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data(start_date='2023-01-01', end_date='2023-01-31', timeframe='5min'):
    """
    Load or generate sample data for demonstration.
    
    Parameters:
    -----------
    start_date : str
        Start date for data generation
    end_date : str
        End date for data generation
    timeframe : str
        Timeframe for data
        
    Returns:
    --------
    dict
        Dictionary with price, volume, and signal data
    """
    # Check if we have real data
    try:
        # Attempt to load real data if available
        data_dir = "data/processed"
        symbols = ["ES", "NQ"]
        
        prices = {}
        volume_data = {}
        
        for symbol in symbols:
            file_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
            
            if os.path.exists(file_path):
                # Load real data
                df = pd.read_parquet(file_path)
                
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                # Resample if needed
                if timeframe and 'close' in df.columns:
                    df = df.resample(timeframe).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                
                prices[symbol] = df
                volume_data[symbol] = pd.DataFrame({'volume': df['volume']}, index=df.index)
            
        if len(prices) == len(symbols):
            logger.info(f"Loaded real data for {len(prices)} symbols")
            
            # Generate sample spread
            pair_data = calculate_sample_spread(prices["ES"], prices["NQ"])
            
            # Generate sample signals
            signals = generate_sample_signals(pair_data)
            
            return {
                'prices': prices,
                'volume_data': volume_data,
                'signals': signals,
                'pair_data': pair_data
            }
    
    except Exception as e:
        logger.warning(f"Could not load real data: {e}. Generating synthetic data instead.")
    
    # Generate synthetic data if real data not available
    logger.info("Generating synthetic data")
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
    
    # Only keep dates during market hours (9:30 AM - 4:00 PM, Monday-Friday)
    dates = dates[
        (dates.dayofweek < 5) &  # Monday-Friday
        (dates.time >= time(9, 30)) &  # After 9:30 AM
        (dates.time < time(16, 0))  # Before 4:00 PM
    ]
    
    # Generate synthetic price data
    prices = {}
    volume_data = {}
    
    # Symbol 1: ES (S&P 500 E-mini futures)
    es_data = pd.DataFrame({
        'open': np.linspace(4000, 4200, len(dates)) + np.random.normal(0, 20, len(dates)),
        'high': np.linspace(4000, 4200, len(dates)) + np.random.normal(0, 30, len(dates)),
        'low': np.linspace(4000, 4200, len(dates)) + np.random.normal(0, 25, len(dates)),
        'close': np.linspace(4000, 4200, len(dates)) + np.random.normal(0, 15, len(dates)),
        'volume': np.random.randint(500, 5000, len(dates))
    }, index=dates)
    
    # Ensure high >= open, low <= open, etc.
    es_data['high'] = es_data[['open', 'high']].max(axis=1)
    es_data['low'] = es_data[['open', 'low']].min(axis=1)
    prices['ES'] = es_data
    volume_data['ES'] = pd.DataFrame({'volume': es_data['volume']}, index=es_data.index)
    
    # Symbol 2: NQ (Nasdaq 100 E-mini futures)
    nq_data = pd.DataFrame({
        'open': np.linspace(12000, 12600, len(dates)) + np.random.normal(0, 40, len(dates)),
        'high': np.linspace(12000, 12600, len(dates)) + np.random.normal(0, 60, len(dates)),
        'low': np.linspace(12000, 12600, len(dates)) + np.random.normal(0, 50, len(dates)),
        'close': np.linspace(12000, 12600, len(dates)) + np.random.normal(0, 30, len(dates)),
        'volume': np.random.randint(400, 4000, len(dates))
    }, index=dates)
    
    # Ensure high >= open, low <= open, etc.
    nq_data['high'] = nq_data[['open', 'high']].max(axis=1)
    nq_data['low'] = nq_data[['open', 'low']].min(axis=1)
    prices['NQ'] = nq_data
    volume_data['NQ'] = pd.DataFrame({'volume': nq_data['volume']}, index=nq_data.index)
    
    # Calculate spread
    pair_data = calculate_sample_spread(prices["ES"], prices["NQ"])
    
    # Generate sample signals
    signals = generate_sample_signals(pair_data)
    
    return {
        'prices': prices,
        'volume_data': volume_data,
        'signals': signals,
        'pair_data': pair_data
    }

def calculate_sample_spread(price_df1, price_df2, hedge_ratio=None):
    """
    Calculate a sample spread between two price series.
    
    Parameters:
    -----------
    price_df1 : pd.DataFrame
        Price data for first instrument
    price_df2 : pd.DataFrame
        Price data for second instrument
    hedge_ratio : float, optional
        Hedge ratio to use (calculated if None)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with spread data
    """
    # Align price data
    common_index = price_df1.index.intersection(price_df2.index)
    price1 = price_df1.loc[common_index, 'close']
    price2 = price_df2.loc[common_index, 'close']
    
    # Calculate hedge ratio if not provided
    if hedge_ratio is None:
        # Simple OLS regression
        X = sm.add_constant(price1)
        model = sm.OLS(price2, X).fit()
        hedge_ratio = model.params[1]
    
    # Calculate spread
    spread = price2 - hedge_ratio * price1
    
    # Calculate z-score (standardized spread)
    window = min(30, len(spread))
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    
    # Combine into DataFrame
    spread_df = pd.DataFrame({
        'price1': price1,
        'price2': price2,
        'spread': spread,
        'mean': mean,
        'std': std,
        'zscore': zscore
    })
    
    return spread_df

def generate_sample_signals(spread_df, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate sample trading signals based on z-score.
    
    Parameters:
    -----------
    spread_df : pd.DataFrame
        DataFrame with spread data including z-score
    entry_threshold : float
        Z-score threshold for entries
    exit_threshold : float
        Z-score threshold for exits
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with trading signals
    """
    # Initialize signals
    signals = pd.DataFrame(0, index=spread_df.index, columns=['pair_signal'])
    
    # Generate signals based on z-score
    position = 0
    
    for i in range(len(spread_df)):
        zscore = spread_df['zscore'].iloc[i]
        
        # Skip if zscore is NaN (e.g., during warmup period)
        if pd.isna(zscore):
            continue
        
        # Long entry
        if position == 0 and zscore < -entry_threshold:
            signals.iloc[i] = 1
            position = 1
        
        # Short entry
        elif position == 0 and zscore > entry_threshold:
            signals.iloc[i] = -1
            position = -1
        
        # Long exit
        elif position == 1 and zscore > -exit_threshold:
            signals.iloc[i] = 0
            position = 0
        
        # Short exit
        elif position == -1 and zscore < exit_threshold:
            signals.iloc[i] = 0
            position = 0
        
        # Maintain position
        else:
            signals.iloc[i] = position
    
    return signals

def generate_sample_regimes(dates, n_regimes=3, seed=42):
    """
    Generate sample market regime data.
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        Dates for which to generate regime data
    n_regimes : int
        Number of regimes to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with regime labels
    """
    np.random.seed(seed)
    
    # Generate initial regime
    regime = np.random.randint(0, n_regimes)
    
    # Probability of regime change
    p_change = 0.05
    
    # Generate regimes
    regimes = []
    for i in range(len(dates)):
        # Randomly change regime with low probability
        if np.random.random() < p_change:
            # Pick a new regime different from current
            new_regimes = list(range(n_regimes))
            new_regimes.remove(regime)
            regime = np.random.choice(new_regimes)
        
        regimes.append(regime)
    
    # Create DataFrame
    regime_df = pd.DataFrame({'regime': regimes}, index=dates)
    
    return regime_df

def main():
    """Main function to run enhanced intraday backtest."""
    logger.info("Starting enhanced intraday backtest demonstration")
    
    # Create output directory
    output_dir = os.path.join("output", "intraday_backtest_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample data
    data = load_sample_data(start_date='2023-01-01', end_date='2023-01-31')
    
    # Define intraday parameters
    intraday_params = {
        "max_holding_period": 180,  # 3 hours max
        "time_filters": {
            "avoid_first_15min": True,
            "avoid_lunch_hour": True,
            "high_liquidity_windows": [
                {"start": "09:45", "end": "11:30"},
                {"start": "13:30", "end": "15:45"}
            ]
        },
        "exit_buffer_minutes": 15  # Exit 15 minutes before market close
    }
    
    # Define transaction cost model
    transaction_cost_model = {
        "commission_model": "ibkr_pro",
        "commission_params": {
            "per_contract": 0.85,  # IBKR futures commission
            "per_share": 0.005,    # IBKR stock commission
            "minimum": 1.0         # Minimum commission
        },
        "slippage_model": "volume_based",
        "slippage_params": {
            "base_points": 1.0,       # Base slippage in price points
            "volume_factor": 0.5,     # Volume impact on slippage
            "volatility_factor": 0.3  # Volatility impact on slippage
        }
    }
    
    # Create IntradayBacktestEngine
    engine = IntradayBacktestEngine(
        signals=data['signals'],
        prices=data['prices'],
        account_size=100000,
        volume_data=data['volume_data'],
        intraday_params=intraday_params,
        transaction_cost_model=transaction_cost_model
    )
    
    # Run backtest
    logger.info("Running intraday backtest with realistic constraints")
    results = engine.run_backtest()
    
    # Generate sample regime data for demonstration
    regime_data = generate_sample_regimes(data['signals'].index)
    
    # Create performance dashboard
    logger.info("Creating performance visualization dashboard")
    dashboard_figures = create_intraday_performance_dashboard(
        backtest_results=results,
        regime_data=regime_data,
        figsize=(12, 8)
    )
    
    # Save performance dashboard
    save_performance_dashboard(
        figures=dashboard_figures,
        output_dir=output_dir,
        prefix="intraday_demo"
    )
    
    # Save summary results to CSV
    equity_curve = results.get('equity_curve')
    if equity_curve is not None:
        equity_curve.to_csv(os.path.join(output_dir, "equity_curve.csv"))
    
    # Display summary statistics
    detailed_metrics = engine.calculate_detailed_metrics()
    
    print("\n--- Enhanced Intraday Backtest Results ---")
    print(f"Total Return: {detailed_metrics.get('total_return', 0):.2f}%")
    print(f"Annualized Return: {detailed_metrics.get('annualized_return', 0):.2f}%")
    print(f"Sharpe Ratio: {detailed_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {detailed_metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"Win Rate: {detailed_metrics.get('win_rate', 0) * 100:.2f}%")
    print(f"Profit Factor: {detailed_metrics.get('profit_factor', 0):.2f}")
    
    print("\n--- Transaction Costs ---")
    print(f"Total Commission: ${detailed_metrics.get('transaction_costs', {}).get('commission', 0):.2f}")
    print(f"Total Slippage: ${detailed_metrics.get('transaction_costs', {}).get('slippage', 0):.2f}")
    print(f"% of Gross Profit: {detailed_metrics.get('transaction_costs', {}).get('pct_of_gross_profit', 0) * 100:.2f}%")
    
    print("\n--- Intraday Constraints Impact ---")
    print(f"Forced Exits: {detailed_metrics.get('constraints', {}).get('forced_exits', 0)}")
    print(f"Time Violations: {detailed_metrics.get('constraints', {}).get('time_violations', 0)}")
    
    logger.info(f"Enhanced intraday backtest completed. Results saved to {output_dir}")
    
    # Return paths to output files for convenience
    equity_curve_path = os.path.join(output_dir, "equity_curve.csv")
    dashboard_paths = [os.path.join(output_dir, f"intraday_demo_{i+1}.png") for i in range(len(dashboard_figures))]
    
    return {
        'equity_curve': equity_curve_path,
        'dashboard': dashboard_paths,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.error("statsmodels is required but not installed. Please install it with 'pip install statsmodels'.")
        exit(1)
    
    main() 