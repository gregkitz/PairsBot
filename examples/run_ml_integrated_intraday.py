#!/usr/bin/env python
"""
Example script demonstrating the integrated ML system for intraday trading.

This script shows how to:
1. Set up the integrated ML system
2. Connect ML models to signal generation
3. Integrate regime detection with parameter adaptation
4. Run ML-enhanced backtests
5. Generate intraday trading plans
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import logging

from src.ml_enhancements.intraday_integration import IntradayMLSystem, integrate_with_paper_trading
from src.ml_enhancements.intraday_signals import IntradaySignalProcessor
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
        Dictionary with price, volume, spread, and signal data
    """
    # Check if we have real data
    try:
        # Attempt to load real data if available
        data_dir = "data/processed"
        symbols = ["ES", "NQ"]
        
        prices = {}
        volumes = {}
        
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
                volumes[symbol] = pd.DataFrame({'volume': df['volume']}, index=df.index)
            
        if len(prices) == len(symbols):
            logger.info(f"Loaded real data for {len(prices)} symbols")
            
            # Generate sample spread
            pair_data = calculate_sample_spread(prices["ES"], prices["NQ"])
            
            # Generate sample signals
            signals = generate_sample_signals(pair_data)
            
            # Generate sample performance
            performance = generate_sample_performance(signals, pair_data)
            
            return {
                'prices': prices,
                'volumes': volumes,
                'spreads': pair_data,
                'signals': signals,
                'performance': performance
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
    volumes = {}
    
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
    volumes['ES'] = pd.DataFrame({'volume': es_data['volume']}, index=es_data.index)
    
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
    volumes['NQ'] = pd.DataFrame({'volume': nq_data['volume']}, index=nq_data.index)
    
    # Calculate spread
    pair_data = calculate_sample_spread(prices["ES"], prices["NQ"])
    
    # Generate sample signals
    signals = generate_sample_signals(pair_data)
    
    # Generate sample performance
    performance = generate_sample_performance(signals, pair_data)
    
    return {
        'prices': prices,
        'volumes': volumes,
        'spreads': pair_data,
        'signals': signals,
        'performance': performance
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
    import statsmodels.api as sm
    
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


def generate_sample_performance(signals, spread_df):
    """
    Generate sample performance data for ML training.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        DataFrame with trading signals
    spread_df : pd.DataFrame
        DataFrame with spread data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with performance metrics
    """
    # Initialize performance data
    performance = pd.DataFrame(index=signals.index)
    
    # Calculate PnL (simplified)
    position = 0
    entry_price = 0
    
    pnl = []
    trade_duration = []
    current_trade_start = None
    
    for i in range(len(signals)):
        signal = signals.iloc[i, 0]
        price = spread_df['spread'].iloc[i]
        
        # Entry
        if position == 0 and signal != 0:
            position = signal
            entry_price = price
            current_trade_start = signals.index[i]
            pnl.append(0)
        
        # Exit
        elif position != 0 and signal == 0:
            # Calculate PnL
            trade_pnl = position * (price - entry_price)
            pnl.append(trade_pnl)
            
            # Calculate trade duration
            if current_trade_start is not None:
                duration = (signals.index[i] - current_trade_start).total_seconds() / 60  # in minutes
                trade_duration.append(duration)
            
            position = 0
            entry_price = 0
            current_trade_start = None
        
        # Holding
        elif position != 0:
            # Calculate unrealized PnL
            unrealized_pnl = position * (price - entry_price)
            pnl.append(unrealized_pnl)
        
        # No position
        else:
            pnl.append(0)
    
    performance['pnl'] = pnl
    
    # Calculate cumulative PnL
    performance['cum_pnl'] = performance['pnl'].cumsum()
    
    # Calculate drawdown
    performance['high_water_mark'] = performance['cum_pnl'].cummax()
    performance['drawdown'] = performance['cum_pnl'] - performance['high_water_mark']
    
    # Add additional metrics
    performance['volatility'] = spread_df['std']
    performance['zscore'] = spread_df['zscore']
    
    return performance


def get_default_ml_config():
    """
    Get default configuration for the ML system.
    
    Returns:
    --------
    dict
        Default configuration
    """
    config = {
        "signal_enhancer": {
            "feature_lookback": 20,
            "prediction_threshold": 0.6,
            "use_rsi_filter": True,
            "use_volume_filter": True,
            "use_volatility_filter": True,
            "enable_ml_filtering": True
        },
        "signal_processor": {
            "enable_ml_enhancement": True,
            "enable_regime_adaptation": True
        },
        "regime_detection": {
            "n_regimes": 3,
            "lookback_window": 60,
            "method": "kmeans",
            "features": [
                "volatility_20d",
                "rsi_14d",
                "atr_14d",
                "hurst_exponent",
                "mean_reversion_strength"
            ]
        },
        "backtest": {
            "account_size": 100000,
            "intraday_params": {
                "max_holding_period": 180,
                "time_filters": {
                    "avoid_first_15min": True,
                    "avoid_lunch_hour": True,
                    "high_liquidity_windows": [
                        {"start": "09:45", "end": "11:30"},
                        {"start": "13:30", "end": "15:45"}
                    ]
                },
                "exit_buffer_minutes": 15
            },
            "transaction_cost_model": {
                "commission_model": "ibkr_pro",
                "commission_params": {
                    "per_contract": 0.85,
                    "per_share": 0.005,
                    "minimum": 1.0
                },
                "slippage_model": "volume_based",
                "slippage_params": {
                    "base_points": 1.0,
                    "volume_factor": 0.5,
                    "volatility_factor": 0.3
                }
            }
        },
        "default_parameters": {
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss": 3.0,
            "position_size": 0.1,
            "time_filters": {
                "avoid_first_15min": True,
                "avoid_lunch_hour": True,
                "high_liquidity_windows": [
                    {"start": "09:45", "end": "11:30"},
                    {"start": "13:30", "end": "15:45"}
                ]
            }
        },
        "adaptation_frequency": "daily"
    }
    
    return config


def demo_ml_integration():
    """
    Demonstrate the integrated ML system for intraday trading.
    """
    logger.info("Starting ML integration demonstration")
    
    # Create output directory
    output_dir = os.path.join("output", "ml_integration_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure ML system
    ml_config = get_default_ml_config()
    
    # Initialize ML system
    ml_system = IntradayMLSystem(
        config=ml_config,
        models_dir=os.path.join(output_dir, "models"),
        output_dir=output_dir
    )
    
    # Load sample data
    data = load_sample_data(start_date='2023-01-01', end_date='2023-01-31')
    
    # Step 1: Train ML models
    logger.info("Step 1: Training ML models")
    ml_system.train_models(
        prices_data=data['prices'],
        spreads_data=data['spreads'],
        signals_data=data['signals'],
        performance_data=data['performance'],
        volumes_data=data['volumes']
    )
    
    # Step 2: Detect market regime
    logger.info("Step 2: Detecting market regime")
    regimes = ml_system.detect_market_regime(
        prices_data=data['prices'],
        volumes_data=data['volumes']
    )
    
    # Step 3: Adapt parameters based on market regime
    logger.info("Step 3: Adapting parameters based on market regime")
    adapted_parameters = ml_system.adapt_parameters()
    
    print("\nAdapted Parameters:")
    print(f"Entry Threshold: {adapted_parameters.get('entry_threshold', 2.0)}")
    print(f"Exit Threshold: {adapted_parameters.get('exit_threshold', 0.5)}")
    print(f"Stop Loss: {adapted_parameters.get('stop_loss', 3.0)}")
    print(f"Position Size: {adapted_parameters.get('position_size', 0.1)}")
    
    # Step 4: Enhance signals with ML
    logger.info("Step 4: Enhancing signals with ML")
    enhanced_signals = ml_system.enhance_signals(
        original_signals=data['signals'],
        prices_data=data['prices'],
        spreads_data=data['spreads'],
        volumes_data=data['volumes']
    )
    
    # Step 5: Run ML-enhanced backtest
    logger.info("Step 5: Running ML-enhanced backtest")
    backtest_results = ml_system.run_backtest(
        original_signals=data['signals'],
        prices_data=data['prices'],
        spreads_data=data['spreads'],
        volumes_data=data['volumes'],
        save_results=True
    )
    
    # Step 6: Generate intraday trading plan
    logger.info("Step 6: Generating intraday trading plan")
    trading_plan = ml_system.generate_intraday_trading_plan()
    
    print("\nIntraday Trading Plan:")
    print(f"Date: {trading_plan.get('date')}")
    print(f"Current Regime: {trading_plan.get('current_regime')}")
    print(f"Regime Description: {trading_plan.get('regime_description')}")
    
    schedule = trading_plan.get('trading_schedule', {})
    print("\nTrading Schedule:")
    print(f"First Entry: {schedule.get('first_entry')}")
    print(f"Last Exit: {schedule.get('last_exit')}")
    print(f"Max Holding: {schedule.get('max_holding_minutes')} minutes")
    
    params = trading_plan.get('trading_parameters', {})
    print("\nTrading Parameters:")
    print(f"Entry Threshold: {params.get('entry_threshold')}")
    print(f"Exit Threshold: {params.get('exit_threshold')}")
    print(f"Stop Loss: {params.get('stop_loss')}")
    print(f"Position Size: {params.get('position_size')}")
    
    # Step 7: Integrate with paper trading
    logger.info("Step 7: Integrating with paper trading")
    paper_trading_config = {
        "account_size": 100000,
        "pairs": [
            {
                "pair_id": "ES_NQ",
                "symbol1": "ES",
                "symbol2": "NQ",
                "config": {
                    "entry_zscore": 2.0,
                    "exit_zscore": 0.5,
                    "stop_loss_std": 3.0
                }
            }
        ],
        "max_allocation_per_pair": 0.1
    }
    
    updated_config = integrate_with_paper_trading(ml_system, paper_trading_config)
    
    # Save updated paper trading config
    with open(os.path.join(output_dir, "paper_trading_config.json"), 'w') as f:
        json.dump(updated_config, f, indent=4)
    
    # Display final metrics
    if 'detailed_metrics' in backtest_results:
        metrics = backtest_results['detailed_metrics']
        
        print("\nML-Enhanced Backtest Results:")
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
        
        print("\nTransaction Costs:")
        costs = metrics.get('transaction_costs', {})
        print(f"Commission: ${costs.get('commission', 0):.2f}")
        print(f"Slippage: ${costs.get('slippage', 0):.2f}")
        print(f"% of Gross Profit: {costs.get('pct_of_gross_profit', 0) * 100:.2f}%")
    
    logger.info(f"ML integration demonstration completed. Results saved to {output_dir}")
    
    return {
        'ml_system': ml_system,
        'regimes': regimes,
        'enhanced_signals': enhanced_signals,
        'backtest_results': backtest_results,
        'trading_plan': trading_plan,
        'paper_trading_config': updated_config,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.error("statsmodels is required but not installed. Please install it with 'pip install statsmodels'.")
        exit(1)
    
    demo_results = demo_ml_integration() 