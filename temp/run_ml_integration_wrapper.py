
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_days=60, n_bars_per_day=78):
    """Generate sample intraday data for ML demonstration."""
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1, 9, 30)  # 9:30 AM
    timestamps = []
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        # Skip weekends
        if current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            continue
        
        # Market hours: 9:30 AM to 4:00 PM
        for bar in range(n_bars_per_day):
            timestamps.append(current_date + timedelta(minutes=5*bar))
    
    # Generate price series with some correlation and mean-reverting spread
    n_bars = len(timestamps)
    common_factor = np.random.randn(n_bars).cumsum() * 0.3
    
    # Mean-reverting component
    mean_reverting = np.sin(np.linspace(0, 20*np.pi, n_bars)) * 5
    
    # Create two correlated series
    series1 = 100 + common_factor + mean_reverting + np.random.randn(n_bars) * 0.5
    series2 = 50 + common_factor * 1.2 - mean_reverting * 0.5 + np.random.randn(n_bars) * 0.7
    
    # Create DataFrames with OHLCV data
    data1 = pd.DataFrame({
        'open': series1,
        'high': series1 + np.random.rand(n_bars) * 0.5,
        'low': series1 - np.random.rand(n_bars) * 0.5,
        'close': series1 + np.random.randn(n_bars) * 0.2,
        'volume': np.random.randint(100, 1000, size=n_bars)
    }, index=timestamps)
    
    data2 = pd.DataFrame({
        'open': series2,
        'high': series2 + np.random.rand(n_bars) * 0.5,
        'low': series2 - np.random.rand(n_bars) * 0.5,
        'close': series2 + np.random.randn(n_bars) * 0.2,
        'volume': np.random.randint(100, 1000, size=n_bars)
    }, index=timestamps)
    
    return {'asset1': data1, 'asset2': data2}

def demo_ml_integration():
    """Run a simplified ML integration demo."""
    logger.info("Starting ML integration demonstration")
    
    # Generate sample data
    logger.info("Generating sample intraday data...")
    data = generate_sample_data()
    
    asset1 = data['asset1']
    asset2 = data['asset2']
    
    # Calculate a simple spread
    hedge_ratio = 0.5
    spread = asset1['close'] - hedge_ratio * asset2['close']
    
    # Calculate z-score
    z_score = (spread - spread.rolling(window=20).mean()) / spread.rolling(window=20).std()
    
    # Generate some basic signals
    entry_threshold = 2.0
    exit_threshold = 0.5
    
    signals = pd.Series(0, index=spread.index)
    position = 0
    
    for i in range(1, len(z_score)):
        if pd.isna(z_score[i]):
            continue
            
        # Entry logic
        if position == 0:
            if z_score[i] > entry_threshold:
                signals[i] = -1  # Short position when spread is high
                position = -1
            elif z_score[i] < -entry_threshold:
                signals[i] = 1  # Long position when spread is low
                position = 1
        
        # Exit logic
        elif position == 1:  # In a long position
            if z_score[i] > -exit_threshold:
                signals[i] = 0  # Exit long position
                position = 0
        elif position == -1:  # In a short position
            if z_score[i] < exit_threshold:
                signals[i] = 0  # Exit short position
                position = 0
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(asset1.index, asset1['close'], label='Asset 1')
    plt.plot(asset2.index, asset2['close'] * hedge_ratio, label='Asset 2 (adjusted)')
    plt.legend()
    plt.title('Asset Prices')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(spread.index, spread)
    plt.title('Spread')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(z_score.index, z_score)
    plt.axhline(entry_threshold, color='r', linestyle='--', alpha=0.5)
    plt.axhline(-entry_threshold, color='g', linestyle='--', alpha=0.5)
    plt.axhline(exit_threshold, color='r', linestyle=':', alpha=0.5)
    plt.axhline(-exit_threshold, color='g', linestyle=':', alpha=0.5)
    
    # Plot signals
    for i in range(len(signals)):
        if signals[i] == 1:
            plt.plot(signals.index[i], z_score[i], '^', color='g', markersize=8)
        elif signals[i] == -1:
            plt.plot(signals.index[i], z_score[i], 'v', color='r', markersize=8)
        elif signals[i] == 0 and i > 0 and (signals[i-1] == 1 or signals[i-1] == -1):
            plt.plot(signals.index[i], z_score[i], 'o', color='k', markersize=6)
    
    plt.title('Z-Score and Signals')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    logger.info("ML integration demonstration completed")
    return data

if __name__ == "__main__":
    demo_ml_integration()
            