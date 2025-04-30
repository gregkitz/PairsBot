
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
    """Generate sample intraday data for backtesting."""
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
    
    # Generate price series with strong mean-reversion
    n_bars = len(timestamps)
    common_factor = np.random.randn(n_bars).cumsum() * 0.3
    
    # Mean-reverting component
    mean_reverting = np.sin(np.linspace(0, 10*np.pi, n_bars)) * 5
    
    # Create two cointegrated series
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

def run_enhanced_backtest():
    """Run an enhanced backtest demonstration."""
    logger.info("Starting enhanced backtest demonstration")
    
    # Generate sample data
    logger.info("Generating sample intraday data...")
    data = generate_sample_data()
    
    asset1 = data['asset1']
    asset2 = data['asset2']
    
    # Calculate spread and z-score
    hedge_ratio = 0.5
    spread = asset1['close'] - hedge_ratio * asset2['close']
    z_score = (spread - spread.rolling(window=20).mean()) / spread.rolling(window=20).std()
    
    # Generate base signals
    entry_threshold = 2.0
    exit_threshold = 0.5
    
    signals = pd.Series(0, index=spread.index)
    position = 0
    
    for i in range(20, len(z_score.index)):
        idx = z_score.index[i]
        # Skip nan values
        if pd.isna(z_score.loc[idx]):
            continue
            
        # Entry logic
        if position == 0:
            if z_score.loc[idx] > entry_threshold:
                signals.loc[idx] = -1  # Short position when spread is high
                position = -1
            elif z_score.loc[idx] < -entry_threshold:
                signals.loc[idx] = 1  # Long position when spread is low
                position = 1
        
        # Exit logic
        elif position == 1:  # In a long position
            if z_score.loc[idx] > -exit_threshold:
                signals.loc[idx] = 0  # Exit long position
                position = 0
        elif position == -1:  # In a short position
            if z_score.loc[idx] < exit_threshold:
                signals.loc[idx] = 0  # Exit short position
                position = 0
    
    # Add intraday constraints
    constrained_signals = signals.copy()
    
    # Force close positions at end of day
    for i in range(1, len(signals.index)):
        idx = signals.index[i]
        prev_idx = signals.index[i-1]
        
        # If crossing day boundary and have position
        if idx.date() != prev_idx.date() and signals.loc[prev_idx] != 0:
            # Close position at day end
            constrained_signals.loc[prev_idx] = 0
            # Re-enter position at day start if signal still valid
            constrained_signals.loc[idx] = signals.loc[idx]
    
    # Calculate PnL
    pnl = pd.Series(0.0, index=spread.index)
    cum_pnl = pd.Series(0.0, index=spread.index)
    
    # Initial position
    position = 0
    entry_price = 0
    
    for i in range(1, len(constrained_signals.index)):
        idx = constrained_signals.index[i]
        prev_idx = constrained_signals.index[i-1]
        
        # Entry
        if constrained_signals.loc[prev_idx] == 0 and constrained_signals.loc[idx] != 0:
            position = constrained_signals.loc[idx]
            entry_price = spread.loc[idx]
            pnl.loc[idx] = 0
        
        # Exit
        elif constrained_signals.loc[prev_idx] != 0 and constrained_signals.loc[idx] == 0:
            exit_price = spread.loc[idx]
            pnl.loc[idx] = position * (exit_price - entry_price)
            position = 0
            entry_price = 0
        
        # Hold
        elif constrained_signals.loc[prev_idx] != 0 and constrained_signals.loc[idx] == constrained_signals.loc[prev_idx]:
            pnl.loc[idx] = position * (spread.loc[idx] - spread.loc[prev_idx])
    
    # Calculate cumulative PnL
    cum_pnl = pnl.cumsum()
    
    # Add transaction costs (slippage & commission)
    transaction_costs = pd.Series(0.0, index=spread.index)
    
    for i in range(1, len(constrained_signals.index)):
        idx = constrained_signals.index[i]
        prev_idx = constrained_signals.index[i-1]
        
        if constrained_signals.loc[idx] != constrained_signals.loc[prev_idx]:
            # Fixed cost per trade
            transaction_costs.loc[idx] = 2.0
    
    # Adjust PnL for transaction costs
    adjusted_pnl = pnl - transaction_costs
    adjusted_cum_pnl = adjusted_pnl.cumsum()
    
    # Plot the results
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Asset Prices
    plt.subplot(4, 1, 1)
    plt.plot(asset1.index, asset1['close'], label='Asset 1')
    plt.plot(asset2.index, asset2['close'] * hedge_ratio, label='Asset 2 (adjusted)')
    plt.legend()
    plt.title('Asset Prices')
    plt.grid(True)
    
    # Plot 2: Spread and Z-Score
    plt.subplot(4, 1, 2)
    plt.plot(spread.index, spread, label='Spread', color='blue')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Twin axis for z-score
    ax2 = plt.twinx()
    ax2.plot(z_score.index, z_score, label='Z-Score', color='green', alpha=0.7)
    ax2.axhline(entry_threshold, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(-entry_threshold, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(exit_threshold, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(-exit_threshold, color='green', linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right')
    plt.title('Spread and Z-Score')
    
    # Plot 3: Signals
    plt.subplot(4, 1, 3)
    plt.plot(constrained_signals.index, constrained_signals, label='Signals')
    plt.title('Trading Signals')
    plt.grid(True)
    
    # Plot 4: PnL
    plt.subplot(4, 1, 4)
    plt.plot(cum_pnl.index, cum_pnl, label='Gross PnL', color='blue')
    plt.plot(adjusted_cum_pnl.index, adjusted_cum_pnl, label='Net PnL (after costs)', color='red')
    plt.title('Cumulative PnL')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate performance metrics
    total_trades = (constrained_signals != 0).sum() / 2  # Divide by 2 because each trade has entry and exit
    winning_trades = (adjusted_pnl > 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_pnl = adjusted_cum_pnl.iloc[-1]
    
    logger.info(f"Enhanced Backtest Results:")
    logger.info(f"Total PnL: {total_pnl:.2f}")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Total Transaction Costs: {transaction_costs.sum():.2f}")
    
    logger.info("Enhanced backtest demonstration completed")
    return data

if __name__ == "__main__":
    run_enhanced_backtest()
            