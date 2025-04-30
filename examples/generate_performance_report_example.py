#!/usr/bin/env python
"""
Example script demonstrating how to generate a performance report directly.

This script shows how to use the reporting framework to create a complete
performance report from custom trading data.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reporting import BacktestReportGenerator, calculate_performance_metrics

def create_sample_data():
    """Create sample trading data for demonstration."""
    # Create date range for a trading month
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    # Create market hours timestamps (9:30 AM to 4:00 PM on weekdays)
    timestamps = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday
            for hour in range(9, 16):
                for minute in range(0, 60, 5):  # 5-minute bars
                    if hour == 9 and minute < 30:
                        continue  # Market opens at 9:30 AM
                    if hour == 16 and minute > 0:
                        continue  # Market closes at 4:00 PM
                    
                    timestamps.append(current_date.replace(hour=hour, minute=minute))
        
        current_date += timedelta(days=1)
    
    # Create equity curve with some randomness and upward trend
    np.random.seed(42)  # For reproducibility
    
    # Initial value
    initial_value = 10000.0
    
    # Simulate equity curve with momentum and mean reversion
    equity_values = [initial_value]
    momentum = 0
    
    for i in range(1, len(timestamps)):
        # Random component
        noise = np.random.normal(0, 0.0005)
        
        # Trend component (slight upward bias)
        trend = 0.0001
        
        # Momentum component (persistence)
        momentum = 0.8 * momentum + 0.2 * noise
        
        # Mean reversion component
        current_return = equity_values[-1] / initial_value - 1
        mean_reversion = -0.05 * current_return
        
        # Combine components
        daily_return = trend + momentum + mean_reversion + noise
        
        # Update equity
        new_value = equity_values[-1] * (1 + daily_return)
        equity_values.append(new_value)
    
    # Create equity curve Series
    equity_curve = pd.Series(equity_values, index=timestamps)
    
    # Create trades
    trade_count = 30
    trades = []
    
    for i in range(trade_count):
        # Randomly select entry time
        entry_idx = np.random.randint(0, len(timestamps) - 20)
        entry_time = timestamps[entry_idx]
        
        # Hold for 5-20 bars
        holding_periods = np.random.randint(5, 21)
        exit_idx = min(entry_idx + holding_periods, len(timestamps) - 1)
        exit_time = timestamps[exit_idx]
        
        # Determine side (60% long, 40% short)
        side = 'long' if np.random.random() < 0.6 else 'short'
        
        # Generate P&L (positive expectancy)
        pnl_factor = 1 if side == 'long' else -1
        pnl_base = np.random.normal(0.002, 0.005) * pnl_factor
        
        # Calculate prices
        base_price = 100 + np.random.random() * 50
        entry_price = base_price
        exit_price = base_price * (1 + pnl_base)
        pnl = base_price * pnl_base * 100  # Scale for dollars
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': f"PAIR_{i % 5}",  # 5 different pairs
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': 100,
            'pnl': pnl
        })
    
    trades_df = pd.DataFrame(trades)
    
    # Create benchmark (S&P 500-like)
    benchmark_values = [initial_value]
    for i in range(1, len(timestamps)):
        # Random component with less volatility than strategy
        noise = np.random.normal(0, 0.0003)
        
        # Slight upward bias (traditional markets tend to go up)
        trend = 0.00005
        
        benchmark_values.append(benchmark_values[-1] * (1 + trend + noise))
    
    benchmark = pd.Series(benchmark_values, index=timestamps)
    
    return equity_curve, trades_df, benchmark

def generate_report():
    """Generate a complete performance report."""
    print("Generating sample trading data...")
    equity_curve, trades, benchmark = create_sample_data()
    
    print(f"Generated {len(equity_curve)} price points and {len(trades)} trades")
    
    # Calculate key statistics
    print("\nCalculating performance metrics...")
    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=trades,
        benchmark=benchmark,
        risk_free_rate=0.04  # 4% annual risk-free rate
    )
    
    # Print key metrics
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    
    # Ensure output directory exists
    os.makedirs('reports', exist_ok=True)
    
    # Calculate unique trading days
    trading_days = pd.Series([d.date() for d in equity_curve.index]).nunique()
    
    # Create report generator
    report_generator = BacktestReportGenerator(
        output_dir='reports'
    )
    
    # Generate report
    report_path = report_generator.generate_report(
        title='Intraday Pairs Trading Performance',
        strategy_description="""
        This performance report shows the results of an intraday statistical arbitrage strategy
        trading cointegrated pairs with ML enhancements for signal generation and regime detection.
        The strategy aims to profit from temporary mispricings between related assets while
        maintaining low overall market exposure.
        """,
        equity_curve=equity_curve,
        trades=trades,
        benchmark=benchmark,
        risk_free_rate=0.04,
        output_filename='intraday_ml_performance_report.html',
        additional_metrics={
            'strategy_type': 'Intraday Statistical Arbitrage',
            'trading_pairs': 5,
            'avg_trades_per_day': len(trades) / (trading_days or 1),
            'avg_holding_period_minutes': (trades['exit_time'] - trades['entry_time']).mean().total_seconds() / 60
        }
    )
    
    print(f"Report generated successfully at: {report_path}")
    print("Open this file in a web browser to view the interactive report.")

if __name__ == "__main__":
    print("=" * 80)
    print("PERFORMANCE REPORT GENERATION EXAMPLE")
    print("=" * 80)
    print()
    
    generate_report()
    
    print()
    print("=" * 80) 