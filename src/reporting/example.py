"""
Example usage of the reporting module for generating backtest reports.

This script demonstrates how to use the reporting module to generate
comprehensive HTML reports from backtest results.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import reporting module
from src.reporting import BacktestReportGenerator, calculate_performance_metrics

# Create example data
def create_sample_data():
    """Create sample backtest data for demonstration."""
    # Create date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create equity curve with some randomness
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(0.0005, 0.01, len(dates))
    equity = 10000 * (1 + daily_returns).cumprod()
    equity_curve = pd.Series(equity, index=dates)
    
    # Create benchmark (e.g., S&P 500-like)
    benchmark_returns = np.random.normal(0.0004, 0.012, len(dates))
    benchmark = 10000 * (1 + benchmark_returns).cumprod()
    benchmark_curve = pd.Series(benchmark, index=dates)
    
    # Create sample trades
    num_trades = 100
    trade_indices = np.sort(np.random.choice(range(len(dates) - 1), num_trades, replace=False))
    
    trades = []
    for i in trade_indices:
        entry_time = dates[i]
        days_held = np.random.randint(1, 7)  # Hold for 1-7 days
        exit_time = entry_time + timedelta(days=days_held)
        
        if exit_time > dates[-1]:
            exit_time = dates[-1]
            
        # Random P&L with positive skew
        pnl = np.random.normal(20, 100)
        side = 'long' if np.random.random() > 0.4 else 'short'  # 60% long, 40% short
        symbol = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB'])
        entry_price = np.random.uniform(100, 200)
        exit_price = entry_price * (1 + pnl / entry_price)
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'side': side,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price
        })
    
    trades_df = pd.DataFrame(trades)
    
    return equity_curve, benchmark_curve, trades_df

def main():
    """Generate a sample backtest report."""
    print("Generating sample backtest data...")
    equity_curve, benchmark, trades = create_sample_data()
    
    print("Calculating performance metrics...")
    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=trades,
        benchmark=benchmark,
        risk_free_rate=0.02  # 2% risk-free rate
    )
    
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    print("\nGenerating HTML report...")
    
    # Create report generator
    report_generator = BacktestReportGenerator(
        output_dir='reports'
    )
    
    # Generate report
    report_path = report_generator.generate_report(
        title='Sample Pairs Trading Strategy',
        strategy_description="""
        This is a sample pairs trading strategy that trades mean-reverting spreads between
        cointegrated assets. The strategy uses statistical methods to identify profitable
        trading opportunities based on temporary mispricings.
        """,
        equity_curve=equity_curve,
        trades=trades,
        benchmark=benchmark,
        risk_free_rate=0.02,
        output_filename='sample_backtest_report.html'
    )
    
    print(f"Report generated successfully at: {report_path}")
    print(f"Open this file in a web browser to view the interactive report.")

if __name__ == "__main__":
    main() 