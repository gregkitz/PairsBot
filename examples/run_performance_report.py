"""
Example script demonstrating how to use the performance reporting framework.

This script shows how to generate performance reports for different trading modes
and date ranges using the generate_performance_report.py script.
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function directly for programmatic usage
from scripts.generate_performance_report import generate_report


def run_report_via_cli():
    """Demonstrate how to run the report generator via command line."""
    print("Running performance report generator via command line...")
    
    # Example 1: Generate report for the last 5 days of paper trading
    cmd = ["python", "scripts/generate_performance_report.py", "--mode", "paper", "--days", "5"]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Report generated successfully via CLI")
    except subprocess.CalledProcessError as e:
        print(f"Error running report via CLI: {e}")
    
    print()
    
    # Example 2: Generate report for a specific date of backtest results
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    cmd = ["python", "scripts/generate_performance_report.py", "--mode", "backtest", "--date", yesterday]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Report generated successfully via CLI")
    except subprocess.CalledProcessError as e:
        print(f"Error running report via CLI: {e}")
    
    print()


def run_report_programmatically():
    """Demonstrate how to run the report generator programmatically."""
    print("Running performance report generator programmatically...")
    
    # Example 1: Generate report for the last week of live trading
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    output_file = f"live_trading_last_week_{datetime.now().strftime('%Y%m%d')}.html"
    
    print(f"Generating report for live trading from {start_date.date()} to {end_date.date()}")
    
    try:
        report_path = generate_report(
            start_date=start_date,
            end_date=end_date,
            mode="live",
            output_file=output_file
        )
        if report_path:
            print(f"Report generated successfully at: {report_path}")
        else:
            print("Failed to generate report")
    except Exception as e:
        print(f"Error generating report: {e}")
    
    print()
    
    # Example 2: Generate report for a specific date range of paper trading
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    output_file = "paper_trading_january_2023.html"
    
    print(f"Generating report for paper trading from {start_date.date()} to {end_date.date()}")
    
    try:
        report_path = generate_report(
            start_date=start_date,
            end_date=end_date,
            mode="paper",
            output_file=output_file
        )
        if report_path:
            print(f"Report generated successfully at: {report_path}")
        else:
            print("Failed to generate report")
    except Exception as e:
        print(f"Error generating report: {e}")


def create_sample_data_for_demo():
    """Create sample data files for demonstration purposes."""
    print("Creating sample data files for demonstration...")
    
    # Ensure directories exist
    for mode in ['backtest', 'paper', 'live']:
        os.makedirs(f"data/{mode}_results", exist_ok=True)
    
    # Create sample backtest data for yesterday
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    import numpy as np
    import pandas as pd
    
    # Create equity curve
    dates = pd.date_range(
        start=yesterday.replace(hour=9, minute=30), 
        end=yesterday.replace(hour=16, minute=0),
        freq='1min'
    )
    
    np.random.seed(42)  # For reproducibility
    
    # Create sample equity curve with some randomness but upward trend
    daily_returns = np.random.normal(0.00002, 0.0002, len(dates))
    equity = 10000 * (1 + daily_returns).cumprod()
    equity_curve = pd.Series(equity, index=dates)
    
    # Save equity curve
    equity_file = f"data/backtest_results/equity_{yesterday_str}.csv"
    equity_curve.to_csv(equity_file)
    print(f"Sample equity curve saved to {equity_file}")
    
    # Create sample trades
    num_trades = 15
    
    trades = []
    for i in range(num_trades):
        entry_idx = np.random.randint(0, len(dates) - 60)
        entry_time = dates[entry_idx]
        
        # Hold position for 10-60 minutes
        minutes_held = np.random.randint(10, 60)
        exit_time = entry_time + timedelta(minutes=minutes_held)
        
        if exit_time > dates[-1]:
            exit_time = dates[-1]
        
        # Random P&L with positive skew
        pnl = np.random.normal(10, 50)
        side = 'long' if np.random.random() > 0.4 else 'short'
        symbol_1, symbol_2 = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB'], size=2, replace=False)
        entry_price = np.random.uniform(100, 200)
        exit_price = entry_price * (1 + pnl / (entry_price * 100))
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'side': side,
            'symbol': f"{symbol_1}/{symbol_2}",
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pair_ratio': np.random.uniform(0.8, 1.2),
            'quantity': np.random.randint(1, 10) * 100
        })
    
    trades_df = pd.DataFrame(trades)
    
    # Save trades
    trades_file = f"data/backtest_results/trades_{yesterday_str}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"Sample trades saved to {trades_file}")
    
    # Copy the same data to paper trading for demonstration
    equity_file_paper = f"data/paper_trading/equity_{yesterday_str}.csv"
    trades_file_paper = f"data/paper_trading/trades_{yesterday_str}.csv"
    
    equity_curve.to_csv(equity_file_paper)
    trades_df.to_csv(trades_file_paper, index=False)
    
    print(f"Sample data copied to paper trading directory for demonstration")
    print()


def main():
    """Main function to demonstrate the performance reporting framework."""
    print("=" * 80)
    print("PERFORMANCE REPORTING FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create sample data for demonstration
    create_sample_data_for_demo()
    
    # Demonstrate CLI usage
    run_report_via_cli()
    
    # Demonstrate programmatic usage
    run_report_programmatically()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Generated reports can be found in the 'reports' directory.")
    print("Open the HTML files in a web browser to view the interactive reports.")


if __name__ == "__main__":
    main() 