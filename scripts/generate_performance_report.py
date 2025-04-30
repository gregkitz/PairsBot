#!/usr/bin/env python
"""
Performance Report Generator Script

This script generates daily performance reports for the intraday trading system.
It can be used to generate reports for backtest results, paper trading, or live trading.

Usage:
    python scripts/generate_performance_report.py --mode [backtest|paper|live] --days [N] --output [filename]
    python scripts/generate_performance_report.py --date [YYYY-MM-DD] --mode [backtest|paper|live] --output [filename]
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Union, Optional, Any, Tuple

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reporting import BacktestReportGenerator, calculate_performance_metrics
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/performance_report.log')
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory_path: str):
    """Ensure the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def get_data_for_date_range(start_date: datetime, end_date: datetime, mode: str) -> Tuple[pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """
    Load trading data for the specified date range and trading mode.
    
    Parameters:
    -----------
    start_date : datetime
        Start date for the report
    end_date : datetime
        End date for the report
    mode : str
        Trading mode ('backtest', 'paper', or 'live')
        
    Returns:
    --------
    Tuple containing:
    - equity_curve (pd.Series): Portfolio equity values
    - trades (pd.DataFrame): Trade data
    - benchmark (pd.Series, optional): Benchmark data if available
    """
    # Define data directory based on mode
    if mode == 'backtest':
        data_dir = 'data/backtest_results'
    elif mode == 'paper':
        data_dir = 'data/paper_trading'
    elif mode == 'live':
        data_dir = 'data/live_trading'
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Ensure the data directory exists
    ensure_directory_exists(data_dir)
    
    # Format date strings for file names
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Load equity curve
    equity_file = os.path.join(data_dir, f'equity_{start_str}_to_{end_str}.csv')
    if not os.path.exists(equity_file):
        # Try loading individual day files and combining them
        logger.info("Combined equity file not found, attempting to load daily files")
        equity_data = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_file = os.path.join(data_dir, f'equity_{date_str}.csv')
            if os.path.exists(daily_file):
                daily_equity = pd.read_csv(daily_file, index_col=0, parse_dates=True)
                equity_data.append(daily_equity)
            current_date += timedelta(days=1)
        
        if equity_data:
            equity_curve = pd.concat(equity_data).sort_index()
        else:
            logger.error(f"No equity data found for {start_str} to {end_str}")
            return None, None, None
    else:
        equity_curve = pd.read_csv(equity_file, index_col=0, parse_dates=True).squeeze()
    
    # Load trades
    trades_file = os.path.join(data_dir, f'trades_{start_str}_to_{end_str}.csv')
    if not os.path.exists(trades_file):
        # Try loading individual day files and combining them
        logger.info("Combined trades file not found, attempting to load daily files")
        trades_data = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_file = os.path.join(data_dir, f'trades_{date_str}.csv')
            if os.path.exists(daily_file):
                daily_trades = pd.read_csv(daily_file, parse_dates=['entry_time', 'exit_time'])
                trades_data.append(daily_trades)
            current_date += timedelta(days=1)
        
        if trades_data:
            trades = pd.concat(trades_data, ignore_index=True)
        else:
            logger.warning(f"No trade data found for {start_str} to {end_str}")
            trades = pd.DataFrame()
    else:
        trades = pd.read_csv(trades_file, parse_dates=['entry_time', 'exit_time'])
    
    # Load benchmark (if available)
    benchmark_file = os.path.join(data_dir, f'benchmark_{start_str}_to_{end_str}.csv')
    if os.path.exists(benchmark_file):
        benchmark = pd.read_csv(benchmark_file, index_col=0, parse_dates=True).squeeze()
    else:
        logger.info("No benchmark data found")
        benchmark = None
    
    return equity_curve, trades, benchmark

def generate_report(start_date: datetime, end_date: datetime, mode: str, output_file: str):
    """
    Generate a performance report for the specified date range.
    
    Parameters:
    -----------
    start_date : datetime
        Start date for the report
    end_date : datetime
        End date for the report
    mode : str
        Trading mode ('backtest', 'paper', or 'live')
    output_file : str
        Output file path for the HTML report
    """
    logger.info(f"Generating {mode} trading report for {start_date.date()} to {end_date.date()}")
    
    # Get data for the specified date range
    equity_curve, trades, benchmark = get_data_for_date_range(start_date, end_date, mode)
    
    if equity_curve is None:
        logger.error("Failed to load trading data, cannot generate report")
        return
    
    # Load regime information if available
    try:
        regime_classifier = MarketRegimeClassifier()
        current_regime = regime_classifier.detect_regime(equity_curve)
        regime_info = {
            'regime': current_regime,
            'description': regime_classifier.describe_regime(current_regime)
        }
        logger.info(f"Current market regime: {current_regime}")
    except Exception as e:
        logger.warning(f"Failed to load regime information: {e}")
        regime_info = {
            'regime': 'unknown',
            'description': 'Market regime detection failed'
        }
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=trades,
        benchmark=benchmark,
        risk_free_rate=0.02  # Default risk-free rate (can be made configurable)
    )
    
    # Add regime information to metrics
    metrics['market_regime'] = regime_info
    
    # Generate report title and description
    title = f"{mode.capitalize()} Trading Performance Report"
    description = f"""
    Performance report for the intraday ML trading system in {mode} mode for the period
    from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
    Market Regime: {regime_info['regime']} - {regime_info['description']}
    """
    
    # Create report generator
    ensure_directory_exists('reports')
    report_generator = BacktestReportGenerator(
        output_dir='reports'
    )
    
    # Generate HTML report
    report_path = report_generator.generate_report(
        title=title,
        strategy_description=description,
        equity_curve=equity_curve,
        trades=trades,
        benchmark=benchmark,
        risk_free_rate=0.02,
        output_filename=output_file,
        additional_metrics={'market_regime': regime_info}
    )
    
    # Also save metrics to JSON for programmatic access
    metrics_file = os.path.splitext(report_path)[0] + '_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v) 
                  for k, v in metrics.items()}, f, indent=2)
    
    logger.info(f"Report generated successfully at: {report_path}")
    logger.info(f"Metrics saved to: {metrics_file}")
    
    return report_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate performance reports for intraday trading')
    
    # Date options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days', type=int, help='Number of past days to include in the report')
    date_group.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD) or "today" for today\'s report')
    date_group.add_argument('--range', type=str, help='Date range in format YYYY-MM-DD:YYYY-MM-DD')
    
    # Other options
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper', 'live'], default='paper',
                      help='Trading mode (default: paper)')
    parser.add_argument('--output', type=str, help='Output file name (default: auto-generated based on date)')
    
    return parser.parse_args()

def main():
    """Main function for the performance report generator."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine date range for the report
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if args.days:
        # Use last N days
        start_date = end_date - timedelta(days=args.days)
    elif args.date:
        # Use specific date
        if args.date.lower() == 'today':
            start_date = end_date
        else:
            try:
                start_date = datetime.strptime(args.date, '%Y-%m-%d')
                end_date = start_date  # For a single day report
            except ValueError:
                logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD format.")
                return
    elif args.range:
        # Use date range
        try:
            date_parts = args.range.split(':')
            if len(date_parts) != 2:
                logger.error("Invalid date range format. Use YYYY-MM-DD:YYYY-MM-DD")
                return
            start_date = datetime.strptime(date_parts[0], '%Y-%m-%d')
            end_date = datetime.strptime(date_parts[1], '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date range format: {args.range}. Use YYYY-MM-DD:YYYY-MM-DD format.")
            return
    
    # Create default output filename if not specified
    if args.output:
        output_file = args.output
    else:
        date_str = start_date.strftime('%Y%m%d')
        if start_date != end_date:
            date_str += f"_to_{end_date.strftime('%Y%m%d')}"
        output_file = f"{args.mode}_performance_{date_str}.html"
    
    # Ensure output file has .html extension
    if not output_file.endswith('.html'):
        output_file += '.html'
    
    # Generate the report
    report_path = generate_report(start_date, end_date, args.mode, output_file)
    
    if report_path:
        print(f"Report generated successfully: {report_path}")
    else:
        print("Failed to generate report. Check logs for details.")

if __name__ == "__main__":
    # Ensure log directory exists
    ensure_directory_exists('logs')
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Error generating performance report: {e}")
        sys.exit(1) 