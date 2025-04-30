#!/usr/bin/env python3
"""
Entry point script for the Intraday Statistical Arbitrage System.

This script provides a command-line interface to run the pairs trading system.
"""

import os
import sys
import time
import argparse

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main module
from src.main import main as run_main

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Intraday Statistical Arbitrage System for Futures Pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/default_config.json',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper_trade', 'live_trade'],
                      default='backtest', help='Trading mode')
    
    # Data options
    parser.add_argument('--start_date', type=str, 
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, 
                      help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str,
                      choices=['1min', '5min', '15min', '30min', '1hour', '4hour', '1day', '1week'],
                      help='Data timeframe')
    
    # Pair selection
    parser.add_argument('--pairs', type=str, nargs='+',
                      help='Specific pairs to test (e.g., CL GC)')
    parser.add_argument('--ticker_file', type=str,
                      help='File listing ticker symbols to test')
    
    # Output options
    parser.add_argument('--output_dir', type=str,
                      help='Output directory for results')
    parser.add_argument('--no_plots', action='store_true',
                      help='Disable plot generation')
    parser.add_argument('--save_plots', action='store_true',
                      help='Save plots to file')
    
    # Backtesting options
    parser.add_argument('--commission', type=float,
                      help='Commission rate')
    parser.add_argument('--slippage', type=float,
                      help='Slippage rate')
    
    return parser.parse_args()

def main():
    """Main entry point function."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create required directories if they don't exist
    for directory in ['data', 'logs', 'output']:
        os.makedirs(directory, exist_ok=True)
    
    # Run the main function from src.main
    sys.argv = [sys.argv[0]]  # Clear argv to avoid argparse conflicts
    run_main(args)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 