#!/usr/bin/env python3
"""
Entry point script for running the Paper Trader.

This script sets up the Python path and imports the paper trader module
properly using the correct package structure.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import and run the paper trader
from src.paper_trading.paper_trader import PaperTrader

if __name__ == '__main__':
    # Create a paper trader instance
    paper_trader = PaperTrader(
        initial_capital=100000.0,
        ib_host='127.0.0.1',
        ib_port=7497,
        ib_client_id=1
    )
    
    # Start the paper trader
    if paper_trader.start():
        print("Paper trader started successfully!")
    else:
        print("Failed to start paper trader.") 