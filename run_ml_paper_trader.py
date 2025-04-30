#!/usr/bin/env python3
"""
Entry point script for running the ML-enhanced Paper Trader.

This script sets up the Paper Trader with ML capabilities, regime detection,
and real-time monitoring dashboard for intraday trading.
"""

import os
import sys
import time
import logging
import signal
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import utilities and ML-enhanced paper trader
from src.utils.json_utils import patch_json_encoder
from src.paper_trading.intraday_ml_paper_trader import IntradayMLPaperTrader

# Create necessary directories
os.makedirs(os.path.join(project_root, 'output', 'paper_trading', 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'output', 'paper_trading', 'dashboard'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'output', 'paper_trading', 'signals'), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                project_root, 
                'output', 
                'paper_trading', 
                'logs', 
                f'paper_trader_{datetime.now().strftime("%Y%m%d")}.log'
            )
        )
    ]
)
logger = logging.getLogger(__name__)

# Global variables
paper_trader = None
keep_running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop the paper trader."""
    global keep_running
    logger.info("Stopping paper trader...")
    keep_running = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ML-enhanced paper trader')
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=os.path.join(project_root, 'config', 'paper_trading.json'),
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--capital', '-k',
        type=float,
        default=100000.0,
        help='Initial capital for paper trading'
    )
    
    parser.add_argument(
        '--ib-host',
        type=str,
        default='127.0.0.1',
        help='Interactive Brokers TWS/Gateway host'
    )
    
    parser.add_argument(
        '--ib-port',
        type=int,
        default=7497,
        help='Interactive Brokers TWS/Gateway port'
    )
    
    parser.add_argument(
        '--ib-client-id',
        type=int,
        default=1,
        help='Interactive Brokers client ID'
    )
    
    parser.add_argument(
        '--dashboard-refresh',
        type=int,
        default=300,
        help='Dashboard refresh interval in seconds'
    )
    
    parser.add_argument(
        '--data-refresh',
        type=int,
        default=60,
        help='Market data refresh interval in seconds'
    )
    
    parser.add_argument(
        '--auto-shutdown',
        type=str,
        default='16:00',
        help='Time to automatically shut down paper trading (HH:MM)'
    )
    
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable performance dashboard'
    )
    
    parser.add_argument(
        '--no-alerts',
        action='store_true',
        help='Disable alerts'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode without actual IB connection'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    global paper_trader, keep_running
    
    # Apply JSON encoder patch for datetime objects
    patch_json_encoder()
    logger.info("JSON encoder patched for datetime handling")
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create ML-enhanced paper trader
        logger.info("Creating ML-enhanced paper trader...")
        
        # Check if config file exists
        if not os.path.exists(args.config):
            logger.warning(f"Configuration file not found: {args.config}")
            logger.info("Using default configuration")
        
        # Create paper trader
        paper_trader = IntradayMLPaperTrader(
            initial_capital=args.capital,
            ib_host=args.ib_host,
            ib_port=args.ib_port,
            ib_client_id=args.ib_client_id,
            ml_config_path=args.config,
            dashboard_update_interval_seconds=args.dashboard_refresh,
            refresh_interval_seconds=args.data_refresh,
            enable_dashboard=not args.no_dashboard,
            enable_alerts=not args.no_alerts,
            auto_shutdown_time=args.auto_shutdown,
            test_mode=args.test_mode
        )
        
        # Configure pairs from config file
        if os.path.exists(args.config):
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                
                if 'pairs' in config:
                    paper_trader.set_pairs(config['pairs'])
                    logger.info(f"Loaded {len(config['pairs'])} pairs from configuration")
            except Exception as e:
                logger.error(f"Error setting pairs from config: {e}")
        
        # Start paper trader
        logger.info("Starting ML-enhanced paper trader...")
        if paper_trader.start():
            logger.info("ML-enhanced paper trader started successfully!")
            
            # Print dashboard URL if enabled
            if not args.no_dashboard:
                dashboard_path = os.path.join(paper_trader.output_dir, "dashboard", "index.html")
                logger.info(f"Dashboard available at: file://{os.path.abspath(dashboard_path)}")
            
            # Keep running until stopped
            while keep_running:
                time.sleep(1)
            
            # Stop paper trader
            logger.info("Stopping ML-enhanced paper trader...")
            paper_trader.stop()
            logger.info("ML-enhanced paper trader stopped.")
        else:
            logger.error("Failed to start ML-enhanced paper trader!")
    
    except Exception as e:
        logger.error(f"Error running ML-enhanced paper trader: {e}", exc_info=True)
        if paper_trader and hasattr(paper_trader, 'is_running') and paper_trader.is_running:
            paper_trader.stop()

if __name__ == '__main__':
    main() 