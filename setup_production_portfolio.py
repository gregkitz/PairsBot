"""
Script to set up our optimized portfolio for production monitoring.

This script registers our portfolio with the distributed system via the API.
"""

import os
import json
import requests
import pandas as pd
import logging
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
API_URL = "http://localhost:8000"

def register_portfolio(portfolio_file):
    """
    Register a portfolio configuration with the API for monitoring.
    
    Parameters:
    -----------
    portfolio_file : str
        Path to the portfolio configuration file
        
    Returns:
    --------
    dict
        Response from the API
    """
    logger.info(f"Registering portfolio: {portfolio_file}")
    
    # Load portfolio config
    with open(portfolio_file, 'r') as f:
        portfolio_config = json.load(f)
    
    # Create API request payload
    payload = {
        "name": f"Optimized Portfolio {datetime.now().strftime('%Y%m%d')}",
        "description": "Optimized pairs trading portfolio with low correlation pairs",
        "config": portfolio_config,
        "auto_start": True
    }
    
    # Send request to API
    url = f"{API_URL}/portfolio/register"
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Portfolio registered successfully with ID: {result.get('portfolio_id')}")
        return result
    else:
        logger.error(f"Failed to register portfolio: {response.text}")
        return None

def setup_monitors(portfolio_id):
    """
    Set up monitoring for a portfolio.
    
    Parameters:
    -----------
    portfolio_id : str
        ID of the registered portfolio
        
    Returns:
    --------
    dict
        Response from the API
    """
    logger.info(f"Setting up monitors for portfolio: {portfolio_id}")
    
    # Define monitoring configuration
    monitor_config = {
        "portfolio_id": portfolio_id,
        "monitors": [
            {
                "type": "drawdown",
                "threshold": -0.15,
                "action": "notify"
            },
            {
                "type": "variance",
                "lookback": 20,
                "threshold": 2.0,
                "action": "pause_trading"
            },
            {
                "type": "correlation_breakdown",
                "lookback": 60,
                "threshold": 0.3,
                "action": "recalibrate"
            }
        ]
    }
    
    # Send request to API
    url = f"{API_URL}/monitoring/setup"
    response = requests.post(url, json=monitor_config)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Monitors set up successfully: {result}")
        return result
    else:
        logger.error(f"Failed to set up monitors: {response.text}")
        return None

def schedule_recalibration(portfolio_id, interval_days=30):
    """
    Schedule periodic recalibration for a portfolio.
    
    Parameters:
    -----------
    portfolio_id : str
        ID of the registered portfolio
    interval_days : int
        Interval in days for recalibration
        
    Returns:
    --------
    dict
        Response from the API
    """
    logger.info(f"Scheduling recalibration for portfolio: {portfolio_id}")
    
    # Define recalibration configuration
    recalibration_config = {
        "portfolio_id": portfolio_id,
        "interval_days": interval_days,
        "params": {
            "lookback_days": 252,
            "min_half_life": 1.0,
            "max_half_life": 30.0,
            "adf_threshold": 0.05,
            "auto_update": True
        }
    }
    
    # Send request to API
    url = f"{API_URL}/portfolio/schedule_recalibration"
    response = requests.post(url, json=recalibration_config)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Recalibration scheduled successfully: {result}")
        return result
    else:
        logger.error(f"Failed to schedule recalibration: {response.text}")
        return None

def main():
    """Main function to set up the production portfolio."""
    # Find most recent portfolio file
    portfolio_dir = "data/results/portfolio"
    portfolio_files = [f for f in os.listdir(portfolio_dir) if f.endswith('.json') and f.startswith('simple_portfolio_')]
    
    if not portfolio_files:
        logger.error("No portfolio configuration files found")
        return
    
    # Sort by timestamp in filename
    portfolio_files.sort(reverse=True)
    latest_portfolio = os.path.join(portfolio_dir, portfolio_files[0])
    
    logger.info(f"Using latest portfolio: {latest_portfolio}")
    
    # Register portfolio with API
    result = register_portfolio(latest_portfolio)
    
    if result:
        portfolio_id = result.get('portfolio_id')
        
        # Set up monitoring
        setup_monitors(portfolio_id)
        
        # Schedule recalibration
        schedule_recalibration(portfolio_id, interval_days=30)
        
        logger.info(f"Portfolio {portfolio_id} is now set up for production monitoring")
        
        # Wait for initial status update
        time.sleep(5)
        
        # Get portfolio status
        url = f"{API_URL}/portfolio/{portfolio_id}/status"
        response = requests.get(url)
        
        if response.status_code == 200:
            status = response.json()
            logger.info(f"Portfolio status: {status}")
            
            # Print summary
            print("\nProduction Portfolio Setup Complete")
            print(f"Portfolio ID: {portfolio_id}")
            print(f"Status: {status.get('status', 'Unknown')}")
            print(f"Pairs: {len(status.get('active_pairs', []))}")
            print(f"Allocation: ${status.get('allocation', 0):,.2f}")
            print("\nMonitoring is active. Use the web interface at http://localhost:8000/docs for management.")
    else:
        logger.error("Failed to set up production portfolio")

if __name__ == "__main__":
    main() 