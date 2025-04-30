
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

# Import example data generator
def generate_sample_data(n_days=252):
    """Generate sample price data for backtesting."""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price series with correlation
    common_factor = np.random.randn(n_days).cumsum()
    
    series1 = common_factor + np.random.randn(n_days).cumsum() * 0.5 + 100
    series2 = common_factor * 1.2 + np.random.randn(n_days).cumsum() * 0.7 + 50
    
    # Convert to pandas Series
    asset1 = pd.Series(series1, index=dates)
    asset2 = pd.Series(series2, index=dates)
    
    return {'asset1': asset1, 'asset2': asset2}

def demo_strategies():
    """Run a demonstration of different strategy variants."""
    logger.info("Generating sample data...")
    data = generate_sample_data()
    
    # Print some statistics about the data
    asset1 = data['asset1']
    asset2 = data['asset2']
    
    correlation = asset1.corr(asset2)
    logger.info(f"Asset Correlation: {correlation:.4f}")
    
    # Plot the price series
    plt.figure(figsize=(12, 6))
    plt.plot(asset1.index, asset1, label='Asset 1')
    plt.plot(asset2.index, asset2, label='Asset 2')
    plt.title('Sample Asset Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Calculate the spread
    spread = asset1 - 0.5 * asset2
    
    # Plot the spread
    plt.figure(figsize=(12, 6))
    plt.plot(spread.index, spread, label='Spread')
    plt.title('Price Spread')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    logger.info("Strategy demonstration completed.")
    return data

if __name__ == "__main__":
    demo_strategies()
            