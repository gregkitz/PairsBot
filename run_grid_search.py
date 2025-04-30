"""
Script to run a parameter grid search for the pairs trading strategy.

This script uses Celery to perform a grid search over different parameter combinations
to find the optimal parameters for the pairs trading strategy.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import time
import itertools
from tasks import parameter_grid_search, collect_grid_search_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_pair_analysis():
    """
    Load the pair analysis results.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe containing pair analysis results
    """
    analysis_file = "data/results/pairs_analysis.csv"
    if not os.path.exists(analysis_file):
        logger.error(f"Analysis file not found: {analysis_file}")
        return None
    
    return pd.read_csv(analysis_file)

def main():
    """Main function to run the grid search."""
    # Load pair analysis results
    analysis_df = load_pair_analysis()
    if analysis_df is None:
        return
    
    # Get top 3 pairs by half-life
    analysis_df = analysis_df.sort_values('half_life')
    top_pairs = analysis_df.head(3)
    
    # Print selected pairs
    logger.info("Selected pairs for grid search:")
    for idx, pair in top_pairs.iterrows():
        logger.info(f"{pair['symbol1']}-{pair['symbol2']} with half-life {pair['half_life']:.2f} days")
    
    # Create parameter grid for each pair
    for idx, pair in top_pairs.iterrows():
        symbol1 = pair['symbol1']
        symbol2 = pair['symbol2']
        hedge_ratio = pair['hedge_ratio']
        
        # Base configuration for the backtest
        base_config = {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'hedge_ratio': hedge_ratio,
            'data_dir': 'data/processed',
            'lookback': 20,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'account_size': 100000,
            'commission': 0.0002,
            'slippage': 0.0001,
            'trade_delay': 1
        }
        
        # Parameter grid
        param_grid = {
            'lookback': [10, 20, 30, 50],
            'entry_zscore': [1.5, 2.0, 2.5, 3.0],
            'exit_zscore': [0.0, 0.5, 1.0]
        }
        
        # Run grid search
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/results/grid_search/{symbol1}_{symbol2}_{timestamp}"
        
        logger.info(f"Starting grid search for {symbol1}-{symbol2}")
        logger.info(f"Base config: {base_config}")
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Output directory: {output_dir}")
        
        # Start grid search task
        task = parameter_grid_search.delay(base_config, param_grid, output_dir)
        task_id = task.id
        
        logger.info(f"Grid search task started with ID: {task_id}")
        
        # Wait for task to complete or timeout after 30 minutes
        timeout = 1800  # 30 minutes
        start_time = time.time()
        completed = False
        
        while time.time() - start_time < timeout:
            if task.ready():
                completed = True
                break
            
            logger.info(f"Waiting for grid search task {task_id} to complete...")
            time.sleep(60)  # Check every minute
        
        if not completed:
            logger.warning(f"Grid search task {task_id} did not complete within timeout")
            continue
        
        # Collect results
        try:
            results = task.get()
            logger.info(f"Grid search completed for {symbol1}-{symbol2}")
            logger.info(f"Results: {results}")
            
            # Find best parameters
            best_params = results.get('best_params', None)
            best_metrics = results.get('best_metrics', None)
            
            if best_params and best_metrics:
                logger.info(f"Best parameters for {symbol1}-{symbol2}:")
                logger.info(f"Parameters: {best_params}")
                logger.info(f"Metrics: {best_metrics}")
                
                print(f"\nBest parameters for {symbol1}-{symbol2}:")
                print(f"Parameters: {best_params}")
                print(f"Metrics:")
                for metric, value in best_metrics.items():
                    if isinstance(value, float):
                        if metric in ['total_return', 'annual_return', 'max_drawdown']:
                            print(f"  {metric}: {value:.2%}")
                        else:
                            print(f"  {metric}: {value:.2f}")
                    else:
                        print(f"  {metric}: {value}")
            
        except Exception as e:
            logger.error(f"Error processing grid search results: {e}")

if __name__ == "__main__":
    main() 