"""
Optimization tasks for the Celery task queue.
These tasks handle parameter optimization operations that can be run asynchronously.
"""

import os
import logging
import sys
from datetime import datetime
import json
from src.tasks.celery_app import celery_app

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='optimize_parameters')
def optimize_parameters(self, pairs_file, start_date, end_date, quick_mode=False, config_path=None):
    """
    Run parameter optimization for pairs specified in a file.
    
    Args:
        pairs_file (str): Path to the file containing pairs to optimize
        start_date (str): Start date for optimization data (YYYY-MM-DD)
        end_date (str): End date for optimization data (YYYY-MM-DD)
        quick_mode (bool): Whether to use quick mode (fewer iterations)
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Optimization results summary
    """
    try:
        logger.info(f"Starting parameter optimization with pairs from {pairs_file}")
        
        # Update task state for monitoring
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs_file': pairs_file,
                'start_date': start_date,
                'end_date': end_date,
                'quick_mode': quick_mode,
                'status': 'loading_data'
            }
        )
        
        # Verify pairs file exists
        if not os.path.exists(pairs_file):
            raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
        
        # Set up output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("data", "results", "optimization", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Update task state
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs_file': pairs_file,
                'start_date': start_date,
                'end_date': end_date,
                'quick_mode': quick_mode,
                'status': 'running_optimization',
                'output_dir': output_dir
            }
        )
        
        try:
            # Add parent directory to sys.path to allow importing run_intraday_parameter_optimization
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            # Import the parameter optimization module
            import run_intraday_parameter_optimization
            
            # Create arguments for the optimization
            from argparse import Namespace
            
            # Use config if provided, otherwise set defaults
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Create args with config values and required parameters
                opt_args = Namespace(
                    pairs_file=pairs_file,
                    start_date=start_date,
                    end_date=end_date,
                    quick_mode=quick_mode,
                    output_dir=output_dir,
                    n_regimes=config.get('n_regimes', 3),
                    lookback_window=config.get('lookback_window', 60),
                    timeframe=config.get('timeframe', '1hour'),
                    commission=config.get('commission', 2.0),
                    slippage=config.get('slippage', 1.0),
                    account_size=config.get('account_size', 25000),
                    n_jobs=config.get('n_jobs', 1)
                )
            else:
                # Use default settings
                opt_args = Namespace(
                    pairs_file=pairs_file,
                    start_date=start_date,
                    end_date=end_date,
                    quick_mode=quick_mode,
                    output_dir=output_dir,
                    n_regimes=3,
                    lookback_window=60,
                    timeframe='1hour',
                    commission=2.0,
                    slippage=1.0,
                    account_size=25000,
                    n_jobs=1
                )
            
            # Run the optimization
            logger.info(f"Running parameter optimization for pairs in {pairs_file}")
            results = run_intraday_parameter_optimization.main(opt_args)
            
            # Find the generated config file in the output directory
            config_files = [f for f in os.listdir(output_dir) if f.startswith('adaptive_parameters') and f.endswith('.json')]
            
            if config_files:
                # Sort by timestamp (newest first)
                config_files.sort(reverse=True)
                adaptive_config_file = os.path.join(output_dir, config_files[0])
                
                # Load the adaptive configuration
                with open(adaptive_config_file, 'r') as f:
                    adaptive_config = json.load(f)
                
                # Return success with config file location
                logger.info(f"Parameter optimization completed successfully")
                return {
                    'status': 'completed',
                    'message': 'Parameter optimization completed successfully',
                    'pairs_file': pairs_file,
                    'start_date': start_date,
                    'end_date': end_date,
                    'quick_mode': quick_mode,
                    'output_dir': output_dir,
                    'adaptive_config_file': adaptive_config_file,
                    'regimes': list(adaptive_config['regime_parameters'].keys()),
                    'plots_dir': os.path.join(output_dir, 'plots')
                }
            else:
                # Return success but note missing config file
                logger.warning("Parameter optimization completed but could not find adaptive parameter config file")
                return {
                    'status': 'completed',
                    'message': 'Parameter optimization completed but could not find adaptive parameter config file',
                    'pairs_file': pairs_file,
                    'start_date': start_date,
                    'end_date': end_date,
                    'quick_mode': quick_mode,
                    'output_dir': output_dir
                }
            
        except ImportError as e:
            logger.error(f"Error importing run_intraday_parameter_optimization: {str(e)}")
            raise Exception(f"Failed to import optimization module: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error optimizing parameters: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='grid_search')
def grid_search(self, pairs, parameter_grid, start_date, end_date, timeframe, config_path=None):
    """
    Run a grid search optimization over a parameter space.
    
    Args:
        pairs (list): List of pairs to optimize
        parameter_grid (dict): Dictionary of parameters and their ranges
        start_date (str): Start date for optimization data (YYYY-MM-DD)
        end_date (str): End date for optimization data (YYYY-MM-DD)
        timeframe (str): Timeframe (e.g., 5min, 1hour)
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Grid search results summary
    """
    try:
        # Import here to avoid circular imports
        # This will be implemented with the actual grid search
        # from src.optimization.grid_search.grid_search import GridSearchOptimizer
        
        logger.info(f"Starting grid search for pairs {pairs}")
        
        # Update task state for monitoring
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs': pairs,
                'parameter_count': len(parameter_grid),
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe
            }
        )
        
        # Placeholder for actual implementation
        # This will be replaced with actual grid search execution
        logger.info("Grid search task implementation pending")
        
        # Return placeholder results
        return {
            'status': 'not_implemented',
            'message': 'Grid search task implementation is pending',
            'pairs': pairs,
            'parameter_grid': parameter_grid,
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe
        }
        
    except Exception as e:
        logger.error(f"Error running grid search: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='genetic_optimization')
def genetic_optimization(self, pairs, parameter_space, generations, population_size, start_date, end_date, timeframe, config_path=None):
    """
    Run genetic algorithm optimization for strategy parameters.
    
    Args:
        pairs (list): List of pairs to optimize
        parameter_space (dict): Dictionary of parameters and their ranges
        generations (int): Number of generations to run
        population_size (int): Size of population in each generation
        start_date (str): Start date for optimization data (YYYY-MM-DD)
        end_date (str): End date for optimization data (YYYY-MM-DD)
        timeframe (str): Timeframe (e.g., 5min, 1hour)
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Genetic optimization results summary
    """
    try:
        # Import here to avoid circular imports
        # This will be implemented with the actual genetic optimizer
        # from src.optimization.genetic_algorithm.genetic_algorithm import GeneticOptimizer
        
        logger.info(f"Starting genetic optimization for pairs {pairs} with {generations} generations")
        
        # Update task state for monitoring
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs': pairs,
                'generations': generations,
                'population_size': population_size,
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe
            }
        )
        
        # Placeholder for actual implementation
        # This will be replaced with actual genetic optimization execution
        logger.info("Genetic optimization task implementation pending")
        
        # Return placeholder results
        return {
            'status': 'not_implemented',
            'message': 'Genetic optimization task implementation is pending',
            'pairs': pairs,
            'parameter_space': parameter_space,
            'generations': generations,
            'population_size': population_size,
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe
        }
        
    except Exception as e:
        logger.error(f"Error running genetic optimization: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise 