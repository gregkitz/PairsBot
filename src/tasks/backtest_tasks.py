"""
Backtesting tasks for the Celery task queue.
These tasks handle backtesting operations that can be run asynchronously.
"""

import os
import logging
import sys
from datetime import datetime
import json
from src.tasks.celery_app import celery_app

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='run_backtest')
def run_backtest(self, pairs, start_date, end_date, timeframe, use_ml=False, config_path=None):
    """
    Run a backtest for specified pairs and timeframe.
    
    Args:
        pairs (list): List of pairs to backtest (e.g., ["GC_SI", "ZN_ZB"])
        start_date (str): Start date for backtest data (YYYY-MM-DD)
        end_date (str): End date for backtest data (YYYY-MM-DD)
        timeframe (str): Timeframe (e.g., 5min, 1hour)
        use_ml (bool): Whether to use ML enhancements
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Backtest results summary
    """
    try:
        logger.info(f"Starting backtest for pairs {pairs} from {start_date} to {end_date}")
        
        # Update task state for monitoring
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'status': 'loading_data'
            }
        )
        
        # Create a temporary configuration file for the backtest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = os.path.join("data", "configs")
        os.makedirs(config_dir, exist_ok=True)
        temp_config_file = os.path.join(config_dir, f"temp_backtest_config_{timestamp}.json")
        
        # Create backtest configuration
        backtest_config = {
            "backtest_config": {
                "data_settings": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "timeframe": timeframe
                }
            },
            "pairs": []
        }
        
        # Process pairs and add them to config
        for pair in pairs:
            # Split pair ID to get individual symbols
            symbols = pair.split("_")
            if len(symbols) != 2:
                logger.warning(f"Invalid pair format: {pair}. Expected format: SYMBOL1_SYMBOL2")
                continue
                
            symbol1, symbol2 = symbols
            
            # Create pair configuration
            pair_config = {
                "pair_id": pair,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "hedge_ratio": 1.0,  # Default, will be calculated dynamically
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "stop_loss_threshold": 4.0,
                "config": {
                    "lookback": 60,  # lookback period for z-score calculation
                    "hedge_ratio_method": "ols",  # method for calculating hedge ratio
                    "zscore_method": "standard",  # method for calculating z-score
                    "use_log_prices": False,  # whether to use log prices for spread calculation
                    "entry_zscore": 2.0,  # z-score threshold for entry
                    "exit_zscore": 0.5,  # z-score threshold for exit
                    "stop_loss_zscore": 4.0,  # z-score threshold for stop loss
                    "stop_loss_std": 4.0,  # stop loss in standard deviations
                    "take_profit_zscore": 0.0,  # z-score threshold for take profit (0 = disabled)
                    "max_holding_period": 120,  # maximum holding period in minutes
                    "intraday_constraints": {
                        "enabled": True,  # whether to apply intraday constraints
                        "start_trading_time": "09:30",  # start trading time (market hours)
                        "end_trading_time": "15:45",  # end trading time (market hours)
                        "close_all_positions_time": "15:55",  # time to close all positions
                        "avoid_overnight": True  # whether to avoid overnight positions
                    },
                    "transaction_costs": {
                        "slippage": 1.0,  # slippage in points
                        "commission": 2.0  # commission in points
                    },
                    "ml_enhancements": {
                        "use_ml_filter": use_ml,  # whether to use ML-based signal filtering
                        "use_ml_entry_timing": use_ml,  # whether to use ML-based entry timing
                        "use_regime_detection": use_ml  # whether to use ML-based regime detection
                    }
                }
            }
            
            backtest_config["pairs"].append(pair_config)
        
        # Save the temporary config
        with open(temp_config_file, 'w') as f:
            json.dump(backtest_config, f, indent=2)
        
        logger.info(f"Created temporary config file: {temp_config_file}")
        
        # Set up output directory
        output_dir = os.path.join("data", "results", "backtest_tasks", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Update task state
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'status': 'running_backtest',
                'config_file': temp_config_file,
                'output_dir': output_dir
            }
        )
        
        # Import module here to avoid circular imports
        try:
            # Add parent directory to sys.path to allow importing run_intraday_backtest
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            # Import the backtest runner
            import run_intraday_backtest
            
            # Create arguments namespace for the backtest
            from argparse import Namespace
            backtest_args = Namespace(
                config=temp_config_file,
                output_dir=output_dir,
                no_plots=False,
                save_plots=True
            )
            
            # Run the backtest
            results = run_intraday_backtest.main(backtest_args)
            
            # Process and return results
            logger.info("Backtest completed successfully")
            return {
                'status': 'completed',
                'message': 'Backtest completed successfully',
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'use_ml': use_ml,
                'results_summary': results.get('summary', {}),
                'output_dir': output_dir,
                'plots_dir': os.path.join(output_dir, 'plots')
            }
            
        except ImportError as e:
            logger.error(f"Error importing run_intraday_backtest: {str(e)}")
            raise Exception(f"Failed to import backtest module: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='run_intraday_backtest')
def run_intraday_backtest(self, pairs, start_date, end_date, timeframe, use_ml=True, config_path=None):
    """
    Run an intraday backtest with ML enhancements.
    
    Args:
        pairs (list): List of pairs to backtest
        start_date (str): Start date for backtest data (YYYY-MM-DD)
        end_date (str): End date for backtest data (YYYY-MM-DD)
        timeframe (str): Timeframe (e.g., 5min)
        use_ml (bool): Whether to use ML enhancements
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Backtest results summary
    """
    try:
        logger.info(f"Starting intraday backtest for pairs {pairs} from {start_date} to {end_date}")
        
        # Update task state for monitoring
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'use_ml': use_ml,
                'status': 'preparing_config'
            }
        )
        
        # Either use provided config or create a temporary one
        if config_path and os.path.exists(config_path):
            temp_config_file = config_path
            logger.info(f"Using provided config file: {config_path}")
            
            # Load the config to check and update ML settings
            with open(config_path, 'r') as f:
                backtest_config = json.load(f)
                
            # Update ML settings if necessary
            for pair_config in backtest_config.get('pairs', []):
                if 'config' in pair_config and 'ml_enhancements' in pair_config['config']:
                    pair_config['config']['ml_enhancements'] = {
                        'use_ml_filter': use_ml,
                        'use_ml_entry_timing': use_ml,
                        'use_regime_detection': use_ml
                    }
            
            # Save the updated config
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = os.path.join("data", "configs")
            os.makedirs(config_dir, exist_ok=True)
            temp_config_file = os.path.join(config_dir, f"temp_intraday_config_{timestamp}.json")
            
            with open(temp_config_file, 'w') as f:
                json.dump(backtest_config, f, indent=2)
        else:
            # Create a temporary configuration file for the backtest
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = os.path.join("data", "configs")
            os.makedirs(config_dir, exist_ok=True)
            temp_config_file = os.path.join(config_dir, f"temp_intraday_config_{timestamp}.json")
            
            # Create backtest configuration
            backtest_config = {
                "backtest_config": {
                    "data_settings": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "timeframe": timeframe
                    }
                },
                "pairs": []
            }
            
            # Process pairs and add them to config
            for pair in pairs:
                # Split pair ID to get individual symbols
                symbols = pair.split("_")
                if len(symbols) != 2:
                    logger.warning(f"Invalid pair format: {pair}. Expected format: SYMBOL1_SYMBOL2")
                    continue
                    
                symbol1, symbol2 = symbols
                
                # Create pair configuration
                pair_config = {
                    "pair_id": pair,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "hedge_ratio": 1.0,  # Default, will be calculated dynamically
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 4.0,
                    "config": {
                        "lookback": 60,  # lookback period for z-score calculation
                        "hedge_ratio_method": "ols",  # method for calculating hedge ratio
                        "zscore_method": "standard",  # method for calculating z-score
                        "use_log_prices": False,  # whether to use log prices for spread calculation
                        "entry_zscore": 2.0,  # z-score threshold for entry
                        "exit_zscore": 0.5,  # z-score threshold for exit
                        "stop_loss_zscore": 4.0,  # z-score threshold for stop loss
                        "stop_loss_std": 4.0,  # stop loss in standard deviations
                        "take_profit_zscore": 0.0,  # z-score threshold for take profit (0 = disabled)
                        "max_holding_period": 120,  # maximum holding period in minutes
                        "intraday_constraints": {
                            "enabled": True,  # whether to apply intraday constraints
                            "start_trading_time": "09:30",  # start trading time (market hours)
                            "end_trading_time": "15:45",  # end trading time (market hours)
                            "close_all_positions_time": "15:55",  # time to close all positions
                            "avoid_overnight": True  # whether to avoid overnight positions
                        },
                        "transaction_costs": {
                            "slippage": 1.0,  # slippage in points
                            "commission": 2.0  # commission in points
                        },
                        "ml_enhancements": {
                            "use_ml_filter": use_ml,  # whether to use ML-based signal filtering
                            "use_ml_entry_timing": use_ml,  # whether to use ML-based entry timing
                            "use_regime_detection": use_ml  # whether to use ML-based regime detection
                        }
                    }
                }
                
                backtest_config["pairs"].append(pair_config)
            
            # Save the temporary config
            with open(temp_config_file, 'w') as f:
                json.dump(backtest_config, f, indent=2)
            
            logger.info(f"Created temporary config file: {temp_config_file}")
        
        # Set up output directory
        output_dir = os.path.join("data", "results", "intraday_backtest_tasks", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Update task state
        self.update_state(
            state='PROGRESS', 
            meta={
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'use_ml': use_ml,
                'status': 'running_backtest',
                'config_file': temp_config_file,
                'output_dir': output_dir
            }
        )
        
        try:
            # Add parent directory to sys.path to allow importing run_intraday_backtest
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            # Import the backtest runner
            import run_intraday_backtest
            
            # Create arguments namespace for the backtest
            from argparse import Namespace
            backtest_args = Namespace(
                config=temp_config_file,
                output_dir=output_dir,
                no_plots=False,
                save_plots=True,
                use_ml=use_ml
            )
            
            # Run the backtest with ML enhancements
            results = run_intraday_backtest.main(backtest_args)
            
            # Process and return results
            logger.info("Intraday backtest completed successfully")
            
            # Check if the results contain a summary
            if results and isinstance(results, dict) and 'summary' in results:
                summary = results['summary']
            else:
                # Create a minimal summary if not available
                summary = {
                    'pairs': pairs,
                    'timeframe': timeframe,
                    'start_date': start_date, 
                    'end_date': end_date,
                    'use_ml': use_ml
                }
            
            return {
                'status': 'completed',
                'message': 'Intraday backtest completed successfully',
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'use_ml': use_ml,
                'results_summary': summary,
                'output_dir': output_dir,
                'plots_dir': os.path.join(output_dir, 'plots')
            }
            
        except ImportError as e:
            logger.error(f"Error importing run_intraday_backtest: {str(e)}")
            raise Exception(f"Failed to import backtest module: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error running intraday backtest: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise 