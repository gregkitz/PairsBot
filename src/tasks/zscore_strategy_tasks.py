"""
Z-Score Strategy tasks for the Celery task queue.
These tasks handle z-score strategy backtesting operations that can be run asynchronously.
"""

import os
import logging
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from src.tasks.celery_app import celery_app
from src.backtest.zscore_strategy_backtest import ZScoreStrategyBacktest

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='run_zscore_backtest')
def run_zscore_backtest(self, pairs, start_date, end_date, timeframe='1d', 
                        zscore_params=None, use_log_prices=False, transaction_costs=None, 
                        output_dir=None):
    """
    Run a z-score strategy backtest for specified pairs and timeframe.
    
    Parameters:
    -----------
    pairs : list
        List of pairs to backtest (e.g., ["GC_SI", "ZN_ZB"])
    start_date : str
        Start date for backtest data (YYYY-MM-DD)
    end_date : str
        End date for backtest data (YYYY-MM-DD)
    timeframe : str
        Timeframe (e.g., 1d, 1h, 5min)
    zscore_params : dict, optional
        Parameters for z-score strategy 
        (entry_threshold, exit_threshold, stop_loss_threshold, window_size, etc.)
    use_log_prices : bool
        Whether to use log prices for spread calculation
    transaction_costs : dict, optional
        Transaction costs parameters (commission_per_trade, slippage_per_trade)
    output_dir : str, optional
        Output directory for results
        
    Returns:
    --------
    dict: 
        Backtest results summary
    """
    try:
        logger.info(f"Starting z-score backtest for pairs {pairs} from {start_date} to {end_date}")
        
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
        
        # Set default parameters if not provided
        if zscore_params is None:
            zscore_params = {
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.0,
                'window_size': 20,
                'max_holding_period': None,
                'use_trailing_stop': False,
                'calculate_method': 'rolling'
            }
            
        if transaction_costs is None:
            transaction_costs = {
                'commission_per_trade': 2.0,
                'slippage_per_trade': 1.0
            }
        
        # Create timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join("data", "results", "zscore_strategy", timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each pair
        results = {}
        
        for pair_id in pairs:
            try:
                # Split pair ID to get individual symbols
                symbols = pair_id.split("_")
                if len(symbols) != 2:
                    logger.warning(f"Invalid pair format: {pair_id}. Expected format: SYMBOL1_SYMBOL2")
                    continue
                    
                symbol1, symbol2 = symbols
                
                # Update task state
                self.update_state(
                    state='PROGRESS', 
                    meta={
                        'pairs': pairs,
                        'current_pair': pair_id,
                        'timeframe': timeframe,
                        'start_date': start_date,
                        'end_date': end_date,
                        'status': 'loading_data'
                    }
                )
                
                # Load price data
                data = load_price_data([symbol1, symbol2], start_date, end_date, timeframe)
                
                if data is None or symbol1 not in data or symbol2 not in data:
                    logger.error(f"Failed to load data for pair {pair_id}")
                    continue
                
                # Update task state
                self.update_state(
                    state='PROGRESS', 
                    meta={
                        'pairs': pairs,
                        'current_pair': pair_id,
                        'timeframe': timeframe,
                        'start_date': start_date,
                        'end_date': end_date,
                        'status': 'running_backtest'
                    }
                )
                
                # Run backtest
                backtest = ZScoreStrategyBacktest(
                    entry_threshold=zscore_params.get('entry_threshold', 2.0),
                    exit_threshold=zscore_params.get('exit_threshold', 0.5),
                    stop_loss_threshold=zscore_params.get('stop_loss_threshold', 3.0),
                    window_size=zscore_params.get('window_size', 20),
                    max_holding_period=zscore_params.get('max_holding_period'),
                    use_trailing_stop=zscore_params.get('use_trailing_stop', False),
                    use_time_filter=zscore_params.get('use_time_filter', False),
                    commission_per_trade=transaction_costs.get('commission_per_trade', 2.0),
                    slippage_per_trade=transaction_costs.get('slippage_per_trade', 1.0),
                    account_size=zscore_params.get('account_size', 100000),
                    max_position_size=zscore_params.get('max_position_size', 0.25),
                    calculate_method=zscore_params.get('calculate_method', 'rolling'),
                    use_log_prices=use_log_prices
                )
                
                # Run the backtest
                backtest_result = backtest.backtest(data[symbol1], data[symbol2])
                
                # Generate plots
                if not zscore_params.get('no_plots', False):
                    # Save plots to output directory
                    plot_file = os.path.join(output_dir, f"zscore_backtest_{pair_id}_{timestamp}.png")
                    backtest.plot_results(save_path=plot_file)
                
                # Save results to CSV files
                signals_file = os.path.join(output_dir, f"zscore_signals_{pair_id}_{timestamp}.csv")
                backtest_result['signals'].to_csv(signals_file)
                
                equity_file = os.path.join(output_dir, f"zscore_equity_{pair_id}_{timestamp}.csv")
                backtest_result['equity_curve'].to_csv(equity_file)
                
                # Save metrics to JSON
                metrics_file = os.path.join(output_dir, f"zscore_metrics_{pair_id}_{timestamp}.json")
                with open(metrics_file, 'w') as f:
                    json.dump(backtest_result['metrics'], f, indent=2)
                
                # Store results
                results[pair_id] = {
                    'metrics': backtest_result['metrics'],
                    'files': {
                        'signals': signals_file,
                        'equity': equity_file,
                        'metrics': metrics_file,
                        'plot': plot_file if not zscore_params.get('no_plots', False) else None
                    }
                }
                
                logger.info(f"Completed backtest for pair {pair_id}")
                
            except Exception as e:
                logger.error(f"Error processing pair {pair_id}: {str(e)}", exc_info=True)
                results[pair_id] = {'error': str(e)}
        
        # Create summary report
        summary_file = os.path.join(output_dir, f"zscore_summary_{timestamp}.json")
        summary = {
            'timestamp': timestamp,
            'parameters': {
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'zscore_params': zscore_params,
                'transaction_costs': transaction_costs
            },
            'results': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Return the summary results
        return {
            'status': 'completed',
            'message': 'Z-Score strategy backtest completed successfully',
            'pairs': pairs,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'output_dir': output_dir,
            'summary_file': summary_file,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error running z-score backtest: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


def load_price_data(symbols, start_date, end_date, timeframe):
    """
    Load price data for the specified symbols.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to load data for
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    timeframe : str
        Timeframe (e.g., 1d, 1h, 5min)
        
    Returns:
    --------
    dict:
        Dictionary with symbols as keys and price series as values
    """
    try:
        # Import the data loading module
        from src.data_processor.data_loader import load_market_data
        
        # Load data
        data = load_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        if data is None or data.empty:
            logger.error(f"No data loaded for symbols {symbols}")
            return None
        
        # Convert to dictionary of series
        result = {}
        for symbol in symbols:
            if symbol in data.columns:
                result[symbol] = data[symbol]
            else:
                logger.warning(f"Symbol {symbol} not found in loaded data")
        
        return result
        
    except ImportError:
        logger.error("Could not import data_loader module")
        
        # Fallback to loading data from CSV files (for testing purposes)
        try:
            # Try to load from data directory
            data_dir = os.path.join("data", "market_data")
            result = {}
            
            for symbol in symbols:
                try:
                    csv_file = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
                        if 'close' in df.columns:
                            result[symbol] = df['close']
                        elif 'Close' in df.columns:
                            result[symbol] = df['Close']
                        else:
                            # Use the first column as price
                            result[symbol] = df.iloc[:, 0]
                    else:
                        logger.warning(f"CSV file for {symbol} not found")
                except Exception as e:
                    logger.error(f"Error loading data for symbol {symbol}: {str(e)}")
            
            if not result:
                logger.error("No data loaded from CSV files")
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"Error in fallback data loading: {str(e)}")
            return None
    
    except Exception as e:
        logger.error(f"Error loading price data: {str(e)}")
        return None


@celery_app.task(bind=True, name='optimize_zscore_parameters')
def optimize_zscore_parameters(self, pairs, start_date, end_date, timeframe='1d', 
                              param_grid=None, use_log_prices=False, transaction_costs=None,
                              output_dir=None):
    """
    Optimize parameters for the z-score strategy.
    
    Parameters:
    -----------
    pairs : list
        List of pairs to optimize (e.g., ["GC_SI", "ZN_ZB"])
    start_date : str
        Start date for optimization data (YYYY-MM-DD)
    end_date : str
        End date for optimization data (YYYY-MM-DD)
    timeframe : str
        Timeframe (e.g., 1d, 1h, 5min)
    param_grid : dict, optional
        Parameter grid for optimization
    use_log_prices : bool
        Whether to use log prices for spread calculation
    transaction_costs : dict, optional
        Transaction costs parameters (commission_per_trade, slippage_per_trade)
    output_dir : str, optional
        Output directory for results
        
    Returns:
    --------
    dict:
        Optimization results summary
    """
    try:
        logger.info(f"Starting z-score parameter optimization for pairs {pairs}")
        
        # Update task state
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
        
        # Set default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'entry_threshold': [1.5, 2.0, 2.5, 3.0],
                'exit_threshold': [0.0, 0.5, 1.0],
                'window_size': [10, 20, 30, 50],
                'calculate_method': ['rolling', 'ewm']
            }
            
        if transaction_costs is None:
            transaction_costs = {
                'commission_per_trade': 2.0,
                'slippage_per_trade': 1.0
            }
        
        # Create timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join("data", "results", "zscore_optimization", timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each pair
        optimization_results = {}
        
        for pair_id in pairs:
            try:
                # Split pair ID to get individual symbols
                symbols = pair_id.split("_")
                if len(symbols) != 2:
                    logger.warning(f"Invalid pair format: {pair_id}. Expected format: SYMBOL1_SYMBOL2")
                    continue
                    
                symbol1, symbol2 = symbols
                
                # Update task state
                self.update_state(
                    state='PROGRESS', 
                    meta={
                        'pairs': pairs,
                        'current_pair': pair_id,
                        'timeframe': timeframe,
                        'start_date': start_date,
                        'end_date': end_date,
                        'status': 'loading_data'
                    }
                )
                
                # Load price data
                data = load_price_data([symbol1, symbol2], start_date, end_date, timeframe)
                
                if data is None or symbol1 not in data or symbol2 not in data:
                    logger.error(f"Failed to load data for pair {pair_id}")
                    continue
                
                # Update task state
                self.update_state(
                    state='PROGRESS', 
                    meta={
                        'pairs': pairs,
                        'current_pair': pair_id,
                        'timeframe': timeframe,
                        'start_date': start_date,
                        'end_date': end_date,
                        'status': 'running_optimization'
                    }
                )
                
                # Implement grid search over parameter combinations
                from itertools import product
                
                # Generate all parameter combinations
                param_keys = list(param_grid.keys())
                param_values = [param_grid[key] for key in param_keys]
                param_combinations = list(product(*param_values))
                
                logger.info(f"Running {len(param_combinations)} parameter combinations for pair {pair_id}")
                
                # Initialize results storage
                pair_results = []
                
                # Import tqdm if available for progress tracking
                try:
                    from tqdm import tqdm
                    param_combinations_iter = tqdm(param_combinations, desc=f"Optimizing {pair_id}")
                except ImportError:
                    param_combinations_iter = param_combinations
                
                # Run backtest for each parameter combination
                for i, combo in enumerate(param_combinations_iter):
                    # Create parameter dictionary
                    params = {key: value for key, value in zip(param_keys, combo)}
                    
                    # Update progress
                    if i % 10 == 0 or i == len(param_combinations) - 1:
                        self.update_state(
                            state='PROGRESS', 
                            meta={
                                'pairs': pairs,
                                'current_pair': pair_id,
                                'timeframe': timeframe,
                                'progress': f"{i+1}/{len(param_combinations)} combinations",
                                'status': 'running_optimization'
                            }
                        )
                    
                    # Set up backtest parameters
                    backtest_params = {
                        'entry_threshold': params.get('entry_threshold', 2.0),
                        'exit_threshold': params.get('exit_threshold', 0.5),
                        'stop_loss_threshold': params.get('stop_loss_threshold', 3.0),
                        'window_size': params.get('window_size', 20),
                        'max_holding_period': params.get('max_holding_period'),
                        'use_trailing_stop': params.get('use_trailing_stop', False),
                        'commission_per_trade': transaction_costs.get('commission_per_trade', 2.0),
                        'slippage_per_trade': transaction_costs.get('slippage_per_trade', 1.0),
                        'calculate_method': params.get('calculate_method', 'rolling'),
                        'use_log_prices': use_log_prices
                    }
                    
                    # Run backtest
                    backtest = ZScoreStrategyBacktest(**backtest_params)
                    result = backtest.backtest(data[symbol1], data[symbol2])
                    
                    # Store results
                    pair_results.append({
                        'parameters': params,
                        'metrics': result['metrics']
                    })
                
                # Sort results by Sharpe ratio (descending)
                pair_results.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
                
                # Save results to CSV
                results_df = pd.DataFrame([
                    {**r['parameters'], **{f"metric_{k}": v for k, v in r['metrics'].items()}}
                    for r in pair_results
                ])
                
                results_file = os.path.join(output_dir, f"zscore_optimization_{pair_id}_{timestamp}.csv")
                results_df.to_csv(results_file, index=False)
                
                # Save best parameters
                best_params = pair_results[0]['parameters']
                best_metrics = pair_results[0]['metrics']
                
                optimization_results[pair_id] = {
                    'best_parameters': best_params,
                    'best_metrics': best_metrics,
                    'results_file': results_file,
                    'num_combinations': len(param_combinations)
                }
                
                logger.info(f"Completed optimization for pair {pair_id}. Best Sharpe: {best_metrics['sharpe_ratio']:.2f}")
                
            except Exception as e:
                logger.error(f"Error optimizing pair {pair_id}: {str(e)}", exc_info=True)
                optimization_results[pair_id] = {'error': str(e)}
        
        # Create summary report
        summary_file = os.path.join(output_dir, f"zscore_optimization_summary_{timestamp}.json")
        summary = {
            'timestamp': timestamp,
            'parameters': {
                'pairs': pairs,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'param_grid': param_grid,
                'transaction_costs': transaction_costs
            },
            'results': optimization_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Return the summary results
        return {
            'status': 'completed',
            'message': 'Z-Score parameter optimization completed successfully',
            'pairs': pairs,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'output_dir': output_dir,
            'summary_file': summary_file,
            'results': optimization_results
        }
        
    except Exception as e:
        logger.error(f"Error running z-score parameter optimization: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise 