#!/usr/bin/env python
"""
Quant-Trader - Intraday Statistical Arbitrage System
Main Entry Point

This serves as the central entry point for all system functionality.
"""

import os
import sys
import argparse
import importlib
import logging
from datetime import datetime
import json
import yaml

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from the specified path."""
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.error(f"Unsupported config file format: {config_path}")
            sys.exit(1)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)

def setup_backtest_parser(subparsers):
    """Set up the parser for the backtest command."""
    parser = subparsers.add_parser('backtest', help='Run backtests on pairs')
    parser.add_argument('--config', type=str, default='config/default_config.json',
                      help='Path to configuration file')
    parser.add_argument('--pairs', nargs='+', help='Specific pairs to backtest (e.g., GC SI ZB ZN)')
    parser.add_argument('--ticker-file', type=str, help='File containing ticker pairs to backtest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 5min, 1hour)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to disk')
    parser.add_argument('--commission', type=float, help='Commission rate')
    parser.add_argument('--slippage', type=float, help='Slippage rate')
    
    parser.set_defaults(func=run_backtest)

def setup_intraday_backtest_parser(subparsers):
    """Set up the parser for the intraday-backtest command."""
    parser = subparsers.add_parser('intraday-backtest', help='Run intraday backtests on pairs')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--pairs', nargs='+', help='Specific pairs to backtest (e.g., GC SI ZB ZN)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='5min', help='Timeframe (e.g., 5min, 1hour)')
    parser.add_argument('--output-dir', type=str, default='data/results/intraday', help='Output directory for results')
    parser.add_argument('--use-ml', action='store_true', help='Use ML enhancements for signal generation')
    parser.add_argument('--commission', type=float, default=2.0, help='Commission per contract')
    parser.add_argument('--slippage', type=float, default=1.0, help='Slippage per contract')
    
    parser.set_defaults(func=run_intraday_backtest)

def setup_analyze_pairs_parser(subparsers):
    """Set up the parser for the analyze-pairs command."""
    parser = subparsers.add_parser('analyze-pairs', help='Analyze pairs for cointegration')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    parser.add_argument('--ticker-file', type=str, help='File containing tickers to analyze')
    parser.add_argument('--min-correlation', type=float, default=0.7, help='Minimum correlation threshold')
    parser.add_argument('--output-file', type=str, default='output/pairs_analysis.json', help='Output file for pair analysis')
    parser.add_argument('--lookback-days', type=int, default=60, help='Number of days for lookback period')
    parser.add_argument('--timeframe', type=str, default='1hour', help='Timeframe for analysis')
    
    parser.set_defaults(func=run_analyze_pairs)

def setup_train_models_parser(subparsers):
    """Set up the parser for the train-models command."""
    parser = subparsers.add_parser('train-models', help='Train ML models for signal enhancement')
    parser.add_argument('--config', type=str, default='config/models/ml_training.json',
                      help='Path to configuration file')
    parser.add_argument('--pair', type=str, help='Specific pair to train on (e.g., GC_SI)')
    parser.add_argument('--timeframe', type=str, default='5min', help='Timeframe (e.g., 5min, 1hour)')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-dir', type=str, default='data/models', help='Directory to save trained models')
    parser.add_argument('--test-ratio', type=float, default=0.3, help='Ratio of data to use for testing (0.0-1.0)')
    
    parser.set_defaults(func=run_train_models)

def setup_train_regime_classifier_parser(subparsers):
    """Set up the parser for the train-regime-classifier command."""
    parser = subparsers.add_parser('train-regime-classifier', help='Train market regime classifier')
    parser.add_argument('--config', type=str, default='config/models/regime_classifier.json',
                      help='Path to configuration file')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to use for training')
    parser.add_argument('--timeframe', type=str, default='1day', help='Timeframe (e.g., 1hour, 1day)')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-dir', type=str, default='data/models', help='Directory to save trained models')
    parser.add_argument('--n-regimes', type=int, default=3, help='Number of regimes to detect')
    
    parser.set_defaults(func=run_train_regime_classifier)

def setup_optimize_parameters_parser(subparsers):
    """Set up the parser for the optimize-parameters command."""
    parser = subparsers.add_parser('optimize-parameters', help='Optimize strategy parameters')
    parser.add_argument('--config', type=str, default='config/optimization/parameter_optimization.json',
                      help='Path to configuration file')
    parser.add_argument('--pairs-file', type=str, required=True, help='File containing pairs to optimize')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-regimes', type=int, default=3, help='Number of regimes to detect')
    parser.add_argument('--lookback-window', type=int, default=60, help='Lookback window for regime detection')
    parser.add_argument('--timeframe', type=str, default='1hour', help='Timeframe for data')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--output-dir', type=str, default='output/optimization', help='Directory for output files')
    parser.add_argument('--quick-mode', action='store_true', help='Run a quick optimization with fewer iterations')
    
    parser.set_defaults(func=run_optimize_parameters)

def setup_paper_trading_parser(subparsers):
    """Set up the parser for the paper-trade command."""
    parser = subparsers.add_parser('paper-trade', help='Run paper trading simulation')
    parser.add_argument('--config', type=str, default='config/paper_trading.json',
                      help='Path to configuration file')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--ib-host', type=str, default='127.0.0.1', help='IB Gateway host')
    parser.add_argument('--ib-port', type=int, default=7497, help='IB Gateway port')
    parser.add_argument('--ib-client-id', type=int, default=1, help='IB client ID')
    parser.add_argument('--dashboard-refresh', type=int, default=300, help='Dashboard refresh interval in seconds')
    parser.add_argument('--data-refresh', type=int, default=60, help='Market data refresh interval in seconds')
    parser.add_argument('--auto-shutdown', type=str, default='16:00', help='Time to auto-shutdown (HH:MM)')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable performance dashboard')
    parser.add_argument('--no-alerts', action='store_true', help='Disable alerts')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode without actual IB connection')
    
    parser.set_defaults(func=run_paper_trading)

def setup_api_parser(subparsers):
    """Set up the parser for the api command."""
    parser = subparsers.add_parser('api', help='Start the web API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload on code changes')
    
    parser.set_defaults(func=run_api)

def setup_data_processor_parser(subparsers):
    """Set up the parser for the process-data command."""
    parser = subparsers.add_parser('process-data', help='Process and prepare data for analysis')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to process')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='5min', help='Timeframe to process')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    
    parser.set_defaults(func=run_data_processor)

def setup_worker_parser(subparsers):
    """Set up the parser for the worker command."""
    parser = subparsers.add_parser('worker', help='Start Celery worker')
    parser.add_argument('--concurrency', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--queue', type=str, default='default', help='Queue to process')
    parser.add_argument('--loglevel', type=str, default='info', help='Worker log level')
    
    parser.set_defaults(func=run_worker)

def run_backtest(args):
    """Run the backtest command."""
    try:
        # Import the run_intraday_backtest module
        import run_intraday_backtest
        import os
        import json
        from datetime import datetime
        
        # Create minimal intraday configuration
        intraday_config = {
            "backtest_config": {
                "data_settings": {
                    "start_date": args.start_date if args.start_date else "2023-01-01",
                    "end_date": args.end_date if args.end_date else "2025-03-01",
                    "timeframe": args.timeframe if args.timeframe else "5min"
                }
            },
            "pairs": []
        }
        
        # If specific pairs are provided, add them to the config
        if args.pairs:
            if len(args.pairs) % 2 != 0:
                logger.error("Pairs must be provided in pairs (even number of tickers)")
                return
            
            for i in range(0, len(args.pairs), 2):
                symbol1 = args.pairs[i]
                symbol2 = args.pairs[i+1]
                pair_id = f"{symbol1}_{symbol2}"
                
                pair_config = {
                    "pair_id": pair_id,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "hedge_ratio": 1.0,  # Default, will be calculated dynamically
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 4.0,
                    "config": {
                        "lookback": 60,  # lookback period for z-score calculation
                        "hedge_ratio_method": "ols",  # method for calculating hedge ratio (ols, kalman)
                        "zscore_method": "standard",  # method for calculating z-score (standard, modified)
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
                            "slippage": args.slippage if args.slippage is not None else 1.0,  # slippage in points
                            "commission": args.commission if args.commission is not None else 2.0  # commission in points
                        },
                        "ml_enhancements": {
                            "use_ml_filter": False,  # whether to use ML-based signal filtering
                            "use_ml_entry_timing": False,  # whether to use ML-based entry timing
                            "use_regime_detection": False  # whether to use ML-based regime detection
                        }
                    }
                }
                
                intraday_config["pairs"].append(pair_config)
        else:
            logger.error("No pairs specified for backtest")
            return
        
        # Set up output directory
        output_dir = args.output_dir if args.output_dir else "data/results/intraday"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the temporary config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = "data/configs"
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, f"temp_intraday_config_{timestamp}.json")
        
        with open(config_file, 'w') as f:
            json.dump(intraday_config, f, indent=2)
        
        logger.info(f"Created temporary config file: {config_file}")
        
        # Get parameters from the config
        start_date = intraday_config["backtest_config"]["data_settings"]["start_date"]
        end_date = intraday_config["backtest_config"]["data_settings"]["end_date"]
        timeframe = intraday_config["backtest_config"]["data_settings"]["timeframe"]
        
        logger.info(f"Running backtest for pairs: {', '.join([p['pair_id'] for p in intraday_config['pairs']])}")
        logger.info(f"Period: {start_date} to {end_date}, Timeframe: {timeframe}")
        
        # Set up args for run_intraday_backtest
        parsed_args = argparse.Namespace()
        parsed_args.config = config_file
        parsed_args.output_dir = output_dir
        parsed_args.no_plots = args.no_plots if hasattr(args, 'no_plots') else False
        parsed_args.save_plots = args.save_plots if hasattr(args, 'save_plots') else True
        
        # Run the backtest
        run_intraday_backtest.main(parsed_args)
        
    except ImportError:
        logger.error("Could not import run_intraday_backtest module")
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()

def run_intraday_backtest(args):
    """Run the intraday backtest command."""
    try:
        import run_intraday_backtest
        
        # Run the intraday backtest directly
        run_intraday_backtest.main(args)
    except ImportError:
        logger.error("Could not import run_intraday_backtest module")
    except Exception as e:
        logger.error(f"Error running intraday backtest: {e}")
        import traceback
        traceback.print_exc()

def run_analyze_pairs(args):
    """Run the analyze-pairs command."""
    try:
        # We'll look for the pairs analysis code in src/test_pair_selection.py or similar modules
        from src.test_pair_selection import main as analyze_pairs_main
        
        # Set sys.argv for the analyze_pairs_main function instead of passing args directly
        sys.argv = [sys.argv[0]]  # Clear command line arguments
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.tickers:
            sys.argv.extend(["--tickers"] + args.tickers)
        if args.ticker_file:
            sys.argv.extend(["--ticker-file", args.ticker_file])
        if args.min_correlation:
            sys.argv.extend(["--min-correlation", str(args.min_correlation)])
        if args.output_file:
            sys.argv.extend(["--output-file", args.output_file])
        if args.lookback_days:
            sys.argv.extend(["--lookback-days", str(args.lookback_days)])
        if args.timeframe:
            sys.argv.extend(["--timeframe", args.timeframe])
        
        # Run the pair analysis
        analyze_pairs_main()
    except ImportError:
        logger.error("Could not import pair analysis module. Make sure the necessary modules are installed.")
    except Exception as e:
        logger.error(f"Error analyzing pairs: {e}")
        import traceback
        traceback.print_exc()

def run_train_models(args):
    """Run the train-models command."""
    try:
        import train_intraday_models
        
        # Run the training directly
        sys.argv = [sys.argv[0]]  # Clear command line arguments
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.start_date:
            sys.argv.extend(["--start_date", args.start_date])
        if args.end_date:
            sys.argv.extend(["--end_date", args.end_date])
        if args.timeframe:
            sys.argv.extend(["--timeframe", args.timeframe])
        if args.test_ratio:
            sys.argv.extend(["--test_ratio", str(args.test_ratio)])
            
        train_intraday_models.main()
    except ImportError:
        logger.error("Could not import train_intraday_models module")
    except Exception as e:
        logger.error(f"Error training models: {e}")
        import traceback
        traceback.print_exc()

def run_train_regime_classifier(args):
    """Run the train-regime-classifier command."""
    try:
        import train_market_regime_classifier
        
        # Run the training directly
        train_market_regime_classifier.main()
    except ImportError:
        logger.error("Could not import train_market_regime_classifier module")
    except Exception as e:
        logger.error(f"Error training regime classifier: {e}")
        import traceback
        traceback.print_exc()

def run_optimize_parameters(args):
    """Run the optimize-parameters command."""
    try:
        import run_intraday_parameter_optimization
        
        # Convert args to match the run_intraday_parameter_optimization format
        if args.quick_mode:
            # Use simplified args for quick mode
            parsed_args = argparse.Namespace()
            parsed_args.pairs_file = args.pairs_file
            parsed_args.start_date = args.start_date
            parsed_args.end_date = args.end_date
            parsed_args.n_regimes = args.n_regimes
            parsed_args.lookback_window = args.lookback_window
            parsed_args.timeframe = args.timeframe
            parsed_args.n_jobs = args.n_jobs
            parsed_args.output_dir = args.output_dir
            
            run_intraday_parameter_optimization.main(parsed_args)
        else:
            # Use full args
            run_intraday_parameter_optimization.main(args)
    except ImportError:
        logger.error("Could not import run_intraday_parameter_optimization module")
    except Exception as e:
        logger.error(f"Error optimizing parameters: {e}")
        import traceback
        traceback.print_exc()

def run_paper_trading(args):
    """Run the paper-trading command."""
    try:
        import run_ml_paper_trader
        
        # Convert args to match the run_ml_paper_trader format
        if hasattr(args, 'timeframe') and hasattr(args, 'pairs'):
            # Create custom args with user-specified parameters
            parsed_args = argparse.Namespace()
            parsed_args.config = args.config
            parsed_args.capital = args.capital
            parsed_args.ib_host = args.ib_host
            parsed_args.ib_port = args.ib_port
            parsed_args.ib_client_id = args.ib_client_id
            parsed_args.dashboard_refresh = args.dashboard_refresh
            parsed_args.data_refresh = args.data_refresh
            parsed_args.auto_shutdown = args.auto_shutdown
            parsed_args.no_dashboard = args.no_dashboard
            parsed_args.no_alerts = args.no_alerts
            parsed_args.test_mode = args.test_mode
            
            run_ml_paper_trader.main(parsed_args)
        else:
            # Use default args
            run_ml_paper_trader.main(args)
    except ImportError:
        logger.error("Could not import run_ml_paper_trader module")
    except Exception as e:
        logger.error(f"Error running paper trading: {e}")
        import traceback
        traceback.print_exc()

def run_api(args):
    """Run the API server."""
    try:
        import run_web
        
        # Start the API server
        sys.argv = [sys.argv[0]]  # Clear command line arguments
        if args.host:
            sys.argv.extend(["--host", args.host])
        if args.port:
            sys.argv.extend(["--port", str(args.port)])
        if args.reload:
            sys.argv.append("--reload")
            
        run_web.main()
    except ImportError:
        logger.error("Could not import run_web module")
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        import traceback
        traceback.print_exc()

def run_data_processor(args):
    """Run the data processor."""
    try:
        from scripts.download_data import main as download_data_main
        
        # Process data for the specified symbols
        download_data_main()
    except ImportError:
        logger.error("Could not import data processing module")
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

def run_worker(args):
    """Run the Celery worker."""
    try:
        from subprocess import Popen
        
        # Build the Celery command
        cmd = [
            "celery",
            "-A", "tasks",
            "worker",
            "--concurrency", str(args.concurrency),
            "--loglevel", args.loglevel
        ]
        
        if args.queue != "default":
            cmd.extend(["-Q", args.queue])
        
        # Start the Celery worker
        logger.info(f"Starting Celery worker with command: {' '.join(cmd)}")
        process = Popen(cmd)
        
        # Wait for the process to complete
        process.wait()
    except Exception as e:
        logger.error(f"Error starting Celery worker: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Quant-Trader: Intraday Statistical Arbitrage System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add version information
    parser.add_argument('--version', action='version', version='Quant-Trader 1.0.0')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Set up individual command parsers
    setup_backtest_parser(subparsers)
    setup_intraday_backtest_parser(subparsers)
    setup_analyze_pairs_parser(subparsers)
    setup_train_models_parser(subparsers)
    setup_train_regime_classifier_parser(subparsers)
    setup_optimize_parameters_parser(subparsers)
    setup_paper_trading_parser(subparsers)
    setup_api_parser(subparsers)
    setup_data_processor_parser(subparsers)
    setup_worker_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return
    
    # Execute the command
    args.func(args)

if __name__ == '__main__':
    main() 