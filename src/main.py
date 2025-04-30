"""
Main module for the Intraday Statistical Arbitrage System.

This module serves as the central controller for the system, coordinating between
different components based on the configuration settings.
"""

import os
import json
import logging
from datetime import datetime

from src.data_processor import DataProcessor
from src.cointegration import PairFinder
from src.spread_analytics import SpreadAnalyzer
from src.signals import SignalGenerator, SignalType
from src.risk_management import PositionSizer, RiskManager
from src.backtest import BacktestEngine
from src.utils import create_directory

def setup_logging(config):
    """Set up logging configuration."""
    log_level = getattr(logging, config['system']['log_level'])
    log_file = config['system']['log_file']
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from the specified path."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise Exception(f"Error loading configuration file: {e}")

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    # Only update if args value is not None
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    if args.timeframe:
        config['data']['timeframe'] = args.timeframe
    if args.output_dir:
        config['system']['output_dir'] = args.output_dir
    if args.no_plots:
        config['system']['show_plots'] = False
    if args.save_plots:
        config['system']['save_plots'] = True
    if args.commission is not None:
        config['backtest']['commission'] = args.commission
    if args.slippage is not None:
        config['backtest']['slippage'] = args.slippage
    
    return config

def run_backtest(config, pairs=None, logger=None):
    """Run the backtesting process."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Initialize data processor
    data_processor = DataProcessor(
        historical_dir=config['data']['historical_dir'],
        timeframe=config['data']['timeframe']
    )
    
    # If specific pairs are provided, use them
    if pairs:
        logger.info(f"Using specified pairs: {pairs}")
        # Assuming pairs is a list of tuples [(ticker1, ticker2), ...]
        all_tickers = set()
        for pair in pairs:
            all_tickers.update(pair)
        
        # Load data for all tickers
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        ticker_data = {}
        
        for ticker in all_tickers:
            try:
                data = data_processor.load_data(ticker, start_date, end_date)
                if data is not None and not data.empty:
                    ticker_data[ticker] = data
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
        
        # Convert pairs to the format expected by the system
        trading_pairs = []
        for ticker1, ticker2 in pairs:
            if ticker1 in ticker_data and ticker2 in ticker_data:
                trading_pairs.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'data1': ticker_data[ticker1],
                    'data2': ticker_data[ticker2]
                })
        
        if not trading_pairs:
            logger.error("No valid pairs found with available data")
            return
    else:
        # Automatically find cointegrated pairs
        logger.info("Finding cointegrated pairs...")
        available_tickers = data_processor.get_available_tickers()
        
        if not available_tickers:
            logger.error("No ticker data available")
            return
        
        # Load data for all available tickers
        ticker_data = {}
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        
        for ticker in available_tickers:
            try:
                data = data_processor.load_data(ticker, start_date, end_date)
                if data is not None and not data.empty:
                    ticker_data[ticker] = data
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
        
        if len(ticker_data) < 2:
            logger.error("Insufficient tickers with data for pair finding")
            return
        
        # Initialize pair finder
        pair_finder = PairFinder(
            min_correlation=config['cointegration']['min_correlation'],
            max_half_life=config['cointegration']['max_half_life'],
            min_half_life=config['cointegration']['min_half_life'],
            pvalue_threshold=config['cointegration']['pvalue_threshold'],
            train_test_split=config['cointegration']['train_test_split'],
            min_cointegration_pct=config['cointegration']['min_cointegration_pct']
        )
        
        # Find cointegrated pairs
        try:
            cointegrated_pairs = pair_finder.find_pairs(ticker_data)
            
            if not cointegrated_pairs:
                logger.error("No cointegrated pairs found")
                return
            
            max_pairs = config['pairs_selection']['max_pairs']
            trading_pairs = cointegrated_pairs[:max_pairs]
            
            logger.info(f"Found {len(trading_pairs)} trading pairs")
        except Exception as e:
            logger.error(f"Error finding pairs: {e}")
            return
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        commission=config['backtest']['commission'],
        slippage=config['backtest']['slippage'],
        trade_delay=config['backtest']['trade_delay'],
        allow_simultaneous_positions=config['backtest']['allow_simultaneous_positions']
    )
    
    # Process each pair
    results = []
    for pair in trading_pairs:
        try:
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            data1 = pair['data1']
            data2 = pair['data2']
            
            logger.info(f"Analyzing pair: {ticker1}-{ticker2}")
            
            # Initialize spread analyzer
            spread_analyzer = SpreadAnalyzer(
                hedge_ratio_method=config['spread_analytics']['hedge_ratio_method'],
                window_size=config['spread_analytics']['window_size'],
                half_life=config['spread_analytics']['half_life'],
                use_log_prices=config['spread_analytics']['use_log_prices']
            )
            
            # Calculate spread
            spread_data = spread_analyzer.calculate_spread(data1, data2)
            
            # Initialize signal generator
            signal_generator = SignalGenerator(
                entry_threshold=config['signal_generation']['entry_threshold'],
                exit_threshold=config['signal_generation']['exit_threshold'],
                stop_loss_threshold=config['signal_generation']['stop_loss_threshold'],
                take_profit_threshold=config['signal_generation']['take_profit_threshold'],
                signal_type=getattr(SignalType, config['signal_generation']['signal_type']),
                use_trailing_stop=config['signal_generation']['use_trailing_stop']
            )
            
            # Generate signals
            signals = signal_generator.generate_signals(spread_data)
            
            # Initialize position sizer
            position_sizer = PositionSizer(
                account_size=config['risk_management']['account_size'],
                max_risk_per_trade=config['risk_management']['max_risk_per_trade'],
                max_allocation=config['risk_management']['max_allocation'],
                method=config['risk_management']['position_sizing_method']
            )
            
            # Initialize risk manager
            risk_manager = RiskManager(
                max_drawdown=config['risk_management']['max_drawdown_limit'],
                daily_loss_limit=config['risk_management']['daily_loss_limit'],
                max_positions=config['risk_management']['max_pairs']
            )
            
            # Run backtest
            pair_result = backtest_engine.run_backtest(
                data1=data1,
                data2=data2,
                spread_data=spread_data,
                signals=signals,
                position_sizer=position_sizer,
                risk_manager=risk_manager
            )
            
            # Save result
            results.append({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'metrics': pair_result['metrics'],
                'trades': pair_result['trades'],
                'equity_curve': pair_result['equity_curve']
            })
            
            # Generate plots if enabled
            if config['system']['show_plots'] or config['system']['save_plots']:
                from src.visualization import (
                    plot_pair_prices,
                    plot_spread_zscore,
                    plot_signals_positions,
                    plot_equity_curve,
                    plot_backtest_summary
                )
                
                plot_dir = config['system']['plot_dir']
                create_directory(plot_dir)
                pair_name = f"{ticker1}_{ticker2}"
                
                # Plot pairs
                plot_pair_prices(
                    data1, data2, 
                    ticker1, ticker2,
                    save_path=f"{plot_dir}/{pair_name}_prices.png" if config['system']['save_plots'] else None,
                    show=config['system']['show_plots']
                )
                
                # Plot spread
                plot_spread_zscore(
                    spread_data,
                    save_path=f"{plot_dir}/{pair_name}_spread.png" if config['system']['save_plots'] else None,
                    show=config['system']['show_plots']
                )
                
                # Plot signals and positions
                plot_signals_positions(
                    spread_data, signals, pair_result['positions'],
                    save_path=f"{plot_dir}/{pair_name}_signals.png" if config['system']['save_plots'] else None,
                    show=config['system']['show_plots']
                )
                
                # Plot equity curve
                plot_equity_curve(
                    pair_result['equity_curve'],
                    save_path=f"{plot_dir}/{pair_name}_equity.png" if config['system']['save_plots'] else None,
                    show=config['system']['show_plots']
                )
                
                # Plot backtest summary
                plot_backtest_summary(
                    pair_result,
                    save_path=f"{plot_dir}/{pair_name}_summary.png" if config['system']['save_plots'] else None,
                    show=config['system']['show_plots']
                )
            
        except Exception as e:
            logger.error(f"Error processing pair {ticker1}-{ticker2}: {e}")
    
    # Save aggregated results
    if results:
        output_dir = config['system']['output_dir']
        create_directory(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        with open(f"{output_dir}/backtest_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        logger.info(f"Backtest results saved to {output_dir}/backtest_results_{timestamp}.json")
    
    return results

def main(args):
    """Main function to run the system."""
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("Starting Intraday Statistical Arbitrage System")
    
    # Handle different modes
    if args.mode == 'backtest':
        # Handle specific pairs if provided
        pairs = None
        if args.pairs and len(args.pairs) >= 2:
            # Convert flat list to pairs
            pairs = [(args.pairs[i], args.pairs[i+1]) 
                     for i in range(0, len(args.pairs), 2)
                     if i+1 < len(args.pairs)]
            
        # Handle ticker file if provided
        elif args.ticker_file:
            try:
                with open(args.ticker_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                
                # Convert flat list to pairs
                pairs = [(tickers[i], tickers[i+1]) 
                         for i in range(0, len(tickers), 2)
                         if i+1 < len(tickers)]
                
                if not pairs:
                    logger.error("No valid pairs found in the ticker file")
                    return
            except Exception as e:
                logger.error(f"Error reading ticker file: {e}")
                return
        
        # Run backtest
        results = run_backtest(config, pairs, logger)
        
        if results:
            logger.info("Backtest completed successfully")
        else:
            logger.warning("Backtest completed with no results")
            
    elif args.mode == 'paper_trade':
        logger.info("Paper trading not yet implemented")
        # TODO: Implement paper trading mode
        
    elif args.mode == 'live_trade':
        logger.info("Live trading not yet implemented")
        # TODO: Implement live trading mode
    
    logger.info("System execution completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Intraday Statistical Arbitrage System'
    )
    parser.add_argument('--config', type=str, default='config/default_config.json',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper_trade', 'live_trade'],
                      default='backtest', help='Trading mode')
    
    args = parser.parse_args()
    
    main(args) 