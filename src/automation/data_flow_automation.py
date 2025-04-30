import argparse
import logging
import json
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the required modules
from data_processor.data_loader import download_historical_data
from data_processor.feature_calculator import generate_features
from signal_generation.signal_processor import process_signals, combine_pair_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_flow_automation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(parents=True, exist_ok=True)


def load_config(config_file: str) -> Dict:
    """
    Load configuration from a YAML file
    
    Parameters:
    -----------
    config_file : str
        Path to the configuration file
        
    Returns:
    --------
    Dict
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_file}: {str(e)}")
        return {}


def run_data_collection(config: Dict) -> bool:
    """
    Run the data collection process
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        logger.info("Starting data collection process")
        
        symbols = config.get('symbols', [])
        timeframes = config.get('timeframes', ['1day'])
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        data_dir = config.get('data_dir', 'data/historical')
        api_key = config.get('api_key')
        parallel = config.get('parallel', True)
        
        if not symbols:
            logger.error("No symbols specified for data collection")
            return False
        
        logger.info(f"Collecting data for {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        # Download data
        result = download_historical_data(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            api_key=api_key,
            parallel=parallel
        )
        
        # Check result
        success = all(any(tf_data for tf_data in sym_data.values()) for sym_data in result.values())
        if success:
            logger.info("Data collection process completed successfully")
        else:
            logger.warning("Data collection process completed with some errors")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in data collection process: {str(e)}")
        return False


def run_feature_generation(config: Dict) -> bool:
    """
    Run the feature generation process
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        logger.info("Starting feature generation process")
        
        symbols = config.get('symbols', [])
        timeframes = config.get('timeframes', ['1day'])
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        data_dir = config.get('data_dir', 'data/historical')
        feature_dir = config.get('feature_dir', 'data/features')
        parallel = config.get('parallel', True)
        
        if not symbols:
            logger.error("No symbols specified for feature generation")
            return False
        
        logger.info(f"Generating features for {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        # Generate features
        result = generate_features(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            feature_dir=feature_dir,
            parallel=parallel
        )
        
        # Check result
        success = all(any(tf_data for tf_data in sym_data.values()) for sym_data in result.values())
        if success:
            logger.info("Feature generation process completed successfully")
        else:
            logger.warning("Feature generation process completed with some errors")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in feature generation process: {str(e)}")
        return False


def run_signal_processing(config: Dict) -> bool:
    """
    Run the signal processing workflow
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        logger.info("Starting signal processing workflow")
        
        symbols = config.get('symbols', [])
        timeframes = config.get('timeframes', ['1day'])
        strategies = config.get('strategies', ['combined'])
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        feature_dir = config.get('feature_dir', 'data/features')
        signal_dir = config.get('signal_dir', 'data/signals')
        model_dir = config.get('model_dir', 'models')
        use_ml = config.get('use_ml', False)
        parallel = config.get('parallel', True)
        process_pairs = config.get('process_pairs', False)
        
        if not symbols:
            logger.error("No symbols specified for signal processing")
            return False
        
        logger.info(f"Processing signals for {len(symbols)} symbols, {len(timeframes)} timeframes, {len(strategies)} strategies")
        
        # Process signals
        result = process_signals(
            symbols=symbols,
            timeframes=timeframes,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            feature_dir=feature_dir,
            signal_dir=signal_dir,
            model_dir=model_dir,
            use_ml=use_ml,
            parallel=parallel
        )
        
        # Process pair signals if requested
        if process_pairs and len(symbols) >= 2:
            logger.info(f"Processing pair signals for {len(symbols)} symbols")
            
            # Process all possible pairs
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    for tf in timeframes:
                        for strategy in strategies:
                            try:
                                combine_pair_signals(
                                    symbol1=symbols[i],
                                    symbol2=symbols[j],
                                    timeframe=tf,
                                    strategy=strategy,
                                    signal_dir=signal_dir
                                )
                            except Exception as e:
                                logger.error(f"Error processing pair signals for {symbols[i]}-{symbols[j]} {tf} {strategy}: {str(e)}")
        
        # Check result
        success = all(
            all(
                any(strat_data for strat_data in tf_data.values())
                for tf_data in sym_data.values()
            )
            for sym_data in result.values()
        )
        
        if success:
            logger.info("Signal processing workflow completed successfully")
        else:
            logger.warning("Signal processing workflow completed with some errors")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in signal processing workflow: {str(e)}")
        return False


def run_full_pipeline(config: Dict) -> bool:
    """
    Run the full data processing pipeline
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    logger.info("Starting full data processing pipeline")
    
    # Run data collection
    if config.get('run_data_collection', True):
        data_success = run_data_collection(config)
        if not data_success and config.get('stop_on_failure', True):
            logger.error("Data collection failed, stopping pipeline")
            return False
    else:
        logger.info("Skipping data collection")
        data_success = True
    
    # Run feature generation
    if config.get('run_feature_generation', True):
        feature_success = run_feature_generation(config)
        if not feature_success and config.get('stop_on_failure', True):
            logger.error("Feature generation failed, stopping pipeline")
            return False
    else:
        logger.info("Skipping feature generation")
        feature_success = True
    
    # Run signal processing
    if config.get('run_signal_processing', True):
        signal_success = run_signal_processing(config)
        if not signal_success and config.get('stop_on_failure', True):
            logger.error("Signal processing failed, stopping pipeline")
            return False
    else:
        logger.info("Skipping signal processing")
        signal_success = True
    
    # Return overall success
    overall_success = data_success and feature_success and signal_success
    if overall_success:
        logger.info("Full data processing pipeline completed successfully")
    else:
        logger.warning("Full data processing pipeline completed with some errors")
    
    return overall_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the automated data processing pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--data-only", action="store_true", help="Run only data collection")
    parser.add_argument("--features-only", action="store_true", help="Run only feature generation")
    parser.add_argument("--signals-only", action="store_true", help="Run only signal processing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration, exiting")
        sys.exit(1)
    
    # Determine what to run
    if args.data_only:
        config['run_data_collection'] = True
        config['run_feature_generation'] = False
        config['run_signal_processing'] = False
    elif args.features_only:
        config['run_data_collection'] = False
        config['run_feature_generation'] = True
        config['run_signal_processing'] = False
    elif args.signals_only:
        config['run_data_collection'] = False
        config['run_feature_generation'] = False
        config['run_signal_processing'] = True
    else:
        # Run everything by default
        config['run_data_collection'] = True
        config['run_feature_generation'] = True
        config['run_signal_processing'] = True
    
    # Run pipeline
    success = run_full_pipeline(config)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 