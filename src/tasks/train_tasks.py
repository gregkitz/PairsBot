"""
Training tasks for the Celery task queue.
These tasks handle model training operations that can be run asynchronously.
"""

import os
import logging
from src.tasks.celery_app import celery_app

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='train_models')
def train_models(self, pair, timeframe, start_date, end_date, config_path=None):
    """
    Train ML models for a specific pair and timeframe.
    
    Args:
        pair (str): Trading pair (e.g., GC_SI)
        timeframe (str): Timeframe (e.g., 5min, 1hour)
        start_date (str): Start date for training data (YYYY-MM-DD)
        end_date (str): End date for training data (YYYY-MM-DD)
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Training results summary
    """
    try:
        # Import here to avoid circular imports
        from src.ml_enhancements.model_retraining import ModelTrainer
        
        logger.info(f"Starting model training for {pair} on {timeframe} timeframe")
        
        # Update task state for monitoring
        self.update_state(state='PROGRESS', meta={'pair': pair, 'timeframe': timeframe})
        
        # Initialize trainer
        trainer = ModelTrainer(config_path=config_path)
        
        # Run training
        results = trainer.train(
            pair=pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Model training completed for {pair}")
        return {
            'status': 'success',
            'pair': pair,
            'timeframe': timeframe,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}", exc_info=True)
        # Properly handle task failure
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='train_regime_classifier')
def train_regime_classifier(self, tickers, timeframe, n_regimes=3, config_path=None):
    """
    Train a market regime classifier.
    
    Args:
        tickers (list): List of tickers to include
        timeframe (str): Timeframe (e.g., 1day)
        n_regimes (int): Number of regimes to detect
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Training results summary
    """
    try:
        # Import here to avoid circular imports
        from src.ml_enhancements.regime_detection.market_regime_classifier import RegimeClassifierTrainer
        
        logger.info(f"Starting regime classifier training for {len(tickers)} tickers")
        
        # Update task state
        self.update_state(
            state='PROGRESS', 
            meta={'tickers': tickers, 'n_regimes': n_regimes}
        )
        
        # Initialize trainer
        trainer = RegimeClassifierTrainer(config_path=config_path)
        
        # Run training
        results = trainer.train(
            tickers=tickers,
            timeframe=timeframe,
            n_regimes=n_regimes
        )
        
        logger.info(f"Regime classifier training completed")
        return {
            'status': 'success',
            'tickers': tickers,
            'timeframe': timeframe,
            'n_regimes': n_regimes,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error training regime classifier: {str(e)}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise 