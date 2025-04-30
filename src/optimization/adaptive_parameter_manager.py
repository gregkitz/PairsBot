"""
Adaptive Parameter Manager Module

This module provides functionality to manage and apply regime-specific parameters
for intraday trading based on detected market regimes.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class AdaptiveParameterManager:
    """
    Manages and applies regime-specific parameters for intraday trading.
    
    This class loads a configuration file with regime-specific parameters,
    detects the current market regime, and provides the appropriate parameters
    for the current market conditions.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 update_frequency: int = 60):
        """
        Initialize the adaptive parameter manager.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to configuration file with regime-specific parameters
        config_dict : dict, optional
            Configuration dictionary with regime-specific parameters
        update_frequency : int
            Frequency in minutes to update regime detection and parameters
        """
        self.config = self._load_config(config_file) if config_file else config_dict or {}
        self.update_frequency = update_frequency
        
        # Initialize regime classifier
        n_regimes = self.config.get('regime_detection', {}).get('n_regimes', 3)
        self.regime_classifier = MarketRegimeClassifier(n_regimes=n_regimes)
        
        # Track current regime and parameters
        self.current_regime = None
        self.current_parameters = None
        self.regime_history = []
        self.last_update_time = None
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Parameters:
        -----------
        config_file : str
            Path to configuration file
            
        Returns:
        --------
        dict
            Configuration dictionary
        """
        logger.info(f"Loading adaptive parameter configuration from {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration with {len(config.get('regime_responses', {}))} regime responses")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def save_config(self, output_file: str) -> bool:
        """
        Save current configuration to a JSON file.
        
        Parameters:
        -----------
        output_file : str
            Path to output file
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Parameters:
        -----------
        config_dict : dict
            New configuration values
        """
        self.config.update(config_dict)
        logger.info(f"Updated configuration with {len(config_dict)} new values")
    
    def detect_regime(self, prices_df: pd.DataFrame) -> Tuple[int, str]:
        """
        Detect the current market regime.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for multiple instruments
            
        Returns:
        --------
        tuple
            (regime_index, regime_name)
        """
        # Calculate features
        features_df = self.regime_classifier.calculate_features(prices_df)
        
        # Fit classifier if not already fitted
        if not hasattr(self.regime_classifier, 'model') or self.regime_classifier.model is None:
            self.regime_classifier.fit(features_df)
        
        # Predict regime for the latest data point
        latest_features = features_df.iloc[-1:].copy()
        regime_idx = self.regime_classifier.predict(latest_features).iloc[0]
        
        # Get regime description
        regime_desc = self.regime_classifier.get_regime_description(regime_idx)
        
        # Map to simplified regime name
        regime_name = self._map_regime_to_name(regime_desc)
        
        logger.info(f"Detected regime: {regime_idx} ({regime_name})")
        
        # Store current regime
        self.current_regime = (regime_idx, regime_name)
        self.regime_history.append((datetime.now(), regime_idx, regime_name))
        self.last_update_time = datetime.now()
        
        return regime_idx, regime_name
    
    def _map_regime_to_name(self, regime_desc: str) -> str:
        """
        Map regime description to simplified name.
        
        Parameters:
        -----------
        regime_desc : str
            Regime description from classifier
            
        Returns:
        --------
        str
            Simplified regime name
        """
        if "High Volatility" in regime_desc:
            return "high_volatility"
        elif "Low Volatility" in regime_desc:
            return "low_volatility"
        elif "Strong Trend" in regime_desc:
            return "trending"
        elif "Weak Trend" in regime_desc and "High Correlation" in regime_desc:
            return "mean_reverting"
        elif "Low Correlation" in regime_desc:
            return "low_correlation"
        else:
            return "default"
    
    def get_parameters_for_regime(self, regime_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for a specific regime.
        
        Parameters:
        -----------
        regime_name : str, optional
            Regime name to get parameters for. If None, uses current regime.
            
        Returns:
        --------
        dict
            Parameters for the specified regime
        """
        if regime_name is None:
            if self.current_regime is None:
                logger.warning("No current regime detected. Using fallback parameters.")
                return self.config.get('fallback_parameters', {})
            regime_name = self.current_regime[1]
        
        # Get parameters for the specified regime
        regime_responses = self.config.get('regime_responses', {})
        
        if regime_name in regime_responses:
            parameters = regime_responses[regime_name]
            logger.info(f"Using parameters for regime '{regime_name}'")
            return parameters
        
        # If regime not found, use fallback parameters
        logger.warning(f"No parameters found for regime '{regime_name}'. Using fallback parameters.")
        return self.config.get('fallback_parameters', {})
    
    def should_update(self) -> bool:
        """
        Determine if parameters should be updated based on update frequency.
        
        Returns:
        --------
        bool
            True if parameters should be updated, False otherwise
        """
        if self.last_update_time is None:
            return True
        
        time_since_update = datetime.now() - self.last_update_time
        update_minutes = self.config.get('update_frequency', {}).get('parameter_update_minutes', self.update_frequency)
        
        return time_since_update > timedelta(minutes=update_minutes)
    
    def adapt_parameters(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Adapt parameters based on current market regime.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for multiple instruments
            
        Returns:
        --------
        dict
            Adapted parameters for current market regime
        """
        # Check if update is needed
        if not self.should_update():
            if self.current_parameters is not None:
                return self.current_parameters
        
        # Detect current regime
        _, regime_name = self.detect_regime(prices_df)
        
        # Get parameters for current regime
        parameters = self.get_parameters_for_regime(regime_name)
        
        # Store current parameters
        self.current_parameters = parameters
        
        return parameters
    
    def adapt_pair_config(self, pair_config: Dict[str, Any], prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Adapt pair configuration based on current market regime.
        
        Parameters:
        -----------
        pair_config : dict
            Original pair configuration
        prices_df : pd.DataFrame
            DataFrame with price data for multiple instruments
            
        Returns:
        --------
        dict
            Adapted pair configuration
        """
        # Get adapted parameters
        adapted_params = self.adapt_parameters(prices_df)
        
        # Create a copy of the original config
        adapted_config = pair_config.copy()
        
        # Update config with adapted parameters
        if 'config' not in adapted_config:
            adapted_config['config'] = {}
        
        # Update entry/exit parameters
        adapted_config['config']['entry_threshold'] = adapted_params.get('entry_zscore', 2.0)
        adapted_config['config']['exit_threshold'] = adapted_params.get('exit_zscore', 0.5)
        adapted_config['config']['stop_loss_std'] = adapted_params.get('stop_loss_std', 2.5)
        
        # Update risk parameters
        adapted_config['config']['max_risk_per_trade'] = adapted_params.get('max_risk_per_trade', 0.01)
        
        # Update position sizing
        max_allocation = pair_config.get('config', {}).get('max_allocation', 0.1)
        position_size_factor = adapted_params.get('position_size_factor', 1.0)
        adapted_config['config']['adjusted_max_allocation'] = max_allocation * position_size_factor
        
        # Add regime information
        adapted_config['regime_info'] = {
            'regime_name': self.current_regime[1] if self.current_regime else 'unknown',
            'regime_description': adapted_params.get('regime_description', 'Unknown'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add optimal parameters if available
        if 'optimal_parameters' in adapted_params:
            adapted_config['optimal_parameters'] = adapted_params['optimal_parameters']
        
        return adapted_config
    
    def get_regime_history(self, as_dataframe: bool = False) -> Union[List[Tuple], pd.DataFrame]:
        """
        Get history of regime changes.
        
        Parameters:
        -----------
        as_dataframe : bool
            Whether to return as a DataFrame
            
        Returns:
        --------
        list or DataFrame
            History of regime changes
        """
        if not self.regime_history:
            return [] if not as_dataframe else pd.DataFrame(columns=['timestamp', 'regime_idx', 'regime_name'])
        
        if as_dataframe:
            df = pd.DataFrame(self.regime_history, columns=['timestamp', 'regime_idx', 'regime_name'])
            return df
        
        return self.regime_history
    
    def plot_regime_history(self, output_file: Optional[str] = None):
        """
        Plot history of regime changes.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to output file. If None, displays the plot.
        """
        if not self.regime_history:
            logger.warning("No regime history to plot")
            return
        
        # Convert to DataFrame
        df = self.get_regime_history(as_dataframe=True)
        
        # Plot regime history
        plt.figure(figsize=(12, 6))
        
        # Create a scatter plot with different colors for each regime
        for regime in df['regime_idx'].unique():
            regime_df = df[df['regime_idx'] == regime]
            plt.scatter(regime_df['timestamp'], regime_df['regime_idx'],
                      label=f"Regime {regime} ({regime_df['regime_name'].iloc[0]})")
        
        plt.xlabel('Time')
        plt.ylabel('Regime')
        plt.title('Market Regime History')
        plt.legend()
        plt.grid(True)
        
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save plot
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Saved regime history plot to {output_file}")
        else:
            plt.show()


def load_adaptive_parameters(config_file: str) -> Optional[AdaptiveParameterManager]:
    """
    Load adaptive parameters from a configuration file.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration file
        
    Returns:
    --------
    AdaptiveParameterManager or None
        Adaptive parameter manager instance, or None if loading failed
    """
    try:
        manager = AdaptiveParameterManager(config_file=config_file)
        return manager
    except Exception as e:
        logger.error(f"Error loading adaptive parameters: {e}")
        return None 