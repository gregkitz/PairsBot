import json
import os
import yaml
import logging
from datetime import datetime
from enum import Enum
import pandas as pd


class TimeFilter(Enum):
    """Enum for time-based filtering"""
    NONE = "none"
    US_MARKET_HOURS = "us_market_hours"
    EU_MARKET_HOURS = "eu_market_hours"
    ASIA_MARKET_HOURS = "asia_market_hours"
    CUSTOM = "custom"


class ConfigurationManager:
    """
    Manages configuration settings for the pairs trading system.
    
    This class provides functionality to load, save, and access 
    configuration settings from various sources (JSON, YAML, etc.).
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the ConfigurationManager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
        
        # Set up logging
        self._setup_logging()
    
    def _load_default_config(self):
        """
        Load default configuration settings.
        
        Returns:
        --------
        dict
            Default configuration
        """
        return {
            # System settings
            "system": {
                "output_dir": "output",
                "log_level": "INFO",
                "log_file": "logs/pairs_trading.log",
                "show_plots": True,
                "save_plots": True,
                "plot_dir": "output/plots"
            },
            
            # Data settings
            "data": {
                "historical_dir": "data/historical",
                "start_date": "2020-01-01",
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "resample_freq": None,
                "min_data_points": 252,  # Minimum data points required
                "timeframe": "1day",     # Default timeframe
                "time_filter": TimeFilter.NONE.value,
                "custom_time_filter": {
                    "start_time": "09:30",
                    "end_time": "16:00",
                    "timezone": "US/Eastern"
                }
            },
            
            # Cointegration settings
            "cointegration": {
                "test_method": "johansen",  # Options: adf, johansen, engle_granger
                "pvalue_threshold": 0.05,
                "min_half_life": 1,
                "max_half_life": 60,
                "min_correlation": 0.5,
                "min_mean_rev_strength": 0.1,
                "train_test_split": 0.7,
                "rolling_window": 252,
                "min_cointegration_pct": 0.7
            },
            
            # Pairs selection settings
            "pairs_selection": {
                "max_pairs": 10,
                "ranking_method": "score",  # Options: score, sharpe, return
                "score_weights": {
                    "correlation": 0.25,
                    "cointegration": 0.25,
                    "half_life": 0.25,
                    "hurst": 0.25
                },
                "allow_cross_asset": True,
                "use_log_prices": True,
                "exclude_pairs": []
            },
            
            # Spread calculation settings
            "spread": {
                "hedge_ratio_method": "ols",  # Options: ols, rolling, kalman
                "window_size": 60,
                "half_life": 30,
                "use_log_prices": True,
                "zscore_method": "rolling",  # Options: rolling, ewm, full
                "kalman_settings": {
                    "transition_covariance": 0.01,
                    "observation_covariance": 0.1
                }
            },
            
            # Signal generation settings
            "signals": {
                "entry_threshold": 2.0,
                "exit_threshold": 0.0,
                "stop_loss_threshold": 3.5,
                "take_profit_threshold": None,
                "time_stop": 10,
                "signal_type": "zscore",  # Options: zscore, bollinger, cusum, regression, kalman
                "signal_smoothing": 0,
                "use_trailing_stop": False
            },
            
            # Risk management settings
            "risk_management": {
                "account_size": 100000,
                "max_risk_per_trade": 0.02,
                "max_allocation": 0.25,
                "max_pairs": 5,
                "volatility_lookback_windows": [20, 60],
                "position_sizing_method": "volatility",  # Options: equal, volatility, kelly
                "use_stop_loss": True,
                "use_max_drawdown_limit": True,
                "max_drawdown_limit": 0.1,
                "correlation_threshold": 0.7,
                "daily_loss_limit": 0.03
            },
            
            # Backtest settings
            "backtest": {
                "commission": 0.0005,  # 0.05%
                "slippage": 0.0001,    # 0.01%
                "trade_delay": 1,
                "allow_simultaneous_positions": True
            }
        }
    
    def load_config(self, config_path=None):
        """
        Load configuration from file.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the configuration file. If None, uses the previously set path.
            
        Returns:
        --------
        dict
            Loaded configuration
        """
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            logging.warning("No configuration path specified, using default settings")
            return self.config
        
        if not os.path.exists(self.config_path):
            logging.warning(f"Configuration file not found at {self.config_path}, using default settings")
            return self.config
        
        try:
            _, ext = os.path.splitext(self.config_path)
            
            if ext.lower() == '.json':
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logging.error(f"Unsupported configuration file format: {ext}")
                return self.config
            
            # Update config with loaded values
            self._update_config_recursive(self.config, loaded_config)
            logging.info(f"Configuration loaded from {self.config_path}")
            
            return self.config
            
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            return self.config
    
    def _update_config_recursive(self, target, source):
        """
        Recursively update configuration dictionary.
        
        Parameters:
        -----------
        target : dict
            Target dictionary to update
        source : dict
            Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, config_path=None):
        """
        Save configuration to file.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to save the configuration file. If None, uses the previously set path.
            
        Returns:
        --------
        bool
            True if save was successful, False otherwise
        """
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            logging.error("No configuration path specified for saving")
            return False
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(self.config_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            _, ext = os.path.splitext(self.config_path)
            
            if ext.lower() == '.json':
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4, default=str)
            elif ext.lower() in ['.yaml', '.yml']:
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                logging.error(f"Unsupported configuration file format: {ext}")
                return False
            
            logging.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, section, key=None, default=None):
        """
        Get a configuration value.
        
        Parameters:
        -----------
        section : str
            Configuration section
        key : str, optional
            Configuration key within the section. If None, returns the entire section.
        default : Any, optional
            Default value to return if the key is not found
            
        Returns:
        --------
        Any
            Configuration value
        """
        if section not in self.config:
            logging.warning(f"Configuration section '{section}' not found")
            return default
        
        if key is None:
            return self.config[section]
        
        if key not in self.config[section]:
            logging.warning(f"Configuration key '{key}' not found in section '{section}'")
            return default
        
        return self.config[section][key]
    
    def set(self, section, key, value):
        """
        Set a configuration value.
        
        Parameters:
        -----------
        section : str
            Configuration section
        key : str
            Configuration key within the section
        value : Any
            Value to set
            
        Returns:
        --------
        bool
            True if set was successful, False otherwise
        """
        if section not in self.config:
            logging.warning(f"Creating new configuration section '{section}'")
            self.config[section] = {}
        
        self.config[section][key] = value
        return True
    
    def _setup_logging(self):
        """Set up logging based on configuration."""
        log_level = self.get('system', 'log_level', 'INFO')
        log_file = self.get('system', 'log_file', None)
        
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        
        # Configure logging
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler if log_file is specified
        if log_file:
            # Create directory if it doesn't exist
            directory = os.path.dirname(log_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logging.getLogger().addHandler(file_handler)
    
    def get_time_filter(self):
        """
        Get time filter function based on configuration.
        
        Returns:
        --------
        callable
            Function to filter by time
        """
        time_filter_name = self.get('data', 'time_filter', TimeFilter.NONE.value)
        
        if time_filter_name == TimeFilter.NONE.value:
            return lambda df: df
        
        elif time_filter_name == TimeFilter.US_MARKET_HOURS.value:
            def us_market_hours_filter(df):
                # US market hours: 9:30 AM - 4:00 PM Eastern
                if not isinstance(df.index, pd.DatetimeIndex):
                    return df
                
                market_hours = (df.index.time >= pd.Timestamp('09:30').time()) & \
                               (df.index.time <= pd.Timestamp('16:00').time())
                return df[market_hours]
            
            return us_market_hours_filter
        
        elif time_filter_name == TimeFilter.EU_MARKET_HOURS.value:
            def eu_market_hours_filter(df):
                # European market hours: 8:00 AM - 4:30 PM Central European
                if not isinstance(df.index, pd.DatetimeIndex):
                    return df
                
                market_hours = (df.index.time >= pd.Timestamp('08:00').time()) & \
                               (df.index.time <= pd.Timestamp('16:30').time())
                return df[market_hours]
            
            return eu_market_hours_filter
        
        elif time_filter_name == TimeFilter.ASIA_MARKET_HOURS.value:
            def asia_market_hours_filter(df):
                # Asian market hours: 9:00 AM - 3:00 PM Japan/Tokyo
                if not isinstance(df.index, pd.DatetimeIndex):
                    return df
                
                market_hours = (df.index.time >= pd.Timestamp('09:00').time()) & \
                               (df.index.time <= pd.Timestamp('15:00').time())
                return df[market_hours]
            
            return asia_market_hours_filter
        
        elif time_filter_name == TimeFilter.CUSTOM.value:
            custom_filter = self.get('data', 'custom_time_filter', {})
            start_time = custom_filter.get('start_time', '09:30')
            end_time = custom_filter.get('end_time', '16:00')
            
            def custom_time_filter(df):
                if not isinstance(df.index, pd.DatetimeIndex):
                    return df
                
                market_hours = (df.index.time >= pd.Timestamp(start_time).time()) & \
                               (df.index.time <= pd.Timestamp(end_time).time())
                return df[market_hours]
            
            return custom_time_filter
        
        else:
            logging.warning(f"Unknown time filter: {time_filter_name}, using NONE")
            return lambda df: df 