"""
Time Series Strategy Implementation for the Intraday Statistical Arbitrage System.

This module implements a time-series based approach to pairs trading,
using advanced time-series models like ARIMA, GARCH, VAR to predict
spreads and generate signals.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model

from src.pairs_trading_strategy import PairsTradingStrategy
from src.data_processor.data_processor import DataProcessor
from src.signal_generation.signal_generator import SignalGenerator
from src.risk_management.risk_manager import RiskManager
from src.spread_analytics.spread_analyzer import SpreadAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class TimeSeriesStrategy(PairsTradingStrategy):
    """
    Time Series Strategy for pairs trading that extends the base
    pairs trading strategy with advanced time-series modeling.
    
    This strategy uses ARIMA, GARCH, VAR and other time-series models
    to forecast spreads and generate more robust trading signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time series strategy.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary for the strategy
        """
        super().__init__(config)
        
        # Time-series specific configuration
        self.ts_config = config.get('time_series', {})
        self.model_type = self.ts_config.get('model_type', 'arima')  # 'arima', 'var', 'garch', 'combined'
        self.forecast_horizon = self.ts_config.get('forecast_horizon', 10)
        self.min_history_bars = self.ts_config.get('min_history_bars', 100)
        self.confidence_level = self.ts_config.get('confidence_level', 0.95)
        
        # Model parameters
        self.arima_order = self.ts_config.get('arima_order', (2, 1, 2))
        self.garch_order = self.ts_config.get('garch_order', (1, 1))
        self.var_lags = self.ts_config.get('var_lags', 5)
        
        # Model storage
        self.models = {}
        self.forecasts = {}
        self.last_model_update = {}
        self.model_update_frequency = self.ts_config.get('model_update_frequency', 60)  # minutes
        
        logger.info(f"Time Series Strategy initialized with model type: {self.model_type}")
    
    def analyze_pair(self, pair_id: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze a pair using time-series models and predict future spreads.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        data : Dict[str, pd.DataFrame]
            Historical price data for each symbol in the pair
        
        Returns:
        --------
        Dict[str, Any]
            Analysis results including spread prediction and signals
        """
        # Get the standard pairs trading analysis first
        base_analysis = super().analyze_pair(pair_id, data)
        
        # Check if we have enough data for time-series analysis
        leg1_data = data[self.pairs[pair_id]['leg1']]
        leg2_data = data[self.pairs[pair_id]['leg2']]
        
        if len(leg1_data) < self.min_history_bars or len(leg2_data) < self.min_history_bars:
            logger.warning(f"Not enough data for time-series analysis of pair {pair_id}")
            return base_analysis
        
        # Get the spread series from the base analysis
        spread_series = base_analysis['spread_series']
        
        # Generate time-series forecasts
        forecast_result = self._generate_forecast(pair_id, spread_series)
        
        # Update the base analysis with time-series forecasts
        base_analysis.update({
            'forecast': forecast_result['forecast'],
            'forecast_upper': forecast_result.get('upper', None),
            'forecast_lower': forecast_result.get('lower', None),
            'model_type': self.model_type,
            'model_quality': forecast_result.get('model_quality', None)
        })
        
        # Generate enhanced signals using forecasts
        signals = self._generate_signals_with_forecast(pair_id, base_analysis)
        base_analysis['signals'] = signals
        
        return base_analysis
    
    def _generate_forecast(self, pair_id: str, spread_series: pd.Series) -> Dict[str, Any]:
        """
        Generate forecasts for the spread using time-series models.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        
        Returns:
        --------
        Dict[str, Any]
            Forecast results including predicted values and confidence intervals
        """
        # Check if we need to update the model
        current_time = datetime.now()
        last_update = self.last_model_update.get(pair_id)
        
        if last_update is None or (current_time - last_update) > timedelta(minutes=self.model_update_frequency):
            # Time to update the model
            self._update_model(pair_id, spread_series)
            self.last_model_update[pair_id] = current_time
        
        # Generate forecast based on model type
        if self.model_type == 'arima':
            return self._forecast_arima(pair_id, spread_series)
        elif self.model_type == 'garch':
            return self._forecast_garch(pair_id, spread_series)
        elif self.model_type == 'var':
            return self._forecast_var(pair_id, spread_series)
        elif self.model_type == 'combined':
            return self._forecast_combined(pair_id, spread_series)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using ARIMA as fallback")
            return self._forecast_arima(pair_id, spread_series)
    
    def _update_model(self, pair_id: str, spread_series: pd.Series) -> None:
        """
        Update time-series models with new data.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        """
        logger.info(f"Updating time-series model for pair {pair_id}")
        
        try:
            if self.model_type == 'arima':
                # Fit ARIMA model
                model = ARIMA(spread_series, order=self.arima_order)
                fitted_model = model.fit()
                self.models[pair_id] = fitted_model
                
            elif self.model_type == 'garch':
                # Fit GARCH model
                model = arch_model(spread_series, vol='GARCH', p=self.garch_order[0], q=self.garch_order[1])
                fitted_model = model.fit(disp='off')
                self.models[pair_id] = fitted_model
                
            elif self.model_type == 'var':
                # For VAR, we need both leg prices
                leg1_symbol = self.pairs[pair_id]['leg1']
                leg2_symbol = self.pairs[pair_id]['leg2']
                
                # Prepare data for VAR
                var_data = pd.DataFrame({
                    'spread': spread_series,
                    'leg1_returns': self.data_processor.get_returns(leg1_symbol),
                    'leg2_returns': self.data_processor.get_returns(leg2_symbol)
                }).dropna()
                
                # Fit VAR model
                model = VAR(var_data)
                fitted_model = model.fit(maxlags=self.var_lags)
                self.models[pair_id] = fitted_model
                
            elif self.model_type == 'combined':
                # Fit multiple models
                # ARIMA for mean prediction
                arima_model = ARIMA(spread_series, order=self.arima_order)
                arima_fitted = arima_model.fit()
                
                # GARCH for volatility prediction
                garch_model = arch_model(spread_series, vol='GARCH', p=self.garch_order[0], q=self.garch_order[1])
                garch_fitted = garch_model.fit(disp='off')
                
                self.models[pair_id] = {
                    'arima': arima_fitted,
                    'garch': garch_fitted
                }
            
            logger.info(f"Model update successful for pair {pair_id}")
            
        except Exception as e:
            logger.error(f"Error updating model for pair {pair_id}: {str(e)}")
            # Keep the old model if update fails
    
    def _forecast_arima(self, pair_id: str, spread_series: pd.Series) -> Dict[str, Any]:
        """
        Generate forecasts using ARIMA model.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        
        Returns:
        --------
        Dict[str, Any]
            Forecast results
        """
        if pair_id not in self.models:
            logger.warning(f"No ARIMA model found for pair {pair_id}, fitting a new one")
            self._update_model(pair_id, spread_series)
        
        model = self.models.get(pair_id)
        if model is None:
            # If still no model, return empty forecast
            return {'forecast': pd.Series(dtype=float)}
        
        # Generate forecast with confidence intervals
        forecast_result = model.get_forecast(steps=self.forecast_horizon)
        forecast = forecast_result.predicted_mean
        
        # Get confidence intervals
        conf_int = forecast_result.conf_int(alpha=1-self.confidence_level)
        
        return {
            'forecast': forecast,
            'upper': conf_int.iloc[:, 1],  # Upper bound
            'lower': conf_int.iloc[:, 0],  # Lower bound
            'model_quality': {
                'aic': model.aic,
                'bic': model.bic
            }
        }
    
    def _forecast_garch(self, pair_id: str, spread_series: pd.Series) -> Dict[str, Any]:
        """
        Generate volatility forecasts using GARCH model.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        
        Returns:
        --------
        Dict[str, Any]
            Forecast results
        """
        if pair_id not in self.models:
            logger.warning(f"No GARCH model found for pair {pair_id}, fitting a new one")
            self._update_model(pair_id, spread_series)
        
        model = self.models.get(pair_id)
        if model is None:
            # If still no model, return empty forecast
            return {'forecast': pd.Series(dtype=float)}
        
        # Generate forecast
        forecast = model.forecast(horizon=self.forecast_horizon)
        
        # Extract mean and variance forecasts
        mean_forecast = forecast.mean.iloc[-1]
        variance_forecast = forecast.variance.iloc[-1]
        
        # Compute confidence intervals
        from scipy.stats import norm
        z_value = norm.ppf(self.confidence_level)
        std_forecast = np.sqrt(variance_forecast)
        
        upper = mean_forecast + z_value * std_forecast
        lower = mean_forecast - z_value * std_forecast
        
        return {
            'forecast': pd.Series(mean_forecast),
            'upper': pd.Series(upper),
            'lower': pd.Series(lower),
            'volatility': pd.Series(std_forecast),
            'model_quality': {
                'aic': model.aic,
                'bic': model.bic
            }
        }
    
    def _forecast_var(self, pair_id: str, spread_series: pd.Series) -> Dict[str, Any]:
        """
        Generate forecasts using VAR model.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        
        Returns:
        --------
        Dict[str, Any]
            Forecast results
        """
        if pair_id not in self.models:
            logger.warning(f"No VAR model found for pair {pair_id}, fitting a new one")
            self._update_model(pair_id, spread_series)
        
        model = self.models.get(pair_id)
        if model is None:
            # If still no model, return empty forecast
            return {'forecast': pd.Series(dtype=float)}
        
        # Generate forecast
        lag_order = model.k_ar
        # Get last values for forecast input
        var_data = pd.DataFrame({
            'spread': spread_series,
            'leg1_returns': self.data_processor.get_returns(self.pairs[pair_id]['leg1']),
            'leg2_returns': self.data_processor.get_returns(self.pairs[pair_id]['leg2'])
        }).dropna()
        
        input_data = var_data.values[-lag_order:]
        
        # Forecast
        forecast_values = model.forecast(input_data, steps=self.forecast_horizon)
        
        # Extract spread forecast (assuming it's the first column)
        spread_forecast = pd.Series(forecast_values[:, 0])
        
        return {
            'forecast': spread_forecast,
            'model_quality': {
                'aic': model.aic,
                'bic': model.bic if hasattr(model, 'bic') else None
            }
        }
    
    def _forecast_combined(self, pair_id: str, spread_series: pd.Series) -> Dict[str, Any]:
        """
        Generate forecasts using a combination of ARIMA and GARCH models.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        
        Returns:
        --------
        Dict[str, Any]
            Forecast results
        """
        if pair_id not in self.models:
            logger.warning(f"No combined models found for pair {pair_id}, fitting new ones")
            self._update_model(pair_id, spread_series)
        
        models = self.models.get(pair_id)
        if models is None:
            # If still no models, return empty forecast
            return {'forecast': pd.Series(dtype=float)}
        
        # Get ARIMA and GARCH models
        arima_model = models.get('arima')
        garch_model = models.get('garch')
        
        if arima_model is None or garch_model is None:
            logger.warning(f"Missing one of the required models for combined forecast for pair {pair_id}")
            return {'forecast': pd.Series(dtype=float)}
        
        # Generate ARIMA forecast for mean
        arima_forecast = arima_model.get_forecast(steps=self.forecast_horizon)
        mean_forecast = arima_forecast.predicted_mean
        
        # Generate GARCH forecast for volatility
        garch_forecast = garch_model.forecast(horizon=self.forecast_horizon)
        variance_forecast = garch_forecast.variance.iloc[-1]
        std_forecast = np.sqrt(variance_forecast)
        
        # Compute confidence intervals
        from scipy.stats import norm
        z_value = norm.ppf(self.confidence_level)
        
        upper = mean_forecast + z_value * std_forecast
        lower = mean_forecast - z_value * std_forecast
        
        return {
            'forecast': mean_forecast,
            'upper': upper,
            'lower': lower,
            'volatility': pd.Series(std_forecast),
            'model_quality': {
                'arima_aic': arima_model.aic,
                'garch_aic': garch_model.aic
            }
        }
    
    def _generate_signals_with_forecast(self, pair_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals using spread forecasts in addition to z-scores.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        analysis : Dict[str, Any]
            Analysis results including spread forecast
        
        Returns:
        --------
        Dict[str, Any]
            Trading signals
        """
        # Get base signals from z-score
        base_signals = analysis.get('signals', {})
        
        # Get forecast data
        forecast = analysis.get('forecast')
        upper = analysis.get('forecast_upper')
        lower = analysis.get('forecast_lower')
        
        if forecast is None or len(forecast) == 0:
            # If no forecast, just return base signals
            return base_signals
        
        # Get current signal
        current_signal = base_signals.get('signal', 0)
        
        # Enhance signals using forecast
        # If the forecast is strongly trending, we might want to modify our signals
        forecast_direction = 1 if forecast.iloc[-1] > forecast.iloc[0] else -1
        forecast_magnitude = abs(forecast.iloc[-1] - forecast.iloc[0]) / analysis.get('spread_std', 1)
        
        # Stronger forecast trend = more confident signal
        if forecast_magnitude > 0.5:  # If forecast change is more than 0.5 standard deviations
            # Strengthen existing signals in the direction of the forecast
            if forecast_direction == 1 and current_signal < 0:  # Forecast up, current signal short
                # More confident short signal (price reversal expected)
                enhanced_signal = current_signal * 1.2  # 20% stronger
            elif forecast_direction == -1 and current_signal > 0:  # Forecast down, current signal long
                # More confident long signal (price reversal expected)
                enhanced_signal = current_signal * 1.2  # 20% stronger
            else:
                # Signal and forecast align, no enhancement needed
                enhanced_signal = current_signal
        else:
            # Not a strong forecast, use base signal
            enhanced_signal = current_signal
        
        # Update signals
        enhanced_signals = base_signals.copy()
        enhanced_signals['signal'] = enhanced_signal
        enhanced_signals['forecast_direction'] = forecast_direction
        enhanced_signals['forecast_magnitude'] = forecast_magnitude
        
        return enhanced_signals
    
    def save_state(self, directory: str = None) -> Dict[str, Any]:
        """
        Save the strategy state to file.
        
        Parameters:
        -----------
        directory : str, optional
            Directory to save the state
        
        Returns:
        --------
        Dict[str, Any]
            Summary of saved state
        """
        # Get base state
        base_state = super().save_state(directory)
        
        # We can't easily save statsmodels objects, so we'll just save metadata
        model_metadata = {}
        for pair_id, model in self.models.items():
            if self.model_type == 'combined':
                model_metadata[pair_id] = {
                    'type': 'combined',
                    'arima_order': self.arima_order,
                    'garch_order': self.garch_order,
                    'last_update': self.last_model_update.get(pair_id, datetime.now()).isoformat()
                }
            else:
                model_metadata[pair_id] = {
                    'type': self.model_type,
                    'last_update': self.last_model_update.get(pair_id, datetime.now()).isoformat()
                }
        
        # Save model metadata
        if directory:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, 'time_series_models.json'), 'w') as f:
                json.dump(model_metadata, f, indent=2)
        
        return {
            **base_state,
            'time_series_models': model_metadata
        }
    
    def load_state(self, directory: str) -> Dict[str, Any]:
        """
        Load the strategy state from file.
        
        Parameters:
        -----------
        directory : str
            Directory containing saved state
        
        Returns:
        --------
        Dict[str, Any]
            Summary of loaded state
        """
        # Get base state
        base_state = super().load_state(directory)
        
        # Load model metadata
        model_metadata_file = os.path.join(directory, 'time_series_models.json')
        if os.path.exists(model_metadata_file):
            with open(model_metadata_file, 'r') as f:
                model_metadata = json.load(f)
            
            # Update last model update times
            for pair_id, metadata in model_metadata.items():
                try:
                    self.last_model_update[pair_id] = datetime.fromisoformat(metadata.get('last_update'))
                except (ValueError, TypeError):
                    # If date parsing fails, default to requiring model update
                    self.last_model_update[pair_id] = datetime.now() - timedelta(days=1)
        
        return {
            **base_state,
            'time_series_models_loaded': True
        } 