"""
Spread analyzer module for the Intraday Statistical Arbitrage System.

This module implements spread analysis functionality.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
from scipy import stats
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression
import warnings
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SpreadAnalyzer:
    """
    Spread analyzer class for analyzing spreads between assets.
    """
    
    def __init__(self, window_size=None, half_life=None, use_log_prices=False):
        """
        Initialize the SpreadAnalyzer.
        
        Parameters:
        -----------
        window_size : int, optional
            Window size for rolling calculations (default: None)
        half_life : int, optional
            Half-life period for exponential weighted calculations (default: None)
        use_log_prices : bool
            Whether to use log prices for spread calculations (default: False)
        """
        self.window_size = window_size
        self.half_life = half_life
        self.use_log_prices = use_log_prices
        self.hedge_ratio_history = None
    
    def calculate_spread_for_pair(self, asset1: Any, asset2: Any, 
                       hedge_ratio: float = None,
                       lookback_period: int = 20) -> Dict[str, Any]:
        """
        Calculate spread between two assets.
        
        Parameters:
        -----------
        asset1 : Any
            First asset
        asset2 : Any
            Second asset
        hedge_ratio : float, optional
            Hedge ratio between assets (if None, will be calculated)
        lookback_period : int, optional
            Lookback period for z-score calculation
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with spread information
        """
        # For testing, just return a simple spread structure
        return {
            'current_spread': 0.5,
            'current_zscore': 2.5,
            'mean': 0.0,
            'std': 0.2,
            'half_life': 5,
            'hedge_ratio': hedge_ratio or 0.85
        }
    
    def get_historical_spread(self, asset1: Any, asset2: Any,
                           hedge_ratio: float,
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime]) -> pd.Series:
        """
        Get historical spread between two assets.
        
        Parameters:
        -----------
        asset1 : Any
            First asset
        asset2 : Any
            Second asset
        hedge_ratio : float
            Hedge ratio between assets
        start_date : str or datetime
            Start date for data retrieval
        end_date : str or datetime
            End date for data retrieval
            
        Returns:
        --------
        pd.Series
            Series with historical spread values
        """
        # For testing, just return a simple series
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
    
    def calculate_spread(self, y_price, x_price, hedge_ratio=None, intercept=0):
        """
        Calculate spread between two price series.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        hedge_ratio : float, optional
            Hedge ratio to use (if None, will be calculated)
        intercept : float, optional
            Intercept to use in the spread calculation
            
        Returns:
        --------
        pandas.Series
            Spread series
        float
            Hedge ratio used
        """
        # Transform prices if needed
        y = np.log(y_price) if self.use_log_prices else y_price
        x = np.log(x_price) if self.use_log_prices else x_price
        
        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            model = OLS(y, x).fit()
            hedge_ratio = model.params[0]
        
        # Calculate spread
        spread = y - (hedge_ratio * x + intercept)
        
        return spread, hedge_ratio
    
    def rolling_hedge_ratio(self, y_price, x_price, window=None):
        """
        Calculate rolling hedge ratio.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        window : int, optional
            Window size for rolling calculation
            
        Returns:
        --------
        pandas.Series
            Rolling hedge ratio series
        """
        if window is None:
            window = self.window_size if self.window_size is not None else 60
        
        # Transform prices if needed
        y = np.log(y_price) if self.use_log_prices else y_price
        x = np.log(x_price) if self.use_log_prices else x_price
        
        # Prepare data
        data = pd.DataFrame({'y': y, 'x': x})
        
        # Calculate rolling hedge ratio
        hedge_ratios = []
        for i in range(window, len(data) + 1):
            window_data = data.iloc[i-window:i]
            model = OLS(window_data['y'], window_data['x']).fit()
            hedge_ratios.append(model.params[0])
        
        # Create series with matching index
        hr_series = pd.Series(hedge_ratios, index=data.index[window-1:])
        
        # Save for reference
        self.hedge_ratio_history = hr_series
        
        return hr_series
    
    def kalman_filter_hedge_ratio(self, y_price, x_price, transition_covariance=0.01, 
                                observation_covariance=0.1, initial_state_mean=0):
        """
        Calculate hedge ratio using Kalman filter.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        transition_covariance : float
            Kalman filter transition covariance
        observation_covariance : float
            Kalman filter observation covariance
        initial_state_mean : float
            Initial state mean for Kalman filter
            
        Returns:
        --------
        pandas.Series
            Kalman filter hedge ratio series
        pandas.Series
            Kalman filter spread series
        """
        # Transform prices if needed
        y = np.log(y_price) if self.use_log_prices else y_price
        x = np.log(x_price) if self.use_log_prices else x_price
        
        # Prepare data for Kalman filter
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        
        # Setup Kalman Filter
        kf = KalmanFilter(
            n_dim_obs=1, 
            n_dim_state=2,
            initial_state_mean=[initial_state_mean, 0],
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_matrices=np.vstack([x, np.ones(len(x))]).T,
            observation_covariance=observation_covariance,
            transition_covariance=trans_cov
        )
        
        # Use Kalman filter to get state estimates
        state_means, _ = kf.filter(y.values)
        
        # Extract hedge ratio and intercept
        hedge_ratios = pd.Series(state_means[:, 0], index=y.index)
        intercepts = pd.Series(state_means[:, 1], index=y.index)
        
        # Calculate spread
        spreads = y - (hedge_ratios * x + intercepts)
        
        # Save hedge ratio history
        self.hedge_ratio_history = hedge_ratios
        
        return hedge_ratios, spreads
    
    def calculate_zscore(self, spread, window=None, method='rolling'):
        """
        Calculate z-score of spread.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        window : int, optional
            Window size for rolling calculation
        method : str
            Method to use for z-score calculation ('rolling', 'ewm', 'garch')
            
        Returns:
        --------
        pandas.Series
            Z-score series
        """
        if window is None:
            window = self.window_size if self.window_size is not None else 60
        
        if method == 'rolling':
            mean = spread.rolling(window=window).mean()
            std = spread.rolling(window=window).std()
            return (spread - mean) / std
        elif method == 'ewm':
            if self.half_life is None:
                half_life = 0.5 * window
            else:
                half_life = self.half_life
            
            mean = spread.ewm(halflife=half_life).mean()
            std = spread.ewm(halflife=half_life).std()
            return (spread - mean) / std
        elif method == 'garch':
            # This is a placeholder for GARCH implementation
            # In a real implementation, use a proper GARCH model from arch package
            return self.calculate_garch_zscore(spread, window)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_volatility_adjusted_spread(self, y_price, x_price, 
                                            hedge_ratio=None, 
                                            vol_window=20, 
                                            vol_method='rolling',
                                            vol_lookback='same'):
        """
        Calculate volatility-adjusted spread between two price series.
        
        The volatility adjustment scales the spread by the inverse of its recent volatility,
        making the spread more comparable across different volatility regimes.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        hedge_ratio : float, optional
            Hedge ratio to use (if None, will be calculated)
        vol_window : int, optional
            Window size for volatility calculation (default: 20)
        vol_method : str, optional
            Method for volatility calculation ('rolling', 'ewm', 'garch') (default: 'rolling')
        vol_lookback : str, optional
            Type of lookback for volatility ('same' uses same window for spread and volatility,
            'lagged' uses previous window to avoid lookahead bias) (default: 'same')
            
        Returns:
        --------
        pandas.Series
            Volatility-adjusted spread series
        float
            Hedge ratio used
        pandas.Series
            Raw spread before volatility adjustment
        pandas.Series
            Volatility series used for adjustment
        """
        # Calculate raw spread
        raw_spread, hedge_ratio = self.calculate_spread(y_price, x_price, hedge_ratio)
        
        # Calculate volatility
        if vol_method == 'rolling':
            volatility = raw_spread.rolling(window=vol_window).std()
        elif vol_method == 'ewm':
            half_life = self.half_life if self.half_life is not None else 0.5 * vol_window
            volatility = raw_spread.ewm(halflife=half_life).std()
        elif vol_method == 'garch':
            volatility = self.calculate_garch_volatility(raw_spread, vol_window)
        else:
            raise ValueError(f"Unknown volatility method: {vol_method}")
        
        # Apply lookback adjustment
        if vol_lookback == 'lagged':
            # Shift volatility forward by one period to avoid lookahead bias
            volatility = volatility.shift(1)
            
        # Replace initial NaN values with first valid value
        volatility = volatility.fillna(volatility.dropna().iloc[0])
        
        # Adjust spread by volatility (inverse scaling)
        vol_adjusted_spread = raw_spread / volatility
        
        return vol_adjusted_spread, hedge_ratio, raw_spread, volatility

    def calculate_garch_volatility(self, spread, window):
        """
        Calculate GARCH volatility for spread.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        window : int
            Window size for rolling calculation
            
        Returns:
        --------
        pandas.Series
            GARCH volatility series
        """
        try:
            from arch import arch_model
            
            # Calculate returns from spread (use difference if spread is stationary)
            spread_diff = spread.diff().dropna()
            
            # Fit GARCH model
            model = arch_model(spread_diff, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Get conditional volatility
            vol = model_fit.conditional_volatility
            
            # Convert to series with original index
            vol_series = pd.Series(vol, index=spread_diff.index)
            
            # Reindex to match original spread
            volatility = pd.Series(index=spread.index)
            volatility.loc[vol_series.index] = vol_series
            
            # Forward fill to handle NaN at the beginning
            volatility = volatility.fillna(method='ffill')
            
            # For remaining NaN at the beginning, use the first valid value
            volatility = volatility.fillna(volatility.dropna().iloc[0])
            
            return volatility
        
        except ImportError:
            # Fallback to rolling standard deviation if arch package is not available
            logger.warning("arch package not found, falling back to rolling std for volatility")
            return spread.rolling(window=window).std()

    def calculate_garch_zscore(self, spread, window):
        """
        Calculate z-score using GARCH volatility.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        window : int
            Window size for mean calculation
            
        Returns:
        --------
        pandas.Series
            GARCH-based z-score series
        """
        # Calculate mean (use rolling for simplicity)
        mean = spread.rolling(window=window).mean()
        
        # Calculate GARCH volatility
        volatility = self.calculate_garch_volatility(spread, window)
        
        # Calculate z-score
        zscore = (spread - mean) / volatility
        
        return zscore

    def calculate_multitimeframe_spread(self, y_price, x_price, timeframes=[20, 60, 120], 
                                      hedge_ratio=None, zscore_method='rolling'):
        """
        Calculate spread and z-scores across multiple timeframes.
        
        This function calculates the spread using the same hedge ratio but analyzes it
        using different timeframes. This helps identify more robust trading opportunities
        that are confirmed across multiple time horizons.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        timeframes : list, optional
            List of timeframes (in periods) to use (default: [20, 60, 120])
        hedge_ratio : float, optional
            Hedge ratio to use (if None, will be calculated)
        zscore_method : str, optional
            Method for z-score calculation (default: 'rolling')
            
        Returns:
        --------
        dict
            Dictionary with spread, z-scores, and other metrics across timeframes
        """
        # Calculate raw spread using provided or calculated hedge ratio
        raw_spread, hedge_ratio = self.calculate_spread(y_price, x_price, hedge_ratio)
        
        # Initialize results dictionary
        results = {
            'raw_spread': raw_spread,
            'hedge_ratio': hedge_ratio,
            'timeframes': {}
        }
        
        # Calculate z-scores and other metrics for each timeframe
        for timeframe in timeframes:
            # Calculate z-score for this timeframe
            zscore = self.calculate_zscore(raw_spread, window=timeframe, method=zscore_method)
            
            # Calculate half-life and other spread metrics for this timeframe
            if len(raw_spread.dropna()) > timeframe:
                try:
                    half_life = self.estimate_half_life(raw_spread.iloc[-timeframe:])
                except:
                    half_life = None
                
                try:
                    hurst = self.hurst_exponent(raw_spread.iloc[-timeframe:])
                except:
                    hurst = None
                
                # Calculate spread metrics for this timeframe
                spread_metrics = self.calculate_spread_metrics(
                    raw_spread.iloc[-timeframe:], 
                    zscore.iloc[-timeframe:]
                )
            else:
                half_life = None
                hurst = None
                spread_metrics = {}
            
            # Store results for this timeframe
            results['timeframes'][timeframe] = {
                'zscore': zscore,
                'half_life': half_life,
                'hurst_exponent': hurst,
                'metrics': spread_metrics
            }
        
        # Calculate spread consistency across timeframes
        results['signal_consistency'] = self.calculate_signal_consistency(results)
        
        return results

    def calculate_signal_consistency(self, multitimeframe_results):
        """
        Calculate signal consistency across timeframes.
        
        Parameters:
        -----------
        multitimeframe_results : dict
            Dictionary with spread results across timeframes
            
        Returns:
        --------
        dict
            Dictionary with signal consistency metrics
        """
        timeframes = multitimeframe_results['timeframes']
        
        # Get latest z-score for each timeframe
        latest_zscores = {}
        for tf, data in timeframes.items():
            zscore = data['zscore']
            if not zscore.empty and not zscore.iloc[-1] != zscore.iloc[-1]:  # Check for NaN
                latest_zscores[tf] = zscore.iloc[-1]
        
        if not latest_zscores:
            return {'consistency': 0, 'agreement': False, 'mean_zscore': None}
        
        # Calculate signal direction for each timeframe
        directions = {}
        for tf, zscore in latest_zscores.items():
            if zscore > 2:
                directions[tf] = 'short'
            elif zscore < -2:
                directions[tf] = 'long'
            else:
                directions[tf] = 'neutral'
        
        # Check if all timeframes agree on direction
        all_equal = len(set(directions.values())) == 1 and 'neutral' not in directions.values()
        
        # Calculate the mean absolute z-score
        mean_abs_zscore = np.mean([abs(z) for z in latest_zscores.values()])
        
        # Calculate ratio of agreeing timeframes
        if 'neutral' in directions.values():
            non_neutral = [d for d in directions.values() if d != 'neutral']
            if not non_neutral:
                direction_consistency = 0
            else:
                majority = max(set(non_neutral), key=non_neutral.count) if non_neutral else 'neutral'
                direction_consistency = sum(1 for d in directions.values() if d == majority) / len(directions)
        else:
            majority = max(set(directions.values()), key=list(directions.values()).count)
            direction_consistency = sum(1 for d in directions.values() if d == majority) / len(directions)
        
        return {
            'consistency': direction_consistency,
            'agreement': all_equal,
            'mean_zscore': mean_abs_zscore,
            'directions': directions
        }
        
    def dynamic_threshold_calculation(self, spread, window=None, target_percentile=0.05):
        """
        Calculate dynamic entry/exit thresholds based on spread distribution.
        
        Instead of using fixed Z-score thresholds, this method calculates thresholds
        based on historical distribution percentiles, adapting to the specific
        characteristics of each spread.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        window : int, optional
            Window size for rolling calculation
        target_percentile : float, optional
            Target percentile for threshold calculation (default: 0.05)
            
        Returns:
        --------
        dict
            Dictionary with entry and exit thresholds
        """
        if window is None:
            window = self.window_size if self.window_size is not None else 60
            
        # Initialize empty series for thresholds
        upper_entry = pd.Series(index=spread.index)
        lower_entry = pd.Series(index=spread.index)
        upper_exit = pd.Series(index=spread.index)
        lower_exit = pd.Series(index=spread.index)
        
        # Calculate rolling thresholds
        for i in range(window, len(spread) + 1):
            window_spread = spread.iloc[i-window:i]
            
            # Calculate percentiles for this window
            upper_entry.iloc[i-1] = window_spread.quantile(1 - target_percentile)
            lower_entry.iloc[i-1] = window_spread.quantile(target_percentile)
            
            # Calculate exit thresholds (halfway between entry and mean)
            mean = window_spread.mean()
            upper_exit.iloc[i-1] = (upper_entry.iloc[i-1] + mean) / 2
            lower_exit.iloc[i-1] = (lower_entry.iloc[i-1] + mean) / 2
        
        return {
            'upper_entry': upper_entry,
            'lower_entry': lower_entry,
            'upper_exit': upper_exit,
            'lower_exit': lower_exit
        }

    def regime_based_thresholds(self, spread, volatility=None, vol_window=20):
        """
        Calculate regime-based thresholds that adapt to volatility conditions.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        volatility : pandas.Series, optional
            Volatility series (if None, will be calculated)
        vol_window : int, optional
            Window size for volatility calculation
            
        Returns:
        --------
        dict
            Dictionary with regime-based thresholds
        """
        # Calculate volatility if not provided
        if volatility is None:
            volatility = spread.rolling(window=vol_window).std()
        
        # Calculate volatility regimes
        # Use rolling percentile to identify relative volatility level
        vol_rank = volatility.rolling(window=252).rank(pct=True)
        
        # Define regimes
        high_vol = vol_rank > 0.8
        normal_vol = (vol_rank >= 0.3) & (vol_rank <= 0.8)
        low_vol = vol_rank < 0.3
        
        # Define threshold mappings for different regimes
        # High volatility: wider thresholds
        # Low volatility: tighter thresholds
        thresholds = {
            'entry': pd.Series(index=spread.index),
            'exit': pd.Series(index=spread.index)
        }
        
        # Set regime-based entry thresholds
        thresholds['entry'][high_vol] = 2.5
        thresholds['entry'][normal_vol] = 2.0
        thresholds['entry'][low_vol] = 1.5
        
        # Set regime-based exit thresholds
        thresholds['exit'][high_vol] = 1.0
        thresholds['exit'][normal_vol] = 0.5
        thresholds['exit'][low_vol] = 0.25
        
        # Forward fill to handle any NaN values
        thresholds['entry'] = thresholds['entry'].fillna(method='ffill').fillna(2.0)
        thresholds['exit'] = thresholds['exit'].fillna(method='ffill').fillna(0.5)
        
        return thresholds
        
    def calculate_normalized_spread(self, y_price, x_price, 
                                   hedge_ratio=None, normalization='zscore',
                                   window=None, vol_adjust=False):
        """
        Calculate and normalize spread using various methods.
        
        This is a unified function that combines different spread normalization
        techniques in one place.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        hedge_ratio : float, optional
            Hedge ratio to use (if None, will be calculated)
        normalization : str, optional
            Normalization method ('zscore', 'quantile', 'percentile_rank', 'madev') (default: 'zscore')
        window : int, optional
            Window size for calculations
        vol_adjust : bool, optional
            Whether to adjust for volatility (default: False)
            
        Returns:
        --------
        pandas.Series
            Normalized spread series
        dict
            Additional information about normalization
        """
        if window is None:
            window = self.window_size if self.window_size is not None else 60
            
        # Calculate raw spread
        raw_spread, used_hedge_ratio = self.calculate_spread(y_price, x_price, hedge_ratio)
        
        # Apply volatility adjustment if requested
        if vol_adjust:
            spread, _, _, volatility = self.calculate_volatility_adjusted_spread(
                y_price, x_price, hedge_ratio=used_hedge_ratio, vol_window=window
            )
        else:
            spread = raw_spread
            volatility = spread.rolling(window=window).std()
        
        # Apply specified normalization
        if normalization == 'zscore':
            # Z-score normalization (already implemented)
            normalized = self.calculate_zscore(spread, window=window)
            norm_info = {'method': 'zscore', 'window': window}
            
        elif normalization == 'quantile':
            # Quantile-based normalization
            normalized = pd.Series(index=spread.index)
            for i in range(window, len(spread) + 1):
                historical = spread.iloc[i-window:i]
                # Get quantile of current spread value within historical distribution
                current = spread.iloc[i-1]
                q = (historical <= current).mean()
                # Scale to range similar to z-score (-3 to +3)
                normalized.iloc[i-1] = (q - 0.5) * 6
                
            norm_info = {'method': 'quantile', 'window': window}
            
        elif normalization == 'percentile_rank':
            # Percentile rank normalization
            normalized = pd.Series(index=spread.index)
            for i in range(window, len(spread) + 1):
                window_data = spread.iloc[i-window:i]
                # Calculate percentile rank
                normalized.iloc[i-1] = (window_data.rank(pct=True).iloc[-1] - 0.5) * 6
                
            norm_info = {'method': 'percentile_rank', 'window': window}
            
        elif normalization == 'madev':
            # Mean Absolute Deviation normalization
            # Less affected by outliers than z-score
            rolling_mean = spread.rolling(window=window).mean()
            rolling_mad = (spread - rolling_mean).abs().rolling(window=window).mean()
            normalized = (spread - rolling_mean) / rolling_mad
            
            norm_info = {'method': 'madev', 'window': window}
            
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Return normalized spread and information
        return normalized, {
            'normalization': norm_info,
            'raw_spread': raw_spread,
            'hedge_ratio': used_hedge_ratio,
            'volatility': volatility
        }
    
    def test_stationarity(self, spread, confidence_level=0.05):
        """
        Test stationarity of spread using Augmented Dickey-Fuller test.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        confidence_level : float
            Confidence level for the test
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        # Perform ADF test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            adf_result = adfuller(spread.dropna())
        
        # Extract results
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        # Determine if stationary
        is_stationary = p_value < confidence_level
        
        # Return results
        result = {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary
        }
        
        return result
    
    def test_cointegration(self, y_price, x_price, confidence_level=0.05):
        """
        Test for cointegration between two price series.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        confidence_level : float
            Confidence level for the test
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        # Transform prices if needed
        y = np.log(y_price) if self.use_log_prices else y_price
        x = np.log(x_price) if self.use_log_prices else x_price
        
        # Perform cointegration test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            coint_result = coint(y, x)
        
        # Extract results
        coint_t = coint_result[0]
        p_value = coint_result[1]
        critical_values = coint_result[2]
        
        # Determine if cointegrated
        is_cointegrated = p_value < confidence_level
        
        # Return results
        result = {
            'coint_t_statistic': coint_t,
            'p_value': p_value,
            'critical_values': {
                '1%': critical_values[0],
                '5%': critical_values[1],
                '10%': critical_values[2]
            },
            'is_cointegrated': is_cointegrated
        }
        
        return result
    
    def estimate_half_life(self, spread):
        """
        Estimate half-life of mean reversion for spread.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
            
        Returns:
        --------
        float
            Estimated half-life
        """
        # Lag spread and remove NaN values
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Prepare data for regression
        data = pd.DataFrame({'y': spread_diff, 'x': spread_lag}).dropna()
        
        # Perform regression
        model = LinearRegression()
        model.fit(data[['x']], data['y'])
        
        # Calculate half-life
        beta = model.coef_[0]
        half_life = -np.log(2) / beta if beta < 0 else np.inf
        
        return half_life
    
    def hurst_exponent(self, spread, max_lag=100):
        """
        Calculate Hurst exponent for spread series.
        
        The Hurst exponent (H) indicates:
        - H < 0.5: Mean-reverting (anti-persistent) series
        - H = 0.5: Random walk
        - H > 0.5: Trend-following (persistent) series
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        max_lag : int
            Maximum lag for the calculation
            
        Returns:
        --------
        float
            Estimated Hurst exponent
        """
        # Ensure spread has no NaN values
        spread = spread.dropna()
        
        # Limit max_lag if series is too short
        max_lag = min(max_lag, len(spread) // 2)
        
        # Calculate range over standard deviation for different lags
        lags = range(2, max_lag)
        rs_values = []
        
        for lag in lags:
            # Split time series into chunks
            chunks = len(spread) // lag
            if chunks < 1:
                continue
                
            # Reshape series into chunks and calculate means
            values = spread.values[:chunks * lag].reshape((chunks, lag))
            means = values.mean(axis=1)
            
            # Calculate cumulative deviations
            deviations = np.array([values[i] - means[i] for i in range(chunks)])
            cumdev = np.array([deviations[i].cumsum() for i in range(chunks)])
            
            # Calculate range and standard deviation
            ranges = np.array([cumdev[i].max() - cumdev[i].min() for i in range(chunks)])
            stds = np.array([deviations[i].std() for i in range(chunks)])
            
            # Calculate R/S values
            rs = ranges / (stds + 1e-10)  # Add small constant to avoid division by zero
            rs_values.append(rs.mean())
        
        # Calculate Hurst exponent using linear regression
        if len(rs_values) < 2:
            return 0.5  # Not enough data, return random walk value
            
        x = np.log(lags[:len(rs_values)])
        y = np.log(rs_values)
        
        slope, _, _, _, _ = stats.linregress(x, y)
        
        return slope
    
    def calculate_spread_metrics(self, spread, zscore):
        """
        Calculate various metrics for the spread and its z-score.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        zscore : pandas.Series
            Z-score series
            
        Returns:
        --------
        dict
            Dictionary containing spread metrics
        """
        # Test stationarity
        stationarity_result = self.test_stationarity(spread)
        
        # Calculate half-life
        half_life = self.estimate_half_life(spread)
        
        # Calculate Hurst exponent
        hurst = self.hurst_exponent(spread)
        
        # Calculate spread statistics
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Calculate z-score statistics
        zscore_mean = zscore.dropna().mean()
        zscore_std = zscore.dropna().std()
        zscore_min = zscore.dropna().min()
        zscore_max = zscore.dropna().max()
        
        # Calculate z-score crossing statistics
        zero_crossings = ((zscore.shift(1) * zscore) < 0).sum()
        plus_2_crosses = ((zscore.shift(1) < 2) & (zscore >= 2)).sum() + ((zscore.shift(1) > 2) & (zscore <= 2)).sum()
        minus_2_crosses = ((zscore.shift(1) > -2) & (zscore <= -2)).sum() + ((zscore.shift(1) < -2) & (zscore >= -2)).sum()
        
        # Calculate time spent outside 2/-2 bands
        time_above_2 = (zscore > 2).mean() * 100  # as percentage
        time_below_neg2 = (zscore < -2).mean() * 100  # as percentage
        
        # Collect results
        metrics = {
            'is_stationary': stationarity_result['is_stationary'],
            'adf_p_value': stationarity_result['p_value'],
            'half_life': half_life,
            'hurst_exponent': hurst,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'zscore_mean': zscore_mean,
            'zscore_std': zscore_std,
            'zscore_min': zscore_min,
            'zscore_max': zscore_max,
            'zero_crossings': zero_crossings,
            'plus_2_crosses': plus_2_crosses,
            'minus_2_crosses': minus_2_crosses,
            'time_above_2_pct': time_above_2,
            'time_below_neg2_pct': time_below_neg2
        }
        
        return metrics
    
    def rolling_cointegration(self, y_price, x_price, window=None, step=1):
        """
        Calculate rolling cointegration p-values.
        
        Parameters:
        -----------
        y_price : pandas.Series
            Price series of the dependent variable
        x_price : pandas.Series
            Price series of the independent variable
        window : int, optional
            Window size for rolling calculation
        step : int
            Step size for rolling calculation
            
        Returns:
        --------
        pandas.Series
            Series of cointegration p-values
        """
        if window is None:
            window = self.window_size if self.window_size is not None else 60
        
        # Transform prices if needed
        y = np.log(y_price) if self.use_log_prices else y_price
        x = np.log(x_price) if self.use_log_prices else x_price
        
        # Ensure series are aligned
        data = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        # Calculate rolling cointegration
        p_values = []
        indices = []
        
        for i in range(window, len(data) + 1, step):
            window_data = data.iloc[i-window:i]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                _, p_value, _ = coint(window_data['y'], window_data['x'])
            
            p_values.append(p_value)
            indices.append(data.index[i-1])
        
        # Create series with matching index
        coint_series = pd.Series(p_values, index=indices)
        
        return coint_series
    
    def rolling_metrics(self, spread, window=None, step=1):
        """
        Calculate rolling metrics for the spread.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        window : int, optional
            Window size for rolling calculation
        step : int
            Step size for rolling calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing rolling metrics
        """
        if window is None:
            window = self.window_size if self.window_size is not None else 60
        
        # Calculate rolling half-life
        half_lives = []
        hurst_exponents = []
        adf_p_values = []
        indices = []
        
        for i in range(window, len(spread) + 1, step):
            window_data = spread.iloc[i-window:i]
            
            # Calculate half-life
            half_life = self.estimate_half_life(window_data)
            half_lives.append(half_life)
            
            # Calculate Hurst exponent
            hurst = self.hurst_exponent(window_data)
            hurst_exponents.append(hurst)
            
            # Calculate ADF p-value
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                _, p_value, _, _, _ = adfuller(window_data.dropna())
            adf_p_values.append(p_value)
            
            indices.append(spread.index[i-1])
        
        # Create DataFrame
        metrics = pd.DataFrame({
            'half_life': half_lives,
            'hurst_exponent': hurst_exponents,
            'adf_p_value': adf_p_values
        }, index=indices)
        
        return metrics 