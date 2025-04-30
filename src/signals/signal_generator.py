import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class SignalType(Enum):
    """Enum for signal types."""
    ZSCORE = "zscore"
    BOLLINGER = "bollinger"
    CUMULATIVE_SUM = "cusum"
    REGRESSION = "regression"
    KALMAN = "kalman"
    COMBINED = "combined"


class ConfirmationType(Enum):
    """Enum for confirmation types."""
    NONE = "none"
    VOLUME_IMBALANCE = "volume_imbalance"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MULTI_TIMEFRAME = "multi_timeframe"
    COMBINED = "combined"


class StopType(Enum):
    """Enum for stop-loss types."""
    FIXED = "fixed"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TRAILING = "trailing"
    DYNAMIC = "dynamic"
    TIME_BASED = "time_based"
    NONE = "none"


class TakeProfitType(Enum):
    """Enum for take-profit types."""
    FIXED = "fixed"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TRAILING = "trailing"
    DYNAMIC = "dynamic"
    TIME_BASED = "time_based"
    NONE = "none"


class SignalGenerator:
    """
    Generates trading signals based on spread analysis.
    
    This class provides methods for generating entry and exit signals
    for pairs trading using various statistical approaches.
    """
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.0, 
                 stop_loss_threshold=3.5, take_profit_threshold=None, 
                 time_stop=None, signal_type=SignalType.ZSCORE,
                 signal_smoothing=0, use_trailing_stop=False,
                 regime_classifier=None, confirmation_type=ConfirmationType.NONE,
                 confirmation_params=None, stop_type=StopType.FIXED,
                 take_profit_type=TakeProfitType.FIXED,
                 trailing_stop_params=None, dynamic_thresholds=False,
                 regime_threshold_mapping=None):
        """
        Initialize SignalGenerator.
        
        Parameters:
        -----------
        entry_threshold : float
            Z-score threshold to enter position
        exit_threshold : float
            Z-score threshold to exit position
        stop_loss_threshold : float, optional
            Z-score threshold for stop loss
        take_profit_threshold : float, optional
            Z-score threshold for take profit
        time_stop : int, optional
            Maximum holding period in bars
        signal_type : SignalType
            Type of signal generation method
        signal_smoothing : int
            Number of periods for signal smoothing (moving average)
        use_trailing_stop : bool
            Whether to use trailing stop
        regime_classifier : MarketRegimeClassifier, optional
            Classifier for market regime detection
        confirmation_type : ConfirmationType
            Type of confirmation filter to use
        confirmation_params : dict, optional
            Parameters for confirmation filters
        stop_type : StopType
            Type of stop-loss mechanism
        take_profit_type : TakeProfitType
            Type of take-profit mechanism
        trailing_stop_params : dict, optional
            Parameters for trailing stop (e.g., {'activation': 1.0, 'step': 0.5})
        dynamic_thresholds : bool
            Whether to use dynamic thresholds based on spread distribution
        regime_threshold_mapping : dict, optional
            Mapping of regime to threshold parameters
            Example: {0: {'entry': 2.5, 'exit': 0.5}, 1: {'entry': 2.0, 'exit': 0.0}}
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.time_stop = time_stop
        self.signal_type = signal_type
        self.signal_smoothing = signal_smoothing
        self.use_trailing_stop = use_trailing_stop
        
        # New parameters
        self.regime_classifier = regime_classifier
        self.confirmation_type = confirmation_type
        self.confirmation_params = confirmation_params or {}
        self.stop_type = stop_type
        self.take_profit_type = take_profit_type
        self.trailing_stop_params = trailing_stop_params or {'activation': 1.0, 'step': 0.5}
        self.dynamic_thresholds = dynamic_thresholds
        self.regime_threshold_mapping = regime_threshold_mapping or {}
        
        # Initialize state variables
        self.current_regime = None
        self.highest_profit = None
        self.lowest_drawdown = None
        self.entry_time = None
        self.trailing_stop_level = None
        self.trailing_profit_level = None
        
        # For dynamic thresholds
        self.historical_thresholds = None
    
    def generate_signals(self, spread=None, zscore=None, additional_data=None):
        """
        Generate trading signals based on spread and z-score.
        
        Parameters:
        -----------
        spread : pandas.Series, optional
            Spread series
        zscore : pandas.Series, optional
            Z-score series
        additional_data : dict, optional
            Additional data for signal generation. May include:
            - volume_data: DataFrame with volume information
            - price_data: DataFrame with price information
            - volatility: Series with volatility estimates
            - market_data: DataFrame with broader market data
            - timeframes: Dictionary with data from multiple timeframes
            - regime_features: Features for regime detection
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signals
        """
        # Check inputs
        if zscore is None and spread is None:
            raise ValueError("Either zscore or spread must be provided")
        
        # If z-score is not provided but spread is, we need to calculate z-score
        if zscore is None:
            # This is a fallback - ideally z-score should be pre-calculated
            mean = spread.rolling(window=20).mean()
            std = spread.rolling(window=20).std()
            zscore = (spread - mean) / std
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=zscore.index)
        
        # Apply smoothing if requested
        if self.signal_smoothing > 0:
            zscore = zscore.rolling(window=self.signal_smoothing).mean()
        
        # Detect market regime if classifier is provided
        if self.regime_classifier is not None and additional_data is not None:
            regime_features = additional_data.get('regime_features', None)
            if regime_features is not None:
                self.current_regime = self.regime_classifier.predict(regime_features)
                signals['regime'] = self.current_regime
                
                # Adapt thresholds based on regime
                self._adapt_thresholds_to_regime()
        
        # Apply dynamic thresholds if enabled
        if self.dynamic_thresholds and spread is not None:
            self._calculate_dynamic_thresholds(spread)
        
        # Choose signal generation method
        if self.signal_type == SignalType.ZSCORE:
            signals = self._generate_zscore_signals(zscore)
        elif self.signal_type == SignalType.BOLLINGER:
            signals = self._generate_bollinger_signals(spread, additional_data)
        elif self.signal_type == SignalType.CUMULATIVE_SUM:
            signals = self._generate_cusum_signals(zscore)
        elif self.signal_type == SignalType.REGRESSION:
            signals = self._generate_regression_signals(spread, zscore, additional_data)
        elif self.signal_type == SignalType.KALMAN:
            signals = self._generate_kalman_signals(spread, zscore, additional_data)
        elif self.signal_type == SignalType.COMBINED:
            signals = self._generate_combined_signals(spread, zscore, additional_data)
        else:
            signals = self._generate_zscore_signals(zscore)  # Default to zscore
        
        # Apply confirmation filters if enabled
        if self.confirmation_type != ConfirmationType.NONE:
            signals = self._apply_confirmation_filters(signals, spread, zscore, additional_data)
        
        # Apply appropriate stop-loss mechanism
        if self.stop_type != StopType.NONE:
            signals = self._apply_stop_loss(signals, zscore, additional_data)
        
        # Apply appropriate take-profit mechanism
        if self.take_profit_type != TakeProfitType.NONE:
            signals = self._apply_take_profit(signals, zscore, additional_data)
        
        # Add time-based exit signals if time_stop is specified
        if self.time_stop is not None:
            signals = self._add_time_stop_signals(signals)
        
        # Add trailing stop signals if requested
        if self.use_trailing_stop:
            signals = self._add_advanced_trailing_stop(signals, zscore, additional_data)
        
        # Calculate final positions
        signals = self._calculate_positions(signals)
        
        return signals
    
    def _generate_zscore_signals(self, zscore):
        """
        Generate signals based on z-score thresholds.
        
        Parameters:
        -----------
        zscore : pandas.Series
            Z-score series
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signals
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        
        # Entry signals
        signals['entry_long'] = 0  # Initialize
        signals['entry_short'] = 0  # Initialize
        
        # Long entry: z-score < -entry_threshold
        signals.loc[zscore < -self.entry_threshold, 'entry_long'] = 1
        
        # Short entry: z-score > entry_threshold
        signals.loc[zscore > self.entry_threshold, 'entry_short'] = 1
        
        # Exit signals
        signals['exit_long'] = 0  # Initialize
        signals['exit_short'] = 0  # Initialize
        
        # Long exit: z-score > -exit_threshold
        signals.loc[zscore > -self.exit_threshold, 'exit_long'] = 1
        
        # Short exit: z-score < exit_threshold
        signals.loc[zscore < self.exit_threshold, 'exit_short'] = 1
        
        # Stop loss signals
        if self.stop_loss_threshold is not None:
            # Long stop: z-score < -stop_loss_threshold
            signals.loc[zscore < -self.stop_loss_threshold, 'exit_long'] = 1
            
            # Short stop: z-score > stop_loss_threshold
            signals.loc[zscore > self.stop_loss_threshold, 'exit_short'] = 1
        
        # Take profit signals
        if self.take_profit_threshold is not None:
            # Long take profit: z-score > take_profit_threshold
            signals.loc[zscore > self.take_profit_threshold, 'exit_long'] = 1
            
            # Short take profit: z-score < -take_profit_threshold
            signals.loc[zscore < -self.take_profit_threshold, 'exit_short'] = 1
        
        return signals
    
    def _generate_bollinger_signals(self, spread, additional_data=None):
        """
        Generate signals based on Bollinger Bands.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        additional_data : dict, optional
            Additional data including Bollinger Band parameters
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signals
        """
        # Default parameters
        window = 20
        num_std = 2.0
        
        # Override with additional_data if provided
        if additional_data is not None:
            window = additional_data.get('window', window)
            num_std = additional_data.get('num_std', num_std)
        
        # Calculate Bollinger Bands
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['rolling_mean'] = rolling_mean
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        
        # Entry signals
        signals['entry_long'] = 0  # Initialize
        signals['entry_short'] = 0  # Initialize
        
        # Long entry: spread < lower_band
        signals.loc[spread < lower_band, 'entry_long'] = 1
        
        # Short entry: spread > upper_band
        signals.loc[spread > upper_band, 'entry_short'] = 1
        
        # Exit signals
        signals['exit_long'] = 0  # Initialize
        signals['exit_short'] = 0  # Initialize
        
        # Long exit: spread > rolling_mean
        signals.loc[spread > rolling_mean, 'exit_long'] = 1
        
        # Short exit: spread < rolling_mean
        signals.loc[spread < rolling_mean, 'exit_short'] = 1
        
        # Calculate z-score for consistency
        signals['zscore'] = (spread - rolling_mean) / rolling_std
        
        return signals
    
    def _generate_cusum_signals(self, zscore):
        """
        Generate signals using CUSUM (Cumulative Sum) filter.
        
        Parameters:
        -----------
        zscore : pandas.Series
            Z-score series
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signals
        """
        # Parameters for CUSUM filter
        threshold = self.entry_threshold
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        
        # Initialize CUSUM arrays
        cusum_plus = np.zeros(len(zscore))
        cusum_minus = np.zeros(len(zscore))
        
        # Initialize signals
        signals['entry_long'] = 0  # Initialize
        signals['entry_short'] = 0  # Initialize
        signals['exit_long'] = 0  # Initialize
        signals['exit_short'] = 0  # Initialize
        
        # Calculate CUSUM+ and CUSUM-
        for i in range(1, len(zscore)):
            # CUSUM+ calculation
            cusum_plus[i] = max(0, cusum_plus[i-1] + zscore.iloc[i])
            
            # CUSUM- calculation
            cusum_minus[i] = min(0, cusum_minus[i-1] + zscore.iloc[i])
            
            # Generate signals based on CUSUM values
            if cusum_plus[i] > threshold:
                signals.iloc[i, signals.columns.get_loc('entry_short')] = 1
                cusum_plus[i] = 0  # Reset after signal
            
            if cusum_minus[i] < -threshold:
                signals.iloc[i, signals.columns.get_loc('entry_long')] = 1
                cusum_minus[i] = 0  # Reset after signal
        
        # Add CUSUM values to signals DataFrame
        signals['cusum_plus'] = cusum_plus
        signals['cusum_minus'] = cusum_minus
        
        # Exit logic: exit when z-score crosses zero
        signals.loc[zscore.shift(1) < 0, 'exit_short'] = (zscore >= 0).astype(int)
        signals.loc[zscore.shift(1) > 0, 'exit_long'] = (zscore <= 0).astype(int)
        
        return signals
    
    def _generate_regression_signals(self, spread, zscore, additional_data=None):
        """
        Generate signals based on linear regression prediction.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        zscore : pandas.Series
            Z-score series
        additional_data : dict, optional
            Additional data including regression parameters
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signals
        """
        # Default parameters
        lookback = 20
        forecast_periods = 1
        
        # Override with additional_data if provided
        if additional_data is not None:
            lookback = additional_data.get('lookback', lookback)
            forecast_periods = additional_data.get('forecast_periods', forecast_periods)
            
            # If forecast is provided, use it directly
            if 'forecast' in additional_data:
                forecast = additional_data['forecast']
                signals = self._generate_zscore_signals(zscore)
                signals['forecast'] = forecast
                return signals
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        
        # Calculate regression forecast
        forecast = pd.Series(index=zscore.index, dtype=float)
        forecast.iloc[:lookback] = np.nan
        
        for i in range(lookback, len(zscore)):
            # Get lookback window
            y = zscore.iloc[i-lookback:i].values
            X = np.arange(lookback).reshape(-1, 1)
            
            # Simple linear regression
            coeffs = np.polyfit(X[:, 0], y, 1)
            
            # Forecast
            forecast.iloc[i] = coeffs[0] * (lookback + forecast_periods - 1) + coeffs[1]
        
        signals['forecast'] = forecast
        
        # Entry signals based on forecast and current zscore
        signals['entry_long'] = 0  # Initialize
        signals['entry_short'] = 0  # Initialize
        
        # Long entry: current z-score < -entry_threshold and forecast > current
        signals.loc[(zscore < -self.entry_threshold) & 
                   (forecast > zscore), 'entry_long'] = 1
        
        # Short entry: current z-score > entry_threshold and forecast < current
        signals.loc[(zscore > self.entry_threshold) & 
                   (forecast < zscore), 'entry_short'] = 1
        
        # Exit signals
        signals['exit_long'] = 0  # Initialize
        signals['exit_short'] = 0  # Initialize
        
        # Long exit: current z-score > -exit_threshold or forecast < current
        signals.loc[(zscore > -self.exit_threshold) | 
                   (forecast < zscore), 'exit_long'] = 1
        
        # Short exit: current z-score < exit_threshold or forecast > current
        signals.loc[(zscore < self.exit_threshold) | 
                   (forecast > zscore), 'exit_short'] = 1
        
        return signals
    
    def _generate_kalman_signals(self, spread, zscore, additional_data=None):
        """
        Generate signals based on Kalman filter hedge ratio.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        zscore : pandas.Series
            Z-score series
        additional_data : dict, optional
            Additional data including Kalman filter state
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signals
        """
        # Check if Kalman filter state is provided
        if additional_data is None or 'kalman_spread' not in additional_data:
            # If no Kalman filter state, use standard z-score signals
            return self._generate_zscore_signals(zscore)
        
        # Extract Kalman filter results
        kalman_spread = additional_data['kalman_spread']
        kalman_zscore = additional_data.get('kalman_zscore', zscore)
        hedge_ratios = additional_data.get('hedge_ratios', None)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=kalman_zscore.index)
        signals['zscore'] = kalman_zscore
        
        if hedge_ratios is not None:
            signals['hedge_ratio'] = hedge_ratios
        
        # Entry signals
        signals['entry_long'] = 0  # Initialize
        signals['entry_short'] = 0  # Initialize
        
        # Long entry: kalman_zscore < -entry_threshold
        signals.loc[kalman_zscore < -self.entry_threshold, 'entry_long'] = 1
        
        # Short entry: kalman_zscore > entry_threshold
        signals.loc[kalman_zscore > self.entry_threshold, 'entry_short'] = 1
        
        # Exit signals
        signals['exit_long'] = 0  # Initialize
        signals['exit_short'] = 0  # Initialize
        
        # Long exit: kalman_zscore > -exit_threshold
        signals.loc[kalman_zscore > -self.exit_threshold, 'exit_long'] = 1
        
        # Short exit: kalman_zscore < exit_threshold
        signals.loc[kalman_zscore < self.exit_threshold, 'exit_short'] = 1
        
        # Add standard z-score for comparison
        signals['standard_zscore'] = zscore
        
        return signals
    
    def _generate_combined_signals(self, spread, zscore, additional_data=None):
        """
        Generate signals using multiple methods and combine them.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        zscore : pandas.Series
            Z-score series
        additional_data : dict, optional
            Additional data for signal generation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signals
        """
        # Generate signals using different methods
        zscore_signals = self._generate_zscore_signals(zscore)
        
        # If additional_data contains required info, generate Bollinger signals
        if spread is not None:
            bollinger_signals = self._generate_bollinger_signals(spread, additional_data)
        else:
            bollinger_signals = None
        
        # If additional_data contains required info, generate regression signals
        regression_signals = None
        if additional_data is not None and 'forecast' in additional_data:
            regression_data = {'forecast': additional_data['forecast']}
            regression_signals = self._generate_regression_signals(spread, zscore, regression_data)
        
        # Initialize combined signals DataFrame
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        
        # Initialize combined entry signals
        signals['entry_long'] = zscore_signals['entry_long']
        signals['entry_short'] = zscore_signals['entry_short']
        
        # Combine with Bollinger signals if available
        if bollinger_signals is not None:
            # Require both signals to trigger entry (logical AND)
            signals['entry_long'] = (signals['entry_long'] & bollinger_signals['entry_long']).astype(int)
            signals['entry_short'] = (signals['entry_short'] & bollinger_signals['entry_short']).astype(int)
        
        # Combine with regression signals if available
        if regression_signals is not None:
            # Require both signals to trigger entry (logical AND)
            signals['entry_long'] = (signals['entry_long'] & regression_signals['entry_long']).astype(int)
            signals['entry_short'] = (signals['entry_short'] & regression_signals['entry_short']).astype(int)
        
        # Initialize combined exit signals (more sensitive - exit if ANY signal says to exit)
        signals['exit_long'] = zscore_signals['exit_long']
        signals['exit_short'] = zscore_signals['exit_short']
        
        # Combine with Bollinger signals if available
        if bollinger_signals is not None:
            # Exit if either signal says to exit (logical OR)
            signals['exit_long'] = (signals['exit_long'] | bollinger_signals['exit_long']).astype(int)
            signals['exit_short'] = (signals['exit_short'] | bollinger_signals['exit_short']).astype(int)
        
        # Combine with regression signals if available
        if regression_signals is not None:
            # Exit if either signal says to exit (logical OR)
            signals['exit_long'] = (signals['exit_long'] | regression_signals['exit_long']).astype(int)
            signals['exit_short'] = (signals['exit_short'] | regression_signals['exit_short']).astype(int)
        
        return signals
    
    def _add_time_stop_signals(self, signals):
        """
        Add time-based exit signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with updated signals
        """
        # Initialize position tracking
        position_long_duration = np.zeros(len(signals))
        position_short_duration = np.zeros(len(signals))
        
        # Track position duration
        position_long = 0
        position_short = 0
        
        for i in range(1, len(signals)):
            # Update long position tracking
            if signals['entry_long'].iloc[i] == 1:
                position_long = 1
                position_long_duration[i] = 1
            elif position_long == 1:
                if signals['exit_long'].iloc[i] == 1:
                    position_long = 0
                    position_long_duration[i] = 0
                else:
                    position_long_duration[i] = position_long_duration[i-1] + 1
            
            # Update short position tracking
            if signals['entry_short'].iloc[i] == 1:
                position_short = 1
                position_short_duration[i] = 1
            elif position_short == 1:
                if signals['exit_short'].iloc[i] == 1:
                    position_short = 0
                    position_short_duration[i] = 0
                else:
                    position_short_duration[i] = position_short_duration[i-1] + 1
        
        # Add time-based exit signals
        time_stop = self.time_stop
        if time_stop is not None:
            signals.loc[position_long_duration >= time_stop, 'exit_long'] = 1
            signals.loc[position_short_duration >= time_stop, 'exit_short'] = 1
        
        # Add duration to signals
        signals['position_long_duration'] = position_long_duration
        signals['position_short_duration'] = position_short_duration
        
        return signals
    
    def _add_trailing_stop_signals(self, signals, zscore):
        """
        Add trailing stop loss signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        zscore : pandas.Series
            Z-score series
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with updated signals
        """
        # Initialize max favorable excursion tracking
        max_favorable_long = np.zeros(len(signals))  # For long positions, track lowest z-score
        max_favorable_short = np.zeros(len(signals))  # For short positions, track highest z-score
        
        # Initialize position tracking
        position_long = 0
        position_short = 0
        
        # Trailing stop parameters
        trail_amount = 1.0  # How much to trail (in z-score units)
        
        for i in range(1, len(signals)):
            # Update long position tracking
            if signals['entry_long'].iloc[i] == 1:
                position_long = 1
                max_favorable_long[i] = zscore.iloc[i]
            elif position_long == 1:
                if signals['exit_long'].iloc[i] == 1:
                    position_long = 0
                    max_favorable_long[i] = 0
                else:
                    # Track lowest z-score
                    max_favorable_long[i] = min(max_favorable_long[i-1], zscore.iloc[i])
            
            # Update short position tracking
            if signals['entry_short'].iloc[i] == 1:
                position_short = 1
                max_favorable_short[i] = zscore.iloc[i]
            elif position_short == 1:
                if signals['exit_short'].iloc[i] == 1:
                    position_short = 0
                    max_favorable_short[i] = 0
                else:
                    # Track highest z-score
                    max_favorable_short[i] = max(max_favorable_short[i-1], zscore.iloc[i])
        
        # Add trailing stop signals
        for i in range(1, len(signals)):
            # Check if long position is active
            if position_long == 1 and max_favorable_long[i] != 0:
                # If z-score has moved away from favorable by trail_amount, exit
                if zscore.iloc[i] > max_favorable_long[i] + trail_amount:
                    signals.iloc[i, signals.columns.get_loc('exit_long')] = 1
            
            # Check if short position is active
            if position_short == 1 and max_favorable_short[i] != 0:
                # If z-score has moved away from favorable by trail_amount, exit
                if zscore.iloc[i] < max_favorable_short[i] - trail_amount:
                    signals.iloc[i, signals.columns.get_loc('exit_short')] = 1
        
        return signals
    
    def _calculate_positions(self, signals):
        """
        Calculate position flags based on entry and exit signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with entry and exit signals
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with position flags
        """
        # Initialize position flags
        signals['position_long'] = 0
        signals['position_short'] = 0
        signals['position'] = 0
        
        position_long = 0
        position_short = 0
        
        # Calculate positions
        for i in range(1, len(signals)):
            # Update long position
            if signals['entry_long'].iloc[i] == 1:
                position_long = 1
            elif position_long == 1 and signals['exit_long'].iloc[i] == 1:
                position_long = 0
            
            # Update short position
            if signals['entry_short'].iloc[i] == 1:
                position_short = 1
            elif position_short == 1 and signals['exit_short'].iloc[i] == 1:
                position_short = 0
            
            # Store positions
            signals.iloc[i, signals.columns.get_loc('position_long')] = position_long
            signals.iloc[i, signals.columns.get_loc('position_short')] = position_short
            
            # Combine positions (1 = long, -1 = short, 0 = flat)
            signals.iloc[i, signals.columns.get_loc('position')] = position_long - position_short
        
        return signals
    
    def _adapt_thresholds_to_regime(self):
        """
        Adapt thresholds based on the current market regime.
        
        This method adjusts entry, exit, stop-loss, and take-profit thresholds
        based on the detected market regime.
        """
        if self.current_regime is None or not self.regime_threshold_mapping:
            return
        
        # Get threshold mapping for current regime
        regime_id = self.current_regime.iloc[-1] if isinstance(self.current_regime, pd.Series) else self.current_regime
        
        thresholds = self.regime_threshold_mapping.get(regime_id, None)
        if thresholds is None:
            return
        
        # Apply regime-specific thresholds
        self.entry_threshold = thresholds.get('entry', self.entry_threshold)
        self.exit_threshold = thresholds.get('exit', self.exit_threshold)
        self.stop_loss_threshold = thresholds.get('stop_loss', self.stop_loss_threshold)
        self.take_profit_threshold = thresholds.get('take_profit', self.take_profit_threshold)
    
    def _calculate_dynamic_thresholds(self, spread, window=60, percentile=0.05):
        """
        Calculate dynamic thresholds based on historical spread distribution.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
        window : int
            Window size for threshold calculation
        percentile : float
            Percentile for threshold calculation (e.g., 0.05 for 95% confidence)
        """
        # Initialize threshold series
        thresholds = pd.DataFrame(index=spread.index)
        
        # Calculate thresholds for each point
        for i in range(window, len(spread)):
            hist_data = spread.iloc[i-window:i]
            
            # Calculate percentiles
            upper = hist_data.quantile(1 - percentile)
            lower = hist_data.quantile(percentile)
            
            # Store as multiple of standard deviation
            std = hist_data.std()
            if std > 0:  # Avoid division by zero
                thresholds.loc[spread.index[i], 'entry_upper'] = upper / std
                thresholds.loc[spread.index[i], 'entry_lower'] = lower / std
                
                # Exit thresholds are halfway between entry and mean
                mean = hist_data.mean()
                thresholds.loc[spread.index[i], 'exit_upper'] = (upper + mean) / (2 * std)
                thresholds.loc[spread.index[i], 'exit_lower'] = (lower + mean) / (2 * std)
        
        # Store for use in signal generation
        self.historical_thresholds = thresholds
    
    def _apply_confirmation_filters(self, signals, spread, zscore, additional_data=None):
        """
        Apply confirmation filters to entry signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        spread : pandas.Series
            Spread series
        zscore : pandas.Series
            Z-score series
        additional_data : dict, optional
            Additional data for confirmation filters
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with filtered signals
        """
        if additional_data is None or self.confirmation_type == ConfirmationType.NONE:
            return signals
            
        # Create a copy to avoid modifying the original
        filtered_signals = signals.copy()
        
        if self.confirmation_type == ConfirmationType.VOLUME_IMBALANCE:
            # Apply volume imbalance filter
            filtered_signals = self._apply_volume_imbalance_filter(filtered_signals, additional_data)
            
        elif self.confirmation_type == ConfirmationType.MOMENTUM:
            # Apply momentum confirmation filter
            filtered_signals = self._apply_momentum_filter(filtered_signals, zscore)
            
        elif self.confirmation_type == ConfirmationType.MEAN_REVERSION:
            # Apply mean-reversion strength filter
            filtered_signals = self._apply_mean_reversion_filter(filtered_signals, spread)
            
        elif self.confirmation_type == ConfirmationType.MULTI_TIMEFRAME:
            # Apply multi-timeframe confirmation filter
            filtered_signals = self._apply_multi_timeframe_filter(filtered_signals, additional_data)
            
        elif self.confirmation_type == ConfirmationType.COMBINED:
            # Apply combined confirmation filters
            for filter_type in self.confirmation_params.get('filters', []):
                if filter_type == 'volume_imbalance':
                    filtered_signals = self._apply_volume_imbalance_filter(filtered_signals, additional_data)
                elif filter_type == 'momentum':
                    filtered_signals = self._apply_momentum_filter(filtered_signals, zscore)
                elif filter_type == 'mean_reversion':
                    filtered_signals = self._apply_mean_reversion_filter(filtered_signals, spread)
                elif filter_type == 'multi_timeframe':
                    filtered_signals = self._apply_multi_timeframe_filter(filtered_signals, additional_data)
        
        return filtered_signals
    
    def _apply_volume_imbalance_filter(self, signals, additional_data):
        """
        Apply volume imbalance filter to confirm entry signals.
        
        Volume imbalance suggests directional pressure that might 
        reinforce or contradict the statistical signal.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        additional_data : dict
            Contains volume data for assets
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with filtered signals
        """
        if 'volume_data' not in additional_data:
            return signals
            
        volume_data = additional_data['volume_data']
        if len(volume_data.columns) < 2:
            return signals
            
        # Calculate volume ratio between assets
        vol_ratio = volume_data.iloc[:, 0] / volume_data.iloc[:, 1]
        
        # Calculate moving average of volume ratio
        vol_ratio_ma = vol_ratio.rolling(
            window=self.confirmation_params.get('volume_window', 5)).mean()
        
        # Calculate z-score of volume ratio
        vol_ratio_std = vol_ratio.rolling(
            window=self.confirmation_params.get('volume_window', 5)).std()
        vol_zscore = (vol_ratio - vol_ratio_ma) / vol_ratio_std
        
        # Filter entry signals
        threshold = self.confirmation_params.get('volume_threshold', 1.0)
        
        # For long entries: First asset should have relatively lower volume (undervalued)
        mask_long = vol_zscore < -threshold
        signals.loc[~mask_long, 'entry_long'] = 0
        
        # For short entries: First asset should have relatively higher volume (overvalued)
        mask_short = vol_zscore > threshold
        signals.loc[~mask_short, 'entry_short'] = 0
        
        return signals
    
    def _apply_momentum_filter(self, signals, zscore):
        """
        Apply momentum filter to confirm entry signals.
        
        Confirms that z-score movement is in the right direction
        and has sufficient momentum.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        zscore : pandas.Series
            Z-score series
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with filtered signals
        """
        # Calculate momentum (rate of change)
        momentum_window = self.confirmation_params.get('momentum_window', 3)
        momentum = zscore.diff(momentum_window)
        
        # Set momentum threshold
        threshold = self.confirmation_params.get('momentum_threshold', 0.1)
        
        # For long entries: Z-score should be decreasing (getting more negative)
        mask_long = momentum < -threshold
        signals.loc[~mask_long, 'entry_long'] = 0
        
        # For short entries: Z-score should be increasing (getting more positive)
        mask_short = momentum > threshold
        signals.loc[~mask_short, 'entry_short'] = 0
        
        return signals
    
    def _apply_mean_reversion_filter(self, signals, spread):
        """
        Apply mean-reversion strength filter to confirm entry signals.
        
        Confirms that the spread is exhibiting strong mean-reverting behavior
        before generating entry signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        spread : pandas.Series
            Spread series
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with filtered signals
        """
        # Calculate Hurst exponent or half-life
        window = self.confirmation_params.get('reversion_window', 60)
        
        # Simple half-life calculation (more complex implementations in spread_analyzer)
        half_lives = []
        lag = spread.shift(1).dropna()
        diff = spread.diff().dropna()
        
        # Calculate half-life for rolling windows
        for i in range(window, len(spread)):
            y = diff.iloc[i-window:i]
            x = lag.iloc[i-window:i]
            
            try:
                model = np.polyfit(x, y, 1)
                beta = model[0]
                half_life = -np.log(2) / beta if beta < 0 else np.inf
                half_lives.append(half_life)
            except:
                half_lives.append(np.inf)
        
        # Create Series aligned with signals
        half_life_series = pd.Series(half_lives, index=spread.index[window:])
        half_life_threshold = self.confirmation_params.get('half_life_threshold', 20)
        
        # Only enter if half-life is below threshold (strong mean reversion)
        mask = half_life_series < half_life_threshold
        signals.loc[~mask, 'entry_long'] = 0
        signals.loc[~mask, 'entry_short'] = 0
        
        return signals
    
    def _apply_multi_timeframe_filter(self, signals, additional_data):
        """
        Apply multi-timeframe filter to confirm entry signals.
        
        Confirms that signals are consistent across multiple timeframes.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        additional_data : dict
            Contains data for multiple timeframes
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with filtered signals
        """
        if 'timeframes' not in additional_data:
            return signals
            
        timeframes = additional_data['timeframes']
        
        # Process each timeframe
        agreement_threshold = self.confirmation_params.get('timeframe_agreement', 0.7)
        tf_signals = []
        
        for tf, data in timeframes.items():
            if 'zscore' in data:
                # Generate signals for this timeframe
                tf_zscore = data['zscore']
                tf_signal = pd.DataFrame(index=tf_zscore.index)
                
                # Apply simple Z-score rules for this timeframe
                tf_signal['long'] = (tf_zscore < -self.entry_threshold).astype(int)
                tf_signal['short'] = (tf_zscore > self.entry_threshold).astype(int)
                
                tf_signals.append(tf_signal)
        
        if not tf_signals:
            return signals
            
        # Create a composite signal based on agreement across timeframes
        composite = pd.DataFrame(index=signals.index)
        composite['long_agreement'] = 0
        composite['short_agreement'] = 0
        
        # Calculate agreement percentage
        for tf_signal in tf_signals:
            # Reindex to match the main signal's index
            reindexed = tf_signal.reindex(signals.index, method='ffill')
            composite['long_agreement'] += reindexed['long']
            composite['short_agreement'] += reindexed['short']
        
        # Convert to percentage
        composite['long_agreement'] = composite['long_agreement'] / len(tf_signals)
        composite['short_agreement'] = composite['short_agreement'] / len(tf_signals)
        
        # Filter signals based on agreement threshold
        signals.loc[composite['long_agreement'] < agreement_threshold, 'entry_long'] = 0
        signals.loc[composite['short_agreement'] < agreement_threshold, 'entry_short'] = 0
        
        return signals
    
    def _apply_stop_loss(self, signals, zscore, additional_data=None):
        """
        Apply advanced stop-loss mechanisms.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        zscore : pandas.Series
            Z-score series
        additional_data : dict, optional
            Additional data for stop-loss calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stop-loss signals
        """
        if self.stop_loss_threshold is None:
            return signals
            
        # Create a copy to avoid modifying the original
        signals_with_stops = signals.copy()
        
        if self.stop_type == StopType.FIXED:
            # Apply fixed stop-loss (already implemented in _generate_zscore_signals)
            pass
            
        elif self.stop_type == StopType.VOLATILITY_ADJUSTED:
            # Volatility-adjusted stop-loss
            if additional_data is not None and 'volatility' in additional_data:
                vol = additional_data['volatility']
                vol_multiple = self.confirmation_params.get('vol_stop_multiple', 2.0)
                
                # Adjust stop-loss based on volatility
                # Higher volatility -> wider stops
                stop_level = self.stop_loss_threshold * vol / vol.mean()
                
                # Apply stop level capped at reasonable values
                stop_level = np.minimum(stop_level, self.stop_loss_threshold * vol_multiple)
                stop_level = np.maximum(stop_level, self.stop_loss_threshold / vol_multiple)
                
                # Apply stops to signals
                for i in range(1, len(signals_with_stops)):
                    if signals_with_stops['entry_long'].iloc[i-1] == 1:
                        # Check for stop-loss on long positions
                        if zscore.iloc[i] < -stop_level.iloc[i]:
                            signals_with_stops.loc[zscore.index[i], 'exit_long'] = 1
                    
                    if signals_with_stops['entry_short'].iloc[i-1] == 1:
                        # Check for stop-loss on short positions
                        if zscore.iloc[i] > stop_level.iloc[i]:
                            signals_with_stops.loc[zscore.index[i], 'exit_short'] = 1
        
        elif self.stop_type == StopType.DYNAMIC:
            # Dynamic stop-loss based on spread behavior
            window = self.confirmation_params.get('dynamic_stop_window', 20)
            
            # Calculate dynamic stop levels
            if additional_data is not None and 'spread' in additional_data:
                spread = additional_data['spread']
                
                for i in range(window, len(signals_with_stops)):
                    # Get recent spread data
                    recent_spread = spread.iloc[i-window:i]
                    
                    # Calculate dynamic threshold based on recent distribution
                    percentile = self.confirmation_params.get('dynamic_stop_percentile', 0.01)
                    upper_stop = recent_spread.quantile(1 - percentile)
                    lower_stop = recent_spread.quantile(percentile)
                    
                    # Calculate z-score equivalents
                    mean = recent_spread.mean()
                    std = recent_spread.std()
                    
                    if std > 0:  # Avoid division by zero
                        upper_z = (upper_stop - mean) / std
                        lower_z = (lower_stop - mean) / std
                        
                        # Apply stops to signals
                        if signals_with_stops['entry_long'].iloc[i-1] == 1:
                            # Check for stop-loss on long positions
                            if zscore.iloc[i] < lower_z:
                                signals_with_stops.loc[zscore.index[i], 'exit_long'] = 1
                        
                        if signals_with_stops['entry_short'].iloc[i-1] == 1:
                            # Check for stop-loss on short positions
                            if zscore.iloc[i] > upper_z:
                                signals_with_stops.loc[zscore.index[i], 'exit_short'] = 1
        
        return signals_with_stops
    
    def _apply_take_profit(self, signals, zscore, additional_data=None):
        """
        Apply advanced take-profit mechanisms.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        zscore : pandas.Series
            Z-score series
        additional_data : dict, optional
            Additional data for take-profit calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with take-profit signals
        """
        if self.take_profit_threshold is None:
            return signals
            
        # Create a copy to avoid modifying the original
        signals_with_tp = signals.copy()
        
        if self.take_profit_type == TakeProfitType.FIXED:
            # Apply fixed take-profit (already implemented in _generate_zscore_signals)
            pass
            
        elif self.take_profit_type == TakeProfitType.VOLATILITY_ADJUSTED:
            # Volatility-adjusted take-profit
            if additional_data is not None and 'volatility' in additional_data:
                vol = additional_data['volatility']
                vol_multiple = self.confirmation_params.get('vol_tp_multiple', 1.5)
                
                # Adjust take-profit based on volatility
                # Higher volatility -> wider take-profit levels
                tp_level = self.take_profit_threshold * vol / vol.mean()
                
                # Apply take-profit level capped at reasonable values
                tp_level = np.minimum(tp_level, self.take_profit_threshold * vol_multiple)
                tp_level = np.maximum(tp_level, self.take_profit_threshold / vol_multiple)
                
                # Apply take-profit to signals
                for i in range(1, len(signals_with_tp)):
                    if signals_with_tp['entry_long'].iloc[i-1] == 1:
                        # Check for take-profit on long positions
                        if zscore.iloc[i] > tp_level.iloc[i]:
                            signals_with_tp.loc[zscore.index[i], 'exit_long'] = 1
                    
                    if signals_with_tp['entry_short'].iloc[i-1] == 1:
                        # Check for take-profit on short positions
                        if zscore.iloc[i] < -tp_level.iloc[i]:
                            signals_with_tp.loc[zscore.index[i], 'exit_short'] = 1
        
        elif self.take_profit_type == TakeProfitType.DYNAMIC:
            # Dynamic take-profit based on spread behavior
            window = self.confirmation_params.get('dynamic_tp_window', 20)
            
            # Calculate dynamic take-profit levels
            if additional_data is not None and 'spread' in additional_data:
                spread = additional_data['spread']
                
                for i in range(window, len(signals_with_tp)):
                    # Get recent spread data
                    recent_spread = spread.iloc[i-window:i]
                    
                    # Calculate dynamic threshold based on recent distribution
                    percentile = self.confirmation_params.get('dynamic_tp_percentile', 0.25)
                    upper_tp = recent_spread.quantile(1 - percentile)
                    lower_tp = recent_spread.quantile(percentile)
                    
                    # Calculate z-score equivalents
                    mean = recent_spread.mean()
                    std = recent_spread.std()
                    
                    if std > 0:  # Avoid division by zero
                        upper_z = (upper_tp - mean) / std
                        lower_z = (lower_tp - mean) / std
                        
                        # Apply take-profit to signals
                        if signals_with_tp['entry_long'].iloc[i-1] == 1:
                            # Check for take-profit on long positions
                            if zscore.iloc[i] > upper_z:
                                signals_with_tp.loc[zscore.index[i], 'exit_long'] = 1
                        
                        if signals_with_tp['entry_short'].iloc[i-1] == 1:
                            # Check for take-profit on short positions
                            if zscore.iloc[i] < lower_z:
                                signals_with_tp.loc[zscore.index[i], 'exit_short'] = 1
        
        return signals_with_tp
    
    def _add_advanced_trailing_stop(self, signals, zscore, additional_data=None):
        """
        Add advanced trailing stop signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signals
        zscore : pandas.Series
            Z-score series
        additional_data : dict, optional
            Additional data for trailing stop calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with trailing stop signals
        """
        # Create a copy to avoid modifying the original
        signals_with_ts = signals.copy()
        
        # Get parameters for trailing stop
        activation = self.trailing_stop_params.get('activation', 1.0)
        step = self.trailing_stop_params.get('step', 0.5)
        
        # Initialize tracking variables
        position = 0  # 1 for long, -1 for short, 0 for flat
        trailing_level = None
        
        # Process signals
        for i in range(1, len(signals_with_ts)):
            # Update position based on entry/exit signals
            if signals_with_ts['entry_long'].iloc[i-1] == 1:
                position = 1
                trailing_level = None  # Reset trailing level
            elif signals_with_ts['entry_short'].iloc[i-1] == 1:
                position = -1
                trailing_level = None  # Reset trailing level
            elif signals_with_ts['exit_long'].iloc[i-1] == 1 and position == 1:
                position = 0
            elif signals_with_ts['exit_short'].iloc[i-1] == 1 and position == -1:
                position = 0
            
            # Update trailing stop level
            if position == 1:  # Long position
                # Initialize trailing stop if zscore crosses activation level
                if trailing_level is None and zscore.iloc[i] > -activation:
                    trailing_level = -activation
                
                # Update trailing level if spread moves favorably
                elif trailing_level is not None and zscore.iloc[i] > trailing_level + step:
                    trailing_level = min(0, trailing_level + step)
                
                # Check for trailing stop exit
                if trailing_level is not None and zscore.iloc[i] < trailing_level:
                    signals_with_ts.loc[zscore.index[i], 'exit_long'] = 1
                    position = 0
                    
            elif position == -1:  # Short position
                # Initialize trailing stop if zscore crosses activation level
                if trailing_level is None and zscore.iloc[i] < activation:
                    trailing_level = activation
                
                # Update trailing level if spread moves favorably
                elif trailing_level is not None and zscore.iloc[i] < trailing_level - step:
                    trailing_level = max(0, trailing_level - step)
                
                # Check for trailing stop exit
                if trailing_level is not None and zscore.iloc[i] > trailing_level:
                    signals_with_ts.loc[zscore.index[i], 'exit_short'] = 1
                    position = 0
        
        return signals_with_ts 