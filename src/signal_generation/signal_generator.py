"""
Signal generator module for the Intraday Statistical Arbitrage System.

This module implements signal generation functionality.
"""

from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Signal generator class for generating trading signals.
    """
    
    def __init__(self, entry_threshold: float = 2.0, 
                exit_threshold: float = 0.5,
                use_confirmation: bool = False):
        """
        Initialize the signal generator.
        
        Parameters:
        -----------
        entry_threshold : float, optional
            Z-score threshold for entry signals
        exit_threshold : float, optional
            Z-score threshold for exit signals
        use_confirmation : bool, optional
            Whether to use confirmation filters
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.use_confirmation = use_confirmation
    
    def generate_signal(self, zscore: float, 
                      pair_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on z-score.
        
        Parameters:
        -----------
        zscore : float
            Current z-score value
        pair_info : Dict[str, Any]
            Information about the pair
            
        Returns:
        --------
        Dict[str, Any]
            Trading signal
        """
        # Check if we already have a position
        has_position = 'leg1_position' in pair_info and pair_info['leg1_position'] != 0
        
        if has_position:
            # Check for exit signal
            position_direction = 1 if pair_info.get('leg1_position', 0) > 0 else -1
            
            # For long positions (negative z-score entry)
            if position_direction < 0 and zscore <= self.exit_threshold:
                return {'action': 'close', 'reason': 'target_reached'}
            
            # For short positions (positive z-score entry)
            if position_direction > 0 and zscore >= -self.exit_threshold:
                return {'action': 'close', 'reason': 'target_reached'}
                
            # Hold existing position
            return {'action': 'hold'}
        else:
            # Check for entry signal
            if zscore >= self.entry_threshold:
                return {'action': 'enter', 'direction': 'short', 'zscore': zscore}
            elif zscore <= -self.entry_threshold:
                return {'action': 'enter', 'direction': 'long', 'zscore': zscore}
        
        # No signal
        return {'action': 'none'}

    def generate_signals(self, zscore, volume_data=None, time_data=None):
        """
        Generate entry and exit signals based on z-score and filters.
        
        Parameters:
        -----------
        zscore : pandas.Series
            Z-score series
        volume_data : pandas.DataFrame, optional
            Volume data for both assets in pair
        time_data : pandas.Series, optional
            Series with timestamp data (for time-of-day filtering)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns:
            - entry_long: 1 for long entry, 0 otherwise
            - entry_short: 1 for short entry, 0 otherwise
            - exit_long: 1 for long exit, 0 otherwise
            - exit_short: 1 for short exit, 0 otherwise
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        
        # Generate raw entry and exit signals based on z-score
        signals['entry_long'] = (zscore < -self.entry_threshold).astype(int)
        signals['entry_short'] = (zscore > self.entry_threshold).astype(int)
        signals['exit_long'] = (zscore > -self.exit_threshold).astype(int)
        signals['exit_short'] = (zscore < self.exit_threshold).astype(int)
        
        # Apply volume filter if provided
        if volume_data is not None and self.volume_threshold is not None:
            if isinstance(volume_data, pd.DataFrame) and 'volume' in volume_data.columns:
                # Single volume series
                volume_filter = volume_data['volume'] >= self.volume_threshold
            elif isinstance(volume_data, pd.DataFrame) and len(volume_data.columns) >= 2:
                # Multiple volume series (e.g., for each asset in the pair)
                # Require both assets to have sufficient volume
                volume_filter = (volume_data.iloc[:, 0] >= self.volume_threshold) & \
                                (volume_data.iloc[:, 1] >= self.volume_threshold)
            else:
                # Fallback
                volume_filter = pd.Series(True, index=zscore.index)
                
            # Apply volume filter to entry signals
            signals['entry_long'] = signals['entry_long'] & volume_filter
            signals['entry_short'] = signals['entry_short'] & volume_filter
        
        # Apply time-of-day filter if provided
        if time_data is not None and self.time_filters is not None:
            # Extract hour and minute
            if isinstance(time_data.index, pd.DatetimeIndex):
                hours = time_data.index.hour
                minutes = time_data.index.minute
                
                # Create time filter
                start_time = self.time_filters.get('start_time', time(0, 0))
                end_time = self.time_filters.get('end_time', time(23, 59))
                
                # Convert to minutes since midnight for comparison
                start_minutes = start_time.hour * 60 + start_time.minute
                end_minutes = end_time.hour * 60 + end_time.minute
                current_minutes = hours * 60 + minutes
                
                if start_minutes <= end_minutes:
                    # Normal case (e.g., 9:30 to 16:00)
                    time_filter = (current_minutes >= start_minutes) & (current_minutes <= end_minutes)
                else:
                    # Overnight case (e.g., 22:00 to 8:00)
                    time_filter = (current_minutes >= start_minutes) | (current_minutes <= end_minutes)
                
                # Apply time filter to entry signals
                signals['entry_long'] = signals['entry_long'] & time_filter
                signals['entry_short'] = signals['entry_short'] & time_filter
        
        return signals
    
    def apply_holding_period(self, signals, prices=None, frequency='1min'):
        """
        Apply maximum holding period constraint to signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signal columns
        prices : pandas.DataFrame, optional
            Price data (used for stop loss calculation)
        frequency : str
            Data frequency (e.g., '1min', '5min')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with updated signals and position columns
        """
        # Add position columns
        signals['position_long'] = 0
        signals['position_short'] = 0
        
        # Add holding period counters
        signals['holding_period_long'] = 0
        signals['holding_period_short'] = 0
        
        # Convert max_holding_period to periods based on frequency
        if frequency == '1min':
            max_periods = self.max_holding_period
        elif frequency == '5min':
            max_periods = self.max_holding_period // 5
        elif frequency == '15min':
            max_periods = self.max_holding_period // 15
        elif frequency == '30min':
            max_periods = self.max_holding_period // 30
        elif frequency == '1hour':
            max_periods = self.max_holding_period // 60
        else:
            # Default to the raw value
            max_periods = self.max_holding_period
        
        # Process signals in sequence
        for i in range(1, len(signals)):
            prev_idx = signals.index[i-1]
            curr_idx = signals.index[i]
            
            # Long position logic
            if signals.loc[prev_idx, 'position_long'] == 0 and signals.loc[curr_idx, 'entry_long'] == 1:
                # Enter new long position
                signals.loc[curr_idx, 'position_long'] = 1
                signals.loc[curr_idx, 'holding_period_long'] = 1
            elif signals.loc[prev_idx, 'position_long'] == 1:
                if signals.loc[curr_idx, 'exit_long'] == 1:
                    # Exit existing long position
                    signals.loc[curr_idx, 'position_long'] = 0
                    signals.loc[curr_idx, 'holding_period_long'] = 0
                else:
                    # Continue holding long position
                    signals.loc[curr_idx, 'position_long'] = 1
                    signals.loc[curr_idx, 'holding_period_long'] = signals.loc[prev_idx, 'holding_period_long'] + 1
                    
                    # Force exit if max holding period reached
                    if signals.loc[curr_idx, 'holding_period_long'] >= max_periods:
                        signals.loc[curr_idx, 'position_long'] = 0
                        signals.loc[curr_idx, 'holding_period_long'] = 0
            
            # Short position logic
            if signals.loc[prev_idx, 'position_short'] == 0 and signals.loc[curr_idx, 'entry_short'] == 1:
                # Enter new short position
                signals.loc[curr_idx, 'position_short'] = 1
                signals.loc[curr_idx, 'holding_period_short'] = 1
            elif signals.loc[prev_idx, 'position_short'] == 1:
                if signals.loc[curr_idx, 'exit_short'] == 1:
                    # Exit existing short position
                    signals.loc[curr_idx, 'position_short'] = 0
                    signals.loc[curr_idx, 'holding_period_short'] = 0
                else:
                    # Continue holding short position
                    signals.loc[curr_idx, 'position_short'] = 1
                    signals.loc[curr_idx, 'holding_period_short'] = signals.loc[prev_idx, 'holding_period_short'] + 1
                    
                    # Force exit if max holding period reached
                    if signals.loc[curr_idx, 'holding_period_short'] >= max_periods:
                        signals.loc[curr_idx, 'position_short'] = 0
                        signals.loc[curr_idx, 'holding_period_short'] = 0
        
        # Calculate overall position (1 for long, -1 for short, 0 for flat)
        signals['position'] = signals['position_long'] - signals['position_short']
        
        return signals
    
    def apply_trailing_exit(self, signals, zscore, trail_percent=0.5):
        """
        Apply trailing exit to improve exits based on z-score.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signal columns
        zscore : pandas.Series
            Z-score series
        trail_percent : float
            Percent of max favorable z-score to trail by
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with updated signals
        """
        # Initialize tracking variables
        signals['max_favorable_zscore'] = np.nan
        signals['trailing_exit_level'] = np.nan
        
        # Calculate trailing exit levels
        for i in range(1, len(signals)):
            prev_idx = signals.index[i-1]
            curr_idx = signals.index[i]
            
            # Long position logic (negative z-score is favorable)
            if signals.loc[curr_idx, 'position_long'] == 1:
                curr_zscore = zscore.loc[curr_idx]
                
                # Initialize or update max favorable z-score
                if np.isnan(signals.loc[prev_idx, 'max_favorable_zscore']) or \
                   signals.loc[prev_idx, 'position_long'] == 0:
                    # New position or first tracking
                    signals.loc[curr_idx, 'max_favorable_zscore'] = curr_zscore
                else:
                    # Update if more favorable (more negative)
                    signals.loc[curr_idx, 'max_favorable_zscore'] = min(
                        signals.loc[prev_idx, 'max_favorable_zscore'], 
                        curr_zscore
                    )
                
                # Calculate trailing exit level (move by trail_percent towards zero)
                max_favorable = signals.loc[curr_idx, 'max_favorable_zscore']
                signals.loc[curr_idx, 'trailing_exit_level'] = max_favorable * (1 - trail_percent)
                
                # Add trailing exit signal
                if curr_zscore > signals.loc[curr_idx, 'trailing_exit_level']:
                    signals.loc[curr_idx, 'exit_long'] = 1
            
            # Short position logic (positive z-score is favorable)
            if signals.loc[curr_idx, 'position_short'] == 1:
                curr_zscore = zscore.loc[curr_idx]
                
                # Initialize or update max favorable z-score
                if np.isnan(signals.loc[prev_idx, 'max_favorable_zscore']) or \
                   signals.loc[prev_idx, 'position_short'] == 0:
                    # New position or first tracking
                    signals.loc[curr_idx, 'max_favorable_zscore'] = curr_zscore
                else:
                    # Update if more favorable (more positive)
                    signals.loc[curr_idx, 'max_favorable_zscore'] = max(
                        signals.loc[prev_idx, 'max_favorable_zscore'], 
                        curr_zscore
                    )
                
                # Calculate trailing exit level (move by trail_percent towards zero)
                max_favorable = signals.loc[curr_idx, 'max_favorable_zscore']
                signals.loc[curr_idx, 'trailing_exit_level'] = max_favorable * (1 - trail_percent)
                
                # Add trailing exit signal
                if curr_zscore < signals.loc[curr_idx, 'trailing_exit_level']:
                    signals.loc[curr_idx, 'exit_short'] = 1
        
        return signals
    
    def apply_stop_loss(self, signals, zscore, stop_threshold=3.0):
        """
        Apply stop loss based on extreme z-score levels.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signal columns
        zscore : pandas.Series
            Z-score series
        stop_threshold : float
            Z-score threshold for stop loss
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with updated signals
        """
        # Process signals with stop loss
        for i in range(1, len(signals)):
            curr_idx = signals.index[i]
            curr_zscore = zscore.loc[curr_idx]
            
            # Long position stop loss (z-score becomes too positive)
            if signals.loc[curr_idx, 'position_long'] == 1 and curr_zscore > stop_threshold:
                signals.loc[curr_idx, 'exit_long'] = 1
                signals.loc[curr_idx, 'position_long'] = 0
                signals.loc[curr_idx, 'holding_period_long'] = 0
            
            # Short position stop loss (z-score becomes too negative)
            if signals.loc[curr_idx, 'position_short'] == 1 and curr_zscore < -stop_threshold:
                signals.loc[curr_idx, 'exit_short'] = 1
                signals.loc[curr_idx, 'position_short'] = 0
                signals.loc[curr_idx, 'holding_period_short'] = 0
        
        return signals 