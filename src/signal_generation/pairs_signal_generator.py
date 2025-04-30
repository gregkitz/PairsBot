"""
Pairs Trading Signal Generator Module

This module provides signal generation for pairs trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class PairsSignalGenerator:
    """
    Generate trading signals for pairs trading strategies.
    """
    
    def __init__(self, 
                 lookback=20, 
                 entry_zscore=2.0, 
                 exit_zscore=0.5,
                 use_kalman=False,
                 use_rolling_regression=False,
                 regression_window=60,
                 adf_threshold=0.05,
                 min_half_life=1.0,
                 max_half_life=30.0,
                 stop_loss_std=4.0):
        """
        Initialize the PairsSignalGenerator with parameters.
        
        Parameters:
        -----------
        lookback : int
            Number of periods to use for z-score calculation
        entry_zscore : float
            Z-score threshold for entry signals
        exit_zscore : float
            Z-score threshold for exit signals
        use_kalman : bool
            Whether to use Kalman filter for dynamic hedge ratio
        use_rolling_regression : bool
            Whether to use rolling regression for dynamic hedge ratio
        regression_window : int
            Window size for rolling regression
        adf_threshold : float
            Threshold for ADF test p-value
        min_half_life : float
            Minimum half-life for valid pairs
        max_half_life : float
            Maximum half-life for valid pairs
        stop_loss_std : float
            Stop loss as number of standard deviations
        """
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.use_kalman = use_kalman
        self.use_rolling_regression = use_rolling_regression
        self.regression_window = regression_window
        self.adf_threshold = adf_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.stop_loss_std = stop_loss_std
        
        # Store calculation results
        self.hedge_ratio = None
        self.spread = None
        self.z_score = None
        self.half_life = None
        self.adf_p_value = None
        
    def validate_pair(self, price1, price2):
        """
        Validate if a pair is suitable for trading.
        
        Parameters:
        -----------
        price1 : pd.Series
            Price series for the first ticker
        price2 : pd.Series
            Price series for the second ticker
            
        Returns:
        --------
        bool
            True if the pair is valid, False otherwise
        """
        # Calculate spread with fixed hedge ratio first
        hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        
        # Calculate half-life
        half_life = self.calculate_half_life(spread)
        
        # Run ADF test
        adf_result = adfuller(spread.dropna())
        p_value = adf_result[1]
        
        # Store results
        self.half_life = half_life
        self.adf_p_value = p_value
        
        # Check validity
        if p_value <= self.adf_threshold and \
           half_life >= self.min_half_life and \
           half_life <= self.max_half_life:
            return True
        else:
            logger.info(f"Pair validation failed: p-value={p_value:.4f}, half-life={half_life:.2f}")
            return False
    
    def calculate_hedge_ratio(self, price1, price2):
        """
        Calculate the hedge ratio between two price series.
        
        Parameters:
        -----------
        price1 : pd.Series
            Price series for the first ticker
        price2 : pd.Series
            Price series for the second ticker
            
        Returns:
        --------
        float
            Hedge ratio
        """
        # Ensure price series have matching indices
        common_idx = price1.index.intersection(price2.index)
        price1 = price1[common_idx]
        price2 = price2[common_idx]
        
        # Use log prices
        log_price1 = np.log(price1)
        log_price2 = np.log(price2)
        
        # Calculate hedge ratio using OLS
        X = sm.add_constant(log_price2)
        model = OLS(log_price1, X).fit()
        hedge_ratio = model.params.iloc[1]
        
        self.hedge_ratio = hedge_ratio
        return hedge_ratio
    
    def calculate_rolling_hedge_ratio(self, price1, price2):
        """
        Calculate rolling hedge ratio between two price series.
        
        Parameters:
        -----------
        price1 : pd.Series
            Price series for the first ticker
        price2 : pd.Series
            Price series for the second ticker
            
        Returns:
        --------
        pd.Series
            Series of hedge ratios
        """
        # Ensure price series have matching indices
        common_idx = price1.index.intersection(price2.index)
        price1 = price1[common_idx]
        price2 = price2[common_idx]
        
        # Use log prices
        log_price1 = np.log(price1)
        log_price2 = np.log(price2)
        
        # Calculate rolling hedge ratio
        hedge_ratios = pd.Series(index=log_price1.index, dtype=float)
        hedge_ratios[:] = np.nan
        
        for i in range(self.regression_window, len(log_price1)):
            window_y = log_price1.iloc[i-self.regression_window:i]
            window_x = log_price2.iloc[i-self.regression_window:i]
            
            # Add constant
            X = sm.add_constant(window_x)
            model = OLS(window_y, X).fit()
            hedge_ratios.iloc[i] = model.params.iloc[1]
        
        # Forward fill NaN values
        hedge_ratios = hedge_ratios.fillna(method='ffill')
        
        return hedge_ratios
    
    def calculate_spread(self, price1, price2, hedge_ratio=None, dynamic_hedge_ratio=None):
        """
        Calculate the spread between two price series.
        
        Parameters:
        -----------
        price1 : pd.Series
            Price series for the first ticker
        price2 : pd.Series
            Price series for the second ticker
        hedge_ratio : float, optional
            Fixed hedge ratio to use
        dynamic_hedge_ratio : pd.Series, optional
            Series of hedge ratios to use
            
        Returns:
        --------
        pd.Series
            Spread series
        """
        # Ensure price series have matching indices
        common_idx = price1.index.intersection(price2.index)
        price1 = price1[common_idx]
        price2 = price2[common_idx]
        
        # Use log prices
        log_price1 = np.log(price1)
        log_price2 = np.log(price2)
        
        # Calculate spread
        if dynamic_hedge_ratio is not None:
            # Use dynamic hedge ratio
            dynamic_hedge_ratio = dynamic_hedge_ratio[common_idx]
            spread = log_price1 - dynamic_hedge_ratio * log_price2
        else:
            # Use fixed hedge ratio
            if hedge_ratio is None:
                hedge_ratio = self.calculate_hedge_ratio(price1, price2)
            
            spread = log_price1 - hedge_ratio * log_price2
        
        self.spread = spread
        return spread
    
    def calculate_half_life(self, spread):
        """
        Calculate the half-life of mean reversion for a spread series.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread series
            
        Returns:
        --------
        float
            Half-life of mean reversion
        """
        # Filter out any NaN values
        spread = spread.dropna()
        
        # Check if we have enough data points
        if len(spread) < 20:
            logger.warning("Not enough data points to calculate half-life")
            return np.nan
        
        try:
            # Calculate lagged spread and delta
            spread_lag = spread.shift(1)
            delta_spread = spread - spread_lag
            spread_lag = spread_lag.dropna()
            delta_spread = delta_spread.dropna()
            
            # Regression delta_spread on spread_lag
            spread_lag = sm.add_constant(spread_lag)
            model = OLS(delta_spread, spread_lag).fit()
            
            # Calculate half-life
            half_life = -np.log(2) / model.params.iloc[1]
            
            # Return half-life if it's positive and finite
            if np.isfinite(half_life) and half_life > 0:
                return half_life
            else:
                return np.nan
        except Exception as e:
            logger.warning(f"Error calculating half-life: {e}")
            return np.nan
    
    def generate_signals(self, price1, price2, hedge_ratio=None):
        """
        Generate trading signals for a pair of price series.
        
        Parameters:
        -----------
        price1 : pd.Series
            Price series for the first ticker
        price2 : pd.Series
            Price series for the second ticker
        hedge_ratio : float, optional
            Hedge ratio to use. If None, calculate it.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals (position, entry_price, exit_price)
        """
        # Ensure price series have matching indices
        common_idx = price1.index.intersection(price2.index)
        price1 = price1[common_idx]
        price2 = price2[common_idx]
        
        # Calculate hedge ratio and spread
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        
        # Calculate dynamic hedge ratio if needed
        dynamic_hedge_ratio = None
        if self.use_rolling_regression:
            dynamic_hedge_ratio = self.calculate_rolling_hedge_ratio(price1, price2)
            spread = self.calculate_spread(price1, price2, dynamic_hedge_ratio=dynamic_hedge_ratio)
        else:
            spread = self.calculate_spread(price1, price2, hedge_ratio)
        
        # Calculate z-score
        spread_mean = spread.rolling(window=self.lookback).mean()
        spread_std = spread.rolling(window=self.lookback).std()
        z_score = (spread - spread_mean) / spread_std
        
        self.z_score = z_score
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['position'] = 0  # 1 for long, -1 for short, 0 for flat
        signals['spread_mean'] = spread_mean
        signals['spread_std'] = spread_std
        signals['upper_band'] = spread_mean + self.entry_zscore * spread_std
        signals['lower_band'] = spread_mean - self.entry_zscore * spread_std
        signals['stop_loss_upper'] = spread_mean + self.stop_loss_std * spread_std
        signals['stop_loss_lower'] = spread_mean - self.stop_loss_std * spread_std
        
        # Generate signals based on z-score
        for i in range(self.lookback, len(signals)):
            # Skip if NaN values
            if np.isnan(signals['z_score'].iloc[i]):
                continue
                
            # Set current position to previous position by default
            signals.loc[signals.index[i], 'position'] = signals['position'].iloc[i-1]
            
            # Check for stop loss
            if signals['position'].iloc[i-1] == 1 and spread.iloc[i] > signals['stop_loss_upper'].iloc[i]:
                signals.loc[signals.index[i], 'position'] = 0  # Stop loss on long
            elif signals['position'].iloc[i-1] == -1 and spread.iloc[i] < signals['stop_loss_lower'].iloc[i]:
                signals.loc[signals.index[i], 'position'] = 0  # Stop loss on short
            # Check for entry/exit signals
            elif signals['position'].iloc[i-1] == 0:  # If no position
                if signals['z_score'].iloc[i] < -self.entry_zscore:
                    signals.loc[signals.index[i], 'position'] = 1  # Go long
                elif signals['z_score'].iloc[i] > self.entry_zscore:
                    signals.loc[signals.index[i], 'position'] = -1  # Go short
            elif signals['position'].iloc[i-1] == 1:  # If long
                if signals['z_score'].iloc[i] > -self.exit_zscore:
                    signals.loc[signals.index[i], 'position'] = 0  # Exit long
            elif signals['position'].iloc[i-1] == -1:  # If short
                if signals['z_score'].iloc[i] < self.exit_zscore:
                    signals.loc[signals.index[i], 'position'] = 0  # Exit short
        
        # Add trade entry and exit prices
        signals['entry_price1'] = np.nan
        signals['entry_price2'] = np.nan
        signals['exit_price1'] = np.nan
        signals['exit_price2'] = np.nan
        
        # Find entry and exit points
        position_changes = signals['position'].diff()
        
        # Entry points (non-zero position after zero position)
        entries = (position_changes != 0) & (signals['position'] != 0)
        signals.loc[entries, 'entry_price1'] = price1[entries]
        signals.loc[entries, 'entry_price2'] = price2[entries]
        
        # Exit points (zero position after non-zero position)
        exits = (position_changes != 0) & (signals['position'] == 0)
        signals.loc[exits, 'exit_price1'] = price1[exits]
        signals.loc[exits, 'exit_price2'] = price2[exits]
        
        return signals
    
    def plot_signals(self, signals, price1=None, price2=None, title=None, save_path=None):
        """
        Plot the spread, z-score, and trading signals.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with signals
        price1 : pd.Series, optional
            Price series for the first ticker
        price2 : pd.Series, optional
            Price series for the second ticker
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        None
        """
        fig = plt.figure(figsize=(15, 12))
        
        # Plot spread
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(signals['spread'], label='Spread')
        ax1.plot(signals['spread_mean'], label='Mean', color='r')
        ax1.plot(signals['upper_band'], label='Upper Band', color='g', linestyle='--')
        ax1.plot(signals['lower_band'], label='Lower Band', color='g', linestyle='--')
        ax1.plot(signals['stop_loss_upper'], label='Stop Loss Upper', color='r', linestyle=':')
        ax1.plot(signals['stop_loss_lower'], label='Stop Loss Lower', color='r', linestyle=':')
        
        # Mark entry and exit points on spread
        long_entries = (signals['position'].shift(1) == 0) & (signals['position'] == 1)
        long_exits = (signals['position'].shift(1) == 1) & (signals['position'] == 0)
        short_entries = (signals['position'].shift(1) == 0) & (signals['position'] == -1)
        short_exits = (signals['position'].shift(1) == -1) & (signals['position'] == 0)
        
        ax1.scatter(signals.index[long_entries], signals['spread'][long_entries], 
                   marker='^', color='g', s=100, label='Long Entry')
        ax1.scatter(signals.index[long_exits], signals['spread'][long_exits], 
                   marker='v', color='r', s=100, label='Long Exit')
        ax1.scatter(signals.index[short_entries], signals['spread'][short_entries], 
                   marker='v', color='r', s=100, label='Short Entry')
        ax1.scatter(signals.index[short_exits], signals['spread'][short_exits], 
                   marker='^', color='g', s=100, label='Short Exit')
        
        ax1.set_title('Spread')
        ax1.legend()
        ax1.grid(True)
        
        # Plot z-score
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(signals['z_score'], label='Z-Score')
        ax2.axhline(y=0, color='r', linestyle='-')
        ax2.axhline(y=self.entry_zscore, color='g', linestyle='--', label='Entry Threshold')
        ax2.axhline(y=-self.entry_zscore, color='g', linestyle='--')
        ax2.axhline(y=self.exit_zscore, color='b', linestyle='--', label='Exit Threshold')
        ax2.axhline(y=-self.exit_zscore, color='b', linestyle='--')
        ax2.set_title('Z-Score')
        ax2.legend()
        ax2.grid(True)
        
        # Plot position
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(signals['position'], label='Position')
        ax3.set_title('Position')
        ax3.legend()
        ax3.grid(True)
        
        # Set main title
        if title is None:
            if hasattr(price1, 'name') and hasattr(price2, 'name'):
                title = f"{price1.name}-{price2.name} Pair Trading Signals"
            else:
                title = "Pair Trading Signals"
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def run_strategy(self, price1, price2, hedge_ratio=None, plot=True, save_path=None):
        """
        Run the pairs trading strategy on a pair of price series.
        
        Parameters:
        -----------
        price1 : pd.Series
            Price series for the first ticker
        price2 : pd.Series
            Price series for the second ticker
        hedge_ratio : float, optional
            Hedge ratio to use. If None, calculate it.
        plot : bool
            Whether to plot the signals
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals and statistics
        """
        # Set the hedge ratio
        if hedge_ratio is not None:
            self.hedge_ratio = hedge_ratio
            
        # Validate the pair
        if not self.validate_pair(price1, price2):
            logger.warning("Pair failed validation, not suitable for trading")
            return None
        
        # Generate signals
        signals = self.generate_signals(price1, price2, hedge_ratio)
        
        # Plot signals if requested
        if plot:
            self.plot_signals(signals, price1, price2, save_path=save_path)
        
        # Calculate performance statistics
        stats = self.calculate_performance(signals, price1, price2)
        
        # Add statistics to signals dataframe
        for key, value in stats.items():
            signals.loc[:, f'stat_{key}'] = value
        
        return signals
    
    def calculate_performance(self, signals, price1, price2):
        """
        Calculate performance statistics for the strategy.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with signals
        price1 : pd.Series
            Price series for the first ticker
        price2 : pd.Series
            Price series for the second ticker
            
        Returns:
        --------
        dict
            Dictionary with performance statistics
        """
        # Ensure hedge_ratio is set
        if self.hedge_ratio is None:
            self.hedge_ratio = 1.0  # Default to 1.0 if not set
            
        # Count trades
        position_changes = signals['position'].diff()
        entries = (position_changes != 0) & (signals['position'] != 0)
        exits = (position_changes != 0) & (signals['position'] == 0)
        
        total_trades = entries.sum()
        
        # Calculate P&L for each trade
        trade_pnl = []
        open_position = 0
        entry_price1 = 0
        entry_price2 = 0
        
        for i in range(1, len(signals)):
            curr_pos = signals['position'].iloc[i]
            prev_pos = signals['position'].iloc[i-1]
            
            # Entry
            if curr_pos != 0 and prev_pos == 0:
                open_position = curr_pos
                entry_price1 = price1.iloc[i]
                entry_price2 = price2.iloc[i]
            
            # Exit
            elif curr_pos == 0 and prev_pos != 0:
                exit_price1 = price1.iloc[i]
                exit_price2 = price2.iloc[i]
                
                # Calculate P&L (assumption: equal dollar amounts in each leg)
                if prev_pos == 1:  # Long spread (long ticker1, short ticker2)
                    pnl = (exit_price1 / entry_price1 - 1) - self.hedge_ratio * (exit_price2 / entry_price2 - 1)
                else:  # Short spread (short ticker1, long ticker2)
                    pnl = -1 * ((exit_price1 / entry_price1 - 1) - self.hedge_ratio * (exit_price2 / entry_price2 - 1))
                
                trade_pnl.append(pnl)
                
                open_position = 0
        
        # Calculate statistics
        if trade_pnl:
            win_trades = sum(1 for pnl in trade_pnl if pnl > 0)
            lose_trades = sum(1 for pnl in trade_pnl if pnl <= 0)
            win_rate = win_trades / len(trade_pnl) if trade_pnl else 0
            avg_win = np.mean([pnl for pnl in trade_pnl if pnl > 0]) if any(pnl > 0 for pnl in trade_pnl) else 0
            avg_loss = np.mean([pnl for pnl in trade_pnl if pnl <= 0]) if any(pnl <= 0 for pnl in trade_pnl) else 0
            profit_factor = -sum(pnl for pnl in trade_pnl if pnl > 0) / sum(pnl for pnl in trade_pnl if pnl < 0) if sum(pnl for pnl in trade_pnl if pnl < 0) else np.inf
            
            stats = {
                'total_trades': total_trades,
                'win_trades': win_trades,
                'lose_trades': lose_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': sum(trade_pnl),
                'sharpe_ratio': np.mean(trade_pnl) / np.std(trade_pnl) * np.sqrt(252 / total_trades) if np.std(trade_pnl) > 0 else 0
            }
        else:
            stats = {
                'total_trades': 0,
                'win_trades': 0,
                'lose_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0
            }
        
        return stats 