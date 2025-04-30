"""
Z-Score Strategy Backtest module.

This module implements a basic z-score strategy for pairs trading backtesting.
It includes spread calculation, z-score computation, signal generation based on thresholds,
and position management with proper risk controls and transaction costs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class ZScoreStrategyBacktest:
    """
    Z-Score Strategy Backtest class for pairs trading.
    
    This class implements a basic mean-reversion strategy using z-scores
    of the spread between two cointegrated assets.
    """
    
    def __init__(self, 
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0,
                 window_size: int = 20,
                 max_holding_period: Optional[int] = None,
                 use_trailing_stop: bool = False,
                 use_time_filter: bool = False,
                 commission_per_trade: float = 0.0,
                 slippage_per_trade: float = 0.0,
                 account_size: float = 100000,
                 max_position_size: float = 0.25,
                 calculate_method: str = 'rolling',
                 use_log_prices: bool = False):
        """
        Initialize the Z-Score Strategy Backtest.
        
        Parameters:
        -----------
        entry_threshold : float
            Z-score threshold for trade entry (default: 2.0)
        exit_threshold : float
            Z-score threshold for trade exit (default: 0.5)
        stop_loss_threshold : float
            Z-score threshold for stop loss (default: 3.0)
        window_size : int
            Window size for z-score calculation (default: 20)
        max_holding_period : int, optional
            Maximum holding period in bars (default: None, no limit)
        use_trailing_stop : bool
            Whether to use trailing stop (default: False)
        use_time_filter : bool
            Whether to use time of day filter (default: False)
        commission_per_trade : float
            Commission per trade in dollars (default: 0.0)
        slippage_per_trade : float
            Slippage per trade in dollars (default: 0.0)
        account_size : float
            Initial account size in dollars (default: 100000)
        max_position_size : float
            Maximum position size as fraction of account (default: 0.25)
        calculate_method : str
            Method for z-score calculation ('rolling', 'ewm', 'full')
        use_log_prices : bool
            Whether to use log prices (default: False)
        """
        # Strategy parameters
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.window_size = window_size
        self.max_holding_period = max_holding_period
        self.use_trailing_stop = use_trailing_stop
        self.use_time_filter = use_time_filter
        
        # Transaction costs
        self.commission_per_trade = commission_per_trade
        self.slippage_per_trade = slippage_per_trade
        
        # Account parameters
        self.initial_account_size = account_size
        self.max_position_size = max_position_size
        
        # Calculation parameters
        self.calculate_method = calculate_method
        self.use_log_prices = use_log_prices
        
        # Results
        self.results = None
        self.equity_curve = None
        self.trade_history = []
        self.metrics = None
    
    def calculate_hedge_ratio(self, price_series1: pd.Series, price_series2: pd.Series) -> float:
        """
        Calculate the hedge ratio between two price series using OLS regression.
        
        Parameters:
        -----------
        price_series1 : pandas.Series
            Price series of the first asset
        price_series2 : pandas.Series
            Price series of the second asset
            
        Returns:
        --------
        float
            Hedge ratio
        """
        # Apply log transformation if specified
        if self.use_log_prices:
            y = np.log(price_series1)
            x = np.log(price_series2)
        else:
            y = price_series1
            x = price_series2
        
        # Add constant to the independent variable
        x = sm.add_constant(x)
        
        # Run OLS regression
        model = sm.OLS(y, x).fit()
        
        # Extract the beta coefficient (hedge ratio)
        hedge_ratio = model.params[1]
        
        return hedge_ratio
    
    def calculate_spread(self, price_series1: pd.Series, price_series2: pd.Series, 
                         hedge_ratio: Optional[float] = None) -> pd.Series:
        """
        Calculate the spread between two price series.
        
        Parameters:
        -----------
        price_series1 : pandas.Series
            Price series of the first asset
        price_series2 : pandas.Series
            Price series of the second asset
        hedge_ratio : float, optional
            Hedge ratio (if None, will be calculated)
            
        Returns:
        --------
        pandas.Series
            Spread series
        """
        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(price_series1, price_series2)
        
        # Apply log transformation if specified
        if self.use_log_prices:
            y = np.log(price_series1)
            x = np.log(price_series2)
        else:
            y = price_series1
            x = price_series2
        
        # Calculate spread
        spread = y - hedge_ratio * x
        
        return spread
    
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Calculate z-score of the spread.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread series
            
        Returns:
        --------
        pandas.Series
            Z-score series
        """
        # Calculate z-score based on the specified method
        if self.calculate_method == 'rolling':
            # Use rolling window
            mean = spread.rolling(window=self.window_size).mean()
            std = spread.rolling(window=self.window_size).std()
            zscore = (spread - mean) / std
        
        elif self.calculate_method == 'ewm':
            # Use exponentially weighted moving average
            half_life = self.window_size // 2
            mean = spread.ewm(halflife=half_life).mean()
            std = spread.ewm(halflife=half_life).std()
            zscore = (spread - mean) / std
        
        elif self.calculate_method == 'full':
            # Use full history
            mean = spread.mean()
            std = spread.std()
            zscore = (spread - mean) / std
        
        else:
            raise ValueError("Method must be one of: 'rolling', 'ewm', or 'full'")
        
        return zscore
    
    def generate_signals(self, zscore: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on z-score.
        
        Parameters:
        -----------
        zscore : pandas.Series
            Z-score series
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        
        # Generate entry signals
        signals['entry_long'] = (zscore <= -self.entry_threshold).astype(int)
        signals['entry_short'] = (zscore >= self.entry_threshold).astype(int)
        
        # Generate exit signals
        signals['exit_long'] = (zscore >= -self.exit_threshold).astype(int)
        signals['exit_short'] = (zscore <= self.exit_threshold).astype(int)
        
        # Generate stop loss signals
        signals['stop_long'] = (zscore <= -self.stop_loss_threshold).astype(int)
        signals['stop_short'] = (zscore >= self.stop_loss_threshold).astype(int)
        
        return signals
    
    def apply_position_logic(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position logic to signals to create a position series.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with signal columns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with position column added
        """
        # Add position column
        signals['position'] = 0
        
        # Add holding period counter
        signals['holding_period'] = 0
        
        # Process signals sequentially to determine positions
        current_position = 0
        holding_period = 0
        
        for i in range(1, len(signals)):
            prev_idx = signals.index[i-1]
            curr_idx = signals.index[i]
            
            # Get previous position
            current_position = signals.loc[prev_idx, 'position']
            
            # Update holding period if in a position
            if current_position != 0:
                holding_period += 1
            else:
                holding_period = 0
            
            # Check for exit/stop conditions first
            if current_position > 0:  # Long position
                if (signals.loc[curr_idx, 'exit_long'] == 1 or 
                    signals.loc[curr_idx, 'stop_long'] == 1 or
                    (self.max_holding_period is not None and holding_period >= self.max_holding_period)):
                    # Exit long position
                    current_position = 0
                    holding_period = 0
            
            elif current_position < 0:  # Short position
                if (signals.loc[curr_idx, 'exit_short'] == 1 or 
                    signals.loc[curr_idx, 'stop_short'] == 1 or
                    (self.max_holding_period is not None and holding_period >= self.max_holding_period)):
                    # Exit short position
                    current_position = 0
                    holding_period = 0
            
            # Check for entry conditions only if not in a position
            if current_position == 0:
                if signals.loc[curr_idx, 'entry_long'] == 1:
                    # Enter long position
                    current_position = 1
                    holding_period = 0
                elif signals.loc[curr_idx, 'entry_short'] == 1:
                    # Enter short position
                    current_position = -1
                    holding_period = 0
            
            # Store position and holding period
            signals.loc[curr_idx, 'position'] = current_position
            signals.loc[curr_idx, 'holding_period'] = holding_period
        
        return signals
    
    def calculate_returns(self, signals: pd.DataFrame, price1: pd.Series, price2: pd.Series, 
                          hedge_ratio: float) -> pd.DataFrame:
        """
        Calculate strategy returns based on positions and price data.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with position column
        price1 : pandas.Series
            Price series of the first asset
        price2 : pandas.Series
            Price series of the second asset
        hedge_ratio : float
            Hedge ratio between the two assets
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with returns column added
        """
        # Calculate asset returns
        asset1_returns = price1.pct_change()
        asset2_returns = price2.pct_change()
        
        # Initialize returns column
        signals['returns'] = 0.0
        
        # Calculate position sizes based on account size and max position size
        position_value = self.initial_account_size * self.max_position_size
        
        # Calculate pairs returns
        for i in range(1, len(signals)):
            if signals['position'].iloc[i-1] != 0:
                # Calculate returns based on position
                if signals['position'].iloc[i-1] > 0:  # Long spread
                    pair_return = asset1_returns.iloc[i] - hedge_ratio * asset2_returns.iloc[i]
                else:  # Short spread
                    pair_return = -(asset1_returns.iloc[i] - hedge_ratio * asset2_returns.iloc[i])
                
                # Check for position changes (to apply transaction costs)
                is_entry = signals['position'].iloc[i-1] != 0 and signals['position'].iloc[i-2] == 0
                is_exit = signals['position'].iloc[i-1] != 0 and signals['position'].iloc[i] == 0
                
                transaction_costs = 0
                
                if is_entry:
                    # Apply entry transaction costs
                    transaction_costs = (self.commission_per_trade * 2 + self.slippage_per_trade * 2) / position_value
                
                if is_exit:
                    # Apply exit transaction costs
                    transaction_costs += (self.commission_per_trade * 2 + self.slippage_per_trade * 2) / position_value
                
                # Apply transaction costs
                pair_return -= transaction_costs
                
                # Store returns
                signals['returns'].iloc[i] = pair_return
        
        return signals
    
    def backtest(self, price1: pd.Series, price2: pd.Series, hedge_ratio: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a backtest of the z-score strategy.
        
        Parameters:
        -----------
        price1 : pandas.Series
            Price series of the first asset
        price2 : pandas.Series
            Price series of the second asset
        hedge_ratio : float, optional
            Hedge ratio between the two assets (if None, will be calculated)
            
        Returns:
        --------
        dict
            Dictionary with backtest results
        """
        # Ensure index alignment
        common_index = price1.index.intersection(price2.index)
        price1 = price1.loc[common_index]
        price2 = price2.loc[common_index]
        
        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        
        # Calculate spread
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        
        # Calculate z-score
        zscore = self.calculate_zscore(spread)
        
        # Generate signals
        signals = self.generate_signals(zscore)
        
        # Apply position logic
        signals = self.apply_position_logic(signals)
        
        # Calculate returns
        signals = self.calculate_returns(signals, price1, price2, hedge_ratio)
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
        
        # Calculate equity curve
        equity_curve = self.initial_account_size * signals['cumulative_returns']
        
        # Calculate drawdown
        signals['drawdown'] = 1 - signals['cumulative_returns'] / signals['cumulative_returns'].cummax()
        
        # Generate trade history
        trade_history = self._generate_trade_history(signals, price1, price2, hedge_ratio)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(signals)
        
        # Store results
        self.results = signals
        self.equity_curve = equity_curve
        self.trade_history = trade_history
        self.metrics = metrics
        
        # Return results
        return {
            'signals': signals,
            'equity_curve': equity_curve,
            'trade_history': trade_history,
            'metrics': metrics
        }
    
    def _generate_trade_history(self, signals: pd.DataFrame, price1: pd.Series, price2: pd.Series, 
                               hedge_ratio: float) -> List[Dict[str, Any]]:
        """
        Generate trade history from signals.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with position column
        price1 : pandas.Series
            Price series of the first asset
        price2 : pandas.Series
            Price series of the second asset
        hedge_ratio : float
            Hedge ratio between the two assets
            
        Returns:
        --------
        list
            List of trade dictionaries
        """
        trade_history = []
        current_trade = None
        
        for i in range(1, len(signals)):
            prev_idx = signals.index[i-1]
            curr_idx = signals.index[i]
            
            # Check for entries
            if signals['position'].iloc[i-1] == 0 and signals['position'].iloc[i] != 0:
                # New trade entry
                direction = 'long' if signals['position'].iloc[i] > 0 else 'short'
                
                current_trade = {
                    'entry_date': curr_idx,
                    'direction': direction,
                    'entry_price1': price1.iloc[i],
                    'entry_price2': price2.iloc[i],
                    'entry_zscore': signals['zscore'].iloc[i],
                    'entry_spread': price1.iloc[i] - hedge_ratio * price2.iloc[i]
                }
            
            # Check for exits
            if signals['position'].iloc[i-1] != 0 and signals['position'].iloc[i] == 0:
                # Trade exit
                if current_trade is not None:
                    # Complete trade information
                    current_trade['exit_date'] = curr_idx
                    current_trade['exit_price1'] = price1.iloc[i]
                    current_trade['exit_price2'] = price2.iloc[i]
                    current_trade['exit_zscore'] = signals['zscore'].iloc[i]
                    current_trade['exit_spread'] = price1.iloc[i] - hedge_ratio * price2.iloc[i]
                    
                    # Calculate P&L
                    if current_trade['direction'] == 'long':
                        price_change1 = current_trade['exit_price1'] / current_trade['entry_price1'] - 1
                        price_change2 = current_trade['exit_price2'] / current_trade['entry_price2'] - 1
                        spread_return = price_change1 - hedge_ratio * price_change2
                    else:  # short
                        price_change1 = current_trade['exit_price1'] / current_trade['entry_price1'] - 1
                        price_change2 = current_trade['exit_price2'] / current_trade['entry_price2'] - 1
                        spread_return = -(price_change1 - hedge_ratio * price_change2)
                    
                    # Apply transaction costs
                    transaction_costs = (self.commission_per_trade * 4 + self.slippage_per_trade * 4) / (self.initial_account_size * self.max_position_size)
                    net_return = spread_return - transaction_costs
                    
                    current_trade['spread_return'] = spread_return
                    current_trade['transaction_costs'] = transaction_costs
                    current_trade['net_return'] = net_return
                    current_trade['duration'] = (current_trade['exit_date'] - current_trade['entry_date']).total_seconds() / 3600  # in hours
                    
                    # Determine reason for exit
                    if 'holding_period' in signals.columns and signals['holding_period'].iloc[i-1] >= self.max_holding_period:
                        current_trade['exit_reason'] = 'time_limit'
                    elif (current_trade['direction'] == 'long' and signals['zscore'].iloc[i] >= -self.exit_threshold):
                        current_trade['exit_reason'] = 'target'
                    elif (current_trade['direction'] == 'short' and signals['zscore'].iloc[i] <= self.exit_threshold):
                        current_trade['exit_reason'] = 'target'
                    elif (current_trade['direction'] == 'long' and signals['zscore'].iloc[i] <= -self.stop_loss_threshold):
                        current_trade['exit_reason'] = 'stop_loss'
                    elif (current_trade['direction'] == 'short' and signals['zscore'].iloc[i] >= self.stop_loss_threshold):
                        current_trade['exit_reason'] = 'stop_loss'
                    else:
                        current_trade['exit_reason'] = 'unknown'
                    
                    # Add to trade history
                    trade_history.append(current_trade)
                    current_trade = None
        
        return trade_history
    
    def _calculate_metrics(self, signals: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with returns and positions
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        # Calculate basic metrics
        total_return = signals['cumulative_returns'].iloc[-1] - 1
        
        # Calculate annualized return (assuming 252 trading days per year)
        days = (signals.index[-1] - signals.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (252 / days) - 1
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio
        daily_returns = signals['returns'].resample('D').sum()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Calculate drawdown metrics
        max_drawdown = signals['drawdown'].max()
        
        # Calculate trade metrics
        winning_trades = [t for t in self.trade_history if t['net_return'] > 0]
        losing_trades = [t for t in self.trade_history if t['net_return'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        avg_win = np.mean([t['net_return'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['net_return'] for t in losing_trades]) if losing_trades else 0
        profit_factor = sum(t['net_return'] for t in winning_trades) / abs(sum(t['net_return'] for t in losing_trades)) if losing_trades and sum(t['net_return'] for t in losing_trades) != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
    
    def plot_results(self, figsize=(15, 12), save_path=None):
        """
        Plot backtest results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot equity curve
        axs[0].plot(self.equity_curve.index, self.equity_curve, label='Equity Curve')
        axs[0].set_title('Z-Score Strategy Backtest Results')
        axs[0].set_ylabel('Equity ($)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot Z-scores
        axs[1].plot(self.results.index, self.results['zscore'], label='Z-Score')
        axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[1].axhline(y=self.entry_threshold, color='red', linestyle='--', alpha=0.5, label='Entry Threshold')
        axs[1].axhline(y=-self.entry_threshold, color='green', linestyle='--', alpha=0.5)
        axs[1].axhline(y=self.exit_threshold, color='blue', linestyle=':', alpha=0.5, label='Exit Threshold')
        axs[1].axhline(y=-self.exit_threshold, color='blue', linestyle=':', alpha=0.5)
        axs[1].set_ylabel('Z-Score')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot positions
        axs[2].plot(self.results.index, self.results['position'], label='Position')
        axs[2].set_ylabel('Position')
        axs[2].set_xlabel('Date')
        axs[2].legend()
        axs[2].grid(True)
        
        # Add metrics as text box
        if self.metrics:
            metrics_text = '\n'.join([
                f'Total Return: {self.metrics["total_return"]:.2%}',
                f'Annual Return: {self.metrics["annualized_return"]:.2%}',
                f'Sharpe Ratio: {self.metrics["sharpe_ratio"]:.2f}',
                f'Max Drawdown: {self.metrics["max_drawdown"]:.2%}',
                f'Win Rate: {self.metrics["win_rate"]:.2%}',
                f'Profit Factor: {self.metrics["profit_factor"]:.2f}',
                f'Total Trades: {self.metrics["total_trades"]}'
            ])
            
            # Add text box to the plot
            axs[0].text(0.02, 0.05, metrics_text, transform=axs[0].transAxes,
                      fontsize=9, verticalalignment='bottom', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def save_results(self, file_path):
        """
        Save backtest results to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the results
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if self.results is None:
            print("No backtest results available. Run backtest first.")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save results to CSV
        self.results.to_csv(f"{file_path}_signals.csv")
        
        # Save equity curve to CSV
        self.equity_curve.to_csv(f"{file_path}_equity.csv")
        
        # Save metrics to JSON
        with open(f"{file_path}_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save trade history to CSV
        trade_history_df = pd.DataFrame(self.trade_history)
        if not trade_history_df.empty:
            trade_history_df.to_csv(f"{file_path}_trades.csv", index=False)
        
        return True


def run_zscore_backtest(price_data, ticker1, ticker2, **kwargs):
    """
    Run Z-Score strategy backtest function.
    
    Parameters:
    -----------
    price_data : pandas.DataFrame
        DataFrame with price data
    ticker1 : str
        First ticker symbol
    ticker2 : str
        Second ticker symbol
    **kwargs : dict
        Additional parameters for the backtest
        
    Returns:
    --------
    dict
        Dictionary with backtest results
    """
    # Initialize the backtest with provided parameters
    backtest = ZScoreStrategyBacktest(**kwargs)
    
    # Extract price series
    price1 = price_data[ticker1]
    price2 = price_data[ticker2]
    
    # Run the backtest
    results = backtest.backtest(price1, price2)
    
    # Plot results
    if kwargs.get('plot_results', True):
        backtest.plot_results()
    
    # Save results if path provided
    save_path = kwargs.get('save_path')
    if save_path:
        backtest.save_results(save_path)
    
    return results 