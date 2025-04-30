"""
Script to run intraday pairs trading strategy with ML enhancements.

This script implements a real-time or simulated real-time intraday pairs trading
strategy that uses machine learning to enhance signals and adapt to changing
market conditions.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import logging
import joblib
import time as tm
import argparse
import pytz

from src.ml_enhancements.intraday_signals import IntradaySignalProcessor
from train_intraday_models import (
    load_portfolio_config,
    calculate_spread
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntradayStrategy:
    """
    Class to implement intraday pairs trading strategy with ML enhancements.
    """
    
    def __init__(self, config_file, data_source="file", live_mode=False):
        """
        Initialize the intraday strategy.
        
        Parameters:
        -----------
        config_file : str
            Path to the configuration file
        data_source : str
            Source of data ('file' for historical data, 'api' for live data)
        live_mode : bool
            Whether to run in live mode (True) or simulation mode (False)
        """
        self.config_file = config_file
        self.data_source = data_source
        self.live_mode = live_mode
        
        # Load configuration
        self.config = self._load_config()
        
        if self.config is None:
            raise ValueError("Failed to load configuration")
        
        # Initialize signal processors for each pair
        self.signal_processors = {}
        self.initialize_processors()
        
        # Initialize data structures
        self.prices = {}
        self.volumes = {}
        self.spreads = {}
        self.signals = {}
        self.positions = {}
        self.trades = []
        
        # Initialize performance tracking
        self.portfolio_value = 1.0
        self.portfolio_history = []
        
        # Initialize market hours
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Initialize timezone
        self.timezone = pytz.timezone('US/Eastern')
        
        logger.info("Initialized intraday strategy")
    
    def _load_config(self):
        """
        Load strategy configuration from file.
        
        Returns:
        --------
        dict
            Strategy configuration
        """
        logger.info(f"Loading configuration from {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration with {len(config['pairs'])} pairs")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return None
    
    def initialize_processors(self):
        """Initialize signal processors for each pair."""
        for pair in self.config['pairs']:
            pair_id = pair['pair_id']
            
            # Create signal processor
            processor = IntradaySignalProcessor()
            
            # Load models
            if not processor.signal_enhancer.load_models():
                logger.warning(f"Failed to load models for pair {pair_id}")
            
            self.signal_processors[pair_id] = processor
            
            # Initialize empty positions
            self.positions[pair_id] = {
                'position': 0,
                'entry_price': 0,
                'entry_time': None,
                'size': 0,
                'pair_config': pair
            }
        
        logger.info(f"Initialized signal processors for {len(self.signal_processors)} pairs")
    
    def is_market_open(self, current_time=None):
        """
        Check if the market is currently open.
        
        Parameters:
        -----------
        current_time : datetime, optional
            Time to check. If None, use current time.
            
        Returns:
        --------
        bool
            True if market is open, False otherwise
        """
        if current_time is None:
            current_time = datetime.now(self.timezone)
        
        # Check if it's a weekday
        if current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check if it's within market hours
        current_time_of_day = current_time.time()
        return self.market_open <= current_time_of_day < self.market_close 
    
    def load_historical_data(self, symbol, start_date, end_date, timeframe="5min", data_dir="data/processed"):
        """
        Load historical data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to load data for
        start_date : str
            Start date for data (YYYY-MM-DD)
        end_date : str
            End date for data (YYYY-MM-DD)
        timeframe : str
            Timeframe to resample data to
        data_dir : str
            Directory containing processed data files
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with historical data
        """
        try:
            file_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
            
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                return None
            
            # Load data
            df = pd.read_parquet(file_path)
            
            # Ensure data is sorted by timestamp
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    logger.warning(f"No timestamp column found for {symbol}")
                    return None
            
            df.sort_index(inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Resample to desired timeframe
            if timeframe != "1min":
                # For OHLCV data
                ohlcv_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                
                # Extract available columns
                resample_dict = {col: ohlcv_dict[col] for col in ohlcv_dict.keys() if col in df.columns}
                
                # Resample
                df = df.resample(timeframe).agg(resample_dict)
            
            # Keep only market hours (9:30 AM - 4:00 PM Eastern)
            df = df.between_time('09:30', '16:00')
            
            logger.info(f"Loaded data for {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def initialize_simulation(self, start_date, end_date, timeframe="5min"):
        """
        Initialize a simulation run with historical data.
        
        Parameters:
        -----------
        start_date : str
            Start date for simulation (YYYY-MM-DD)
        end_date : str
            End date for simulation (YYYY-MM-DD)
        timeframe : str
            Timeframe to use for simulation
            
        Returns:
        --------
        bool
            True if initialization was successful, False otherwise
        """
        logger.info(f"Initializing simulation from {start_date} to {end_date}")
        
        # Load data for each pair
        for pair in self.config['pairs']:
            pair_id = pair['pair_id']
            symbol1, symbol2 = pair_id.split('_')
            
            # Load data for each symbol
            symbol1_data = self.load_historical_data(symbol1, start_date, end_date, timeframe)
            symbol2_data = self.load_historical_data(symbol2, start_date, end_date, timeframe)
            
            if symbol1_data is None or symbol2_data is None:
                logger.error(f"Failed to load data for pair {pair_id}")
                return False
            
            # Store data
            self.prices[pair_id] = pd.DataFrame({
                symbol1: symbol1_data['close'],
                symbol2: symbol2_data['close']
            })
            
            # Store volume data if available
            if 'volume' in symbol1_data.columns and 'volume' in symbol2_data.columns:
                self.volumes[pair_id] = pd.DataFrame({
                    symbol1: symbol1_data['volume'],
                    symbol2: symbol2_data['volume']
                })
            
            # Calculate initial spread
            self.spreads[pair_id] = calculate_spread(
                self.prices[pair_id], 
                symbol1, 
                symbol2, 
                pair['config']
            )
            
            # Initialize signals
            self.signals[pair_id] = pd.Series(0, index=self.prices[pair_id].index)
        
        # Set the simulation timeframe
        self.simulation_timeframe = timeframe
        
        # Initialize simulation index
        self.current_index = 0
        self.simulation_indices = list(self.prices[list(self.prices.keys())[0]].index)
        
        logger.info(f"Simulation initialized with {len(self.simulation_indices)} time steps")
        return True
    
    def get_current_data(self, current_time=None):
        """
        Get current market data for all pairs.
        
        In simulation mode, this returns data for the current simulation step.
        In live mode, this fetches the latest data from the data source.
        
        Parameters:
        -----------
        current_time : datetime, optional
            Current time for simulation mode
            
        Returns:
        --------
        dict
            Dictionary with current data
        """
        if not self.live_mode:
            # Simulation mode
            if self.current_index >= len(self.simulation_indices):
                return None  # End of simulation
            
            current_time = self.simulation_indices[self.current_index]
            
            # Get data for current time
            current_data = {}
            
            for pair_id in self.prices:
                symbol1, symbol2 = pair_id.split('_')
                
                if current_time in self.prices[pair_id].index:
                    current_data[pair_id] = {
                        'prices': self.prices[pair_id].loc[current_time],
                        'spread': self.spreads[pair_id].loc[current_time] if current_time in self.spreads[pair_id].index else None,
                        'volume': self.volumes[pair_id].loc[current_time] if pair_id in self.volumes and current_time in self.volumes[pair_id].index else None
                    }
                else:
                    current_data[pair_id] = None
            
            self.current_index += 1
            return current_data
        else:
            # Live mode - implement API fetching here
            # For now, return placeholder data
            logger.warning("Live mode not yet implemented")
            return None
    
    def process_signals(self, current_data, current_time):
        """
        Process signals for all pairs based on current data.
        
        Parameters:
        -----------
        current_data : dict
            Dictionary with current market data
        current_time : datetime
            Current time
            
        Returns:
        --------
        dict
            Dictionary with updated signals
        """
        updated_signals = {}
        
        for pair_id, data in current_data.items():
            if data is None:
                continue
            
            symbol1, symbol2 = pair_id.split('_')
            pair_config = next(pair for pair in self.config['pairs'] if pair['pair_id'] == pair_id)
            
            # Get the last N data points for processing
            lookback = pair_config['config'].get('lookback', 50)
            
            # Get historical prices
            if pair_id not in self.prices or len(self.prices[pair_id]) < lookback:
                logger.warning(f"Not enough historical data for pair {pair_id}")
                continue
            
            # Get the current position
            current_position = self.positions[pair_id]['position']
            
            # Get the signal processor
            processor = self.signal_processors[pair_id]
            
            # Process signals
            try:
                # Get latest prices for the pair
                latest_prices = self.prices[pair_id].iloc[-lookback:]
                
                # Get latest spread
                if pair_id in self.spreads:
                    latest_spread = self.spreads[pair_id].iloc[-lookback:]
                else:
                    latest_spread = calculate_spread(
                        latest_prices, 
                        symbol1, 
                        symbol2, 
                        pair_config['config']
                    )
                
                # Get latest signals (last N observations)
                latest_signals = self.signals[pair_id].iloc[-lookback:]
                
                # Get latest volumes if available
                latest_volumes = None
                if pair_id in self.volumes:
                    latest_volumes = self.volumes[pair_id].iloc[-lookback:]
                
                # Process intraday signals with ML enhancement
                enhanced_signal, metrics = processor.process_intraday_signals(
                    latest_prices,
                    latest_spread,
                    latest_signals,
                    latest_volumes
                )
                
                # Get the latest signal (most recent)
                latest_signal = enhanced_signal.iloc[-1] if len(enhanced_signal) > 0 else 0
                
                # Update signal
                updated_signals[pair_id] = latest_signal
                
                # Log signal change
                if latest_signal != current_position:
                    logger.info(f"Signal change for {pair_id}: {current_position} -> {latest_signal}")
            except Exception as e:
                logger.error(f"Error processing signals for {pair_id}: {e}")
                updated_signals[pair_id] = current_position  # Maintain current position on error 
        
        return updated_signals
    
    def execute_trades(self, updated_signals, current_data, current_time):
        """
        Execute trades based on updated signals.
        
        Parameters:
        -----------
        updated_signals : dict
            Dictionary with updated signals
        current_data : dict
            Dictionary with current market data
        current_time : datetime
            Current time
            
        Returns:
        --------
        list
            List of executed trades
        """
        executed_trades = []
        
        for pair_id, signal in updated_signals.items():
            if pair_id not in current_data or current_data[pair_id] is None:
                continue
            
            symbol1, symbol2 = pair_id.split('_')
            pair_config = next(pair for pair in self.config['pairs'] if pair['pair_id'] == pair_id)
            
            # Get the current position
            current_position = self.positions[pair_id]['position']
            
            # Check if signal has changed
            if signal != current_position:
                # Get current spread price
                if 'spread' in current_data[pair_id] and current_data[pair_id]['spread'] is not None:
                    spread_price = current_data[pair_id]['spread']['spread']
                else:
                    # Calculate spread price from individual prices
                    hedge_ratio = pair_config['config'].get('hedge_ratio', 1.0)
                    prices = current_data[pair_id]['prices']
                    spread_price = prices[symbol1] - (hedge_ratio * prices[symbol2])
                
                # Create trade record
                trade = {
                    'pair_id': pair_id,
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'time': current_time,
                    'action': 'close' if signal == 0 else ('open_long' if signal > 0 else 'open_short'),
                    'prev_position': current_position,
                    'new_position': signal,
                    'spread_price': spread_price,
                    'price1': current_data[pair_id]['prices'][symbol1],
                    'price2': current_data[pair_id]['prices'][symbol2]
                }
                
                # If closing a position, calculate PnL
                if signal == 0 and current_position != 0:
                    entry_price = self.positions[pair_id]['entry_price']
                    entry_time = self.positions[pair_id]['entry_time']
                    
                    if current_position > 0:  # Long position
                        pnl = -(spread_price - entry_price)
                    else:  # Short position
                        pnl = spread_price - entry_price
                    
                    trade['entry_price'] = entry_price
                    trade['entry_time'] = entry_time
                    trade['exit_price'] = spread_price
                    trade['exit_time'] = current_time
                    trade['pnl'] = pnl
                    trade['holding_minutes'] = (current_time - entry_time).total_seconds() / 60 if entry_time is not None else 0
                
                # Update position
                self.positions[pair_id]['position'] = signal
                
                # If opening a position, record entry price and time
                if signal != 0 and current_position == 0:
                    self.positions[pair_id]['entry_price'] = spread_price
                    self.positions[pair_id]['entry_time'] = current_time
                
                # Log trade
                if current_position == 0 and signal != 0:
                    logger.info(f"Opening {pair_id} {'LONG' if signal > 0 else 'SHORT'} at {spread_price:.4f}")
                elif current_position != 0 and signal == 0:
                    pnl = trade.get('pnl', 0)
                    logger.info(f"Closing {pair_id} {'LONG' if current_position > 0 else 'SHORT'} at {spread_price:.4f} with PnL {pnl:.4f}")
                elif current_position != 0 and signal != 0 and current_position != signal:
                    logger.info(f"Flipping {pair_id} {current_position} -> {signal} at {spread_price:.4f}")
                
                # Add trade to executed trades
                executed_trades.append(trade)
                
                # Add to trade history
                self.trades.append(trade)
        
        return executed_trades
    
    def update_portfolio_value(self, executed_trades, current_data, current_time):
        """
        Update portfolio value based on executed trades and current positions.
        
        Parameters:
        -----------
        executed_trades : list
            List of executed trades
        current_data : dict
            Dictionary with current market data
        current_time : datetime
            Current time
            
        Returns:
        --------
        float
            Updated portfolio value
        """
        # Process closed trades for PnL
        for trade in executed_trades:
            if trade['action'] == 'close':
                pnl = trade.get('pnl', 0)
                
                # Update portfolio value
                self.portfolio_value *= (1 + pnl * self.config.get('position_size', 0.1))
        
        # Calculate unrealized PnL for open positions
        unrealized_pnl = 0
        
        for pair_id, position in self.positions.items():
            if position['position'] == 0:
                continue
            
            if pair_id not in current_data or current_data[pair_id] is None:
                continue
            
            # Get current spread price
            if 'spread' in current_data[pair_id] and current_data[pair_id]['spread'] is not None:
                spread_price = current_data[pair_id]['spread']['spread']
            else:
                # Calculate spread price from individual prices
                symbol1, symbol2 = pair_id.split('_')
                hedge_ratio = position['pair_config']['config'].get('hedge_ratio', 1.0)
                prices = current_data[pair_id]['prices']
                spread_price = prices[symbol1] - (hedge_ratio * prices[symbol2])
            
            # Calculate unrealized PnL
            entry_price = position['entry_price']
            
            if position['position'] > 0:  # Long position
                pos_pnl = -(spread_price - entry_price)
            else:  # Short position
                pos_pnl = spread_price - entry_price
            
            unrealized_pnl += pos_pnl * self.config.get('position_size', 0.1)
        
        # Add current portfolio value to history
        self.portfolio_history.append({
            'time': current_time,
            'value': self.portfolio_value,
            'unrealized_pnl': unrealized_pnl
        })
        
        return self.portfolio_value
    
    def run_simulation(self, start_date, end_date, timeframe="5min"):
        """
        Run a simulation of the intraday strategy.
        
        Parameters:
        -----------
        start_date : str
            Start date for simulation (YYYY-MM-DD)
        end_date : str
            End date for simulation (YYYY-MM-DD)
        timeframe : str
            Timeframe to use for simulation
            
        Returns:
        --------
        dict
            Simulation results
        """
        # Initialize simulation
        if not self.initialize_simulation(start_date, end_date, timeframe):
            logger.error("Failed to initialize simulation")
            return None
        
        # Run simulation
        logger.info("Starting simulation")
        
        # Reset portfolio value
        self.portfolio_value = 1.0
        self.portfolio_history = []
        self.trades = []
        
        # Simulation loop
        while True:
            # Get current data
            current_data = self.get_current_data()
            
            if current_data is None:
                break  # End of simulation
            
            current_time = self.simulation_indices[self.current_index - 1]
            
            # Process signals
            updated_signals = self.process_signals(current_data, current_time)
            
            # Execute trades
            executed_trades = self.execute_trades(updated_signals, current_data, current_time)
            
            # Update portfolio value
            self.update_portfolio_value(executed_trades, current_data, current_time)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # Prepare results
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe,
            'portfolio_history': self.portfolio_history,
            'trades': self.trades,
            'metrics': metrics
        }
        
        logger.info(f"Simulation completed with {len(self.trades)} trades")
        logger.info(f"Final portfolio value: {self.portfolio_value:.4f}")
        
        return results
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the strategy.
        
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        if not self.portfolio_history:
            return {}
        
        # Extract portfolio values
        values = [entry['value'] for entry in self.portfolio_history]
        times = [entry['time'] for entry in self.portfolio_history]
        
        # Calculate returns
        returns = np.diff(values) / values[:-1]
        
        # Calculate metrics
        total_return = (values[-1] / values[0]) - 1
        
        # Calculate daily returns
        if isinstance(times[0], pd.Timestamp):
            daily_returns = pd.Series(returns, index=times[1:]).resample('D').sum()
        else:
            daily_returns = pd.Series(returns)
        
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        max_drawdown = min(drawdown)
        
        # Calculate trade metrics
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        profit_factor = sum(t.get('pnl', 0) for t in winning_trades) / abs(sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades else float('inf')
        
        # Calculate annualized return
        days = (times[-1] - times[0]).days
        annual_return = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(self.trades),
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades)
        }

    def plot_portfolio_performance(self, output_file=None):
        """
        Plot portfolio performance over time.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the plot to. If None, display the plot.
            
        Returns:
        --------
        str
            Path to the saved plot file
        """
        if not self.portfolio_history:
            logger.warning("No portfolio history to plot")
            return None
        
        # Extract portfolio values
        values = [entry['value'] for entry in self.portfolio_history]
        times = [entry['time'] for entry in self.portfolio_history]
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio value
        axs[0].plot(times, values, label='Portfolio Value')
        axs[0].set_title('Portfolio Performance')
        axs[0].set_ylabel('Portfolio Value')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Calculate and plot drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        
        axs[1].fill_between(times, drawdown, 0, color='red', alpha=0.3)
        axs[1].set_title('Drawdown')
        axs[1].set_ylabel('Drawdown')
        axs[1].set_xlabel('Time')
        axs[1].grid(True, alpha=0.3)
        
        # Add metrics as text box
        metrics = self.calculate_performance_metrics()
        
        metrics_text = (
            f"Performance Metrics:\n"
            f"  Total Return: {metrics.get('total_return', 0):.2%}\n"
            f"  Annual Return: {metrics.get('annual_return', 0):.2%}\n"
            f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            f"  Number of Trades: {metrics.get('num_trades', 0)}\n"
            f"  Winning Trades: {metrics.get('num_winning_trades', 0)}\n"
            f"  Losing Trades: {metrics.get('num_losing_trades', 0)}"
        )
        
        plt.figtext(0.02, 0.01, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save or display plot
        if output_file:
            plt.savefig(output_file)
            plt.close()
            return output_file
        else:
            plt.show()
            return None
    
    def plot_pair_performance(self, pair_id, output_file=None):
        """
        Plot performance for a specific pair.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier
        output_file : str, optional
            Path to save the plot to. If None, display the plot.
            
        Returns:
        --------
        str
            Path to the saved plot file
        """
        if not self.prices or pair_id not in self.prices:
            logger.warning(f"No data for pair {pair_id}")
            return None
        
        # Extract trades for this pair
        pair_trades = [t for t in self.trades if t['pair_id'] == pair_id]
        
        if not pair_trades:
            logger.warning(f"No trades for pair {pair_id}")
            return None
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot spread
        if pair_id in self.spreads:
            spread_values = self.spreads[pair_id]['spread']
            spread_times = self.spreads[pair_id].index
            
            axs[0].plot(spread_times, spread_values, label='Spread')
            
            # Plot z-score
            if 'zscore' in self.spreads[pair_id].columns:
                z_values = self.spreads[pair_id]['zscore']
                axs[1].plot(spread_times, z_values, label='Z-Score')
                axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axs[1].axhline(y=2, color='red', linestyle='--', alpha=0.5)
                axs[1].axhline(y=-2, color='green', linestyle='--', alpha=0.5)
        
        # Mark trades on the plot
        for trade in pair_trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                
                if trade['prev_position'] > 0:  # Long position
                    color = 'green'
                else:  # Short position
                    color = 'red'
                
                # Mark position on spread chart
                axs[0].axvspan(entry_time, exit_time, color=color, alpha=0.2)
                axs[0].plot(entry_time, trade['entry_price'], 'o', color=color)
                axs[0].plot(exit_time, trade['exit_price'], 'x', color=color)
                
                # Mark position on z-score chart if available
                if 'zscore' in self.spreads[pair_id].columns:
                    axs[1].axvspan(entry_time, exit_time, color=color, alpha=0.2)
        
        # Set titles and labels
        axs[0].set_title(f'Spread Performance - {pair_id}')
        axs[0].set_ylabel('Spread')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        axs[1].set_title('Z-Score')
        axs[1].set_ylabel('Z-Score')
        axs[1].set_xlabel('Time')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        # Add pair metrics as text box
        winning_trades = [t for t in pair_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in pair_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(pair_trades) if pair_trades else 0
        profit_factor = sum(t.get('pnl', 0) for t in winning_trades) / abs(sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades else float('inf')
        
        metrics_text = (
            f"Pair Metrics - {pair_id}:\n"
            f"  Win Rate: {win_rate:.2%}\n"
            f"  Profit Factor: {profit_factor:.2f}\n"
            f"  Number of Trades: {len(pair_trades)}\n"
            f"  Winning Trades: {len(winning_trades)}\n"
            f"  Losing Trades: {len(losing_trades)}\n"
            f"  Total PnL: {sum(t.get('pnl', 0) for t in pair_trades):.4f}"
        )
        
        plt.figtext(0.02, 0.01, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save or display plot
        if output_file:
            plt.savefig(output_file)
            plt.close()
            return output_file
        else:
            plt.show()
            return None
    
    def save_results(self, results, output_dir="data/results/intraday"):
        """
        Save simulation results to disk.
        
        Parameters:
        -----------
        results : dict
            Simulation results
        output_dir : str
            Directory to save results
            
        Returns:
        --------
        dict
            Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save trades
        trades_file = os.path.join(output_dir, f"trades_{timestamp}.csv")
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(results['trades'])
        
        # Convert datetime columns to string
        for col in trades_df.columns:
            if col.endswith('_time'):
                trades_df[col] = trades_df[col].astype(str)
        
        trades_df.to_csv(trades_file, index=False)
        
        # Save portfolio history
        portfolio_file = os.path.join(output_dir, f"portfolio_{timestamp}.csv")
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(results['portfolio_history'])
        
        # Convert datetime columns to string
        if 'time' in portfolio_df.columns:
            portfolio_df['time'] = portfolio_df['time'].astype(str)
        
        portfolio_df.to_csv(portfolio_file, index=False)
        
        # Plot portfolio performance
        plot_file = os.path.join(output_dir, f"portfolio_plot_{timestamp}.png")
        self.plot_portfolio_performance(output_file=plot_file)
        
        # Plot individual pair performance
        pair_plots = {}
        
        for pair in self.config['pairs']:
            pair_id = pair['pair_id']
            pair_plot_file = os.path.join(output_dir, f"pair_plot_{pair_id}_{timestamp}.png")
            pair_plot = self.plot_pair_performance(pair_id, output_file=pair_plot_file)
            
            if pair_plot:
                pair_plots[pair_id] = pair_plot
        
        return {
            'metrics': metrics_file,
            'trades': trades_file,
            'portfolio': portfolio_file,
            'portfolio_plot': plot_file,
            'pair_plots': pair_plots
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run intraday pairs trading strategy with ML enhancements"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="data/configs/intraday_config_latest.json",
        help="Path to intraday configuration file"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["backtest", "live"], 
        default="backtest",
        help="Mode to run the strategy in (backtest or live)"
    )
    
    parser.add_argument(
        "--start_date", 
        type=str, 
        default="2023-01-01",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end_date", 
        type=str, 
        default="2023-12-31",
        help="End date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--timeframe", 
        type=str, 
        default="5min",
        help="Timeframe to use for backtest"
    )
    
    parser.add_argument(
        "--data_source", 
        type=str, 
        choices=["file", "api"], 
        default="file",
        help="Source of data (file for historical data, api for live data)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/results/intraday",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Find the latest config file if using default
    if args.config == "data/configs/intraday_config_latest.json":
        config_dir = "data/configs"
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json') and f.startswith('intraday_config_')]
        
        if not config_files:
            raise ValueError("No intraday configuration files found")
        
        # Sort by timestamp in filename (most recent first)
        config_files.sort(reverse=True)
        args.config = os.path.join(config_dir, config_files[0])
    
    return args


def main():
    """Main function to run the intraday strategy."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize strategy
    live_mode = args.mode == "live"
    
    try:
        strategy = IntradayStrategy(
            config_file=args.config,
            data_source=args.data_source,
            live_mode=live_mode
        )
    except Exception as e:
        logger.error(f"Failed to initialize strategy: {e}")
        return
    
    # Run strategy
    if args.mode == "backtest":
        logger.info(f"Running backtest from {args.start_date} to {args.end_date}")
        
        results = strategy.run_simulation(
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe
        )
        
        if results:
            # Save results
            saved_files = strategy.save_results(results, output_dir=args.output_dir)
            
            # Display summary
            metrics = results['metrics']
            
            print("\nBacktest Results:")
            print(f"  Period: {args.start_date} to {args.end_date}")
            print(f"  Timeframe: {args.timeframe}")
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"  Number of Trades: {metrics.get('num_trades', 0)}")
            
            print("\nFiles saved:")
            for file_type, file_path in saved_files.items():
                if isinstance(file_path, dict):
                    print(f"  {file_type}:")
                    for key, path in file_path.items():
                        print(f"    {key}: {path}")
                else:
                    print(f"  {file_type}: {file_path}")
        else:
            logger.error("Backtest failed")
    else:
        logger.error("Live mode not yet implemented")


if __name__ == "__main__":
    main() 