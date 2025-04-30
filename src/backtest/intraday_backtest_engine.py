"""
Intraday backtesting engine with realistic constraints and transaction cost modeling.

This module extends the base BacktestEngine with intraday-specific functionality
including time-of-day constraints, realistic transaction costs, and execution modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import logging
from typing import Dict, List, Tuple, Union, Optional

from src.backtest.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

class IntradayBacktestEngine(BacktestEngine):
    """
    Specialized backtesting engine for intraday trading strategies.
    
    This class extends the base BacktestEngine with additional functionality
    specific to intraday trading, including:
    - Realistic time-of-day constraints
    - Sophisticated transaction cost modeling
    - Volume-based execution modeling
    - Liquidity-aware position sizing
    """
    
    def __init__(
        self, 
        signals=None, 
        prices=None, 
        position_sizes=None,
        account_size=100000, 
        trade_delay=0,
        allow_simultaneous_positions=True,
        pairs_data=None,
        intraday_params=None,
        transaction_cost_model=None,
        market_hours=None,
        volume_data=None
    ):
        """
        Initialize the IntradayBacktestEngine.
        
        Parameters:
        -----------
        signals : pandas.DataFrame, optional
            DataFrame containing trading signals
        prices : dict, optional
            Dictionary of price data for each instrument
        position_sizes : pandas.DataFrame, optional
            DataFrame containing position sizes
        account_size : float
            Initial account size
        trade_delay : int
            Delay in bars between signal and execution
        allow_simultaneous_positions : bool
            Whether to allow multiple positions at the same time
        pairs_data : dict, optional
            Dictionary containing pairs trading data
        intraday_params : dict, optional
            Intraday-specific parameters including:
            - max_holding_period: Maximum position holding time in minutes
            - time_filters: Dict with time-of-day trading restrictions
            - exit_buffer_minutes: Minutes before market close to exit positions
        transaction_cost_model : dict, optional
            Configuration for transaction cost modeling including:
            - commission_model: Type of commission model ('ibkr_pro', 'flat', etc.)
            - commission_params: Parameters for commission calculation
            - slippage_model: Type of slippage model ('fixed', 'variable', 'volume_based')
            - slippage_params: Parameters for slippage calculation
        market_hours : dict, optional
            Market session hours configuration
            - start: Session start time (e.g., '09:30')
            - end: Session end time (e.g., '16:00')
        volume_data : dict, optional
            Dictionary of volume data for each instrument for volume-based execution
        """
        # Initialize with 0 commission and slippage - we'll handle this more precisely
        super().__init__(
            signals=signals,
            prices=prices,
            position_sizes=position_sizes,
            account_size=account_size,
            commission=0.0,
            slippage=0.0,
            trade_delay=trade_delay,
            allow_simultaneous_positions=allow_simultaneous_positions,
            pairs_data=pairs_data
        )
        
        # Set default intraday parameters if not provided
        self.intraday_params = intraday_params or {
            "max_holding_period": 180,  # 3 hours in minutes
            "time_filters": {
                "avoid_first_15min": True,
                "avoid_lunch_hour": True,
                "high_liquidity_windows": [
                    {"start": "09:45", "end": "11:30"},
                    {"start": "13:30", "end": "15:45"}
                ]
            },
            "exit_buffer_minutes": 15  # Exit 15 minutes before market close
        }
        
        # Set default transaction cost model if not provided
        self.transaction_cost_model = transaction_cost_model or {
            "commission_model": "ibkr_pro",
            "commission_params": {
                "per_contract": 0.85,   # IBKR futures commission
                "per_share": 0.005,     # IBKR stock commission
                "minimum": 1.0          # Minimum commission
            },
            "slippage_model": "volume_based",
            "slippage_params": {
                "base_points": 1.0,      # Base slippage in price points
                "volume_factor": 0.5,    # Volume impact on slippage
                "volatility_factor": 0.3  # Volatility impact on slippage
            }
        }
        
        # Set default market hours if not provided
        self.market_hours = market_hours or {
            "start": "09:30",
            "end": "16:00"
        }
        
        # Store volume data for volume-based execution and slippage
        self.volume_data = volume_data or {}
        
        # Initialize additional tracking metrics
        self.intraday_metrics = {
            "missed_trades": [],
            "time_violations": [],
            "forced_exits": [],
            "transaction_costs": {
                "commission": [],
                "slippage": []
            }
        }
    
    def run_backtest(self, signals=None, prices=None, position_sizes=None, pairs_data=None,
                    volume_data=None, intraday_params=None):
        """
        Run backtest with intraday-specific constraints and costs.
        
        Parameters:
        -----------
        signals : pandas.DataFrame, optional
            DataFrame containing trading signals
        prices : dict, optional
            Dictionary of price data for each instrument
        position_sizes : pandas.DataFrame, optional
            DataFrame containing position sizes
        pairs_data : dict, optional
            Dictionary containing pairs trading data
        volume_data : dict, optional
            Dictionary of volume data for each instrument
        intraday_params : dict, optional
            Intraday-specific parameters to override defaults
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Update parameters if provided
        self.volume_data = volume_data or self.volume_data
        if intraday_params:
            self.intraday_params.update(intraday_params)
        
        # Reset intraday metrics
        self.intraday_metrics = {
            "missed_trades": [],
            "time_violations": [],
            "forced_exits": [],
            "transaction_costs": {
                "commission": [],
                "slippage": []
            }
        }
        
        # Apply intraday constraints to signals
        constrained_signals = self._apply_intraday_constraints(
            signals if signals is not None else self.signals
        )
        
        # Run the backtest with constrained signals
        results = super().run_backtest(
            signals=constrained_signals,
            prices=prices,
            position_sizes=position_sizes,
            pairs_data=pairs_data
        )
        
        # Add intraday-specific metrics to results
        results['intraday_metrics'] = self.intraday_metrics
        
        return results
    
    def _apply_intraday_constraints(self, signals):
        """
        Apply intraday-specific constraints to trading signals.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with trading signals
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with constrained signals
        """
        if signals is None or len(signals) == 0:
            return signals
        
        constrained_signals = signals.copy()
        
        # Check if timestamp index is available
        if not isinstance(signals.index, pd.DatetimeIndex):
            logger.warning("Cannot apply intraday constraints - signals do not have DatetimeIndex")
            return constrained_signals
        
        # Apply maximum holding period
        constrained_signals = self._apply_max_holding_period(constrained_signals)
        
        # Apply time-of-day filters
        constrained_signals = self._apply_time_filters(constrained_signals)
        
        # Force close positions before market close
        constrained_signals = self._apply_market_close_constraint(constrained_signals)
        
        return constrained_signals
    
    def _apply_max_holding_period(self, signals):
        """
        Close positions that exceed maximum holding period.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with trading signals
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with constrained signals
        """
        constrained_signals = signals.copy()
        max_holding_period = self.intraday_params.get("max_holding_period", 180)  # in minutes
        
        # Track entry timestamps and current positions
        entry_times = {}
        current_positions = {}
        
        # For each pair or instrument in the signals DataFrame
        for column in constrained_signals.columns:
            position = 0
            entry_time = None
            
            for i in range(len(constrained_signals)):
                timestamp = constrained_signals.index[i]
                current_signal = constrained_signals.loc[timestamp, column]
                
                # Check for new position
                if position == 0 and current_signal != 0:
                    position = current_signal
                    entry_time = timestamp
                # Check for closed position
                elif position != 0 and current_signal == 0:
                    position = 0
                    entry_time = None
                
                # Close position if holding period exceeded
                if position != 0 and entry_time is not None:
                    holding_time = timestamp - entry_time
                    if holding_time.total_seconds() / 60 > max_holding_period:
                        constrained_signals.loc[timestamp, column] = 0
                        position = 0
                        entry_time = None
                        self.intraday_metrics["forced_exits"].append({
                            "timestamp": timestamp,
                            "instrument": column,
                            "reason": "max_holding_period",
                            "holding_minutes": holding_time.total_seconds() / 60
                        })
        
        return constrained_signals
    
    def _apply_time_filters(self, signals):
        """
        Apply time-of-day filters to avoid trading during specified periods.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with trading signals
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with constrained signals
        """
        constrained_signals = signals.copy()
        time_filters = self.intraday_params.get("time_filters", {})
        
        # a. Avoid first 15 minutes after market open
        if time_filters.get("avoid_first_15min", False):
            start_str = self.market_hours.get("start", "09:30")
            start_hour, start_minute = map(int, start_str.split(":"))
            market_open = time(start_hour, start_minute)
            
            # Calculate market open + 15 minutes
            market_open_plus_15 = (
                datetime.combine(datetime.today(), market_open) + timedelta(minutes=15)
            ).time()
            
            morning_mask = (
                (signals.index.time >= market_open) & 
                (signals.index.time < market_open_plus_15)
            )
            
            # Don't enter new positions during first 15 minutes
            for i in range(1, len(signals)):
                if morning_mask[i]:
                    for column in constrained_signals.columns:
                        # Check if this would be a new position
                        prev_signal = constrained_signals.iloc[i-1][column]
                        curr_signal = constrained_signals.iloc[i][column]
                        
                        if prev_signal == 0 and curr_signal != 0:
                            constrained_signals.iloc[i][column] = 0
                            self.intraday_metrics["time_violations"].append({
                                "timestamp": signals.index[i],
                                "instrument": column,
                                "reason": "avoid_first_15min"
                            })
        
        # b. Avoid lunch hour
        if time_filters.get("avoid_lunch_hour", False):
            lunch_start = time(12, 0)
            lunch_end = time(13, 0)
            
            lunch_mask = (
                (signals.index.time >= lunch_start) & 
                (signals.index.time < lunch_end)
            )
            
            # Don't enter new positions during lunch hour
            for i in range(1, len(signals)):
                if lunch_mask[i]:
                    for column in constrained_signals.columns:
                        # Check if this would be a new position
                        prev_signal = constrained_signals.iloc[i-1][column]
                        curr_signal = constrained_signals.iloc[i][column]
                        
                        if prev_signal == 0 and curr_signal != 0:
                            constrained_signals.iloc[i][column] = 0
                            self.intraday_metrics["time_violations"].append({
                                "timestamp": signals.index[i],
                                "instrument": column,
                                "reason": "avoid_lunch_hour"
                            })
        
        # c. Only trade during high liquidity windows if specified
        high_liquidity_windows = time_filters.get("high_liquidity_windows", [])
        
        if high_liquidity_windows:
            # Initialize mask for high liquidity periods
            in_liquidity_window = pd.Series(False, index=signals.index)
            
            for window in high_liquidity_windows:
                start_str = window.get("start", "09:30")
                end_str = window.get("end", "16:00")
                
                start_hour, start_minute = map(int, start_str.split(":"))
                end_hour, end_minute = map(int, end_str.split(":"))
                
                start_time = time(start_hour, start_minute)
                end_time = time(end_hour, end_minute)
                
                # Update mask for this window
                window_mask = (signals.index.time >= start_time) & (signals.index.time < end_time)
                in_liquidity_window = in_liquidity_window | window_mask
            
            # Only allow entries during high liquidity windows
            for i in range(1, len(signals)):
                if not in_liquidity_window.iloc[i]:
                    for column in constrained_signals.columns:
                        # Check if this would be a new position
                        prev_signal = constrained_signals.iloc[i-1][column]
                        curr_signal = constrained_signals.iloc[i][column]
                        
                        if prev_signal == 0 and curr_signal != 0:
                            constrained_signals.iloc[i][column] = 0
                            self.intraday_metrics["time_violations"].append({
                                "timestamp": signals.index[i],
                                "instrument": column,
                                "reason": "outside_liquidity_window"
                            })
        
        return constrained_signals
    
    def _apply_market_close_constraint(self, signals):
        """
        Force close all positions before market close.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with trading signals
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with constrained signals
        """
        constrained_signals = signals.copy()
        exit_buffer = self.intraday_params.get("exit_buffer_minutes", 15)
        
        # Parse market close time
        close_str = self.market_hours.get("end", "16:00")
        close_hour, close_minute = map(int, close_str.split(":"))
        market_close = time(close_hour, close_minute)
        
        # Calculate force exit time (market close - buffer)
        force_exit_time = (
            datetime.combine(datetime.today(), market_close) - timedelta(minutes=exit_buffer)
        ).time()
        
        # Force close all positions at or after force exit time
        close_mask = (signals.index.time >= force_exit_time)
        
        # For each time period in the close mask
        for i in range(len(signals)):
            if close_mask[i]:
                for column in constrained_signals.columns:
                    if constrained_signals.iloc[i][column] != 0:
                        constrained_signals.iloc[i][column] = 0
                        self.intraday_metrics["forced_exits"].append({
                            "timestamp": signals.index[i],
                            "instrument": column,
                            "reason": "market_close"
                        })
        
        return constrained_signals
    
    def _process_trade(self, timestamp, signals, prices, positions):
        """
        Override parent method to add realistic transaction costs.
        
        Parameters:
        -----------
        timestamp : datetime
            Current timestamp
        signals : pd.Series
            Current signals
        prices : dict
            Current prices
        positions : dict
            Current positions
            
        Returns:
        --------
        dict
            Trade information
        """
        # Call parent method to get base trade info
        trade = super()._process_trade(timestamp, signals, prices, positions)
        
        # Add realistic transaction costs if a trade was executed
        if trade and trade.get('type') != 'hold':
            # Calculate transaction costs
            transaction_costs = self._calculate_transaction_costs(
                timestamp, trade, prices
            )
            
            # Apply costs to trade PnL
            trade['commission'] = transaction_costs['commission']
            trade['slippage'] = transaction_costs['slippage']
            trade['total_cost'] = transaction_costs['total']
            trade['pnl'] -= transaction_costs['total']
            
            # Track costs
            self.intraday_metrics["transaction_costs"]["commission"].append(transaction_costs['commission'])
            self.intraday_metrics["transaction_costs"]["slippage"].append(transaction_costs['slippage'])
        
        return trade
    
    def _calculate_transaction_costs(self, timestamp, trade, prices):
        """
        Calculate realistic transaction costs including commission and slippage.
        
        Parameters:
        -----------
        timestamp : datetime
            Current timestamp
        trade : dict
            Trade information
        prices : dict
            Current prices
            
        Returns:
        --------
        dict
            Dictionary with commission, slippage, and total costs
        """
        # Extract instruments and quantities
        symbol1 = trade.get('symbol1')
        symbol2 = trade.get('symbol2')
        qty1 = abs(trade.get('qty1', 0))
        qty2 = abs(trade.get('qty2', 0))
        
        # Get current prices
        price1 = prices.get(symbol1, {}).get('close', 0) if symbol1 in prices else 0
        price2 = prices.get(symbol2, {}).get('close', 0) if symbol2 in prices else 0
        
        # Calculate trade value
        trade_value = (qty1 * price1) + (qty2 * price2)
        
        # 1. Calculate commission
        commission = self._calculate_commission(symbol1, price1, qty1)
        commission += self._calculate_commission(symbol2, price2, qty2)
        
        # 2. Calculate slippage
        slippage1 = self._calculate_slippage(timestamp, symbol1, price1, qty1)
        slippage2 = self._calculate_slippage(timestamp, symbol2, price2, qty2)
        slippage = slippage1 + slippage2
        
        return {
            'commission': commission,
            'slippage': slippage,
            'total': commission + slippage
        }
    
    def _calculate_commission(self, symbol, price, quantity):
        """
        Calculate commission for a trade based on the selected model.
        
        Parameters:
        -----------
        symbol : str
            Instrument symbol
        price : float
            Trade price
        quantity : float
            Trade quantity
            
        Returns:
        --------
        float
            Commission amount
        """
        if quantity == 0 or price == 0:
            return 0.0
            
        commission_model = self.transaction_cost_model.get("commission_model", "ibkr_pro")
        params = self.transaction_cost_model.get("commission_params", {})
        
        # Determine if this is a futures contract or stock
        is_futures = any(fut_symbol in symbol for fut_symbol in ['ES', 'NQ', 'CL', 'GC', 'ZB', 'ZN', 'ZF', 'ZT', 'MES', 'MNQ'])
        
        if commission_model == "none":
            return 0.0
        
        elif commission_model == "ibkr_pro":
            if is_futures:
                # IBKR Pro futures commission
                per_contract = params.get("per_contract", 0.85)
                return quantity * per_contract
            else:
                # IBKR Pro stock commission: min($1.00, 0.005 * shares)
                per_share = params.get("per_share", 0.005)
                minimum = params.get("minimum", 1.0)
                return max(minimum, quantity * per_share)
        
        elif commission_model == "flat":
            # Flat fee per trade
            flat_fee = params.get("flat_fee", 1.0)
            return flat_fee
        
        elif commission_model == "percentage":
            # Percentage of trade value
            percentage = params.get("percentage", 0.001)  # 0.1% default
            return price * quantity * percentage
            
        else:
            # Default to zero if unknown model
            logger.warning(f"Unknown commission model: {commission_model}. Using zero commission.")
            return 0.0
    
    def _calculate_slippage(self, timestamp, symbol, price, quantity):
        """
        Calculate slippage for a trade based on the selected model.
        
        Parameters:
        -----------
        timestamp : datetime
            Trade timestamp
        symbol : str
            Instrument symbol
        price : float
            Trade price
        quantity : float
            Trade quantity
            
        Returns:
        --------
        float
            Slippage amount
        """
        if quantity == 0 or price == 0:
            return 0.0
            
        slippage_model = self.transaction_cost_model.get("slippage_model", "fixed")
        params = self.transaction_cost_model.get("slippage_params", {})
        
        # Get volume data if available
        volume = None
        if symbol in self.volume_data and timestamp in self.volume_data[symbol].index:
            volume = self.volume_data[symbol].loc[timestamp, 'volume']
        
        if slippage_model == "none":
            return 0.0
            
        elif slippage_model == "fixed":
            # Fixed percentage of price
            fixed_percentage = params.get("fixed_percentage", 0.0001)  # 0.01% default
            return price * quantity * fixed_percentage
            
        elif slippage_model == "variable":
            # Variable slippage based on random factor
            base_percentage = params.get("base_percentage", 0.0001)
            max_factor = params.get("max_factor", 3.0)
            random_factor = np.random.random() * max_factor
            return price * quantity * base_percentage * random_factor
            
        elif slippage_model == "volume_based":
            # Volume-based slippage model
            base_points = params.get("base_points", 1.0)
            volume_factor = params.get("volume_factor", 0.5)
            volatility_factor = params.get("volatility_factor", 0.3)
            
            # Default slippage if no volume data
            if volume is None or volume == 0:
                return price * quantity * 0.0001  # Default to 0.01%
            
            # Calculate relative volume (normalized trade size)
            relative_volume = min(1.0, quantity / volume)
            
            # Calculate relative volatility if volatility data available
            relative_volatility = 1.0
            if hasattr(self, 'volatility_data') and symbol in self.volatility_data:
                if timestamp in self.volatility_data[symbol].index:
                    vol = self.volatility_data[symbol].loc[timestamp]
                    avg_vol = self.volatility_data[symbol].mean()
                    relative_volatility = vol / avg_vol if avg_vol > 0 else 1.0
            
            # Calculate slippage in basis points (1bp = 0.01%)
            slippage_bps = base_points * (1 + volume_factor * relative_volume + volatility_factor * relative_volatility)
            
            # Convert basis points to percentage (100bp = 1%)
            slippage_percentage = slippage_bps / 10000
            
            return price * quantity * slippage_percentage
            
        else:
            # Default to fixed slippage if unknown model
            logger.warning(f"Unknown slippage model: {slippage_model}. Using fixed slippage.")
            return price * quantity * 0.0001  # Default to 0.01%
    
    def calculate_detailed_metrics(self):
        """
        Calculate detailed intraday performance metrics.
        
        Returns:
        --------
        dict
            Dictionary with detailed metrics
        """
        # Call parent method to get base metrics
        base_metrics = super().calculate_metrics()
        
        # Calculate additional intraday-specific metrics
        
        # 1. Calculate transaction cost impact
        total_commission = sum(self.intraday_metrics["transaction_costs"]["commission"])
        total_slippage = sum(self.intraday_metrics["transaction_costs"]["slippage"])
        total_costs = total_commission + total_slippage
        
        # 2. Calculate time-of-day performance
        hourly_returns = {}
        if self.equity_curve is not None and isinstance(self.equity_curve.index, pd.DatetimeIndex):
            # Calculate returns by hour
            returns = self.equity_curve['equity'].pct_change().dropna()
            for hour in range(9, 17):  # Market hours typically 9:30 AM to 4:00 PM
                hour_returns = returns[returns.index.hour == hour]
                if not hour_returns.empty:
                    hourly_returns[hour] = {
                        'mean': hour_returns.mean(),
                        'std': hour_returns.std(),
                        'sharpe': hour_returns.mean() / hour_returns.std() if hour_returns.std() > 0 else 0,
                        'total': hour_returns.sum(),
                        'count': len(hour_returns)
                    }
        
        # 3. Calculate statistics on missed trades and forced exits
        missed_trades_count = len(self.intraday_metrics["missed_trades"])
        time_violations_count = len(self.intraday_metrics["time_violations"])
        forced_exits_count = len(self.intraday_metrics["forced_exits"])
        
        # Compile detailed metrics
        detailed_metrics = {
            **base_metrics,
            'transaction_costs': {
                'total': total_costs,
                'commission': total_commission,
                'slippage': total_slippage,
                'pct_of_gross_profit': total_costs / base_metrics['gross_profit'] if base_metrics.get('gross_profit', 0) > 0 else 0
            },
            'time_of_day': {
                'hourly_returns': hourly_returns
            },
            'constraints': {
                'missed_trades': missed_trades_count,
                'time_violations': time_violations_count,
                'forced_exits': forced_exits_count,
                'forced_exits_by_reason': self._count_by_reason(self.intraday_metrics["forced_exits"])
            }
        }
        
        return detailed_metrics
    
    def _count_by_reason(self, events):
        """
        Count events by reason.
        
        Parameters:
        -----------
        events : list
            List of event dictionaries with 'reason' key
            
        Returns:
        --------
        dict
            Count of events by reason
        """
        counts = {}
        for event in events:
            reason = event.get('reason', 'unknown')
            counts[reason] = counts.get(reason, 0) + 1
        return counts 