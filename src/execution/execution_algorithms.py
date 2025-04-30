"""
Execution Algorithms

This module provides execution algorithms for trading operations, including:
1. VWAP (Volume-Weighted Average Price) strategy
2. TWAP (Time-Weighted Average Price) strategy
3. Adaptive execution strategies based on market conditions
4. Limit order placement strategies

These algorithms help minimize market impact and optimize execution performance.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import datetime as dt


class ExecutionType(Enum):
    """Enumeration of execution algorithm types."""
    
    MARKET = "market"  # Immediate execution at market price
    LIMIT = "limit"  # Limit order at specified price
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ADAPTIVE = "adaptive"  # Adaptive execution based on market conditions
    ICEBERG = "iceberg"  # Iceberg/reserve orders
    CUSTOM = "custom"  # Custom execution strategy


@dataclass
class ExecutionConfig:
    """Base configuration for execution algorithms."""
    
    start_time: Optional[dt.datetime] = None  # Start time for execution
    end_time: Optional[dt.datetime] = None  # End time for execution
    max_participation: float = 0.3  # Maximum market participation rate
    urgency: float = 0.5  # Execution urgency (0-1)
    allow_partial: bool = True  # Whether to allow partial fills


@dataclass
class TWAPConfig(ExecutionConfig):
    """Configuration for TWAP execution."""
    
    num_slices: int = 10  # Number of equal time slices
    random_variance: float = 0.0  # Random variance in slice sizes (0-1)


@dataclass
class VWAPConfig(ExecutionConfig):
    """Configuration for VWAP execution."""
    
    num_slices: int = 10  # Number of volume slices
    historical_volume_window: int = 5  # Days of historical volume data to use
    volume_profile_percentile: int = 50  # Percentile of volume profile to target


@dataclass
class AdaptiveConfig(ExecutionConfig):
    """Configuration for adaptive execution."""
    
    min_participation: float = 0.05  # Minimum participation rate
    volatility_factor: float = 1.0  # Adjustment for volatility
    price_improvement_threshold: float = 0.0001  # Required price improvement
    spread_factor: float = 0.5  # Spread factor for limit price calculation


class ExecutionAlgorithm:
    """Base class for execution algorithms."""
    
    def __init__(self, execution_type: ExecutionType, config: ExecutionConfig):
        """
        Initialize the execution algorithm.
        
        Args:
            execution_type: Type of execution algorithm
            config: Configuration for the algorithm
        """
        self.execution_type = execution_type
        self.config = config
        
    def generate_execution_plan(self, quantity: float, 
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an execution plan for the given order.
        
        Args:
            quantity: Total quantity to execute
            market_data: Market data for execution planning
            
        Returns:
            DataFrame with execution plan details
        """
        raise NotImplementedError("Subclasses must implement generate_execution_plan method")
        
    def _validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """
        Validate market data for execution planning.
        
        Args:
            market_data: Market data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['timestamp', 'price']
        return all(col in market_data.columns for col in required_columns)


class MarketExecutionAlgorithm(ExecutionAlgorithm):
    """Simple market execution algorithm."""
    
    def __init__(self, config: ExecutionConfig = None):
        """
        Initialize the market execution algorithm.
        
        Args:
            config: Optional configuration
        """
        if config is None:
            config = ExecutionConfig()
        super().__init__(ExecutionType.MARKET, config)
        
    def generate_execution_plan(self, quantity: float, 
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a simple market execution plan.
        
        Args:
            quantity: Total quantity to execute
            market_data: Market data for execution planning
            
        Returns:
            DataFrame with single-slice execution plan
        """
        if not self._validate_market_data(market_data):
            raise ValueError("Invalid market data for execution planning")
            
        current_time = market_data['timestamp'].iloc[-1]
        current_price = market_data['price'].iloc[-1]
        
        # Create a simple single-slice execution plan
        execution_plan = pd.DataFrame({
            'timestamp': [current_time],
            'quantity': [quantity],
            'price': [current_price],
            'type': ['market'],
            'participation_rate': [1.0]
        })
        
        return execution_plan


class TWAPExecutionAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price (TWAP) execution algorithm."""
    
    def __init__(self, config: TWAPConfig = None):
        """
        Initialize the TWAP execution algorithm.
        
        Args:
            config: TWAP configuration
        """
        if config is None:
            config = TWAPConfig()
        super().__init__(ExecutionType.TWAP, config)
        
    def generate_execution_plan(self, quantity: float, 
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a TWAP execution plan.
        
        Args:
            quantity: Total quantity to execute
            market_data: Market data for execution planning
            
        Returns:
            DataFrame with time-sliced execution plan
        """
        if not self._validate_market_data(market_data):
            raise ValueError("Invalid market data for execution planning")
            
        config = self.config
        
        # Determine execution time window
        if config.start_time is None:
            start_time = market_data['timestamp'].iloc[-1]
        else:
            start_time = config.start_time
            
        if config.end_time is None:
            # Default to 1 hour execution window if not specified
            end_time = start_time + dt.timedelta(hours=1)
        else:
            end_time = config.end_time
            
        # Calculate time slices
        time_delta = (end_time - start_time) / config.num_slices
        
        # Calculate base quantity per slice
        base_quantity_per_slice = quantity / config.num_slices
        
        # Create execution plan
        execution_plan = []
        
        for i in range(config.num_slices):
            # Add randomization if specified
            if config.random_variance > 0:
                random_factor = 1.0 + np.random.uniform(
                    -config.random_variance, config.random_variance)
                slice_quantity = base_quantity_per_slice * random_factor
            else:
                slice_quantity = base_quantity_per_slice
                
            slice_time = start_time + time_delta * i
            
            # Find closest price to the slice time
            closest_price_idx = (market_data['timestamp'] - slice_time).abs().idxmin()
            slice_price = market_data.loc[closest_price_idx, 'price']
            
            execution_plan.append({
                'timestamp': slice_time,
                'quantity': slice_quantity,
                'price': slice_price,
                'type': 'limit',
                'participation_rate': config.max_participation
            })
            
        # Adjust final slice to ensure total quantity matches
        total_planned = sum(slice['quantity'] for slice in execution_plan)
        execution_plan[-1]['quantity'] += (quantity - total_planned)
        
        return pd.DataFrame(execution_plan)


class VWAPExecutionAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price (VWAP) execution algorithm."""
    
    def __init__(self, config: VWAPConfig = None):
        """
        Initialize the VWAP execution algorithm.
        
        Args:
            config: VWAP configuration
        """
        if config is None:
            config = VWAPConfig()
        super().__init__(ExecutionType.VWAP, config)
        
    def _estimate_volume_profile(self, market_data: pd.DataFrame) -> Dict[int, float]:
        """
        Estimate volume profile based on historical data.
        
        Args:
            market_data: Historical market data with volume
            
        Returns:
            Dictionary mapping hour of day to percentage of daily volume
        """
        if 'volume' not in market_data.columns:
            raise ValueError("Market data must contain volume for VWAP execution")
            
        # Group by hour of day and calculate average volume
        market_data['hour'] = market_data['timestamp'].dt.hour
        
        # Calculate average volume by hour
        volume_by_hour = market_data.groupby('hour')['volume'].mean()
        
        # Calculate percentage of daily volume for each hour
        total_volume = volume_by_hour.sum()
        volume_profile = {hour: vol/total_volume for hour, vol in volume_by_hour.items()}
        
        return volume_profile
        
    def generate_execution_plan(self, quantity: float, 
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a VWAP execution plan.
        
        Args:
            quantity: Total quantity to execute
            market_data: Market data for execution planning
            
        Returns:
            DataFrame with volume-weighted execution plan
        """
        if not self._validate_market_data(market_data) or 'volume' not in market_data.columns:
            raise ValueError("Invalid market data for VWAP execution planning")
            
        config = self.config
        
        # Determine execution time window
        if config.start_time is None:
            start_time = market_data['timestamp'].iloc[-1]
        else:
            start_time = config.start_time
            
        if config.end_time is None:
            # Default to 1 day execution window if not specified
            end_time = start_time + dt.timedelta(days=1)
        else:
            end_time = config.end_time
            
        # Estimate volume profile
        volume_profile = self._estimate_volume_profile(market_data)
        
        # Calculate execution plan based on volume profile
        trading_hours = sorted([hour for hour in volume_profile.keys() 
                               if start_time.replace(hour=hour) < end_time])
        
        # Create execution plan
        execution_plan = []
        
        for hour in trading_hours:
            # Calculate quantity for this hour based on volume profile
            hour_volume_percentage = volume_profile[hour]
            hour_quantity = quantity * hour_volume_percentage
            
            # Calculate slice time
            slice_time = start_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            if slice_time < start_time:
                slice_time += dt.timedelta(days=1)
                
            if slice_time >= end_time:
                continue
                
            # Find closest price to the slice time
            closest_price_idx = (market_data['timestamp'] - slice_time).abs().idxmin()
            slice_price = market_data.loc[closest_price_idx, 'price']
            
            # Calculate participation rate based on historical volume
            hour_market_data = market_data[market_data['timestamp'].dt.hour == hour]
            avg_hour_volume = hour_market_data['volume'].mean()
            participation_rate = min(hour_quantity / avg_hour_volume, config.max_participation)
            
            execution_plan.append({
                'timestamp': slice_time,
                'quantity': hour_quantity,
                'price': slice_price,
                'type': 'limit',
                'participation_rate': participation_rate
            })
            
        # Adjust final slice to ensure total quantity matches
        total_planned = sum(slice['quantity'] for slice in execution_plan)
        if execution_plan:
            execution_plan[-1]['quantity'] += (quantity - total_planned)
        
        return pd.DataFrame(execution_plan)


class AdaptiveExecutionAlgorithm(ExecutionAlgorithm):
    """Adaptive execution algorithm based on market conditions."""
    
    def __init__(self, config: AdaptiveConfig = None):
        """
        Initialize the adaptive execution algorithm.
        
        Args:
            config: Adaptive execution configuration
        """
        if config is None:
            config = AdaptiveConfig()
        super().__init__(ExecutionType.ADAPTIVE, config)
        
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """
        Calculate recent market volatility.
        
        Args:
            market_data: Recent market data
            
        Returns:
            Volatility estimate
        """
        if 'price' in market_data.columns:
            returns = market_data['price'].pct_change().dropna()
        elif 'close' in market_data.columns:
            returns = market_data['close'].pct_change().dropna()
        else:
            raise ValueError("Market data must contain price or close column")
            
        return returns.std() * np.sqrt(252)  # Annualized volatility
        
    def _calculate_spread(self, market_data: pd.DataFrame) -> float:
        """
        Calculate average bid-ask spread.
        
        Args:
            market_data: Market data with bid and ask prices
            
        Returns:
            Average spread in percentage terms
        """
        if 'bid' in market_data.columns and 'ask' in market_data.columns:
            spread = (market_data['ask'] - market_data['bid']) / market_data['mid']
            return spread.mean()
        else:
            # Default to estimated spread if bid/ask not available
            return 0.0001  # 1 basis point default
        
    def generate_execution_plan(self, quantity: float, 
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an adaptive execution plan based on market conditions.
        
        Args:
            quantity: Total quantity to execute
            market_data: Market data for execution planning
            
        Returns:
            DataFrame with adaptive execution plan
        """
        if not self._validate_market_data(market_data):
            raise ValueError("Invalid market data for execution planning")
            
        config = self.config
        
        # Determine execution time window
        if config.start_time is None:
            start_time = market_data['timestamp'].iloc[-1]
        else:
            start_time = config.start_time
            
        if config.end_time is None:
            # Default to 1 day execution window if not specified
            end_time = start_time + dt.timedelta(days=1)
        else:
            end_time = config.end_time
            
        # Calculate volatility
        volatility = self._calculate_volatility(market_data)
        
        # Calculate spread
        spread = self._calculate_spread(market_data)
        
        # Determine number of slices based on volatility
        # Higher volatility -> more slices
        volatility_factor = max(0.5, min(2.0, config.volatility_factor * volatility / 0.2))
        base_slices = 10  # Default number of slices
        num_slices = int(base_slices * volatility_factor)
        
        # Determine time between slices
        time_delta = (end_time - start_time) / num_slices
        
        # Determine participation rate based on volatility and urgency
        # Lower volatility and higher urgency -> higher participation
        base_participation = config.min_participation + (config.max_participation - config.min_participation) * config.urgency
        adjusted_participation = base_participation / volatility_factor
        
        # Create execution plan
        execution_plan = []
        
        slice_quantity = quantity / num_slices
        
        for i in range(num_slices):
            slice_time = start_time + time_delta * i
            
            # Find closest data to the slice time
            closest_idx = (market_data['timestamp'] - slice_time).abs().idxmin()
            slice_data = market_data.loc[closest_idx]
            
            # Determine order type based on spread and market conditions
            if spread > config.price_improvement_threshold:
                # Use limit orders when spread is wide enough
                limit_price = slice_data['price'] * (1 - spread * config.spread_factor)
                order_type = 'limit'
            else:
                # Use market orders when spread is tight
                limit_price = slice_data['price']
                order_type = 'market'
                
            execution_plan.append({
                'timestamp': slice_time,
                'quantity': slice_quantity,
                'price': limit_price,
                'type': order_type,
                'participation_rate': adjusted_participation
            })
            
        # Adjust final slice to ensure total quantity matches
        total_planned = sum(slice['quantity'] for slice in execution_plan)
        execution_plan[-1]['quantity'] += (quantity - total_planned)
        
        return pd.DataFrame(execution_plan)


class IcebergExecutionAlgorithm(ExecutionAlgorithm):
    """Iceberg/reserve order execution algorithm."""
    
    def __init__(self, visible_quantity: float, config: ExecutionConfig = None):
        """
        Initialize the iceberg execution algorithm.
        
        Args:
            visible_quantity: Maximum visible quantity at any time
            config: Optional execution configuration
        """
        if config is None:
            config = ExecutionConfig()
        super().__init__(ExecutionType.ICEBERG, config)
        self.visible_quantity = visible_quantity
        
    def generate_execution_plan(self, quantity: float, 
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an iceberg execution plan.
        
        Args:
            quantity: Total quantity to execute
            market_data: Market data for execution planning
            
        Returns:
            DataFrame with iceberg execution plan
        """
        if not self._validate_market_data(market_data):
            raise ValueError("Invalid market data for execution planning")
            
        config = self.config
        
        # Determine execution time window
        if config.start_time is None:
            start_time = market_data['timestamp'].iloc[-1]
        else:
            start_time = config.start_time
            
        if config.end_time is None:
            # Default to 30 minutes execution window if not specified
            end_time = start_time + dt.timedelta(minutes=30)
        else:
            end_time = config.end_time
            
        # Calculate number of iceberg replenishments needed
        num_replenishments = int(np.ceil(quantity / self.visible_quantity))
        
        # Calculate time between replenishments
        time_delta = (end_time - start_time) / num_replenishments
        
        # Create execution plan
        execution_plan = []
        
        remaining_quantity = quantity
        
        for i in range(num_replenishments):
            slice_time = start_time + time_delta * i
            
            # Determine quantity for this slice
            slice_quantity = min(self.visible_quantity, remaining_quantity)
            remaining_quantity -= slice_quantity
            
            # Find closest price to the slice time
            closest_price_idx = (market_data['timestamp'] - slice_time).abs().idxmin()
            slice_price = market_data.loc[closest_price_idx, 'price']
            
            execution_plan.append({
                'timestamp': slice_time,
                'quantity': slice_quantity,
                'price': slice_price,
                'type': 'limit',
                'participation_rate': config.max_participation,
                'display_quantity': self.visible_quantity
            })
            
        return pd.DataFrame(execution_plan)


class ExecutionSimulator:
    """
    Simulates execution of trading plans for backtesting and performance analysis.
    """
    
    def __init__(self, 
                 market_data: pd.DataFrame,
                 spread_model: Optional[Callable[[pd.Series], float]] = None,
                 volatility_model: Optional[Callable[[pd.DataFrame], float]] = None,
                 latency_model: Optional[Callable[[], float]] = None):
        """
        Initialize the execution simulator.
        
        Args:
            market_data: Historical market data for simulation
            spread_model: Optional function to model bid-ask spread
            volatility_model: Optional function to model market volatility
            latency_model: Optional function to model execution latency
        """
        self.market_data = market_data
        self.spread_model = spread_model
        self.volatility_model = volatility_model
        self.latency_model = latency_model
        
    def _default_spread_model(self, data: pd.Series) -> float:
        """
        Default model for bid-ask spread.
        
        Args:
            data: Market data for a specific timestamp
            
        Returns:
            Estimated spread in price units
        """
        price = data['price']
        return price * 0.0001  # Default 1 basis point spread
        
    def _default_volatility_model(self, data: pd.DataFrame) -> float:
        """
        Default model for market volatility.
        
        Args:
            data: Recent market data
            
        Returns:
            Estimated volatility
        """
        if len(data) < 2:
            return 0.001  # Default low volatility if insufficient data
            
        returns = data['price'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized volatility
        
    def _default_latency_model(self) -> float:
        """
        Default model for execution latency.
        
        Returns:
            Simulated latency in seconds
        """
        return np.random.exponential(0.1)  # Average 100ms latency
        
    def _get_execution_price(self, 
                            target_time: dt.datetime,
                            target_price: float,
                            order_type: str,
                            participation_rate: float) -> Tuple[float, dt.datetime]:
        """
        Calculate actual execution price based on market conditions.
        
        Args:
            target_time: Target execution time
            target_price: Target execution price
            order_type: Order type (market, limit)
            participation_rate: Market participation rate
            
        Returns:
            Tuple of (execution_price, execution_time)
        """
        # Find nearest market data to target time
        nearest_idx = (self.market_data['timestamp'] - target_time).abs().idxmin()
        execution_data = self.market_data.loc[nearest_idx:nearest_idx+5]
        
        # Calculate spread
        spread_model = self.spread_model or self._default_spread_model
        spread = spread_model(execution_data.iloc[0])
        
        # Calculate volatility
        volatility_model = self.volatility_model or self._default_volatility_model
        volatility = volatility_model(execution_data)
        
        # Calculate latency
        latency_model = self.latency_model or self._default_latency_model
        latency = latency_model()
        
        # Adjust execution time for latency
        execution_time = target_time + dt.timedelta(seconds=latency)
        
        # Find market price at execution time
        exec_nearest_idx = (self.market_data['timestamp'] - execution_time).abs().idxmin()
        market_price = self.market_data.loc[exec_nearest_idx, 'price']
        
        # Calculate execution price based on order type
        if order_type == 'market':
            # Market orders - add half spread plus impact based on participation
            market_impact = market_price * 0.0001 * participation_rate * 10  # Simple impact model
            execution_price = market_price + (spread / 2) + market_impact
        elif order_type == 'limit':
            # Limit orders - execute at target price if market price crosses it
            if target_price >= market_price:
                execution_price = target_price
            else:
                # Limit price not reached - no execution
                execution_price = np.nan
                
        return execution_price, execution_time
        
    def simulate_execution(self, execution_plan: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate execution of a trading plan.
        
        Args:
            execution_plan: Execution plan from an algorithm
            
        Returns:
            DataFrame with simulated execution results
        """
        execution_results = []
        
        for _, order in execution_plan.iterrows():
            target_time = order['timestamp']
            target_price = order['price']
            order_type = order['type']
            quantity = order['quantity']
            participation_rate = order.get('participation_rate', 0.1)
            
            # Calculate actual execution price and time
            execution_price, execution_time = self._get_execution_price(
                target_time, target_price, order_type, participation_rate)
            
            # Check if order was executed (limit orders might not execute)
            executed = not np.isnan(execution_price)
            executed_quantity = quantity if executed else 0
            
            execution_results.append({
                'planned_time': target_time,
                'actual_time': execution_time,
                'planned_price': target_price,
                'actual_price': execution_price if executed else np.nan,
                'planned_quantity': quantity,
                'executed_quantity': executed_quantity,
                'order_type': order_type,
                'executed': executed,
                'slippage': (execution_price - target_price) / target_price if executed else np.nan,
                'slippage_value': (execution_price - target_price) * executed_quantity if executed else 0,
                'latency': (execution_time - target_time).total_seconds()
            })
            
        return pd.DataFrame(execution_results)
        
    def evaluate_execution(self, execution_results: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate execution performance.
        
        Args:
            execution_results: Simulated execution results
            
        Returns:
            Dict with performance metrics
        """
        # Filter to executed orders only
        executed = execution_results[execution_results['executed']]
        
        if executed.empty:
            return {
                'fill_rate': 0.0,
                'average_slippage_bps': 0.0,
                'total_slippage_value': 0.0,
                'average_latency': 0.0,
                'implementation_shortfall': 0.0
            }
        
        # Calculate fill rate
        fill_rate = executed['executed_quantity'].sum() / execution_results['planned_quantity'].sum()
        
        # Calculate average slippage in basis points
        avg_slippage_bps = executed['slippage'].mean() * 10000
        
        # Calculate total slippage value
        total_slippage_value = executed['slippage_value'].sum()
        
        # Calculate average latency
        avg_latency = executed['latency'].mean()
        
        # Calculate implementation shortfall
        vwap_executed = (executed['actual_price'] * executed['executed_quantity']).sum() / executed['executed_quantity'].sum()
        vwap_planned = (execution_results['planned_price'] * execution_results['planned_quantity']).sum() / execution_results['planned_quantity'].sum()
        implementation_shortfall = (vwap_executed - vwap_planned) / vwap_planned * 10000  # In basis points
        
        return {
            'fill_rate': fill_rate,
            'average_slippage_bps': avg_slippage_bps,
            'total_slippage_value': total_slippage_value,
            'average_latency': avg_latency,
            'implementation_shortfall': implementation_shortfall,
            'executed_quantity': executed['executed_quantity'].sum(),
            'planned_quantity': execution_results['planned_quantity'].sum()
        } 