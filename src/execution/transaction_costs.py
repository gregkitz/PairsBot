"""
Transaction Cost Models

This module provides models to estimate transaction costs for trading operations, including:
1. Exchange fees for different venues
2. Slippage models based on volume and volatility
3. Market impact modeling for different order sizes
4. Cost analysis tools for strategy assessment

These models are crucial for realistic backtesting and strategy assessment.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass


class ExchangeType(Enum):
    """Enumeration of exchange types for fee modeling."""
    
    STOCK_EXCHANGE = "stock_exchange"
    FUTURES_EXCHANGE = "futures_exchange"
    CRYPTO_EXCHANGE = "crypto_exchange"
    FOREX_EXCHANGE = "forex_exchange"
    OPTIONS_EXCHANGE = "options_exchange"


class SlippageModel(Enum):
    """Enumeration of slippage model types."""
    
    FIXED = "fixed"  # Fixed basis points
    PERCENTAGE = "percentage"  # Percentage of price
    VOLUME_BASED = "volume_based"  # Based on order size relative to volume
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Adjusted based on volatility
    MARKET_IMPACT = "market_impact"  # Full market impact model
    CUSTOM = "custom"  # Custom model function


@dataclass
class ExchangeFeeConfig:
    """Configuration for exchange fee calculation."""
    
    maker_fee: float  # Fee for maker orders (providing liquidity), in bps
    taker_fee: float  # Fee for taker orders (taking liquidity), in bps
    minimum_fee: float  # Minimum fee per trade
    maximum_fee: Optional[float] = None  # Maximum fee per trade, if applicable
    fixed_fees: Optional[Dict[str, float]] = None  # Fixed fees by instrument
    tier_levels: Optional[List[Tuple[float, float]]] = None  # Volume-based tier levels
    rebates: Optional[Dict[str, float]] = None  # Rebates for specific conditions


class ExchangeFeeModel:
    """Model for calculating exchange fees based on order characteristics."""
    
    def __init__(self, exchange_type: ExchangeType, config: ExchangeFeeConfig):
        """
        Initialize the exchange fee model.
        
        Args:
            exchange_type: Type of exchange for fee structure
            config: Configuration for fee calculation
        """
        self.exchange_type = exchange_type
        self.config = config
        
    def calculate_fee(self, order_value: float, is_maker: bool = False, 
                     monthly_volume: Optional[float] = None,
                     symbol: Optional[str] = None) -> float:
        """
        Calculate the fee for a given order based on the exchange fee structure.
        
        Args:
            order_value: Monetary value of the order
            is_maker: Whether the order provides liquidity (maker) or takes liquidity (taker)
            monthly_volume: Monthly trading volume for tier-based fees
            symbol: Trading symbol/instrument
            
        Returns:
            The calculated fee amount
        """
        # Start with basic maker/taker fee
        if is_maker:
            fee_rate = self.config.maker_fee / 10000  # Convert from bps to decimal
        else:
            fee_rate = self.config.taker_fee / 10000
            
        # Apply volume-based tiers if applicable
        if monthly_volume is not None and self.config.tier_levels is not None:
            for tier_volume, tier_fee in sorted(self.config.tier_levels):
                if monthly_volume >= tier_volume:
                    fee_rate = tier_fee / 10000
                else:
                    break
                    
        # Apply fixed fees for specific instruments if applicable
        if symbol is not None and self.config.fixed_fees is not None and symbol in self.config.fixed_fees:
            return self.config.fixed_fees[symbol]
            
        # Calculate the fee
        fee = order_value * fee_rate
        
        # Apply minimum fee if necessary
        fee = max(fee, self.config.minimum_fee)
        
        # Apply maximum fee if specified
        if self.config.maximum_fee is not None:
            fee = min(fee, self.config.maximum_fee)
            
        # Apply rebates if applicable
        if is_maker and self.config.rebates is not None and symbol in self.config.rebates:
            fee -= order_value * (self.config.rebates[symbol] / 10000)
            fee = max(0, fee)  # Ensure fee doesn't go negative
            
        return fee


class SlippageModelBase:
    """Base class for slippage models."""
    
    def __init__(self):
        pass
    
    def calculate_slippage(self, price: float, quantity: float, 
                           market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate slippage for a given order.
        
        Args:
            price: Execution price
            quantity: Order quantity
            market_data: Additional market data for calculation
            
        Returns:
            The calculated slippage amount
        """
        raise NotImplementedError("Subclasses must implement calculate_slippage method")


class FixedSlippageModel(SlippageModelBase):
    """Fixed basis points slippage model."""
    
    def __init__(self, basis_points: float):
        """
        Initialize the fixed slippage model.
        
        Args:
            basis_points: Fixed slippage in basis points
        """
        super().__init__()
        self.basis_points = basis_points
        
    def calculate_slippage(self, price: float, quantity: float, 
                           market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate slippage based on fixed basis points.
        
        Args:
            price: Execution price
            quantity: Order quantity
            market_data: Not used in this model
            
        Returns:
            The calculated slippage amount
        """
        return price * quantity * (self.basis_points / 10000)


class VolumeBasedSlippageModel(SlippageModelBase):
    """Volume-based slippage model that scales with order size relative to market volume."""
    
    def __init__(self, base_slippage_bps: float, volume_impact_factor: float):
        """
        Initialize the volume-based slippage model.
        
        Args:
            base_slippage_bps: Base slippage in basis points
            volume_impact_factor: Factor to scale impact with volume ratio
        """
        super().__init__()
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor
        
    def calculate_slippage(self, price: float, quantity: float, 
                           market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate slippage based on order volume relative to market volume.
        
        Args:
            price: Execution price
            quantity: Order quantity
            market_data: Must contain 'volume' column
            
        Returns:
            The calculated slippage amount
        """
        if market_data is None or 'volume' not in market_data.columns:
            raise ValueError("Market data with volume is required for volume-based slippage")
            
        # Calculate average daily volume
        avg_volume = market_data['volume'].mean()
        
        # Calculate volume ratio (order size / average volume)
        volume_ratio = (quantity / avg_volume)
        
        # Apply non-linear scaling to volume impact
        impact_multiplier = np.power(volume_ratio, self.volume_impact_factor)
        
        # Calculate adjusted slippage in basis points
        adjusted_slippage_bps = self.base_slippage_bps * (1 + impact_multiplier)
        
        # Calculate and return slippage amount
        return price * quantity * (adjusted_slippage_bps / 10000)


class VolatilityAdjustedSlippageModel(SlippageModelBase):
    """Volatility-adjusted slippage model scaling with price volatility."""
    
    def __init__(self, base_slippage_bps: float, volatility_impact_factor: float):
        """
        Initialize the volatility-adjusted slippage model.
        
        Args:
            base_slippage_bps: Base slippage in basis points
            volatility_impact_factor: Factor to scale impact with volatility
        """
        super().__init__()
        self.base_slippage_bps = base_slippage_bps
        self.volatility_impact_factor = volatility_impact_factor
        
    def calculate_slippage(self, price: float, quantity: float, 
                           market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate slippage based on market volatility.
        
        Args:
            price: Execution price
            quantity: Order quantity
            market_data: Must contain price data for volatility calculation
            
        Returns:
            The calculated slippage amount
        """
        if market_data is None or len(market_data) < 2:
            raise ValueError("Market data is required for volatility-adjusted slippage")
            
        # Calculate daily volatility (annualized)
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().dropna()
        elif 'price' in market_data.columns:
            returns = market_data['price'].pct_change().dropna()
        else:
            raise ValueError("Market data must contain 'close' or 'price' column")
            
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Adjust slippage based on volatility
        volatility_multiplier = np.power(volatility, self.volatility_impact_factor)
        
        # Calculate adjusted slippage in basis points
        adjusted_slippage_bps = self.base_slippage_bps * volatility_multiplier
        
        # Calculate and return slippage amount
        return price * quantity * (adjusted_slippage_bps / 10000)


class MarketImpactModel:
    """
    Full market impact model accounting for both temporary and permanent impact.
    
    Based on academic research and standard market impact models, this accounts for:
    - Temporary price impact from immediate execution
    - Permanent price impact that affects future trading
    - Non-linear scaling with order size and market characteristics
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.3, gamma: float = 0.5,
                temporary_impact_factor: float = 1.0, permanent_impact_factor: float = 0.5):
        """
        Initialize the market impact model.
        
        Args:
            alpha: Scaling parameter for market impact (typical range 0.05-0.3)
            beta: Exponent for order size scaling (typical range 0.2-0.5)
            gamma: Exponent for volatility scaling (typical range 0.3-0.7)
            temporary_impact_factor: Multiplier for temporary impact component
            permanent_impact_factor: Multiplier for permanent impact component
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temporary_impact_factor = temporary_impact_factor
        self.permanent_impact_factor = permanent_impact_factor
        
    def calculate_market_impact(self, price: float, quantity: float, 
                               market_data: pd.DataFrame,
                               is_buy: bool = True,
                               participation_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate market impact for an order.
        
        Args:
            price: Current market price
            quantity: Order quantity
            market_data: Market data including volume and price
            is_buy: Whether order is a buy (True) or sell (False)
            participation_rate: Desired participation rate (e.g., 0.1 for 10%)
            
        Returns:
            Dict containing temporary_impact, permanent_impact, total_impact values
        """
        if market_data is None or len(market_data) < 10:
            raise ValueError("Sufficient market data is required for market impact calculation")
            
        # Calculate average daily volume (ADV)
        if 'volume' in market_data.columns:
            adv = market_data['volume'].mean()
        else:
            raise ValueError("Market data must contain 'volume' column")
            
        # Calculate volatility (annualized)
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().dropna()
        elif 'price' in market_data.columns:
            returns = market_data['price'].pct_change().dropna()
        else:
            raise ValueError("Market data must contain 'close' or 'price' column")
            
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate participation rate if not provided
        if participation_rate is None:
            participation_rate = quantity / adv
            
        # Calculate market impact in basis points (theoretical model)
        # Uses the standard square-root formula with adjustable parameters
        impact_bps = self.alpha * np.power(participation_rate, self.beta) * np.power(volatility, self.gamma) * 10000
        
        # Calculate temporary and permanent impact
        temporary_impact = price * quantity * (impact_bps / 10000) * self.temporary_impact_factor
        permanent_impact = price * quantity * (impact_bps / 10000) * self.permanent_impact_factor
        
        # Direction adjustment (negative impact for sells, positive for buys)
        direction_multiplier = 1 if is_buy else -1
        
        return {
            "temporary_impact": temporary_impact * direction_multiplier,
            "permanent_impact": permanent_impact * direction_multiplier,
            "total_impact": (temporary_impact + permanent_impact) * direction_multiplier,
            "impact_bps": impact_bps
        }


class CustomSlippageModel(SlippageModelBase):
    """Custom slippage model using a user-provided function."""
    
    def __init__(self, slippage_function: Callable[[float, float, Optional[pd.DataFrame]], float]):
        """
        Initialize the custom slippage model.
        
        Args:
            slippage_function: A function that takes price, quantity, and market_data
                              and returns the slippage amount
        """
        super().__init__()
        self.slippage_function = slippage_function
        
    def calculate_slippage(self, price: float, quantity: float, 
                           market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate slippage using the provided custom function.
        
        Args:
            price: Execution price
            quantity: Order quantity
            market_data: Market data for calculation
            
        Returns:
            The calculated slippage amount
        """
        return self.slippage_function(price, quantity, market_data)


class TransactionCostAnalyzer:
    """
    Analyzes and summarizes transaction costs for trading strategies.
    
    Combines fees, slippage, and market impact to provide comprehensive 
    cost analysis for backtesting and strategy evaluation.
    """
    
    def __init__(self, 
                 exchange_fee_model: ExchangeFeeModel,
                 slippage_model: SlippageModelBase,
                 market_impact_model: Optional[MarketImpactModel] = None):
        """
        Initialize the transaction cost analyzer.
        
        Args:
            exchange_fee_model: Model for exchange fee calculation
            slippage_model: Model for slippage calculation
            market_impact_model: Optional model for market impact calculation
        """
        self.exchange_fee_model = exchange_fee_model
        self.slippage_model = slippage_model
        self.market_impact_model = market_impact_model
        
    def calculate_transaction_costs(self, 
                                   price: float, 
                                   quantity: float,
                                   market_data: pd.DataFrame,
                                   is_buy: bool = True,
                                   is_maker: bool = False,
                                   monthly_volume: Optional[float] = None,
                                   symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate comprehensive transaction costs for a trade.
        
        Args:
            price: Execution price
            quantity: Order quantity
            market_data: Market data for calculations
            is_buy: Whether order is buy (True) or sell (False)
            is_maker: Whether order is maker (True) or taker (False)
            monthly_volume: Monthly trading volume for tier-based fees
            symbol: Trading symbol/instrument
            
        Returns:
            Dict containing detailed transaction cost breakdown
        """
        order_value = price * quantity
        
        # Calculate exchange fees
        exchange_fee = self.exchange_fee_model.calculate_fee(
            order_value, is_maker, monthly_volume, symbol)
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(price, quantity, market_data)
        
        # Calculate market impact if model is provided
        if self.market_impact_model is not None:
            market_impact = self.market_impact_model.calculate_market_impact(
                price, quantity, market_data, is_buy)
            impact_cost = market_impact["total_impact"]
        else:
            impact_cost = 0.0
            market_impact = {"temporary_impact": 0.0, "permanent_impact": 0.0, "total_impact": 0.0, "impact_bps": 0.0}
        
        # Total transaction cost
        total_cost = exchange_fee + slippage + impact_cost
        
        # Transaction cost as percentage of order value
        cost_percentage = (total_cost / order_value) * 100 if order_value > 0 else 0.0
        
        return {
            "order_value": order_value,
            "exchange_fee": exchange_fee,
            "slippage": slippage,
            "impact_cost": impact_cost,
            "temporary_impact": market_impact["temporary_impact"],
            "permanent_impact": market_impact["permanent_impact"],
            "total_cost": total_cost,
            "cost_percentage": cost_percentage,
            "cost_bps": cost_percentage * 100  # Convert percentage to basis points
        }
        
    def analyze_strategy_costs(self, trades: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze transaction costs for a series of trades.
        
        Args:
            trades: DataFrame containing trade information including:
                  - price: Execution price
                  - quantity: Order quantity
                  - is_buy: Whether trade is buy (True) or sell (False)
                  - is_maker: Whether trade is maker (True) or taker (False)
                  - timestamp: Trade timestamp
                  - symbol: Trading symbol/instrument
            market_data: Market data indexed by timestamp and symbol
            
        Returns:
            Dict containing aggregated transaction cost statistics
        """
        if trades.empty:
            return {
                "total_trades": 0,
                "total_volume": 0.0,
                "total_value": 0.0,
                "total_cost": 0.0,
                "total_cost_percentage": 0.0
            }
        
        results = []
        
        for _, trade in trades.iterrows():
            # Get market data for the trade
            symbol = trade.get('symbol', None)
            timestamp = trade.get('timestamp', None)
            
            # Filter market data for the specific symbol and around the timestamp
            if symbol is not None and timestamp is not None:
                trade_market_data = market_data[
                    (market_data['symbol'] == symbol) & 
                    (market_data.index <= timestamp)
                ].tail(20)  # Use last 20 data points before the trade
            else:
                trade_market_data = market_data.tail(20)
                
            # Calculate costs for this trade
            costs = self.calculate_transaction_costs(
                price=trade['price'],
                quantity=trade['quantity'],
                market_data=trade_market_data,
                is_buy=trade.get('is_buy', True),
                is_maker=trade.get('is_maker', False),
                monthly_volume=trade.get('monthly_volume', None),
                symbol=symbol
            )
            
            results.append(costs)
            
        # Convert results to DataFrame for analysis
        costs_df = pd.DataFrame(results)
        
        # Calculate aggregated statistics
        total_value = costs_df['order_value'].sum()
        total_cost = costs_df['total_cost'].sum()
        
        return {
            "total_trades": len(trades),
            "total_volume": trades['quantity'].sum(),
            "total_value": total_value,
            "total_cost": total_cost,
            "average_cost_per_trade": total_cost / len(trades),
            "total_cost_percentage": (total_cost / total_value) * 100 if total_value > 0 else 0.0,
            "total_cost_bps": (total_cost / total_value) * 10000 if total_value > 0 else 0.0,
            "fee_cost": costs_df['exchange_fee'].sum(),
            "slippage_cost": costs_df['slippage'].sum(),
            "impact_cost": costs_df['impact_cost'].sum(),
            "fee_percentage": costs_df['exchange_fee'].sum() / total_cost * 100 if total_cost > 0 else 0.0,
            "slippage_percentage": costs_df['slippage'].sum() / total_cost * 100 if total_cost > 0 else 0.0,
            "impact_percentage": costs_df['impact_cost'].sum() / total_cost * 100 if total_cost > 0 else 0.0
        }


# Pre-configured exchange fee models for common exchanges
def create_futures_exchange_fee_model() -> ExchangeFeeModel:
    """Create a pre-configured fee model for a typical futures exchange."""
    config = ExchangeFeeConfig(
        maker_fee=1.0,    # 1.0 bps (0.01%)
        taker_fee=3.5,    # 3.5 bps (0.035%)
        minimum_fee=1.0,  # $1.0 minimum fee
        tier_levels=[
            (1000000, 3.0),  # $1M monthly volume: 3.0 bps taker fee
            (5000000, 2.5),  # $5M monthly volume: 2.5 bps taker fee
            (10000000, 2.0), # $10M monthly volume: 2.0 bps taker fee
            (25000000, 1.5), # $25M monthly volume: 1.5 bps taker fee
            (50000000, 1.0), # $50M monthly volume: 1.0 bps taker fee
        ]
    )
    return ExchangeFeeModel(ExchangeType.FUTURES_EXCHANGE, config)


def create_stock_exchange_fee_model() -> ExchangeFeeModel:
    """Create a pre-configured fee model for a typical stock exchange."""
    config = ExchangeFeeConfig(
        maker_fee=-0.2,   # -0.2 bps (-0.002%) - rebate
        taker_fee=3.0,    # 3.0 bps (0.03%)
        minimum_fee=1.0,  # $1.0 minimum fee
        maximum_fee=None  # No maximum fee
    )
    return ExchangeFeeModel(ExchangeType.STOCK_EXCHANGE, config)


def create_crypto_exchange_fee_model() -> ExchangeFeeModel:
    """Create a pre-configured fee model for a typical crypto exchange."""
    config = ExchangeFeeConfig(
        maker_fee=10.0,   # 10.0 bps (0.1%)
        taker_fee=20.0,   # 20.0 bps (0.2%)
        minimum_fee=1.0,  # $1.0 minimum fee
        tier_levels=[
            (100000, 15.0),   # $100K monthly volume: 15 bps taker fee
            (500000, 10.0),   # $500K monthly volume: 10 bps taker fee
            (1000000, 7.5),   # $1M monthly volume: 7.5 bps taker fee
            (5000000, 5.0),   # $5M monthly volume: 5 bps taker fee
            (10000000, 3.5),  # $10M monthly volume: 3.5 bps taker fee
        ]
    )
    return ExchangeFeeModel(ExchangeType.CRYPTO_EXCHANGE, config) 