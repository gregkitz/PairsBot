"""
Multi-Pair Portfolio Strategy Implementation for the Intraday Statistical Arbitrage System.

This module implements a portfolio approach for managing multiple pairs simultaneously
with advanced risk management and allocation techniques.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.strategies.base import Strategy
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.risk_manager import RiskManager
from src.signal_generation.signal_generator import SignalGenerator
from src.pair_trading.pair_finder import PairFinder
from src.spread_analytics.spread_analyzer import SpreadAnalyzer
from src.asset_classes.base import Asset

# Configure logging
logger = logging.getLogger(__name__)

class MultiPairPortfolio(Strategy):
    """
    Multi-pair portfolio strategy that manages multiple pairs simultaneously.
    This strategy balances risk across pairs and optimizes overall portfolio performance.
    """
    
    def __init__(self, 
                max_pairs: int = 5,
                max_allocation_per_pair: float = 0.2,
                total_risk_allocation: float = 0.01,
                correlation_threshold: float = 0.3,
                rebalance_frequency: str = '1D',
                use_sector_constraints: bool = True,
                max_sector_allocation: float = 0.4,
                position_sizer: Optional[PositionSizer] = None,
                risk_manager: Optional[RiskManager] = None,
                signal_generator: Optional[SignalGenerator] = None,
                pair_finder: Optional[PairFinder] = None,
                spread_analyzer: Optional[SpreadAnalyzer] = None,
                **kwargs):
        """
        Initialize the multi-pair portfolio strategy.
        
        Parameters:
        -----------
        max_pairs : int, optional
            Maximum number of pairs to trade simultaneously
        max_allocation_per_pair : float, optional
            Maximum capital allocation to any single pair (0.0-1.0)
        total_risk_allocation : float, optional
            Total risk budget for the portfolio (0.0-1.0)
        correlation_threshold : float, optional
            Maximum correlation allowed between different pairs in the portfolio
        rebalance_frequency : str, optional
            How often to rebalance the portfolio (pandas offset string)
        use_sector_constraints : bool, optional
            Whether to apply sector concentration constraints
        max_sector_allocation : float, optional
            Maximum allocation to any single sector (0.0-1.0)
        position_sizer : PositionSizer, optional
            Position sizing component
        risk_manager : RiskManager, optional
            Risk management component
        signal_generator : SignalGenerator, optional
            Signal generation component
        pair_finder : PairFinder, optional
            Pair finding component
        spread_analyzer : SpreadAnalyzer, optional
            Spread analysis component
        **kwargs : dict
            Additional strategy parameters
        """
        super().__init__(**kwargs)
        self.max_pairs = max_pairs
        self.max_allocation_per_pair = max_allocation_per_pair
        self.total_risk_allocation = total_risk_allocation
        self.correlation_threshold = correlation_threshold
        self.rebalance_frequency = rebalance_frequency
        self.use_sector_constraints = use_sector_constraints
        self.max_sector_allocation = max_sector_allocation
        
        # Components
        self.position_sizer = position_sizer or PositionSizer()
        self.risk_manager = risk_manager or RiskManager()
        self.signal_generator = signal_generator or SignalGenerator()
        self.pair_finder = pair_finder or PairFinder()
        self.spread_analyzer = spread_analyzer or SpreadAnalyzer()
        
        # Portfolio state
        self.active_pairs = {}  # {pair_id: {pair_info, position_size, entry_price, etc.}}
        self.pair_candidates = []  # List of potential pairs to trade
        self.last_rebalance_time = None
        self.sector_allocations = {}  # {sector: allocation}
        self.pair_correlations = pd.DataFrame()  # Correlation matrix between pairs
    
    def find_tradable_pairs(self, universe: List[Asset], 
                           lookback_period: int = 60,
                           min_half_life: int = 1,
                           max_half_life: int = 20,
                           p_value_threshold: float = 0.05,
                           min_zscore: float = 2.0) -> List[Dict[str, Any]]:
        """
        Find tradable pairs from a universe of assets.
        
        Parameters:
        -----------
        universe : List[Asset]
            List of assets to analyze
        lookback_period : int, optional
            Days of historical data to use
        min_half_life : int, optional
            Minimum half-life in days for pair mean reversion
        max_half_life : int, optional
            Maximum half-life in days for pair mean reversion
        p_value_threshold : float, optional
            Maximum p-value for cointegration test
        min_zscore : float, optional
            Minimum z-score for considering a pair for trading
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of tradable pairs with their properties
        """
        logger.info(f"Finding tradable pairs from universe of {len(universe)} assets")
        
        # Get historical data
        end_date = datetime.now()
        start_date = pd.Timestamp(end_date) - pd.Timedelta(days=lookback_period)
        
        # Use the pair finder to identify cointegrated pairs
        pairs = self.pair_finder.find_pairs(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            p_value_threshold=p_value_threshold,
            min_half_life=min_half_life,
            max_half_life=max_half_life
        )
        
        # Filter pairs based on current z-score
        tradable_pairs = []
        for pair in pairs:
            # Calculate current spread and z-score
            spread_data = self.spread_analyzer.calculate_spread(
                asset1=pair['asset1'],
                asset2=pair['asset2'],
                hedge_ratio=pair['hedge_ratio'],
                lookback_period=20  # For z-score calculation
            )
            
            current_zscore = spread_data.get('current_zscore', 0)
            
            # Only consider pairs with significant deviation
            if abs(current_zscore) >= min_zscore:
                pair['current_zscore'] = current_zscore
                tradable_pairs.append(pair)
        
        logger.info(f"Found {len(tradable_pairs)} tradable pairs")
        return tradable_pairs
    
    def calculate_pair_correlations(self, pairs: List[Dict[str, Any]], 
                                  lookback_period: int = 30) -> pd.DataFrame:
        """
        Calculate correlation matrix between pair spreads.
        
        Parameters:
        -----------
        pairs : List[Dict[str, Any]]
            List of pairs to analyze
        lookback_period : int, optional
            Days of historical data to use
        
        Returns:
        --------
        pd.DataFrame
            Correlation matrix between pair spreads
        """
        logger.info(f"Calculating correlation matrix for {len(pairs)} pairs")
        
        if len(pairs) <= 1:
            return pd.DataFrame()
            
        # Get historical spread data for each pair
        end_date = datetime.now()
        start_date = pd.Timestamp(end_date) - pd.Timedelta(days=lookback_period)
        
        spread_data = {}
        for i, pair in enumerate(pairs):
            pair_id = f"{pair['asset1'].symbol}_{pair['asset2'].symbol}"
            spread_series = self.spread_analyzer.get_historical_spread(
                asset1=pair['asset1'],
                asset2=pair['asset2'],
                hedge_ratio=pair['hedge_ratio'],
                start_date=start_date,
                end_date=end_date
            )
            if not spread_series.empty:
                spread_data[pair_id] = spread_series
        
        # Create a DataFrame with all spreads
        if not spread_data:
            return pd.DataFrame()
            
        spread_df = pd.DataFrame(spread_data)
        
        # Calculate correlation matrix
        return spread_df.corr()
    
    def optimize_portfolio(self, tradable_pairs: List[Dict[str, Any]],
                         account_value: float) -> List[Dict[str, Any]]:
        """
        Optimize portfolio allocation across multiple pairs.
        
        Parameters:
        -----------
        tradable_pairs : List[Dict[str, Any]]
            List of tradable pairs
        account_value : float
            Current account value
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of pairs with optimized allocations
        """
        logger.info(f"Optimizing portfolio for {len(tradable_pairs)} pairs")
        
        if not tradable_pairs:
            return []
            
        # Calculate correlation matrix
        self.pair_correlations = self.calculate_pair_correlations(tradable_pairs)
        
        # Initial ranking based on cointegration strength, half-life and zscore
        for pair in tradable_pairs:
            # Create a composite score (lower is better)
            pair['score'] = (
                pair['p_value'] +  # Lower p-value is better
                pair['half_life'] / 20 +  # Shorter half-life is better (normalized by max)
                (3.0 - min(abs(pair['current_zscore']), 3.0)) / 3.0  # Higher absolute z-score is better (cap at 3)
            )
        
        # Sort by score (ascending)
        tradable_pairs.sort(key=lambda x: x['score'])
        
        # Select pairs while respecting correlation constraints
        selected_pairs = []
        selected_ids = set()
        
        for pair in tradable_pairs:
            if len(selected_pairs) >= self.max_pairs:
                break
                
            pair_id = f"{pair['asset1'].symbol}_{pair['asset2'].symbol}"
            
            # Check correlation with already selected pairs
            correlated = False
            for selected_pair in selected_pairs:
                selected_id = f"{selected_pair['asset1'].symbol}_{selected_pair['asset2'].symbol}"
                if pair_id in self.pair_correlations.index and selected_id in self.pair_correlations.columns:
                    corr = self.pair_correlations.loc[pair_id, selected_id]
                    if abs(corr) > self.correlation_threshold:
                        correlated = True
                        break
            
            if not correlated and pair_id not in selected_ids:
                # Check sector constraints if enabled
                if self.use_sector_constraints:
                    asset1_sector = getattr(pair['asset1'], 'sector', None) 
                    asset2_sector = getattr(pair['asset2'], 'sector', None)
                    
                    # Update sector allocations
                    sector_allocation_ok = True
                    temp_sector_allocations = self.sector_allocations.copy()
                    
                    for sector in [asset1_sector, asset2_sector]:
                        if sector:
                            temp_sector_allocations[sector] = temp_sector_allocations.get(sector, 0) + 0.5 * self.max_allocation_per_pair
                            if temp_sector_allocations[sector] > self.max_sector_allocation:
                                sector_allocation_ok = False
                                break
                    
                    if not sector_allocation_ok:
                        continue
                    
                    # Update sector allocations if pair is selected
                    for sector in [asset1_sector, asset2_sector]:
                        if sector:
                            self.sector_allocations[sector] = temp_sector_allocations[sector]
                
                # Calculate position size
                position_size = self.position_sizer.calculate_position_size(
                    spread_volatility=pair.get('spread_volatility', 0.01),
                    account_value=account_value,
                    max_risk=self.total_risk_allocation * self.max_allocation_per_pair
                )
                
                # Update pair with allocation
                pair['allocation'] = min(position_size / account_value, self.max_allocation_per_pair)
                selected_pairs.append(pair)
                selected_ids.add(pair_id)
        
        # Normalize allocations to fit within total risk budget
        total_allocation = sum(pair['allocation'] for pair in selected_pairs)
        if total_allocation > self.total_risk_allocation and total_allocation > 0:
            scale_factor = self.total_risk_allocation / total_allocation
            for pair in selected_pairs:
                pair['allocation'] *= scale_factor
        
        logger.info(f"Selected {len(selected_pairs)} pairs for the portfolio")
        return selected_pairs
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with portfolio state information
        """
        return {
            'active_pairs': len(self.active_pairs),
            'pair_details': self.active_pairs,
            'sector_allocations': self.sector_allocations,
            'last_rebalance': self.last_rebalance_time,
            'total_allocation': sum(pair.get('allocation', 0) for pair in self.active_pairs.values())
        }
    
    def should_rebalance(self, current_time: datetime) -> bool:
        """
        Determine if the portfolio should be rebalanced.
        
        Parameters:
        -----------
        current_time : datetime
            Current time
        
        Returns:
        --------
        bool
            True if portfolio should be rebalanced
        """
        if self.last_rebalance_time is None:
            return True
            
        elapsed = pd.Timestamp(current_time) - pd.Timestamp(self.last_rebalance_time)
        threshold = pd.Timedelta(self.rebalance_frequency)
        
        return elapsed >= threshold
    
    def rebalance_portfolio(self, universe: List[Asset], account_value: float) -> Dict[str, Any]:
        """
        Rebalance the portfolio based on current market conditions.
        
        Parameters:
        -----------
        universe : List[Asset]
            List of assets to consider
        account_value : float
            Current account value
        
        Returns:
        --------
        Dict[str, Any]
            Rebalancing actions to take
        """
        logger.info("Rebalancing portfolio")
        
        # Reset sector allocations
        self.sector_allocations = {}
        
        # Find tradable pairs
        tradable_pairs = self.find_tradable_pairs(universe)
        
        # Optimize portfolio
        selected_pairs = self.optimize_portfolio(tradable_pairs, account_value)
        
        # Prepare actions
        actions = {
            'close': [],  # Pairs to close
            'open': [],   # Pairs to open
            'adjust': []  # Pairs to adjust position size
        }
        
        # Identify pairs to close
        for pair_id, pair_info in list(self.active_pairs.items()):
            found = False
            for new_pair in selected_pairs:
                new_pair_id = f"{new_pair['asset1'].symbol}_{new_pair['asset2'].symbol}"
                if new_pair_id == pair_id:
                    found = True
                    break
                    
            if not found:
                actions['close'].append(pair_info)
                del self.active_pairs[pair_id]
        
        # Identify pairs to open or adjust
        for new_pair in selected_pairs:
            new_pair_id = f"{new_pair['asset1'].symbol}_{new_pair['asset2'].symbol}"
            
            if new_pair_id in self.active_pairs:
                # Check if position size needs adjustment
                current_allocation = self.active_pairs[new_pair_id].get('allocation', 0)
                new_allocation = new_pair.get('allocation', 0)
                
                if abs(current_allocation - new_allocation) / max(current_allocation, 1e-10) > 0.1:  # 10% change threshold
                    actions['adjust'].append({
                        'pair_id': new_pair_id,
                        'current_allocation': current_allocation,
                        'new_allocation': new_allocation,
                        'pair_info': new_pair
                    })
                    # Update the active pair info
                    self.active_pairs[new_pair_id].update({
                        'allocation': new_allocation,
                        'current_zscore': new_pair.get('current_zscore')
                    })
            else:
                # New pair to open
                actions['open'].append(new_pair)
                self.active_pairs[new_pair_id] = new_pair
        
        # Update last rebalance time
        self.last_rebalance_time = datetime.now()
        
        logger.info(f"Rebalance actions: close {len(actions['close'])}, open {len(actions['open'])}, adjust {len(actions['adjust'])}")
        return actions
    
    def update(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the strategy with current market data.
        
        Parameters:
        -----------
        current_data : Dict[str, Any]
            Current market data, including time and account information
        
        Returns:
        --------
        Dict[str, Any]
            Dict with signal updates and actions
        """
        current_time = current_data.get('timestamp', datetime.now())
        account_value = current_data.get('account_value', 0)
        universe = current_data.get('universe', [])
        
        actions = {'signals': [], 'rebalance': None}
        
        # Check if we need to rebalance
        if self.should_rebalance(current_time):
            rebalance_actions = self.rebalance_portfolio(universe, account_value)
            actions['rebalance'] = rebalance_actions
        
        # Update signals for active pairs
        for pair_id, pair_info in self.active_pairs.items():
            # Get current spread and z-score
            spread_data = self.spread_analyzer.calculate_spread(
                asset1=pair_info['asset1'],
                asset2=pair_info['asset2'],
                hedge_ratio=pair_info['hedge_ratio']
            )
            
            current_zscore = spread_data.get('current_zscore', 0)
            pair_info['current_zscore'] = current_zscore
            
            # Generate signals
            signal = self.signal_generator.generate_signal(
                zscore=current_zscore,
                pair_info=pair_info
            )
            
            # Apply risk management
            risk_adjusted_signal = self.risk_manager.adjust_signal(
                signal=signal,
                pair_info=pair_info,
                portfolio_state=self.get_portfolio_state()
            )
            
            # Add to actions
            actions['signals'].append({
                'pair_id': pair_id,
                'signal': risk_adjusted_signal,
                'zscore': current_zscore,
                'pair_info': pair_info
            })
        
        return actions
    
    def generate_orders(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert signals into executable orders.
        
        Parameters:
        -----------
        signals : Dict[str, Any]
            Dictionary of signals and actions
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of orders to execute
        """
        orders = []
        
        # Process rebalance actions
        if signals.get('rebalance'):
            rebalance = signals['rebalance']
            
            # Close positions
            for pair_info in rebalance.get('close', []):
                pair_id = f"{pair_info['asset1'].symbol}_{pair_info['asset2'].symbol}"
                
                # Create closing orders
                if 'leg1_position' in pair_info and pair_info['leg1_position'] != 0:
                    orders.append({
                        'action': 'close',
                        'asset': pair_info['asset1'],
                        'quantity': pair_info.get('leg1_position', 0),
                        'pair_id': pair_id,
                        'reason': 'portfolio_rebalance'
                    })
                
                if 'leg2_position' in pair_info and pair_info['leg2_position'] != 0:
                    orders.append({
                        'action': 'close',
                        'asset': pair_info['asset2'],
                        'quantity': pair_info.get('leg2_position', 0),
                        'pair_id': pair_id,
                        'reason': 'portfolio_rebalance'
                    })
            
            # Open new positions
            for pair_info in rebalance.get('open', []):
                pair_id = f"{pair_info['asset1'].symbol}_{pair_info['asset2'].symbol}"
                
                if 'allocation' not in pair_info or pair_info['allocation'] <= 0:
                    continue
                    
                allocation = pair_info.get('allocation', 0)
                current_zscore = pair_info.get('current_zscore', 0)
                
                # Determine direction based on z-score
                # Negative z-score: asset1 is undervalued (buy asset1, sell asset2)
                # Positive z-score: asset1 is overvalued (sell asset1, buy asset2)
                if current_zscore > 0:
                    leg1_direction = 'sell'
                    leg2_direction = 'buy'
                else:
                    leg1_direction = 'buy'
                    leg2_direction = 'sell'
                
                # Calculate position sizes
                try:
                    leg1_price = pair_info['asset1'].get_current_price()
                    leg2_price = pair_info['asset2'].get_current_price()
                    
                    if leg1_price <= 0 or leg2_price <= 0:
                        logger.warning(f"Invalid prices for pair {pair_id}: {leg1_price}, {leg2_price}")
                        continue
                    
                    # For testing purposes, assume a notional value of $100,000
                    account_value = 100000
                    
                    # Simple distribution: split allocation equally between legs
                    leg1_notional = account_value * allocation * 0.5
                    leg2_notional = account_value * allocation * 0.5 * pair_info['hedge_ratio']
                    
                    leg1_quantity = round(leg1_notional / leg1_price)
                    leg2_quantity = round(leg2_notional / leg2_price)
                    
                    # Store position size in pair info
                    pair_info['leg1_position'] = leg1_quantity if leg1_direction == 'buy' else -leg1_quantity
                    pair_info['leg2_position'] = leg2_quantity if leg2_direction == 'buy' else -leg2_quantity
                    
                    # Create opening orders
                    if leg1_quantity > 0:
                        orders.append({
                            'action': leg1_direction,
                            'asset': pair_info['asset1'],
                            'quantity': leg1_quantity,
                            'pair_id': pair_id,
                            'reason': 'new_position'
                        })
                    
                    if leg2_quantity > 0:
                        orders.append({
                            'action': leg2_direction,
                            'asset': pair_info['asset2'],
                            'quantity': leg2_quantity,
                            'pair_id': pair_id,
                            'reason': 'new_position'
                        })
                except Exception as e:
                    logger.error(f"Error calculating position sizes for pair {pair_id}: {str(e)}")
        
        # Process signal actions 
        for signal_info in signals.get('signals', []):
            signal = signal_info.get('signal')
            pair_info = signal_info.get('pair_info')
            pair_id = signal_info.get('pair_id')
            
            if signal and pair_info and pair_id:
                if signal.get('action') == 'close':
                    # Create closing orders
                    if 'leg1_position' in pair_info and pair_info['leg1_position'] != 0:
                        orders.append({
                            'action': 'close',
                            'asset': pair_info['asset1'],
                            'quantity': pair_info.get('leg1_position', 0),
                            'pair_id': pair_id,
                            'reason': signal.get('reason', 'signal_close')
                        })
                    
                    if 'leg2_position' in pair_info and pair_info['leg2_position'] != 0:
                        orders.append({
                            'action': 'close',
                            'asset': pair_info['asset2'],
                            'quantity': pair_info.get('leg2_position', 0),
                            'pair_id': pair_id,
                            'reason': signal.get('reason', 'signal_close')
                        })
        
        return orders 