"""
Position Manager Component for the Paper Trading module.

This component is responsible for tracking and managing positions,
including entries, exits, and risk management.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class PositionManager:
    """
    Component responsible for tracking and managing trading positions.
    
    This class extracts position management functionality from
    the IntradayMLPaperTrader class, providing a focused interface for
    working with trading positions.
    """
    
    def __init__(
        self,
        paper_trader: Any = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the position manager component.
        
        Parameters:
        -----------
        paper_trader : Any
            Reference to the paper trader for executing orders
        config : dict, optional
            Configuration options
        """
        self.paper_trader = paper_trader
        self.config = config or {}
        self.positions = {}  # Pair ID -> Position details
        self.position_history = []  # List of closed positions
        self.pair_configs = {}  # Pair ID -> Pair configuration
    
    def add_pair(self, pair_config: Dict[str, Any]) -> bool:
        """
        Add a trading pair to be managed.
        
        Parameters:
        -----------
        pair_config : dict
            Pair configuration
            
        Returns:
        --------
        bool
            Success flag
        """
        if not pair_config.get('pair_id'):
            logger.error("Pair configuration missing pair_id")
            return False
        
        try:
            pair_id = pair_config['pair_id']
            self.pair_configs[pair_id] = pair_config
            logger.info(f"Added pair {pair_id} to position manager")
            return True
            
        except Exception as e:
            logger.error(f"Error adding pair: {str(e)}")
            return False
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all current positions.
        
        Returns:
        --------
        dict
            Dictionary of positions, keyed by pair ID
        """
        return self.positions.copy()
    
    def get_position(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific pair.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier
            
        Returns:
        --------
        dict or None
            Position details or None if no position
        """
        return self.positions.get(pair_id)
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get history of closed positions.
        
        Returns:
        --------
        list
            List of closed positions
        """
        return self.position_history.copy()
    
    def execute_signals(
        self,
        pair_id: str,
        signals: pd.Series,
        prices: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Execute trading signals for a pair.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        signals : pd.Series
            Series with trading signals
        prices : pd.DataFrame, optional
            DataFrame with price data
            
        Returns:
        --------
        dict
            Execution results
        """
        if not signals.any():
            logger.info(f"No signals to execute for {pair_id}")
            return {"executed": 0, "errors": 0}
        
        if self.paper_trader is None:
            logger.error("Paper trader not available for executing signals")
            return {"executed": 0, "errors": 1}
        
        pair_config = self.pair_configs.get(pair_id)
        if not pair_config:
            logger.error(f"No configuration found for pair {pair_id}")
            return {"executed": 0, "errors": 1}
        
        try:
            # Get the latest signal
            latest_signal = signals.iloc[-1]
            
            # Get current position
            current_position = self.get_position(pair_id)
            
            # If no position and signal is non-zero, enter position
            if current_position is None and latest_signal != 0:
                result = self._enter_position(pair_id, latest_signal, pair_config)
                return {"executed": 1 if result else 0, "errors": 0 if result else 1}
            
            # If position exists and signal is opposite, exit position
            elif current_position is not None:
                current_dir = current_position['direction']
                
                if (current_dir > 0 and latest_signal < 0) or (current_dir < 0 and latest_signal > 0):
                    # Exit current position
                    exit_result = self._exit_position(pair_id, "signal_reversal")
                    
                    # Enter new position
                    if exit_result and latest_signal != 0:
                        entry_result = self._enter_position(pair_id, latest_signal, pair_config)
                        return {
                            "executed": 2 if entry_result else 1, 
                            "errors": 0 if entry_result else 1
                        }
                    
                    return {
                        "executed": 1 if exit_result else 0,
                        "errors": 0 if exit_result else 1
                    }
                
                # If position exists and signal is zero, exit position
                elif latest_signal == 0:
                    exit_result = self._exit_position(pair_id, "signal_exit")
                    return {
                        "executed": 1 if exit_result else 0,
                        "errors": 0 if exit_result else 1
                    }
            
            # No action needed
            return {"executed": 0, "errors": 0}
            
        except Exception as e:
            logger.error(f"Error executing signals for {pair_id}: {str(e)}")
            return {"executed": 0, "errors": 1}
    
    def _enter_position(
        self,
        pair_id: str,
        signal: int,
        pair_config: Dict[str, Any]
    ) -> bool:
        """
        Enter a new position.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        signal : int
            Signal value (1 for long, -1 for short)
        pair_config : dict
            Pair configuration
            
        Returns:
        --------
        bool
            Success flag
        """
        if self.paper_trader is None:
            logger.error("Paper trader not available for entering position")
            return False
        
        try:
            # Get leg symbols
            leg1 = pair_config.get('leg1')
            leg2 = pair_config.get('leg2')
            
            if not leg1 or not leg2:
                logger.error(f"Pair {pair_id} missing leg symbols")
                return False
            
            # Get multipliers
            leg1_multiplier = pair_config.get('leg1_multiplier', 1.0)
            leg2_multiplier = pair_config.get('leg2_multiplier', 1.0)
            
            # Get entry parameters (use regime parameters if available)
            hedge_ratio = pair_config.get('hedge_ratio', 1.0)
            
            # Calculate quantities
            leg1_quantity = 1.0 * leg1_multiplier
            leg2_quantity = hedge_ratio * leg2_multiplier
            
            # Determine direction (long spread = long leg1, short leg2)
            direction = signal  # 1 for long spread, -1 for short spread
            
            # Place orders
            if direction > 0:
                # Long spread: Buy leg1, sell leg2
                leg1_action = 'BUY'
                leg2_action = 'SELL'
            else:
                # Short spread: Sell leg1, buy leg2
                leg1_action = 'SELL'
                leg2_action = 'BUY'
            
            # Execute leg1 order
            leg1_order_id = self.paper_trader.place_order(
                symbol=leg1,
                action=leg1_action,
                quantity=leg1_quantity,
                order_type='MKT'
            )
            
            # Execute leg2 order
            leg2_order_id = self.paper_trader.place_order(
                symbol=leg2,
                action=leg2_action,
                quantity=leg2_quantity,
                order_type='MKT'
            )
            
            # Create position record
            position = {
                'pair_id': pair_id,
                'direction': direction,
                'entry_time': datetime.now(),
                'leg1': {
                    'symbol': leg1,
                    'action': leg1_action,
                    'quantity': leg1_quantity,
                    'order_id': leg1_order_id
                },
                'leg2': {
                    'symbol': leg2,
                    'action': leg2_action,
                    'quantity': leg2_quantity,
                    'order_id': leg2_order_id
                },
                'status': 'open'
            }
            
            # Store position
            self.positions[pair_id] = position
            
            logger.info(f"Entered position for {pair_id}: direction={direction}")
            return True
            
        except Exception as e:
            logger.error(f"Error entering position for {pair_id}: {str(e)}")
            return False
    
    def _exit_position(
        self,
        pair_id: str,
        exit_reason: str = "manual"
    ) -> bool:
        """
        Exit an existing position.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        exit_reason : str, optional
            Reason for exiting the position
            
        Returns:
        --------
        bool
            Success flag
        """
        if self.paper_trader is None:
            logger.error("Paper trader not available for exiting position")
            return False
        
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            logger.warning(f"No position found for {pair_id}")
            return False
        
        try:
            # Get leg details
            leg1 = position['leg1']
            leg2 = position['leg2']
            
            # Determine exit actions (opposite of entry)
            leg1_exit_action = 'SELL' if leg1['action'] == 'BUY' else 'BUY'
            leg2_exit_action = 'SELL' if leg2['action'] == 'BUY' else 'BUY'
            
            # Execute leg1 exit order
            leg1_exit_order_id = self.paper_trader.place_order(
                symbol=leg1['symbol'],
                action=leg1_exit_action,
                quantity=leg1['quantity'],
                order_type='MKT'
            )
            
            # Execute leg2 exit order
            leg2_exit_order_id = self.paper_trader.place_order(
                symbol=leg2['symbol'],
                action=leg2_exit_action,
                quantity=leg2['quantity'],
                order_type='MKT'
            )
            
            # Update position record
            position['exit_time'] = datetime.now()
            position['exit_reason'] = exit_reason
            position['leg1_exit_order_id'] = leg1_exit_order_id
            position['leg2_exit_order_id'] = leg2_exit_order_id
            position['status'] = 'closed'
            
            # Calculate PnL (simplified, actual PnL would come from paper_trader)
            leg1_position = self.paper_trader.get_position(leg1['symbol'])
            leg2_position = self.paper_trader.get_position(leg2['symbol'])
            
            # Update PnL if available
            if leg1_position and 'unrealized_pnl' in leg1_position:
                position['leg1_pnl'] = leg1_position['unrealized_pnl']
            
            if leg2_position and 'unrealized_pnl' in leg2_position:
                position['leg2_pnl'] = leg2_position['unrealized_pnl']
            
            # Calculate total PnL
            position['pnl'] = position.get('leg1_pnl', 0.0) + position.get('leg2_pnl', 0.0)
            
            # Move position to history
            self.position_history.append(position)
            del self.positions[pair_id]
            
            logger.info(f"Exited position for {pair_id}: reason={exit_reason}, pnl={position.get('pnl', 0.0)}")
            return True
            
        except Exception as e:
            logger.error(f"Error exiting position for {pair_id}: {str(e)}")
            return False
    
    def check_stop_losses(
        self,
        pair_id: str,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if stop loss conditions are met for a position.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        current_data : pd.DataFrame
            Current market data
            
        Returns:
        --------
        bool
            True if stop loss triggered, False otherwise
        """
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            return False
        
        # Get pair config
        pair_config = self.pair_configs.get(pair_id)
        if not pair_config:
            return False
        
        try:
            # Check for stop loss conditions
            stop_loss_z = pair_config.get('stop_loss_z', 3.0)
            
            # Get current z-score
            if 'zscore' in current_data.columns:
                current_z = current_data['zscore'].iloc[-1]
                
                # Check if z-score exceeds stop loss threshold
                if position['direction'] > 0 and current_z <= -stop_loss_z:
                    # Long position: stop if z-score drops too low
                    logger.info(f"Stop loss triggered for {pair_id}: zscore={current_z}, threshold=-{stop_loss_z}")
                    return self._exit_position(pair_id, "stop_loss")
                
                elif position['direction'] < 0 and current_z >= stop_loss_z:
                    # Short position: stop if z-score rises too high
                    logger.info(f"Stop loss triggered for {pair_id}: zscore={current_z}, threshold={stop_loss_z}")
                    return self._exit_position(pair_id, "stop_loss")
            
            # No stop loss triggered
            return False
            
        except Exception as e:
            logger.error(f"Error checking stop loss for {pair_id}: {str(e)}")
            return False
    
    def check_take_profits(
        self,
        pair_id: str,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if take profit conditions are met for a position.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        current_data : pd.DataFrame
            Current market data
            
        Returns:
        --------
        bool
            True if take profit triggered, False otherwise
        """
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            return False
        
        # Get pair config
        pair_config = self.pair_configs.get(pair_id)
        if not pair_config:
            return False
        
        try:
            # Get take profit parameters
            take_profit_z = pair_config.get('z_exit', 0.5)
            
            # Get current z-score
            if 'zscore' in current_data.columns:
                current_z = current_data['zscore'].iloc[-1]
                
                # Check if z-score crosses take profit threshold
                if position['direction'] > 0 and current_z <= take_profit_z:
                    # Long position: take profit when z-score drops below threshold
                    logger.info(f"Take profit triggered for {pair_id}: zscore={current_z}, threshold={take_profit_z}")
                    return self._exit_position(pair_id, "take_profit")
                
                elif position['direction'] < 0 and current_z >= -take_profit_z:
                    # Short position: take profit when z-score rises above threshold
                    logger.info(f"Take profit triggered for {pair_id}: zscore={current_z}, threshold=-{take_profit_z}")
                    return self._exit_position(pair_id, "take_profit")
            
            # No take profit triggered
            return False
            
        except Exception as e:
            logger.error(f"Error checking take profit for {pair_id}: {str(e)}")
            return False
    
    def check_holding_limits(self, pair_id: str) -> bool:
        """
        Check if maximum holding period is exceeded for a position.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
            
        Returns:
        --------
        bool
            True if holding limit triggered, False otherwise
        """
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            return False
        
        # Get pair config
        pair_config = self.pair_configs.get(pair_id)
        if not pair_config:
            return False
        
        try:
            # Get max holding period (in minutes)
            max_holding_period = pair_config.get('max_holding_period', 180)
            
            # Calculate holding time
            entry_time = position['entry_time']
            current_time = datetime.now()
            holding_time = (current_time - entry_time).total_seconds() / 60
            
            # Check if holding time exceeds maximum
            if holding_time >= max_holding_period:
                logger.info(f"Max holding period exceeded for {pair_id}: {holding_time:.1f} min > {max_holding_period} min")
                return self._exit_position(pair_id, "max_holding_period")
            
            # No holding limit triggered
            return False
            
        except Exception as e:
            logger.error(f"Error checking holding limit for {pair_id}: {str(e)}")
            return False
    
    def adjust_position_size(
        self,
        pair_id: str,
        market_conditions: Dict[str, Any]
    ) -> bool:
        """
        Adjust position size based on market conditions.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        market_conditions : Dict[str, Any]
            Current market conditions including volatility metrics
            
        Returns:
        --------
        bool
            Success flag
        """
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            logger.warning(f"No position found for {pair_id} to adjust")
            return False
        
        # Get pair config
        pair_config = self.pair_configs.get(pair_id)
        if not pair_config:
            logger.error(f"No configuration found for pair {pair_id}")
            return False
        
        try:
            # Extract market volatility metrics
            current_volatility = market_conditions.get('current_volatility')
            baseline_volatility = market_conditions.get('baseline_volatility')
            
            if not current_volatility or not baseline_volatility:
                logger.warning(f"Missing volatility metrics for position adjustment")
                return False
            
            # Calculate volatility ratio
            vol_ratio = current_volatility / baseline_volatility
            
            # Determine if adjustment is needed
            if vol_ratio > 1.5:  # If current volatility is 50% higher than baseline
                # Calculate adjustment factor (inverse relationship with volatility)
                adjustment_factor = 1 / vol_ratio
                
                # Get original size parameters
                original_size = position.get('original_size', 1.0)
                current_size = position.get('size', original_size)
                
                # Calculate new size
                new_size = original_size * adjustment_factor
                
                # Don't adjust if change is minimal
                if abs(new_size - current_size) / current_size < 0.1:  # Less than 10% change
                    return False
                    
                logger.info(f"Adjusting position size for {pair_id}: {current_size:.2f} -> {new_size:.2f} " +
                           f"(vol_ratio: {vol_ratio:.2f})")
                
                # Update position record
                position['size'] = new_size
                position['size_adjustments'] = position.get('size_adjustments', []) + [
                    {
                        'timestamp': datetime.now(),
                        'old_size': current_size,
                        'new_size': new_size,
                        'reason': 'volatility',
                        'vol_ratio': vol_ratio
                    }
                ]
                
                # TODO: Implement actual position size adjustment
                # This would require closing part of the position or adding to it
                
                return True
                
            return False  # No adjustment needed
            
        except Exception as e:
            logger.error(f"Error adjusting position size for {pair_id}: {str(e)}")
            return False

    def track_position_performance(
        self,
        pair_id: str,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Track and update performance metrics for an open position.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        current_data : pd.DataFrame
            Current market data
            
        Returns:
        --------
        Dict[str, Any]
            Updated performance metrics
        """
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            return {}
        
        try:
            # Get leg details
            leg1 = position['leg1']
            leg2 = position['leg2']
            
            # Get current leg positions from paper trader
            leg1_position = self.paper_trader.get_position(leg1['symbol'])
            leg2_position = self.paper_trader.get_position(leg2['symbol'])
            
            if not leg1_position or not leg2_position:
                logger.warning(f"Missing leg position data for {pair_id}")
                return {}
            
            # Calculate current P&L
            leg1_pnl = leg1_position.get('unrealized_pnl', 0.0)
            leg2_pnl = leg2_position.get('unrealized_pnl', 0.0)
            total_pnl = leg1_pnl + leg2_pnl
            
            # Calculate current spread value if available in data
            current_spread = None
            current_zscore = None
            if 'spread' in current_data.columns:
                current_spread = current_data['spread'].iloc[-1]
            if 'zscore' in current_data.columns:
                current_zscore = current_data['zscore'].iloc[-1]
            
            # Calculate holding time
            entry_time = position['entry_time']
            current_time = datetime.now()
            holding_time_minutes = (current_time - entry_time).total_seconds() / 60
            
            # Create performance metrics
            metrics = {
                'pair_id': pair_id,
                'timestamp': current_time,
                'leg1_pnl': leg1_pnl,
                'leg2_pnl': leg2_pnl,
                'total_pnl': total_pnl,
                'holding_time_minutes': holding_time_minutes,
                'current_spread': current_spread,
                'current_zscore': current_zscore
            }
            
            # Calculate additional metrics
            if 'entry_spread' in position:
                metrics['spread_change'] = current_spread - position['entry_spread'] if current_spread else None
                
            if 'entry_zscore' in position:
                metrics['zscore_change'] = current_zscore - position['entry_zscore'] if current_zscore else None
            
            # Update position record with latest metrics
            position['current_pnl'] = total_pnl
            position['current_spread'] = current_spread
            position['current_zscore'] = current_zscore
            position['holding_time_minutes'] = holding_time_minutes
            
            # Add to performance history
            if 'performance_history' not in position:
                position['performance_history'] = []
                
            position['performance_history'].append(metrics)
            
            # Limit history size
            max_history_points = 100
            if len(position['performance_history']) > max_history_points:
                position['performance_history'] = position['performance_history'][-max_history_points:]
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking position performance for {pair_id}: {str(e)}")
            return {}

    def analyze_position_risk(
        self,
        pair_id: str,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze risk metrics for an open position.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        current_data : pd.DataFrame
            Current market data
            
        Returns:
        --------
        Dict[str, Any]
            Risk metrics
        """
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            return {}
        
        # Get pair config
        pair_config = self.pair_configs.get(pair_id)
        if not pair_config:
            return {}
        
        try:
            # Extract relevant metrics
            direction = position['direction']
            entry_zscore = position.get('entry_zscore')
            current_zscore = None
            
            if 'zscore' in current_data.columns:
                current_zscore = current_data['zscore'].iloc[-1]
            
            # Get risk thresholds
            stop_loss_z = pair_config.get('stop_loss_z', 3.0)
            take_profit_z = pair_config.get('z_exit', 0.5)
            
            # Initialize risk metrics
            risk_metrics = {
                'pair_id': pair_id,
                'timestamp': datetime.now(),
                'direction': direction,
                'current_zscore': current_zscore,
                'distance_to_stop_loss': None,
                'distance_to_take_profit': None,
                'risk_reward_ratio': None,
                'time_decay_factor': None
            }
            
            # Calculate distance to stop loss and take profit
            if current_zscore is not None:
                if direction > 0:  # Long position
                    risk_metrics['distance_to_stop_loss'] = abs(current_zscore - (-stop_loss_z))
                    risk_metrics['distance_to_take_profit'] = abs(current_zscore - take_profit_z)
                else:  # Short position
                    risk_metrics['distance_to_stop_loss'] = abs(current_zscore - stop_loss_z)
                    risk_metrics['distance_to_take_profit'] = abs(current_zscore - (-take_profit_z))
                
                # Calculate risk-reward ratio
                if risk_metrics['distance_to_stop_loss'] > 0:
                    risk_metrics['risk_reward_ratio'] = (
                        risk_metrics['distance_to_take_profit'] / risk_metrics['distance_to_stop_loss']
                    )
            
            # Calculate time decay factor based on half-life
            half_life = pair_config.get('half_life', 24)  # In hours
            if half_life and 'holding_time_minutes' in position:
                holding_time_hours = position['holding_time_minutes'] / 60
                time_decay_factor = np.exp(-np.log(2) * holding_time_hours / half_life)
                risk_metrics['time_decay_factor'] = time_decay_factor
            
            # Update position with risk metrics
            position['risk_metrics'] = risk_metrics
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing position risk for {pair_id}: {str(e)}")
            return {}

    def check_correlation_breakdown(
        self,
        pair_id: str,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if correlation between pair legs has broken down.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        current_data : pd.DataFrame
            Current market data
            
        Returns:
        --------
        bool
            True if correlation breakdown detected, False otherwise
        """
        # Get current position
        position = self.positions.get(pair_id)
        if not position:
            return False
        
        # Get pair config
        pair_config = self.pair_configs.get(pair_id)
        if not pair_config:
            return False
        
        try:
            # Extract leg symbols
            leg1_symbol = position['leg1']['symbol']
            leg2_symbol = position['leg2']['symbol']
            
            # Get correlation threshold
            min_correlation = pair_config.get('min_correlation', 0.5)
            
            # Check if both leg price data is available
            if leg1_symbol not in current_data.columns or leg2_symbol not in current_data.columns:
                logger.warning(f"Missing leg price data for correlation check on {pair_id}")
                return False
            
            # Calculate recent correlation (last 30 periods)
            window_size = min(30, len(current_data) - 1)
            if window_size < 10:
                return False  # Not enough data points
            
            recent_data = current_data.iloc[-window_size:]
            correlation = recent_data[leg1_symbol].corr(recent_data[leg2_symbol])
            
            # Update position with current correlation
            position['current_correlation'] = correlation
            
            # Check for correlation breakdown
            if correlation < min_correlation:
                logger.warning(f"Correlation breakdown detected for {pair_id}: {correlation:.2f} < {min_correlation:.2f}")
                return self._exit_position(pair_id, "correlation_breakdown")
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking correlation for {pair_id}: {str(e)}")
            return False

    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all current positions.
        
        Returns:
        --------
        Dict[str, Any]
            Position summary
        """
        summary = {
            'timestamp': datetime.now(),
            'position_count': len(self.positions),
            'total_exposure': 0.0,
            'unrealized_pnl': 0.0,
            'positions': {}
        }
        
        # Calculate aggregate metrics
        for pair_id, position in self.positions.items():
            # Add individual position summary
            summary['positions'][pair_id] = {
                'direction': position['direction'],
                'entry_time': position['entry_time'],
                'holding_time_minutes': position.get('holding_time_minutes', 0),
                'unrealized_pnl': position.get('current_pnl', 0.0)
            }
            
            # Add to totals
            summary['unrealized_pnl'] += position.get('current_pnl', 0.0)
            
            # Calculate exposure (simplified)
            leg1_exposure = abs(position['leg1'].get('quantity', 0) * position['leg1'].get('price', 0))
            leg2_exposure = abs(position['leg2'].get('quantity', 0) * position['leg2'].get('price', 0))
            position_exposure = leg1_exposure + leg2_exposure
            
            summary['total_exposure'] += position_exposure
            summary['positions'][pair_id]['exposure'] = position_exposure
        
        return summary

    def monitor_all_positions(
        self,
        current_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Monitor all open positions and apply risk management rules.
        
        Parameters:
        -----------
        current_data : Dict[str, pd.DataFrame]
            Dictionary of current market data for each pair
            
        Returns:
        --------
        Dict[str, Any]
            Monitoring results
        """
        results = {
            'timestamp': datetime.now(),
            'total_positions': len(self.positions),
            'positions_checked': 0,
            'stop_losses_triggered': 0,
            'take_profits_triggered': 0,
            'holding_limits_triggered': 0,
            'correlation_breakdowns': 0,
            'position_metrics': {}
        }
        
        # Check each position
        for pair_id in list(self.positions.keys()):  # Create a copy of keys for safe iteration
            # Skip if position was already closed
            if pair_id not in self.positions:
                continue
            
            # Get data for this pair
            pair_data = current_data.get(pair_id)
            if pair_data is None or pair_data.empty:
                logger.warning(f"No data available for {pair_id}")
                continue
            
            results['positions_checked'] += 1
            position_result = {'pair_id': pair_id, 'checks': {}}
            
            # Track performance
            metrics = self.track_position_performance(pair_id, pair_data)
            position_result['performance'] = metrics
            
            # Analyze risk
            risk_metrics = self.analyze_position_risk(pair_id, pair_data)
            position_result['risk'] = risk_metrics
            
            # Apply checks - note that each check might close the position
            # Check stop losses
            stop_loss_triggered = self.check_stop_losses(pair_id, pair_data)
            position_result['checks']['stop_loss'] = stop_loss_triggered
            if stop_loss_triggered:
                results['stop_losses_triggered'] += 1
                continue  # Position closed, skip further checks
            
            # Check take profits
            take_profit_triggered = self.check_take_profits(pair_id, pair_data)
            position_result['checks']['take_profit'] = take_profit_triggered
            if take_profit_triggered:
                results['take_profits_triggered'] += 1
                continue  # Position closed, skip further checks
            
            # Check holding limits
            holding_limit_triggered = self.check_holding_limits(pair_id)
            position_result['checks']['holding_limit'] = holding_limit_triggered
            if holding_limit_triggered:
                results['holding_limits_triggered'] += 1
                continue  # Position closed, skip further checks
            
            # Check correlation breakdown
            correlation_breakdown = self.check_correlation_breakdown(pair_id, pair_data)
            position_result['checks']['correlation_breakdown'] = correlation_breakdown
            if correlation_breakdown:
                results['correlation_breakdowns'] += 1
                continue  # Position closed, skip further checks
            
            # Store position metrics
            results['position_metrics'][pair_id] = position_result
        
        return results 