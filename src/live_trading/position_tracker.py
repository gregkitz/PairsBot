"""
Position Tracking module for the Intraday Statistical Arbitrage System.

This module provides functionality for tracking and managing positions
in the live trading environment, including P&L calculation, position
lifecycle management, and reporting.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Position tracking system that provides:
    - Position lifecycle management
    - P&L calculation and tracking
    - Trade history recording
    - Position status reporting
    """
    
    def __init__(self, 
                data_directory: Optional[str] = None,
                commission_model: Dict = None,
                slippage_model: Dict = None,
                debug_mode: bool = False):
        """
        Initialize the position tracker.
        
        Parameters:
        -----------
        data_directory : str, optional
            Directory to store position data. If None, a default directory will be created.
        commission_model : dict, optional
            Settings for commission calculation (e.g., per_contract, per_order, percent)
        slippage_model : dict, optional
            Settings for slippage modeling (e.g., fixed_ticks, percent, market_impact)
        debug_mode : bool
            Whether to run in debug mode with additional logging
        """
        # Set data directory
        if data_directory is None:
            self.data_directory = os.path.join(os.getcwd(), 'position_data')
        else:
            self.data_directory = data_directory
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Set commission model with defaults
        self.commission_model = {
            'per_contract': 0.85,  # $0.85 per contract
            'per_order': 0.0,      # $0.00 per order
            'percent': 0.0,        # 0% of trade value
            'minimum': 0.0         # $0.00 minimum commission
        }
        if commission_model:
            self.commission_model.update(commission_model)
        
        # Set slippage model with defaults
        self.slippage_model = {
            'fixed_ticks': 1,      # 1 tick of slippage
            'percent': 0.0,        # 0% of price
            'market_impact': 0.0   # 0% market impact factor
        }
        if slippage_model:
            self.slippage_model.update(slippage_model)
        
        # Initialize position and trade storage
        self._active_positions = {}  # Current open positions
        self._closed_positions = []  # Historical closed positions
        self._trades = []            # Individual trades
        self._pairs_positions = {}   # Positions grouped by pair
        
        # Set debug mode
        self.debug_mode = debug_mode
        if debug_mode:
            logging.getLogger('src.live_trading.position_tracker').setLevel(logging.DEBUG)
        
        # Load existing data if available
        self._load_data()
    
    def create_position(self, 
                      symbol: str,
                      quantity: float,
                      entry_price: float,
                      entry_time: Optional[Union[str, datetime]] = None,
                      order_id: Optional[str] = None,
                      pair_id: Optional[str] = None,
                      position_type: str = 'single',
                      metadata: Optional[Dict] = None) -> str:
        """
        Create a new position.
        
        Parameters:
        -----------
        symbol : str
            The symbol of the position
        quantity : float
            The quantity of the position (positive for long, negative for short)
        entry_price : float
            The entry price of the position
        entry_time : str or datetime, optional
            The entry time of the position. If None, current time will be used.
        order_id : str, optional
            The order ID associated with the position
        pair_id : str, optional
            The pair ID if this position is part of a pair
        position_type : str
            The type of position ('single', 'pair_leg', 'spread')
        metadata : dict, optional
            Additional metadata for the position
        
        Returns:
        --------
        str
            The position ID
        """
        try:
            # Generate position ID
            position_id = f"POS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}"
            
            # Set entry time if not provided
            if entry_time is None:
                entry_time = datetime.now()
            elif isinstance(entry_time, str):
                # Parse ISO format string to datetime
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            
            # Format entry time as ISO string
            entry_time_str = entry_time.isoformat()
            
            # Create position object
            position = {
                'position_id': position_id,
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': entry_time_str,
                'order_id': order_id,
                'pair_id': pair_id,
                'position_type': position_type,
                'metadata': metadata or {},
                'status': 'open',
                'exit_price': None,
                'exit_time': None,
                'exit_order_id': None,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'commission': self._calculate_commission(quantity, entry_price),
                'slippage': self._calculate_slippage(quantity, entry_price),
                'total_cost': 0.0,  # Will be calculated below
                'net_pnl': 0.0,     # Will be calculated below
            }
            
            # Calculate total cost (entry cost + commission + slippage)
            position['total_cost'] = (
                abs(quantity) * entry_price + 
                position['commission'] + 
                position['slippage']
            )
            
            # Store position
            self._active_positions[position_id] = position
            
            # If this is part of a pair, update pairs positions
            if pair_id:
                if pair_id not in self._pairs_positions:
                    self._pairs_positions[pair_id] = {
                        'pair_id': pair_id,
                        'positions': [],
                        'total_cost': 0.0,
                        'realized_pnl': 0.0,
                        'unrealized_pnl': 0.0,
                        'net_pnl': 0.0,
                        'status': 'open',
                        'entry_time': entry_time_str
                    }
                
                self._pairs_positions[pair_id]['positions'].append(position_id)
                self._pairs_positions[pair_id]['total_cost'] += position['total_cost']
            
            # Record this as a trade
            trade = {
                'trade_id': f"TRADE_{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}",
                'position_id': position_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': entry_price,
                'time': entry_time_str,
                'order_id': order_id,
                'trade_type': 'entry',
                'commission': position['commission'],
                'slippage': position['slippage']
            }
            self._trades.append(trade)
            
            # Log position creation
            direction = 'LONG' if quantity > 0 else 'SHORT'
            logger.info(f"Created {direction} position {position_id} for {symbol}: {abs(quantity)} @ {entry_price}")
            
            # Save data
            self._save_data()
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error creating position for {symbol}: {str(e)}")
            return None
    
    def update_position(self, 
                      position_id: str,
                      current_price: Optional[float] = None,
                      current_time: Optional[Union[str, datetime]] = None,
                      metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing position with current market data.
        
        Parameters:
        -----------
        position_id : str
            The ID of the position to update
        current_price : float, optional
            The current market price to update the position with
        current_time : str or datetime, optional
            The current time. If None, current time will be used.
        metadata : dict, optional
            Additional metadata to update
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Check if position exists
            if position_id not in self._active_positions:
                logger.warning(f"Cannot update non-existent position: {position_id}")
                return False
            
            # Get position
            position = self._active_positions[position_id]
            
            # Set current time if not provided
            if current_time is None:
                current_time = datetime.now()
            elif isinstance(current_time, str):
                # Parse ISO format string to datetime
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            
            # Format current time as ISO string
            current_time_str = current_time.isoformat()
            
            # Update position with current price if provided
            if current_price is not None:
                # Calculate unrealized P&L
                position['unrealized_pnl'] = self._calculate_unrealized_pnl(
                    position['quantity'], 
                    position['entry_price'], 
                    current_price
                )
                
                # Calculate net P&L (unrealized P&L - commission - slippage)
                position['net_pnl'] = position['unrealized_pnl'] - position['commission'] - position['slippage']
                
                # If this is part of a pair, update pair unrealized P&L
                if position['pair_id']:
                    pair_id = position['pair_id']
                    if pair_id in self._pairs_positions:
                        # Recalculate entire pair's unrealized P&L
                        self._update_pair_unrealized_pnl(pair_id)
            
            # Update metadata if provided
            if metadata:
                position['metadata'].update(metadata)
            
            # Save data
            self._save_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {str(e)}")
            return False
    
    def close_position(self, 
                     position_id: str,
                     exit_price: float,
                     exit_time: Optional[Union[str, datetime]] = None,
                     exit_order_id: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Close an existing position.
        
        Parameters:
        -----------
        position_id : str
            The ID of the position to close
        exit_price : float
            The exit price of the position
        exit_time : str or datetime, optional
            The exit time of the position. If None, current time will be used.
        exit_order_id : str, optional
            The order ID associated with the exit
        metadata : dict, optional
            Additional metadata to update
        
        Returns:
        --------
        dict
            The closed position details
        """
        try:
            # Check if position exists
            if position_id not in self._active_positions:
                logger.warning(f"Cannot close non-existent position: {position_id}")
                return None
            
            # Get position
            position = self._active_positions[position_id]
            
            # Set exit time if not provided
            if exit_time is None:
                exit_time = datetime.now()
            elif isinstance(exit_time, str):
                # Parse ISO format string to datetime
                exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
            
            # Format exit time as ISO string
            exit_time_str = exit_time.isoformat()
            
            # Update position with exit information
            position['exit_price'] = exit_price
            position['exit_time'] = exit_time_str
            position['exit_order_id'] = exit_order_id
            position['status'] = 'closed'
            
            # Update metadata if provided
            if metadata:
                position['metadata'].update(metadata)
            
            # Calculate realized P&L
            position['realized_pnl'] = self._calculate_realized_pnl(
                position['quantity'], 
                position['entry_price'], 
                exit_price
            )
            
            # Calculate net P&L (realized P&L - commission - slippage)
            position['net_pnl'] = position['realized_pnl'] - position['commission'] - position['slippage']
            
            # Clear unrealized P&L
            position['unrealized_pnl'] = 0.0
            
            # If this is part of a pair, update pair P&L
            if position['pair_id']:
                pair_id = position['pair_id']
                if pair_id in self._pairs_positions:
                    self._update_pair_realized_pnl(pair_id)
                    
                    # Check if all positions in the pair are closed
                    all_closed = True
                    for pos_id in self._pairs_positions[pair_id]['positions']:
                        if self._active_positions.get(pos_id, {}).get('status') != 'closed':
                            all_closed = False
                            break
                    
                    if all_closed:
                        self._pairs_positions[pair_id]['status'] = 'closed'
                        self._pairs_positions[pair_id]['exit_time'] = exit_time_str
            
            # Record this as a trade
            trade = {
                'trade_id': f"TRADE_{datetime.now().strftime('%Y%m%d%H%M%S')}_{position['symbol']}",
                'position_id': position_id,
                'symbol': position['symbol'],
                'quantity': -position['quantity'],  # Opposite sign for exit
                'price': exit_price,
                'time': exit_time_str,
                'order_id': exit_order_id,
                'trade_type': 'exit',
                'commission': position['commission'],  # Using same commission for simplicity
                'slippage': position['slippage']       # Using same slippage for simplicity
            }
            self._trades.append(trade)
            
            # Move position to closed positions
            self._closed_positions.append(position)
            del self._active_positions[position_id]
            
            # Log position closure
            direction = 'LONG' if position['quantity'] > 0 else 'SHORT'
            logger.info(f"Closed {direction} position {position_id} for {position['symbol']}: "
                       f"{abs(position['quantity'])} @ {exit_price}, "
                       f"P&L: ${position['realized_pnl']:.2f}, "
                       f"Net P&L: ${position['net_pnl']:.2f}")
            
            # Save data
            self._save_data()
            
            return position
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {str(e)}")
            return None
    
    def create_pair_position(self, 
                           pair_id: str,
                           leg1_symbol: str,
                           leg1_quantity: float,
                           leg1_price: float,
                           leg2_symbol: str,
                           leg2_quantity: float,
                           leg2_price: float,
                           entry_time: Optional[Union[str, datetime]] = None,
                           leg1_order_id: Optional[str] = None,
                           leg2_order_id: Optional[str] = None,
                           metadata: Optional[Dict] = None) -> str:
        """
        Create a new pair position with two legs.
        
        Parameters:
        -----------
        pair_id : str
            The ID for the pair
        leg1_symbol : str
            The symbol of the first leg
        leg1_quantity : float
            The quantity of the first leg
        leg1_price : float
            The entry price of the first leg
        leg2_symbol : str
            The symbol of the second leg
        leg2_quantity : float
            The quantity of the second leg
        leg2_price : float
            The entry price of the second leg
        entry_time : str or datetime, optional
            The entry time of the position
        leg1_order_id : str, optional
            The order ID for the first leg
        leg2_order_id : str, optional
            The order ID for the second leg
        metadata : dict, optional
            Additional metadata for the pair
        
        Returns:
        --------
        str
            The pair ID
        """
        try:
            # Set entry time if not provided
            if entry_time is None:
                entry_time = datetime.now()
            elif isinstance(entry_time, str):
                # Parse ISO format string to datetime
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            
            # Create the pair entry in pairs_positions if it doesn't exist
            if pair_id not in self._pairs_positions:
                self._pairs_positions[pair_id] = {
                    'pair_id': pair_id,
                    'positions': [],
                    'total_cost': 0.0,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0,
                    'net_pnl': 0.0,
                    'status': 'open',
                    'entry_time': entry_time.isoformat(),
                    'exit_time': None,
                    'metadata': metadata or {}
                }
            
            # Create metadata for each leg
            leg1_metadata = {
                'leg': 'leg1',
                'pair_symbol': f"{leg1_symbol}/{leg2_symbol}",
                'is_long_leg': leg1_quantity > 0,
                'is_short_leg': leg1_quantity < 0
            }
            if metadata:
                leg1_metadata.update(metadata)
            
            leg2_metadata = {
                'leg': 'leg2',
                'pair_symbol': f"{leg1_symbol}/{leg2_symbol}",
                'is_long_leg': leg2_quantity > 0,
                'is_short_leg': leg2_quantity < 0
            }
            if metadata:
                leg2_metadata.update(metadata)
            
            # Create positions for each leg
            leg1_position_id = self.create_position(
                symbol=leg1_symbol,
                quantity=leg1_quantity,
                entry_price=leg1_price,
                entry_time=entry_time,
                order_id=leg1_order_id,
                pair_id=pair_id,
                position_type='pair_leg',
                metadata=leg1_metadata
            )
            
            leg2_position_id = self.create_position(
                symbol=leg2_symbol,
                quantity=leg2_quantity,
                entry_price=leg2_price,
                entry_time=entry_time,
                order_id=leg2_order_id,
                pair_id=pair_id,
                position_type='pair_leg',
                metadata=leg2_metadata
            )
            
            # Log pair creation
            logger.info(f"Created pair position {pair_id}: "
                       f"Leg1: {leg1_symbol} {leg1_quantity} @ {leg1_price}, "
                       f"Leg2: {leg2_symbol} {leg2_quantity} @ {leg2_price}")
            
            return pair_id
            
        except Exception as e:
            logger.error(f"Error creating pair position {pair_id}: {str(e)}")
            return None
    
    def close_pair_position(self, 
                          pair_id: str,
                          leg1_exit_price: float,
                          leg2_exit_price: float,
                          exit_time: Optional[Union[str, datetime]] = None,
                          leg1_exit_order_id: Optional[str] = None,
                          leg2_exit_order_id: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> Dict:
        """
        Close a pair position by closing both legs.
        
        Parameters:
        -----------
        pair_id : str
            The ID of the pair to close
        leg1_exit_price : float
            The exit price for the first leg
        leg2_exit_price : float
            The exit price for the second leg
        exit_time : str or datetime, optional
            The exit time of the position
        leg1_exit_order_id : str, optional
            The order ID for the first leg exit
        leg2_exit_order_id : str, optional
            The order ID for the second leg exit
        metadata : dict, optional
            Additional metadata for the pair closure
        
        Returns:
        --------
        dict
            The closed pair details
        """
        try:
            # Check if pair exists
            if pair_id not in self._pairs_positions:
                logger.warning(f"Cannot close non-existent pair: {pair_id}")
                return None
            
            # Get pair
            pair = self._pairs_positions[pair_id]
            
            # Check if pair is already closed
            if pair['status'] == 'closed':
                logger.warning(f"Pair {pair_id} is already closed")
                return pair
            
            # Set exit time if not provided
            if exit_time is None:
                exit_time = datetime.now()
            elif isinstance(exit_time, str):
                # Parse ISO format string to datetime
                exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
            
            # Get leg positions
            leg_positions = []
            for position_id in pair['positions']:
                # Position might be in active or closed positions
                if position_id in self._active_positions:
                    leg_positions.append(self._active_positions[position_id])
                else:
                    # Check in closed positions
                    for pos in self._closed_positions:
                        if pos['position_id'] == position_id:
                            leg_positions.append(pos)
                            break
            
            if len(leg_positions) != 2:
                logger.warning(f"Pair {pair_id} doesn't have exactly 2 legs")
                return None
            
            # Identify leg1 and leg2
            leg1 = None
            leg2 = None
            for pos in leg_positions:
                if pos['metadata'].get('leg') == 'leg1':
                    leg1 = pos
                elif pos['metadata'].get('leg') == 'leg2':
                    leg2 = pos
            
            if not leg1 or not leg2:
                logger.warning(f"Could not identify leg1 and leg2 for pair {pair_id}")
                return None
            
            # Close each leg if still open
            closed_positions = []
            
            if leg1['status'] == 'open':
                closed_leg1 = self.close_position(
                    position_id=leg1['position_id'],
                    exit_price=leg1_exit_price,
                    exit_time=exit_time,
                    exit_order_id=leg1_exit_order_id
                )
                closed_positions.append(closed_leg1)
            else:
                closed_positions.append(leg1)
            
            if leg2['status'] == 'open':
                closed_leg2 = self.close_position(
                    position_id=leg2['position_id'],
                    exit_price=leg2_exit_price,
                    exit_time=exit_time,
                    exit_order_id=leg2_exit_order_id
                )
                closed_positions.append(closed_leg2)
            else:
                closed_positions.append(leg2)
            
            # Update pair status and exit time
            pair['status'] = 'closed'
            pair['exit_time'] = exit_time.isoformat()
            
            # Update pair metadata if provided
            if metadata:
                if 'metadata' not in pair:
                    pair['metadata'] = {}
                pair['metadata'].update(metadata)
            
            # Log pair closure
            logger.info(f"Closed pair position {pair_id}: "
                       f"Total P&L: ${pair['realized_pnl']:.2f}, "
                       f"Net P&L: ${pair['net_pnl']:.2f}")
            
            # Save data
            self._save_data()
            
            return pair
            
        except Exception as e:
            logger.error(f"Error closing pair position {pair_id}: {str(e)}")
            return None
    
    def get_active_positions(self) -> Dict[str, Dict]:
        """
        Get all active positions.
        
        Returns:
        --------
        dict
            Dictionary of active positions by position ID
        """
        return self._active_positions
    
    def get_closed_positions(self) -> List[Dict]:
        """
        Get all closed positions.
        
        Returns:
        --------
        list
            List of closed position dictionaries
        """
        return self._closed_positions
    
    def get_trades(self) -> List[Dict]:
        """
        Get all recorded trades.
        
        Returns:
        --------
        list
            List of trade dictionaries
        """
        return self._trades
    
    def get_pairs_positions(self) -> Dict[str, Dict]:
        """
        Get all pair positions.
        
        Returns:
        --------
        dict
            Dictionary of pair positions by pair ID
        """
        return self._pairs_positions
    
    def get_position(self, position_id: str) -> Optional[Dict]:
        """
        Get a specific position by ID.
        
        Parameters:
        -----------
        position_id : str
            The ID of the position to get
        
        Returns:
        --------
        dict or None
            The position details, or None if not found
        """
        # Check active positions
        if position_id in self._active_positions:
            return self._active_positions[position_id]
        
        # Check closed positions
        for position in self._closed_positions:
            if position['position_id'] == position_id:
                return position
        
        # Not found
        return None
    
    def get_pair_position(self, pair_id: str) -> Optional[Dict]:
        """
        Get a specific pair position by ID.
        
        Parameters:
        -----------
        pair_id : str
            The ID of the pair to get
        
        Returns:
        --------
        dict or None
            The pair details, or None if not found
        """
        return self._pairs_positions.get(pair_id)
    
    def get_position_by_symbol(self, symbol: str) -> List[Dict]:
        """
        Get all positions for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to search for
        
        Returns:
        --------
        list
            List of positions for the symbol
        """
        positions = []
        
        # Check active positions
        for pos in self._active_positions.values():
            if pos['symbol'] == symbol:
                positions.append(pos)
        
        # Check closed positions
        for pos in self._closed_positions:
            if pos['symbol'] == symbol:
                positions.append(pos)
        
        return positions
    
    def get_daily_pnl(self, date: Optional[str] = None) -> float:
        """
        Get the total P&L for a specific date.
        
        Parameters:
        -----------
        date : str, optional
            The date to get P&L for (format: 'YYYY-MM-DD'). If None, use today.
        
        Returns:
        --------
        float
            Total P&L for the date
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Sum P&L from all trades on this date
        pnl = 0.0
        
        # We need to consider both entry and exit dates for accurate P&L
        for trade in self._trades:
            trade_date = trade['time'].split('T')[0]  # Extract date part
            
            if trade_date == date:
                # For entry trades, this is unrealized P&L or cost
                if trade['trade_type'] == 'entry':
                    # This is a cost (negative P&L temporarily)
                    pnl -= abs(trade['quantity']) * trade['price']
                
                # For exit trades, this is realized P&L
                elif trade['trade_type'] == 'exit':
                    # This is revenue (positive P&L)
                    pnl += abs(trade['quantity']) * trade['price']
        
        return pnl
    
    def get_performance_summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        Get a performance summary for a date range.
        
        Parameters:
        -----------
        start_date : str, optional
            The start date (format: 'YYYY-MM-DD'). If None, use all history.
        end_date : str, optional
            The end date (format: 'YYYY-MM-DD'). If None, use today.
        
        Returns:
        --------
        dict
            Performance summary statistics
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Filter closed positions by date range
        positions = []
        
        for pos in self._closed_positions:
            # Extract date from exit_time
            if pos['exit_time']:
                pos_date = pos['exit_time'].split('T')[0]  # Extract date part
                
                # Check if within date range
                if (start_date is None or pos_date >= start_date) and pos_date <= end_date:
                    positions.append(pos)
        
        # Calculate summary statistics
        if not positions:
            return {
                'start_date': start_date,
                'end_date': end_date,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'breakeven_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_net_pnl': 0.0,
                'total_commissions': 0.0,
                'total_slippage': 0.0,
                'avg_pnl_per_trade': 0.0,
                'avg_net_pnl_per_trade': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        # Calculate statistics
        total_trades = len(positions)
        winning_trades = sum(1 for p in positions if p['realized_pnl'] > 0)
        losing_trades = sum(1 for p in positions if p['realized_pnl'] < 0)
        breakeven_trades = total_trades - winning_trades - losing_trades
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        total_pnl = sum(p['realized_pnl'] for p in positions)
        total_net_pnl = sum(p['net_pnl'] for p in positions)
        total_commissions = sum(p['commission'] for p in positions)
        total_slippage = sum(p['slippage'] for p in positions)
        
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        avg_net_pnl_per_trade = total_net_pnl / total_trades if total_trades > 0 else 0.0
        
        wins = [p['realized_pnl'] for p in positions if p['realized_pnl'] > 0]
        losses = [p['realized_pnl'] for p in positions if p['realized_pnl'] < 0]
        
        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        # Calculate max consecutive wins/losses
        results = [1 if p['realized_pnl'] > 0 else -1 for p in positions]
        max_consecutive_wins = self._calc_max_consecutive(results, 1)
        max_consecutive_losses = self._calc_max_consecutive(results, -1)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_net_pnl': total_net_pnl,
            'total_commissions': total_commissions,
            'total_slippage': total_slippage,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'avg_net_pnl_per_trade': avg_net_pnl_per_trade,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission cost for a trade."""
        # Per-contract commission
        per_contract = abs(quantity) * self.commission_model['per_contract']
        
        # Per-order commission
        per_order = self.commission_model['per_order']
        
        # Percentage commission
        percentage = abs(quantity) * price * self.commission_model['percent'] / 100.0
        
        # Calculate total commission
        commission = per_contract + per_order + percentage
        
        # Apply minimum commission if needed
        if commission < self.commission_model['minimum'] and self.commission_model['minimum'] > 0:
            commission = self.commission_model['minimum']
        
        return commission
    
    def _calculate_slippage(self, quantity: float, price: float) -> float:
        """Calculate slippage cost for a trade."""
        # Fixed ticks slippage (e.g., 1 tick per contract)
        # Assuming 1 tick is 0.01 for simplicity, should be adjusted per instrument
        fixed_ticks_slippage = abs(quantity) * self.slippage_model['fixed_ticks'] * 0.01
        
        # Percentage slippage
        percent_slippage = abs(quantity) * price * self.slippage_model['percent'] / 100.0
        
        # Market impact slippage (increases with order size)
        impact_factor = self.slippage_model['market_impact']
        market_impact = abs(quantity) * price * impact_factor * (abs(quantity) / 100.0) / 100.0
        
        # Total slippage
        return fixed_ticks_slippage + percent_slippage + market_impact
    
    def _calculate_unrealized_pnl(self, quantity: float, entry_price: float, current_price: float) -> float:
        """Calculate unrealized P&L for a position."""
        # For long positions: current_price - entry_price
        # For short positions: entry_price - current_price
        if quantity > 0:  # Long position
            return quantity * (current_price - entry_price)
        else:  # Short position
            return abs(quantity) * (entry_price - current_price)
    
    def _calculate_realized_pnl(self, quantity: float, entry_price: float, exit_price: float) -> float:
        """Calculate realized P&L for a closed position."""
        # For long positions: exit_price - entry_price
        # For short positions: entry_price - exit_price
        if quantity > 0:  # Long position
            return quantity * (exit_price - entry_price)
        else:  # Short position
            return abs(quantity) * (entry_price - exit_price)
    
    def _update_pair_unrealized_pnl(self, pair_id: str) -> None:
        """Update the unrealized P&L for a pair position."""
        if pair_id not in self._pairs_positions:
            return
        
        pair = self._pairs_positions[pair_id]
        total_unrealized_pnl = 0.0
        
        # Sum the unrealized P&L from all active positions in the pair
        for position_id in pair['positions']:
            if position_id in self._active_positions:
                total_unrealized_pnl += self._active_positions[position_id]['unrealized_pnl']
        
        # Update pair unrealized P&L
        pair['unrealized_pnl'] = total_unrealized_pnl
        
        # Update pair net P&L (realized + unrealized - total costs)
        pair['net_pnl'] = pair['realized_pnl'] + pair['unrealized_pnl']
    
    def _update_pair_realized_pnl(self, pair_id: str) -> None:
        """Update the realized P&L for a pair position."""
        if pair_id not in self._pairs_positions:
            return
        
        pair = self._pairs_positions[pair_id]
        total_realized_pnl = 0.0
        
        # Sum the realized P&L from all closed positions in the pair
        for position_id in pair['positions']:
            # Check closed positions
            for position in self._closed_positions:
                if position['position_id'] == position_id:
                    total_realized_pnl += position['realized_pnl']
                    break
        
        # Update pair realized P&L
        pair['realized_pnl'] = total_realized_pnl
        
        # Update pair net P&L (realized + unrealized - total costs)
        pair['net_pnl'] = pair['realized_pnl'] + pair['unrealized_pnl']
    
    def _calc_max_consecutive(self, results: List[int], value: int) -> int:
        """Calculate maximum consecutive occurrences of a value in a list."""
        max_count = 0
        current_count = 0
        
        for result in results:
            if result == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _save_data(self) -> None:
        """Save position and trade data to disk."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.data_directory, exist_ok=True)
            
            # Save active positions
            active_positions_file = os.path.join(self.data_directory, 'active_positions.json')
            with open(active_positions_file, 'w') as f:
                json.dump(self._active_positions, f, indent=2)
            
            # Save closed positions
            closed_positions_file = os.path.join(self.data_directory, 'closed_positions.json')
            with open(closed_positions_file, 'w') as f:
                json.dump(self._closed_positions, f, indent=2)
            
            # Save trades
            trades_file = os.path.join(self.data_directory, 'trades.json')
            with open(trades_file, 'w') as f:
                json.dump(self._trades, f, indent=2)
            
            # Save pair positions
            pairs_file = os.path.join(self.data_directory, 'pairs_positions.json')
            with open(pairs_file, 'w') as f:
                json.dump(self._pairs_positions, f, indent=2)
            
            # If in debug mode, log save operation
            if self.debug_mode:
                logger.debug(f"Position data saved to {self.data_directory}")
            
        except Exception as e:
            logger.error(f"Error saving position data: {str(e)}")
    
    def _load_data(self) -> None:
        """Load position and trade data from disk."""
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_directory):
                logger.info(f"Data directory {self.data_directory} does not exist. Creating...")
                os.makedirs(self.data_directory, exist_ok=True)
                return
            
            # Load active positions if file exists
            active_positions_file = os.path.join(self.data_directory, 'active_positions.json')
            if os.path.exists(active_positions_file):
                with open(active_positions_file, 'r') as f:
                    self._active_positions = json.load(f)
                logger.info(f"Loaded {len(self._active_positions)} active positions")
            
            # Load closed positions if file exists
            closed_positions_file = os.path.join(self.data_directory, 'closed_positions.json')
            if os.path.exists(closed_positions_file):
                with open(closed_positions_file, 'r') as f:
                    self._closed_positions = json.load(f)
                logger.info(f"Loaded {len(self._closed_positions)} closed positions")
            
            # Load trades if file exists
            trades_file = os.path.join(self.data_directory, 'trades.json')
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self._trades = json.load(f)
                logger.info(f"Loaded {len(self._trades)} trades")
            
            # Load pair positions if file exists
            pairs_file = os.path.join(self.data_directory, 'pairs_positions.json')
            if os.path.exists(pairs_file):
                with open(pairs_file, 'r') as f:
                    self._pairs_positions = json.load(f)
                logger.info(f"Loaded {len(self._pairs_positions)} pair positions")
            
        except Exception as e:
            logger.error(f"Error loading position data: {str(e)}")

# Example usage:
#
# # Create position tracker
# tracker = PositionTracker(
#     data_directory='./position_data',
#     commission_model={
#         'per_contract': 0.85,  # $0.85 per contract
#         'per_order': 0.0,      # $0.00 per order
#         'percent': 0.0,        # 0% of trade value
#         'minimum': 0.0         # $0.00 minimum commission
#     },
#     slippage_model={
#         'fixed_ticks': 1,      # 1 tick of slippage
#         'percent': 0.0,        # 0% of price
#         'market_impact': 0.0   # 0% market impact factor
#     }
# )
#
# # Create a position
# position_id = tracker.create_position(
#     symbol='ES',
#     quantity=1,
#     entry_price=4000.50,
#     entry_time=datetime.now()
# )
#
# # Update position with current price
# tracker.update_position(
#     position_id=position_id,
#     current_price=4010.25
# )
#
# # Close position
# tracker.close_position(
#     position_id=position_id,
#     exit_price=4010.25,
#     exit_time=datetime.now()
# )
#
# # Create a pair position
# pair_id = tracker.create_pair_position(
#     pair_id='GC_SI_PAIR',
#     leg1_symbol='GC',
#     leg1_quantity=1,
#     leg1_price=1800.50,
#     leg2_symbol='SI',
#     leg2_quantity=-10,  # Short 10 contracts
#     leg2_price=22.75,
#     entry_time=datetime.now()
# )
#
# # Get performance summary
# summary = tracker.get_performance_summary() 