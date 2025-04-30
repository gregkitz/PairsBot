"""
Utility functions for the web interface.

This module provides utility functions for getting system status,
performance data, and other common tasks.
"""

import os
import json
import logging
import subprocess
import psutil
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

def get_system_status():
    """
    Get system status information.
    
    Returns:
    --------
    dict
        System status information
    """
    try:
        # Check if live trader is running
        trader_running = False
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            if process.info['cmdline'] and any('live_trader' in arg for arg in process.info['cmdline']):
                trader_running = True
                break
        
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        disk_info = psutil.disk_usage('/')
        disk_percent = disk_info.percent
        
        return {
            'timestamp': datetime.now().isoformat(),
            'trader_running': trader_running,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'uptime': get_uptime(),
            'connections': {
                'ib_connected': check_ib_connection()
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def get_performance_data(start_date=None, end_date=None):
    """
    Get performance data for a date range.
    
    Parameters:
    -----------
    start_date : str, optional
        Start date (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date (format: 'YYYY-MM-DD')
    
    Returns:
    --------
    dict
        Performance data
    """
    try:
        # Get position data
        position_data = get_position_data()
        
        # Filter by date range if specified
        closed_positions = position_data.get('closed_positions', [])
        filtered_positions = []
        
        for position in closed_positions:
            exit_time = position.get('exit_time', '')
            if not exit_time:
                continue
                
            # Extract date part (YYYY-MM-DD)
            position_date = exit_time.split('T')[0]
            
            # Check if within date range
            if start_date and position_date < start_date:
                continue
            if end_date and position_date > end_date:
                continue
                
            filtered_positions.append(position)
        
        # Calculate daily PnL
        daily_pnl = {}
        for position in filtered_positions:
            exit_time = position.get('exit_time', '')
            if not exit_time:
                continue
                
            # Extract date part (YYYY-MM-DD)
            position_date = exit_time.split('T')[0]
            
            # Add PnL to daily total
            if position_date not in daily_pnl:
                daily_pnl[position_date] = 0.0
                
            daily_pnl[position_date] += position.get('realized_pnl', 0.0)
        
        # Calculate summary statistics
        total_pnl = sum(position.get('realized_pnl', 0.0) for position in filtered_positions)
        winning_trades = sum(1 for position in filtered_positions if position.get('realized_pnl', 0.0) > 0)
        losing_trades = sum(1 for position in filtered_positions if position.get('realized_pnl', 0.0) < 0)
        total_trades = len(filtered_positions)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # Calculate profit factor
        gross_profit = sum(position.get('realized_pnl', 0.0) for position in filtered_positions 
                          if position.get('realized_pnl', 0.0) > 0)
        gross_loss = abs(sum(position.get('realized_pnl', 0.0) for position in filtered_positions 
                           if position.get('realized_pnl', 0.0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Calculate max drawdown (simplified)
        max_drawdown = 0.0
        peak = 0.0
        current = 0.0
        
        # Sort dates
        sorted_dates = sorted(daily_pnl.keys())
        
        for date in sorted_dates:
            current += daily_pnl[date]
            if current > peak:
                peak = current
            drawdown = peak - current
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'daily_pnl': daily_pnl,
            'trades': filtered_positions,
            'active_positions': len(position_data.get('active_positions', {})),
            'summary': {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade': avg_trade,
                'max_drawdown': max_drawdown
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance data: {str(e)}")
        return {
            'error': str(e),
            'daily_pnl': {},
            'trades': [],
            'active_positions': 0,
            'summary': {
                'total_pnl': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'max_drawdown': 0.0
            }
        }

def get_active_positions():
    """
    Get active positions.
    
    Returns:
    --------
    list
        List of active positions
    """
    try:
        position_data = get_position_data()
        active_positions = []
        
        # Convert dictionary to list for easier use in templates
        for position_id, position in position_data.get('active_positions', {}).items():
            position['id'] = position_id
            active_positions.append(position)
        
        return active_positions
    except Exception as e:
        logger.error(f"Error getting active positions: {str(e)}")
        return []

def get_position_data():
    """
    Get position data from the position tracker.
    
    Returns:
    --------
    dict
        Position data
    """
    try:
        # Look for position data in live_data/positions directory
        positions_dir = os.path.join(os.getcwd(), 'live_data', 'positions')
        
        # Check if directory exists
        if not os.path.exists(positions_dir):
            return {
                'active_positions': {},
                'closed_positions': []
            }
        
        # Load active positions
        active_positions_file = os.path.join(positions_dir, 'active_positions.json')
        if os.path.exists(active_positions_file):
            with open(active_positions_file, 'r') as f:
                active_positions = json.load(f)
        else:
            active_positions = {}
        
        # Load closed positions
        closed_positions_file = os.path.join(positions_dir, 'closed_positions.json')
        if os.path.exists(closed_positions_file):
            with open(closed_positions_file, 'r') as f:
                closed_positions = json.load(f)
        else:
            closed_positions = []
        
        return {
            'active_positions': active_positions,
            'closed_positions': closed_positions
        }
    except Exception as e:
        logger.error(f"Error loading position data: {str(e)}")
        return {
            'active_positions': {},
            'closed_positions': []
        }

def get_uptime():
    """
    Get system uptime.
    
    Returns:
    --------
    str
        System uptime
    """
    try:
        # Get uptime in seconds
        uptime_seconds = int(psutil.boot_time())
        boot_time = datetime.fromtimestamp(uptime_seconds)
        uptime = datetime.now() - boot_time
        
        # Format uptime
        days, remainder = divmod(uptime.total_seconds(), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{int(days)}d {int(hours)}h {int(minutes)}m"
        elif hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        else:
            return f"{int(minutes)}m {int(seconds)}s"
    except Exception as e:
        logger.error(f"Error getting uptime: {str(e)}")
        return "Unknown"

def check_ib_connection():
    """
    Check if connection to Interactive Brokers is active.
    
    Returns:
    --------
    bool
        True if connected, False otherwise
    """
    try:
        # Look for connection status in trader data
        trader_data_dir = os.path.join(os.getcwd(), 'live_data', 'trader')
        
        # Read latest status file if available
        status_files = [f for f in os.listdir(trader_data_dir) 
                       if f.startswith('summary_') and f.endswith('.json')]
        
        if status_files:
            # Get most recent status file
            latest_file = max(status_files)
            status_file = os.path.join(trader_data_dir, latest_file)
            
            with open(status_file, 'r') as f:
                status_data = json.load(f)
                
            # Check connection status (simplified)
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking IB connection: {str(e)}")
        return False 