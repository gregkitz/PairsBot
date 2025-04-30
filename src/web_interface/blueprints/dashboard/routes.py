"""
Routes for the dashboard blueprint.

This module defines the routes for the dashboard, including
the main dashboard, strategy performance, and system status.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import render_template, request, jsonify, current_app
from flask_login import login_required, current_user

from . import dashboard
from ...utils import get_system_status, get_performance_data, get_active_positions

# Configure logging
logger = logging.getLogger(__name__)

@dashboard.route('/')
@login_required
def index():
    """Display the main dashboard."""
    # Get system status
    system_status = get_system_status()
    
    # Get performance summary
    try:
        performance = get_performance_data()
    except Exception as e:
        logger.error(f"Error getting performance data: {str(e)}")
        performance = {
            'daily_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'active_positions': 0
        }
    
    # Get active positions
    try:
        positions = get_active_positions()
    except Exception as e:
        logger.error(f"Error getting active positions: {str(e)}")
        positions = []
    
    return render_template('dashboard/index.html',
                          system_status=system_status,
                          performance=performance,
                          positions=positions)

@dashboard.route('/performance')
@login_required
def performance():
    """Display detailed performance metrics."""
    # Get start and end dates from query parameters
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    start_date = request.args.get('start_date', 
                                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    
    # Get performance data
    try:
        performance_data = get_performance_data(start_date, end_date)
        
        # Get daily PnL
        daily_pnl = []
        for date, pnl in performance_data.get('daily_pnl', {}).items():
            daily_pnl.append({
                'date': date,
                'pnl': pnl
            })
        
        # Sort by date
        daily_pnl.sort(key=lambda x: x['date'])
        
        # Calculate cumulative PnL
        cumulative_pnl = []
        running_total = 0
        for day in daily_pnl:
            running_total += day['pnl']
            cumulative_pnl.append({
                'date': day['date'],
                'pnl': running_total
            })
        
        # Get trade statistics
        trades = performance_data.get('trades', [])
    except Exception as e:
        logger.error(f"Error getting detailed performance data: {str(e)}")
        daily_pnl = []
        cumulative_pnl = []
        trades = []
    
    return render_template('dashboard/performance.html',
                          start_date=start_date,
                          end_date=end_date,
                          daily_pnl=daily_pnl,
                          cumulative_pnl=cumulative_pnl,
                          trades=trades,
                          summary=performance_data.get('summary', {}))

@dashboard.route('/status')
@login_required
def status():
    """Display detailed system status."""
    # Get system status
    system_status = get_system_status()
    
    return render_template('dashboard/status.html',
                          system_status=system_status)

@dashboard.route('/api/status')
@login_required
def api_status():
    """API endpoint for system status data."""
    return jsonify(get_system_status())

@dashboard.route('/api/performance')
@login_required
def api_performance():
    """API endpoint for performance data."""
    # Get start and end dates from query parameters
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    start_date = request.args.get('start_date', 
                                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    
    try:
        return jsonify(get_performance_data(start_date, end_date))
    except Exception as e:
        logger.error(f"Error in API performance endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@dashboard.route('/api/positions')
@login_required
def api_positions():
    """API endpoint for active positions data."""
    try:
        return jsonify(get_active_positions())
    except Exception as e:
        logger.error(f"Error in API positions endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500 