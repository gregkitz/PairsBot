"""
Routes for the API blueprint.

This module defines API endpoints for external systems to interact
with the trading platform.
"""

import os
import json
import logging
from datetime import datetime
from functools import wraps
from flask import jsonify, request, current_app
from flask_login import current_user, login_required

from . import api
from ...utils import get_system_status, get_performance_data, get_active_positions
from ..strategy.routes import get_available_strategies, is_strategy_running

# Configure logging
logger = logging.getLogger(__name__)

# API key validation decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        # Load API keys from file
        api_keys_file = os.path.join(current_app.config['DATA_DIRECTORY'], 'web_users', 'api_keys.json')
        if not os.path.exists(api_keys_file):
            return jsonify({'error': 'API authentication not configured'}), 500
        
        with open(api_keys_file, 'r') as f:
            api_keys = json.load(f)
        
        if api_key not in api_keys:
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

@api.route('/status')
@require_api_key
def status():
    """Get system status."""
    try:
        return jsonify(get_system_status())
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/performance')
@require_api_key
def performance():
    """Get performance data."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        return jsonify(get_performance_data(start_date, end_date))
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/positions')
@require_api_key
def positions():
    """Get active positions."""
    try:
        return jsonify(get_active_positions())
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/strategies')
@require_api_key
def strategies():
    """Get available strategies."""
    try:
        # Add running status to strategies
        strategies = get_available_strategies()
        for strategy in strategies:
            strategy['running'] = is_strategy_running(strategy['id'])
        
        return jsonify(strategies)
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/strategies/<strategy_id>/start', methods=['POST'])
@require_api_key
def start_strategy(strategy_id):
    """Start a trading strategy."""
    try:
        # Get trading mode
        mode = request.json.get('mode', 'paper')
        if mode not in ['paper', 'live']:
            return jsonify({'error': 'Invalid trading mode. Must be "paper" or "live"'}), 400
        
        # Get strategy configuration
        strategies = get_available_strategies()
        strategy_config = next((s for s in strategies if s['id'] == strategy_id), None)
        
        if not strategy_config:
            return jsonify({'error': f"Strategy '{strategy_id}' not found"}), 404
        
        # Check if strategy is already running
        if is_strategy_running(strategy_id):
            return jsonify({'error': f"Strategy '{strategy_config['name']}' is already running"}), 400
        
        # Start the strategy
        import subprocess
        cmd = [
            'python', 'run.py', 
            '--mode', mode,
            '--strategy', strategy_id
        ]
        
        # Start the process in the background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True
        )
        
        # Log the process start
        logger.info(f"API started strategy '{strategy_config['name']}' in {mode} mode (PID: {process.pid})")
        
        return jsonify({
            'success': True,
            'message': f"Strategy '{strategy_config['name']}' started in {mode} mode",
            'pid': process.pid
        })
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/strategies/<strategy_id>/stop', methods=['POST'])
@require_api_key
def stop_strategy(strategy_id):
    """Stop a running trading strategy."""
    try:
        # Get strategy configuration
        strategies = get_available_strategies()
        strategy_config = next((s for s in strategies if s['id'] == strategy_id), None)
        
        if not strategy_config:
            return jsonify({'error': f"Strategy '{strategy_id}' not found"}), 404
        
        # Check if strategy is running
        if not is_strategy_running(strategy_id):
            return jsonify({'error': f"Strategy '{strategy_config['name']}' is not running"}), 400
        
        # Find and stop the strategy process
        for process in get_running_processes():
            if strategy_id in process['cmdline']:
                # Kill the process
                import os
                import signal
                os.kill(process['pid'], signal.SIGTERM)
                
                # Log the process stop
                logger.info(f"API stopped strategy '{strategy_config['name']}' (PID: {process['pid']})")
                
                return jsonify({
                    'success': True,
                    'message': f"Strategy '{strategy_config['name']}' stopped"
                })
        
        return jsonify({'error': f"Could not find running process for strategy '{strategy_config['name']}'"}), 400
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/generate-api-key', methods=['POST'])
@login_required
def generate_api_key():
    """Generate a new API key."""
    # Only administrators can generate API keys
    if not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Generate a random API key
        import secrets
        api_key = secrets.token_hex(16)
        
        # Load existing API keys
        api_keys_file = os.path.join(current_app.config['DATA_DIRECTORY'], 'web_users', 'api_keys.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(api_keys_file), exist_ok=True)
        
        # Load existing keys or create empty dict
        if os.path.exists(api_keys_file):
            with open(api_keys_file, 'r') as f:
                try:
                    api_keys = json.load(f)
                except:
                    api_keys = {}
        else:
            api_keys = {}
        
        # Add new key
        api_keys[api_key] = {
            'created_by': current_user.username,
            'created_at': datetime.now().isoformat()
        }
        
        # Save API keys
        with open(api_keys_file, 'w') as f:
            json.dump(api_keys, f, indent=2)
        
        return jsonify({
            'success': True,
            'api_key': api_key
        })
    except Exception as e:
        logger.error(f"Error generating API key: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/revoke-api-key', methods=['POST'])
@login_required
def revoke_api_key():
    """Revoke an API key."""
    # Only administrators can revoke API keys
    if not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get API key to revoke
        api_key = request.json.get('api_key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 400
        
        # Load existing API keys
        api_keys_file = os.path.join(current_app.config['DATA_DIRECTORY'], 'web_users', 'api_keys.json')
        
        if not os.path.exists(api_keys_file):
            return jsonify({'error': 'No API keys found'}), 404
        
        with open(api_keys_file, 'r') as f:
            api_keys = json.load(f)
        
        # Check if key exists
        if api_key not in api_keys:
            return jsonify({'error': 'API key not found'}), 404
        
        # Remove key
        del api_keys[api_key]
        
        # Save API keys
        with open(api_keys_file, 'w') as f:
            json.dump(api_keys, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'API key revoked'
        })
    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_running_processes():
    """
    Get running Python processes.
    
    This is duplicated from strategy/routes.py to avoid circular imports.
    
    Returns:
    --------
    list
        List of running Python processes
    """
    try:
        import psutil
        
        processes = []
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Filter for Python processes related to our strategies
                if process.info['name'] == 'python' and process.info['cmdline'] and 'run.py' in ' '.join(process.info['cmdline']):
                    processes.append({
                        'pid': process.info['pid'],
                        'cmdline': ' '.join(process.info['cmdline'])
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return processes
    except Exception as e:
        logger.error(f"Error getting running processes: {str(e)}")
        return [] 