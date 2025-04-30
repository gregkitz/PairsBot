"""
Routes for the strategy blueprint.

This module defines the routes for managing trading strategies,
including starting, stopping, and configuring strategies.
"""

import os
import json
import logging
import subprocess
from flask import render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user

from . import strategy

# Configure logging
logger = logging.getLogger(__name__)

@strategy.route('/')
@login_required
def index():
    """Display the strategy management page."""
    strategies = get_available_strategies()
    return render_template('strategy/index.html', strategies=strategies)

@strategy.route('/start/<strategy_id>', methods=['POST'])
@login_required
def start(strategy_id):
    """Start a trading strategy."""
    if not current_user.is_admin():
        flash('You do not have permission to start strategies', 'danger')
        return redirect(url_for('strategy.index'))
    
    try:
        # Get strategy configuration
        strategies = get_available_strategies()
        strategy_config = next((s for s in strategies if s['id'] == strategy_id), None)
        
        if not strategy_config:
            flash(f"Strategy '{strategy_id}' not found", 'danger')
            return redirect(url_for('strategy.index'))
        
        # Check if strategy is already running
        if is_strategy_running(strategy_id):
            flash(f"Strategy '{strategy_config['name']}' is already running", 'warning')
            return redirect(url_for('strategy.index'))
        
        # Start the strategy
        mode = request.form.get('mode', 'paper')  # Default to paper trading
        if mode == 'live' and not current_user.is_admin():
            flash('Only administrators can start live trading', 'danger')
            return redirect(url_for('strategy.index'))
        
        # Build the command to start the strategy
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
        logger.info(f"Started strategy '{strategy_config['name']}' in {mode} mode (PID: {process.pid})")
        
        flash(f"Strategy '{strategy_config['name']}' started in {mode} mode", 'success')
    except Exception as e:
        logger.error(f"Error starting strategy '{strategy_id}': {str(e)}")
        flash(f"Error starting strategy: {str(e)}", 'danger')
    
    return redirect(url_for('strategy.index'))

@strategy.route('/stop/<strategy_id>', methods=['POST'])
@login_required
def stop(strategy_id):
    """Stop a running trading strategy."""
    if not current_user.is_admin():
        flash('You do not have permission to stop strategies', 'danger')
        return redirect(url_for('strategy.index'))
    
    try:
        # Get strategy configuration
        strategies = get_available_strategies()
        strategy_config = next((s for s in strategies if s['id'] == strategy_id), None)
        
        if not strategy_config:
            flash(f"Strategy '{strategy_id}' not found", 'danger')
            return redirect(url_for('strategy.index'))
        
        # Check if strategy is running
        if not is_strategy_running(strategy_id):
            flash(f"Strategy '{strategy_config['name']}' is not running", 'warning')
            return redirect(url_for('strategy.index'))
        
        # Find and stop the strategy process
        for process in get_running_processes():
            if strategy_id in process['cmdline']:
                # Kill the process
                import os
                import signal
                os.kill(process['pid'], signal.SIGTERM)
                
                # Log the process stop
                logger.info(f"Stopped strategy '{strategy_config['name']}' (PID: {process['pid']})")
                
                flash(f"Strategy '{strategy_config['name']}' stopped", 'success')
                return redirect(url_for('strategy.index'))
        
        flash(f"Could not find running process for strategy '{strategy_config['name']}'", 'warning')
    except Exception as e:
        logger.error(f"Error stopping strategy '{strategy_id}': {str(e)}")
        flash(f"Error stopping strategy: {str(e)}", 'danger')
    
    return redirect(url_for('strategy.index'))

@strategy.route('/edit/<strategy_id>', methods=['GET', 'POST'])
@login_required
def edit(strategy_id):
    """Edit a strategy configuration."""
    if not current_user.is_admin():
        flash('You do not have permission to edit strategy configurations', 'danger')
        return redirect(url_for('strategy.index'))
    
    # Get strategy configuration
    strategies = get_available_strategies()
    strategy_config = next((s for s in strategies if s['id'] == strategy_id), None)
    
    if not strategy_config:
        flash(f"Strategy '{strategy_id}' not found", 'danger')
        return redirect(url_for('strategy.index'))
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form.get('name')
            description = request.form.get('description')
            pairs = request.form.get('pairs')
            entry_threshold = float(request.form.get('entry_threshold'))
            exit_threshold = float(request.form.get('exit_threshold'))
            stop_loss_threshold = float(request.form.get('stop_loss_threshold'))
            max_holding_period = int(request.form.get('max_holding_period'))
            
            # Parse pairs
            pairs_list = []
            for pair_str in pairs.split('\n'):
                pair = pair_str.strip()
                if pair:
                    parts = pair.split(',')
                    if len(parts) >= 2:
                        leg1 = parts[0].strip()
                        leg2 = parts[1].strip()
                        ratio = float(parts[2].strip()) if len(parts) > 2 else 1.0
                        pairs_list.append({
                            'leg1': leg1,
                            'leg2': leg2,
                            'ratio': ratio
                        })
            
            # Update strategy configuration
            new_config = {
                'id': strategy_id,
                'name': name,
                'description': description,
                'pairs': pairs_list,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'stop_loss_threshold': stop_loss_threshold,
                'max_holding_period_minutes': max_holding_period,
                'type': strategy_config['type']
            }
            
            # Save configuration
            save_strategy_config(strategy_id, new_config)
            
            flash(f"Strategy '{name}' configuration updated", 'success')
            return redirect(url_for('strategy.index'))
        except Exception as e:
            logger.error(f"Error updating strategy configuration: {str(e)}")
            flash(f"Error updating strategy configuration: {str(e)}", 'danger')
    
    # Format pairs for display
    pairs_text = '\n'.join([f"{p['leg1']}, {p['leg2']}, {p['ratio']}" for p in strategy_config.get('pairs', [])])
    
    return render_template('strategy/edit.html', 
                          strategy=strategy_config,
                          pairs_text=pairs_text)

@strategy.route('/status/<strategy_id>')
@login_required
def status(strategy_id):
    """Get the status of a running strategy."""
    try:
        # Get strategy configuration
        strategies = get_available_strategies()
        strategy_config = next((s for s in strategies if s['id'] == strategy_id), None)
        
        if not strategy_config:
            return jsonify({'error': f"Strategy '{strategy_id}' not found"}), 404
        
        # Check if strategy is running
        running = is_strategy_running(strategy_id)
        
        # Get process information if running
        process_info = None
        if running:
            for process in get_running_processes():
                if strategy_id in process['cmdline']:
                    process_info = process
                    break
        
        return jsonify({
            'id': strategy_id,
            'name': strategy_config['name'],
            'running': running,
            'process': process_info
        })
    except Exception as e:
        logger.error(f"Error getting strategy status: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_available_strategies():
    """
    Get available trading strategies.
    
    Returns:
    --------
    list
        List of strategy configurations
    """
    try:
        config_dir = os.path.join(os.getcwd(), 'config', 'strategies')
        
        # Check if directory exists
        if not os.path.exists(config_dir):
            return []
        
        # Get all strategy configuration files
        strategy_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        strategies = []
        
        for file in strategy_files:
            try:
                with open(os.path.join(config_dir, file), 'r') as f:
                    config = json.load(f)
                    
                    # Add file name as ID if not present
                    if 'id' not in config:
                        config['id'] = os.path.splitext(file)[0]
                    
                    strategies.append(config)
            except Exception as e:
                logger.error(f"Error loading strategy configuration from {file}: {str(e)}")
        
        return strategies
    except Exception as e:
        logger.error(f"Error getting available strategies: {str(e)}")
        return []

def is_strategy_running(strategy_id):
    """
    Check if a strategy is currently running.
    
    Parameters:
    -----------
    strategy_id : str
        The strategy ID to check
    
    Returns:
    --------
    bool
        True if the strategy is running, False otherwise
    """
    try:
        for process in get_running_processes():
            if strategy_id in process['cmdline']:
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking if strategy is running: {str(e)}")
        return False

def get_running_processes():
    """
    Get running Python processes.
    
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

def save_strategy_config(strategy_id, config):
    """
    Save strategy configuration.
    
    Parameters:
    -----------
    strategy_id : str
        The strategy ID
    config : dict
        The strategy configuration
    """
    try:
        config_dir = os.path.join(os.getcwd(), 'config', 'strategies')
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configuration
        config_file = os.path.join(config_dir, f"{strategy_id}.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving strategy configuration: {str(e)}")
        raise 