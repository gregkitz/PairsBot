"""
Routes for the configuration blueprint.

This module defines the routes for system configuration management.
"""

import os
import json
import logging
from flask import render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user

from . import configuration

# Configure logging
logger = logging.getLogger(__name__)

@configuration.route('/')
@login_required
def index():
    """Display the configuration management page."""
    if not current_user.is_admin():
        flash('You do not have permission to access system configuration', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get configuration files
    configs = get_config_files()
    
    return render_template('configuration/index.html', configs=configs)

@configuration.route('/edit/<config_id>', methods=['GET', 'POST'])
@login_required
def edit(config_id):
    """Edit a configuration file."""
    if not current_user.is_admin():
        flash('You do not have permission to edit system configuration', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get configuration file
    configs = get_config_files()
    config = next((c for c in configs if c['id'] == config_id), None)
    
    if not config:
        flash(f"Configuration '{config_id}' not found", 'danger')
        return redirect(url_for('configuration.index'))
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Get form data
            content = request.form.get('content')
            
            # Parse JSON to validate
            json_content = json.loads(content)
            
            # Format JSON with indentation
            formatted_content = json.dumps(json_content, indent=2)
            
            # Save configuration
            config_file = os.path.join(os.getcwd(), 'config', f"{config_id}.json")
            with open(config_file, 'w') as f:
                f.write(formatted_content)
            
            flash(f"Configuration '{config['name']}' updated", 'success')
            return redirect(url_for('configuration.index'))
        except json.JSONDecodeError as e:
            flash(f"Invalid JSON format: {str(e)}", 'danger')
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            flash(f"Error updating configuration: {str(e)}", 'danger')
    
    return render_template('configuration/edit.html', config=config)

@configuration.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    """Create a new configuration file."""
    if not current_user.is_admin():
        flash('You do not have permission to create system configuration', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form.get('name')
            file_id = request.form.get('id')
            content = request.form.get('content', '{}')
            
            # Validate file ID
            if not file_id:
                flash("Configuration ID is required", 'danger')
                return render_template('configuration/create.html')
            
            # Check if file already exists
            config_file = os.path.join(os.getcwd(), 'config', f"{file_id}.json")
            if os.path.exists(config_file):
                flash(f"Configuration with ID '{file_id}' already exists", 'danger')
                return render_template('configuration/create.html')
            
            # Parse JSON to validate
            try:
                json_content = json.loads(content)
            except json.JSONDecodeError as e:
                flash(f"Invalid JSON format: {str(e)}", 'danger')
                return render_template('configuration/create.html')
            
            # Format JSON with indentation
            formatted_content = json.dumps(json_content, indent=2)
            
            # Save configuration
            with open(config_file, 'w') as f:
                f.write(formatted_content)
            
            flash(f"Configuration '{name}' created", 'success')
            return redirect(url_for('configuration.index'))
        except Exception as e:
            logger.error(f"Error creating configuration: {str(e)}")
            flash(f"Error creating configuration: {str(e)}", 'danger')
    
    return render_template('configuration/create.html')

@configuration.route('/delete/<config_id>', methods=['POST'])
@login_required
def delete(config_id):
    """Delete a configuration file."""
    if not current_user.is_admin():
        flash('You do not have permission to delete system configuration', 'danger')
        return redirect(url_for('dashboard.index'))
    
    try:
        # Get configuration file
        configs = get_config_files()
        config = next((c for c in configs if c['id'] == config_id), None)
        
        if not config:
            flash(f"Configuration '{config_id}' not found", 'danger')
            return redirect(url_for('configuration.index'))
        
        # Delete configuration file
        config_file = os.path.join(os.getcwd(), 'config', f"{config_id}.json")
        os.remove(config_file)
        
        flash(f"Configuration '{config['name']}' deleted", 'success')
    except Exception as e:
        logger.error(f"Error deleting configuration: {str(e)}")
        flash(f"Error deleting configuration: {str(e)}", 'danger')
    
    return redirect(url_for('configuration.index'))

def get_config_files():
    """
    Get available configuration files.
    
    Returns:
    --------
    list
        List of configuration file dictionaries
    """
    try:
        config_dir = os.path.join(os.getcwd(), 'config')
        
        # Check if directory exists
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
            return []
        
        # Get all JSON files in the config directory
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        configs = []
        
        for file in config_files:
            try:
                # Get file path
                file_path = os.path.join(config_dir, file)
                
                # Read file content
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Try to parse as JSON
                try:
                    json_content = json.loads(content)
                    
                    # Get name from content or use filename
                    name = json_content.get('name', os.path.splitext(file)[0])
                except:
                    # If parsing fails, use filename as name
                    name = os.path.splitext(file)[0]
                    json_content = {}
                
                # Create configuration entry
                config = {
                    'id': os.path.splitext(file)[0],
                    'name': name,
                    'file': file,
                    'content': content,
                    'json': json_content
                }
                
                configs.append(config)
            except Exception as e:
                logger.error(f"Error loading configuration from {file}: {str(e)}")
        
        return configs
    except Exception as e:
        logger.error(f"Error getting configuration files: {str(e)}")
        return [] 