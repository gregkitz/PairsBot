"""
Flask application module for the web interface.

This module initializes the Flask application, configures routes,
and handles requests for the web interface.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_socketio import SocketIO
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logger = logging.getLogger(__name__)

# Initialize SocketIO for real-time updates
socketio = SocketIO()

def create_app(config=None):
    """
    Create and configure the Flask application.
    
    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary. If None, default config is used.
    
    Returns:
    --------
    Flask
        The configured Flask application
    """
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Default configuration
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-for-development-only'),
        DEBUG=os.environ.get('FLASK_DEBUG', 'False') == 'True',
        SESSION_TYPE='filesystem',
        TEMPLATES_AUTO_RELOAD=True,
        DATA_DIRECTORY=os.path.join(os.getcwd(), 'data')
    )
    
    # Apply configuration from parameter if provided
    if config:
        app.config.update(config)
    
    # Initialize extensions
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Initialize login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    
    # Import and register blueprints
    from .blueprints.auth import auth as auth_blueprint
    from .blueprints.dashboard import dashboard as dashboard_blueprint
    from .blueprints.strategy import strategy as strategy_blueprint
    from .blueprints.configuration import configuration as config_blueprint
    from .blueprints.api import api as api_blueprint
    
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(dashboard_blueprint)
    app.register_blueprint(strategy_blueprint)
    app.register_blueprint(config_blueprint)
    app.register_blueprint(api_blueprint)
    
    # Configure error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error/404.html'), 404
        
    @app.errorhandler(500)
    def server_error(e):
        return render_template('error/500.html'), 500
    
    # User loader for Flask-Login
    from .models import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)
    
    # Health check route
    @app.route('/health')
    def health_check():
        return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
    
    # Initialize default user if not exists
    with app.app_context():
        try:
            # Create config directory if not exists
            os.makedirs(os.path.join(app.config['DATA_DIRECTORY'], 'web_users'), exist_ok=True)
            
            # Check if users file exists
            users_file = os.path.join(app.config['DATA_DIRECTORY'], 'web_users', 'users.json')
            if not os.path.exists(users_file):
                # Create default admin user
                users = {
                    'admin': {
                        'username': 'admin',
                        'password_hash': generate_password_hash('admin'),
                        'role': 'admin'
                    }
                }
                
                with open(users_file, 'w') as f:
                    json.dump(users, f, indent=2)
                
                logger.info("Created default admin user")
                
                # Log warning to change default password
                logger.warning("Default admin user created with password 'admin'. "
                             "Please change this password immediately.")
        except Exception as e:
            logger.error(f"Error initializing default user: {str(e)}")
    
    # Done!
    logger.info("Flask application initialized")
    return app 