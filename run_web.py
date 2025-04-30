#!/usr/bin/env python
"""
Run script for the web interface.

This script starts the web interface for the intraday statistical arbitrage system.
"""

import os
import argparse
import logging
from src.web_interface import create_app
from src.web_interface.app import socketio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_interface.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the web interface for the intraday statistical arbitrage system.')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set Flask environment variables
    os.environ['FLASK_ENV'] = 'development' if args.debug else 'production'
    os.environ['FLASK_DEBUG'] = str(args.debug).lower()
    
    # Create application
    app = create_app()
    
    # Set secret key from environment variable or generate a random one
    if 'SECRET_KEY' not in os.environ:
        import secrets
        os.environ['SECRET_KEY'] = secrets.token_hex(16)
    
    # Run application with SocketIO
    logger.info(f"Starting web interface on {args.host}:{args.port} (debug={args.debug})")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 