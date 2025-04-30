"""
API blueprint for the web interface.

This module provides API endpoints for external systems to interact
with the trading platform.
"""

from flask import Blueprint

api = Blueprint('api', __name__, url_prefix='/api')

from . import routes 