"""
Dashboard blueprint for the web interface.

This module provides the main dashboard for monitoring strategy performance
and system status.
"""

from flask import Blueprint

dashboard = Blueprint('dashboard', __name__, url_prefix='/dashboard')

from . import routes 