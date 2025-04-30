"""
Strategy blueprint for the web interface.

This module provides routes for managing trading strategies,
including starting, stopping, and configuring strategies.
"""

from flask import Blueprint

strategy = Blueprint('strategy', __name__, url_prefix='/strategy')

from . import routes 