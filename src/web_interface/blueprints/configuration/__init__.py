"""
Configuration blueprint for the web interface.

This module provides routes for system configuration management.
"""

from flask import Blueprint

configuration = Blueprint('configuration', __name__, url_prefix='/config')

from . import routes 