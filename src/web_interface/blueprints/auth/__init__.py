"""
Authentication blueprint for the web interface.

This module handles user authentication, including login, logout,
and user management.
"""

from flask import Blueprint

auth = Blueprint('auth', __name__, url_prefix='/auth')

from . import routes 