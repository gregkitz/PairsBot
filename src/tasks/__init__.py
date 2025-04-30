"""
Tasks package for distributed processing.
This package contains Celery tasks for running long-running processes like 
training, backtesting, and optimization.
"""

from src.tasks.celery_app import celery_app

__all__ = ['celery_app'] 