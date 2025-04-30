"""
Celery application setup for distributed task processing.
This module configures the Celery app used for running long-running tasks.
"""

import os
from celery import Celery

# Configure Redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/0'
BACKEND_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/1'

# Create Celery app
celery_app = Celery(
    'quant_trader',
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=[
        'src.tasks.train_tasks',
        'src.tasks.backtest_tasks',
        'src.tasks.optimization_tasks',
        'src.tasks.zscore_strategy_tasks',
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=86400,  # 24 hours max runtime
    worker_prefetch_multiplier=1,  # Don't prefetch tasks (better for long-running tasks)
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
)

# Define queues for different task types
celery_app.conf.task_routes = {
    'src.tasks.train_tasks.*': {'queue': 'train'},
    'src.tasks.backtest_tasks.*': {'queue': 'backtest'},
    'src.tasks.optimization_tasks.*': {'queue': 'optimize'},
    'src.tasks.zscore_strategy_tasks.*': {'queue': 'zscore'},
}

if __name__ == '__main__':
    celery_app.start() 