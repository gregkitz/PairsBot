"""
Minimal self-contained Celery example.
This script creates a task, starts a worker, and runs the task.
"""

from celery import Celery
import time
import threading
import sys

# Create a simple Celery app
app = Celery('minimal',
             broker='redis://redis:6379/0',
             backend='redis://redis:6379/0')

# Configure Celery to run tasks eagerly (in the same process)
app.conf.update(
    task_always_eager=True,
    task_eager_propagates=True,
)

@app.task
def add(x, y):
    """Add two numbers together and return the result."""
    result = x + y
    print(f"Task executed: {x} + {y} = {result}")
    return result

if __name__ == '__main__':
    print("Starting minimal Celery example...")
    print("This will run the task in the same process.")
    
    # Run the task
    result = add.delay(4, 4)
    
    # Wait for the task to complete (should be immediate in eager mode)
    time.sleep(1)
    
    # Check the result
    print(f"Task ID: {result.id}")
    print(f"Task status: {result.status}")
    print(f"Task result: {result.result}")
    print("Done!") 