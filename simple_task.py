"""
A very simple Celery task for testing.
"""

from celery import Celery

# Create a simple Celery app
app = Celery('simple_tasks',
             broker='redis://redis:6379/0',
             backend='redis://redis:6379/0')

@app.task
def add(x, y):
    """Add two numbers together and return the result."""
    result = x + y
    print(f"Task executed: {x} + {y} = {result}")
    return result

if __name__ == '__main__':
    # Submit a task
    result = add.delay(4, 4)
    print(f"Task ID: {result.id}")
    print("Task submitted.") 