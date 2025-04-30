"""
Simple test script to verify Celery task queue functionality.
"""

from src.tasks.celery_app import celery_app

# Define a simple test task
@celery_app.task(name='test_task')
def test_task():
    print("Test task running...")
    return {'success': True, 'message': 'Task completed successfully'}

if __name__ == "__main__":
    # Run the test task
    result = test_task.delay()
    print(f"Task submitted with ID: {result.id}")
    print("Check the worker logs to see if the task completes successfully.") 