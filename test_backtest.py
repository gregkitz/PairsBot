"""
Test script to invoke the run_backtest task.
"""

from src.tasks.celery_app import celery_app
from src.tasks.backtest_tasks import run_backtest

if __name__ == "__main__":
    # Run the backtest task
    result = run_backtest.delay(
        pairs=["GC_SI"],
        start_date="2023-01-01",
        end_date="2023-01-31",
        timeframe="5min"
    )
    print(f"Backtest task submitted with ID: {result.id}")
    print("Check the worker logs to see if the task completes successfully.") 