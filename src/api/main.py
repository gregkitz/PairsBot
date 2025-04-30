"""
FastAPI application for task management and system control.
This module provides REST API endpoints for submitting and monitoring tasks.
"""

import os
from typing import List, Dict, Any, Optional, Union, Annotated
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, constr, StringConstraints
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Quant Trader API",
    description="API for managing trading system tasks and monitoring results",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Task models with enhanced validation
class TrainModelRequest(BaseModel):
    """Request model for training ML models."""
    pair: Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9]+_[A-Za-z0-9]+$')] = Field(..., description="Pair ID in format SYMBOL1_SYMBOL2")
    timeframe: Annotated[str, StringConstraints(pattern=r'^[0-9]+[smhd]in$')] = Field(..., description="Timeframe (e.g., 5min, 1hour)")
    start_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="End date (YYYY-MM-DD)")
    config_path: Optional[str] = Field(None, description="Path to configuration file")
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    @validator('config_path')
    def config_path_must_exist(cls, v):
        if v is not None and not os.path.exists(v):
            raise ValueError(f'config_path {v} does not exist')
        return v

class RegimeClassifierRequest(BaseModel):
    """Request model for training regime classifiers."""
    tickers: List[str] = Field(..., description="List of tickers to train on", min_items=1)
    timeframe: Annotated[str, StringConstraints(pattern=r'^[0-9]+[smhd]in$|^[0-9]+day$')] = Field(..., description="Timeframe (e.g., 1hour, 1day)")
    n_regimes: int = Field(3, description="Number of regimes to detect", ge=2, le=5)
    config_path: Optional[str] = Field(None, description="Path to configuration file")
    
    @validator('config_path')
    def config_path_must_exist(cls, v):
        if v is not None and not os.path.exists(v):
            raise ValueError(f'config_path {v} does not exist')
        return v

class BacktestRequest(BaseModel):
    """Request model for backtesting operations."""
    pairs: List[Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9]+_[A-Za-z0-9]+$')]] = Field(..., description="List of pairs to backtest", min_items=1)
    start_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="End date (YYYY-MM-DD)")
    timeframe: Annotated[str, StringConstraints(pattern=r'^[0-9]+[smhd]in$')] = Field(..., description="Timeframe (e.g., 5min, 1hour)")
    use_ml: bool = Field(False, description="Whether to use ML enhancements")
    config_path: Optional[str] = Field(None, description="Path to configuration file")
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    @validator('config_path')
    def config_path_must_exist(cls, v):
        if v is not None and not os.path.exists(v):
            raise ValueError(f'config_path {v} does not exist')
        return v

class ZScoreBacktestRequest(BaseModel):
    """Request model for z-score strategy backtesting operations."""
    pairs: List[Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9]+_[A-Za-z0-9]+$')]] = Field(..., description="List of pairs to backtest", min_items=1)
    start_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="End date (YYYY-MM-DD)")
    timeframe: Annotated[str, StringConstraints(pattern=r'^[0-9]+[smhd]in$|^[0-9]+day$')] = Field(..., description="Timeframe (e.g., 5min, 1hour, 1day)")
    zscore_params: Optional[Dict[str, Any]] = Field(None, description="Z-Score strategy parameters")
    use_log_prices: bool = Field(False, description="Whether to use log prices for spread calculation")
    transaction_costs: Optional[Dict[str, float]] = Field(None, description="Transaction costs parameters")
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class ZScoreOptimizationRequest(BaseModel):
    """Request model for z-score strategy parameter optimization."""
    pairs: List[Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9]+_[A-Za-z0-9]+$')]] = Field(..., description="List of pairs to optimize", min_items=1)
    start_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="End date (YYYY-MM-DD)")
    timeframe: Annotated[str, StringConstraints(pattern=r'^[0-9]+[smhd]in$|^[0-9]+day$')] = Field(..., description="Timeframe (e.g., 5min, 1hour, 1day)")
    param_grid: Optional[Dict[str, List[Any]]] = Field(None, description="Parameter grid for optimization")
    use_log_prices: bool = Field(False, description="Whether to use log prices for spread calculation")
    transaction_costs: Optional[Dict[str, float]] = Field(None, description="Transaction costs parameters")
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class OptimizationRequest(BaseModel):
    """Request model for parameter optimization operations."""
    pairs_file: str = Field(..., description="Path to file containing pairs to optimize")
    start_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: Annotated[str, StringConstraints(pattern=r'^\d{4}-\d{2}-\d{2}$')] = Field(..., description="End date (YYYY-MM-DD)")
    quick_mode: bool = Field(False, description="Whether to use quick mode (fewer iterations)")
    config_path: Optional[str] = Field(None, description="Path to configuration file")
    
    @validator('pairs_file')
    def pairs_file_must_exist(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'pairs_file {v} does not exist')
        return v
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    @validator('config_path')
    def config_path_must_exist(cls, v):
        if v is not None and not os.path.exists(v):
            raise ValueError(f'config_path {v} does not exist')
        return v

class TaskResponse(BaseModel):
    """Response model for task submission."""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task")
    status: str = Field(..., description="Task status")
    timestamp: str = Field(..., description="Task submission timestamp")

class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task")
    status: str = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[Dict[str, Any]] = Field(None, description="Progress information")
    timestamp: str = Field(..., description="Status timestamp")

class TaskCancelRequest(BaseModel):
    """Request model for canceling a task."""
    task_id: str = Field(..., description="Unique task identifier")
    terminate: bool = Field(False, description="Whether to terminate the task or just revoke it")

# Health check endpoint
@app.get("/health", 
         summary="Health check",
         description="Simple health check endpoint to verify API availability",
         response_description="Health status and timestamp")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# System status endpoint
@app.get("/system/status",
         summary="System status",
         description="Get system status information including worker availability",
         response_description="System status information")
def system_status():
    """Get system status information."""
    try:
        # Import Celery app
        from src.tasks.celery_app import celery_app
        
        # Get basic status information
        worker_status = {}
        try:
            # Inspect workers
            inspector = celery_app.control.inspect()
            active_workers = inspector.active()
            ping_workers = inspector.ping()
            
            if active_workers:
                worker_status["active"] = active_workers
            if ping_workers:
                worker_status["available"] = list(ping_workers.keys())
            
            if not worker_status:
                worker_status["error"] = "No workers available"
        except Exception as e:
            worker_status["error"] = str(e)
        
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "workers": worker_status,
            "queues": {
                "default": {"status": "available"},
                "backtest": {"status": "available"},
                "optimize": {"status": "available"},
                "train": {"status": "available"},
                "zscore": {"status": "available"}
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get system status: {str(e)}"
        )

# Task submission endpoints
@app.post("/tasks/train-models", 
          response_model=TaskResponse,
          summary="Train ML models",
          description="Submit a task to train ML models for a specific pair",
          response_description="Task submission details and ID",
          status_code=status.HTTP_202_ACCEPTED)
def submit_train_models(request: TrainModelRequest):
    """Submit a model training task."""
    try:
        # Import here to prevent circular imports
        from src.tasks.train_tasks import train_models
        
        # Record submission time
        timestamp = datetime.now().isoformat()
        
        # Submit task to Celery
        task = train_models.delay(
            pair=request.pair,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            config_path=request.config_path
        )
        
        logger.info(f"Submitted training task with ID: {task.id}")
        return {
            "task_id": task.id,
            "task_type": "train_models",
            "status": "submitted",
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error submitting training task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to submit task: {str(e)}"
        )

@app.post("/tasks/train-regime-classifier", 
          response_model=TaskResponse,
          summary="Train regime classifier",
          description="Submit a task to train market regime classifier",
          response_description="Task submission details and ID",
          status_code=status.HTTP_202_ACCEPTED)
def submit_train_regime_classifier(request: RegimeClassifierRequest):
    """Submit a regime classifier training task."""
    try:
        # Import here to prevent circular imports
        from src.tasks.train_tasks import train_regime_classifier
        
        # Record submission time
        timestamp = datetime.now().isoformat()
        
        # Submit task to Celery
        task = train_regime_classifier.delay(
            tickers=request.tickers,
            timeframe=request.timeframe,
            n_regimes=request.n_regimes,
            config_path=request.config_path
        )
        
        logger.info(f"Submitted regime classifier training task with ID: {task.id}")
        return {
            "task_id": task.id,
            "task_type": "train_regime_classifier",
            "status": "submitted",
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error submitting regime classifier task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )

@app.post("/tasks/backtest", 
          response_model=TaskResponse,
          summary="Run backtest",
          description="Submit a task to run backtest on specified pairs",
          response_description="Task submission details and ID",
          status_code=status.HTTP_202_ACCEPTED)
def submit_backtest(request: BacktestRequest):
    """Submit a backtest task."""
    try:
        # Import here to prevent circular imports
        from src.tasks.backtest_tasks import run_backtest
        
        # Record submission time
        timestamp = datetime.now().isoformat()
        
        # Submit task to Celery
        task = run_backtest.delay(
            pairs=request.pairs,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            use_ml=request.use_ml,
            config_path=request.config_path
        )
        
        logger.info(f"Submitted backtest task with ID: {task.id}")
        return {
            "task_id": task.id,
            "task_type": "backtest",
            "status": "submitted",
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error submitting backtest task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )

@app.post("/tasks/intraday-backtest", 
          response_model=TaskResponse,
          summary="Run intraday backtest",
          description="Submit a task to run intraday backtest with ML enhancements",
          response_description="Task submission details and ID",
          status_code=status.HTTP_202_ACCEPTED)
def submit_intraday_backtest(request: BacktestRequest):
    """Submit an intraday backtest task."""
    try:
        # Import here to prevent circular imports
        from src.tasks.backtest_tasks import run_intraday_backtest
        
        # Record submission time
        timestamp = datetime.now().isoformat()
        
        # Submit task to Celery
        task = run_intraday_backtest.delay(
            pairs=request.pairs,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            use_ml=request.use_ml,
            config_path=request.config_path
        )
        
        logger.info(f"Submitted intraday backtest task with ID: {task.id}")
        return {
            "task_id": task.id,
            "task_type": "intraday_backtest",
            "status": "submitted",
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error submitting intraday backtest task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )

@app.post("/tasks/optimize-parameters", 
          response_model=TaskResponse,
          summary="Optimize parameters",
          description="Submit a task to optimize strategy parameters",
          response_description="Task submission details and ID",
          status_code=status.HTTP_202_ACCEPTED)
def submit_optimization(request: OptimizationRequest):
    """Submit a parameter optimization task."""
    try:
        # Import here to prevent circular imports
        from src.tasks.optimization_tasks import optimize_parameters
        
        # Record submission time
        timestamp = datetime.now().isoformat()
        
        # Submit task to Celery
        task = optimize_parameters.delay(
            pairs_file=request.pairs_file,
            start_date=request.start_date,
            end_date=request.end_date,
            quick_mode=request.quick_mode,
            config_path=request.config_path
        )
        
        logger.info(f"Submitted optimization task with ID: {task.id}")
        return {
            "task_id": task.id,
            "task_type": "optimize_parameters",
            "status": "submitted",
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error submitting optimization task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )

@app.post("/tasks/zscore-backtest", 
          response_model=TaskResponse,
          summary="Run Z-Score strategy backtest",
          description="Submit a task to run Z-Score strategy backtest on specified pairs",
          response_description="Task submission details and ID",
          status_code=status.HTTP_202_ACCEPTED)
def submit_zscore_backtest(request: ZScoreBacktestRequest):
    """Submit a Z-Score strategy backtest task."""
    try:
        # Import here to prevent circular imports
        from src.tasks.zscore_strategy_tasks import run_zscore_backtest
        
        # Record submission time
        timestamp = datetime.now().isoformat()
        
        # Submit task to Celery
        task = run_zscore_backtest.delay(
            pairs=request.pairs,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            zscore_params=request.zscore_params,
            use_log_prices=request.use_log_prices,
            transaction_costs=request.transaction_costs
        )
        
        logger.info(f"Submitted Z-Score backtest task with ID: {task.id}")
        return {
            "task_id": task.id,
            "task_type": "zscore_backtest",
            "status": "submitted",
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error submitting Z-Score backtest task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )

@app.post("/tasks/zscore-optimize", 
          response_model=TaskResponse,
          summary="Optimize Z-Score strategy parameters",
          description="Submit a task to optimize Z-Score strategy parameters",
          response_description="Task submission details and ID",
          status_code=status.HTTP_202_ACCEPTED)
def submit_zscore_optimization(request: ZScoreOptimizationRequest):
    """Submit a Z-Score parameter optimization task."""
    try:
        # Import here to prevent circular imports
        from src.tasks.zscore_strategy_tasks import optimize_zscore_parameters
        
        # Record submission time
        timestamp = datetime.now().isoformat()
        
        # Submit task to Celery
        task = optimize_zscore_parameters.delay(
            pairs=request.pairs,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            param_grid=request.param_grid,
            use_log_prices=request.use_log_prices,
            transaction_costs=request.transaction_costs
        )
        
        logger.info(f"Submitted Z-Score optimization task with ID: {task.id}")
        return {
            "task_id": task.id,
            "task_type": "zscore_optimization",
            "status": "submitted",
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error submitting Z-Score optimization task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )

# Task management endpoints
@app.get("/tasks/{task_id}", 
         response_model=TaskStatusResponse,
         summary="Get task status",
         description="Get the status and results of a task by its ID",
         response_description="Task status information and results if available")
def get_task_status(task_id: str = Path(..., description="Unique task identifier")):
    """Get the status of a task by its ID."""
    try:
        # Import Celery app
        from src.tasks.celery_app import celery_app
        
        # Get task result
        result = celery_app.AsyncResult(task_id)
        
        # Determine task type from result info
        task_type = "unknown"
        if result.info and isinstance(result.info, dict) and 'task_type' in result.info:
            task_type = result.info['task_type']
        
        # Get progress information
        progress = None
        if result.state == 'PROGRESS' and result.info:
            progress = result.info
        
        # Return status information
        response = {
            "task_id": task_id,
            "task_type": task_type,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "error": str(result.traceback) if result.failed() else None,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )

@app.post("/tasks/cancel",
          summary="Cancel task",
          description="Cancel a running task by its ID",
          response_description="Cancellation status")
def cancel_task(request: TaskCancelRequest):
    """Cancel a running task."""
    try:
        # Import Celery app
        from src.tasks.celery_app import celery_app
        
        # Get task
        task = celery_app.AsyncResult(request.task_id)
        
        # Check if task exists and is not completed
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {request.task_id} not found"
            )
        
        # Revoke task
        celery_app.control.revoke(request.task_id, terminate=request.terminate)
        
        return {
            "task_id": request.task_id,
            "status": "canceled",
            "timestamp": datetime.now().isoformat(),
            "message": f"Task {request.task_id} has been {'terminated' if request.terminate else 'revoked'}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )

@app.get("/tasks",
         summary="List tasks",
         description="List all active and recently completed tasks",
         response_description="List of tasks with their status")
def list_tasks(
    limit: int = Query(10, description="Number of tasks to return", ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by task status"),
    task_type: Optional[str] = Query(None, description="Filter by task type")
):
    """List tasks with optional filtering."""
    try:
        # Import Celery app
        from src.tasks.celery_app import celery_app
        
        # Get active and reserved tasks
        inspector = celery_app.control.inspect()
        
        active_tasks = inspector.active() or {}
        reserved_tasks = inspector.reserved() or {}
        scheduled_tasks = inspector.scheduled() or {}
        
        # Combine all tasks
        all_tasks = []
        
        # Process active tasks
        for worker, tasks in active_tasks.items():
            for task in tasks:
                task_info = {
                    "task_id": task['id'],
                    "task_type": task['name'].split('.')[-1],
                    "status": "active",
                    "worker": worker,
                    "time_start": task.get('time_start', None)
                }
                all_tasks.append(task_info)
        
        # Process reserved tasks
        for worker, tasks in reserved_tasks.items():
            for task in tasks:
                task_info = {
                    "task_id": task['id'],
                    "task_type": task['name'].split('.')[-1],
                    "status": "reserved",
                    "worker": worker
                }
                all_tasks.append(task_info)
        
        # Process scheduled tasks
        for worker, tasks in scheduled_tasks.items():
            for task in tasks:
                task_info = {
                    "task_id": task['request']['id'],
                    "task_type": task['request']['name'].split('.')[-1],
                    "status": "scheduled",
                    "worker": worker
                }
                all_tasks.append(task_info)
        
        # Apply filters
        if status:
            all_tasks = [t for t in all_tasks if t['status'] == status]
        
        if task_type:
            all_tasks = [t for t in all_tasks if t['task_type'] == task_type]
        
        # Sort by start time (if available)
        all_tasks.sort(key=lambda x: x.get('time_start', 0) or 0, reverse=True)
        
        # Apply limit
        all_tasks = all_tasks[:limit]
        
        return {
            "tasks": all_tasks,
            "count": len(all_tasks),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Get host and port from environment variables or use defaults
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    # Run the API server
    uvicorn.run(app, host=host, port=port) 