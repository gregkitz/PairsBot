"""
Error Handling Framework for the Intraday ML Trading System.

This module provides consistent error handling and logging throughout the codebase
with standardized exception classes and error reporting utilities.
"""

import logging
import traceback
import functools
import inspect
import sys
from typing import Any, Callable, Dict, Optional, Type, Union, List, Tuple
from datetime import datetime

# Get a logger for this module
logger = logging.getLogger(__name__)


# ======================================================================
# Base Exception Classes
# ======================================================================

class BaseError(Exception):
    """Base exception class for all custom exceptions in the system."""
    
    def __init__(self, 
                message: str,
                error_code: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None,
                source: Optional[str] = None,
                cause: Optional[Exception] = None) -> None:
        """
        Initialize the error with detailed information.
        
        Parameters:
        -----------
        message : str
            Human-readable error message
        error_code : str, optional
            Error code for categorization
        details : dict, optional
            Additional error details as key-value pairs
        source : str, optional
            Source of the error (e.g., module name, function name)
        cause : Exception, optional
            The original exception that caused this error
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.source = source
        self.cause = cause
        self.timestamp = datetime.now()
        
        # Set the cause as __cause__ for proper chained exception handling
        if cause:
            self.__cause__ = cause
        
        # Construct the full error message
        full_message = message
        if error_code:
            full_message = f"[{error_code}] {full_message}"
        if source:
            full_message = f"{full_message} (source: {source})"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary representation.
        
        Returns:
        --------
        dict
            Dictionary representation of the error
        """
        result = {
            'error_code': self.error_code,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
        }
        
        if self.source:
            result['source'] = self.source
            
        if self.details:
            result['details'] = self.details
            
        if self.cause:
            result['cause'] = str(self.cause)
            
        return result


# ======================================================================
# Data Related Exceptions
# ======================================================================

class DataError(BaseError):
    """Base exception for data-related errors."""
    pass


class DataNotFoundError(DataError):
    """Exception raised when required data is not found."""
    pass


class DataValidationError(DataError):
    """Exception raised when data validation fails."""
    pass


class DataProcessingError(DataError):
    """Exception raised when data processing fails."""
    pass


# ======================================================================
# Configuration Related Exceptions
# ======================================================================

class ConfigurationError(BaseError):
    """Base exception for configuration-related errors."""
    pass


class ConfigNotFoundError(ConfigurationError):
    """Exception raised when a configuration file is not found."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Exception raised when configuration is invalid."""
    pass


# ======================================================================
# Trading Related Exceptions
# ======================================================================

class TradingError(BaseError):
    """Base exception for trading-related errors."""
    pass


class OrderExecutionError(TradingError):
    """Exception raised when order execution fails."""
    pass


class PositionTrackingError(TradingError):
    """Exception raised when position tracking fails."""
    pass


# ======================================================================
# Model Related Exceptions
# ======================================================================

class ModelError(BaseError):
    """Base exception for model-related errors."""
    pass


class ModelTrainingError(ModelError):
    """Exception raised when model training fails."""
    pass


class ModelPredictionError(ModelError):
    """Exception raised when model prediction fails."""
    pass


class ModelLoadingError(ModelError):
    """Exception raised when model loading fails."""
    pass


# ======================================================================
# System Related Exceptions
# ======================================================================

class SystemError(BaseError):
    """Base exception for system-related errors."""
    pass


class ResourceNotAvailableError(SystemError):
    """Exception raised when a required resource is not available."""
    pass


class OperationTimeoutError(SystemError):
    """Exception raised when an operation times out."""
    pass


# ======================================================================
# Error Handling Utilities
# ======================================================================

def log_exception(exc: Exception, logger_obj: Optional[logging.Logger] = None) -> None:
    """
    Log an exception with detailed information.
    
    Parameters:
    -----------
    exc : Exception
        The exception to log
    logger_obj : logging.Logger, optional
        Logger to use. If None, the default module logger is used.
    """
    log = logger_obj or logger
    
    if isinstance(exc, BaseError):
        # Log custom exceptions with their structured information
        error_info = exc.to_dict()
        error_msg = f"Error: {error_info['message']}"
        
        if 'source' in error_info:
            error_msg += f" (source: {error_info['source']})"
            
        log.error(error_msg, exc_info=True)
        
        # Log additional details at debug level
        if error_info.get('details'):
            details_str = ', '.join(f"{k}={v}" for k, v in error_info['details'].items())
            log.debug(f"Error details: {details_str}")
    else:
        # Log standard exceptions
        log.error(f"Error: {str(exc)}", exc_info=True)


def create_error_dict(exc: Exception, include_traceback: bool = False) -> Dict[str, Any]:
    """
    Create a dictionary representation of an error.
    
    Parameters:
    -----------
    exc : Exception
        The exception to convert
    include_traceback : bool
        Whether to include the traceback in the result
        
    Returns:
    --------
    dict
        Dictionary representation of the error
    """
    if isinstance(exc, BaseError):
        result = exc.to_dict()
    else:
        result = {
            'error_code': exc.__class__.__name__,
            'message': str(exc),
            'timestamp': datetime.now().isoformat()
        }
    
    if include_traceback:
        result['traceback'] = traceback.format_exception(
            type(exc), exc, exc.__traceback__
        )
    
    return result


def handle_exceptions(target_exception: Type[Exception] = Exception,
                    fallback_value: Any = None,
                    reraise: bool = False,
                    error_handler: Optional[Callable[[Exception], Any]] = None,
                    logger_obj: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator for handling exceptions in functions.
    
    Parameters:
    -----------
    target_exception : Type[Exception]
        The exception type to catch
    fallback_value : Any
        Value to return if an exception is caught
    reraise : bool
        Whether to reraise the exception after handling
    error_handler : Callable
        Function to call with the exception. Takes the exception as an argument.
    logger_obj : logging.Logger
        Logger to use for error logging
        
    Returns:
    --------
    Callable
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except target_exception as e:
                # Get the source information
                module_name = func.__module__
                func_name = func.__qualname__
                source = f"{module_name}.{func_name}"
                
                # Log the exception
                log = logger_obj or logger
                log.error(f"Exception in {source}: {str(e)}", exc_info=True)
                
                # Call the custom error handler if provided
                if error_handler:
                    return error_handler(e)
                
                # Reraise the exception if requested
                if reraise:
                    raise
                
                # Return the fallback value
                return fallback_value
        return wrapper
    return decorator


def safe_execute(func: Callable, 
               *args, 
               error_dict: Optional[Dict[Type[Exception], Any]] = None,
               default: Any = None, 
               logger_obj: Optional[logging.Logger] = None, 
               **kwargs) -> Any:
    """
    Execute a function safely with exception handling.
    
    Parameters:
    -----------
    func : Callable
        Function to execute
    *args
        Positional arguments to pass to the function
    error_dict : Dict[Type[Exception], Any]
        Dictionary mapping exception types to return values
    default : Any
        Default value to return for unspecified exceptions
    logger_obj : logging.Logger
        Logger to use for error logging
    **kwargs
        Keyword arguments to pass to the function
        
    Returns:
    --------
    Any
        Result of the function or fallback value
    """
    error_dict = error_dict or {}
    log = logger_obj or logger
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Get the source information
        module_name = func.__module__
        func_name = func.__qualname__
        source = f"{module_name}.{func_name}"
        
        # Log the exception
        log.error(f"Exception in {source}: {str(e)}", exc_info=True)
        
        # Check if we have a specific handler for this exception type
        for exc_type, fallback in error_dict.items():
            if isinstance(e, exc_type):
                return fallback() if callable(fallback) else fallback
        
        # Return the default value
        return default() if callable(default) else default


def try_with_fallback(operations: List[Tuple[Callable, List, Dict]],
                    logger_obj: Optional[logging.Logger] = None) -> Any:
    """
    Try a series of operations with fallbacks.
    
    Parameters:
    -----------
    operations : List[Tuple[Callable, List, Dict]]
        List of (function, args, kwargs) tuples to try in order
    logger_obj : logging.Logger
        Logger to use for error logging
        
    Returns:
    --------
    Any
        Result of the first successful operation
        
    Raises:
    -------
    Exception
        If all operations fail
    """
    log = logger_obj or logger
    last_exception = None
    
    for i, (func, args, kwargs) in enumerate(operations):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the source information
            module_name = func.__module__
            func_name = func.__qualname__
            source = f"{module_name}.{func_name}"
            
            # Log the exception
            log.warning(f"Operation {i+1}/{len(operations)} failed in {source}: {str(e)}")
            last_exception = e
    
    # If we get here, all operations failed
    if last_exception:
        log.error("All operations failed", exc_info=last_exception)
        raise last_exception
    else:
        raise RuntimeError("All operations failed") 