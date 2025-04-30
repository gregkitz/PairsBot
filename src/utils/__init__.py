"""
Utility functions for common operations.
"""

from .utils import (
    create_directory,
    save_results,
    load_results,
    format_number,
    print_results_table,
    get_project_root,
    plot_comparison
)

from .technical_indicators import (
    rsi,
    macd,
    bollinger_bands,
    moving_averages,
    exponential_moving_averages,
    atr,
    z_score,
    stochastic_oscillator,
    rate_of_change,
    add_all_indicators
)

from .error_handling import (
    # Exception classes
    BaseError,
    # Data exceptions
    DataError, DataNotFoundError, DataValidationError, DataProcessingError,
    # Configuration exceptions
    ConfigurationError, ConfigNotFoundError, InvalidConfigurationError,
    # Trading exceptions
    TradingError, OrderExecutionError, PositionTrackingError,
    # Model exceptions
    ModelError, ModelTrainingError, ModelPredictionError, ModelLoadingError,
    # System exceptions
    SystemError, ResourceNotAvailableError, OperationTimeoutError,
    # Utility functions
    log_exception, create_error_dict, handle_exceptions, safe_execute, try_with_fallback
)

__all__ = [
    'create_directory',
    'save_results',
    'load_results',
    'format_number',
    'print_results_table',
    'get_project_root',
    'plot_comparison',
    'rsi',
    'macd',
    'bollinger_bands',
    'moving_averages',
    'exponential_moving_averages',
    'atr',
    'z_score',
    'stochastic_oscillator',
    'rate_of_change',
    'add_all_indicators',
    # Error handling framework
    'BaseError',
    'DataError', 'DataNotFoundError', 'DataValidationError', 'DataProcessingError',
    'ConfigurationError', 'ConfigNotFoundError', 'InvalidConfigurationError',
    'TradingError', 'OrderExecutionError', 'PositionTrackingError',
    'ModelError', 'ModelTrainingError', 'ModelPredictionError', 'ModelLoadingError',
    'SystemError', 'ResourceNotAvailableError', 'OperationTimeoutError',
    'log_exception', 'create_error_dict', 'handle_exceptions', 'safe_execute', 'try_with_fallback'
] 