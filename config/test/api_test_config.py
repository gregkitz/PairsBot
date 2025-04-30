"""
API endpoint configuration for testing.

This module contains endpoint configurations used for API testing,
making it easier to maintain endpoint URLs in a central location.
"""

# Base URL for API testing
BASE_URL = "http://localhost:5000/api"

# API endpoint definitions
ENDPOINTS = {
    # Health and status endpoints
    "health": "/health",
    "system_status": "/system/status",
    "system_metrics": "/system/metrics",
    
    # Task management
    "tasks": "/tasks",
    "task_cancel": "/tasks/cancel",
    
    # Backtest endpoints
    "backtest_submit": "/backtest/submit",
    "backtest_status": "/backtest/status/{}",  # Format with task_id
    "backtest_results": "/backtest/results/{}", # Format with task_id
    
    # Optimization endpoints
    "optimization_submit": "/optimization/submit",
    "optimization_status": "/optimization/status/{}",  # Format with task_id
    "optimization_results": "/optimization/results/{}", # Format with task_id
    
    # Data endpoints
    "historical_data": "/data/historical/{}/{}",  # Format with ticker, timeframe
    "pairs": "/pairs",
    
    # API versions
    "v1": "/v1",
    "latest": "",  # empty string for latest version
}

# Construct full URLs
def get_endpoint(name, *args):
    """
    Get full URL for an endpoint.
    
    Parameters:
    -----------
    name : str
        Endpoint name as defined in ENDPOINTS dictionary
    *args : list
        Format arguments for endpoint that requires parameters
        
    Returns:
    --------
    str
        Full URL for the endpoint
    """
    if name not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {name}")
        
    endpoint = ENDPOINTS[name]
    
    # Format endpoint with arguments if provided
    if args and '{}' in endpoint:
        endpoint = endpoint.format(*args)
        
    return f"{BASE_URL}{endpoint}"

# Authentication settings
AUTH_ENABLED = False
AUTH_HEADER = "X-API-Key"
AUTH_KEY = "test_api_key_123"

def get_auth_headers():
    """
    Get authentication headers for API requests.
    
    Returns:
    --------
    dict
        Headers containing authentication information if enabled
    """
    if AUTH_ENABLED:
        return {AUTH_HEADER: AUTH_KEY}
    return {} 