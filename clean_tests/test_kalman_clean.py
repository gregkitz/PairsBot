import pytest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def test_simple():
    assert True

def test_linear_kalman_filter_init():
    """Test initialization of LinearKalmanFilter."""
    from src.cointegration.kalman_filter import LinearKalmanFilter
    kf = LinearKalmanFilter()
    assert kf.transition_covariance == 1e-4
    assert kf.observation_covariance == 1e-2
    assert kf.initial_state_mean is None
    assert kf.initial_state_covariance is None
    assert kf.adapt_observation_noise is False
    assert kf.em_iterations == 0
    assert kf.is_fitted is False
