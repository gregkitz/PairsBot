# Kalman Filter Implementation for Time-Varying Cointegration

This document provides a detailed overview of the Kalman filter implementation used for time-varying cointegration and dynamic hedge ratio estimation in our pair trading system.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
    - [Linear Kalman Filter](#linear-kalman-filter)
    - [Extended Kalman Filter](#extended-kalman-filter)
    - [Unscented Kalman Filter](#unscented-kalman-filter)
3. [Implementation Details](#implementation-details)
    - [Base Kalman Filter Class](#base-kalman-filter-class)
    - [Linear Kalman Filter Implementation](#linear-kalman-filter-implementation)
    - [Extended Kalman Filter Implementation](#extended-kalman-filter-implementation)
    - [Unscented Kalman Filter Implementation](#unscented-kalman-filter-implementation)
4. [Time-Varying Hedge Ratio Estimation](#time-varying-hedge-ratio-estimation)
5. [Model Comparisons and Selection](#model-comparisons-and-selection)
6. [Diagnostic Methods](#diagnostic-methods)
7. [Integration with Spread Analytics](#integration-with-spread-analytics)
8. [Performance Optimization](#performance-optimization)
9. [Usage Examples](#usage-examples)
10. [References](#references)

## Introduction

Kalman filtering provides a powerful framework for estimating time-varying parameters in cointegration relationships. In pair trading, it allows us to adapt the hedge ratio dynamically as market conditions change, leading to more robust trading strategies. Our implementation supports three variants of the Kalman filter:

1. **Linear Kalman Filter**: For standard linear systems with Gaussian noise
2. **Extended Kalman Filter**: For nonlinear systems through local linearization
3. **Unscented Kalman Filter**: For highly nonlinear systems with non-Gaussian noise

These implementations are applied to estimate time-varying hedge ratios, detect regime changes, and provide adaptive spread calculations for trading signals.

## Mathematical Foundation

### Linear Kalman Filter

The Kalman filter is an optimal recursive estimator for linear systems with Gaussian noise. In the context of pairs trading, we model the relationship between two assets as:

$$y_t = \beta_t x_t + \alpha_t + \nu_t$$

where:
- $y_t$ is the price of the second asset at time $t$
- $x_t$ is the price of the first asset at time $t$
- $\beta_t$ is the time-varying hedge ratio
- $\alpha_t$ is the time-varying intercept
- $\nu_t$ is the observation noise, $\nu_t \sim \mathcal{N}(0, R)$

We model the time evolution of parameters as a random walk:

$$\begin{bmatrix} \alpha_t \\ \beta_t \end{bmatrix} = \begin{bmatrix} \alpha_{t-1} \\ \beta_{t-1} \end{bmatrix} + \omega_t$$

where $\omega_t \sim \mathcal{N}(0, Q)$ is the process noise.

The Kalman filter algorithm consists of two main steps:

1. **Prediction Step**:
   - State prediction: $\hat{x}_{t|t-1} = F_t \hat{x}_{t-1|t-1}$
   - Covariance prediction: $P_{t|t-1} = F_t P_{t-1|t-1} F_t^T + Q_t$

2. **Update Step**:
   - Innovation: $\tilde{y}_t = y_t - H_t \hat{x}_{t|t-1}$
   - Innovation covariance: $S_t = H_t P_{t|t-1} H_t^T + R_t$
   - Kalman gain: $K_t = P_{t|t-1} H_t^T S_t^{-1}$
   - State update: $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \tilde{y}_t$
   - Covariance update: $P_{t|t} = (I - K_t H_t) P_{t|t-1}$

For hedge ratio estimation, the state vector is $x_t = [\alpha_t, \beta_t]^T$, the transition matrix $F_t = I$ (identity for random walk), and the observation matrix $H_t = [1, x_t]$.

### Extended Kalman Filter

The Extended Kalman Filter (EKF) extends the linear Kalman filter to nonlinear systems by linearizing the system around the current state estimate. It can be used to model more complex relationships between assets.

For nonlinear systems, we have:

$$x_t = f(x_{t-1}) + \omega_t$$
$$y_t = h(x_t) + \nu_t$$

Where $f$ and $h$ are nonlinear functions. The EKF linearizes these functions using Jacobians:

$$F_t = \frac{\partial f}{\partial x}|_{x=\hat{x}_{t-1|t-1}}$$
$$H_t = \frac{\partial h}{\partial x}|_{x=\hat{x}_{t|t-1}}$$

For pairs trading, this allows us to model threshold-based relationships where the hedge ratio changes abruptly when certain conditions are met.

### Unscented Kalman Filter

The Unscented Kalman Filter (UKF) addresses limitations of the EKF by using a deterministic sampling approach known as the unscented transform. Instead of linearizing the nonlinear functions, it approximates the probability distribution using carefully chosen sigma points.

The key steps in the UKF algorithm are:

1. **Generate Sigma Points**:
   Generate 2n+1 sigma points (where n is the state dimension) using:
   $$\mathcal{X}_0 = \hat{x}$$
   $$\mathcal{X}_i = \hat{x} + \sqrt{(n+\lambda)P}_i$$
   $$\mathcal{X}_{i+n} = \hat{x} - \sqrt{(n+\lambda)P}_i$$

   Where $\sqrt{(n+\lambda)P}_i$ is the ith column of the matrix square root, and $\lambda = \alpha^2(n+\kappa)-n$ is a scaling parameter.

2. **Prediction Step**:
   - Transform sigma points: $\mathcal{Y}_i = f(\mathcal{X}_i)$
   - Predicted state: $\hat{x}_{t|t-1} = \sum_{i=0}^{2n} W_i^m \mathcal{Y}_i$
   - Predicted covariance: $P_{t|t-1} = \sum_{i=0}^{2n} W_i^c (\mathcal{Y}_i - \hat{x}_{t|t-1})(\mathcal{Y}_i - \hat{x}_{t|t-1})^T + Q$

3. **Update Step**:
   - Transform sigma points through measurement function: $\mathcal{Z}_i = h(\mathcal{Y}_i)$
   - Predicted measurement: $\hat{y}_{t|t-1} = \sum_{i=0}^{2n} W_i^m \mathcal{Z}_i$
   - Innovation covariance: $S_t = \sum_{i=0}^{2n} W_i^c (\mathcal{Z}_i - \hat{y}_{t|t-1})(\mathcal{Z}_i - \hat{y}_{t|t-1})^T + R$
   - Cross-covariance: $C_t = \sum_{i=0}^{2n} W_i^c (\mathcal{Y}_i - \hat{x}_{t|t-1})(\mathcal{Z}_i - \hat{y}_{t|t-1})^T$
   - Kalman gain: $K_t = C_t S_t^{-1}$
   - State update: $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - \hat{y}_{t|t-1})$
   - Covariance update: $P_{t|t} = P_{t|t-1} - K_t S_t K_t^T$

In our implementation, the UKF handles complex non-Gaussian models like stochastic volatility, Student-t innovations, and jump diffusion processes.

## Implementation Details

Our implementation in `src/cointegration/kalman_filter.py` follows a modular, object-oriented approach with a base class and specialized subclasses for each filter variant.

### Base Kalman Filter Class

The `KalmanFilterBase` class provides common functionality and interfaces for all Kalman filter implementations:

```python
class KalmanFilterBase:
    def __init__(self):
        self.is_fitted = False
        self.states = None
        self.state_covariances = None
        self.measurements = None
        self.log_likelihood = None
        self.innovations = None
        self.innovation_covariances = None
    
    def fit(self, endog, exog, **kwargs):
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, exog=None):
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_states(self):
        # Return estimated states
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing states")
        return self.states
    
    def plot_states(self, state_names=None, confidence_interval=0.95, figsize=(12, 8), title=None):
        # Plot estimated states with confidence intervals
        # Implementation details...
        
    def get_innovations(self):
        # Return innovations (prediction errors)
        # Implementation details...
        
    def get_innovation_statistics(self):
        # Compute statistics of innovations for diagnostics
        # Implementation details...
        
    def plot_innovation_diagnostics(self, measurement_names=None, figsize=(15, 10), lags=20):
        # Plot innovation diagnostics
        # Implementation details...
        
    def calculate_filter_performance_metrics(self, true_states=None):
        # Calculate performance metrics
        # Implementation details...
```

### Linear Kalman Filter Implementation

The `LinearKalmanFilter` class implements the standard Kalman filter for linear systems:

```python
class LinearKalmanFilter(KalmanFilterBase):
    def __init__(self, 
                transition_covariance=1e-4,
                observation_covariance=1e-2,
                initial_state_mean=None,
                initial_state_covariance=None,
                adapt_observation_noise=False,
                em_iterations=0):
        # Initialization code...
        
    def fit(self, endog, exog, dates=None):
        # Implementation of the linear Kalman filter
        # ...
        
    def predict(self, exog=None):
        # Prediction using fitted model
        # ...
        
    def _kalman_filter(self, endog, transition_matrices, observation_matrices,
                      transition_covariance, observation_covariance,
                      initial_state_mean, initial_state_covariance):
        # Core Kalman filter algorithm
        # ...
```

### Extended Kalman Filter Implementation

The `ExtendedKalmanFilter` class handles nonlinear models through local linearization:

```python
class ExtendedKalmanFilter(KalmanFilterBase):
    def __init__(self,
                state_transition_function,
                observation_function,
                state_transition_jacobian,
                observation_jacobian,
                transition_covariance=1e-4,
                observation_covariance=1e-2,
                initial_state_mean=None,
                initial_state_covariance=None):
        # Initialization code...
        
    def fit(self, observations, n_states, exog=None, dates=None):
        # Implementation of the extended Kalman filter
        # ...
        
    def predict(self, exog=None):
        # Prediction using fitted model
        # ...
```

### Unscented Kalman Filter Implementation

The `UnscentedKalmanFilter` class implements the unscented transform for highly nonlinear systems:

```python
class UnscentedKalmanFilter(KalmanFilterBase):
    def __init__(self, 
                state_transition_function,
                observation_function,
                process_noise_covariance=1e-4,
                observation_noise_covariance=1e-2,
                initial_state_mean=None,
                initial_state_covariance=None,
                alpha=1e-3,
                beta=2.0,
                kappa=0.0):
        # Initialization code...
        
    def _compute_sigma_points(self, mean, covariance):
        # Compute sigma points for the unscented transform
        # ...
        
    def _compute_weights(self, n_states):
        # Compute weights for the mean and covariance in unscented transform
        # ...
        
    def fit(self, observations, n_states, exog=None, dates=None):
        # Implementation of the unscented Kalman filter
        # ...
        
    def _unscented_kalman_filter(self, observations, exog=None):
        # Core unscented Kalman filter algorithm
        # ...
        
    def predict(self, exog=None):
        # Prediction using fitted model
        # ...
```

## Time-Varying Hedge Ratio Estimation

Our implementation includes specialized functions for estimating time-varying hedge ratios:

1. `estimate_timevarying_hedge_ratio()`: Uses the Linear Kalman Filter for basic hedge ratio estimation
2. `estimate_nonlinear_timevarying_hedge_ratio()`: Uses the Extended Kalman Filter for threshold or regime-switching models
3. `estimate_ukf_timevarying_hedge_ratio()`: Uses the Unscented Kalman Filter for complex non-Gaussian models

Example usage:

```python
# Linear model
results = estimate_timevarying_hedge_ratio(price1, price2, transition_covariance=1e-4)

# Nonlinear threshold model
results = estimate_nonlinear_timevarying_hedge_ratio(
    price1, price2, model_type='threshold', transition_covariance=1e-4
)

# Unscented Kalman Filter with stochastic volatility
results = estimate_ukf_timevarying_hedge_ratio(
    price1, price2, model_type='stochastic_volatility', transition_covariance=1e-4
)
```

The `results` DataFrame contains the estimated time-varying parameters (intercept, hedge ratio) and derived metrics like spread.

## Model Comparisons and Selection

We provide utilities for comparing different Kalman filter models:

1. `compare_kalman_models()`: Compares multiple Kalman filter models on the same data
2. `optimize_kalman_parameters()`: Finds optimal parameters for a given model
3. `optimize_adaptive_kalman_filter_parameters()`: Uses cross-validation for robust parameter selection

This allows users to select the most appropriate model for their specific pair:

```python
# Compare different models
model_comparison = compare_kalman_models(
    price1, price2, 
    models=['linear', 'threshold', 'regime_switch', 'log_price']
)

# Optimize parameters for selected model
best_params = optimize_kalman_parameters(
    price1, price2, model_type='linear'
)

# Adaptive parameter optimization with cross-validation
adaptive_params = optimize_adaptive_kalman_filter_parameters(
    price1, price2, validation_size=0.3, model_type='linear'
)
```

## Diagnostic Methods

Our implementation includes comprehensive diagnostic methods to evaluate filter performance:

1. `get_innovation_statistics()`: Computes statistics of innovations (prediction errors)
2. `plot_innovation_diagnostics()`: Visual diagnostics for filter performance
3. `calculate_filter_performance_metrics()`: Computes performance metrics
4. `compare_kalman_filter_with_reference()`: Compares our implementation with reference libraries

These methods help ensure that the Kalman filter is working correctly and optimally:

```python
# Get innovation statistics
innovation_stats = kalman_model.get_innovation_statistics()

# Plot innovation diagnostics
fig = kalman_model.plot_innovation_diagnostics()

# Calculate performance metrics
metrics = kalman_model.calculate_filter_performance_metrics()

# Compare with reference implementation
comparison = compare_kalman_filter_with_reference(
    price1, price2, reference_implementation='pykalman'
)
```

## Integration with Spread Analytics

The Kalman filter is integrated with the spread analytics framework through specialized functions:

1. `create_kalman_spread_analyzer()`: Combines Kalman filter with SpreadAnalyzer for comprehensive analysis
2. `backtest_kalman_zscore_strategy()`: Backtests a Z-score strategy using Kalman filter for dynamic hedge ratio

This integration enables advanced trading strategies:

```python
# Create spread analyzer with Kalman filter
spread_results, spread_analyzer = create_kalman_spread_analyzer(
    price1, price2, model_type='linear', window_size=20, use_volatility_adjustment=True
)

# Backtest strategy
backtest_results = backtest_kalman_zscore_strategy(
    price1, price2, model_type='linear', entry_threshold=2.0, exit_threshold=0.5,
    use_volatility_adjustment=True, include_costs=True
)
```

## Performance Optimization

The implementation includes several performance optimizations:

1. Efficient matrix operations using NumPy
2. Robust numerical stability through careful covariance handling
3. Vectorized operations where possible
4. Adaptive parameter updating

For extremely large datasets, consider using these optimization techniques:

```python
# Decimating data for initial parameter search
decimated_price1 = price1.iloc[::10]  # Use every 10th point
decimated_price2 = price2.iloc[::10]

# Find optimal parameters on reduced dataset
params = optimize_kalman_parameters(decimated_price1, decimated_price2)

# Apply to full dataset
results = estimate_timevarying_hedge_ratio(price1, price2, **params)
```

## Usage Examples

### Basic Linear Kalman Filter

```python
import pandas as pd
from src.cointegration.kalman_filter import estimate_timevarying_hedge_ratio, plot_timevarying_hedge_ratio

# Load price data
price1 = pd.read_csv('data/price1.csv', index_col='date', parse_dates=True)['close']
price2 = pd.read_csv('data/price2.csv', index_col='date', parse_dates=True)['close']

# Estimate time-varying hedge ratio using linear Kalman filter
results = estimate_timevarying_hedge_ratio(
    price1, price2,
    transition_covariance=1e-4,
    observation_covariance=1e-2
)

# Plot the results
fig = plot_timevarying_hedge_ratio(results, price1, price2)
```

### Threshold Model with Extended Kalman Filter

```python
from src.cointegration.kalman_filter import estimate_nonlinear_timevarying_hedge_ratio, plot_nonlinear_hedge_ratio

# Estimate with threshold model (EKF)
threshold_results = estimate_nonlinear_timevarying_hedge_ratio(
    price1, price2,
    model_type='threshold',
    transition_covariance=1e-4,
    observation_covariance=1e-2
)

# Plot the results
fig = plot_nonlinear_hedge_ratio(threshold_results, price1, price2, model_type='threshold')
```

### Stochastic Volatility Model with Unscented Kalman Filter

```python
from src.cointegration.kalman_filter import estimate_ukf_timevarying_hedge_ratio, plot_ukf_hedge_ratio

# Estimate with stochastic volatility model (UKF)
ukf_results = estimate_ukf_timevarying_hedge_ratio(
    price1, price2,
    model_type='stochastic_volatility',
    transition_covariance=1e-4,
    observation_covariance=1e-2
)

# Plot the results
fig = plot_ukf_hedge_ratio(ukf_results, price1, price2, model_type='stochastic_volatility')
```

### Trading Strategy Integration

```python
from src.cointegration.kalman_filter import backtest_kalman_zscore_strategy

# Backtest trading strategy
backtest_results = backtest_kalman_zscore_strategy(
    price1, price2,
    model_type='linear',
    entry_threshold=2.0,
    exit_threshold=0.5,
    window_size=20,
    use_volatility_adjustment=True,
    include_costs=True,
    commission_rate=0.001,
    slippage_rate=0.0005,
    max_holding_period=10,
    use_trailing_stop=True
)

# Print performance metrics
print(backtest_results['metrics'])
```

## References

1. Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems." Journal of Basic Engineering, 82(1), 35-45.

2. Harvey, A. C. (1990). "Forecasting, Structural Time Series Models and the Kalman Filter." Cambridge University Press.

3. Julier, S. J., & Uhlmann, J. K. (1997). "A new extension of the Kalman filter to nonlinear systems." In Int. symp. aerospace/defense sensing, simul. and controls (Vol. 3, No. 26, pp. 182-193).

4. Chan, N. H., et al. (2017). "Time-varying cointegration models for longitudinal data." Journal of Business & Economic Statistics, 35(3), 349-362.

5. Dunis, C. L., & Shannon, G. (2005). "Emerging markets of south-east and central Asia: Do they still offer a diversification benefit?" Journal of Asset Management, 6(3), 168-190.

6. Wan, E. A., & Van Der Merwe, R. (2000). "The unscented Kalman filter for nonlinear estimation." In Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (pp. 153-158).

7. Simon, D. (2006). "Optimal state estimation: Kalman, H infinity, and nonlinear approaches." John Wiley & Sons. 