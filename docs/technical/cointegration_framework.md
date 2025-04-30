# Cointegration Framework Documentation

## Overview

This document provides a comprehensive overview of our cointegration framework for pairs trading. It explains how different components interact, validation approaches, and implementation details to ensure proper understanding of the system architecture.

## Table of Contents
1. [Framework Architecture](#framework-architecture)
2. [Component Interactions](#component-interactions)
3. [Implementation Details](#implementation-details)
4. [Validation Approaches](#validation-approaches)
5. [Examples](#examples)
6. [Performance Considerations](#performance-considerations)
7. [References](#references)

## Framework Architecture

Our cointegration framework is designed as a modular system with clear separation of concerns. The framework consists of the following primary components:

1. **Statistical Methods** - Core mathematical implementations of cointegration tests
2. **Pair Selection** - Processes for identifying potentially cointegrated pairs
3. **Cointegration Testing** - Components for validating cointegration relationships
4. **Rolling Analysis** - Time-varying analysis of relationship stability
5. **Out-of-Sample Validation** - Verification of relationship persistence
6. **Z-Score Strategy** - Basic trading strategy based on cointegration

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Pair Selection │───>│  Cointegration  │───>│  Rolling Window │
│                 │    │     Testing     │    │    Analysis     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Out-of-Sample  │<───│  Half-life      │
                       │   Validation    │    │   Estimation    │
                       └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │    Z-Score      │
                       │    Strategy     │
                       └─────────────────┘
```

## Component Interactions

### Data Flow

1. **Pair Selection → Cointegration Testing**
   - Pair Selection identifies promising pairs based on correlation and sector/industry relationships
   - These candidate pairs are passed to Cointegration Testing for formal statistical evaluation

2. **Cointegration Testing → Rolling Window Analysis**
   - Pairs that pass cointegration tests are analyzed for relationship stability over time
   - Both Engle-Granger and Johansen tests are employed based on the context

3. **Cointegration Testing → Half-life Estimation**
   - Cointegrated pairs have their mean-reversion speed estimated through half-life calculations
   - This information is critical for setting appropriate trading parameters

4. **Rolling Window Analysis → Out-of-Sample Validation**
   - Stability metrics from rolling analysis inform the out-of-sample validation process
   - Pairs with unstable relationships are filtered out

5. **Out-of-Sample Validation → Z-Score Strategy**
   - Validated pairs are used to generate trading signals based on z-score deviations
   - Trading parameters are optimized based on half-life and stability metrics

### Key Interfaces

1. **Statistical Methods Interface**
   - All components access statistical methods through a standardized interface
   - This ensures consistency in statistical calculations across the framework
   - Located in `src/cointegration/statistical_methods.py`

2. **Data Interface**
   - Standardized data format for all components (pandas DataFrames with consistent indexing)
   - Common preprocessing methods ensure data quality

3. **Results Interface**
   - All tests return structured dictionaries with consistent keys
   - Enables easy integration between components and reliable result interpretation

## Implementation Details

### Statistical Methods

The core statistical implementations are found in `src/cointegration/statistical_methods.py` and include:

1. **Johansen Test**
   - Multivariate cointegration analysis
   - Allows for testing multiple time series simultaneously
   - Implemented according to Johansen's original papers with numerical improvements

2. **Engle-Granger Test**
   - Bivariate cointegration analysis with direct hedge ratio estimation
   - Multiple regression methods (OLS, DOLS, TLS)
   - Enhanced with proper critical value adjustments

3. **Half-life Estimation**
   - AR(1) model-based estimation
   - Enhanced with confidence intervals and stability metrics
   - Multiple calculation methods for robustness

### Cointegration Testing

The main testing framework is implemented in `src/cointegration/cointegration_tests.py` and provides:

1. **Combined Testing Approach**
   - Integrates both Engle-Granger and Johansen methods
   - Provides comprehensive results including hedge ratios and half-life
   - Automatically handles data preprocessing

2. **Rolling Window Analysis**
   - Time-varying analysis of cointegration stability
   - Multiple window sizes for robustness
   - Stability metrics for relationship quality assessment

3. **Out-of-Sample Validation**
   - Train/test split methodology
   - Performance metrics on validation period
   - Statistical significance testing

### Z-Score Strategy

The basic trading strategy is implemented in:

1. **Signal Generation**
   - Z-score-based entry and exit signals
   - Adaptive thresholds based on half-life
   - Transaction cost consideration

2. **Backtesting Framework**
   - Historical performance evaluation
   - Risk-adjusted metrics calculation
   - Parameter optimization utilities

## Validation Approaches

Our framework incorporates several validation methodologies to ensure robustness:

1. **Statistical Validation**
   - Proper critical values for hypothesis testing
   - Multiple test statistics (trace, maximum eigenvalue)
   - P-value adjustments for multiple testing

2. **Time-Varying Validation**
   - Rolling window analysis to assess relationship stability
   - Metrics for coefficient stability and mean-reversion consistency
   - Detection of structural breaks

3. **Out-of-Sample Testing**
   - Train/test split validation
   - Forward walk testing
   - Parameter stability analysis

4. **Economic Validation**
   - Transaction cost inclusion
   - Risk-adjusted performance metrics
   - Comparison to benchmark strategies

## Examples

### Basic Usage Example

```python
from src.cointegration.cointegration_tests import test_cointegration

# Test a pair for cointegration
result = test_cointegration(
    price_series1=stock1_prices,
    price_series2=stock2_prices,
    test_type='both',           # Use both Engle-Granger and Johansen
    train_test_split=0.7,       # 70/30 train/test split
    use_log_prices=True,        # Use log prices for better properties
    out_of_sample=True          # Validate out-of-sample
)

# Check if cointegrated
if result['combined_result']['is_cointegrated']:
    # Extract hedge ratio and half-life
    hedge_ratio = result['hedge_ratio']
    half_life = result['half_life']
    
    # Check out-of-sample validation
    if result['out_of_sample']['is_cointegrated_oos']:
        print(f"Valid pair with hedge ratio {hedge_ratio} and half-life {half_life}")
```

### Rolling Window Example

```python
from src.cointegration.cointegration_tests import rolling_cointegration

# Analyze stability over time
rolling_result = rolling_cointegration(
    price_series1=stock1_prices,
    price_series2=stock2_prices,
    window=60,                 # 60-day window
    step=5,                    # Move forward 5 days each step
    window_sizes=[30, 60, 90]  # Test multiple window sizes for robustness
)

# Check stability metrics
stability = rolling_result['stability_metrics']
if stability['cointegration_frequency'] > 0.8 and stability['hedge_ratio_stability'] < 0.2:
    print("Stable relationship suitable for trading")
```

## Performance Considerations

1. **Computational Efficiency**
   - Matrix operations optimization for Johansen test
   - Vectorized operations where possible
   - Caching for repeated calculations

2. **Memory Management**
   - Efficient handling of large rolling window calculations
   - Proper cleanup of temporary data structures

3. **Parallelization**
   - Parallel processing for universe testing
   - Batch processing for multiple pairs

4. **Edge Cases**
   - Handling of missing data
   - Numerical stability for near-singular matrices
   - Graceful degradation for insufficient data

## References

1. Johansen, S. (1988). "Statistical analysis of cointegration vectors." Journal of Economic Dynamics and Control, 12(2-3), 231-254.
2. Johansen, S. (1991). "Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models." Econometrica, 59(6), 1551-1580.
3. Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction: representation, estimation, and testing." Econometrica, 55(2), 251-276.
4. MacKinnon, J. G. (2010). "Critical values for cointegration tests." Queen's Economics Department Working Paper, (1227).
5. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis." Wiley, Chichester.
6. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley, Hoboken, NJ. 