# Statistical Methods for Cointegration Testing

This document provides a comprehensive overview of the statistical methods implemented for cointegration testing in our pairs trading system.

## Table of Contents
1. [Introduction to Cointegration](#introduction-to-cointegration)
2. [Engle-Granger Two-Step Method](#engle-granger-two-step-method)
3. [Johansen Cointegration Test](#johansen-cointegration-test)
4. [Half-Life Estimation](#half-life-estimation)
5. [Implementation Details](#implementation-details)
6. [Usage Examples](#usage-examples)
7. [References](#references)

## Introduction to Cointegration

Cointegration is a statistical property of multiple time series that allows us to identify stable, long-term relationships between non-stationary variables. Two or more time series are said to be cointegrated if they share a common stochastic trend, and a linear combination of these series is stationary.

In the context of pairs trading, cointegration is crucial because:
- It provides a statistical foundation for identifying pairs that tend to move together
- It helps identify relationships that are likely to revert to their mean
- It reduces the risk of spurious relationships by requiring a formal statistical criterion

Our system implements two primary methods for testing cointegration:
1. The Engle-Granger two-step method (for bivariate analysis)
2. The Johansen test (for multivariate analysis)

## Engle-Granger Two-Step Method

### Mathematical Foundation

The Engle-Granger method is a two-step procedure for testing cointegration between two time series:

**Step 1**: Estimate the cointegrating relationship by regressing one series on the other:

$$y_t = \alpha + \beta x_t + u_t$$

Where:
- $y_t$ and $x_t$ are the two time series
- $\alpha$ is the intercept
- $\beta$ is the slope coefficient (or hedge ratio)
- $u_t$ is the residual term

**Step 2**: Test the residuals $u_t$ for stationarity using the Augmented Dickey-Fuller (ADF) test.

If the residuals are stationary, then the series are cointegrated, and $\beta$ represents the hedge ratio for the pair.

### Important Considerations

1. **Critical Values**: Standard ADF critical values are not appropriate when testing residuals from a cointegrating regression. The critical values need to be adjusted using methods outlined by MacKinnon (2010).

2. **Regression Method Options**:
   - Ordinary Least Squares (OLS): Standard approach
   - Dynamic OLS (DOLS): Includes leads and lags of differenced explanatory variables to account for potential endogeneity
   - Total Least Squares (TLS): Orthogonal regression, useful when both variables contain measurement error

3. **Limitations**:
   - Only suitable for bivariate analysis
   - Sensitive to the ordering of variables (which one is dependent vs. independent)
   - Cannot identify multiple cointegrating relationships

### Implementation Details

Our implementation of the Engle-Granger test includes:
- Support for different regression methods (OLS, DOLS, TLS)
- Automatic handling of missing values
- Calculation of the half-life of mean reversion
- Comprehensive diagnostic information
- Adjusted critical values for residual-based tests

## Johansen Cointegration Test

### Mathematical Foundation

The Johansen test is a maximum likelihood method for determining the number of cointegrating relationships among multiple time series. It's based on the vector error correction model (VECM) representation:

$$\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \mu + \varepsilon_t$$

Where:
- $Y_t$ is a vector of time series
- $\Pi$ is a matrix whose rank determines the number of cointegrating relationships
- $\Gamma_i$ are coefficient matrices for the lagged differences
- $\mu$ is a vector of deterministic terms
- $\varepsilon_t$ is a vector of innovations

The test focuses on the rank of matrix $\Pi$, which is equal to the number of cointegrating relationships.

### Test Statistics

The Johansen procedure produces two test statistics:

1. **Trace Statistic**:
   
   $$\lambda_{trace}(r) = -T \sum_{i=r+1}^{n} \ln(1-\hat{\lambda}_i)$$

   Tests the null hypothesis of at most $r$ cointegrating relations against the alternative of $n$ cointegrating relations.

2. **Maximum Eigenvalue Statistic**:
   
   $$\lambda_{max}(r,r+1) = -T \ln(1-\hat{\lambda}_{r+1})$$

   Tests the null hypothesis of $r$ cointegrating relations against the alternative of $r+1$ cointegrating relations.

Where:
- $T$ is the sample size
- $\hat{\lambda}_i$ are the estimated eigenvalues of $\Pi$ (ordered from largest to smallest)
- $n$ is the number of variables in the system

### Important Considerations

1. **Deterministic Terms**: The inclusion of deterministic terms (constant, trend) affects the distribution of the test statistics. Our implementation supports different specifications:
   - No deterministic terms
   - Restricted constant
   - Unrestricted constant
   - Restricted trend
   - Unrestricted trend

2. **Lag Selection**: The choice of lag length in the VECM model can significantly affect results. Our implementation supports automatic lag selection using information criteria (AIC, BIC).

3. **Advantages Over Engle-Granger**:
   - Can identify multiple cointegrating relationships
   - Not sensitive to variable ordering
   - More powerful in multivariate settings

### Implementation Details

Our implementation of the Johansen test includes:
- Support for different deterministic term specifications
- Automatic lag selection
- Calculation of both trace and maximum eigenvalue statistics
- Extraction of the cointegrating vectors
- P-value approximations based on MacKinnon-Haug-Michelis (1999)

## Half-Life Estimation

Half-life estimation quantifies the speed of mean reversion in the cointegrated relationship and is crucial for assessing the profitability potential of a pairs trading strategy.

### Mathematical Foundation

The half-life is estimated using an Ornstein-Uhlenbeck process, which models mean-reverting behavior:

$$\Delta z_t = (\mu - \theta z_{t-1}) + \varepsilon_t$$

Where:
- $z_t$ is the spread series (residuals from the cointegrating regression)
- $\mu$ is the long-term mean
- $\theta$ is the speed of mean reversion
- $\varepsilon_t$ is the error term

The half-life is calculated as:

$$\text{Half-life} = \frac{\ln(2)}{\theta}$$

### Implementation Details

Our half-life calculation includes:
- Robust estimation methods
- Handling of edge cases (non-stationary and explosive processes)
- Confidence intervals for half-life estimates

## Implementation Details

### Engle-Granger Test Function

Our `engle_granger_test` function:
- Takes two time series as input
- Supports different regression methods and ADF test specifications
- Returns comprehensive diagnostic information
- Includes adjusted critical values for cointegration tests
- Calculates the half-life of mean reversion
- Provides a human-readable conclusion

### Johansen Test Function

Our `johansen_test` function:
- Takes a matrix of time series as input
- Supports different deterministic term specifications and lag structures
- Returns both trace and maximum eigenvalue statistics
- Calculates the number of cointegrating relationships
- Extracts the cointegrating vectors if cointegration exists
- Includes p-value calculations for test statistics

## Usage Examples

### Engle-Granger Test Example

```python
import pandas as pd
import numpy as np
from src.cointegration.statistical_methods import engle_granger_test

# Example with two price series
series1 = pd.Series(data)
series2 = pd.Series(data2)

# Run Engle-Granger test
result = engle_granger_test(
    y=series1,
    x=series2,
    regression_method='ols',
    trend='c',
    maxlag=None,
    autolag='AIC'
)

# Check if cointegrated
if result['is_cointegrated']:
    print(f"Pair is cointegrated with hedge ratio: {result['hedge_ratio']:.4f}")
    print(f"Half-life of mean reversion: {result['half_life']:.2f} periods")
else:
    print("Pair is not cointegrated")
```

### Johansen Test Example

```python
import pandas as pd
import numpy as np
from src.cointegration.statistical_methods import johansen_test

# Example with multiple time series
data = pd.DataFrame({
    'series1': series1,
    'series2': series2,
    'series3': series3
})

# Run Johansen test
result = johansen_test(
    data=data,
    det_order=1,  # Unrestricted constant
    k_ar_diff=1,
    significance_level=0.05
)

# Check number of cointegrating relationships
print(f"Number of cointegrating relationships (trace test): {result['n_cointegrating_relations_trace']}")
print(f"Number of cointegrating relationships (max eigenvalue): {result['n_cointegrating_relations_maxeig']}")

# If cointegrated, examine the cointegrating vectors
if result['n_cointegrating_relations_trace'] > 0:
    print("Cointegrating vectors:")
    print(result['eigenvectors'])
```

## References

1. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica: Journal of the Econometric Society*, 251-276.

2. Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.

3. Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica: Journal of the Econometric Society*, 1551-1580.

4. MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper.

5. MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical distribution functions of likelihood ratio tests for cointegration. *Journal of Applied Econometrics*, 14(5), 563-577. 