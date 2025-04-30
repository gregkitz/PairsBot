# Johansen Cointegration Test Implementation

This document provides a comprehensive guide to the implementation of the Johansen cointegration test in our quantitative trading framework. The document includes mathematical foundations, implementation details, and pseudocode to facilitate understanding and development.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Guidelines](#implementation-guidelines)
4. [Pseudocode](#pseudocode)
5. [Statistical Significance and Interpretation](#statistical-significance-and-interpretation)
6. [Edge Cases and Numerical Stability](#edge-cases-and-numerical-stability)
7. [References](#references)

## Introduction

The Johansen cointegration test is a multivariate statistical framework that allows us to test for the presence of cointegration relationships among multiple time series variables. Unlike the Engle-Granger approach, which is limited to bivariate cases, the Johansen test can identify multiple cointegrating relationships in a system of variables.

The test is particularly valuable in pairs trading because:
1. It can handle more than two time series simultaneously
2. It's invariant to the choice of normalization (which series is dependent vs. independent)
3. It provides estimates of all possible cointegrating vectors
4. It allows for formal hypothesis testing about the cointegrating relationships

Our implementation is based on Johansen's original 1988 and 1991 papers, with additional numerical improvements for stability and robustness.

## Mathematical Foundation

### Vector Error Correction Model (VECM)

The Johansen test is based on the Vector Error Correction Model (VECM) representation of a Vector Autoregressive (VAR) process:

$$\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta Y_{t-i} + \mu + \varepsilon_t$$

Where:
- $Y_t$ is an $n \times 1$ vector of time series variables
- $\Delta Y_t = Y_t - Y_{t-1}$ represents first differences
- $\Pi = \alpha \beta'$ where $\alpha$ and $\beta$ are $n \times r$ matrices
- $\beta$ contains the cointegrating vectors
- $\alpha$ contains the adjustment coefficients
- $\Gamma_i$ are $n \times n$ matrices of short-run coefficients
- $\mu$ is a vector of deterministic terms (constants, trends)
- $\varepsilon_t$ is a vector of innovations
- $k$ is the lag order of the underlying VAR model

### Matrix $\Pi$ and Its Rank

The key to the Johansen procedure is the rank of matrix $\Pi$:
- If rank($\Pi$) = 0, there is no cointegration
- If rank($\Pi$) = $n$, all series are stationary
- If 0 < rank($\Pi$) = $r$ < $n$, there are $r$ cointegrating relationships

### Maximum Likelihood Estimation

The Johansen procedure uses maximum likelihood to estimate the VECM and test for the rank of $\Pi$. The approach involves:

1. Regressing $\Delta Y_t$ on lagged differences $\Delta Y_{t-1}, \ldots, \Delta Y_{t-k+1}$ to get residuals $R_{0t}$
2. Regressing $Y_{t-1}$ on the same lagged differences to get residuals $R_{1t}$
3. Computing the sample covariance matrices:
   - $S_{00} = \frac{1}{T}\sum_{t=1}^{T} R_{0t} R_{0t}'$
   - $S_{01} = \frac{1}{T}\sum_{t=1}^{T} R_{0t} R_{1t}'$
   - $S_{10} = S_{01}'$
   - $S_{11} = \frac{1}{T}\sum_{t=1}^{T} R_{1t} R_{1t}'$
4. Solving the eigenvalue problem:
   - $|\lambda S_{11} - S_{10} S_{00}^{-1} S_{01}| = 0$
5. Find the eigenvalues $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_n$ and corresponding eigenvectors $v_1, v_2, \ldots, v_n$
6. The cointegrating vectors are the eigenvectors corresponding to the $r$ largest eigenvalues

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
- $\hat{\lambda}_i$ are the estimated eigenvalues (ordered from largest to smallest)
- $n$ is the number of variables in the system

### Deterministic Terms

The distribution of the test statistics depends on the deterministic terms included in the model:

1. **Model 0**: No deterministic terms
2. **Model 1**: Restricted constant (within cointegration space)
3. **Model 2**: Unrestricted constant
4. **Model 3**: Restricted linear trend (within cointegration space)
5. **Model 4**: Unrestricted linear trend

Different critical values apply to each model specification.

## Implementation Guidelines

### Preprocessing Steps

1. **Data Validation**:
   - Check for missing values and handle them appropriately (interpolation, removal)
   - Ensure sufficient observations for reliable estimation
   - Verify that input variables are I(1) (integrated of order 1)

2. **Lag Selection**:
   - Determine optimal lag length using information criteria (AIC, BIC, HQ)
   - Consider using multiple criteria and select the most appropriate lag
   - Avoid over-parameterization which could lead to inefficient estimates

3. **Model Selection**:
   - Choose the appropriate deterministic specification based on visual inspection of the data
   - When in doubt, use model 2 (unrestricted constant) as a reasonable default

### Core Algorithm

The core algorithm involves these key steps:

1. **Construct the VECM**:
   - Transform input variables to first differences
   - Create lagged differences according to selected lag order
   - Incorporate deterministic terms based on model specification

2. **Compute Residuals and Covariance Matrices**:
   - Run auxiliary regressions to get residuals $R_{0t}$ and $R_{1t}$
   - Calculate sample covariance matrices $S_{00}$, $S_{01}$, $S_{10}$, and $S_{11}$

3. **Solve the Eigenvalue Problem**:
   - Use numerical linear algebra routines to compute eigenvalues and eigenvectors
   - Ensure numerical stability, especially for near-singular matrices

4. **Calculate Test Statistics**:
   - Compute trace and maximum eigenvalue statistics
   - Compare against appropriate critical values based on model specification

5. **Determine Cointegration Rank**:
   - Use sequential testing procedure to determine rank
   - Start with null hypothesis of no cointegration (r=0)
   - Increase r until failing to reject the null hypothesis

### Post-processing

1. **Extract Cointegrating Vectors**:
   - If cointegration is detected, extract the eigenvectors corresponding to significant eigenvalues
   - Normalize the cointegrating vectors as needed

2. **Calculate Adjustment Coefficients**:
   - Compute the alpha coefficients that represent the speed of adjustment

3. **Perform Restrictions Testing**:
   - Test specific restrictions on the cointegrating vectors if applicable

4. **Generate Diagnostic Information**:
   - P-values for test statistics
   - Eigenvalues and their statistical significance
   - Model fit statistics

## Pseudocode

```
function johansen_test(data, det_order, k_ar_diff, significance_level):
    """
    Implements Johansen cointegration test
    
    Parameters:
    - data: Matrix of time series variables
    - det_order: Deterministic terms specification (0 to 4)
    - k_ar_diff: Lag order for VECM (VAR order minus 1)
    - significance_level: Significance level for critical values
    
    Returns:
    - Dictionary with test results
    """
    
    # 1. Input validation
    if number_of_variables < 2:
        raise ValueError("At least two time series are required")
    
    # Handle missing values if necessary
    if has_missing_values(data):
        data = interpolate_missing_values(data)
    
    # 2. Prepare data
    Y = data  # Original data matrix
    T = number_of_rows(Y)
    n = number_of_columns(Y)
    dY = diff(Y)  # First differences
    
    # 3. Create lags
    Z = []  # Matrix of lagged differences
    for i in range(1, k_ar_diff):
        Z.append(lag(dY, i))
    
    # Add deterministic terms
    if det_order >= 1:  # Add constant
        Z.append(ones(T))
    if det_order >= 3:  # Add trend
        Z.append(trend(T))
    
    # 4. Run auxiliary regressions
    # Regress dY on Z to get R0
    R0 = residuals(dY ~ Z)
    
    # Regress Y_t-1 on Z to get R1
    Y_lag1 = lag(Y, 1)
    R1 = residuals(Y_lag1 ~ Z)
    
    # 5. Compute moment matrices
    S00 = (R0' * R0) / T
    S01 = (R0' * R1) / T
    S10 = S01'
    S11 = (R1' * R1) / T
    
    # 6. Solve eigenvalue problem
    S11_inv_sqrt = matrix_sqrt_inverse(S11)
    M = S11_inv_sqrt * S10 * inv(S00) * S01 * S11_inv_sqrt
    eigenvalues, eigenvectors = eigen(M)
    
    # Sort eigenvalues in descending order
    sort_desc(eigenvalues, eigenvectors)
    
    # 7. Calculate test statistics
    trace_stat = []
    max_eig_stat = []
    
    for i in range(n):
        # Trace statistic for H0: rank <= i
        trace_i = -T * sum(log(1 - eigenvalues[i:]))
        trace_stat.append(trace_i)
        
        # Max eigenvalue statistic for H0: rank = i vs. H1: rank = i+1
        max_eig_i = -T * log(1 - eigenvalues[i])
        max_eig_stat.append(max_eig_i)
    
    # 8. Get critical values
    trace_cv = get_trace_critical_values(n, det_order, significance_level)
    max_eig_cv = get_max_eig_critical_values(n, det_order, significance_level)
    
    # 9. Determine cointegration rank
    # Trace test
    r_trace = 0
    for i in range(n):
        if trace_stat[i] > trace_cv[i]:
            r_trace = i + 1
    
    # Max eigenvalue test
    r_max_eig = 0
    for i in range(n):
        if max_eig_stat[i] > max_eig_cv[i]:
            r_max_eig = i + 1
    
    # 10. Extract cointegrating vectors if cointegration exists
    if r_trace > 0:
        # Transform eigenvectors to get cointegrating vectors
        beta = S11_inv_sqrt * eigenvectors[:, :r_trace]
        
        # Calculate adjustment coefficients
        alpha = S01 * beta * inv(beta' * S11 * beta)
    else:
        beta = None
        alpha = None
    
    # 11. Compute p-values
    p_values_trace = calculate_p_values(trace_stat, n, det_order, "trace")
    p_values_max_eig = calculate_p_values(max_eig_stat, n, det_order, "max_eig")
    
    # 12. Return results
    return {
        "trace_statistic": trace_stat,
        "max_eigenvalue_statistic": max_eig_stat,
        "trace_critical_values": trace_cv,
        "max_eigenvalue_critical_values": max_eig_cv,
        "eigenvalues": eigenvalues,
        "cointegrating_vectors": beta,
        "adjustment_coefficients": alpha,
        "n_cointegrating_relations_trace": r_trace,
        "n_cointegrating_relations_max_eig": r_max_eig,
        "p_values_trace": p_values_trace,
        "p_values_max_eig": p_values_max_eig
    }
```

## Statistical Significance and Interpretation

### Interpreting Test Results

1. **Trace Test vs. Maximum Eigenvalue Test**:
   - The trace test tends to find more cointegrating relationships than the maximum eigenvalue test
   - In case of conflicting results, examine both and consider the specific application context
   - The maximum eigenvalue test often has better small-sample properties

2. **Critical Values**:
   - Critical values depend on:
     - Sample size (asymptotic vs. small sample)
     - Model specification (deterministic terms)
     - Number of variables
     - Significance level

3. **P-values**:
   - P-values provide a more nuanced view of statistical significance
   - MacKinnon-Haug-Michelis (1999) provide response surface approximations for p-values

### Economic Interpretation

1. **Cointegrating Vectors**:
   - Each cointegrating vector represents a long-run equilibrium relationship
   - The coefficients can be interpreted as relative weights in the equilibrium relationship
   - Normalize coefficients for easier interpretation

2. **Adjustment Coefficients**:
   - Alpha coefficients indicate the speed of adjustment to disequilibrium
   - Larger alphas indicate faster adjustment
   - The sign of alpha indicates the direction of adjustment

3. **Common Trends**:
   - In a system with n variables and r cointegrating relationships, there are (n-r) common stochastic trends
   - These common trends drive the long-run behavior of the system

## Edge Cases and Numerical Stability

### Potential Issues and Solutions

1. **Near Unit Roots**:
   - When one or more variables are nearly stationary (near I(0))
   - Solution: Conduct preliminary unit root tests to identify I(0) variables

2. **Near-Singular Matrices**:
   - Can arise with highly correlated variables or small samples
   - Solution: Use regularization techniques like ridge regression or singular value decomposition with truncation

3. **Outliers and Structural Breaks**:
   - Can distort estimation and inference
   - Solution: Use robust covariance estimators or include dummy variables for identified breaks

4. **Small Sample Issues**:
   - Johansen test statistics can be biased in small samples
   - Solution: Apply Bartlett small-sample corrections or bootstrap methods

5. **Lag Selection**:
   - Too few lags can result in autocorrelated errors
   - Too many lags can lead to loss of power
   - Solution: Use multiple information criteria and diagnostic checks for residual autocorrelation

### Numerical Implementation Tips

1. **Use Stable Eigenvalue Algorithms**:
   - Prefer algorithms that compute eigenvalues directly rather than through characteristic polynomials
   - Use specialized libraries like LAPACK for numerical linear algebra

2. **Matrix Inversion**:
   - Avoid explicit matrix inversion when possible
   - Use Cholesky decomposition for positive definite matrices
   - Consider using QR decomposition or singular value decomposition for general matrices

3. **Scaling Variables**:
   - Consider scaling variables to similar magnitudes to improve numerical stability
   - Document any scaling applied for reproducibility

4. **Precision Issues**:
   - Use at least double precision floating point for calculations
   - Be aware of catastrophic cancellation in certain calculations

## References

1. Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.

2. Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica: Journal of the Econometric Society*, 1551-1580.

3. Johansen, S. (1995). *Likelihood-based Inference in Cointegrated Vector Autoregressive Models*. Oxford University Press.

4. MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical distribution functions of likelihood ratio tests for cointegration. *Journal of Applied Econometrics*, 14(5), 563-577.

5. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer Science & Business Media.

6. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

7. Juselius, K. (2006). *The Cointegrated VAR Model: Methodology and Applications*. Oxford University Press. 