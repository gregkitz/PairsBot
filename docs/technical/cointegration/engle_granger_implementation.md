# Engle-Granger Cointegration Test Implementation

This document provides a comprehensive guide to the implementation of the Engle-Granger cointegration test in our quantitative trading framework. It includes mathematical foundations, implementation details, and pseudocode to facilitate understanding and development.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Guidelines](#implementation-guidelines)
4. [Pseudocode](#pseudocode)
5. [Statistical Significance and Interpretation](#statistical-significance-and-interpretation)
6. [Edge Cases and Numerical Stability](#edge-cases-and-numerical-stability)
7. [References](#references)

## Introduction

The Engle-Granger test is a bivariate cointegration testing procedure developed by Nobel laureates Robert F. Engle and Clive W. J. Granger. It is widely used in pairs trading to identify statistically significant long-term relationships between two financial time series.

The method is based on a two-step approach:
1. Estimating the cointegrating relationship using regression
2. Testing the residuals for stationarity

While simpler than the Johansen test, the Engle-Granger approach is particularly well-suited for pairs trading because:
1. It directly estimates the hedge ratio (trading ratio) between two assets
2. It is conceptually simpler and easier to interpret
3. It can be extended to include more sophisticated regression techniques
4. It performs well in the bivariate case common in pairs trading

Our implementation enhances the standard Engle-Granger approach with robust estimation methods and additional diagnostics relevant to trading applications.

## Mathematical Foundation

### The Cointegration Concept

Two time series $X_t$ and $Y_t$ are cointegrated if:
1. Both are integrated of order 1 (I(1)), meaning they become stationary after first differencing
2. There exists a linear combination $Z_t = Y_t - \alpha - \beta X_t$ that is stationary (I(0))

Where:
- $\alpha$ is the intercept term
- $\beta$ is the slope coefficient (hedge ratio)
- $Z_t$ represents the spread or equilibrium error

### The Two-Step Procedure

**Step 1**: Estimate the cointegrating relationship by regressing one series on the other:

$$Y_t = \alpha + \beta X_t + u_t$$

This can be done using various regression methods:
- Ordinary Least Squares (OLS)
- Dynamic OLS (DOLS)
- Total Least Squares (TLS) / Orthogonal regression

**Step 2**: Test the residuals $u_t$ for stationarity using the Augmented Dickey-Fuller (ADF) test:

$$\Delta u_t = \gamma u_{t-1} + \sum_{i=1}^{p} \delta_i \Delta u_{t-i} + \varepsilon_t$$

The null hypothesis is $\gamma = 0$ (unit root, no cointegration).
The alternative hypothesis is $\gamma < 0$ (stationarity, cointegration exists).

### Critical Values for Residual-Based Tests

Standard ADF critical values are not appropriate for testing residuals from a cointegrating regression. Special critical values developed by MacKinnon (2010) must be used instead.

These critical values are more negative (stricter) than standard ADF critical values because the OLS procedure in Step 1 produces residuals that are more likely to appear stationary than the true error process.

### Half-Life Calculation

If cointegration is detected, we calculate the half-life of mean reversion to assess the trading potential:

$$\text{Half-life} = \frac{\ln(2)}{\gamma}$$

Where $\gamma$ is the coefficient on the lagged level term in the ADF regression.

## Implementation Guidelines

### Preprocessing Steps

1. **Data Validation**:
   - Check that both series have the same length
   - Handle missing values through interpolation or other methods
   - Ensure sufficient observations for reliable estimation (preferably 100+ observations)

2. **Unit Root Testing**:
   - Verify that both series are I(1) using ADF or Phillips-Perron tests
   - Check that series are not I(0) or I(2), which would invalidate the cointegration framework

3. **Log Transformation**:
   - Consider using log prices rather than raw prices
   - This often improves the stability of the relationship and addresses heteroscedasticity
   - Especially important for financial time series with exponential growth characteristics

### Regression Method Selection

1. **Ordinary Least Squares (OLS)**:
   - The standard approach, simple and effective
   - Assumes the independent variable is measured without error
   - Suffers from endogeneity bias if there's contemporaneous correlation between regressors and errors

2. **Dynamic OLS (DOLS)**:
   - Adds leads and lags of differenced independent variables
   - Corrects for endogeneity and serial correlation
   - Formula: $Y_t = \alpha + \beta X_t + \sum_{i=-q}^{q} \gamma_i \Delta X_{t+i} + u_t$
   - Recommended when there's suspicion of endogeneity

3. **Total Least Squares (TLS)**:
   - Orthogonal regression that accounts for measurement errors in both variables
   - Minimizes the perpendicular distances from the data points to the fitted line
   - Useful when both series contain measurement error or noise
   - More symmetric treatment of the variables

### ADF Test Configuration

1. **Lag Selection**:
   - Use information criteria (AIC, BIC) for automatic lag selection
   - Consider a maximum lag of $12 * (n/100)^{1/4}$ where $n$ is the sample size
   - Too few lags can lead to residual autocorrelation
   - Too many lags reduce test power

2. **Deterministic Terms**:
   - Include a constant in the ADF test for residuals
   - A trend is typically not included when testing residuals
   - If the spread exhibits a trend, it generally indicates a lack of cointegration

### Result Interpretation

1. **Test Statistic and P-Value**:
   - Compare the ADF test statistic to MacKinnon's critical values
   - Calculate approximate p-values using MacKinnon's response surface
   - Reject the null hypothesis (no cointegration) if p-value < significance level (typically 0.05)

2. **Hedge Ratio**:
   - The estimated $\beta$ coefficient represents the hedge ratio
   - It indicates the relative position sizes for the two assets in a pairs trade
   - A positive $\beta$ indicates a short-long strategy
   - The magnitude of $\beta$ determines the capital allocation ratio

3. **Half-Life**:
   - The half-life indicates how quickly the spread reverts to its mean
   - Trading-relevant half-lives typically range from 1 to 20 periods
   - Very short half-lives may indicate high transaction costs
   - Very long half-lives may indicate weak mean reversion

## Pseudocode

```
function engle_granger_test(y, x, regression_method='ols', trend='c', maxlag=None, autolag='AIC'):
    """
    Implements the Engle-Granger two-step method for testing cointegration
    
    Parameters:
    - y: Dependent variable time series
    - x: Independent variable time series
    - regression_method: Method for estimating cointegrating relationship ('ols', 'dynamic_ols', 'tls')
    - trend: Trend term in ADF regression ('c', 'ct', 'ctt', 'n')
    - maxlag: Maximum number of lags for ADF test
    - autolag: Method to select optimal lag length
    
    Returns:
    - Dictionary with test results
    """
    
    # 1. Input validation
    if len(y) != len(x):
        raise ValueError("Input series must have the same length")
    
    # Handle missing values if necessary
    if has_missing_values(y) or has_missing_values(x):
        y = interpolate_missing_values(y)
        x = interpolate_missing_values(x)
    
    # 2. Step 1: Estimate the cointegrating relationship
    if regression_method == 'ols':
        # Ordinary Least Squares
        X = add_constant(x)
        model = OLS(y, X).fit()
        intercept = model.params[0]
        beta = model.params[1]  # Hedge ratio
        residuals = y - (intercept + beta * x)
        
    elif regression_method == 'dynamic_ols':
        # Dynamic OLS with leads and lags
        # Number of leads and lags based on sample size
        n_leads_lags = min(int(ceil(0.1 * len(x))), 10)
        
        # Create extended regressor matrix with leads and lags
        X_extended = add_constant(x)
        
        for i in range(1, n_leads_lags + 1):
            # Add lags of differenced x
            X_extended = column_stack((X_extended, lag(diff(x), i)))
            
            # Add leads of differenced x
            X_extended = column_stack((X_extended, lead(diff(x), i)))
        
        # Remove rows with NaN values
        valid_idx = non_nan_indices(X_extended)
        X_valid = X_extended[valid_idx]
        y_valid = y[valid_idx]
        
        # Estimate model
        model = OLS(y_valid, X_valid).fit()
        intercept = model.params[0]
        beta = model.params[1]  # Main hedge ratio (contemporaneous effect)
        
        # Calculate residuals using only the contemporary relationship
        residuals = y - (intercept + beta * x)
        
    elif regression_method == 'tls':
        # Total Least Squares (orthogonal regression)
        # Subtract means
        x_mean = mean(x)
        y_mean = mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Calculate the covariance matrix
        cov_matrix = cov(x_centered, y_centered)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(cov_matrix)
        
        # Index of the smallest eigenvalue
        idx = argmin(eigenvalues)
        
        # Get the corresponding eigenvector
        v = eigenvectors[:, idx]
        
        # Calculate beta and intercept
        beta = -v[0] / v[1]
        intercept = y_mean - beta * x_mean
        
        # Calculate residuals
        residuals = y - (intercept + beta * x)
    
    else:
        raise ValueError(f"Unknown regression method: {regression_method}")
    
    # 3. Step 2: Test for stationarity of the residuals using ADF test
    
    # Set maxlag based on sample size if not provided
    if maxlag is None:
        maxlag = int(ceil(12 * (len(residuals) / 100) ** 0.25))
    
    # Perform ADF test on residuals
    adf_result = adfuller(residuals, maxlag=maxlag, regression=trend, autolag=autolag)
    
    # Get test statistic and p-value
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    
    # Get critical values
    critical_values = adf_result[4]
    
    # 4. Determine if the series are cointegrated
    is_cointegrated = p_value < 0.05
    
    # 5. Calculate half-life if cointegrated
    if is_cointegrated:
        half_life = calculate_half_life(residuals)
    else:
        half_life = None
    
    # 6. Prepare results dictionary
    results = {
        "hedge_ratio": beta,
        "intercept": intercept,
        "residuals": residuals,
        "adf_statistic": adf_statistic,
        "p_value": p_value,
        "critical_values": critical_values,
        "is_cointegrated": is_cointegrated,
        "half_life": half_life
    }
    
    # 7. Add human-readable conclusion
    if is_cointegrated:
        if half_life is not None and half_life < 20:
            results['conclusion'] = (
                f"Series are cointegrated with hedge ratio {beta:.4f}. "
                f"The half-life of mean reversion is {half_life:.2f} periods, "
                f"indicating a good mean-reverting relationship."
            )
        else:
            results['conclusion'] = (
                f"Series are cointegrated with hedge ratio {beta:.4f}. "
                f"However, the half-life of mean reversion is "
                f"{'long' if half_life is not None else 'unable to be determined'}, "
                f"which may indicate a weak mean-reverting relationship."
            )
    else:
        results['conclusion'] = (
            f"Series are not cointegrated at the 5% significance level. "
            f"The p-value of {p_value:.4f} exceeds the 0.05 threshold."
        )
    
    return results
```

## Statistical Significance and Interpretation

### Hypothesis Testing Framework

1. **Null Hypothesis (H0)**: The residuals from the cointegrating regression have a unit root (no cointegration).
2. **Alternative Hypothesis (H1)**: The residuals do not have a unit root (cointegration exists).

### Assessing Statistical Significance

1. **Critical Values**:
   - MacKinnon (2010) provides specialized critical values for residual-based cointegration tests
   - These values depend on:
     - Sample size
     - Number of regressors (excluding constant)
     - Significance level (typically 1%, 5%, or 10%)
     - Deterministic terms in the cointegrating regression

2. **P-values**:
   - P-values less than the significance level (typically 0.05) indicate cointegration
   - MacKinnon's response surface approximations can be used to calculate approximate p-values
   - These are more accurate than comparing against tabulated critical values

### Trading Implications

1. **Hedge Ratio Interpretation**:
   - The hedge ratio ($\beta$) determines the relative position sizes
   - For example, if $\beta = 2$, then a position of size 1 in $Y$ should be paired with a position of size 2 in $X$
   - The sign of $\beta$ indicates the direction of the trades

2. **Half-Life and Trading Frequency**:
   - The half-life influences the optimal trading frequency
   - Shorter half-lives support higher-frequency trading strategies
   - A rule of thumb: the holding period should be related to the half-life
   - Typically, consider holding periods of 0.5 to 2 times the half-life

3. **Spread Volatility**:
   - The standard deviation of residuals indicates the spread volatility
   - This affects position sizing and risk management
   - Higher spread volatility suggests wider stop-loss levels but potentially higher returns

## Edge Cases and Numerical Stability

### Common Issues and Solutions

1. **Order of Variables**:
   - Engle-Granger results can depend on which variable is chosen as dependent
   - Solution: Try both orderings or use methods invariant to ordering (like Johansen)
   - TLS regression is less sensitive to variable ordering than OLS

2. **Structural Breaks**:
   - Breaks in the relationship can lead to spurious results
   - Solution: Use rolling windows or regime-switching models
   - Test for stability of the cointegrating relationship over time

3. **Near-Unit Root Processes**:
   - Series that are close to stationary can lead to unreliable inference
   - Solution: Conduct thorough unit root testing before cointegration testing
   - Consider KPSS tests alongside ADF tests for confirmatory analysis

4. **Autocorrelated Residuals**:
   - Residual autocorrelation can distort inference
   - Solution: Use sufficient lags in the ADF test
   - Consider using dynamic regression methods like DOLS

### Implementation Challenges

1. **Missing Values and Data Quality**:
   - Can significantly affect estimation quality
   - Solution: Use robust interpolation methods but be cautious about overreliance on imputed values
   - Document any data cleaning procedures for reproducibility

2. **Computational Efficiency**:
   - Large datasets can lead to computational challenges
   - Solution: Consider batch processing or parallel implementation for large-scale testing
   - Use optimized linear algebra libraries

3. **Multiplicity Problem**:
   - When testing many pairs, false positives become likely
   - Solution: Apply multiple testing corrections (Bonferroni, FDR)
   - Use more stringent significance levels

4. **Numerical Precision**:
   - Especially important for calculating p-values and critical values
   - Solution: Use double precision for all calculations
   - Implement MacKinnon's response surface approximations accurately

## References

1. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica: Journal of the Econometric Society*, 251-276.

2. MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper.

3. Stock, J. H., & Watson, M. W. (1993). A simple estimator of cointegrating vectors in higher order integrated systems. *Econometrica: Journal of the Econometric Society*, 783-820.

4. Phillips, P. C., & Ouliaris, S. (1990). Asymptotic properties of residual based tests for cointegration. *Econometrica: Journal of the Econometric Society*, 165-193.

5. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

6. Murray, M. P. (1994). A drunk and her dog: an illustration of cointegration and error correction. *The American Statistician*, 48(1), 37-39.

7. Zivot, E., & Wang, J. (2006). *Modeling Financial Time Series with S-PLUS®*. Springer Science & Business Media. 