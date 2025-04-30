# Advanced Spread Calculation Methods

This document provides a comprehensive explanation of the advanced spread calculation methods implemented in the `SpreadAnalyzer` class. These methods enhance the basic spread calculation with various normalization techniques, volatility adjustments, and multi-timeframe analysis.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Spread Calculation](#basic-spread-calculation)
3. [Advanced Normalization Techniques](#advanced-normalization-techniques)
   - [Z-Score Normalization](#z-score-normalization)
   - [Volatility-Adjusted Spreads](#volatility-adjusted-spreads)
   - [GARCH-Based Volatility Estimation](#garch-based-volatility-estimation)
   - [Alternative Normalization Methods](#alternative-normalization-methods)
4. [Multiple Timeframe Analysis](#multiple-timeframe-analysis)
5. [Adaptive Thresholds](#adaptive-thresholds)
   - [Dynamic Threshold Calculation](#dynamic-threshold-calculation)
   - [Regime-Based Thresholds](#regime-based-thresholds)
6. [Signal Consistency Analysis](#signal-consistency-analysis)
7. [Mathematical Formulations](#mathematical-formulations)
8. [Usage Examples](#usage-examples)
9. [References](#references)

## Introduction

In pairs trading, the spread between two cointegrated assets is a critical component that forms the basis of trading signals. The quality of spread calculation and normalization directly impacts the strategy's performance. Advanced spread calculation methods aim to:

1. Better identify trading opportunities
2. Reduce false signals
3. Adapt to changing market conditions
4. Improve risk management through more accurate spread modeling

## Basic Spread Calculation

At its core, the spread between two assets is calculated as:

```
spread = y - (hedge_ratio * x + intercept)
```

Where:
- `y` is the price of the dependent asset
- `x` is the price of the independent asset
- `hedge_ratio` is the coefficient that defines the relationship between the assets
- `intercept` is an optional constant term (often set to 0)

This is implemented in the `calculate_spread` method of the `SpreadAnalyzer` class.

## Advanced Normalization Techniques

### Z-Score Normalization

The Z-score normalizes the spread based on its historical mean and standard deviation:

```
z-score = (spread - mean) / standard_deviation
```

The `SpreadAnalyzer` class supports multiple methods for calculating the Z-score:

1. **Rolling Window**: Uses a sliding window of historical data
   ```python
   mean = spread.rolling(window=window).mean()
   std = spread.rolling(window=window).std()
   zscore = (spread - mean) / std
   ```

2. **Exponentially Weighted**: Gives more weight to recent observations
   ```python
   mean = spread.ewm(halflife=half_life).mean()
   std = spread.ewm(halflife=half_life).std()
   zscore = (spread - mean) / std
   ```

3. **GARCH-Based**: Uses GARCH models to estimate time-varying volatility
   ```python
   mean = spread.rolling(window=window).mean()
   volatility = GARCH_model(spread)
   zscore = (spread - mean) / volatility
   ```

### Volatility-Adjusted Spreads

Volatility adjustment scales the spread by its recent volatility, making it more comparable across different market regimes:

```python
volatility = spread.rolling(window=vol_window).std()
vol_adjusted_spread = raw_spread / volatility
```

The `calculate_volatility_adjusted_spread` method implements this with several options:

1. **Multiple volatility estimation methods**:
   - Rolling window standard deviation
   - Exponentially weighted standard deviation
   - GARCH model-based volatility

2. **Lookback options**:
   - `'same'`: Uses contemporaneous volatility
   - `'lagged'`: Uses lagged volatility to avoid lookahead bias in backtesting

### GARCH-Based Volatility Estimation

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models capture volatility clustering and are particularly useful for financial time series:

```python
from arch import arch_model

spread_diff = spread.diff().dropna()
model = arch_model(spread_diff, vol='Garch', p=1, q=1)
model_fit = model.fit(disp='off')
volatility = model_fit.conditional_volatility
```

The implementation includes robust error handling and fallback to simpler methods if the `arch` package is not available.

### Alternative Normalization Methods

Beyond traditional Z-score, several alternative methods are implemented in `calculate_normalized_spread`:

1. **Quantile-Based Normalization**:
   - Maps the spread to its quantile in the historical distribution
   - Scales to a range similar to Z-score (-3 to +3)
   - Less affected by outliers than standard Z-score

2. **Percentile Rank Normalization**:
   - Similar to quantile-based but uses percentile rank within the window
   - Produces a uniform distribution of normalized values

3. **Mean Absolute Deviation (MAD) Normalization**:
   - Uses MAD instead of standard deviation
   - More robust to outliers than standard Z-score
   ```
   normalized = (spread - rolling_mean) / rolling_mad
   ```

## Multiple Timeframe Analysis

The `calculate_multitimeframe_spread` method analyzes the spread across different timeframes to identify more robust trading opportunities:

```python
# Example timeframes (in periods)
timeframes = [20, 60, 120]  # Short, medium, and long term
```

For each timeframe, the method calculates:
- Z-score
- Half-life
- Hurst exponent
- Additional spread metrics

This multi-timeframe approach helps:
1. Identify trading signals confirmed across multiple horizons
2. Reduce false positives by requiring consistency
3. Better understand the temporal characteristics of the spread

## Adaptive Thresholds

### Dynamic Threshold Calculation

Instead of fixed Z-score thresholds (e.g., ±2), dynamic thresholds adapt to the specific characteristics of each spread:

```python
# Calculate percentile-based entry thresholds
upper_entry = window_spread.quantile(1 - target_percentile)
lower_entry = window_spread.quantile(target_percentile)

# Calculate exit thresholds (halfway between entry and mean)
mean = window_spread.mean()
upper_exit = (upper_entry + mean) / 2
lower_exit = (lower_entry + mean) / 2
```

This approach:
- Adapts to non-normal spread distributions
- Maintains a consistent false positive rate
- Better captures the unique behavior of each pair

### Regime-Based Thresholds

Different market regimes (high/normal/low volatility) require different thresholds:

```python
# Define volatility regimes based on historical ranking
vol_rank = volatility.rolling(window=252).rank(pct=True)
high_vol = vol_rank > 0.8
normal_vol = (vol_rank >= 0.3) & (vol_rank <= 0.8)
low_vol = vol_rank < 0.3

# Set different thresholds based on regime
# High volatility: wider thresholds
# Low volatility: tighter thresholds
thresholds['entry'][high_vol] = 2.5
thresholds['entry'][normal_vol] = 2.0
thresholds['entry'][low_vol] = 1.5
```

This approach allows the strategy to:
- Reduce false signals during high volatility
- Capture more opportunities during low volatility
- Automatically adapt to changing market conditions

## Signal Consistency Analysis

The `calculate_signal_consistency` method evaluates the consistency of signals across different timeframes:

```python
# Example output
{
    'consistency': 0.75,  # 75% of timeframes agree
    'agreement': False,   # Not all timeframes agree
    'mean_zscore': 2.3,   # Average absolute z-score
    'directions': {
        20: 'long',
        60: 'long',
        120: 'neutral'
    }
}
```

This provides valuable information for:
- Filtering signals based on cross-timeframe agreement
- Assessing signal strength via consistency metrics
- Prioritizing trades with higher consistency scores

## Mathematical Formulations

### Volatility-Adjusted Z-Score

The standard Z-score formula is:

$$Z_t = \frac{S_t - \mu_t}{\sigma_t}$$

Where:
- $S_t$ is the spread at time $t$
- $\mu_t$ is the mean at time $t$
- $\sigma_t$ is the standard deviation at time $t$

For volatility-adjusted spread, we first adjust the raw spread:

$$S_{adj,t} = \frac{S_t}{\sigma_t}$$

Then calculate the Z-score of this adjusted spread:

$$Z_{adj,t} = \frac{S_{adj,t} - \mu_{adj,t}}{\sigma_{adj,t}}$$

### GARCH(1,1) Model

The GARCH(1,1) model for volatility is:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:
- $\omega$ is the long-term variance
- $\alpha$ captures the impact of recent shocks
- $\beta$ represents the persistence of volatility
- $\epsilon_{t-1}$ is the previous period's residual

## Usage Examples

### Basic Spread and Z-Score Calculation

```python
analyzer = SpreadAnalyzer(window_size=60, use_log_prices=True)

# Calculate spread
spread, hedge_ratio = analyzer.calculate_spread(y_price, x_price)

# Calculate z-score
zscore = analyzer.calculate_zscore(spread, method='ewm')
```

### Volatility-Adjusted Spread

```python
# Calculate volatility-adjusted spread
vol_adjusted_spread, hedge_ratio, raw_spread, volatility = analyzer.calculate_volatility_adjusted_spread(
    y_price, 
    x_price, 
    vol_method='garch',
    vol_lookback='lagged'
)
```

### Multi-Timeframe Analysis

```python
# Analyze spread across multiple timeframes
multi_tf_results = analyzer.calculate_multitimeframe_spread(
    y_price,
    x_price,
    timeframes=[20, 60, 120, 252]
)

# Check signal consistency
consistency = multi_tf_results['signal_consistency']['consistency']
if consistency > 0.8 and multi_tf_results['signal_consistency']['mean_zscore'] > 2:
    print("Strong consistent signal detected")
```

### Adaptive Thresholds

```python
# Calculate dynamic thresholds
spread, _ = analyzer.calculate_spread(y_price, x_price)
thresholds = analyzer.dynamic_threshold_calculation(spread, target_percentile=0.05)

# Calculate regime-based thresholds
regime_thresholds = analyzer.regime_based_thresholds(spread)
```

## References

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. John Wiley & Sons.
2. Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
3. Rad, H., Low, R. K. Y., Faff, R. (2016). "The profitability of pairs trading strategies: distance, cointegration and copula methods." *Quantitative Finance*, 16(10), 1541-1558.
4. Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007.
5. Liew, R. Q., Wu, Y. (2013). "Pairs trading: A copula approach." *Journal of Derivatives & Hedge Funds*, 19(1), 12-30. 