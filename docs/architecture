# Intraday Statistical Arbitrage System for Futures Pairs
## Enhanced Design & Implementation Roadmap

### 1. **Objective & Overview**
Develop a robust intraday statistical arbitrage system for futures pairs that identifies and exploits temporary mispricings between cointegrated instruments. The system focuses on mean-reverting relationships, utilizes optimal entry/exit timing, and implements rigorous risk management suitable for smaller accounts or prop firm capital constraints.

### 2. **Core Components**

#### A. **Pair Selection & Cointegration Framework**
- **Statistical Testing Suite**:
  - **Engle-Granger Two-Step Method**: Regression + ADF test on residuals
  - **Johansen Cointegration Test**: Identify multiple cointegrating relationships (more robust than Engle-Granger alone)
  - **Rolling Window Analysis**: Recalculate cointegration over 60-day windows
  - **Half-Life Estimation**: Ornstein-Uhlenbeck process to quantify mean-reversion speed
  - **Out-of-Sample Validation**: Split historical data into training/validation periods to prevent overfitting
  
- **Pair Universe Management**:
  - **Micros Focus**: Gold/Silver, Crude/Natural Gas, Treasury yield curve spreads
  - **Correlation Filtering**: Minimum 0.7 correlation threshold
  - **Volatility Filtering**: Eliminate pairs with extreme ratio volatility
  - **Liquidity Requirements**: Ensure sufficient intraday volume in both legs
  - **Out-of-Sample Performance**: Only trade pairs with validated cointegration in training and validation sets

#### B. **Spread Calculation & Normalization**
- **Dynamic Hedge Ratio Estimation**:
  - **Kalman Filter Implementation**: Adaptive beta calculation between pairs
  - **Standardization Methods**: z-score calculation on spreads using rolling windows
  - **Volatility Adjustment**: Normalize spread based on recent volatility regime

- **Spread Analytics**:
  - **Mean-Reversion Strength Indicator**: Measure how quickly spreads return to mean
  - **Outlier Detection**: Identify historically extreme spread levels
  - **Seasonality Adjustment**: Account for time-of-day effects on spread behavior

#### C. **Signal Generation Engine**
- **Entry Signal Framework**:
  - **Z-Score Thresholds**: Enter on deviations beyond ±2 standard deviations
  - **Confirmation Filters**: Volume imbalance, momentum exhaustion
  - **Timing Optimization**: Time-of-day filters for highest-probability setups

- **Exit Strategy Components**:
  - **Mean-Reversion Target**: Exit when z-score returns to ±0.5
  - **Maximum Holding Period**: 3-hour limit for intraday positions
  - **Trailing Exits**: Dynamic adjustment of exit threshold based on spread movement

#### D. **Risk & Position Management**
- **Account-Appropriate Position Sizing**:
  - **Adaptive Volatility-Based Sizing**: Scale positions inversely to recent volatility
  - **Volatility Lookback Windows**: Maintain multiple volatility estimates (10-day, 20-day, 30-day)
  - **Maximum Exposure Rules**: Never exceed 15% account margin on any single pair
  - **Volatility Regime Adjustments**: Automatically reduce position size during high volatility periods

- **Risk Controls**:
  - **Stop Loss Mechanism**: Exit if spread exceeds 3x standard deviations
  - **Daily Loss Limits**: Cap daily losses at 1% of account (compatible with prop firm rules)
  - **Correlation Break Protection**: Exit if intraday correlation drops below threshold
  - **Monte Carlo Risk Simulation**: Generate thousands of potential trade sequences based on historical spread behavior to estimate worst-case scenarios

#### E. **Execution Optimization**
- **Order Type Selection**:
  - **Simultaneous Leg Execution**: Reduce execution slippage across pair components
  - **Limit Order Strategies**: Optimal placement relative to bid-ask spread
  - **Market Condition Adaptation**: Switch to market orders during fast-moving conditions

- **Transaction Cost Mitigation**:
  - **Volume-Weighted Execution**: Adjust order timing based on historical volume patterns
  - **Rebalancing Minimization**: Only adjust hedge ratios when significantly changed
  - **Exchange Selection**: Consider spread differentials across venues
  - **Timing Optimization**: Avoid trading during known low-liquidity periods

### 3. **Implementation Roadmap**

#### Phase 1: Foundation & Research (2-3 weeks)
- Set up data pipeline for futures pairs analysis
- Implement cointegration testing framework (Engle-Granger and Johansen)
- Implement out-of-sample validation procedure for pair selection
- Backtest basic z-score strategy on 3-5 most liquid pairs
- Establish baseline performance metrics
- Validate feasibility with target account size

#### Phase 2: Core Strategy Development (3-4 weeks)
- Implement Kalman filter for dynamic hedge ratio estimation
- Develop spread calculation and normalization modules
- Build z-score based entry/exit rules
- Create visualization tools for spread analysis
- Backtest with realistic transaction costs
- Test volume-weighted execution strategies

#### Phase 3: Risk Management Framework (2-3 weeks)
- Implement adaptive volatility-based position sizing algorithm
- Develop stop-loss and maximum holding period logic
- Create daily risk monitoring dashboard
- Add correlation break protection
- Implement Monte Carlo risk testing framework
- Test system with worst-case scenario simulation

#### Phase 4: Execution & Optimization (2-3 weeks)
- Build execution engine with simultaneous order placement
- Implement volume-weighted execution logic
- Optimize order types based on market conditions
- Implement transaction cost analysis
- Develop rebalancing logic to minimize costs
- Perform walk-forward testing

#### Phase 5: Production Deployment (Ongoing)
- Paper trading validation (2 weeks)
- Live trading with minimal size
- Daily performance monitoring
- Periodic pair relationship re-evaluation 
- System refinement based on performance

### 4. **Key Integration Code Elements**

```python
def test_cointegration(price_series1, price_series2, window=60, test_type='both', train_test_split=0.7):
    """
    Test for cointegration between two price series using both Engle-Granger and Johansen methods
    with out-of-sample validation.
    
    Parameters:
    - price_series1: Series of prices for first instrument
    - price_series2: Series of prices for second instrument
    - window: Rolling window size in days
    - test_type: 'engle-granger', 'johansen', or 'both'
    - train_test_split: Proportion of data to use for training vs validation
    
    Returns:
    - Dictionary with cointegration test results, hedge ratio, and validation results
    """
    results = []
    
    # Convert to log prices for better stationarity properties
    log_price1 = np.log(price_series1)
    log_price2 = np.log(price_series2)
    
    # Split data into training and validation sets
    split_idx = int(len(log_price1) * train_test_split)
    train_log_price1 = log_price1[:split_idx]
    train_log_price2 = log_price2[:split_idx]
    valid_log_price1 = log_price1[split_idx:]
    valid_log_price2 = log_price2[split_idx:]
    
    for i in range(window, len(train_log_price1)):
        # Get window of data
        y = train_log_price1[i-window:i]
        x = train_log_price2[i-window:i]
        
        result_dict = {'date': train_log_price1.index[i]}
        
        if test_type in ['engle-granger', 'both']:
            # Engle-Granger test
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            beta = model.params[1]  # Hedge ratio
            residuals = y - (model.params[0] + beta * x)
            adf_result = adfuller(residuals)
            
            result_dict.update({
                'hedge_ratio': beta,
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_cointegrated_eg': adf_result[1] < 0.05,
                'half_life': calculate_half_life(residuals)
            })
        
        if test_type in ['johansen', 'both']:
            # Johansen test
            data = pd.concat([y, x], axis=1)
            johansen_result = coint_johansen(data, 0, 1)
            
            result_dict.update({
                'johansen_stat': johansen_result.lr1[0],
                'johansen_cv': johansen_result.cvt[0, 1],  # 5% critical value
                'is_cointegrated_j': johansen_result.lr1[0] > johansen_result.cvt[0, 1]
            })
        
        results.append(result_dict)
    
    # Perform out-of-sample validation
    train_results = pd.DataFrame(results).set_index('date')
    
    # Use the last hedge ratio from training to test on validation set
    last_beta = train_results['hedge_ratio'].iloc[-1] if 'hedge_ratio' in train_results else None
    
    if last_beta is not None:
        # Calculate spread in validation set using training hedge ratio
        valid_residuals = valid_log_price1 - (model.params[0] + last_beta * valid_log_price2)
        valid_adf_result = adfuller(valid_residuals)
        
        # Add validation results
        train_results['valid_adf_statistic'] = valid_adf_result[0]
        train_results['valid_p_value'] = valid_adf_result[1]
        train_results['valid_is_cointegrated'] = valid_adf_result[1] < 0.05
        train_results['valid_half_life'] = calculate_half_life(valid_residuals)
    
    return train_results
```

```python
def adaptive_position_sizing(account_size, spread_data, lookback_windows=[10, 20, 30], 
                            max_risk=0.01, max_allocation=0.15):
    """
    Calculate position size based on adaptive volatility measures.
    
    Parameters:
    - account_size: Size of trading account
    - spread_data: Historical spread data
    - lookback_windows: List of periods for volatility calculation
    - max_risk: Maximum account risk per trade
    - max_allocation: Maximum allocation to any pair
    
    Returns:
    - Position size for each leg
    """
    # Calculate volatilities for multiple windows
    volatilities = {}
    for window in lookback_windows:
        volatilities[window] = spread_data.ewm(span=window).std().iloc[-1]
    
    # Use the maximum volatility as a conservative estimate
    max_volatility = max(volatilities.values())
    
    # Basic position size calculation
    position_size = (account_size * max_risk) / (max_volatility * account_size)
    
    # Apply volatility scaling
    vol_ratio = volatilities[lookback_windows[0]] / volatilities[lookback_windows[-1]]
    
    # If recent volatility is higher than long-term volatility, reduce size
    if vol_ratio > 1.2:  # Recent volatility is 20% higher
        position_size = position_size * (1 / vol_ratio)
    
    # Enforce maximum allocation constraint
    position_size = min(position_size, max_allocation)
    
    # Additional constraint for small accounts
    if account_size < 25000:
        position_size = min(position_size, 0.1)  # More conservative for small accounts
    
    return position_size
```

```python
def monte_carlo_risk_analysis(spread_data, model_params, num_simulations=1000, max_holding_period=3*60):
    """
    Perform Monte Carlo simulation to estimate worst-case scenarios.
    
    Parameters:
    - spread_data: Historical spread data
    - model_params: Dictionary with model parameters
    - num_simulations: Number of Monte Carlo simulations to run
    - max_holding_period: Maximum holding period in minutes
    
    Returns:
    - Dictionary with risk metrics (VaR, maximum drawdown, etc.)
    """
    # Fit Ornstein-Uhlenbeck parameters to the spread
    ou_params = fit_ou_process(spread_data)
    
    # Generate simulations
    simulations = []
    for _ in range(num_simulations):
        sim_path = simulate_ou_process(
            s0=spread_data.iloc[-1],
            theta=ou_params['theta'],
            mu=ou_params['mu'],
            sigma=ou_params['sigma'],
            n_steps=max_holding_period
        )
        simulations.append(sim_path)
    
    # Convert to DataFrame for analysis
    sim_df = pd.DataFrame(simulations)
    
    # Calculate risk metrics
    risk_metrics = {
        'var_95': np.percentile(sim_df.min(axis=1), 5),  # 95% VaR
        'var_99': np.percentile(sim_df.min(axis=1), 1),  # 99% VaR
        'expected_shortfall': sim_df.min(axis=1)[sim_df.min(axis=1) <= np.percentile(sim_df.min(axis=1), 5)].mean(),
        'max_favorable': np.percentile(sim_df.max(axis=1), 95),
        'probability_loss': (sim_df.iloc[:, -1] < 0).mean(),
        'probability_stop_hit': (sim_df.min(axis=1) < model_params['stop_threshold']).mean()
    }
    
    return risk_metrics
```

```python
def volume_weighted_execution(order, market_data, volume_profile):
    """
    Implement volume-weighted execution strategy.
    
    Parameters:
    - order: Order details (instrument, size, direction)
    - market_data: Current market data
    - volume_profile: Historical volume profile by time of day
    
    Returns:
    - Execution plan with timing and sizes
    """
    current_time = pd.Timestamp.now().time()
    time_bracket = find_time_bracket(current_time, volume_profile)
    
    # Get volume expectations for current and upcoming time brackets
    current_vol_pct = volume_profile.loc[time_bracket, 'volume_percentage']
    upcoming_brackets = get_upcoming_brackets(time_bracket, volume_profile, periods=3)
    
    # Determine if we should execute now or wait
    if current_vol_pct < 0.02:  # Less than 2% of daily volume
        if any(volume_profile.loc[upcoming_brackets, 'volume_percentage'] > 0.05):
            # Wait for higher volume period
            next_high_vol = upcoming_brackets[
                volume_profile.loc[upcoming_brackets, 'volume_percentage'].argmax()]
            
            return {
                'action': 'defer',
                'optimal_time': next_high_vol,
                'reason': 'Low current volume, higher volume expected soon'
            }
    
    # Determine order type based on volume and volatility
    if current_vol_pct > 0.08:  # High volume period
        order_type = 'limit'
        # Place limit slightly better than mid for high volume periods
        limit_adjustment = 0.25  # 25% of the spread
    else:
        # Use market orders during lower volume to ensure execution
        order_type = 'market'
        limit_adjustment = None
    
    return {
        'action': 'execute',
        'order_type': order_type,
        'limit_adjustment': limit_adjustment,
        'volume_context': f"Current period represents {current_vol_pct:.1%} of average daily volume"
    }
```

### 5. **System Architecture**

```
flowchart TD
    A[Market Data Feeds] --> B[Cointegration Testing]
    A --> C[Pair Selection Module]
    
    B --> B1[Engle-Granger Test]
    B --> B2[Johansen Test]
    B1 & B2 --> B3[Out-of-Sample Validation]
    
    B3 --> D[Pair Universe Database]
    C --> D
    
    D --> E[Spread Calculation Engine]
    E --> E1[Static Hedge Ratio]
    E --> E2[Kalman Filter Ratio]
    
    E1 & E2 --> F[Z-Score Calculation]
    F --> G[Signal Generation]
    
    G --> G1[Entry Rules]
    G --> G2[Exit Rules]
    G1 & G2 --> H[Risk Management]
    
    H --> H1[Adaptive Volatility-Based Position Sizing]
    H --> H2[Stop Loss Logic]
    H --> H3[Max Holding Time]
    H --> H4[Monte Carlo Risk Simulation]
    
    H1 & H2 & H3 & H4 --> I[Order Execution Engine]
    I --> I1[Volume-Weighted Execution]
    I --> I2[Simultaneous Order Placement]
    
    I1 & I2 --> K[Performance Analytics]
    K --> L[Pair Re-evaluation Module]
    L --> B
```

### 6. **Implementation Advantages**

- **Lower Capital Requirements**: Viable with micro futures and small accounts (<$20K)
- **Enhanced Robustness**: Out-of-sample validation prevents trading false cointegration relationships
- **Adaptive Risk Management**: Position sizing automatically adjusts to changing market conditions
- **Execution Optimization**: Volume-weighted execution improves entry and exit prices
- **Comprehensive Risk Assessment**: Monte Carlo simulations provide deeper understanding of potential losses
- **Intraday Focus**: Reduced overnight risk and daily reset of positions
- **Market Neutrality**: Inherent hedging provides protection during market stress
- **Flexibility**: Can be expanded to include more pairs as account grows
- **Prop Firm Compatibility**: Conservative risk management aligns with prop firm rules

### 7. **Final Vision**

An adaptive, self-adjusting statistical arbitrage system for futures pairs that:
- **Identifies** optimal cointegrated pairs with multiple robust statistical methods
- **Validates** pair relationships out-of-sample to prevent overfitting
- **Calculates** dynamic hedge ratios using Kalman filtering
- **Generates** high-probability mean-reversion signals based on z-score thresholds
- **Manages** risk appropriately with adaptive position sizing and Monte Carlo simulation
- **Executes** trades efficiently with volume-weighted execution strategies
- **Continuously evaluates** pair relationships to maintain strategy edge
- **Scales** from micro contracts to full-sized contracts as account equity grows

This enhanced system design provides a practical and implementable approach to statistical arbitrage that can be deployed with modest capital while maintaining robust risk management. Its intraday focus minimizes overnight risk and allows for daily performance assessment, making it ideal for both individual traders with smaller accounts and prop firm traders operating under strict risk parameters. The integration of out-of-sample validation, adaptive position sizing, volume-weighted execution, and Monte Carlo risk testing significantly improves the robustness and expected performance of the system.


---

Insights on Pairs Trading from the Books

Pairs trading is a statistical arbitrage strategy that aims to capitalize on price divergences between two historically correlated assets. The books you provided discuss multiple methodologies, challenges, and implementations of pairs trading. Below are key insights:

⸻

1. Approaches to Pairs Trading

(a) Distance Approach (Correlation-Based)
	•	Methodology: Identifies pairs using correlation between price movements or returns.
	•	Pros: Computationally fast, enabling large-scale implementation.
	•	Cons: Cointegrated pairs often have higher volatility in spread, making distance-based pairs less profitable ￼.

(b) Cointegration Approach
	•	Methodology: Uses econometric techniques like the Engle-Granger and Johansen tests to identify pairs that have a long-term stable relationship.
	•	Pros: More reliable than correlation-based methods in identifying tradable pairs.
	•	Cons: Computationally expensive, and large-scale screening requires data snooping bias mitigation ￼.

(c) Time-Series & Mean Reversion Approach
	•	Methodology: Models the spread as a mean-reverting stochastic process (e.g., Ornstein-Uhlenbeck model).
	•	Pros: Optimizes entry/exit points based on expected reversion speed.
	•	Cons: Requires a deep understanding of statistical processes ￼.

(d) Machine Learning (ML)-Based Approach
	•	Methodology: Uses supervised and unsupervised learning models (e.g., PCA, Copulas) to identify pairs with predictive power.
	•	Pros: Captures nonlinear relationships between assets, improving prediction accuracy.
	•	Cons: Requires large amounts of clean financial data and high computational resources ￼.

⸻

2. Key Challenges

(a) Short-Sale Constraints
	•	Some pairs require shorting a stock, which can lead to forced liquidation if the asset becomes hard to borrow ￼.

(b) Regime Shifts
	•	Pairs that appear cointegrated can break down due to changes in market structure, business models, or external shocks (e.g., oil price shocks affecting gold miners) ￼.

(c) Trade Execution Complexity
	•	Bid-ask spreads, dark pools, and high-frequency trading (HFT) can make intraday execution difficult, causing slippage ￼.

⸻

3. Code Example: Pairs Trading Strategy (Python)

The books contain several implementations of pairs trading, including Kalman filters and cointegration-based models. Below is a simplified cointegration-based pairs trading strategy using Python.

Step 1: Import Dependencies

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

Step 2: Load Data & Select a Pair

# Load price data (assumed to be pre-cleaned)
df = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)

# Select two stocks
stock1 = df['AAPL']
stock2 = df['MSFT']

Step 3: Check for Cointegration

score, p_value, _ = coint(stock1, stock2)
print(f'Cointegration Test p-value: {p_value}')

	•	If p_value < 0.05, the pair is cointegrated and suitable for trading.

Step 4: Compute the Spread & Define Trading Signals

# Compute hedge ratio using OLS regression
X = sm.add_constant(stock1)
model = sm.OLS(stock2, X).fit()
hedge_ratio = model.params[1]

# Compute spread
spread = stock2 - hedge_ratio * stock1

# Define entry/exit signals
zscore = (spread - spread.mean()) / spread.std()
long_signal = zscore < -2  # Buy spread
short_signal = zscore > 2   # Sell spread
exit_signal = abs(zscore) < 0.5  # Close position

Step 5: Backtesting & Evaluation

plt.figure(figsize=(12,6))
plt.plot(spread, label='Spread')
plt.axhline(spread.mean(), color='black', linestyle='--', label='Mean')
plt.axhline(spread.mean() + 2*spread.std(), color='red', linestyle='--', label='Sell Threshold')
plt.axhline(spread.mean() - 2*spread.std(), color='green', linestyle='--', label='Buy Threshold')
plt.legend()
plt.title("Pairs Trading: AAPL vs MSFT Spread")
plt.show()

This basic pairs trading strategy can be expanded with Kalman filters, rolling cointegration tests, and machine learning methods to improve robustness ￼.

⸻

4. Advanced Enhancements
	•	Dynamic Hedge Ratio: Use a Kalman filter instead of static OLS regression ￼.
	•	Risk Management: Implement stop-loss rules to prevent large losses.
	•	Intraday Trading: Adapt entry-exit signals to real-time market conditions ￼.

⸻

Final Thoughts
	•	Pairs trading remains a profitable strategy, but success depends on execution efficiency and proper pair selection.
	•	Statistical cointegration is more reliable than correlation-based approaches.
	•	Machine Learning models can enhance pair selection but require extensive data and computational power.
	•	Risk management is crucial to avoid short squeezes and market regime shifts.
