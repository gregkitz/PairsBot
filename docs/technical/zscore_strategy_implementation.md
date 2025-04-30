# Z-Score Strategy Implementation

This document provides a comprehensive overview of the Z-Score Strategy implementation for pairs trading in our system. The Z-Score Strategy is the foundational pairs trading approach implemented in Phase 1.

## Table of Contents
1. [Overview](#overview)
2. [Strategy Concept](#strategy-concept)
3. [Implementation Components](#implementation-components)
4. [Signal Generation](#signal-generation)
5. [Position Management](#position-management)
6. [Transaction Costs](#transaction-costs)
7. [Performance Metrics](#performance-metrics)
8. [Configuration Parameters](#configuration-parameters)
9. [Usage Examples](#usage-examples)
10. [Extensions](#extensions)
11. [References](#references)

## Overview

The Z-Score Strategy is a statistical arbitrage approach for pairs trading that capitalizes on the mean-reverting property of cointegrated asset pairs. The implementation calculates the spread between two assets, normalizes it as a z-score, and generates trading signals when the z-score exceeds specified thresholds.

The strategy is implemented in `src/backtest/zscore_strategy_backtest.py` with the primary class being `ZScoreStrategyBacktest`. This implementation includes:

- Spread calculation with dynamic or static hedge ratios
- Z-score normalization using multiple methods
- Signal generation based on configurable thresholds
- Position management with risk controls
- Transaction cost modeling
- Comprehensive performance metrics

## Strategy Concept

### Theoretical Foundation

Pairs trading with the Z-Score Strategy relies on the following key concepts:

1. **Cointegration**: Two price series are cointegrated if a linear combination of them is stationary, even though the individual series may be non-stationary.

2. **Mean Reversion**: The spread between cointegrated assets tends to revert to its mean over time.

3. **Z-Score Normalization**: By calculating the z-score of the spread (number of standard deviations from the mean), we can identify statistically significant deviations.

4. **Statistical Arbitrage**: When the z-score exceeds certain thresholds, we take positions to capitalize on the expected mean reversion.

### Trading Logic

The basic trading logic is:

1. When the z-score exceeds a positive threshold (e.g., +2.0), the spread is considered too wide, so we:
   - Sell the overvalued asset (numerator of the spread)
   - Buy the undervalued asset (denominator of the spread)

2. When the z-score exceeds a negative threshold (e.g., -2.0), the spread is considered too narrow, so we:
   - Buy the undervalued asset (numerator of the spread)
   - Sell the overvalued asset (denominator of the spread)

3. When the z-score returns to a value close to zero (e.g., between -0.5 and +0.5), we close positions to realize profits.

## Implementation Components

### Core Components

The Z-Score Strategy implementation consists of the following core components:

1. **Hedge Ratio Calculation**
   - OLS regression to determine the optimal hedge ratio
   - Implementation in `calculate_hedge_ratio()` method

2. **Spread Calculation**
   - Linear combination of assets using the hedge ratio
   - Implementation in `calculate_spread()` method
   - Options for log prices or raw prices

3. **Z-Score Calculation**
   - Multiple methods: rolling window, exponentially weighted, full history
   - Implementation in `calculate_zscore()` method

4. **Signal Generation**
   - Threshold-based entry and exit signals
   - Implementation in `generate_signals()` method

5. **Position Management**
   - Long/short position tracking
   - Stop-loss and maximum holding period constraints
   - Implementation in `apply_position_logic()` method

6. **Returns Calculation**
   - Proper handling of trading costs
   - Position-based return calculation
   - Implementation in `calculate_returns()` method

7. **Performance Analytics**
   - Comprehensive metrics calculation
   - Risk-adjusted performance measures
   - Implementation in `_calculate_metrics()` method

### Class Structure

The main class `ZScoreStrategyBacktest` contains:

```
ZScoreStrategyBacktest
├── __init__                # Initialize with configuration parameters
├── calculate_hedge_ratio   # Calculate optimal hedge ratio between assets
├── calculate_spread        # Calculate spread series using hedge ratio
├── calculate_zscore        # Calculate z-score of the spread
├── generate_signals        # Generate entry/exit signals based on z-score
├── apply_position_logic    # Apply position management rules
├── calculate_returns       # Calculate returns with transaction costs
├── backtest                # Run complete backtest with all components
├── _generate_trade_history # Create detailed trade history
├── _calculate_metrics      # Calculate performance metrics
├── plot_results            # Visualize backtest results
└── save_results            # Save results to file
```

## Signal Generation

### Z-Score Calculation Methods

The implementation supports three methods for z-score calculation:

1. **Rolling Window** (`'rolling'`):
   ```python
   spread_mean = spread.rolling(window=window_size).mean()
   spread_std = spread.rolling(window=window_size).std()
   zscore = (spread - spread_mean) / spread_std
   ```

2. **Exponentially Weighted Moving Average** (`'ewm'`):
   ```python
   spread_mean = spread.ewm(span=window_size).mean()
   spread_std = spread.ewm(span=window_size).std()
   zscore = (spread - spread_mean) / spread_std
   ```

3. **Full History** (`'full'`):
   ```python
   spread_mean = spread.expanding().mean()
   spread_std = spread.expanding().std()
   zscore = (spread - spread_mean) / spread_std
   ```

### Signal Rules

The basic signal generation rules are:

```python
def generate_signals(self, zscore: pd.Series) -> pd.DataFrame:
    signals = pd.DataFrame(index=zscore.index)
    signals['zscore'] = zscore
    signals['signal'] = 0
    
    # Entry signals
    signals.loc[zscore < -self.entry_threshold, 'signal'] = 1  # Long spread signal
    signals.loc[zscore > self.entry_threshold, 'signal'] = -1  # Short spread signal
    
    # Exit signals
    signals.loc[
        (zscore > -self.exit_threshold) & 
        (zscore < self.exit_threshold), 'signal'] = 0  # Exit signal
    
    return signals
```

## Position Management

### Position Logic

The position logic tracks the current market position and applies trading rules:

```python
def apply_position_logic(self, signals: pd.DataFrame) -> pd.DataFrame:
    signals['position'] = 0
    current_position = 0
    entry_price = 0
    entry_bar = 0
    
    for i in range(len(signals)):
        # Current signal
        current_signal = signals['signal'].iloc[i]
        
        # Apply existing position logic
        if current_position != 0:
            # Check for stop loss
            if current_position == 1 and signals['zscore'].iloc[i] < -self.stop_loss_threshold:
                current_position = 0  # Stop out of long position
            elif current_position == -1 and signals['zscore'].iloc[i] > self.stop_loss_threshold:
                current_position = 0  # Stop out of short position
                
            # Check for max holding period
            if self.max_holding_period is not None:
                if i - entry_bar >= self.max_holding_period:
                    current_position = 0  # Exit due to max holding period
        
        # Apply new position based on signal
        if current_position == 0 and current_signal != 0:
            current_position = current_signal
            entry_price = signals['zscore'].iloc[i]
            entry_bar = i
        # Exit existing position
        elif current_position != 0 and current_signal == 0:
            current_position = 0
        
        signals['position'].iloc[i] = current_position
    
    return signals
```

### Risk Management

The implementation includes several risk management features:

1. **Stop Loss**:
   - Based on z-score threshold (default: 3.0)
   - Exits position when z-score moves against the position too far

2. **Maximum Holding Period**:
   - Optional time-based exit
   - Exits position after specified number of bars

3. **Position Sizing**:
   - Maximum position size as fraction of account
   - Implemented via the `max_position_size` parameter

4. **Trailing Stop** (Optional):
   - Adjusts stop loss as position moves in favor
   - Implemented when `use_trailing_stop=True`

## Transaction Costs

### Cost Modeling

The implementation models two types of transaction costs:

1. **Commission**:
   - Fixed cost per trade specified in `commission_per_trade`
   - Applied at entry and exit

2. **Slippage**:
   - Fixed cost per trade specified in `slippage_per_trade`
   - Applied at entry and exit

The costs are applied in the `calculate_returns()` method:

```python
# Transaction costs calculation (simplified)
if is_entry:
    # Apply entry transaction costs
    transaction_costs = (self.commission_per_trade * 2 + 
                        self.slippage_per_trade * 2) / position_value

if is_exit:
    # Apply exit transaction costs
    transaction_costs += (self.commission_per_trade * 2 + 
                         self.slippage_per_trade * 2) / position_value

# Apply transaction costs
pair_return -= transaction_costs
```

## Performance Metrics

### Key Metrics

The implementation calculates the following performance metrics:

1. **Return Metrics**:
   - Total return
   - Annualized return
   - Daily/monthly/yearly returns

2. **Risk Metrics**:
   - Volatility (standard deviation of returns)
   - Downside deviation
   - Maximum drawdown
   - Value at Risk (VaR)

3. **Risk-Adjusted Performance**:
   - Sharpe ratio
   - Sortino ratio
   - Calmar ratio
   - Information ratio

4. **Trade Statistics**:
   - Number of trades
   - Win rate
   - Average profit/loss
   - Profit factor
   - Average holding period

### Implementation

The metrics are calculated in the `_calculate_metrics()` method:

```python
def _calculate_metrics(self, signals: pd.DataFrame) -> Dict[str, float]:
    # Key metrics calculation
    returns = signals['returns'].dropna()
    cum_returns = (1 + returns).cumprod() - 1
    
    # Return metrics
    total_return = cum_returns.iloc[-1] if len(cum_returns) > 0 else 0
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    drawdown = 1 - (1 + cum_returns) / (1 + cum_returns.expanding().max())
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
    
    # Risk-adjusted metrics
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    sortino_ratio = annualized_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # Trade statistics
    position_changes = signals['position'].diff().fillna(0)
    entries = position_changes[position_changes != 0]
    num_trades = len(entries) // 2  # Each trade has entry and exit
    
    # Additional metrics...
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'num_trades': num_trades,
        # Additional metrics...
    }
```

## Configuration Parameters

### Strategy Parameters

The Z-Score Strategy accepts the following configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entry_threshold` | float | 2.0 | Z-score threshold for trade entry |
| `exit_threshold` | float | 0.5 | Z-score threshold for trade exit |
| `stop_loss_threshold` | float | 3.0 | Z-score threshold for stop loss |
| `window_size` | int | 20 | Window size for z-score calculation |
| `max_holding_period` | int | None | Maximum holding period in bars |
| `use_trailing_stop` | bool | False | Whether to use trailing stop |
| `use_time_filter` | bool | False | Whether to use time of day filter |
| `commission_per_trade` | float | 0.0 | Commission per trade in dollars |
| `slippage_per_trade` | float | 0.0 | Slippage per trade in dollars |
| `account_size` | float | 100000 | Initial account size in dollars |
| `max_position_size` | float | 0.25 | Maximum position size as fraction of account |
| `calculate_method` | str | 'rolling' | Method for z-score calculation ('rolling', 'ewm', 'full') |
| `use_log_prices` | bool | False | Whether to use log prices |

## Usage Examples

### Basic Usage

```python
from src.backtest.zscore_strategy_backtest import ZScoreStrategyBacktest
import pandas as pd

# Initialize the strategy with default parameters
strategy = ZScoreStrategyBacktest(
    entry_threshold=2.0,
    exit_threshold=0.5,
    stop_loss_threshold=3.0,
    window_size=20,
    commission_per_trade=0.01,
    slippage_per_trade=0.01
)

# Run backtest
results = strategy.backtest(
    price1=stock1_price,
    price2=stock2_price,
    hedge_ratio=None  # Calculate hedge ratio automatically
)

# Access results
equity_curve = results['equity_curve']
metrics = results['metrics']
trade_history = results['trade_history']

# Print key metrics
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Number of Trades: {metrics['num_trades']}")

# Plot results
strategy.plot_results()
```

### Using the Provided Runner Function

```python
from src.backtest.zscore_strategy_backtest import run_zscore_backtest

# Prepare price data dictionary
price_data = {
    'SPY': spy_price_df,
    'QQQ': qqq_price_df
}

# Run the backtest with custom parameters
results = run_zscore_backtest(
    price_data=price_data,
    ticker1='SPY',
    ticker2='QQQ',
    entry_threshold=2.5,
    exit_threshold=0.5,
    window_size=30,
    max_holding_period=10,
    use_log_prices=True
)

# Access results
print(f"Annualized Return: {results['metrics']['annualized_return']:.2%}")
```

### Parameter Optimization Example

```python
from src.backtest.zscore_strategy_backtest import ZScoreStrategyBacktest
import itertools

# Define parameter ranges
entry_thresholds = [1.5, 2.0, 2.5]
exit_thresholds = [0.3, 0.5, 0.8]
window_sizes = [10, 20, 30]

# Store results
optimization_results = []

# Iterate over parameter combinations
for entry, exit, window in itertools.product(entry_thresholds, exit_thresholds, window_sizes):
    strategy = ZScoreStrategyBacktest(
        entry_threshold=entry,
        exit_threshold=exit,
        window_size=window
    )
    
    results = strategy.backtest(price1, price2)
    
    optimization_results.append({
        'entry_threshold': entry,
        'exit_threshold': exit,
        'window_size': window,
        'sharpe_ratio': results['metrics']['sharpe_ratio'],
        'total_return': results['metrics']['total_return'],
        'max_drawdown': results['metrics']['max_drawdown']
    })

# Convert to DataFrame for analysis
import pandas as pd
results_df = pd.DataFrame(optimization_results)

# Find best parameter set based on Sharpe ratio
best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
print(f"Best parameters: Entry={best_params['entry_threshold']}, " +
      f"Exit={best_params['exit_threshold']}, Window={best_params['window_size']}")
```

## Extensions

### Implemented Extensions

The Z-Score Strategy implementation includes several extensions to the basic strategy:

1. **Multiple Z-Score Calculation Methods**:
   - Rolling window
   - Exponentially weighted
   - Full history

2. **Advanced Risk Management**:
   - Stop loss based on z-score
   - Maximum holding period
   - Position sizing

3. **Transaction Costs**:
   - Commission modeling
   - Slippage modeling

### Potential Future Extensions

Potential extensions for Phase 2 and beyond include:

1. **Advanced Signal Filters**:
   - Volume filters
   - Volatility filters
   - Correlation breakdown detection

2. **Dynamic Parameters**:
   - Adaptive entry/exit thresholds based on volatility
   - Regime-dependent parameters

3. **Multi-Pair Portfolio**:
   - Cross-correlation modeling
   - Risk allocation across multiple pairs

4. **Alternative Spread Models**:
   - Kalman filter for dynamic hedge ratio
   - VECM (Vector Error Correction Model)
   - Machine learning enhancements

## References

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.

2. Ehrman, D. S. (2006). "The Handbook of Pairs Trading: Strategies Using Equities, Options, and Futures." Wiley.

3. Do, B., & Faff, R. (2010). "Does Simple Pairs Trading Still Work?" Financial Analysts Journal, 66(4), 83-95.

4. Krauss, C. (2017). "Statistical Arbitrage Pairs Trading Strategies: Review and Outlook." Journal of Economic Surveys, 31(2), 513-545.

5. Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). "Pairs Trading." Quantitative Finance, 5(3), 271-276. 