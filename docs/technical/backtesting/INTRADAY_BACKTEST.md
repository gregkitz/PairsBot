# Intraday Backtesting Framework

This document provides an overview of the intraday backtesting framework, which includes realistic time-of-day constraints, transaction cost modeling, and performance analysis tools.

## Overview

The intraday backtesting framework extends the base backtesting engine with specialized functionality for intraday trading:

1. **Realistic Intraday Constraints**: Implements time-of-day filters, maximum holding periods, and pre-market-close exits
2. **Transaction Cost Modeling**: Provides detailed models for commission and slippage calculation
3. **Performance Analysis**: Offers comprehensive performance metrics with time-of-day and regime-aware attribution

## Key Components

### IntradayBacktestEngine

The `IntradayBacktestEngine` class (in `src/backtest/intraday_backtest_engine.py`) extends the base `BacktestEngine` with intraday-specific functionality:

```python
from src.backtest.intraday_backtest_engine import IntradayBacktestEngine

# Create IntradayBacktestEngine
engine = IntradayBacktestEngine(
    signals=signals_df,
    prices=prices_dict,
    account_size=100000,
    intraday_params={
        "max_holding_period": 180,  # 3 hours max holding period
        "time_filters": {
            "avoid_first_15min": True,
            "avoid_lunch_hour": True,
            "high_liquidity_windows": [
                {"start": "09:45", "end": "11:30"},
                {"start": "13:30", "end": "15:45"}
            ]
        },
        "exit_buffer_minutes": 15  # Exit 15 minutes before market close
    },
    transaction_cost_model={
        "commission_model": "ibkr_pro",
        "commission_params": {
            "per_contract": 0.85,
            "per_share": 0.005,
            "minimum": 1.0
        },
        "slippage_model": "volume_based",
        "slippage_params": {
            "base_points": 1.0,
            "volume_factor": 0.5,
            "volatility_factor": 0.3
        }
    }
)

# Run backtest
results = engine.run_backtest()

# Calculate detailed metrics
detailed_metrics = engine.calculate_detailed_metrics()
```

### Intraday Constraints

The engine implements several intraday-specific constraints:

1. **Maximum Holding Period**: Forces exit of positions held beyond a specified time limit
2. **Time-of-Day Filters**:
   - Avoid trading during the first 15 minutes after market open
   - Avoid trading during lunch hour (12:00-13:00)
   - Only trade during specified high-liquidity windows
3. **Pre-Market Close Exit**: Force close all positions before market close

### Transaction Cost Models

The framework includes realistic transaction cost modeling:

1. **Commission Models**:
   - IBKR Pro: Standard Interactive Brokers Pro commission structure
   - Flat Fee: Fixed commission per trade
   - Percentage: Commission as percentage of trade value

2. **Slippage Models**:
   - Fixed: Fixed percentage of price
   - Variable: Random slippage within specified range
   - Volume-Based: Slippage that increases with relative trade size and market volatility

### Performance Visualization

The `intraday_performance_visualization.py` module provides comprehensive visualization tools:

```python
from src.backtest.intraday_performance_visualization import (
    create_intraday_performance_dashboard,
    save_performance_dashboard
)

# Create performance dashboard
dashboard_figures = create_intraday_performance_dashboard(
    backtest_results=results,
    regime_data=regime_data,  # Optional market regime data
    figsize=(12, 8)
)

# Save dashboard to files
save_performance_dashboard(
    figures=dashboard_figures,
    output_dir="output/backtest_results",
    prefix="intraday_test"
)
```

The dashboard includes:

1. **Equity Curve**: Overall performance with drawdown
2. **Intraday Patterns**: Returns by time of day
3. **Transaction Cost Analysis**: Commission and slippage impact
4. **Regime Performance**: Performance broken down by market regime
5. **Intraday Constraints Analysis**: Impact of time filters and forced exits

## Usage Examples

### Basic Usage

```python
# Create engine
engine = IntradayBacktestEngine(
    signals=signals_df,
    prices=prices_dict,
    account_size=100000
)

# Run backtest with default intraday constraints
results = engine.run_backtest()

# Print metrics
detailed_metrics = engine.calculate_detailed_metrics()
print(f"Total Return: {detailed_metrics.get('total_return', 0):.2f}%")
print(f"Transaction Costs: ${detailed_metrics.get('transaction_costs', {}).get('total', 0):.2f}")
```

### Custom Constraints

```python
# Define custom intraday parameters
intraday_params = {
    "max_holding_period": 90,  # 90 minutes max
    "time_filters": {
        "avoid_first_15min": True,
        "avoid_lunch_hour": False,  # Allow lunch hour trading
        "high_liquidity_windows": [
            {"start": "09:45", "end": "16:00"}  # Trade all day
        ]
    },
    "exit_buffer_minutes": 5  # Exit 5 minutes before market close
}

# Run backtest with custom parameters
results = engine.run_backtest(intraday_params=intraday_params)
```

### Full Example

A complete example is available in `examples/run_enhanced_intraday_backtest.py`, which demonstrates:

1. Loading or generating sample data
2. Configuring the intraday backtest engine
3. Running the backtest with realistic constraints
4. Visualizing the results with performance analysis
5. Incorporating market regime information

## Best Practices

1. **Data Requirements**:
   - Use minute-level data for accurate intraday simulation
   - Include volume data for realistic slippage modeling
   - Ensure timestamps use DatetimeIndex for time-of-day filtering

2. **Parameter Calibration**:
   - Calibrate transaction cost models to match real-world execution
   - Adjust time filters based on known market behavior patterns
   - Set realistic maximum holding periods based on strategy characteristics

3. **Performance Analysis**:
   - Analyze performance by time of day to optimize entry/exit timing
   - Compare transaction costs as percentage of gross profit
   - Examine the impact of constraints on overall performance

## Extending the Framework

The intraday backtesting framework can be extended in several ways:

1. **Custom Constraint Rules**: Implement custom time-of-day filters in the `_apply_intraday_constraints` method
2. **Advanced Transaction Cost Models**: Add more sophisticated models in the `_calculate_transaction_costs` method
3. **Additional Performance Metrics**: Extend the `calculate_detailed_metrics` method with custom metrics

## Troubleshooting

1. **No Trades Executed**: Check that your signals are aligned with your price data and occur during allowed trading windows
2. **High Transaction Costs**: Review and adjust your transaction cost model parameters
3. **Visualization Issues**: Ensure that your data has a proper DatetimeIndex for time-based plots 