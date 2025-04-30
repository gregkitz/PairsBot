# Reporting Module

The Reporting module provides tools for generating comprehensive HTML reports from backtest results, with interactive visualizations and performance metrics.

## Components

### BacktestReportGenerator

The `BacktestReportGenerator` class is the main tool for creating HTML reports from backtest results.

```python
from src.reporting import BacktestReportGenerator
```

#### Initialization

```python
report_generator = BacktestReportGenerator(
    template_dir=None,   # Directory containing HTML templates (default: module's templates directory)
    output_dir=None      # Directory to save the generated reports (default: './reports')
)
```

#### Generating Reports

```python
report_path = report_generator.generate_report(
    title='My Strategy',                            # Report title
    strategy_description='Statistical arbitrage...',  # Strategy description
    equity_curve=backtest_result.equity_curve,      # Equity curve as pd.Series
    trades=backtest_result.trades,                  # Trades as pd.DataFrame (optional)
    benchmark=benchmark_data,                       # Benchmark data as pd.Series (optional)
    risk_free_rate=0.02,                            # Risk-free rate (optional)
    output_filename='my_report.html',               # Output filename (optional)
    additional_metrics={}                           # Additional metrics to include (optional)
)
```

### Performance Metrics Calculation

The module includes a comprehensive performance metrics calculator:

```python
from src.reporting import calculate_performance_metrics

metrics = calculate_performance_metrics(
    equity_curve=backtest_result.equity_curve,  # Required
    trades=backtest_result.trades,              # Optional
    benchmark=benchmark_data,                   # Optional
    risk_free_rate=0.02,                        # Optional
    trading_days_per_year=252                   # Optional
)
```

## Metrics and Visualizations

### Generated Performance Metrics

The report includes a wide range of performance metrics:

#### Return Metrics
- Total Return
- Annualized Return
- Cumulative Returns Over Time
- Monthly/Yearly Returns
- Best/Worst Month/Day

#### Risk Metrics
- Maximum Drawdown
- Drawdown Duration
- Value at Risk (VaR)
- Conditional VaR (Expected Shortfall)
- Daily Volatility
- Annualized Volatility

#### Risk-Adjusted Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Information Ratio (if benchmark provided)
- Beta (if benchmark provided)
- Alpha (if benchmark provided)

#### Trade Statistics (if trades provided)
- Number of Trades
- Win Rate
- Average Win/Loss
- Profit Factor
- Average Holding Period
- Winning/Losing Streaks

### Interactive Visualizations

The generated HTML report includes several interactive visualizations created with Plotly:

- **Equity Curve**: Shows the growth of portfolio value over time
- **Drawdown Chart**: Visualizes underwater periods
- **Monthly Returns Heatmap**: Shows returns by month and year
- **Return Distribution**: Histogram of returns
- **Trade Outcomes**: Pie chart of winning/losing trades
- **Trade P&L Distribution**: Histogram of trade P&L
- **Cumulative P&L by Side**: Long vs. short performance
- **Benchmark Comparison**: Strategy vs benchmark performance (if benchmark provided)

## Report Format

The generated report is a standalone HTML file with the following sections:

1. **Summary**: Overview of key metrics
2. **Performance Metrics**: Detailed performance statistics
3. **Return Analysis**: Monthly/yearly returns with heatmap
4. **Drawdown Analysis**: Drawdown periods and statistics
5. **Trade Analysis**: Trade statistics and visualizations (if trades provided)
6. **Risk Analysis**: Risk metrics and visualizations
7. **Benchmark Comparison**: Comparison to benchmark (if benchmark provided)

The report uses Bootstrap for styling and is fully responsive, making it suitable for viewing on any device. All charts are interactive with zoom, pan, and tooltips.

## Data Requirements

### Equity Curve

The `equity_curve` parameter should be a pandas Series with a DatetimeIndex and portfolio values.

```python
equity_curve = pd.Series(
    [10000, 10050, 10100, 10025, 10200],
    index=pd.date_range(start='2023-01-01', periods=5, freq='D')
)
```

### Trades DataFrame

If provided, the `trades` DataFrame should contain individual trade information with the following columns:

- `entry_time`: Entry timestamp
- `exit_time`: Exit timestamp
- `pnl`: Profit/loss amount
- `side` (optional): 'long' or 'short'
- `symbol` (optional): Trading instrument
- `entry_price` (optional): Entry price
- `exit_price` (optional): Exit price

```python
trades = pd.DataFrame({
    'entry_time': [pd.Timestamp('2023-01-01 10:00:00'), pd.Timestamp('2023-01-02 11:00:00')],
    'exit_time': [pd.Timestamp('2023-01-01 15:30:00'), pd.Timestamp('2023-01-02 15:45:00')],
    'pnl': [50.0, -25.0],
    'side': ['long', 'short'],
    'symbol': ['AAPL', 'MSFT'],
    'entry_price': [150.0, 250.0],
    'exit_price': [155.0, 252.5]
})
```

### Benchmark Data

If provided, the `benchmark` parameter should be a pandas Series with the same DatetimeIndex as the equity curve.

```python
benchmark = pd.Series(
    [10000, 10020, 10010, 10030, 10060],
    index=pd.date_range(start='2023-01-01', periods=5, freq='D')
)
```

## Examples

### Basic Example

```python
from src.reporting import BacktestReportGenerator
from datetime import datetime
import pandas as pd
import numpy as np

# Create sample equity curve (daily values)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
np.random.seed(42)
daily_returns = np.random.normal(0.0005, 0.01, len(dates))
equity = 10000 * (1 + daily_returns).cumprod()
equity_curve = pd.Series(equity, index=dates)

# Create sample trades
trades = pd.DataFrame({
    'entry_time': [
        datetime(2023, 1, 5, 10, 30),
        datetime(2023, 1, 10, 11, 15),
        datetime(2023, 2, 3, 9, 45)
    ],
    'exit_time': [
        datetime(2023, 1, 5, 15, 45),
        datetime(2023, 1, 11, 14, 30),
        datetime(2023, 2, 4, 10, 15)
    ],
    'pnl': [120.0, -50.0, 200.0],
    'side': ['long', 'short', 'long'],
    'symbol': ['AAPL', 'MSFT', 'GOOGL'],
    'entry_price': [150.0, 220.0, 180.0],
    'exit_price': [156.0, 222.5, 190.0]
})

# Generate the report
report_generator = BacktestReportGenerator()
report_path = report_generator.generate_report(
    title='My Trading Strategy',
    strategy_description='A statistical arbitrage strategy for tech stocks',
    equity_curve=equity_curve,
    trades=trades
)

print(f"Report generated at: {report_path}")
```

### With Benchmark Comparison

```python
from src.reporting import BacktestReportGenerator
import pandas as pd
import numpy as np

# Create sample equity curve
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
np.random.seed(42)
strategy_returns = np.random.normal(0.0005, 0.01, len(dates))
strategy_equity = 10000 * (1 + strategy_returns).cumprod()
equity_curve = pd.Series(strategy_equity, index=dates)

# Create sample benchmark (S&P 500)
benchmark_returns = np.random.normal(0.0004, 0.012, len(dates))
benchmark_equity = 10000 * (1 + benchmark_returns).cumprod()
benchmark = pd.Series(benchmark_equity, index=dates)

# Generate the report with benchmark comparison
report_generator = BacktestReportGenerator()
report_path = report_generator.generate_report(
    title='Strategy vs S&P 500',
    strategy_description='Comparing my strategy to the S&P 500 index',
    equity_curve=equity_curve,
    benchmark=benchmark,
    risk_free_rate=0.02  # 2% risk-free rate
)

print(f"Report with benchmark comparison generated at: {report_path}")
```

### Calculating Performance Metrics Separately

```python
from src.reporting import calculate_performance_metrics
import pandas as pd
import numpy as np

# Create sample equity curve
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
np.random.seed(42)
daily_returns = np.random.normal(0.0005, 0.01, len(dates))
equity = 10000 * (1 + daily_returns).cumprod()
equity_curve = pd.Series(equity, index=dates)

# Calculate performance metrics
metrics = calculate_performance_metrics(
    equity_curve=equity_curve,
    risk_free_rate=0.02,
    trading_days_per_year=252
)

# Print key metrics
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Annualized Return: {metrics['annual_return']:.2%}")
print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
```

## Integration with Backtesting Engine

The reporting module can be easily integrated with the backtesting engine:

```python
from src.backtesting import BacktestEngine
from src.reporting import BacktestReportGenerator

# Run backtest
backtest_engine = BacktestEngine()
backtest_result = backtest_engine.run(data, signals, positions)

# Generate report from backtest results
report_generator = BacktestReportGenerator()
report_path = report_generator.generate_report(
    title='Backtest Results',
    strategy_description='Statistical arbitrage strategy for futures pairs',
    equity_curve=backtest_result.equity_curve,
    trades=backtest_result.trades
)
```

## Customization

The report layout and styling can be customized by providing your own HTML templates. Create a directory with modified templates and pass it to the `BacktestReportGenerator`:

```python
report_generator = BacktestReportGenerator(
    template_dir='my_templates',
    output_dir='my_reports'
)
```

The main template file should be named `report_template.html`. You can copy the default template from the module's `templates` directory and modify it as needed. 