# Intraday Statistical Arbitrage System API Documentation

## Overview

The Intraday Statistical Arbitrage System is a comprehensive framework for developing, backtesting, and deploying statistical arbitrage strategies for futures pairs trading. The system is designed with modularity and extensibility in mind, allowing for easy customization and enhancement.

### System Architecture

The system is organized into several interconnected modules:

```
src/
├── data/              # Data loading and preprocessing
├── cointegration/     # Pair selection and cointegration testing
├── spreads/           # Spread calculation and analysis
├── signals/           # Signal generation and management
├── risk/              # Risk management and position sizing
├── backtesting/       # Backtesting engine
├── reporting/         # Performance reporting and visualization
├── performance/       # Performance optimization tools
│   ├── parallel/      # Parallel processing utilities
│   ├── caching/       # Caching mechanisms
│   └── data_structures/ # Efficient data structures
├── connectors/        # Exchange connectors
│   └── ib/            # Interactive Brokers connector
└── paper_trading/     # Paper trading environment
```

### Core Components

1. **Data Processing**: Handles loading, cleaning, and preprocessing of time series data from various sources.
2. **Cointegration Analysis**: Identifies cointegrated pairs and estimates their statistical properties.
3. **Spread Analytics**: Calculates and normalizes spreads between cointegrated assets.
4. **Signal Generation**: Creates trading signals based on spread movements and other factors.
5. **Risk Management**: Manages position sizing and risk constraints.
6. **Backtesting**: Simulates strategy performance on historical data.
7. **Reporting**: Generates comprehensive performance reports.
8. **Performance Optimization**: Improves system speed and efficiency.
9. **Paper Trading**: Tests strategies with real-time data but simulated execution.
10. **Interactive Brokers Integration**: Connects to IB for market data and execution.

## Module Documentation

- [Data Module](data.md)
- [Cointegration Module](cointegration.md)
- [Spreads Module](spreads.md)
- [Signals Module](signals.md)
- [Risk Management Module](risk.md)
- [Backtesting Module](backtesting.md)
- [Reporting Module](reporting.md)
- [Performance Optimization](performance.md)
- [Paper Trading Module](paper_trading.md)
- [IB Connector](ib_connector.md)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quant-trader.git
cd quant-trader

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

A simple example of using the system:

```python
from src.data import DataProcessor
from src.cointegration import PairFinder
from src.spreads import SpreadAnalyzer
from src.signals import SignalGenerator
from src.risk import RiskManager
from src.backtesting import BacktestEngine
from src.reporting import BacktestReportGenerator

# Load and preprocess data
data_processor = DataProcessor()
data = data_processor.load_from_csv('path/to/data.csv')

# Find cointegrated pairs
pair_finder = PairFinder()
pairs = pair_finder.find_pairs(data)

# Analyze spread for the first pair
spread_analyzer = SpreadAnalyzer()
spread_data = spread_analyzer.calculate_spread(
    data[pairs[0][0]], 
    data[pairs[0][1]]
)

# Generate signals
signal_generator = SignalGenerator()
signals = signal_generator.generate_signals(spread_data)

# Manage risk and position sizing
risk_manager = RiskManager(initial_capital=100000)
positions = risk_manager.calculate_positions(signals, spread_data)

# Backtest the strategy
backtest_engine = BacktestEngine()
backtest_result = backtest_engine.run(data, signals, positions)

# Generate a report
report_generator = BacktestReportGenerator()
report_path = report_generator.generate_report(
    'My Strategy',
    'A statistical arbitrage strategy for futures pairs',
    backtest_result.equity_curve,
    backtest_result.trades
)
```

For more detailed examples and advanced usage, please refer to the specific module documentation. 