# Quant-Trader: Intraday Statistical Arbitrage System

This is a sophisticated statistical arbitrage system for futures pairs trading, focusing on mean-reverting relationships with optimal entry/exit timing and rigorous risk management.

## System Overview

The system identifies and exploits temporary mispricings between cointegrated instruments, using various statistical methods and ML enhancements to improve trading signals. It includes functionality for:

- Pair selection and cointegration testing
- Spread calculation with dynamic hedge ratios
- Signal generation with ML-based enhancements
- Risk management with adaptive position sizing
- Paper trading with performance monitoring
- Comprehensive backtesting with transaction costs

## Documentation

For detailed system documentation, refer to these resources:

### Architecture & Design
- [System Architecture Diagram](docs/architecture_dir/system_architecture_diagram.md): Visual representation of system components
- [PAIRS Design Document](docs/architecture_dir/PAIRS_DESIGN.md): Overall system design and architecture
- [Data Flow Architecture](docs/architecture_dir/data_flow.md): Data flow between system components
- [Docker Architecture](docs/architecture_dir/docker_architecture.md): Containerized distributed processing

### Implementation Guides & Plans
- [Next Steps](docs/plans/next_steps.md): Current priorities and immediate focus items
- [Implementation Status](docs/context/implementation_status.md): Current state of implementation
- [Implementation Notes](docs/context/implementation_notes.md): Notes on implementation details

### Technical Documentation
- [Cointegration Framework](docs/technical/cointegration_framework.md): Comprehensive cointegration framework design
- [Statistical Methods](docs/technical/statistical_methods.md): Statistical methods for cointegration testing
- [Johansen Test Implementation](docs/technical/johansen_implementation.md): Detailed Johansen test implementation
- [Engle-Granger Test Implementation](docs/technical/engle_granger_implementation.md): Detailed Engle-Granger test implementation
- [Statistical Validation Methods](docs/technical/statistical_validation_methods.md): Methods for validating cointegration relationships
- [Z-Score Strategy Implementation](docs/technical/zscore_strategy_implementation.md): Comprehensive Z-Score strategy documentation

#### Cointegration Implementation
- [Cointegration Framework](docs/technical/cointegration/cointegration_framework.md): Detailed component interactions
- [Statistical Methods](docs/technical/cointegration/statistical_methods.md): Mathematical foundations
- [Statistical Validation](docs/technical/cointegration/statistical_validation_methods.md): Ensuring robustness

#### Strategy Implementation
- [Z-Score Strategy](docs/technical/strategies/zscore_strategy_implementation.md): Core strategy implementation
- [Strategy Variants](docs/technical/strategies/strategy_variants.md): Different strategy variations

#### Backtesting Implementation
- [Backtest Implementation Guide](docs/technical/backtesting/backtest_implementation_guide.md): Detailed implementation guide
- [Intraday Backtest](docs/technical/backtesting/INTRADAY_BACKTEST.md): Intraday backtesting specifics

### User Guides
- [Paper Trading Guide](docs/paper_trading_guide.md): Guide for paper trading setup and usage
- [Configuration Guide](docs/user_manual/configuration_guide.md): Configuration options and parameters
- [Intraday ML System User Guide](docs/user_manual/intraday_ml_system_user_guide.md): End-user guide for ML system
- [Troubleshooting Guide](docs/user_manual/troubleshooting_guide.md): Solutions for common issues
- [IB Connection Troubleshooting](docs/user_manual/ib_connection_troubleshooting.md): Interactive Brokers connectivity help

### Documentation Index
For a complete overview of all documentation, refer to the [Documentation Guide](docs/documentation_guide.md).

## Main Commands

The system provides a unified command-line interface through `main.py` with the following commands:

### Analyze Pairs

Find and analyze cointegrated pairs from your futures data:

```bash
python main.py analyze-pairs --tickers GC SI ZB ZN --min-correlation 0.7 --timeframe 1hour
```

### Run Backtests

Test trading strategies on historical data:

```bash
python main.py backtest --pairs GC SI ZB ZN --start-date 2023-01-01 --end-date 2023-12-31
```

### Run Intraday Backtests

Test intraday-specific strategies with enhanced features:

```bash
python main.py intraday-backtest --pairs GC SI --start-date 2023-01-01 --end-date 2023-12-31 --timeframe 5min --use-ml
```

### Train ML Models

Train machine learning models to enhance trading signals:

```bash
python main.py train-models --pair GC_SI --start-date 2023-01-01 --end-date 2023-12-31 --timeframe 5min
```

### Train Regime Detection

Train market regime classifier for adaptive parameter selection:

```bash
python main.py train-regime-classifier --tickers GC SI ZB ZN --timeframe 1day --n-regimes 3
```

### Optimize Parameters

Optimize strategy parameters for different market regimes:

```bash
python main.py optimize-parameters --pairs-file output/pairs_analysis.json --start-date 2023-01-01 --end-date 2023-12-31 --n-regimes 3
```

### Run Paper Trading

Test your strategy in a simulated environment:

```bash
python main.py paper-trade --config config/paper_trading.json --capital 100000 --test-mode
```

### Process Data

Prepare data for analysis and backtesting:

```bash
python main.py process-data --symbols GC SI ZB ZN --start-date 2020-01-01 --end-date 2023-12-31 --timeframe 5min
```

### Start API Server

Launch the web API for monitoring and control:

```bash
python main.py api --host 0.0.0.0 --port 8000
```

### Start Worker

Start a Celery worker for background tasks:

```bash
python main.py worker --concurrency 4 --queue default
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Process your historical data: `python main.py process-data --symbols GC SI ZB ZN --start-date 2020-01-01 --end-date 2023-12-31`
4. Analyze pairs: `python main.py analyze-pairs --tickers GC SI ZB ZN`
5. Run a backtest: `python main.py intraday-backtest --pairs GC SI --start-date 2023-01-01 --end-date 2023-12-31`
6. Train ML models: `python main.py train-models --pair GC_SI --start-date 2023-01-01 --end-date 2023-12-31`
7. Start paper trading: `python main.py paper-trade --config config/paper_trading.json --test-mode`

## Example Workflow

A typical workflow might look like:

1. Process your futures data
2. Find cointegrated pairs with the pair analyzer
3. Run backtests to validate the strategy
4. Train ML models to enhance signal generation
5. Optimize parameters for different market regimes
6. Run paper trading with the optimized strategy
7. Monitor performance and refine the strategy

## Data Structure

The system expects futures data in the following structure:

- `/data/processed/`: Processed futures data
- `/data/models/`: Trained ML models
- `/data/results/`: Backtest results and analysis
- `/config/`: System and strategy configuration files

## System Architecture

The system consists of several components:

1. **Asset Classes**: Abstractions for futures, equities, and other asset types
2. **Pair Trading**: Pair selection, spread analytics, and cointegration testing
3. **Signal Generation**: Signal processing, z-score calculation, and filtering
4. **ML Enhancements**: Feature engineering, model training, intraday signal enhancement
5. **Paper/Live Trading**: Order execution, position management, performance tracking
6. **Risk Management**: Position sizing, stop loss management, exposure control
7. **Backtesting**: Backtesting engine, strategy optimization, performance metrics
8. **Infrastructure**: Docker containers, Celery tasks, monitoring dashboard

For a visual representation of the system architecture, see [System Architecture Diagram](docs/architecture_dir/system_architecture_diagram.md).

## Docker-Based Task Processing

The system uses Docker containers for distributed task processing:

```bash
# Start the Docker containers
./scripts/start-containers.ps1

# Stop the Docker containers
./scripts/stop-containers.ps1

# Submit a task to the system
./scripts/submit-task.ps1 -TaskType train-models -Pair GC_SI -Timeframe 1hour
```

For details on the Docker-based architecture, see [Docker Architecture](docs/architecture_dir/docker_architecture.md).

## Technical Debt Resolution

We are actively working on improving code quality by addressing technical debt in these areas:

1. **Large Files Refactoring**:
   - Breaking down large files into smaller, focused modules
   - Implementing design patterns to improve code organization

2. **Complex Functions Simplification**:
   - Extracting helper methods from complex functions
   - Applying the single responsibility principle

3. **Duplicate Code Elimination**:
   - Using template method pattern and inheritance
   - Creating base classes and mixins for common functionality

For details on our technical debt resolution plan, see [Technical Debt Analysis](docs/context/technical_debt_analysis.md).

## Contributing

Guidelines for contributing to the project:

1. Use consistent naming conventions with existing code
2. Add comprehensive tests for new features
3. Document your changes in the appropriate documentation files
4. Follow the code structure and patterns established in the project
5. Run tests before submitting changes

## License

This project is proprietary and not licensed for public use. All rights reserved.
