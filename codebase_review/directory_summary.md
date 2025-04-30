# Quant-Trader Directory Structure Summary

This document provides a high-level overview of the purpose and contents of each directory in the quant-trader codebase.

## Core Source Code (`src/`)

### Asset Classes (`src/asset_classes/`)
Contains the implementation of different asset classes (equities, futures, fixed income, cryptocurrencies) that can be traded by the system. Includes a factory pattern for instantiating appropriate asset types and base classes that define common functionality.

Subdirectories:
- `cryptocurrencies/`: Cryptocurrency-specific asset implementations
- `equities/`: Equity-specific asset implementations 
- `fixed_income/`: Fixed income asset implementations
- `futures/`: Futures asset implementations

### Backtest (`src/backtest/`)
Houses the backtesting framework for simulating and evaluating trading strategies without risking real capital. Includes engines for both daily and intraday backtesting with realistic constraints and performance visualization tools.

### Cointegration (`src/cointegration/`)
Contains tools for testing cointegration relationships between asset pairs, which is a fundamental requirement for statistical arbitrage. Includes implementations of statistical tests and pair selection algorithms.

### Configuration (`src/config/`)
Manages system-wide configuration settings, parameter management, and configuration file handling. Provides interfaces for loading, saving, and accessing configuration values throughout the application.

### Connectors (`src/connectors/`)
Implements connections to external systems like brokers and data providers. Currently focused on Interactive Brokers (IB) integration.

Subdirectories:
- `ib/`: Interactive Brokers connector implementation

### Data Processor (`src/data_processor/`)
Responsible for loading, preprocessing, and managing market data. Includes specialized processors for different data types and timeframes.

### Live Trading (`src/live_trading/`)
Contains components required for executing trades in a live environment. Includes position tracking, monitoring, and real-time management of trades.

### ML Enhancements (`src/ml_enhancements/`)
Machine learning enhancements to the base statistical arbitrage strategy. Includes signal enhancement, regime detection, and predictive models.

Subdirectories:
- `feature_engineering/`: Feature creation and transformation for ML models
- `regime_detection/`: Market regime classification and detection
- `spread_prediction/`: Predictive models for spread behavior

### Optimization (`src/optimization/`)
Tools for optimizing strategy parameters to improve performance. Includes various optimization techniques.

Subdirectories:
- `genetic_algorithm/`: Evolutionary optimization approach
- `grid_search/`: Exhaustive parameter search implementation
- `walk_forward/`: Walk-forward optimization to reduce overfitting

### Pair Trading (`src/pair_trading/`)
Core pair trading strategy implementation. Contains logic for pair selection, spread calculation, and trade management specific to pairs trading.

### Paper Trading (`src/paper_trading/`)
Simulated trading environment that mimics real trading without risking capital. Used for strategy validation before live deployment.

### Performance (`src/performance/`)
Tools for measuring and analyzing strategy performance. Includes metrics calculation and performance optimization techniques.

Subdirectories:
- `caching/`: Performance optimization through caching
- `parallel/`: Parallel processing implementations

### Portfolio (`src/portfolio/`)
Portfolio management components for handling multiple pairs and strategies simultaneously. Includes position sizing, risk allocation, and portfolio-level metrics.

### Reporting (`src/reporting/`)
Reporting and visualization tools for strategy performance and metrics. Generates reports on backtest and live trading results.

### Risk Management (`src/risk_management/`)
Risk management components including position sizing, stop-loss implementations, and risk metrics calculation.

### Signal Generation (`src/signal_generation/`)
Generates trading signals based on spread behavior and other indicators. Core logic for determining entries and exits.

### Signals (`src/signals/`)
Alternative signal generation implementations and signal processing utilities.

### Spread Analytics (`src/spread_analytics/`)
Tools for analyzing the behavior of spreads between pairs. Includes statistical analysis and characteristic measurement.

### Strategies (`src/strategies/`)
Strategy implementations beyond basic pairs trading. May include variations and extensions of the core approach.

### Strategy Variants (`src/strategy_variants/`)
Variations on the base strategy with different signal generation or execution approaches.

Subdirectories:
- `ml_signals/`: Machine learning enhanced signal generation
- `time_series/`: Time series analysis based strategies

### Utils (`src/utils/`)
General utility functions and helpers used throughout the codebase.

### Visualization (`src/visualization/`)
Data visualization tools for analyzing pairs, spreads, and performance metrics.

### Web Interface (`src/web_interface/`)
Web-based user interface for monitoring and controlling the trading system.

Subdirectories:
- `blueprints/`: Flask blueprints for different sections of the web interface
- `static/`: Static assets (CSS, JavaScript)
- `templates/`: HTML templates

## Tests (`tests/`)

Contains all test code organized by test type (unit, integration) and mirroring the structure of the source code.

Subdirectories:
- `fixtures/`: Test fixtures and shared test data
- `integration/`: Integration tests that test multiple components together
- `mocks/`: Mock objects for testing
- `unit/`: Unit tests for individual components

## Documentation (`docs/`)

Contains project documentation, implementation guides, and architecture plans.

## Examples (`examples/`)

Contains example scripts demonstrating how to use various components of the system.

## Scripts (`scripts/`)

Utility scripts for development, deployment, and system maintenance.

## Setup (`setup/`)

Installation and setup scripts, configuration templates, and environment setup tools.

## Codebase Review (`codebase_review/`)

Tools and documentation for reviewing and cleaning up the codebase, including analysis tools and planning documents. 