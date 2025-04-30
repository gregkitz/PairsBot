# Testing Framework for Intraday Statistical Arbitrage System

This directory contains the testing framework for the Intraday Statistical Arbitrage System. It includes unit tests for individual components and integration tests for interactions between components.

## Directory Structure

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── asset_classes/     # Tests for asset class implementations
│   ├── cointegration/     # Tests for cointegration analysis
│   ├── data_processor/    # Tests for data processing
│   ├── risk_management/   # Tests for risk management
│   ├── signal_generation/ # Tests for signal generation
│   ├── spread_analytics/  # Tests for spread analytics
│   ├── strategy_variants/ # Tests for strategy variants
│   ├── connectors/        # Tests for external connectors
│   └── paper_trading/     # Tests for paper trading
├── integration/           # Integration tests for component interactions
├── mocks/                 # Mock objects for testing
├── fixtures/              # Test data and configurations
└── README.md              # This file
```

## Running Tests

### Prerequisites

Ensure you have installed the required testing dependencies:

```bash
pip install -r requirements-test.txt
```

### Running All Tests

To run all tests with coverage reporting:

```bash
./run_tests.sh
```

Or, if you prefer to run pytest directly:

```bash
python -m pytest tests/ --cov=src
```

### Running Specific Test Types

To run only unit tests:

```bash
python -m pytest tests/unit/
```

To run only integration tests:

```bash
python -m pytest tests/integration/
```

To run tests for a specific component:

```bash
python -m pytest tests/unit/data_processor/
```

### Test Markers

Tests are marked with the following markers to organize them:

- `unit`: Unit tests for isolated components
- `integration`: Integration tests for component interactions
- `slow`: Tests that take a long time to run
- `external`: Tests that require external connections

To run tests with a specific marker:

```bash
python -m pytest -m unit
```

To exclude tests with a specific marker:

```bash
python -m pytest -m "not slow"
```

## Mock Components

The `mocks/` directory contains mock implementations of external dependencies, such as:

- `mock_ib_connector.py`: Mock Interactive Brokers API for testing
- `mock_market_data.py`: Mock market data provider
- `mock_exchange.py`: Mock exchange for testing order execution

## Test Fixtures

The `fixtures/` directory contains test data and configurations:

- `cointegrated_pairs.json`: Sample cointegrated pairs data
- `historical_data_samples.csv`: Sample historical price data
- `test_config.json`: Test configuration for system components

## Writing New Tests

When adding new components or features, please follow these guidelines:

1. Create unit tests for individual components in the appropriate directory
2. Add integration tests for interactions with other components
3. Use existing fixtures or add new ones as needed
4. Follow the naming convention: `test_*.py` for files and `test_*` for functions
5. Add appropriate markers to organize tests

## Continuous Integration

The tests are automatically run on every commit through GitHub Actions. The configuration is in `.github/workflows/`.

## Code Coverage

We aim for at least 80% code coverage for critical components. Coverage reports are generated after running tests with `--cov` flag. 