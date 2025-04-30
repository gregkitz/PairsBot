# Testing Plan for Intraday Statistical Arbitrage System

This document outlines the unit and integration testing strategy for the intraday statistical arbitrage system.

## Testing Structure

```
tests/
├── unit/
│   ├── asset_classes/
│   │   ├── test_base.py
│   │   ├── test_factory.py
│   │   └── futures/
│   │       └── test_futures_asset.py
│   ├── cointegration/
│   │   ├── test_pair_finder.py
│   │   └── test_cointegration_tests.py
│   ├── data_processor/
│   │   └── test_data_processor.py
│   ├── signal_generation/
│   │   └── test_signal_generator.py
│   ├── risk_management/
│   │   └── test_risk_manager.py
│   ├── spread_analytics/
│   │   └── test_spread_analyzer.py
│   ├── strategy_variants/
│   │   ├── test_time_series_strategy.py
│   │   └── test_ml_signal_strategy.py
│   ├── connectors/
│   │   └── ib/
│   │       └── test_ib_connector.py
│   └── paper_trading/
│       └── test_paper_trader.py
├── integration/
│   ├── test_data_flow.py
│   ├── test_strategy_execution.py
│   ├── test_paper_trading_system.py
│   └── test_end_to_end.py
├── mocks/
│   ├── mock_market_data.py
│   ├── mock_ib_connector.py
│   └── mock_exchange.py
└── fixtures/
    ├── cointegrated_pairs.json
    ├── historical_data_samples.csv
    └── test_config.json
```

## Unit Testing Approach

Unit tests will validate the functionality of individual components in isolation. We'll use mocks and stubs to simulate interactions with external systems.

### Key Components to Test

1. **Asset Classes**
   - Test asset creation, data retrieval, metadata functionality
   - Confirm futures-specific calculations like continuous contracts work correctly
   - Verify asset factory correctly instantiates appropriate asset classes

2. **Cointegration**
   - Test pair finding algorithms with known cointegrated series
   - Validate statistical tests (Engle-Granger, Johansen) against reference values
   - Test half-life calculation accuracy

3. **Data Processing**
   - Test data loading, cleaning, and merging functionality
   - Verify handling of different data frequencies and formats
   - Test missing data handling

4. **Signal Generation**
   - Test z-score calculation
   - Verify entry/exit signal generation logic
   - Test confirmation filters

5. **Risk Management**
   - Test position sizing algorithms
   - Verify stop-loss and take-profit logic
   - Test exposure limits and drawdown protections

6. **Spread Analytics**
   - Test spread calculation with various hedge ratios
   - Verify Kalman filter implementation
   - Test detection of regime changes

7. **Strategy Variants**
   - Test time series forecasting models
   - Verify ML signal enhancement logic
   - Test feature engineering pipeline

8. **Connectors**
   - Test IB connector with simulated responses
   - Verify proper handling of connection issues
   - Test order submission and market data methods

9. **Paper Trading**
   - Test simulated order execution logic
   - Verify P&L calculation accuracy
   - Test position management

## Integration Testing Approach

Integration tests will validate how components work together through the system.

### Key Integration Tests

1. **Data Flow Integration**
   - Test data flow from data processor through pair finder, spread analyzer, signal generator
   - Verify consistent handling of DataFrames and Series across components
   - Test time synchronization between different data sources

2. **Strategy Execution**
   - Test complete signal generation and execution pipeline
   - Verify strategy variants produce consistent output formats
   - Test handling of different market conditions (trending, ranging, volatile)

3. **Paper Trading System**
   - Test integration of strategy with paper trading system
   - Verify realistic simulation of orders, fills, and P&L
   - Test simultaneous handling of multiple pairs

4. **End-to-End Testing**
   - Test complete workflow from data acquisition to position management
   - Verify performance metrics calculation
   - Test recovery from errors and edge cases

## Testing Tools and Frameworks

1. **pytest**: Primary testing framework
2. **pytest-mock**: For creating and managing mocks
3. **pytest-cov**: For measuring test coverage
4. **hypothesis**: For property-based testing of statistical functions
5. **pandas-datareader**: For obtaining real market data for tests

## Mock Components

For testing in isolation, we'll create mock versions of:

1. **Mock Market Data Provider**: Simulates market data feeds
2. **Mock IB Connector**: Simulates Interactive Brokers API
3. **Mock Exchange**: Simulates exchange behavior for testing order execution

## Test Data Fixtures

We'll create reusable test data fixtures including:

1. **Cointegrated Pair Series**: Known cointegrated series with verified properties
2. **Historical Data Samples**: Representative market data for different scenarios
3. **Test Configurations**: Standard test configurations for various components

## Implementation Plan

### Phase 1: Setup Testing Framework

1. Create `tests` directory structure
2. Set up pytest configuration
3. Create basic mocks and fixtures
4. Implement basic utility tests

### Phase 2: Unit Tests for Core Components

1. Implement tests for asset classes (including futures_asset.py)
2. Create tests for cointegration and pair finding
3. Implement tests for spread analytics and signal generation
4. Develop tests for risk management

### Phase 3: Paper Trading and Strategy Tests

1. Implement tests for paper trading system
2. Create tests for strategy variants
3. Test IB connector functionality
4. Implement tests for web interface components

### Phase 4: Integration Tests

1. Implement data flow integration tests
2. Create strategy execution tests
3. Develop paper trading system integration tests
4. Implement end-to-end tests
5. Test market regime detection and parameter adaptation
   - Verify market regime classification accuracy
   - Test adaptive parameter updates based on regime
   - Test scheduler for regular regime updates

### Phase 5: Continuous Integration Setup

1. Configure GitHub Actions or similar CI system
2. Set up automated test runs on commits
3. Implement test coverage reporting
4. Create test result visualization

## Mock Implementation Examples

### Mock IB Connector

```python
class MockIBConnector:
    def __init__(self, host='localhost', port=7497, client_id=1, read_only=True):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.read_only = read_only
        self.connected = False
        self.market_data = {}
        self.contracts = {}
        
    def connect(self):
        self.connected = True
        return True
        
    def disconnect(self):
        self.connected = False
        return True
        
    def is_connected(self):
        return self.connected
        
    def get_market_data(self, symbol, subscribe=False):
        # Return realistic mock data
        return {
            'last_price': 100.0,
            'bid': 99.5,
            'ask': 100.5,
            'volume': 1000,
            'time': datetime.now()
        }
```

### Mock Data Provider

```python
class MockDataProvider:
    def __init__(self):
        self.data = {}
        
    def load_test_data(self, symbol, test_data_path):
        self.data[symbol] = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
        
    def get_data(self, symbol, start_date, end_date):
        if symbol not in self.data:
            # Generate synthetic data if no test data loaded
            return self._generate_synthetic_data(start_date, end_date)
        
        # Return slice of loaded test data
        return self.data[symbol].loc[start_date:end_date]
        
    def _generate_synthetic_data(self, start_date, end_date):
        # Create synthetic price series for testing
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({
            'Open': np.random.normal(100, 1, len(dates)),
            'High': np.random.normal(102, 1, len(dates)),
            'Low': np.random.normal(98, 1, len(dates)),
            'Close': np.random.normal(100, 1, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
```

## Sample Test Cases

### Unit Test: Futures Asset

```python
def test_futures_asset_creation():
    # Arrange
    symbol = "ES"
    name = "E-mini S&P 500"
    exchange = "CME"
    contract_size = 50.0
    tick_size = 0.25
    
    # Act
    asset = FuturesAsset(symbol=symbol, name=name, exchange=exchange, 
                         contract_size=contract_size, tick_size=tick_size)
    
    # Assert
    assert asset.symbol == symbol
    assert asset.name == name
    assert asset.exchange == exchange
    assert asset.contract_size == contract_size
    assert asset.tick_size == tick_size
    assert asset.asset_type == "futures"
    
def test_futures_continuous_contract():
    # Arrange
    symbol = "CL"  # Crude Oil
    asset = FuturesAsset(symbol=symbol)
    mock_connector = MockIBConnector()
    asset.set_connector(mock_connector)
    
    start_date = "2023-01-01"
    end_date = "2023-03-31"
    
    # Act
    continuous_data = asset.get_continuous_contract(start_date, end_date)
    
    # Assert
    assert not continuous_data.empty
    assert isinstance(continuous_data, pd.DataFrame)
    assert 'Close' in continuous_data.columns
    assert 'Volume' in continuous_data.columns
    assert 'Roll' in continuous_data.columns  # Should indicate contract roll dates
```

### Integration Test: Strategy Execution

```python
def test_pairs_strategy_integration():
    # Arrange
    config = {
        'pairs': {
            'ESM23_NQM23': {
                'leg1': 'ESM23',
                'leg2': 'NQM23',
                'hedge_ratio_method': 'kalman',
                'z_score_window': 20
            }
        },
        'entry_threshold': 2.0,
        'exit_threshold': 0.5
    }
    
    mock_data_provider = MockDataProvider()
    # Load test data that contains a known divergence pattern
    mock_data_provider.load_test_data('ESM23', 'tests/fixtures/ES_sample_data.csv')
    mock_data_provider.load_test_data('NQM23', 'tests/fixtures/NQ_sample_data.csv')
    
    # Create components
    data_processor = DataProcessor(data_provider=mock_data_provider)
    spread_analyzer = SpreadAnalyzer()
    signal_generator = SignalGenerator(config=config)
    
    # Act
    # Process the full strategy pipeline
    data = data_processor.get_data(['ESM23', 'NQM23'], '2023-01-01', '2023-03-31')
    pair_analysis = spread_analyzer.analyze_pair('ESM23_NQM23', data, 
                                               leg1='ESM23', leg2='NQM23', 
                                               hedge_ratio_method='kalman')
    signals = signal_generator.generate_signals(pair_analysis, 
                                              entry_threshold=2.0, 
                                              exit_threshold=0.5)
    
    # Assert
    assert 'hedge_ratio' in pair_analysis
    assert 'spread_series' in pair_analysis
    assert 'z_score_series' in pair_analysis
    assert 'signal' in signals
    
    # Verify we get signals at expected divergence points in the test data
    # This would be specific to the test data pattern
    assert signals['signal'].abs().max() > 0  # At least one trade signal
    
    # Count expected number of entries and exits
    entry_signals = signals['signal'].diff().abs() > 0
    assert entry_signals.sum() > 0  # Should have some entries/exits
```

## Testing Automated Trading Safety

Special care will be taken to ensure trading components don't accidentally place real orders:

1. **Environment Detection**: All tests will verify they're running in a test environment
2. **Forced Read-Only Mode**: Trading connectors will be forced into read-only mode during tests
3. **Mock Brokers**: Always use mock broker interfaces for tests
4. **Port Verification**: Ensure connections are always made to paper trading ports, never live trading

## Coverage Goals

- Unit test coverage: Aim for 80%+ for all critical components
- Integration test coverage: Aim for key workflows to be fully tested
- Priority on risk management and execution components

## Test Documentation

Each test file will include:

1. Module documentation explaining the test purpose
2. Test case descriptions
3. Fixture documentation explaining test data properties

## Next Steps

1. Create tests directory structure
2. Implement mock components
3. Start with unit tests for completed modules
4. Develop fixture data for cointegrated series
5. Implement basic integration tests 