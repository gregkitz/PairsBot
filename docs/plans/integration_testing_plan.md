# Integration Testing Plan with Reduced Data

This document outlines a plan for testing the full pipeline integration without waiting for long-running processes. It focuses on using reduced datasets and mocks to verify functionality quickly.

## Integration Testing Objectives

1. Validate the full workflow from data processing to signal generation to execution
2. Ensure all components can work together as expected
3. Identify integration issues between components
4. Verify main.py commands properly orchestrate the system

## Reduced Data Strategy

### 1. Creating Minimal Test Datasets

- **Time Period**: Use 3 months of data instead of multiple years
- **Instrument Selection**: Limit to 2-3 highly liquid instruments
- **Timeframe**: Use larger timeframes (1 hour instead of 5 minutes) to reduce data points
- **Pre-processed Data**: Create pre-computed intermediate results to skip expensive computations

### 2. Pre-computed ML Models

- **Pre-trained Models**: Use small, pre-trained models saved to disk
- **Simplified Models**: Create simplified models with fewer features for testing
- **Mock Predictions**: For pure integration testing, use mock prediction outputs

### 3. Configuration Overrides

- **Quick Mode Parameters**: Create test-specific configurations with reduced iterations
- **Simplified Strategies**: Use basic strategy variants for testing
- **Limited Validation**: Reduce cross-validation folds and testing periods

## Integration Test Scenarios

### Scenario 1: Data Processing Pipeline
- **Objective**: Verify data can flow from raw files to processed format
- **Command**: `python main.py process-data --symbols GC SI --start-date 2023-01-01 --end-date 2023-03-31 --test-mode`
- **Validation**: Check output files exist and contain expected data structure

### Scenario 2: Pair Analysis Pipeline
- **Objective**: Verify pair selection and cointegration testing
- **Command**: `python main.py analyze-pairs --tickers GC SI ZB ZN --test-mode`
- **Validation**: Check pairs ranking, cointegration statistics, and output files

### Scenario 3: Basic Backtest Pipeline
- **Objective**: Verify historical trading simulation and performance metrics
- **Command**: `python main.py backtest --pairs GC_SI --start-date 2023-01-01 --end-date 2023-03-31 --test-mode`
- **Validation**: Check trades, performance metrics, and output reports

### Scenario 4: ML Model Training Pipeline
- **Objective**: Verify model training workflow with minimal data
- **Command**: `python main.py train-models --pair GC_SI --timeframe 1hour --start-date 2023-01-01 --end-date 2023-03-31 --test-mode`
- **Validation**: Check model files, training metrics, and feature importance

### Scenario 5: Parameter Optimization Pipeline
- **Objective**: Verify optimization routines with limited parameters
- **Command**: `python main.py optimize-parameters --pairs-file output/test_pairs.json --start-date 2023-01-01 --end-date 2023-03-31 --quick-mode --test-mode`
- **Validation**: Check optimization results, parameter ranges, and output files

### Scenario 6: ML-Enhanced Backtest Pipeline
- **Objective**: Verify ML model integration with trading strategy
- **Command**: `python main.py intraday-backtest --pairs GC_SI --start-date 2023-01-01 --end-date 2023-03-31 --use-ml --test-mode`
- **Validation**: Check ML-enhanced signals, performance comparison, and output reports

### Scenario 7: Paper Trading Initialization
- **Objective**: Verify paper trading setup without actual trading
- **Command**: `python main.py paper-trade --pairs GC_SI --test-mode --init-only`
- **Validation**: Check system initialization, configuration loading, and setup process

## Implementation Strategy

### 1. Test Mode Flag

Add a `--test-mode` flag to all main.py commands that:

```python
# In main.py command handlers
if args.test_mode:
    # Override configuration with test settings
    config.update(TEST_CONFIG)
    # Use reduced dataset paths
    data_path = 'data/test/'
    # Set low iteration counts for optimization
    config['optimization']['iterations'] = 5
    # Use pre-trained test models
    config['ml']['model_path'] = 'models/test/'
```

### 2. Test Data Generation Script

Create a script to generate minimal test datasets:

```python
# scripts/generate_test_data.py
def create_test_datasets():
    """Generate minimal datasets for integration testing."""
    # Download or extract a small subset of production data
    # Or generate synthetic data with cointegration properties
    # Save to data/test/ directory
    
def create_test_models():
    """Generate simplified pre-trained models for testing."""
    # Train models on minimal data or create mock models
    # Save to models/test/ directory
```

### 3. Integration Test Runner

Create a script that automatically runs all integration tests:

```python
# scripts/run_integration_tests.py
def run_all_integration_tests():
    """Run all integration test scenarios."""
    # First generate test data if needed
    if not os.path.exists('data/test'):
        generate_test_data()
        
    # Run each test scenario and collect results
    results = []
    for scenario in INTEGRATION_TEST_SCENARIOS:
        result = run_test_scenario(scenario)
        results.append(result)
        
    # Report results
    generate_integration_test_report(results)
```

## Test Environment Setup

### 1. Create Test Data Directory

```
/data/test/
  /raw/              # Small raw data files
  /processed/        # Pre-processed data
  /pairs/            # Pre-calculated pair statistics
  /features/         # Pre-calculated features
```

### 2. Create Test Models Directory

```
/models/test/
  /regime_detection/ # Simplified regime models
  /feature_selection/ # Feature importance results
  /signal_enhancement/ # Signal enhancement models
```

### 3. Create Test Configuration Files

```
/config/test/
  test_config.json          # Main test configuration
  test_optimization.json    # Test optimization parameters
  test_ml_training.json     # Test model training parameters
```

## Expected Outcomes

After running the integration tests, we should be able to:

1. Verify that all system components can work together
2. Identify any integration issues or dependency problems
3. Ensure the main.py interface correctly orchestrates the system
4. Validate the data flow through the entire system

## Timeline

- **Day 1**: Set up test data generation and configurations
- **Day 2**: Implement test mode flags in main.py commands
- **Day 3**: Create and run integration test scenarios 1-3
- **Day 4**: Create and run integration test scenarios 4-7
- **Day 5**: Fix issues, document results, and finalize integration testing

## Next Steps

1. Create the test data generation script
2. Implement the test mode flags in main.py
3. Set up the integration test directory structure
4. Run each scenario individually and fix issues
5. Automate the full integration test suite 