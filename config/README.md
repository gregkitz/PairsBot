# Configuration Directory

This directory contains configuration files for the Quant-Trader system. Configuration files are organized by functionality and can be in either JSON or YAML format.

## Configuration Structure

The configuration directory is organized as follows:

```
config/
├── data_config.yaml               # Data processing configuration
├── default_config.json            # Default system configuration
├── paper_trading.json             # Paper trading configuration
├── data_flow_config.yaml          # Data flow pipeline configuration
├── monitoring.json                # Monitoring and alerting configuration
├── models/                        # ML model configurations
│   ├── signal_filter.json         # Signal filter model configuration
│   ├── entry_exit.json            # Entry/exit timing model configuration  
│   └── regime_detection.json      # Regime detection model configuration
├── optimization/                  # Parameter optimization configurations
│   └── parameter_optimization.json # Parameter optimization configuration
└── strategies/                    # Strategy-specific configurations
    ├── pairs_trading.json         # Pairs trading strategy configuration
    └── intraday_ml.json           # Intraday ML strategy configuration
```

## Configuration Files

### Main Configuration Files

- **`default_config.json`**: The default system configuration used when no specific configuration is provided.
- **`data_config.yaml`**: Configuration for data loading, processing, and validation.
- **`paper_trading.json`**: Configuration for paper trading mode.
- **`data_flow_config.yaml`**: Configuration for data processing pipelines.
- **`monitoring.json`**: Configuration for system monitoring and alerts.

### Model Configurations

The `models/` subdirectory contains configuration files for ML models:

- **`signal_filter.json`**: Configuration for the signal filter model.
- **`entry_exit.json`**: Configuration for the entry/exit timing model.
- **`regime_detection.json`**: Configuration for the market regime detection model.

### Optimization Configurations

The `optimization/` subdirectory contains configuration files for parameter optimization:

- **`parameter_optimization.json`**: Configuration for the parameter optimization process.

### Strategy Configurations

The `strategies/` subdirectory contains configuration files for specific trading strategies:

- **`pairs_trading.json`**: Configuration for the basic pairs trading strategy.
- **`intraday_ml.json`**: Configuration for the intraday ML-enhanced strategy.

## Configuration Format

### JSON Configuration Example

```json
{
  "system": {
    "log_level": "INFO",
    "log_file": "logs/system.log",
    "output_dir": "output/backtest_results",
    "plot_dir": "output/plots",
    "show_plots": true,
    "save_plots": false
  },
  "data": {
    "historical_dir": "data/processed",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "timeframe": "5min"
  },
  "cointegration": {
    "min_correlation": 0.7,
    "max_half_life": 60,
    "min_half_life": 5,
    "pvalue_threshold": 0.05,
    "train_test_split": 0.7,
    "min_cointegration_pct": 0.8
  }
}
```

### YAML Configuration Example

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  timeframe: 5min
  start_date: 2022-01-01
  end_date: 2022-12-31
  symbols:
    - GC  # Gold
    - SI  # Silver
    - ZB  # 30-Year Treasury Bonds
    - ZN  # 10-Year Treasury Notes
  
processing:
  fill_missing: true
  normalize: true
  remove_outliers: true
  outlier_threshold: 3.0
```

## Using Configuration Files

Configuration files can be specified when running commands through the main.py interface:

```bash
# Using the default configuration
python main.py backtest

# Using a specific configuration
python main.py backtest --config config/strategies/intraday_ml.json

# Overriding specific configuration values
python main.py backtest --config config/default_config.json --start-date 2022-06-01 --end-date 2022-12-31
```

## Configuration Management

The system uses the `src/config/configuration.py` module to manage configuration files. This module provides:

- Loading configuration files in different formats (JSON, YAML)
- Validating configuration against schemas
- Merging configurations from different sources
- Environment-specific configuration overrides 