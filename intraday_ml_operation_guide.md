# Intraday ML Trading System: Operation Guide

This guide provides detailed instructions for operating the intraday ML trading system to generate profit. It covers the entire workflow from setup to maintenance.

## System Architecture Overview

The intraday ML trading system consists of several interconnected components:

1. **Data Pipeline**: Handles data loading, preprocessing, and feature generation
2. **ML Models**: Provides signal filtering, entry/exit timing, and regime detection
3. **Backtesting Engine**: Tests strategies with historical data
4. **Parameter Optimization**: Tunes parameters for different market regimes
5. **Paper/Live Trading**: Executes trades based on signals
6. **Monitoring & Alerting**: Tracks performance and detects issues

## Step 1: System Setup

### Install Dependencies

```bash
# Navigate to project root
cd /path/to/quant-trader

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, talib, matplotlib"
```

### Configure Data Sources

Edit `config/data_sources.json` to set up your data providers:

```json
{
  "intraday_data": {
    "provider": "your_data_provider",
    "api_key": "your_api_key",
    "symbols": ["GC", "SI", "ZB", "ZN"],
    "timeframes": ["1min", "5min", "15min", "1hour"]
  }
}
```

### Configure Trading Parameters

Edit `config/trading_parameters.json` to set risk parameters:

```json
{
  "account_size": 100000,
  "max_allocation_per_pair": 0.1,
  "max_pairs_active": 3,
  "max_daily_loss_pct": 0.02,
  "transaction_costs": {
    "commission": 2.0,
    "slippage": 1.0
  }
}
```

## Step 2: Data Collection and Processing

### Historical Data Download

For initial training and backtesting, download historical data:

```bash
# Download and process historical data
python scripts/download_historical_data.py --symbols GC,SI,ZB,ZN --start-date 2021-01-01 --end-date 2023-01-01 --timeframe 5min
```

### Data Validation

Verify data quality before proceeding:

```bash
# Run data validation checks
python scripts/validate_data.py --data-dir data/processed

# Visualize data for inspection
python scripts/visualize_data.py --symbols GC,SI --timeframe 5min --days 5
```

## Step 3: Model Training

### Pair Selection

Identify cointegrated pairs for trading:

```bash
# Analyze pairs for cointegration
python analyze_pairs.py --min-correlation 0.7 --output-file output/pairs_analysis.json
```

### Feature Engineering

Generate features for model training:

```bash
# Extract features for ML models
python test_intraday_features.py --pair GC_SI --timeframe 5min
```

### ML Model Training

Train the ML models for signal enhancement:

```bash
# Train signal filter, entry/exit timing, and regime detection models
python train_intraday_models.py --config config/ml_training.json
```

## Step 4: Backtesting and Parameter Optimization

### Basic Backtesting

Test strategy performance with historical data:

```bash
# Run backtest for a specific pair
python run_intraday_backtest.py --pair GC_SI --start-date 2022-01-01 --end-date 2022-12-31 --timeframe 5min
```

### Parameter Optimization

Optimize parameters for different market regimes:

```bash
# Run parameter optimization
python run_intraday_parameter_optimization.py --pairs-file output/pairs_analysis.json --start-date 2022-01-01 --end-date 2022-12-31

# Test model refinements
python test_model_refinement.py --data-file data/processed/GC_SI_combined.parquet
```

### Performance Analysis

Analyze backtest results:

```bash
# Analyze performance metrics
python evaluate_intraday_models.py --backtest-results output/backtest_results/GC_SI_backtest.json
```

## Step 5: Paper Trading Deployment

### Configure Paper Trading

Set up paper trading with configurations that match your intended live trading:

```bash
# Edit paper trading configuration
vim config/paper_trading.json

# Example configuration content:
# {
#   "account_size": 100000,
#   "pairs": ["GC_SI", "ZB_ZN"],
#   "timeframe": "5min",
#   "trading_hours": {
#     "start": "09:30",
#     "end": "15:45"
#   },
#   "use_ml_enhancements": true,
#   "enable_alerts": true
# }
```

### Start Paper Trading

Launch the paper trading system:

```bash
# Start paper trading with ML enhancements
python run_ml_paper_trader.py --config config/paper_trading.json
```

### Monitor Performance

Track paper trading performance:

```bash
# View daily performance report
python scripts/generate_performance_report.py --mode paper --days 5

# Open the dashboard in your browser
open output/paper_trading/dashboard.html
```

## Step 6: Live Trading Transition

### Broker Setup

Configure your trading broker:

```bash
# Set up broker connection
python setup_broker_connection.py --broker your_broker --credentials config/broker_credentials.json
```

### Small-Scale Live Trading

Start with minimal position sizes:

```bash
# Start live trading with reduced risk
python run_ml_paper_trader.py --config config/live_trading_minimal.json --mode live
```

### Full Deployment

After successful small-scale testing, deploy the full system:

```bash
# Deploy full live trading system
python run_ml_paper_trader.py --config config/live_trading.json --mode live
```

## Step 7: Ongoing Operation

### Daily Operations Checklist

1. **Morning Setup (Before Market Open)**
   ```bash
   # Check system health
   python scripts/check_system_health.py
   
   # Update market regime detection
   python scripts/detect_current_regime.py
   
   # Prepare daily watchlist
   python scripts/generate_watchlist.py
   ```

2. **During Trading Hours**
   ```bash
   # Monitor active trades
   python scripts/monitor_trades.py --refresh-rate 300
   
   # Check alerts
   python scripts/check_alerts.py
   ```

3. **End of Day**
   ```bash
   # Generate performance report
   python scripts/generate_performance_report.py --date today
   
   # Check for model drift
   python scripts/check_model_drift.py
   ```

### Weekly Maintenance

```bash
# Retrain models with new data
python train_intraday_models.py --incremental

# Update parameter optimization
python run_intraday_parameter_optimization.py --quick-mode

# Analyze weekly performance
python scripts/analyze_weekly_performance.py
```

### Monthly Review

```bash
# Comprehensive system review
python scripts/generate_monthly_review.py

# Pair relationship reassessment
python analyze_pairs.py --monthly-update
```

## Troubleshooting

### Common Issues

1. **No Signals Generated**
   ```bash
   # Debug signal generation
   python scripts/debug_signals.py --pair GC_SI --date today
   ```

2. **Poor Model Performance**
   ```bash
   # Analyze model accuracy
   python scripts/analyze_model_performance.py --model signal_filter
   
   # Force model retraining
   python scripts/force_model_retrain.py --all
   ```

3. **Execution Issues**
   ```bash
   # Check execution logs
   python scripts/examine_execution_logs.py --days 1
   
   # Test broker connection
   python scripts/test_broker_connection.py
   ```

## Performance Optimization

### System Tuning

```bash
# Profile system performance
python scripts/profile_system.py --component feature_engineering

# Optimize data loading
python scripts/optimize_data_pipeline.py
```

### Strategy Refinement

```bash
# Analyze underperforming trades
python scripts/analyze_trade_performance.py --filter losing

# Test feature importance
python scripts/analyze_feature_importance.py
```

## Key Files and Locations

- **Configuration**: `config/`
- **ML Models**: `models/intraday/`
- **Trading Logs**: `logs/trading/`
- **Performance Reports**: `output/reports/`
- **Backtest Results**: `output/backtest_results/`
- **Data Storage**: `data/processed/`

## Critical Success Factors

1. **Risk Management First**: Always prioritize risk management over returns. The system has built-in risk controls, but you should monitor them vigilantly.

2. **Start Small**: Begin with small position sizes when transitioning to live trading.

3. **Continuous Monitoring**: Market regimes change; your system needs to adapt. Daily monitoring is essential.

4. **Gradual Enhancement**: Don't modify multiple components simultaneously. Make incremental changes and test thoroughly.

5. **Data Quality**: Garbage in, garbage out. Regularly verify your data quality.

6. **Realistic Expectations**: The system won't be profitable every day. Look for consistent performance over weeks and months.

7. **Discipline**: Follow the system's signals; don't override them based on gut feelings.

By following this operation guide, you'll be able to effectively deploy and manage the intraday ML trading system for profitable operation. 