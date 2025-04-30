# Intraday ML System: User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [For Traders](#for-traders)
4. [For Analysts](#for-analysts)
5. [For System Administrators](#for-system-administrators)
6. [Common Operations](#common-operations)
7. [Troubleshooting](#troubleshooting)

## Introduction
This manual provides comprehensive guidance for using the Intraday ML Trading System. The system combines statistical pairs trading with machine learning enhancements to generate profitable intraday trading signals with high accuracy and adaptability to changing market conditions.

## System Overview

### Architecture
The Intraday ML System consists of several integrated components:
- Data Processing Pipeline
- ML Feature Engineering
- Model Training System
- Regime Detection
- Signal Generation and Enhancement
- Backtesting Engine
- Paper/Live Trading Integration

### Key Features
- Adaptive parameter management based on market regimes
- ML-enhanced signal filtering and timing
- Intraday-specific constraints and optimizations
- Comprehensive performance measurement
- Automated retraining and adaptation

### System Requirements
- CPU: 4+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 100GB+ for historical data
- OS: Linux, macOS, or Windows
- Python 3.8+ with required dependencies

## For Traders

### Getting Started

#### Setting Up Your Environment
1. Ensure you have the required API keys for your data provider
2. Configure your trading parameters in `config/trading_config.yml`
3. Initialize your environment:
   ```bash
   python -m src.initialize_environment
   ```

#### Daily Operations
1. **Start of Day**
   - Check system health dashboard
   - Verify data feeds are operational
   - Review overnight model retraining results

2. **During Trading Hours**
   - Monitor active signals dashboard
   - Review trade executions
   - Check regime classification alerts

3. **End of Day**
   - Review daily performance
   - Export reports
   - Check for system maintenance notifications

### Using the Trading Dashboard
The trading dashboard provides real-time monitoring of:
- Active pairs and positions
- Current market regime
- Recent signals and their confidence scores
- Today's P&L and performance metrics

To access the dashboard:
```bash
python -m src.ui.launch_dashboard --mode trader
```

### Managing Risk
- Set position size limits in `config/risk_config.yml`
- Configure stop-loss and take-profit levels
- Use the regime-specific risk controls

### Interpreting Signals
ML-enhanced signals include:
- Direction (long/short)
- Confidence score (0-1)
- Expected holding time
- Regime context

## For Analysts

### Model Training and Evaluation

#### Training New Models
1. Prepare training data:
   ```bash
   python -m src.data.prepare_training_data --start-date YYYY-MM-DD --end-date YYYY-MM-DD
   ```

2. Configure model parameters in `config/model_config.yml`

3. Train models:
   ```bash
   python -m src.ml_enhancements.train_intraday_models
   ```

4. Evaluate model performance:
   ```bash
   python -m src.ml_enhancements.evaluate_intraday_models --model-dir models/latest
   ```

#### Feature Engineering
- View current feature importance:
  ```bash
  python -m src.ml_enhancements.feature_engineering.analyze_features
  ```

- Add new features by extending the `FeatureGenerator` class
- Test feature performance:
  ```bash
  python -m src.ml_enhancements.evaluate_features --feature-list new_feature1,new_feature2
  ```

### Parameter Optimization
1. Configure optimization parameters in `config/optimization_config.yml`
2. Run optimization:
   ```bash
   python -m src.optimization.intraday_parameter_optimizer --pair GC_SI
   ```
3. Apply optimized parameters:
   ```bash
   python -m src.optimization.apply_parameters --param-set latest
   ```

### Backtesting
1. Configure backtest parameters in `config/backtest_config.yml`
2. Run backtest:
   ```bash
   python -m src.backtest.run_intraday_backtest --start-date YYYY-MM-DD --end-date YYYY-MM-DD
   ```
3. Analyze results:
   ```bash
   python -m src.backtest.analyze_backtest_results --results-file results/backtest_YYYYMMDD.pkl
   ```

### Using the Analysis Dashboard
The analysis dashboard provides tools for:
- Comparing model performance
- Analyzing regime transitions
- Visualizing feature importance
- Debugging trade performance

To access the dashboard:
```bash
python -m src.ui.launch_dashboard --mode analyst
```

## For System Administrators

### Installation
1. Clone repository:
   ```bash
   git clone https://github.com/company/quant-trader.git
   cd quant-trader
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp config/sample.env .env
   # Edit .env with your settings
   ```

4. Initialize database:
   ```bash
   python -m src.db.initialize_database
   ```

### Deployment
- **Development**: Use local environment
- **Testing**: Deploy with Docker Compose
- **Production**: Use Kubernetes with provided manifests

#### Docker Deployment
```bash
docker-compose -f deployments/docker-compose.yml up -d
```

#### Kubernetes Deployment
```bash
kubectl apply -f deployments/kubernetes/
```

### System Monitoring
1. Check system logs:
   ```bash
   python -m src.monitoring.view_logs
   ```

2. Monitor resource usage:
   ```bash
   python -m src.monitoring.resource_dashboard
   ```

3. Check health status:
   ```bash
   python -m src.monitoring.health_check
   ```

### Database Maintenance
- Backup database:
  ```bash
  python -m src.db.backup --output backup/db_YYYYMMDD.dump
  ```

- Restore database:
  ```bash
  python -m src.db.restore --input backup/db_YYYYMMDD.dump
  ```

- Optimize database:
  ```bash
  python -m src.db.optimize
  ```

### Update Procedures
1. Stop services:
   ```bash
   python -m src.system.stop_services
   ```

2. Backup data:
   ```bash
   python -m src.system.backup_data
   ```

3. Update code:
   ```bash
   git pull
   pip install -r requirements.txt
   ```

4. Apply migrations:
   ```bash
   python -m src.db.apply_migrations
   ```

5. Start services:
   ```bash
   python -m src.system.start_services
   ```

## Common Operations

### Data Management

#### Importing Historical Data
```bash
python -m src.data.import_historical_data --source [source] --symbols GC,SI,CL --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

#### Cleaning and Validating Data
```bash
python -m src.data.validate_data --source-path data/raw --output-path data/clean
```

#### Exporting Results
```bash
python -m src.reporting.export_results --format [csv|json|excel] --output reports/results_YYYYMMDD
```

### System Configuration

#### Configuring Trading Pairs
Edit `config/pairs_config.yml` to add or modify trading pairs:
```yaml
pairs:
  - symbols: [GC, SI]
    hedge_ratio_method: kalman
    z_score_window: 48
  - symbols: [CL, NG]
    hedge_ratio_method: ols
    z_score_window: 36
```

#### Setting Trading Hours
Edit `config/trading_hours.yml` to configure market hours:
```yaml
GC:
  market_open: "09:30"
  market_close: "16:00"
  timezone: "America/New_York"
```

#### Configuring Alerts
Edit `config/alerts_config.yml`:
```yaml
alerts:
  email:
    enabled: true
    recipients: ["trader@example.com", "analyst@example.com"]
  slack:
    enabled: true
    channel: "#trading-alerts"
  regime_change:
    notify: true
    channels: ["email", "slack"]
```

## Troubleshooting

### Common Issues

#### Data Feed Connection Issues
**Symptoms**: Missing data, stale prices, connection errors

**Solutions**:
1. Check API credentials in `.env` file
2. Verify network connectivity
3. Check data provider status page
4. Restart data services:
   ```bash
   python -m src.data.restart_data_feeds
   ```

#### Model Performance Degradation
**Symptoms**: Declining win rate, increasing drawdowns

**Solutions**:
1. Check for data drift:
   ```bash
   python -m src.ml_enhancements.check_data_drift
   ```
2. Force model retraining:
   ```bash
   python -m src.ml_enhancements.retrain_models --force
   ```
3. Review latest regime detection:
   ```bash
   python -m src.ml_enhancements.regime_detection.analyze_current_regime
   ```

#### System Errors
**Symptoms**: Services failing, error messages

**Solutions**:
1. Check logs:
   ```bash
   python -m src.system.check_error_logs
   ```
2. Verify resource usage:
   ```bash
   python -m src.monitoring.check_resources
   ```
3. Restart problematic services:
   ```bash
   python -m src.system.restart_service --service [service_name]
   ```
4. Restore from backup if necessary:
   ```bash
   python -m src.system.restore_from_backup --backup-path backups/latest
   ```

### Getting Help
- Internal Support: Contact the ML Trading Systems team
- Documentation: Refer to the comprehensive documentation in the `docs/` directory
- Issue Tracking: Submit detailed bug reports in the internal issue tracker

### Emergency Procedures
1. Halt all trading:
   ```bash
   python -m src.emergency.halt_trading
   ```
2. Close all positions:
   ```bash
   python -m src.emergency.close_all_positions
   ```
3. Contact the on-call support team 