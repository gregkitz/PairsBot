# Intraday ML System: Troubleshooting Guide

This troubleshooting guide provides detailed instructions for diagnosing and resolving common issues with the Intraday ML Trading System.

## Table of Contents
1. [System Health Diagnostics](#system-health-diagnostics)
2. [Data Issues](#data-issues)
3. [Model Performance Issues](#model-performance-issues)
4. [Trade Execution Problems](#trade-execution-problems)
5. [System Performance Issues](#system-performance-issues)
6. [Error Messages Reference](#error-messages-reference)
7. [Recovery Procedures](#recovery-procedures)

## System Health Diagnostics

### Running Full System Health Check
Use the health check tool to perform a full system diagnostic:

```bash
python -m src.diagnostics.system_health_check --verbose
```

This checks:
- Service status
- Database connectivity
- Data feed status
- Model health
- Disk space
- Memory utilization
- System logs for errors

### Checking Component Status
Check individual components:

```bash
# Check data services
python -m src.diagnostics.check_component --component data_services

# Check ML models
python -m src.diagnostics.check_component --component ml_models

# Check trading system
python -m src.diagnostics.check_component --component trading_system
```

### Analyzing Log Files
Examine logs for error patterns:

```bash
# View recent errors
python -m src.diagnostics.log_analyzer --level ERROR --hours 24

# Search for specific error patterns
python -m src.diagnostics.log_analyzer --pattern "ConnectionError"
```

## Data Issues

### Missing Data

#### Symptoms
- Gaps in price charts
- NaN values in DataFrames
- Log messages indicating missing data
- Failed backtests due to data gaps

#### Diagnostic Steps
1. Check data source status:
   ```bash
   python -m src.data.check_data_source_status
   ```

2. Analyze data completeness:
   ```bash
   python -m src.data.analyze_data_completeness --symbol GC --start-date YYYY-MM-DD --end-date YYYY-MM-DD
   ```

3. Inspect specific time periods:
   ```bash
   python -m src.data.inspect_timeframe --symbol GC --date YYYY-MM-DD --hour HH
   ```

#### Solutions
1. Reload data from source:
   ```bash
   python -m src.data.reload_data --symbol GC --date YYYY-MM-DD
   ```

2. Fill gaps with interpolation:
   ```bash
   python -m src.data.fill_missing_data --method interpolate --symbol GC --start-date YYYY-MM-DD --end-date YYYY-MM-DD
   ```

3. Apply alternative data source:
   ```bash
   python -m src.data.use_alternative_source --symbol GC --primary-source false
   ```

### Data Quality Issues

#### Symptoms
- Extreme price movements that don't match market
- Inconsistent spreads
- Signal generation issues
- Suspicious backtest results

#### Diagnostic Steps
1. Run data quality checks:
   ```bash
   python -m src.data.quality_check --symbol GC --date YYYY-MM-DD
   ```

2. Visualize suspicious data:
   ```bash
   python -m src.visualization.plot_data_quality --symbol GC --date YYYY-MM-DD
   ```

3. Compare with alternative sources:
   ```bash
   python -m src.data.compare_sources --symbol GC --date YYYY-MM-DD
   ```

#### Solutions
1. Filter outliers:
   ```bash
   python -m src.data.filter_outliers --symbol GC --method zscore --threshold 4
   ```

2. Replace corrupted data segments:
   ```bash
   python -m src.data.replace_data_segment --symbol GC --start-time "YYYY-MM-DD HH:MM" --end-time "YYYY-MM-DD HH:MM" --source alternate
   ```

3. Regenerate derived data:
   ```bash
   python -m src.data.regenerate_derived_data --pair GC_SI --force
   ```

### Data Feed Connection Issues

#### Symptoms
- Stale prices
- Connection timeout errors
- API rate limit errors
- Incomplete real-time data

#### Diagnostic Steps
1. Test API connectivity:
   ```bash
   python -m src.data.test_api_connection --source [source_name]
   ```

2. Check credentials:
   ```bash
   python -m src.data.verify_credentials --source [source_name]
   ```

3. Monitor rate limits:
   ```bash
   python -m src.monitoring.check_api_rate_limits --source [source_name]
   ```

#### Solutions
1. Reset API connection:
   ```bash
   python -m src.data.reset_connection --source [source_name]
   ```

2. Update API credentials:
   ```bash
   python -m src.config.update_credentials --source [source_name]
   ```

3. Implement rate limiting:
   ```bash
   python -m src.config.set_rate_limit --source [source_name] --max-requests 60 --per-minute true
   ```

4. Switch to backup data source:
   ```bash
   python -m src.data.enable_backup_source --primary [source_name] --backup [backup_source]
   ```

## Model Performance Issues

### Model Drift

#### Symptoms
- Decreasing prediction accuracy
- Growing differences between expected and actual outcomes
- Increasing false positives/negatives

#### Diagnostic Steps
1. Analyze prediction performance over time:
   ```bash
   python -m src.ml_enhancements.analyze_prediction_drift --model signal_filter --days 30
   ```

2. Compare feature distributions:
   ```bash
   python -m src.ml_enhancements.feature_drift_analysis --model signal_filter
   ```

3. Evaluate on recent data:
   ```bash
   python -m src.ml_enhancements.evaluate_model_recent --model signal_filter --days 7
   ```

#### Solutions
1. Retrain model with recent data:
   ```bash
   python -m src.ml_enhancements.retrain_model --model signal_filter --include-recent true
   ```

2. Update feature set:
   ```bash
   python -m src.ml_enhancements.update_feature_set --model signal_filter
   ```

3. Adjust model hyperparameters:
   ```bash
   python -m src.ml_enhancements.tune_hyperparameters --model signal_filter
   ```

4. Deploy fallback model:
   ```bash
   python -m src.ml_enhancements.activate_fallback_model --model signal_filter
   ```

### Incorrect Regime Classification

#### Symptoms
- Sudden changes in trading performance
- Parameters not matching current market conditions
- Unexpected signal patterns

#### Diagnostic Steps
1. Analyze current regime classification:
   ```bash
   python -m src.ml_enhancements.regime_detection.analyze_current_regime --pair GC_SI
   ```

2. Compare regime indicators with historical patterns:
   ```bash
   python -m src.ml_enhancements.regime_detection.compare_regimes --current --historical
   ```

3. Visualize regime features:
   ```bash
   python -m src.visualization.plot_regime_features --days 30
   ```

#### Solutions
1. Force regime reclassification:
   ```bash
   python -m src.ml_enhancements.regime_detection.reclassify --pair GC_SI
   ```

2. Adjust regime boundaries:
   ```bash
   python -m src.ml_enhancements.regime_detection.adjust_boundaries --config config/regime_config.yml
   ```

3. Retrain regime classifier:
   ```bash
   python -m src.ml_enhancements.regime_detection.retrain_classifier
   ```

4. Override regime (temporary):
   ```bash
   python -m src.ml_enhancements.regime_detection.manual_override --regime high_volatility --expiry "YYYY-MM-DD HH:MM"
   ```

### Model Loading Failures

#### Symptoms
- Error messages about missing model files
- Model prediction failures
- ValueError or FileNotFoundError exceptions

#### Diagnostic Steps
1. Check model files:
   ```bash
   python -m src.diagnostics.check_model_files --model signal_filter
   ```

2. Verify model version compatibility:
   ```bash
   python -m src.ml_enhancements.check_model_version --model signal_filter
   ```

3. Test model loading:
   ```bash
   python -m src.ml_enhancements.test_model_load --model signal_filter
   ```

#### Solutions
1. Restore model from backup:
   ```bash
   python -m src.ml_enhancements.restore_model --model signal_filter --version latest_stable
   ```

2. Rebuild model:
   ```bash
   python -m src.ml_enhancements.rebuild_model --model signal_filter
   ```

3. Fix file permissions:
   ```bash
   python -m src.system.fix_permissions --path models/
   ```

## Trade Execution Problems

### Signal Generation Issues

#### Symptoms
- No signals generated when expected
- Erratic signal patterns
- Signals inconsistent with market conditions

#### Diagnostic Steps
1. Check signal generation process:
   ```bash
   python -m src.diagnostics.debug_signal_generation --pair GC_SI --date YYYY-MM-DD
   ```

2. Inspect signal thresholds:
   ```bash
   python -m src.diagnostics.inspect_thresholds --pair GC_SI
   ```

3. Analyze raw vs. enhanced signals:
   ```bash
   python -m src.visualization.compare_signals --pair GC_SI --raw --enhanced --date YYYY-MM-DD
   ```

#### Solutions
1. Reset signal parameters:
   ```bash
   python -m src.config.reset_signal_parameters --pair GC_SI
   ```

2. Adjust confidence thresholds:
   ```bash
   python -m src.config.adjust_threshold --model signal_filter --threshold 0.6
   ```

3. Verify spread calculation:
   ```bash
   python -m src.signal_generation.recalculate_spreads --pair GC_SI
   ```

4. Override signal processing:
   ```bash
   python -m src.diagnostics.override_signal_processing --use-statistical-only --pair GC_SI
   ```

### Order Execution Failures

#### Symptoms
- Orders not being executed
- Execution delays
- Partial fills
- Error messages in execution logs

#### Diagnostic Steps
1. Check execution logs:
   ```bash
   python -m src.execution.check_logs --hours 24
   ```

2. Test exchange connectivity:
   ```bash
   python -m src.execution.test_exchange_connection
   ```

3. Verify order parameters:
   ```bash
   python -m src.execution.verify_order_parameters --pair GC_SI
   ```

#### Solutions
1. Reset execution module:
   ```bash
   python -m src.execution.reset_execution_module
   ```

2. Update exchange credentials:
   ```bash
   python -m src.config.update_exchange_credentials
   ```

3. Adjust execution parameters:
   ```bash
   python -m src.config.adjust_execution_parameters --slippage 2 --timeout 30
   ```

4. Enable backup execution route:
   ```bash
   python -m src.execution.enable_backup_route
   ```

### Position Management Issues

#### Symptoms
- Incorrect position sizing
- Failed exit orders
- Incorrect hedge ratios
- Position drift between pairs

#### Diagnostic Steps
1. Verify current positions:
   ```bash
   python -m src.execution.verify_positions
   ```

2. Check hedge ratio calculation:
   ```bash
   python -m src.diagnostics.check_hedge_ratios --pair GC_SI
   ```

3. Analyze exit signal generation:
   ```bash
   python -m src.diagnostics.analyze_exit_signals --pair GC_SI --days 5
   ```

#### Solutions
1. Resynchronize positions:
   ```bash
   python -m src.execution.resync_positions
   ```

2. Force recalculation of hedge ratios:
   ```bash
   python -m src.signal_generation.recalculate_hedge_ratios --pair GC_SI
   ```

3. Adjust position limits:
   ```bash
   python -m src.config.update_position_limits --pair GC_SI --max-size 10
   ```

4. Force exit for specific position:
   ```bash
   python -m src.execution.force_exit --pair GC_SI --reason "manual intervention"
   ```

## System Performance Issues

### High CPU Usage

#### Symptoms
- System slowdowns
- Delayed signal processing
- High CPU usage alerts
- Missed trading opportunities

#### Diagnostic Steps
1. Identify resource-intensive processes:
   ```bash
   python -m src.diagnostics.analyze_resource_usage --resource cpu
   ```

2. Check process timing:
   ```bash
   python -m src.diagnostics.timing_analysis --component all
   ```

3. Analyze computational bottlenecks:
   ```bash
   python -m src.diagnostics.find_bottlenecks
   ```

#### Solutions
1. Optimize feature calculation:
   ```bash
   python -m src.optimization.optimize_features --reduce-computation
   ```

2. Reduce data resolution when appropriate:
   ```bash
   python -m src.config.adjust_data_resolution --timeframe 5m
   ```

3. Disable non-critical features:
   ```bash
   python -m src.config.disable_features --list "feature1,feature2,feature3"
   ```

4. Scale computational resources:
   ```bash
   python -m src.system.scale_resources --cpu 8
   ```

### Memory Leaks

#### Symptoms
- Increasing memory usage over time
- System slowdowns after running for extended periods
- Out of memory errors
- Frequent garbage collection

#### Diagnostic Steps
1. Monitor memory usage:
   ```bash
   python -m src.diagnostics.monitor_memory --duration 60 --interval 5
   ```

2. Identify memory-intensive components:
   ```bash
   python -m src.diagnostics.analyze_component_memory
   ```

3. Check for large objects:
   ```bash
   python -m src.diagnostics.find_large_objects
   ```

#### Solutions
1. Implement periodic garbage collection:
   ```bash
   python -m src.system.configure_gc --frequency 60
   ```

2. Reduce data caching:
   ```bash
   python -m src.config.adjust_cache_size --max-size 500
   ```

3. Enable memory-efficient mode:
   ```bash
   python -m src.system.enable_memory_efficient_mode
   ```

4. Restart memory-intensive services:
   ```bash
   python -m src.system.restart_service --service feature_engineering --interval 6h
   ```

### Database Performance Issues

#### Symptoms
- Slow queries
- Database timeouts
- Increasing I/O wait times
- Storage filling up

#### Diagnostic Steps
1. Analyze database performance:
   ```bash
   python -m src.db.analyze_performance
   ```

2. Check slow queries:
   ```bash
   python -m src.db.find_slow_queries
   ```

3. Verify index usage:
   ```bash
   python -m src.db.check_indexes
   ```

4. Check database size:
   ```bash
   python -m src.db.check_size
   ```

#### Solutions
1. Optimize database:
   ```bash
   python -m src.db.optimize_database
   ```

2. Add indexes for common queries:
   ```bash
   python -m src.db.add_indexes --fields "date,symbol"
   ```

3. Purge old data:
   ```bash
   python -m src.db.purge_old_data --older-than 90d --backup true
   ```

4. Shard database:
   ```bash
   python -m src.db.configure_sharding --by-date true
   ```

## Error Messages Reference

### Common Error Messages and Solutions

#### "Failed to connect to data source"
**Cause**: API connection issue, network problem, or invalid credentials
**Solution**: 
```bash
python -m src.data.test_api_connection --verbose
python -m src.config.update_credentials
```

#### "Model prediction failed: Input shapes do not match"
**Cause**: Feature calculation issue or model version mismatch
**Solution**:
```bash
python -m src.ml_enhancements.check_feature_compatibility
python -m src.ml_enhancements.reset_model_cache
```

#### "Insufficient margin for order execution"
**Cause**: Risk limits exceeded or position sizing issue
**Solution**:
```bash
python -m src.risk.check_margins
python -m src.config.adjust_position_sizing --reduce-factor 0.8
```

#### "Database deadlock detected"
**Cause**: Competing database transactions
**Solution**:
```bash
python -m src.db.unlock_tables
python -m src.db.optimize_transactions
```

#### "Signal processing timed out"
**Cause**: CPU overload or algorithmic inefficiency
**Solution**:
```bash
python -m src.diagnostics.profile_signal_processing
python -m src.optimization.optimize_signal_processing
```

## Recovery Procedures

### System Crash Recovery
1. Check system status:
   ```bash
   python -m src.diagnostics.check_system_status
   ```

2. Verify data integrity:
   ```bash
   python -m src.data.verify_integrity
   ```

3. Restore system state:
   ```bash
   python -m src.system.restore_state --from-checkpoint latest
   ```

4. Resynchronize with market:
   ```bash
   python -m src.execution.resynchronize_positions
   ```

5. Restart system components:
   ```bash
   python -m src.system.start_services --sequence true
   ```

### Database Recovery
1. Stop dependent services:
   ```bash
   python -m src.system.stop_dependent_services --component database
   ```

2. Check database corruption:
   ```bash
   python -m src.db.check_corruption
   ```

3. Repair database (if needed):
   ```bash
   python -m src.db.repair
   ```

4. Restore from backup (if repair fails):
   ```bash
   python -m src.db.restore --backup backup/db_YYYYMMDD.dump
   ```

5. Validate restored data:
   ```bash
   python -m src.db.validate_data
   ```

6. Restart services:
   ```bash
   python -m src.system.start_dependent_services --component database
   ```

### Model Recovery
1. Identify corrupted models:
   ```bash
   python -m src.ml_enhancements.check_model_integrity
   ```

2. Restore from backup:
   ```bash
   python -m src.ml_enhancements.restore_models --from-backup models_backup/YYYYMMDD
   ```

3. Verify restored models:
   ```bash
   python -m src.ml_enhancements.verify_models
   ```

4. Retrain if necessary:
   ```bash
   python -m src.ml_enhancements.retrain_failed_models
   ```

### Emergency Trading Halt Recovery
1. Assess market conditions:
   ```bash
   python -m src.market_analysis.assess_conditions
   ```

2. Check position status:
   ```bash
   python -m src.execution.check_positions
   ```

3. Perform risk assessment:
   ```bash
   python -m src.risk.assess_current_risk
   ```

4. Create recovery plan:
   ```bash
   python -m src.recovery.generate_plan
   ```

5. Execute phased recovery:
   ```bash
   python -m src.recovery.execute_plan --staged true
   ```

6. Resume normal operations:
   ```bash
   python -m src.system.resume_normal_operations
   ``` 