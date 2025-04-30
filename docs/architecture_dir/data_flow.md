# Data Flow Architecture

This document outlines the flow of data throughout the system, detailing how data moves between different components and how transformations are applied along the way.

## High-Level Data Flow

```
[Data Sources] → [Data Collection] → [Data Processing] → [Feature Engineering] → [Model Training/Inference] → [Signal Generation] → [Trading Execution] → [Performance Monitoring]
```

## Detailed Data Flow

### 1. Data Sources
- **Historical Data**: 15 years of futures data for 130 top tickers
- **Real-time Market Data**: Provided through Interactive Brokers API
- **Alternative Data**: Market regime indicators, economic calendars

### 2. Data Collection
- **Historical Data Pipeline**: `src/data_processor/data_downloader.py`
  - Fetches historical data from stored sources
  - Handles data synchronization and updates
  
- **Real-time Data Pipeline**: `src/connectors/ib/ib_connector.py`
  - Connects to Interactive Brokers API
  - Provides real-time price updates and order execution
  - Maintains subscription to market data for active instruments

### 3. Data Processing
- **Data Cleaning**: `src/data_processor/data_cleaner.py`
  - Handles missing values, outliers, and data errors
  - Ensures data consistency across different sources
  
- **Data Normalization**: `src/data_processor/normalizer.py`
  - Applies appropriate scaling and normalization
  - Handles contract rollovers for continuous futures

- **Pair Construction**: `src/pair_trading/pair_builder.py`
  - Calculates spreads between instruments
  - Applies appropriate hedge ratios (static, rolling, or Kalman)
  - Constructs normalized spread series for analysis

### 4. Feature Engineering
- **Feature Generator**: `src/ml_enhancements/feature_engineering/feature_generator.py`
  - Creates technical indicators, statistical features, and derived metrics
  - Builds temporal features and lag variables
  - Generates pair-specific features like spread momentum and volatility

- **Feature Selection**: `src/ml_enhancements/feature_engineering/feature_selector.py`
  - Uses importance analysis to select relevant features
  - Applies dimensionality reduction when appropriate
  - Creates feature groups for different model purposes

### 5. Model Training & Inference
- **Model Training Pipeline**: `src/ml_enhancements/model_retraining.py`
  - Trains models on historical data with appropriate validation
  - Implements cross-validation to prevent overfitting
  - Handles hyperparameter tuning and model selection
  
- **Market Regime Detection**: `src/ml_enhancements/regime_detection/market_regime_classifier.py`
  - Identifies current market regimes
  - Provides regime-specific model selection
  
- **Inference Engine**: `src/ml_enhancements/intraday_signals.py`
  - Applies trained models to current market data
  - Generates ML-enhanced signal predictions
  - Adapts to current market regime

### 6. Signal Generation
- **Signal Processor**: `src/signal_generation/signal_processor.py`
  - Combines statistical signals with ML enhancements
  - Applies confirmation filters and timing optimization
  - Creates actionable entry and exit signals

- **Position Sizing**: `src/risk_management/position_sizer.py`
  - Determines appropriate position sizes based on volatility
  - Applies account-level risk constraints
  - Adapts sizing to current market conditions

### 7. Trading Execution
- **Paper Trading**: `src/paper_trading/paper_trader.py`
  - Simulates order execution with realistic assumptions
  - Tracks virtual positions and P&L
  
- **Live Trading**: `src/live_trading/live_trader.py`
  - Executes orders in real-time through broker API
  - Manages real positions and monitors execution quality
  - Implements safety controls and emergency procedures

### 8. Performance Monitoring
- **Performance Metrics**: `src/reporting/metrics.py`
  - Calculates key performance indicators
  - Tracks strategy performance against benchmarks
  
- **Monitoring Dashboard**: `src/monitoring/dashboard.py`
  - Provides real-time visualization of system performance
  - Alerts on significant deviations or system issues

## Key Integration Points

### Data Pipeline Integration
- Data collection components write to a standardized data storage format
- Processing components read from and write to this common data storage
- Feature engineering components access processed data using consistent APIs

### ML Component Integration
- Models are stored in a central model registry with versioning
- Inference components load models from this registry based on current needs
- Training results are logged for comparison and evaluation

### Trading System Integration
- Signal generation outputs uniform signal format consumed by trading components
- Position tracking is centralized to maintain consistency across components
- Risk limits are enforced at multiple levels for redundant protection

## Data Flow Considerations

1. **Data Consistency**: All components work with consistently formatted data structures
2. **Incremental Processing**: System supports incremental updates to avoid reprocessing full datasets
3. **Caching Strategy**: Frequent calculations are cached to improve performance
4. **Fault Tolerance**: Data flow includes retry mechanisms and fallback options
5. **Monitoring**: Each step in the data flow is monitored for latency and errors 