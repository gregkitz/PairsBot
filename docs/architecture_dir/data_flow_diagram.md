# Data Flow Diagram

This document provides a visual representation of the data flow through the trading system, making it easier to understand how different components interact.

## Overall System Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Sources                                        │
│                                                                                 │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                   │
│  │  Historical    │   │  Real-time     │   │  Alternative   │                   │
│  │  Market Data   │   │  Market Data   │   │  Data Sources  │                   │
│  └───────┬────────┘   └────────┬───────┘   └────────┬───────┘                   │
└──────────┼─────────────────────┼────────────────────┼───────────────────────────┘
           │                     │                    │
           ▼                     ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Data Collection                                      │
│                                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                    │
│  │  Historical    │   │  Real-time     │   │  Alternative   │                    │
│  │  Data Pipeline │   │  Data Pipeline │   │  Data Pipeline │                    │
│  └───────┬────────┘   └────────┬───────┘   └────────┬───────┘                    │
└──────────┼─────────────────────┼────────────────────┼────────────────────────────┘
           │                     │                    │
           └─────────────────────┼────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Processing                                     │
│                                                                                 │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                   │
│  │  Data Cleaning │──►│   Normalizer   │──►│  Pair Builder  │                   │
│  └────────────────┘   └────────────────┘   └───────┬────────┘                   │
└───────────────────────────────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Feature Engineering                                   │
│                                                                                 │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                   │
│  │    Feature     │──►│    Feature     │──►│   Feature      │                   │
│  │   Generator    │   │    Selector    │   │    Storage     │                   │
│  └───────┬────────┘   └────────────────┘   └───────┬────────┘                   │
└──────────┼────────────────────────────────────────┼─────────────────────────────┘
           │                                        │
           │                                        │
           └───────────────────┐                    │
                              ▼                     │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Model Training & Inference                             │
│                                                                                 │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                   │
│  │     Model      │   │    Regime      │   │   Inference    │                   │
│  │    Training    │──►│   Detection    │──►│     Engine     │                   │
│  └────────────────┘   └────────────────┘   └───────┬────────┘                   │
└────────────────────────────────────────────────────┼─────────────────────────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Signal Generation                                     │
│                                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                    │
│  │     Signal     │   │   Confirmation │   │    Position    │                    │
│  │    Processor   │──►│     Filters    │──►│     Sizing     │                    │
│  └───────┬────────┘   └────────────────┘   └───────┬────────┘                    │
└──────────┼─────────────────────────────────────────┼────────────────────────────┘
           │                                         │
           ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Trading Execution                                     │
│                                                                                 │
│  ┌────────────────┐                           ┌────────────────┐                 │
│  │     Paper      │                           │      Live      │                 │
│  │    Trading     │                           │    Trading     │                 │
│  └───────┬────────┘                           └───────┬────────┘                 │
└──────────┼─────────────────────────────────────────────────────────────────────┘
           │                                            │
           └────────────────────┬─────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Performance Monitoring                                  │
│                                                                                 │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                   │
│  │   Performance  │   │   Monitoring   │   │    Alerting    │                   │
│  │     Metrics    │──►│    Dashboard   │──►│     System     │                   │
│  └────────────────┘   └────────────────┘   └────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Data Flow

### Data Processing Pipeline

```
Historical Data ──► Data Downloader ──► Data Cleaner ──► Data Storage
                                                     │
Real-time Data ──► IBConnector ──────────────────────┘
```

### Feature Engineering Pipeline

```
Processed Data ──► Feature Generator ──► Feature Selection ──► Feature Storage
                    │                                         │
                    ▼                                         │
                Feature Set Metadata                          │
                    │                                         │
                    └──────────────────────────────────────────────► Model Training
```

### Signal Generation Pipeline

```
Market Data ──► Z-Score Calculator ──► Signal Processor ──► Signal Validation
                                                        │
ML Models ──► ML-Enhanced Predictions ──────────────────┘
                                                        │
                                                        ▼
                                               Position Sizing ──► Orders
```

### Trading Execution Pipeline

```
Signal ──► Order Generation ──► Order Placement ──► Execution ──► Position Tracking
                                                                    │
                                                                    ▼
                                                              Performance Metrics
```

## Key Data Structures

### Pair Data Structure
```
{
  "pair_id": "GC_SI",
  "ticker1": "GC",
  "ticker2": "SI",
  "hedge_ratio": 0.0123,
  "cointegration_pvalue": 0.0021,
  "half_life": 3.5,
  "correlation": 0.86,
  "spread_volatility": 0.0045,
  "is_active": true
}
```

### Signal Data Structure
```
{
  "pair_id": "GC_SI",
  "timestamp": "2023-03-21T14:30:00Z",
  "zscore": 2.34,
  "signal": 1,  # 1 = Long spread, -1 = Short spread, 0 = No position
  "confidence": 0.89,
  "ml_enhanced": true,
  "regime": "normal",
  "position_size": 0.1,
  "target_exit": 0.5,
  "stop_loss": 3.5
}
```

### Position Data Structure
```
{
  "position_id": "pos_12345",
  "pair_id": "GC_SI",
  "entry_time": "2023-03-21T14:32:15Z",
  "entry_zscore": 2.34,
  "current_zscore": 1.8,
  "leg1_ticker": "GC",
  "leg1_quantity": 1,
  "leg1_entry_price": 1950.5,
  "leg1_current_price": 1955.0,
  "leg2_ticker": "SI",
  "leg2_quantity": -80,
  "leg2_entry_price": 24.1,
  "leg2_current_price": 24.2,
  "pnl_dollars": 210.5,
  "pnl_percent": 0.42,
  "status": "open"
}
```

## Data Integration Points

### Data Collection ↔ Processing
- **Interface**: `DataProcessor` class in `src/data_processor/data_processor.py`
- **Format**: Pandas DataFrames with standardized column names
- **Frequency**: Daily for historical, real-time for live data

### Processing ↔ Feature Engineering
- **Interface**: `FeatureGenerator` class in `src/ml_enhancements/feature_engineering/feature_generator.py`
- **Format**: Pandas DataFrames with feature columns
- **Validation**: Type checking, range validation, missing value handling

### Feature Engineering ↔ Model Training
- **Interface**: `ModelTrainer` class in `src/ml_enhancements/model_retraining.py`
- **Format**: Numpy arrays for X_train, y_train
- **Metadata**: Feature importance, normalization parameters

### Model Inference ↔ Signal Generation
- **Interface**: `SignalEnhancer` class in `src/ml_enhancements/intraday_signals.py`
- **Format**: Dictionary with signal parameters
- **Validation**: Confidence thresholds, regime-specific adjustments

### Signal Generation ↔ Trading Execution
- **Interface**: `OrderManager` class in `src/paper_trading/paper_trader.py`
- **Format**: Order objects with ticker, quantity, price
- **Validation**: Risk limits, liquidity checks, timing constraints

## Data Storage

### Time Series Storage
- **Historical Data**: Parquet files organized by ticker/timeframe
- **Processed Pairs**: HDF5 files with paired time series
- **Location**: `data/historical/`, `data/processed/`

### Model Storage
- **Format**: Pickle files for models, JSON for metadata
- **Versioning**: Timestamp-based versioning
- **Location**: `models/{pair_id}/{timestamp}/`

### Configuration Storage
- **Format**: YAML configuration files
- **Environment Overrides**: Environment-specific settings
- **Location**: `config/`

### Results Storage
- **Backtest Results**: JSON files with performance metrics
- **Live Trading Results**: Database records for positions and trades
- **Location**: `results/backtest/`, Database
``` 