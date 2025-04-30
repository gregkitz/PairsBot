# Strategy Variants

This document describes the various strategy variants available in the Intraday Statistical Arbitrage System, including their features, configurations, and use cases.

## Overview

The system supports multiple strategy variants that extend the base pairs trading strategy:

1. **Base Pairs Trading Strategy**: The standard statistical arbitrage approach using z-scores of spread for signal generation.
2. **Time Series Strategy**: Uses time-series models (ARIMA, GARCH, VAR) to forecast spreads and generate signals.
3. **Machine Learning Signal Strategy**: Uses machine learning models to predict spread direction and enhance signals.

## Base Pairs Trading Strategy

The base strategy uses traditional statistical methods to trade mean-reverting pairs.

### Key Features

- Z-score based entry and exit signals
- Standard position sizing based on volatility
- Simple risk management with fixed thresholds

### Configuration

```json
{
  "id": "basic_pairs",
  "name": "Basic Pairs Trading Strategy",
  "description": "Standard statistical arbitrage strategy for cointegrated pairs",
  "type": "pairs",
  "pairs": [
    {"leg1": "ES", "leg2": "NQ", "ratio": 1.0}
  ],
  "entry_threshold": 2.0,
  "exit_threshold": 0.5,
  "stop_loss_threshold": 3.0,
  "max_holding_period_minutes": 180,
  "entry_time_filter": {
    "start": "09:30:00",
    "end": "15:30:00"
  },
  "position_sizing": {
    "method": "volatility",
    "max_allocation_pct": 15.0,
    "max_risk_pct": 1.0,
    "volatility_lookback_days": [10, 20, 30]
  }
}
```

## Time Series Strategy

The time series strategy extends the base strategy by incorporating time-series models to forecast spreads.

### Key Features

- Multiple time-series model options (ARIMA, GARCH, VAR, combined)
- Dynamic spread forecasting with confidence intervals
- Enhanced signals using forecast direction and magnitude
- Model retraining based on configurable frequency

### Configuration

```json
{
  "id": "time_series_pairs",
  "name": "Time Series Pairs Trading Strategy",
  "description": "Advanced pairs trading strategy using time-series models for spread prediction",
  "type": "time_series",
  "pairs": [
    {"leg1": "ES", "leg2": "NQ", "ratio": 1.0}
  ],
  "entry_threshold": 2.0,
  "exit_threshold": 0.5,
  "stop_loss_threshold": 3.0,
  "max_holding_period_minutes": 180,
  "entry_time_filter": {
    "start": "09:30:00",
    "end": "15:30:00"
  },
  "position_sizing": {
    "method": "volatility",
    "max_allocation_pct": 15.0,
    "max_risk_pct": 1.0,
    "volatility_lookback_days": [10, 20, 30]
  },
  "time_series": {
    "model_type": "combined",
    "forecast_horizon": 12,
    "min_history_bars": 100,
    "confidence_level": 0.95,
    "arima_order": [2, 1, 2],
    "garch_order": [1, 1],
    "var_lags": 5,
    "model_update_frequency": 60
  }
}
```

### Model Types

1. **ARIMA** (Auto-Regressive Integrated Moving Average):
   - Good for modeling time series with trend and seasonal components
   - Parameters: `arima_order` (p, d, q) where p is the AR order, d is the differencing order, and q is the MA order

2. **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity):
   - Specializes in modeling volatility clusters
   - Parameters: `garch_order` (p, q) where p is the GARCH order and q is the ARCH order

3. **VAR** (Vector Auto-Regression):
   - Models multiple time series simultaneously, capturing relationships between them
   - Parameters: `var_lags` (number of lags to include)

4. **Combined**:
   - Uses ARIMA for mean forecasting and GARCH for volatility forecasting
   - Provides more robust predictions with confidence intervals

## Machine Learning Signal Strategy

The ML signal strategy uses machine learning models to predict spread direction and enhance trading signals.

### Key Features

- Multiple ML model options (Random Forest, GBM, SVM, Neural Network, Logistic Regression)
- Feature engineering from spread, z-score, and price data
- Automated training and periodic model updates
- Signal enhancement based on prediction confidence
- Feature importance analysis

### Configuration

```json
{
  "id": "ml_signals_pairs",
  "name": "ML Signal Generation Pairs Trading Strategy",
  "description": "Advanced pairs trading strategy using machine learning for signal generation",
  "type": "ml_signals",
  "pairs": [
    {"leg1": "ES", "leg2": "NQ", "ratio": 1.0}
  ],
  "entry_threshold": 2.0,
  "exit_threshold": 0.5,
  "stop_loss_threshold": 3.0,
  "max_holding_period_minutes": 180,
  "entry_time_filter": {
    "start": "09:30:00",
    "end": "15:30:00"
  },
  "position_sizing": {
    "method": "volatility",
    "max_allocation_pct": 15.0,
    "max_risk_pct": 1.0,
    "volatility_lookback_days": [10, 20, 30]
  },
  "ml_signals": {
    "model_type": "random_forest",
    "prediction_horizon": 5,
    "min_history_bars": 1000,
    "training_frequency": 7,
    "threshold": 0.65,
    "feature_window": 20,
    "random_forest_params": {
      "n_estimators": 200,
      "max_depth": 15,
      "min_samples_split": 5,
      "random_state": 42
    }
  }
}
```

### Model Types

1. **Random Forest**:
   - Ensemble of decision trees with majority voting
   - Good for feature selection and handling non-linear relationships
   - Parameters: `n_estimators`, `max_depth`, `min_samples_split`

2. **Gradient Boosting Machine (GBM)**:
   - Sequential ensemble that corrects errors of previous trees
   - Often achieves high accuracy but can overfit
   - Parameters: `n_estimators`, `learning_rate`, `max_depth`

3. **Support Vector Machine (SVM)**:
   - Finds optimal hyperplane for classification
   - Works well with high-dimensional data
   - Parameters: `C`, `kernel`, `gamma`

4. **Neural Network (NN)**:
   - Multi-layer perceptron for complex pattern recognition
   - Can capture intricate relationships in the data
   - Parameters: `hidden_layer_sizes`, `activation`, `solver`, `alpha`

5. **Logistic Regression**:
   - Simple linear model for binary classification
   - Provides easily interpretable coefficients
   - Parameters: `C`, `max_iter`

### Features Used

The ML model uses a variety of features including:

- Z-score and z-score changes over different windows
- Z-score moving averages and volatility
- Z-score momentum and acceleration
- Technical indicators like RSI and MACD
- Correlation between legs
- Volume imbalance
- Market regime indicators (Bollinger Band width)
- Time-based features (hour of day, day of week)

## Usage Example

Here's how to create and use the different strategy variants:

```python
from src.strategy_variants import create_strategy

# Load configuration
with open('config/strategies/time_series_pairs.json', 'r') as f:
    config = json.load(f)

# Create strategy
strategy = create_strategy('time_series', config)

# Use the strategy
analysis_results = strategy.analyze_data(data)
```

## Performance Comparison

The different strategy variants typically show these characteristics:

- **Base Strategy**: Reliable performance in stable markets with clear mean reversion
- **Time Series Strategy**: Better performance during regime changes and trending markets
- **ML Signal Strategy**: Superior ability to handle complex market conditions and filter false signals

To compare strategies, use the example script:

```bash
python examples/use_strategy_variants.py
```

## Choosing the Right Strategy

Consider these factors when choosing a strategy variant:

1. **Data Availability**: The ML strategy requires more historical data for training.
2. **Computational Resources**: Time series and ML strategies require more processing power.
3. **Market Conditions**: Different strategies excel in different market regimes.
4. **Stability vs. Adaptability**: Base strategy is more stable, while variants are more adaptive.

For best results, consider running multiple strategies in parallel and allocating capital based on recent performance and market conditions. 