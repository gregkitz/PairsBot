# Advanced Signal Generation Features

This document provides a comprehensive overview of the advanced signal generation features implemented in the `SignalGenerator` class. These enhancements improve trading performance by adapting to market conditions, filtering out false signals, and implementing more sophisticated risk management techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Regime-Based Threshold Adaptation](#regime-based-threshold-adaptation)
3. [Confirmation Filters](#confirmation-filters)
   - [Volume Imbalance Filter](#volume-imbalance-filter)
   - [Momentum Filter](#momentum-filter)
   - [Mean-Reversion Strength Filter](#mean-reversion-strength-filter)
   - [Multi-Timeframe Filter](#multi-timeframe-filter)
   - [Combined Filters](#combined-filters)
4. [Dynamic Stop-Loss Mechanisms](#dynamic-stop-loss-mechanisms)
   - [Fixed Stop-Loss](#fixed-stop-loss)
   - [Volatility-Adjusted Stop-Loss](#volatility-adjusted-stop-loss)
   - [Dynamic Stop-Loss](#dynamic-stop-loss)
5. [Dynamic Take-Profit Mechanisms](#dynamic-take-profit-mechanisms)
   - [Fixed Take-Profit](#fixed-take-profit)
   - [Volatility-Adjusted Take-Profit](#volatility-adjusted-take-profit)
   - [Dynamic Take-Profit](#dynamic-take-profit)
6. [Advanced Trailing Stop Logic](#advanced-trailing-stop-logic)
7. [Dynamic Thresholds](#dynamic-thresholds)
8. [Signal Generator Configuration](#signal-generator-configuration)
9. [Integration with Market Regime Detection](#integration-with-market-regime-detection)
10. [Usage Examples](#usage-examples)

## Introduction

The enhanced `SignalGenerator` class implements advanced features for pairs trading that go beyond simple z-score thresholds. These enhancements aim to:

1. Adapt to changing market conditions through regime-based parameters
2. Reduce false signals with confirmation filters
3. Implement more sophisticated risk management with dynamic stop-loss and take-profit mechanisms
4. Improve exit timing with enhanced trailing stop logic

## Regime-Based Threshold Adaptation

Market conditions are not static, and trading parameters should adapt accordingly. The `SignalGenerator` now supports regime-based threshold adaptation, which automatically adjusts entry/exit thresholds based on the current market regime.

### How It Works

1. A `MarketRegimeClassifier` is provided to the `SignalGenerator`
2. The classifier detects the current market regime based on features such as volatility, trend strength, and correlation
3. The `SignalGenerator` uses a mapping of regimes to threshold parameters to adapt its behavior

### Example Configuration

```python
# Define regime to threshold mapping
regime_mapping = {
    0: {'entry': 2.5, 'exit': 0.5, 'stop_loss': 4.0},  # High volatility regime
    1: {'entry': 2.0, 'exit': 0.0, 'stop_loss': 3.5},  # Normal regime
    2: {'entry': 1.5, 'exit': 0.0, 'stop_loss': 3.0}   # Low volatility regime
}

# Create SignalGenerator with regime adaptation
signal_gen = SignalGenerator(
    entry_threshold=2.0,
    exit_threshold=0.0,
    regime_classifier=market_regime_classifier,
    regime_threshold_mapping=regime_mapping
)
```

## Confirmation Filters

Confirmation filters help reduce false signals by requiring additional evidence before generating entry signals. The `SignalGenerator` now supports several types of confirmation filters.

### Volume Imbalance Filter

The volume imbalance filter confirms entry signals based on relative volume between the assets in a pair. Volume imbalance can indicate market enthusiasm or skepticism about an asset's price level.

#### Implementation

```python
def _apply_volume_imbalance_filter(self, signals, additional_data):
    # Calculate volume ratio between assets
    vol_ratio = volume_data.iloc[:, 0] / volume_data.iloc[:, 1]
    
    # Calculate z-score of volume ratio
    vol_zscore = (vol_ratio - vol_ratio_ma) / vol_ratio_std
    
    # For long entries: First asset should have relatively lower volume (undervalued)
    mask_long = vol_zscore < -threshold
    signals.loc[~mask_long, 'entry_long'] = 0
```

### Momentum Filter

The momentum filter confirms that the z-score is moving in the expected direction with sufficient momentum before generating entry signals.

#### Implementation

```python
def _apply_momentum_filter(self, signals, zscore):
    # Calculate momentum (rate of change)
    momentum = zscore.diff(momentum_window)
    
    # For long entries: Z-score should be decreasing (getting more negative)
    mask_long = momentum < -threshold
    signals.loc[~mask_long, 'entry_long'] = 0
```

### Mean-Reversion Strength Filter

This filter confirms that the spread is exhibiting strong mean-reverting behavior before generating entry signals, by calculating metrics like half-life or Hurst exponent.

#### Implementation

```python
def _apply_mean_reversion_filter(self, signals, spread):
    # Calculate half-life
    half_life_series = pd.Series(half_lives, index=spread.index[window:])
    
    # Only enter if half-life is below threshold (strong mean reversion)
    mask = half_life_series < half_life_threshold
    signals.loc[~mask, 'entry_long'] = 0
    signals.loc[~mask, 'entry_short'] = 0
```

### Multi-Timeframe Filter

The multi-timeframe filter confirms that signals are consistent across multiple timeframes, reducing the likelihood of false signals from short-term noise.

#### Implementation

```python
def _apply_multi_timeframe_filter(self, signals, additional_data):
    # Calculate agreement percentage across timeframes
    composite['long_agreement'] = composite['long_agreement'] / len(tf_signals)
    
    # Filter signals based on agreement threshold
    signals.loc[composite['long_agreement'] < agreement_threshold, 'entry_long'] = 0
```

### Combined Filters

Multiple confirmation filters can be combined for more robust signal generation. The `SignalGenerator` supports a combined filter approach that applies multiple filters sequentially.

## Dynamic Stop-Loss Mechanisms

The enhanced `SignalGenerator` implements several types of dynamic stop-loss mechanisms that adapt to market conditions.

### Fixed Stop-Loss

The basic stop-loss mechanism uses fixed z-score thresholds.

### Volatility-Adjusted Stop-Loss

This stop-loss mechanism adjusts the threshold based on current market volatility. During high volatility periods, the stop-loss is placed wider to avoid being stopped out by normal market noise.

#### Implementation

```python
def _apply_stop_loss(self, signals, zscore, additional_data=None):
    if self.stop_type == StopType.VOLATILITY_ADJUSTED:
        # Adjust stop-loss based on volatility
        stop_level = self.stop_loss_threshold * vol / vol.mean()
        
        # Apply stops to signals
        if zscore.iloc[i] < -stop_level.iloc[i]:
            signals_with_stops.loc[zscore.index[i], 'exit_long'] = 1
```

### Dynamic Stop-Loss

The dynamic stop-loss adjusts based on the recent distribution of the spread, using percentiles to set appropriate stop levels.

## Dynamic Take-Profit Mechanisms

Similar to stop-loss mechanisms, the `SignalGenerator` implements several types of take-profit mechanisms.

### Fixed Take-Profit

The basic take-profit mechanism uses fixed z-score thresholds.

### Volatility-Adjusted Take-Profit

This take-profit mechanism adjusts the threshold based on current market volatility. During high volatility periods, the take-profit is placed wider to capture more potential profit.

### Dynamic Take-Profit

The dynamic take-profit adjusts based on the recent distribution of the spread, using percentiles to set appropriate take-profit levels.

## Advanced Trailing Stop Logic

The enhanced trailing stop logic improves upon the basic trailing stop by adding:

1. Activation level: The trailing stop is only activated once the position moves into profit by a certain amount
2. Step size: The trailing stop moves in discrete steps to avoid micromanagement
3. Favorable movement tracking: The trailing stop only moves in favorable directions

### Implementation

```python
def _add_advanced_trailing_stop(self, signals, zscore, additional_data=None):
    # Get parameters for trailing stop
    activation = self.trailing_stop_params.get('activation', 1.0)
    
    # Update trailing stop level
    if position == 1:  # Long position
        # Initialize trailing stop if zscore crosses activation level
        if trailing_level is None and zscore.iloc[i] > -activation:
            trailing_level = -activation
        
        # Update trailing level if spread moves favorably
        elif trailing_level is not None and zscore.iloc[i] > trailing_level + step:
            trailing_level = min(0, trailing_level + step)
```

## Dynamic Thresholds

Instead of using fixed z-score thresholds, the `SignalGenerator` can calculate dynamic thresholds based on the historical distribution of the spread.

### Implementation

```python
def _calculate_dynamic_thresholds(self, spread, window=60, percentile=0.05):
    # Calculate percentiles
    upper = hist_data.quantile(1 - percentile)
    lower = hist_data.quantile(percentile)
    
    # Store as multiple of standard deviation
    thresholds.loc[spread.index[i], 'entry_upper'] = upper / std
    thresholds.loc[spread.index[i], 'entry_lower'] = lower / std
```

## Signal Generator Configuration

The enhanced `SignalGenerator` supports a rich set of configuration options:

```python
SignalGenerator(
    entry_threshold=2.0,
    exit_threshold=0.0,
    stop_loss_threshold=3.5,
    take_profit_threshold=1.0,
    time_stop=30,
    signal_type=SignalType.ZSCORE,
    signal_smoothing=0,
    use_trailing_stop=True,
    regime_classifier=market_regime_classifier,
    confirmation_type=ConfirmationType.COMBINED,
    confirmation_params={
        'filters': ['volume_imbalance', 'momentum'],
        'volume_window': 5,
        'volume_threshold': 1.0,
        'momentum_window': 3,
        'momentum_threshold': 0.1
    },
    stop_type=StopType.VOLATILITY_ADJUSTED,
    take_profit_type=TakeProfitType.DYNAMIC,
    trailing_stop_params={
        'activation': 1.0,
        'step': 0.5
    },
    dynamic_thresholds=True,
    regime_threshold_mapping={
        0: {'entry': 2.5, 'exit': 0.5},
        1: {'entry': 2.0, 'exit': 0.0}
    }
)
```

## Integration with Market Regime Detection

The `SignalGenerator` integrates with the `MarketRegimeClassifier` to adapt its behavior based on market conditions. The integration works as follows:

1. The `MarketRegimeClassifier` is provided to the `SignalGenerator`
2. During signal generation, the classifier is used to detect the current regime
3. The `SignalGenerator` adapts its parameters based on the detected regime

### Example

```python
# Detect market regime
if self.regime_classifier is not None and additional_data is not None:
    regime_features = additional_data.get('regime_features', None)
    if regime_features is not None:
        self.current_regime = self.regime_classifier.predict(regime_features)
        signals['regime'] = self.current_regime
        
        # Adapt thresholds based on regime
        self._adapt_thresholds_to_regime()
```

## Usage Examples

### Basic Usage with Confirmation Filters

```python
# Create SignalGenerator with confirmation filters
signal_gen = SignalGenerator(
    entry_threshold=2.0,
    exit_threshold=0.0,
    confirmation_type=ConfirmationType.VOLUME_IMBALANCE,
    confirmation_params={'volume_window': 5, 'volume_threshold': 1.0}
)

# Generate signals
signals = signal_gen.generate_signals(
    zscore=zscore_series,
    additional_data={'volume_data': volume_df}
)
```

### Advanced Risk Management

```python
# Create SignalGenerator with advanced risk management
signal_gen = SignalGenerator(
    entry_threshold=2.0,
    exit_threshold=0.0,
    stop_type=StopType.DYNAMIC,
    take_profit_type=TakeProfitType.VOLATILITY_ADJUSTED,
    use_trailing_stop=True,
    trailing_stop_params={'activation': 1.0, 'step': 0.5}
)

# Generate signals
signals = signal_gen.generate_signals(
    zscore=zscore_series,
    additional_data={
        'spread': spread_series,
        'volatility': volatility_series
    }
)
```

### Regime-Based Adaptation

```python
# Create SignalGenerator with regime adaptation
signal_gen = SignalGenerator(
    entry_threshold=2.0,
    exit_threshold=0.0,
    regime_classifier=market_regime_classifier,
    regime_threshold_mapping={
        0: {'entry': 2.5, 'exit': 0.5},
        1: {'entry': 2.0, 'exit': 0.0},
        2: {'entry': 1.5, 'exit': 0.0}
    }
)

# Generate signals
signals = signal_gen.generate_signals(
    zscore=zscore_series,
    additional_data={'regime_features': regime_features_df}
)
``` 