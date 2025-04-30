# Complex Function Simplification Plan

This document outlines specific approaches to simplify complex functions identified in our codebase, starting with the most complex functions.

## IntradaySignalEnhancer.enhance_signals

**Current State**: 
- 232 lines
- Complex data alignment and filtering logic
- Multiple signal enhancement steps with conditional execution
- Complex handling of different input types (Series vs DataFrame)
- Metrics tracking mixed with signal logic

**Complexity Factors**:
1. **Data Alignment**: Multiple steps to align different data sources
2. **Type Handling**: Special cases for Series vs DataFrame inputs
3. **Multiple Phases**: ML filtering, entry timing, exit timing, etc.
4. **Metrics Collection**: Signal metrics collected throughout
5. **Complex Output Preparation**: Conversion back to original format

### Simplification Approach

#### 1. Extract Data Preparation

```python
def _prepare_data(self, original_signals, features, prices_df=None, volumes_df=None):
    """
    Prepare and align data for signal enhancement.
    
    Parameters:
    -----------
    original_signals : pd.Series or pd.DataFrame
        Series or DataFrame with original signals
    features : pd.DataFrame
        DataFrame with calculated features
    prices_df : pd.DataFrame, optional
        DataFrame with price data
    volumes_df : pd.DataFrame, optional
        DataFrame with volume data
        
    Returns:
    --------
    dict
        Dictionary with aligned data:
        - common_index: Common index across all data
        - features: Filtered features
        - original_signals: Standardized original signals
        - enhanced_signals: Initial enhanced signals (copy)
        - is_dataframe: Whether input was a DataFrame
        - signal_column: Column name if DataFrame
    """
    # Find common index between signals and features
    common_index = original_signals.index.intersection(features.index)
    
    if len(common_index) == 0:
        logger.warning("No common timestamps between signals and features")
        return None
    
    # Standardize signal format
    is_dataframe = isinstance(original_signals, pd.DataFrame) and original_signals.shape[1] > 1
    signal_column = None
    
    if is_dataframe:
        if 'signal' in original_signals.columns:
            signal_column = 'signal'
            signal_series = original_signals['signal']
        else:
            signal_column = original_signals.columns[0]
            signal_series = original_signals.iloc[:, 0]
        
        original_signals_std = signal_series
    else:
        original_signals_std = original_signals
    
    # Filter data to common index
    features_filtered = features.loc[common_index]
    original_signals_filtered = original_signals_std.loc[common_index]
    
    # Align with prices if provided
    if prices_df is not None:
        common_index = self._align_with_dataframe(common_index, prices_df, features_filtered, original_signals_filtered)
        if common_index is None:
            return None
        features_filtered = features_filtered.loc[common_index]
        original_signals_filtered = original_signals_filtered.loc[common_index]
    
    # Align with volumes if provided
    if volumes_df is not None:
        common_index = self._align_with_dataframe(common_index, volumes_df, features_filtered, original_signals_filtered)
        if common_index is None:
            return None
        features_filtered = features_filtered.loc[common_index]
        original_signals_filtered = original_signals_filtered.loc[common_index]
    
    return {
        'common_index': common_index,
        'features': features_filtered,
        'original_signals': original_signals_filtered,
        'enhanced_signals': original_signals_filtered.copy(),
        'is_dataframe': is_dataframe,
        'signal_column': signal_column
    }

def _align_with_dataframe(self, common_index, df, features, signals):
    """Helper method to align data with an additional DataFrame."""
    if df is None:
        return common_index
        
    common_df_index = common_index.intersection(df.index)
    if len(common_df_index) == 0:
        logger.warning(f"No common timestamps after aligning with DataFrame")
        return None
        
    return common_df_index
```

#### 2. Extract Signal Enhancement Steps

```python
def _apply_signal_filter(self, data, metrics):
    """Apply ML signal filter to enhance signals."""
    features = data['features']
    enhanced_signals = data['enhanced_signals']
    common_index = data['common_index']
    
    if not self.config["enable_ml_filtering"] or self.signal_filter_model is None:
        return enhanced_signals
    
    signal_quality = self.predict_signal_quality(features)
    metrics.loc[common_index, 'signal_quality'] = signal_quality
    
    # Filter out low quality signals
    filter_mask = signal_quality < self.config["prediction_threshold"]
    enhanced_signals[filter_mask] = 0
    
    return enhanced_signals

def _apply_entry_timing(self, data, metrics):
    """Apply ML entry timing to enhance signals."""
    if not self.config["enable_ml_timing"] or self.entry_timing_model is None:
        return data['enhanced_signals']
    
    features = data['features']
    original_signals = data['original_signals']
    enhanced_signals = data['enhanced_signals']
    common_index = data['common_index']
    
    entry_quality = self.predict_optimal_entry(features)
    metrics.loc[common_index, 'entry_quality'] = entry_quality
    
    # Only enter when entry quality is high
    entry_mask = (
        (original_signals != 0) & 
        (enhanced_signals == 0) & 
        (entry_quality >= self.config["prediction_threshold"])
    )
    enhanced_signals[entry_mask] = original_signals[entry_mask]
    
    return enhanced_signals

def _apply_exit_timing(self, data, metrics):
    """Apply ML exit timing to enhance signals."""
    if not self.config["enable_ml_timing"] or self.exit_timing_model is None:
        return data['enhanced_signals']
    
    features = data['features']
    enhanced_signals = data['enhanced_signals']
    common_index = data['common_index']
    
    exit_quality = self.predict_optimal_exit(features)
    metrics.loc[common_index, 'exit_quality'] = exit_quality
    
    # Exit positions when exit quality is high
    exit_mask = (enhanced_signals != 0) & (exit_quality >= self.config["prediction_threshold"])
    enhanced_signals[exit_mask] = 0
    
    return enhanced_signals
```

#### 3. Extract Results Conversion

```python
def _prepare_result(self, data, enhanced_signals_filtered, metrics, original_signals):
    """Prepare final enhanced signals and metrics result."""
    common_index = data['common_index']
    
    # Track modifications
    original_filtered = data['original_signals']
    modifications = sum((enhanced_signals_filtered != original_filtered).astype(int))
    total_signals = len(original_filtered)
    
    modification_rate = modifications / total_signals if total_signals > 0 else 0
    logger.info(f"Modified {modifications} out of {total_signals} signals ({modification_rate:.2%})")
    
    # Add enhanced signals to metrics
    metrics.loc[common_index, 'enhanced_signal'] = enhanced_signals_filtered
    
    # Create final enhanced signals with same format as input
    enhanced_signals = original_signals.copy()
    
    # Update enhanced signals
    if data['is_dataframe']:
        # For DataFrame inputs
        if data['signal_column'] is not None:
            enhanced_signals[data['signal_column']].loc[common_index] = enhanced_signals_filtered
    else:
        # For Series inputs
        enhanced_signals.loc[common_index] = enhanced_signals_filtered
    
    return enhanced_signals, metrics
```

#### 4. Simplified Main Function

```python
def enhance_signals(self, original_signals, features, prices_df=None, volumes_df=None):
    """
    Enhance trading signals using ML models and filters.
    
    Parameters:
    -----------
    original_signals : pd.Series or pd.DataFrame
        Series or DataFrame with original signals
    features : pd.DataFrame
        DataFrame with calculated features
    prices_df : pd.DataFrame, optional
        DataFrame with price data
    volumes_df : pd.DataFrame, optional
        DataFrame with volume data
            
    Returns:
    --------
    pd.Series
        Series with enhanced signals
    pd.DataFrame
        DataFrame with signal quality metrics
    """
    # Initialize metrics
    metrics = pd.DataFrame(index=original_signals.index)
    
    # Save original signal for metrics
    if isinstance(original_signals, pd.DataFrame) and 'signal' in original_signals.columns:
        metrics['original_signal'] = original_signals['signal']
    else:
        metrics['original_signal'] = original_signals
    
    # Prepare and align data
    data = self._prepare_data(original_signals, features, prices_df, volumes_df)
    if data is None:
        return original_signals.copy(), metrics
    
    # Apply signal filters and enhancements
    enhanced_signals = data['enhanced_signals']
    
    # 1. Apply ML signal filter
    enhanced_signals = self._apply_signal_filter(data, metrics)
    
    # 2. Apply ML entry timing
    enhanced_signals = self._apply_entry_timing(data, metrics)
    
    # 3. Apply ML exit timing
    enhanced_signals = self._apply_exit_timing(data, metrics)
    
    # Additional enhancements can be added here
    
    # 4. Prepare final result
    return self._prepare_result(data, enhanced_signals, metrics, original_signals)
```

### Benefits of This Approach

1. **Reduced Complexity**: Main function reduced from 232 lines to ~30 lines
2. **Clear Phases**: Each phase of signal enhancement is a separate function
3. **Improved Testability**: Each extracted function can be unit tested
4. **Better Maintainability**: Adding new enhancement steps is simpler
5. **Enhanced Readability**: Clearer logic flow with descriptive function names

### Implementation Plan

1. Create unit tests for current `enhance_signals` function to ensure behavior is preserved
2. Extract the data preparation function and test
3. Extract the signal filter function and test
4. Extract the entry/exit timing functions and test
5. Implement the simplified main function
6. Run tests to verify behavior is unchanged
7. Add new test cases for edge conditions

### Estimated Effort: 1 developer day

## Additional Complex Functions to Simplify

Following the same approach, we can also simplify the following complex functions:

1. **IntradaySignalEnhancer.apply_intraday_adaptations** (181 lines)
   - Extract time-based adaptation logic
   - Extract volume-based adaptation logic
   - Extract volatility-based adaptation logic
   - Simplify main function to orchestrate these steps

2. **IntradaySignalEnhancer.calculate_features** (146 lines)
   - Extract spread feature calculation
   - Extract time feature calculation
   - Extract instrument-specific feature calculation
   - Extract volume feature calculation

3. **run_backtest in main.py** (241 lines)
   - Extract parameter preparation
   - Extract data loading and validation
   - Extract backtest execution
   - Extract results processing
   - Extract reporting logic

Each of these simplifications would follow a similar pattern of:
1. Extract related code into focused helper functions
2. Simplify the main function to orchestrate these helpers
3. Test to ensure behavior is preserved

## Conclusion

By systematically applying these simplification techniques to our most complex functions, we can significantly improve code maintainability, readability, and testability without changing behavior. This approach aligns with best practices for code organization and the Single Responsibility Principle.

Each simplified function should be easier to understand, modify, and debug, which will reduce the time needed for future development and minimize the risk of introducing bugs. 