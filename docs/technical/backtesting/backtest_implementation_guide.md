# Systematic Backtesting Implementation Guide

This document provides practical guidance on implementing the systematic backtesting framework using our local futures data.

## Getting Started

### 1. Data Exploration & Inventory

**Objective**: Create a comprehensive inventory of our historical data assets

```python
# Example script to inventory the data files
import os
import pandas as pd
from pathlib import Path

data_dir = Path("data")
futures_data = []

# Recursively find all data files
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(('.csv', '.parquet', '.feather')):
            file_path = Path(root) / file
            futures_data.append({
                'path': str(file_path),
                'filename': file,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': pd.Timestamp.fromtimestamp(file_path.stat().st_mtime)
            })

# Create inventory DataFrame
inventory_df = pd.DataFrame(futures_data)
print(f"Found {len(inventory_df)} data files")
inventory_df.to_csv("data_inventory.csv", index=False)
```

**Key Tasks**:
- Generate data inventory reports
- Identify data format consistency/inconsistencies
- Validate data completeness across the 15-year period
- Document the ticker universe and characteristics

### 2. Data Processing Pipeline

**Objective**: Create a standardized pipeline to clean and prepare the data for analysis

```python
# Example data processing function
def process_futures_data(file_path, output_dir=None):
    """
    Process a futures data file into a standardized format.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the futures data file
    output_dir : str or Path, optional
        Directory to save processed data. If None, returns the DataFrame
    
    Returns:
    --------
    pd.DataFrame or None
        Processed DataFrame if output_dir is None, otherwise None
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Standardize column names
    column_map = {
        'Date': 'date',
        'Time': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        # Add other mappings as needed
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    # Ensure datetime format
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    
    # Set datetime as index
    if 'datetime' in df.columns:
        df = df.set_index('datetime').sort_index()
    
    # Handle missing values
    df = df.interpolate(method='time').ffill().bfill()
    
    # Calculate additional fields
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = df['high'] - df['low']
        
    # Remove outliers (optional)
    # df = remove_outliers(df)
    
    # Save or return
    if output_dir:
        output_path = Path(output_dir) / Path(file_path).with_suffix('.parquet').name
        df.to_parquet(output_path)
        return None
    else:
        return df
```

**Key Tasks**:
- Build standardized data loaders for each data format
- Implement data cleaning and normalization functions
- Create feature engineering pipeline
- Set up parallel processing for large datasets
- Implement data validation checks
- Create a data versioning strategy

### 3. Cointegration Testing Framework

**Objective**: Implement a scalable framework for testing cointegration across all ticker pairs

```python
# Example cointegration testing function
from statsmodels.tsa.stattools import coint
import itertools
import numpy as np

def batch_cointegration_test(ticker_data, lookback_days=252, p_value_threshold=0.05):
    """
    Test cointegration for all possible pairs in the ticker_data dictionary.
    
    Parameters:
    -----------
    ticker_data : dict
        Dictionary with ticker symbols as keys and DataFrames as values
    lookback_days : int
        Number of days to use for cointegration test
    p_value_threshold : float
        P-value threshold for cointegration significance
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cointegration test results
    """
    results = []
    
    # Get all possible ticker pairs
    ticker_pairs = list(itertools.combinations(ticker_data.keys(), 2))
    
    for ticker1, ticker2 in ticker_pairs:
        # Get closing prices
        df1 = ticker_data[ticker1]
        df2 = ticker_data[ticker2]
        
        # Align the two series
        df_merged = pd.merge(
            df1['close'], 
            df2['close'], 
            left_index=True, 
            right_index=True,
            suffixes=('_1', '_2')
        ).tail(lookback_days)
        
        if len(df_merged) < lookback_days * 0.9:  # Require at least 90% of data
            continue
            
        # Run cointegration test
        price1 = df_merged['close_1'].values
        price2 = df_merged['close_2'].values
        
        # Skip if either series is constant
        if np.std(price1) == 0 or np.std(price2) == 0:
            continue
            
        score, pvalue, _ = coint(price1, price2)
        
        # Calculate hedge ratio
        model = sm.OLS(price1, sm.add_constant(price2)).fit()
        hedge_ratio = model.params[1]
        
        # Calculate half-life of mean reversion
        spread = price1 - hedge_ratio * price2
        half_life = calculate_half_life(spread)
        
        results.append({
            'ticker1': ticker1,
            'ticker2': ticker2,
            'score': score,
            'pvalue': pvalue,
            'hedge_ratio': hedge_ratio,
            'half_life': half_life,
            'is_cointegrated': pvalue < p_value_threshold,
            'test_date': df_merged.index[-1]
        })
    
    return pd.DataFrame(results)

def calculate_half_life(spread):
    """Calculate half-life of mean reversion."""
    lagged_spread = np.roll(spread, 1)
    lagged_spread[0] = lagged_spread[1]
    
    delta_spread = spread - lagged_spread
    beta = np.polyfit(lagged_spread, delta_spread, 1)[0]
    
    if beta >= 0:  # Not mean-reverting
        return np.nan
        
    half_life = -np.log(2) / beta
    return half_life
```

**Key Tasks**:
- Implement different cointegration test methods
- Create an efficient process for testing thousands of pairs
- Develop a database for storing and querying cointegration results
- Build a pair selection algorithm based on cointegration metrics
- Create a dashboard for visualizing pair relationships
- Implement rolling window testing

### 4. Backtest Engine Configuration

**Objective**: Configure the backtest engine for systematic evaluation of pairs trading strategies

```python
# Example configuration class for backtesting
class BacktestConfig:
    """Configuration class for backtest engine."""
    
    def __init__(self,
                start_date=None,
                end_date=None,
                pairs=None,
                capital=1000000,
                commission_rate=0.0001,
                slippage=0.0001,
                z_entry=2.0,
                z_exit=0.5,
                stop_loss_z=4.0,
                max_holding_days=10,
                max_pairs=20,
                max_pair_allocation=0.1):
        """
        Initialize backtest configuration.
        
        Parameters:
        -----------
        start_date : str or datetime
            Backtest start date
        end_date : str or datetime
            Backtest end date
        pairs : list
            List of ticker pairs to trade
        capital : float
            Initial capital
        commission_rate : float
            Commission rate as fraction of trade value
        slippage : float
            Slippage as fraction of price
        z_entry : float
            Z-score threshold for entry
        z_exit : float
            Z-score threshold for exit
        stop_loss_z : float
            Z-score threshold for stop-loss
        max_holding_days : int
            Maximum holding period in days
        max_pairs : int
            Maximum number of pairs to trade simultaneously
        max_pair_allocation : float
            Maximum allocation per pair
        """
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.pairs = pairs or []
        self.capital = capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss_z = stop_loss_z
        self.max_holding_days = max_holding_days
        self.max_pairs = max_pairs
        self.max_pair_allocation = max_pair_allocation
        
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'pairs': self.pairs,
            'capital': self.capital,
            'commission_rate': self.commission_rate,
            'slippage': self.slippage,
            'z_entry': self.z_entry,
            'z_exit': self.z_exit,
            'stop_loss_z': self.stop_loss_z,
            'max_holding_days': self.max_holding_days,
            'max_pairs': self.max_pairs,
            'max_pair_allocation': self.max_pair_allocation
        }
        
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
        
    def save(self, file_path):
        """Save config to file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, default=str)
            
    @classmethod
    def load(cls, file_path):
        """Load config from file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            return cls.from_dict(config_dict)
```

**Key Tasks**:
- Extend the existing backtest engine for multi-pair testing
- Implement realistic transaction cost models
- Create scenario analysis capabilities
- Develop walk-forward testing framework
- Set up parameter optimization grid
- Implement benchmark strategies for comparison

### 5. ML Model Integration

**Objective**: Enhance the statistical strategy with machine learning models

```python
# Example feature engineering for ML models
def create_features(spread_data, window_sizes=[5, 10, 20, 50]):
    """
    Create features for ML model from spread data.
    
    Parameters:
    -----------
    spread_data : pd.Series
        Time series of spread values
    window_sizes : list
        List of window sizes for rolling statistics
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    features = pd.DataFrame(index=spread_data.index)
    
    # Basic spread features
    features['spread'] = spread_data
    features['spread_lag1'] = spread_data.shift(1)
    features['spread_lag2'] = spread_data.shift(2)
    features['spread_lag3'] = spread_data.shift(3)
    
    # Returns
    features['spread_return1'] = features['spread'] / features['spread_lag1'] - 1
    
    # Generate rolling window features
    for window in window_sizes:
        features[f'spread_mean_{window}'] = spread_data.rolling(window).mean()
        features[f'spread_std_{window}'] = spread_data.rolling(window).std()
        features[f'spread_zscore_{window}'] = (
            spread_data - features[f'spread_mean_{window}']
        ) / features[f'spread_std_{window}']
        
        # Momentum indicators
        features[f'spread_roc_{window}'] = (
            spread_data / spread_data.shift(window) - 1
        )
        
        # Volatility indicators
        features[f'spread_vol_ratio_{window}'] = (
            features[f'spread_std_{window}'] / 
            features[f'spread_std_{window}'].shift(window)
        )
    
    # Technical indicators
    features['spread_rsi_14'] = calculate_rsi(spread_data, 14)
    
    # Target variable (example: direction of spread change)
    features['target'] = np.sign(spread_data.diff().shift(-1))
    
    return features.dropna()

# Example ML pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_spread_direction_model(features_df, test_size=0.3):
    """
    Train model to predict spread direction.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features and target
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        (Trained model, scaler, feature list, performance metrics)
    """
    # Prepare data
    features = features_df.drop(columns=['target'])
    target = features_df['target']
    
    # Train/test split preserving time order
    split_idx = int(len(features) * (1 - test_size))
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, X_train.columns.tolist(), metrics, feature_importance
```

**Key Tasks**:
- Develop feature engineering pipeline for pairs data
- Create model training and evaluation framework
- Implement model selection and hyperparameter tuning
- Build ensemble methods combining multiple models
- Create model interpretation tools
- Integrate ML predictions with the trading strategy

### 6. Results Analysis & Visualization

**Objective**: Create comprehensive tools for analyzing backtest results

```python
# Example performance metrics calculation
def calculate_performance_metrics(returns, benchmark_returns=None):
    """
    Calculate trading strategy performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns of the strategy
    benchmark_returns : pd.Series, optional
        Daily returns of the benchmark
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Annualization factor (assuming daily returns)
    ann_factor = 252
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_volatility = returns.std() * np.sqrt(ann_factor)
    sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns / rolling_max) - 1
    max_drawdown = drawdowns.min()
    
    # Win/loss metrics
    winning_days = returns > 0
    win_rate = winning_days.sum() / len(returns)
    avg_win = returns[winning_days].mean() if winning_days.sum() > 0 else 0
    avg_loss = returns[~winning_days].mean() if (~winning_days).sum() > 0 else 0
    
    # Risk metrics
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(ann_factor)
    sortino_ratio = ann_return / downside_deviation if downside_deviation > 0 else 0
    
    # Benchmark comparison
    if benchmark_returns is not None:
        # Align benchmark with strategy returns
        benchmark_returns = benchmark_returns.reindex(returns.index)
        
        # Calculate beta and alpha
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Jensen's alpha (annualized)
        rf_rate = 0.0  # Assuming 0% risk-free rate for simplicity
        benchmark_return = (1 + benchmark_returns).prod() - 1
        ann_benchmark_return = (1 + benchmark_return) ** (ann_factor / len(benchmark_returns)) - 1
        alpha = ann_return - (rf_rate + beta * (ann_benchmark_return - rf_rate))
        
        # Information ratio
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(ann_factor)
        information_ratio = (ann_return - ann_benchmark_return) / tracking_error if tracking_error > 0 else 0
    else:
        beta = None
        alpha = None
        information_ratio = None
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': -avg_win * win_rate / (avg_loss * (1 - win_rate)) if avg_loss < 0 else float('inf'),
        'beta': beta,
        'alpha': alpha,
        'information_ratio': information_ratio
    }
```

**Key Tasks**:
- Implement comprehensive performance metrics
- Create interactive visualization dashboards
- Develop trade analysis tools
- Build comparison frameworks for strategy variants
- Generate automated performance reports
- Create market regime analysis tools

## Implementation Workflow

1. **Initial setup** (Week 1)
   - Create dedicated workspace for backtesting
   - Set up version control
   - Create data directory structure
   - Document data formats and sources

2. **Data pipeline** (Weeks 1-2)
   - Build data loading and cleaning modules
   - Create standardized data format
   - Implement data validation
   - Process initial batch of futures data

3. **Pair selection framework** (Weeks 3-4)
   - Implement cointegration testing at scale
   - Create pair database
   - Develop pair selection criteria
   - Build visualization for cointegrated pairs

4. **Basic strategy backtesting** (Weeks 5-6)
   - Configure backtest engine
   - Implement initial strategy logic
   - Run backtests on selected pairs
   - Analyze initial performance

5. **Strategy optimization** (Weeks 7-9)
   - Grid search parameter optimization
   - Implement walk-forward testing
   - Refine strategy rules based on results
   - Test various entry/exit criteria

6. **ML enhancement** (Weeks 10-13)
   - Feature engineering pipeline
   - Model training and evaluation
   - Model integration with strategy
   - Performance comparison

7. **Risk management & portfolio construction** (Weeks 14-16)
   - Implement risk management rules
   - Develop portfolio construction logic
   - Test correlation constraints
   - Optimize capital allocation

8. **Final analysis & reporting** (Weeks 17-18)
   - Comprehensive performance evaluation
   - Stress testing under various scenarios
   - Documentation of findings
   - Generation of final reports

## Development Standards

- **Code organization**: Modular architecture with clear separation of concerns
- **Documentation**: Docstrings for all functions, README files for each module
- **Testing**: Unit tests for critical components, integration tests for workflows
- **Version control**: Git-based workflow with meaningful commit messages
- **Configuration management**: External configuration files for all parameters
- **Logging**: Comprehensive logging throughout the codebase
- **Performance**: Optimization for large-scale data processing
- **Reproducibility**: Seed all random processes, version all data and code 