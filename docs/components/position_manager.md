# Position Manager Component

## Overview

The Position Manager is a core component in the paper trading system responsible for managing all aspects of position tracking, entry/exit execution, risk management, and performance monitoring. It provides a clean, focused API for handling trading positions, abstracting the complexities of pair trading position management from the main trading system.

## Key Features

- **Position Tracking**: Maintains detailed records of all open and closed positions
- **Signal Execution**: Converts trading signals into actual position entries/exits
- **Risk Management**: Implements multiple risk control measures (stop-loss, take-profit, max holding time)
- **Performance Monitoring**: Tracks detailed metrics for each position
- **Dynamic Position Sizing**: Adjusts position size based on market volatility
- **Correlation Monitoring**: Detects and responds to correlation breakdowns between pair legs

## Component Architecture

The Position Manager sits between the signal generation components and the paper trader execution layer:

```
Signal Generation → Position Manager → Paper Trader
```

The Position Manager receives signals from the trading strategy components, makes decisions about position entries/exits, and then instructs the Paper Trader to execute the required orders.

## Usage

### Initialization

To use the Position Manager, you need to initialize it with a reference to a paper trader instance:

```python
from src.paper_trading.components.position_manager import PositionManager
from src.paper_trading.paper_trader import PaperTrader

# Create a paper trader instance
paper_trader = PaperTrader(
    initial_capital=100000.0,
    ib_host='127.0.0.1',
    ib_port=7497,
    ib_client_id=1
)

# Create a Position Manager instance
position_manager = PositionManager(
    paper_trader=paper_trader,
    config={
        'default_position_size': 0.1,  # 10% of account per position
        'risk_per_trade': 0.02         # 2% max risk per trade
    }
)
```

### Adding Trading Pairs

Before you can manage positions, you need to add the trading pairs you want to trade:

```python
# Add a trading pair
position_manager.add_pair({
    'pair_id': 'GC_SI',            # Gold-Silver pair
    'leg1': 'GC',                  # Gold symbol
    'leg2': 'SI',                  # Silver symbol
    'hedge_ratio': 1.5,            # Trade 1.5 units of SI for each GC unit
    'leg1_multiplier': 1.0,
    'leg2_multiplier': 1.0,
    'z_entry': 2.0,                # Enter when z-score exceeds 2.0
    'z_exit': 0.5,                 # Exit when z-score returns to 0.5
    'stop_loss_z': 3.0,            # Stop loss at z-score of 3.0
    'max_holding_period': 180,     # Maximum holding period in minutes
    'min_correlation': 0.5,        # Minimum acceptable correlation
    'half_life': 24                # Half-life in hours for mean reversion
})
```

### Executing Trading Signals

To execute trading signals, pass the pair ID and signal values:

```python
# Example signals dataframe with signal values (-1, 0, 1)
signals = pd.Series([1], index=[datetime.now()])  # 1 for long, -1 for short, 0 for no position

# Execute signals for a specific pair
result = position_manager.execute_signals('GC_SI', signals)

# Check execution result
if result['executed'] > 0:
    print(f"Executed {result['executed']} signal(s)")
else:
    print(f"No signals executed, errors: {result['errors']}")
```

### Monitoring Positions

The Position Manager provides several methods for monitoring positions:

```python
# Get all current positions
positions = position_manager.get_positions()

# Get a specific position
position = position_manager.get_position('GC_SI')

# Get position history (closed positions)
history = position_manager.get_position_history()

# Get a summary of all positions
summary = position_manager.get_position_summary()
```

### Risk Management

To perform risk management checks on all positions, use the monitoring system:

```python
# Create a dictionary with market data for each pair
market_data = {
    'GC_SI': current_data_for_gc_si  # pd.DataFrame with price data
}

# Monitor all positions
monitoring_results = position_manager.monitor_all_positions(market_data)

# Check monitoring results
print(f"Checked {monitoring_results['positions_checked']} positions")
print(f"Stop losses triggered: {monitoring_results['stop_losses_triggered']}")
print(f"Take profits triggered: {monitoring_results['take_profits_triggered']}")
```

## Position Structure

Each position maintained by the Position Manager contains the following information:

```python
position = {
    'pair_id': 'GC_SI',                # Trading pair identifier
    'direction': 1,                    # 1 for long, -1 for short
    'entry_time': datetime.now(),      # Entry timestamp
    'entry_spread': 2.5,               # Spread value at entry
    'entry_zscore': 2.0,               # Z-score at entry
    'leg1': {                          # First leg details
        'symbol': 'GC',
        'action': 'BUY',
        'quantity': 1.0,
        'order_id': 'order_123'
    },
    'leg2': {                          # Second leg details
        'symbol': 'SI',
        'action': 'SELL',
        'quantity': 1.5,
        'order_id': 'order_456'
    },
    'status': 'open',                  # Position status
    'current_pnl': 50.0,               # Current P&L
    'current_spread': 2.2,             # Current spread value
    'current_zscore': 1.0,             # Current z-score
    'holding_time_minutes': 120,       # Holding time in minutes
    'risk_metrics': {...},             # Risk metrics
    'performance_history': [...]       # Historical performance data
}
```

## Risk Management Features

The Position Manager implements several risk management features:

### 1. Stop Loss

Stop losses are triggered when the z-score moves too far against the position:

```python
# Check stop loss for a position
if position_manager.check_stop_losses('GC_SI', current_data):
    print("Stop loss triggered for GC_SI")
```

### 2. Take Profit

Take profit is triggered when the z-score reverts back to the mean:

```python
# Check take profit for a position
if position_manager.check_take_profits('GC_SI', current_data):
    print("Take profit triggered for GC_SI")
```

### 3. Maximum Holding Period

Positions are automatically closed if they exceed the maximum holding period:

```python
# Check holding time limit for a position
if position_manager.check_holding_limits('GC_SI'):
    print("Maximum holding period exceeded for GC_SI")
```

### 4. Correlation Breakdown

Positions are closed if the correlation between the pair legs breaks down:

```python
# Check correlation for a position
if position_manager.check_correlation_breakdown('GC_SI', current_data):
    print("Correlation breakdown detected for GC_SI")
```

## Advanced Features

### 1. Position Size Adjustment

The Position Manager can dynamically adjust position size based on market volatility:

```python
# Calculate volatility metrics
volatility_metrics = {
    'current_volatility': 0.2,    # Current volatility measure
    'baseline_volatility': 0.1    # Baseline/historical volatility
}

# Adjust position size
if position_manager.adjust_position_size('GC_SI', volatility_metrics):
    print("Position size adjusted for GC_SI")
```

### 2. Performance Tracking

Track detailed performance metrics for a position:

```python
# Track performance metrics
metrics = position_manager.track_position_performance('GC_SI', current_data)

# Print key metrics
print(f"Total P&L: ${metrics['total_pnl']:.2f}")
print(f"Holding time: {metrics['holding_time_minutes']:.1f} minutes")
```

### 3. Risk Analysis

Analyze risk metrics for a position:

```python
# Analyze risk metrics
risk = position_manager.analyze_position_risk('GC_SI', current_data)

# Print key risk metrics
print(f"Risk/reward ratio: {risk['risk_reward_ratio']:.2f}")
print(f"Distance to stop loss: {risk['distance_to_stop_loss']:.2f}")
```

## Integration with IntradayMLPaperTrader

To integrate the Position Manager with the IntradayMLPaperTrader:

1. Initialize the Position Manager in the IntradayMLPaperTrader constructor
2. Add trading pairs to the Position Manager during setup
3. Route trading signals to the Position Manager's execute_signals method
4. Call the monitor_all_positions method periodically to perform risk management

Example integration:

```python
class IntradayMLPaperTrader:
    def __init__(self, initial_capital=100000.0, ...):
        # Create paper trader
        self.paper_trader = PaperTrader(initial_capital=initial_capital, ...)
        
        # Create position manager
        self.position_manager = PositionManager(
            paper_trader=self.paper_trader,
            config={'default_position_size': 0.1}
        )
        
        # Other initialization code...
    
    def setup_pairs(self, pairs_config):
        # Add pairs to position manager
        for pair in pairs_config:
            self.position_manager.add_pair(pair)
    
    def process_signals(self, pair_id, signals):
        # Execute signals
        return self.position_manager.execute_signals(pair_id, signals)
    
    def monitor_positions(self, market_data):
        # Monitor all positions
        return self.position_manager.monitor_all_positions(market_data)
```

## Testing

The Position Manager has comprehensive unit tests covering all functionality. Run the tests with:

```bash
pytest tests/unit/paper_trading/components/test_position_manager.py
```

## Configuration Options

The Position Manager accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| default_position_size | Default position size as a fraction of account value | 0.1 (10%) |
| risk_per_trade | Maximum risk per trade as a fraction of account value | 0.02 (2%) |
| min_correlation | Global minimum correlation threshold | 0.5 |
| default_stop_loss_z | Default stop loss z-score threshold | 3.0 |
| default_take_profit_z | Default take profit z-score threshold | 0.5 |
| default_max_holding_period | Default maximum holding period in minutes | 180 |

## Best Practices

- **Pair Configuration**: Configure each pair with appropriate stop-loss and take-profit levels based on historical behavior
- **Position Sizing**: Set appropriate position sizes based on pair volatility and account risk tolerance
- **Monitoring Frequency**: Call monitor_all_positions at an appropriate frequency (e.g., every 5 minutes for intraday trading)
- **Correlation Thresholds**: Set correlation thresholds based on the historical correlation stability of each pair
- **Performance Tracking**: Regularly analyze the performance metrics to identify opportunities for improvement 