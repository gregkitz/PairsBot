# Paper Trading Guide

This guide provides detailed instructions for using the ML-enhanced paper trading system to test and validate trading strategies before deploying them with real capital.

## Overview

The paper trading system allows you to:

1. Test trading strategies using real-time market data
2. Apply ML-enhanced signal filtering and entry/exit timing
3. Track performance metrics in a real-time dashboard
4. Adapt to different market regimes
5. Validate strategies before deploying them to live trading

## Setup

### Prerequisites

1. Python 3.8+ environment
2. Interactive Brokers TWS or IB Gateway running
3. Required Python packages installed:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Edit `config/paper_trading.json` to configure trading pairs and parameters:
   ```json
   {
     "pairs": [
       {
         "pair_id": "GC_SI",
         "symbol1": "GC", 
         "symbol2": "SI",
         "full_symbol1": "GC-202406-FUT-COMEX",
         "full_symbol2": "SI-202406-FUT-COMEX",
         "config": {
           "hedge_ratio": 0.1,
           "lookback": 50,
           "entry_zscore": 2.0,
           "exit_zscore": 0.5,
           "stop_loss_std": 3.0
         }
       }
     ],
     "ml_system": {
       "models_dir": "models/intraday",
       "use_regime_detection": true
     }
   }
   ```

2. Key configuration parameters:
   - **pair_id**: Unique identifier for the pair
   - **symbol1/symbol2**: Trading symbols
   - **full_symbol1/full_symbol2**: Detailed contract specifications in the format "SYMBOL-EXPIRY-TYPE-EXCHANGE"
   - **hedge_ratio**: Fixed hedge ratio between the pair (or use rolling regression)
   - **entry_zscore/exit_zscore**: Z-score thresholds for entering/exiting trades
   - **ml_system**: Settings for ML enhancements and model locations

## Running Paper Trading

### Basic Usage

```bash
python run_ml_paper_trader.py --config config/paper_trading.json
```

### Command-line Options

- `--config PATH`: Path to configuration file (default: config/paper_trading.json)
- `--capital AMOUNT`: Initial capital (default: 100000.0)
- `--ib-host HOST`: IB TWS/Gateway host (default: 127.0.0.1)
- `--ib-port PORT`: IB TWS/Gateway port (default: 7497)
- `--ib-client-id ID`: IB client ID (default: 1)
- `--dashboard-refresh SECONDS`: Dashboard refresh interval (default: 300)
- `--data-refresh SECONDS`: Market data refresh interval (default: 60)
- `--auto-shutdown TIME`: Time to automatically shut down (format: HH:MM, default: 16:00)
- `--no-dashboard`: Disable performance dashboard
- `--no-alerts`: Disable alerts
- `--test-mode`: Run in test mode without real IB connection

### Examples

1. Run with default settings:
   ```bash
   python run_ml_paper_trader.py
   ```

2. Run with custom capital and port:
   ```bash
   python run_ml_paper_trader.py --capital 200000 --ib-port 4002
   ```

3. Run in test mode without IB connection:
   ```bash
   python run_ml_paper_trader.py --test-mode
   ```

## Monitoring Performance

### Dashboard

The paper trading system generates a real-time dashboard in the `output/paper_trading/dashboard` directory. Open `index.html` to view:

- Equity curve
- Drawdown chart
- Performance metrics (Sharpe ratio, win rate, profit factor)
- Regime analysis
- Trade history
- Position status

### Logs

Detailed logs are saved in the `output/paper_trading/logs` directory, including:
- Trading signals
- Executed trades
- Market regime changes
- Error messages

## Advanced Features

### ML Enhancements

The paper trading system uses several ML enhancements:

1. **Signal Filtering**: Reduces false signals based on market conditions
2. **Entry/Exit Timing**: Optimizes entry and exit points
3. **Regime Detection**: Identifies market regimes and adapts parameters
4. **Volatility Adaptation**: Adjusts position sizing based on volatility

### Custom Contracts

For futures contracts, you can specify detailed contract information using the `full_symbol` fields:

```json
"full_symbol1": "ES-202406-FUT-CME"
```

Format: `SYMBOL-EXPIRY-TYPE-EXCHANGE`
- SYMBOL: The contract symbol (e.g., ES, GC)
- EXPIRY: Contract expiration in YYYYMM format (e.g., 202406)
- TYPE: Contract type (FUT for futures)
- EXCHANGE: Exchange where the contract is traded (e.g., CME, COMEX)

### Parameter Adaptation

The system can automatically adapt parameters based on market regimes:

```json
"regime_adaptation": {
  "enabled": true,
  "update_frequency": "daily",
  "parameter_ranges": {
    "entry_zscore": [1.5, 3.0],
    "exit_zscore": [0.0, 1.0]
  }
}
```

## Troubleshooting

### Common Issues

1. **Connection Issues**:
   - Ensure TWS/IB Gateway is running
   - Check that the port number matches your TWS/Gateway settings
   - Verify that API connections are enabled in TWS/Gateway

2. **Market Data Issues**:
   - Verify you have market data subscriptions for the instruments
   - Check connection to market data farms in TWS/Gateway logs

3. **Contract Specification Issues**:
   - Try using the full contract specification with the `full_symbol` field
   - Check contract expiry dates are correct
   - Verify the exchange is correctly specified

### Solutions

1. **Restart TWS/Gateway**:
   Sometimes a simple restart of TWS/Gateway resolves connection issues.

2. **Check Logs**:
   The system logs detailed information in the `output/paper_trading/logs` directory.

3. **Test Mode**:
   Use the `--test-mode` flag to test functionality without an actual IB connection.

## Next Steps

After successfully validating your strategy in paper trading:

1. Review performance metrics and trade history
2. Identify areas for improvement
3. Make necessary adjustments to trading parameters
4. Gradually transition to live trading with minimal position sizes
5. Monitor performance and compare with paper trading results 