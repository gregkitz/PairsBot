# Interactive Brokers TWS Integration Testing Guide

This guide provides steps to set up and run the integration tests with Interactive Brokers TWS.

## Prerequisites

1. Interactive Brokers TWS installed and running
2. Paper Trading account (recommended for testing)
3. Python environment with required packages

## TWS Configuration

Before running the tests, ensure TWS is properly configured:

1. Open TWS and go to Edit > Global Configuration > API > Settings
2. Make sure the following settings are enabled:
   - Enable Active X and Socket Clients
   - Socket port: 7497 (Paper Trading) or 7496 (Live Trading)
   - Allow connections from localhost only (for security)
   - Uncheck "Read-Only API"
   - Set "Master API client ID" to a unique number
3. Optional but recommended:
   - Enable "Let API account queries run in threads for better response time"
   - Set "Maximum number of API message lines in log" to 500 or more
4. Click Apply and OK

## API Permissions

TWS requires explicit permission for API connections:

1. When the test connects, TWS will display a popup asking for permission
2. Make sure to check "Do not show this message again" if you want to avoid this in the future
3. Click "Yes" to allow the connection

## Running the Tests

```bash
# Run all IB integration tests
python run_ib_tests.py

# Run a specific test
python -m unittest tests.integration.test_ib_integration.TestIBIntegration.test_equity_asset_integration
```

## Troubleshooting

### Connection Issues

- **Check TWS is running**: Make sure TWS is open and logged in
- **Verify port**: Ensure TWS is listening on port 7497 (for paper trading)
- **API permissions**: Check if TWS is showing an API permission dialog
- **Client ID conflicts**: Try a different client ID if connection fails
- **Restart TWS**: Sometimes restarting TWS helps resolve connection issues
- **Check logs**: Look at TWS logs (Help > Logs > API) for connection errors

### Data Issues

- **Market hours**: Some assets may not have data outside market hours
- **Subscription requirements**: Some data may require specific subscriptions
- **Exchange permissions**: Your account may need permissions for certain exchanges
- **Symbol format**: Different markets may require specific symbol formats

## Asset Types in TWS

Our tests cover:

1. **Equities**: Standard stocks like "AAPL" on "SMART" exchange
2. **Futures**: Futures contracts like "ES" on "CME" exchange
3. **Cryptocurrencies**: Available through "PAXOS" exchange (limited availability)
4. **Fixed Income**: Bond ETFs like "TLT" as a proxy for fixed income

## Note on Test Skipping

The tests are designed to be resilient and will be skipped if:
- TWS connection fails
- A particular asset type is not available
- Market data cannot be retrieved

This allows the test suite to continue with available functionality.

## Additional Resources

- [IB Insync Documentation](https://ib-insync.readthedocs.io/)
- [TWS API Documentation](https://interactivebrokers.github.io/tws-api/)
- [TWS User Guide](https://guides.interactivebrokers.com/tws/) 