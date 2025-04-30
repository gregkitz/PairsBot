# Duplicate Code Analysis: Asset Class Implementations

This document analyzes duplicate code patterns in asset class implementations and provides recommendations for consolidation.

## Overview

Our codebase contains several asset class implementations with notable code duplication, particularly in:

1. Data retrieval methods
2. Error handling patterns
3. Common API interactions
4. Standard information retrieval

After examining the codebase, the most significant duplication appears in the implementations of `get_data()`, `get_metadata()`, `get_current_price()`, and `get_trading_hours()` methods across different asset classes.

## Current Structure

The asset classes are structured as follows:

```
src/asset_classes/
  ├── base.py                 # Contains Asset and AssetClass abstract base classes
  ├── factory.py              # Factory for creating asset instances
  ├── __init__.py             # Package exports
  ├── futures/
  │   ├── futures_asset.py    # FuturesAsset and FuturesAssetClass implementations
  │   └── __init__.py
  ├── equities/
  │   ├── equity_asset.py     # EquityAsset and EquityAssetClass implementations
  │   └── __init__.py
  ├── fixed_income/
  │   ├── fixed_income_asset.py # FixedIncomeAsset implementations
  │   └── __init__.py
  └── cryptocurrencies/
      ├── crypto_asset.py     # CryptoAsset implementations
      └── __init__.py
```

## Duplicate Code Examples

### Example 1: `get_data()` Method

**FuturesAsset**:
```python
def get_data(self, start_date: Union[str, datetime], 
            end_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Get historical price data for the futures contract.
    """
    logger.info(f"Getting data for {self.symbol} from {start_date} to {end_date}")
    try:
        return self.connector.get_historical_data(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            exchange=self.exchange,
            security_type='FUT'
        )
    except Exception as e:
        logger.error(f"Error getting data for {self.symbol}: {str(e)}")
        return pd.DataFrame()
```

**EquityAsset**:
```python
def get_data(self, start_date: Union[str, datetime], 
            end_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Get historical price data for the equity.
    """
    logger.info(f"Getting data for {self.symbol} from {start_date} to {end_date}")
    try:
        return self.connector.get_historical_data(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            exchange=self.exchange,
            security_type='STK'
        )
    except Exception as e:
        logger.error(f"Error getting data for {self.symbol}: {str(e)}")
        return pd.DataFrame()
```

The only difference is the `security_type` parameter value.

### Example 2: `get_current_price()` Method

**FuturesAsset**:
```python
def get_current_price(self) -> float:
    """
    Get the current price of the futures contract.
    """
    try:
        return self.connector.get_market_data(
            symbol=self.symbol,
            exchange=self.exchange
        ).get("last_price", 0.0)
    except Exception as e:
        logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
        return 0.0
```

**EquityAsset**:
```python
def get_current_price(self) -> float:
    """
    Get the current price of the equity.
    """
    try:
        return self.connector.get_market_data(
            symbol=self.symbol,
            exchange=self.exchange,
            security_type='STK'
        ).get("last_price", 0.0)
    except Exception as e:
        logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
        return 0.0
```

Similar pattern with small parameter differences.

## Common Patterns of Duplication

1. **Data Retrieval**: Each asset class implements `get_data()` with 95% identical code
2. **Error Handling**: Try/except blocks with identical logging and fallback behavior
3. **API Interaction**: Similar connector method calls with slight parameter differences
4. **Metadata Retrieval**: Similar patterns for fetching and returning metadata
5. **Structure**: Each asset class has nearly identical class structure

## Solution: Template Method Pattern and Mixins

### Approach 1: Enhanced Base Class with Template Methods

Enhance the `Asset` base class with template methods:

```python
class Asset(abc.ABC):
    # Existing base class code...
    
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical price data for the asset.
        
        Template method implementation that handles common patterns.
        """
        logger.info(f"Getting data for {self.symbol} from {start_date} to {end_date}")
        try:
            return self.connector.get_historical_data(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                exchange=self.get_exchange(),
                security_type=self.get_security_type()
            )
        except Exception as e:
            logger.error(f"Error getting data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self) -> float:
        """
        Get the current price of the asset.
        
        Template method implementation that handles common patterns.
        """
        try:
            return self.connector.get_market_data(
                symbol=self.symbol,
                exchange=self.get_exchange(),
                security_type=self.get_security_type()
            ).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
            return 0.0
    
    @abc.abstractmethod
    def get_security_type(self) -> str:
        """Return the security type for this asset."""
        pass
    
    def get_exchange(self) -> Optional[str]:
        """Return the exchange for this asset. Can be overridden."""
        return getattr(self, 'exchange', None)
```

### Approach 2: Functional Mixins

Create mixins for common functionality:

```python
class DataRetrievalMixin:
    """Mixin for data retrieval functionality."""
    
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """Get historical price data."""
        logger.info(f"Getting data for {self.symbol} from {start_date} to {end_date}")
        try:
            return self.connector.get_historical_data(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                exchange=self.exchange,
                security_type=self.security_type
            )
        except Exception as e:
            logger.error(f"Error getting data for {self.symbol}: {str(e)}")
            return pd.DataFrame()

class MarketDataMixin:
    """Mixin for market data functionality."""
    
    def get_current_price(self) -> float:
        """Get the current price."""
        try:
            return self.connector.get_market_data(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type=self.security_type
            ).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
            return 0.0
```

## Proposed Implementation

### 1. Enhanced Base Class Implementation

```python
# Enhanced Asset base class
class Asset(abc.ABC):
    """Base class for all asset types."""
    
    def __init__(self, symbol: str, name: Optional[str] = None, 
                 connector: Optional[Any] = None, **kwargs):
        self.symbol = symbol
        self.name = name if name is not None else symbol
        self.connector = connector
        self.metadata = kwargs
    
    # Template method pattern
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """Get historical price data with error handling."""
        logger.info(f"Getting data for {self.symbol} from {start_date} to {end_date}")
        try:
            return self._fetch_historical_data(start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    @abc.abstractmethod
    def _fetch_historical_data(self, start_date, end_date) -> pd.DataFrame:
        """Fetch the historical data - to be implemented by subclasses."""
        pass
    
    def get_current_price(self) -> float:
        """Get current price with error handling."""
        try:
            return self._fetch_current_price()
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
            return 0.0
    
    @abc.abstractmethod
    def _fetch_current_price(self) -> float:
        """Fetch current price - to be implemented by subclasses."""
        pass
```

### 2. Implementation in Asset Subclasses

```python
class FuturesAsset(Asset):
    """Futures asset implementation."""
    
    def __init__(self, symbol: str, name: Optional[str] = None, 
                exchange: Optional[str] = None, 
                contract_size: Optional[float] = None,
                tick_size: Optional[float] = None,
                connector: Optional[Any] = None,
                **kwargs):
        super().__init__(symbol, name, connector, **kwargs)
        self.exchange = exchange
        self.contract_size = contract_size
        self.tick_size = tick_size
        self.security_type = 'FUT'
    
    def _fetch_historical_data(self, start_date, end_date) -> pd.DataFrame:
        """Futures-specific implementation of data fetching."""
        return self.connector.get_historical_data(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            exchange=self.exchange,
            security_type=self.security_type
        )
    
    def _fetch_current_price(self) -> float:
        """Futures-specific implementation of price fetching."""
        return self.connector.get_market_data(
            symbol=self.symbol,
            exchange=self.exchange,
            security_type=self.security_type
        ).get("last_price", 0.0)
```

## Benefits of Consolidation

1. **Reduced Code Duplication**: Eliminates ~500 lines of duplicated code
2. **Centralized Error Handling**: Common error handling in one place
3. **Consistent Behavior**: Ensures consistent behavior across asset classes
4. **Easier Maintenance**: Changes only need to be made in one place
5. **Improved Extensibility**: New asset classes will inherit common functionality

## Implementation Plan

1. **Phase 1: Enhance Base Class (2 days)**
   - Add template methods to `Asset` class
   - Add hooks for subclass customization
   - Add error handling utilities

2. **Phase 2: Refactor Subclasses (3 days)**
   - Update `FuturesAsset` to use template methods
   - Update `EquityAsset` to use template methods
   - Update `FixedIncomeAsset` to use template methods
   - Update `CryptoAsset` to use template methods

3. **Phase 3: Testing and Validation (1 day)**
   - Create tests for the base class
   - Verify subclass behavior matches original
   - Check edge cases and error handling

## Conclusion

The asset class implementations contain significant code duplication that can be consolidated using the Template Method pattern and better base class design. This consolidation would reduce code size, improve maintainability, and ensure consistent behavior across all asset classes.

The estimated effort for this refactoring is 6 developer days, with the primary challenge being to ensure backward compatibility and maintain existing behavior while reducing duplication. 