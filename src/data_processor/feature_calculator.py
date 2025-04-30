import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import concurrent.futures
import time

# Import technical indicators utility module
from src.utils.technical_indicators import (
    rsi, macd, bollinger_bands, moving_averages, 
    exponential_moving_averages, atr, z_score, 
    stochastic_oscillator, rate_of_change, add_all_indicators
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureCalculator:
    """
    Calculates technical indicators and features for financial time series data
    """
    def __init__(self, data_dir: str = "data/historical", feature_dir: str = "data/features"):
        """
        Initialize the FeatureCalculator
        
        Parameters:
        -----------
        data_dir : str
            Directory containing historical price data
        feature_dir : str
            Directory to store generated features
        """
        self.data_dir = Path(data_dir)
        self.feature_dir = Path(feature_dir)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_features(self, symbol: str, timeframe: str = "1day", 
                           start_date: Optional[str] = None, end_date: Optional[str] = None,
                           save_to_file: bool = True) -> pd.DataFrame:
        """
        Calculate technical indicators for a given symbol and timeframe
        
        Parameters:
        -----------
        symbol : str
            Symbol to calculate features for
        timeframe : str
            Timeframe of the data ('1min', '5min', '1hour', '1day')
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        save_to_file : bool
            Whether to save the features to a file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated features
        """
        # Load price data
        price_file = self.data_dir / symbol / timeframe / "data.csv"
        
        if not price_file.exists():
            logger.error(f"Price data file not found: {price_file}")
            return pd.DataFrame()
        
        logger.info(f"Loading price data for {symbol} {timeframe}")
        df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        
        # Filter by date if needed
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        if df.empty:
            logger.error(f"No data found for {symbol} {timeframe} in date range")
            return pd.DataFrame()
        
        # Calculate features
        logger.info(f"Calculating features for {symbol} {timeframe}")
        features_df = self._calculate_all_features(df)
        
        # Save to file if requested
        if save_to_file:
            feature_dir = self.feature_dir / symbol / timeframe
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_file = feature_dir / "features.csv"
            features_df.to_csv(feature_file)
            logger.info(f"Saved features to {feature_file}")
            
        return features_df
    
    def _calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and features
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all calculated features
        """
        result = df.copy()
        
        # Add standard technical indicators using the utility functions
        include_high_low = 'high' in df.columns and 'low' in df.columns
        
        # Use the comprehensive add_all_indicators function for most indicators
        result = add_all_indicators(result, price_col='close', include_high_low=include_high_low)
        
        # Additional custom indicators
        result = self._calculate_volume_indicators(result)
        result = self._calculate_volatility_indicators(result)
        result = self._calculate_trend_indicators(result)
        
        # Add moving average crossovers (not in the standard utility function)
        result['sma_5_10_cross'] = (result['sma_5'] > result['sma_10']).astype(int)
        result['sma_20_50_cross'] = (result['sma_20'] > result['sma_50']).astype(int)
        result['ema_5_10_cross'] = (result['ema_5'] > result['ema_10']).astype(int)
        
        # Price relative to moving averages (not in the standard utility function)
        result['close_vs_sma_20'] = result['close'] / result['sma_20'] - 1
        result['close_vs_sma_50'] = result['close'] / result['sma_50'] - 1
        if 'sma_200' in result.columns:
            result['close_vs_sma_200'] = result['close'] / result['sma_200'] - 1
        
        # Remove NaN values that might have been introduced
        result = result.dropna()
        
        return result
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        result = df.copy()
        
        # Volume Moving Averages
        for period in [5, 10, 20]:
            result[f'volume_sma_{period}'] = result['volume'].rolling(window=period).mean()
        
        # Volume relative to moving average
        result['volume_vs_sma_20'] = result['volume'] / result['volume_sma_20']
        
        # On-Balance Volume (OBV)
        obv = 0
        obvs = []
        
        for i, row in result.iterrows():
            if i == 0 or pd.isna(result.loc[i, 'close']) or pd.isna(result.loc[i-1, 'close']):
                obvs.append(obv)
                continue
                
            if result.loc[i, 'close'] > result.loc[i-1, 'close']:
                obv += result.loc[i, 'volume']
            elif result.loc[i, 'close'] < result.loc[i-1, 'close']:
                obv -= result.loc[i, 'volume']
                
            obvs.append(obv)
            
        result['obv'] = obvs
        
        # OBV moving average
        result['obv_sma'] = result['obv'].rolling(window=20).mean()
        
        return result
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        result = df.copy()
        
        # Return volatility
        result['returns'] = result['close'].pct_change()
        
        for period in [5, 10, 20]:
            result[f'volatility_{period}'] = result['returns'].rolling(window=period).std() * np.sqrt(252)  # Annualized
        
        # High-Low Range
        if 'high' in result.columns and 'low' in result.columns:
            result['high_low_range'] = (result['high'] - result['low']) / result['close'] * 100
            
            # Average High-Low Range
            result['avg_range_10'] = result['high_low_range'].rolling(window=10).mean()
        
        return result
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        result = df.copy()
        
        # ADX (Average Directional Index) - Simplified version
        if 'high' in result.columns and 'low' in result.columns:
            # First, calculate +DI and -DI
            plus_dm = result['high'].diff()
            minus_dm = result['low'].diff(-1).abs()
            
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            # Smooth with EMA
            plus_di = plus_dm.ewm(span=14, adjust=False).mean()
            minus_di = minus_dm.ewm(span=14, adjust=False).mean()
            
            # Calculate DX
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            
            # Calculate ADX
            result['adx'] = dx.ewm(span=14, adjust=False).mean()
            
            # Trend strength indicators
            result['strong_trend'] = (result['adx'] > 25).astype(int)
            result['very_strong_trend'] = (result['adx'] > 50).astype(int)
            
            # Higher highs, lower lows
            result['higher_high'] = ((result['high'] > result['high'].shift(1)) & 
                                    (result['high'].shift(1) > result['high'].shift(2))).astype(int)
            result['lower_low'] = ((result['low'] < result['low'].shift(1)) & 
                                (result['low'].shift(1) < result['low'].shift(2))).astype(int)
        
        return result


def generate_features(symbols: Union[str, List[str]], 
                      timeframes: Union[str, List[str]] = "1day",
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      data_dir: str = "data/historical",
                      feature_dir: str = "data/features",
                      parallel: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate technical features for multiple symbols and timeframes
    
    Parameters:
    -----------
    symbols : str or List[str]
        Symbol or list of symbols to generate features for
    timeframes : str or List[str]
        Timeframe or list of timeframes ('1min', '5min', '1hour', '1day')
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    data_dir : str
        Directory containing historical price data
    feature_dir : str
        Directory to store generated features
    parallel : bool
        Whether to process data in parallel
        
    Returns:
    --------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary with symbol and timeframe as keys, and feature DataFrames as values
    """
    # Convert single inputs to lists
    if isinstance(symbols, str):
        symbols = [symbols]
    if isinstance(timeframes, str):
        timeframes = [timeframes]
        
    calculator = FeatureCalculator(data_dir=data_dir, feature_dir=feature_dir)
    results = {}
    
    if parallel and len(symbols) * len(timeframes) > 1:
        logger.info(f"Generating features for {len(symbols)} symbols and {len(timeframes)} timeframes in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(symbols) * len(timeframes))) as executor:
            # Create calculation tasks
            future_to_params = {}
            for symbol in symbols:
                results[symbol] = {}
                for tf in timeframes:
                    future = executor.submit(
                        calculator.calculate_features, symbol, tf, start_date, end_date, True
                    )
                    future_to_params[future] = (symbol, tf)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_params):
                symbol, tf = future_to_params[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol][tf] = data
                except Exception as e:
                    logger.error(f"Error processing features for {symbol} {tf}: {str(e)}")
    else:
        logger.info(f"Generating features for {len(symbols)} symbols and {len(timeframes)} timeframes sequentially")
        for symbol in symbols:
            results[symbol] = {}
            for tf in timeframes:
                try:
                    data = calculator.calculate_features(symbol, tf, start_date, end_date, True)
                    if data is not None and not data.empty:
                        results[symbol][tf] = data
                except Exception as e:
                    logger.error(f"Error processing features for {symbol} {tf}: {str(e)}")
    
    logger.info(f"Completed feature generation for {len(symbols)} symbols")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate technical features for market data")
    parser.add_argument("--symbols", required=True, help="Comma-separated list of symbols")
    parser.add_argument("--timeframes", default="1day", help="Comma-separated list of timeframes (1min,5min,1hour,1day)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data/historical", help="Data directory")
    parser.add_argument("--feature-dir", default="data/features", help="Feature directory")
    parser.add_argument("--sequential", action="store_true", help="Process sequentially instead of in parallel")
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")
    
    generate_features(
        symbols=symbols,
        timeframes=timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        feature_dir=args.feature_dir,
        parallel=not args.sequential
    ) 