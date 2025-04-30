"""
Intraday Data Processor Module

This module provides specialized functionality for processing intraday data for pairs trading,
with optimizations for efficient loading and preprocessing of high-frequency data.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntradayDataProcessor:
    """
    Specialized processor for intraday data with optimizations for pairs trading.
    """
    
    def __init__(self, data_dir="data/processed", cache_enabled=True, max_workers=4):
        """
        Initialize the IntradayDataProcessor.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing processed data files
        cache_enabled : bool
            Whether to cache loaded data in memory
        max_workers : int
            Maximum number of worker threads for parallel processing
        """
        self.data_dir = data_dir
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        self.data_cache = {}
        self.available_symbols = None
        self.timezone = pytz.timezone('US/Eastern')
        
        # Market hours (9:30 AM - 4:00 PM Eastern)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Create output directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def get_available_symbols(self, refresh=False):
        """
        Get list of available symbols.
        
        Parameters:
        -----------
        refresh : bool
            Whether to refresh the list of available symbols
            
        Returns:
        --------
        list
            List of available symbols
        """
        if self.available_symbols is None or refresh:
            try:
                # Find all parquet files
                files = [f for f in os.listdir(self.data_dir) if f.endswith('_processed.parquet')]
                self.available_symbols = [f.split('_processed.parquet')[0] for f in files]
                return sorted(self.available_symbols)
            except Exception as e:
                logger.error(f"Error getting available symbols: {str(e)}")
                return []
        else:
            return self.available_symbols
    
    def load_intraday_data(self, symbol, start_date, end_date, timeframe="5min"):
        """
        Load intraday data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to load data for
        start_date : str
            Start date for data (YYYY-MM-DD)
        end_date : str
            End date for data (YYYY-MM-DD)
        timeframe : str
            Timeframe to resample data to
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with intraday data
        """
        # Check if data is in cache
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        if self.cache_enabled and cache_key in self.data_cache:
            logger.debug(f"Using cached data for {symbol}")
            return self.data_cache[cache_key]
        
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_processed.parquet")
            
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                return None
            
            # Load data efficiently with filters
            logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
            
            # Read parquet with predicate pushdown if possible
            if start_date is not None and end_date is not None:
                # Convert dates to timestamps for filtering
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                # Use predicate pushdown for efficient filtering
                df = pd.read_parquet(
                    file_path,
                    filters=[('timestamp', '>=', start_ts), ('timestamp', '<=', end_ts)]
                )
            else:
                df = pd.read_parquet(file_path)
            
            # Ensure data is sorted by timestamp
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    logger.warning(f"No timestamp column found for {symbol}")
                    return None
            
            df.sort_index(inplace=True)
            
            # Resample to desired timeframe
            if timeframe != "1min":
                # For OHLCV data
                ohlcv_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                
                # Extract available columns
                resample_dict = {col: ohlcv_dict[col] for col in ohlcv_dict.keys() if col in df.columns}
                
                # Resample
                df = df.resample(timeframe).agg(resample_dict)
            
            # Keep only market hours
            df = df.between_time('09:30', '16:00')
            
            # Clean data
            df = self._clean_data(df)
            
            logger.info(f"Loaded data for {symbol}: {len(df)} rows")
            
            # Cache the result
            if self.cache_enabled:
                self.data_cache[cache_key] = df
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def load_pair_data(self, pair_id, start_date, end_date, timeframe="5min"):
        """
        Load intraday data for a pair.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier (e.g. "BFX_ZN")
        start_date : str
            Start date for data (YYYY-MM-DD)
        end_date : str
            End date for data (YYYY-MM-DD)
        timeframe : str
            Timeframe to resample data to
            
        Returns:
        --------
        dict
            Dictionary with data for each symbol and the spread
        """
        logger.info(f"Loading data for pair {pair_id} from {start_date} to {end_date}")
        
        # Split pair_id to get symbols
        try:
            symbol1, symbol2 = pair_id.split('_')
        except ValueError:
            logger.error(f"Invalid pair_id format: {pair_id}")
            return None
        
        # Load data for both symbols in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_symbol1 = executor.submit(
                self.load_intraday_data, symbol1, start_date, end_date, timeframe)
            future_symbol2 = executor.submit(
                self.load_intraday_data, symbol2, start_date, end_date, timeframe)
            
            data = {}
            data[symbol1] = future_symbol1.result()
            data[symbol2] = future_symbol2.result()
        
        # Check if we have data for both symbols
        if data[symbol1] is None or data[symbol2] is None:
            logger.error(f"Missing data for one or both symbols in pair {pair_id}")
            return None
        
        # Create price DataFrame
        prices_df = pd.DataFrame({
            symbol1: data[symbol1]['close'],
            symbol2: data[symbol2]['close']
        })
        
        # Create volume DataFrame if available
        volumes_df = None
        if 'volume' in data[symbol1].columns and 'volume' in data[symbol2].columns:
            volumes_df = pd.DataFrame({
                symbol1: data[symbol1]['volume'],
                symbol2: data[symbol2]['volume']
            })
        
        return {
            'prices': prices_df,
            'volumes': volumes_df,
            'data': data
        }
    
    def load_multiple_pairs(self, pair_ids, start_date, end_date, timeframe="5min"):
        """
        Load intraday data for multiple pairs in parallel.
        
        Parameters:
        -----------
        pair_ids : list
            List of pair identifiers
        start_date : str
            Start date for data (YYYY-MM-DD)
        end_date : str
            End date for data (YYYY-MM-DD)
        timeframe : str
            Timeframe to resample data to
            
        Returns:
        --------
        dict
            Dictionary with data for each pair
        """
        logger.info(f"Loading data for {len(pair_ids)} pairs")
        
        result = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_pair = {
                executor.submit(self.load_pair_data, pair_id, start_date, end_date, timeframe): pair_id
                for pair_id in pair_ids
            }
            
            # Collect results
            for future in as_completed(future_to_pair):
                pair_id = future_to_pair[future]
                try:
                    data = future.result()
                    result[pair_id] = data
                except Exception as e:
                    logger.error(f"Error loading data for pair {pair_id}: {e}")
                    result[pair_id] = None
        
        return result
    
    def _clean_data(self, df):
        """
        Clean data by handling missing values and outliers.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to clean
            
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        if df is None or len(df) == 0:
            return df
        
        # Handle missing values
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Forward fill first (carry last observation forward)
                df[col] = df[col].ffill()
                
                # Then backward fill (for any remaining NAs at the beginning)
                df[col] = df[col].bfill()
        
        # Handle missing volume
        if 'volume' in df.columns:
            # Fill missing volume with 0
            df['volume'] = df['volume'].fillna(0)
        
        # Ensure OHLC constraints
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Make sure high >= open, close, low
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            
            # Make sure low <= open, close, high
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    def interpolate_missing_bars(self, df, timeframe="5min"):
        """
        Interpolate missing bars in the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with potentially missing bars
        timeframe : str
            Timeframe of the data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with interpolated missing bars
        """
        if df is None or len(df) == 0:
            return df
        
        # Create a complete time index for market hours
        start_date = df.index[0].date()
        end_date = df.index[-1].date()
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter for weekdays (Monday=0, Friday=4)
        weekdays = [d for d in dates if d.weekday() < 5]
        
        # Create complete time index
        time_idx = []
        
        for day in weekdays:
            # Market hours: 9:30 AM to 4:00 PM
            start_time = datetime.combine(day.date(), self.market_open)
            end_time = datetime.combine(day.date(), self.market_close)
            
            # Create time range for the day
            day_times = pd.date_range(start=start_time, end=end_time, freq=timeframe)
            
            # Add to overall index
            time_idx.extend(day_times)
        
        # Convert to DatetimeIndex
        full_idx = pd.DatetimeIndex(time_idx)
        
        # Reindex the DataFrame
        reindexed = df.reindex(full_idx)
        
        # Interpolate
        for col in ['open', 'high', 'low', 'close']:
            if col in reindexed.columns:
                reindexed[col] = reindexed[col].interpolate(method='linear')
        
        # Forward fill any remaining NaNs
        reindexed = reindexed.ffill().bfill()
        
        return reindexed

    def calculate_spread(self, prices_df, symbol1, symbol2, config):
        """
        Calculate spread for a pair.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data
        symbol1 : str
            First symbol in the pair
        symbol2 : str
            Second symbol in the pair
        config : dict
            Configuration for the pair
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with spread data
        """
        logger.info(f"Calculating spread for pair {symbol1}_{symbol2}")
        
        # Ensure we have data for both symbols
        if symbol1 not in prices_df.columns or symbol2 not in prices_df.columns:
            logger.error(f"Missing price data for one or both symbols in pair {symbol1}_{symbol2}")
            return None
        
        # Extract price data
        symbol1_prices = prices_df[symbol1]
        symbol2_prices = prices_df[symbol2]
        
        # Get lookback parameter
        lookback = config.get("lookback", 50)
        
        # Calculate spread based on configuration
        if config.get("use_rolling_regression", False):
            # Use rolling regression to calculate hedge ratio
            regression_window = config.get("regression_window", 60)
            
            # Initialize spread dataframe
            spread_df = pd.DataFrame(index=prices_df.index)
            
            # Calculate returns
            symbol1_returns = symbol1_prices.pct_change()
            symbol2_returns = symbol2_prices.pct_change()
            
            # Calculate rolling hedge ratio
            try:
                rolling_cov = symbol1_returns.rolling(regression_window).cov(symbol2_returns)
                rolling_var = symbol2_returns.rolling(regression_window).var()
                
                # Avoid division by zero
                rolling_var = rolling_var.replace(0, np.nan)
                
                rolling_coef = rolling_cov / rolling_var
                
                # Calculate spread using rolling hedge ratio
                spread_df['hedge_ratio'] = rolling_coef
                spread_df['spread'] = symbol1_prices - (spread_df['hedge_ratio'] * symbol2_prices)
                
                # Calculate z-score
                spread_df['mean'] = spread_df['spread'].rolling(lookback).mean()
                spread_df['std'] = spread_df['spread'].rolling(lookback).std()
                
                # Avoid division by zero
                spread_df['std'] = spread_df['std'].replace(0, np.nan)
                
                spread_df['zscore'] = (spread_df['spread'] - spread_df['mean']) / spread_df['std']
            except Exception as e:
                logger.error(f"Error calculating spread for {symbol1}_{symbol2}: {e}")
                return None
        else:
            # Use fixed hedge ratio from configuration
            hedge_ratio = config.get("hedge_ratio", 1.0)
            
            # Calculate spread
            spread = symbol1_prices - (hedge_ratio * symbol2_prices)
            
            # Calculate z-score
            spread_mean = spread.rolling(lookback).mean()
            spread_std = spread.rolling(lookback).std()
            
            # Avoid division by zero
            spread_std = spread_std.replace(0, np.nan)
            
            zscore = (spread - spread_mean) / spread_std
            
            # Create spread dataframe
            spread_df = pd.DataFrame({
                'hedge_ratio': hedge_ratio,
                'spread': spread,
                'mean': spread_mean,
                'std': spread_std,
                'zscore': zscore
            })
        
        # Add spread change
        spread_df['spread_change'] = spread_df['spread'].diff()
        
        # Drop rows with NaN values
        spread_df = spread_df.dropna()
        
        logger.info(f"Calculated spread for pair {symbol1}_{symbol2}: {len(spread_df)} rows")
        
        return spread_df 