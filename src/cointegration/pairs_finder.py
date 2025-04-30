"""
Cointegration Pairs Finder Module

This module provides functionality to identify cointegrated pairs of financial instruments
using various statistical tests and methods.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class PairsFinder:
    """
    Identify cointegrated pairs of financial instruments.
    """
    
    def __init__(self, data_dir, output_dir):
        """
        Initialize the PairsFinder with input and output directories.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing processed data files
        output_dir : str
            Directory to save cointegration results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store data by symbol
        self.data = {}
        
        # List of available symbols
        self.symbols = []
    
    def load_data(self, resample_to='1D'):
        """
        Load all processed data files.
        
        Parameters:
        -----------
        resample_to : str, optional
            Resample frequency (e.g., '1D', '1H', etc.)
            
        Returns:
        --------
        list
            List of loaded symbols
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        # Get list of processed files
        all_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('_processed.parquet'):
                all_files.append(file)
        
        logger.info(f"Found {len(all_files)} processed data files")
        
        # Load each file
        for file in all_files:
            try:
                file_path = os.path.join(self.data_dir, file)
                symbol = file.split('_')[0]
                
                # Load data
                df = pd.read_parquet(file_path)
                
                # Ensure data is sorted by timestamp
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)
                    else:
                        logger.warning(f"No timestamp column found in {file}")
                        continue
                
                df.sort_index(inplace=True)
                
                # Resample to specified frequency
                if resample_to:
                    df_resampled = df['close'].resample(resample_to).last()
                    
                    # Filter out NaN and Inf values
                    df_resampled = df_resampled.replace([np.inf, -np.inf], np.nan)
                    df_resampled = df_resampled.dropna()
                    
                    # Ensure enough data points after resampling
                    if len(df_resampled) < 252:  # Minimum one year of daily data
                        logger.warning(f"Not enough data points for {symbol} after resampling to {resample_to}")
                        continue
                    
                    # Check for zero or negative values
                    if (df_resampled <= 0).any():
                        logger.warning(f"Found zero or negative values in {symbol}, applying log to filtered data")
                        df_resampled = df_resampled[df_resampled > 0]
                        
                        # Check again for enough data points
                        if len(df_resampled) < 252:
                            logger.warning(f"Not enough data points for {symbol} after filtering")
                            continue
                    
                    self.data[symbol] = df_resampled
                else:
                    # Filter out NaN and Inf values
                    df_filtered = df['close'].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # Check for zero or negative values
                    if (df_filtered <= 0).any():
                        logger.warning(f"Found zero or negative values in {symbol}, applying log to filtered data")
                        df_filtered = df_filtered[df_filtered > 0]
                    
                    self.data[symbol] = df_filtered
                
                self.symbols.append(symbol)
                logger.info(f"Loaded {len(self.data[symbol])} data points for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        logger.info(f"Loaded data for {len(self.symbols)} symbols")
        return self.symbols
    
    def calculate_half_life(self, spread):
        """
        Calculate the half-life of mean reversion for a spread series.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread series to analyze
            
        Returns:
        --------
        float
            Half-life of mean reversion in days
        """
        # Filter out any NaN values
        spread = spread.dropna()
        
        # Check if we have enough data points
        if len(spread) < 20:
            logger.warning("Not enough data points to calculate half-life")
            return np.nan
        
        try:
            # Calculate lagged spread and delta
            spread_lag = spread.shift(1)
            delta_spread = spread - spread_lag
            spread_lag = spread_lag.dropna()
            delta_spread = delta_spread.dropna()
            
            # Regression delta_spread on spread_lag
            spread_lag = sm.add_constant(spread_lag)
            model = sm.OLS(delta_spread, spread_lag)
            res = model.fit()
            
            # Calculate half-life
            half_life = -np.log(2) / res.params[1]
            
            # Return half-life if it's positive and finite
            if np.isfinite(half_life) and half_life > 0:
                return half_life
            else:
                return np.nan
        except Exception as e:
            logger.warning(f"Error calculating half-life: {e}")
            return np.nan
    
    def test_cointegration(self, symbol1, symbol2, lookback_days=252, method='engle-granger'):
        """
        Test for cointegration between two symbols.
        
        Parameters:
        -----------
        symbol1 : str
            First symbol
        symbol2 : str
            Second symbol
        lookback_days : int, optional
            Number of days to look back for cointegration test
        method : str, optional
            Cointegration test method ('engle-granger' or 'johansen')
            
        Returns:
        --------
        dict
            Cointegration test results
        """
        logger.info(f"Testing cointegration between {symbol1} and {symbol2}")
        
        # Get data for both symbols
        if symbol1 not in self.data or symbol2 not in self.data:
            logger.warning(f"Data not available for {symbol1} or {symbol2}")
            return None
        
        # Get price series
        price1 = self.data[symbol1]
        price2 = self.data[symbol2]
        
        # Ensure both series have the same index
        common_idx = price1.index.intersection(price2.index)
        if len(common_idx) < lookback_days:
            logger.warning(f"Not enough common data points for {symbol1} and {symbol2}")
            return None
        
        price1 = price1[common_idx]
        price2 = price2[common_idx]
        
        # Use only the most recent lookback_days
        if len(price1) > lookback_days:
            price1 = price1[-lookback_days:]
            price2 = price2[-lookback_days:]
        
        # Check for NaN or infinite values
        if np.isnan(price1).any() or np.isnan(price2).any() or np.isinf(price1).any() or np.isinf(price2).any():
            logger.warning(f"Found NaN or infinite values in {symbol1} or {symbol2}")
            
            # Filter out NaN and Inf values
            valid_idx = ~(np.isnan(price1) | np.isnan(price2) | np.isinf(price1) | np.isinf(price2))
            price1 = price1[valid_idx]
            price2 = price2[valid_idx]
            
            # Check if we still have enough data points
            if len(price1) < lookback_days * 0.8:  # Allow 20% missing data
                logger.warning(f"Not enough valid data points for {symbol1} and {symbol2} after filtering")
                return None
        
        # Check for zero or negative values
        if (price1 <= 0).any() or (price2 <= 0).any():
            logger.warning(f"Found zero or negative values in {symbol1} or {symbol2}")
            valid_idx = (price1 > 0) & (price2 > 0)
            price1 = price1[valid_idx]
            price2 = price2[valid_idx]
            
            # Check if we still have enough data points
            if len(price1) < lookback_days * 0.8:  # Allow 20% missing data
                logger.warning(f"Not enough valid data points for {symbol1} and {symbol2} after filtering positives")
                return None
        
        try:
            # Convert to log prices
            log_price1 = np.log(price1)
            log_price2 = np.log(price2)
            
            results = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'lookback_days': lookback_days,
                'start_date': price1.index[0].strftime('%Y-%m-%d'),
                'end_date': price1.index[-1].strftime('%Y-%m-%d'),
                'method': method,
                'num_observations': len(price1)
            }
            
            # Perform cointegration test
            if method == 'engle-granger':
                # Engle-Granger test
                eg_result = coint(log_price1, log_price2)
                
                results['t_statistic'] = eg_result[0]
                results['p_value'] = eg_result[1]
                results['critical_values'] = {
                    '1%': eg_result[2][0],
                    '5%': eg_result[2][1],
                    '10%': eg_result[2][2]
                }
                results['is_cointegrated'] = results['p_value'] < 0.05
                
                # Estimate hedge ratio using OLS
                X = sm.add_constant(log_price2)
                model = sm.OLS(log_price1, X).fit()
                hedge_ratio = model.params[1]
                
                results['hedge_ratio'] = hedge_ratio
                
                # Calculate spread
                spread = log_price1 - hedge_ratio * log_price2
                
                # Calculate half-life
                half_life = self.calculate_half_life(spread)
                results['half_life'] = half_life
                
                # Calculate spread metrics
                results['spread_mean'] = spread.mean()
                results['spread_std'] = spread.std()
                results['spread_min'] = spread.min()
                results['spread_max'] = spread.max()
                
                # Calculate correlation
                results['correlation'] = log_price1.corr(log_price2)
                
            elif method == 'johansen':
                # Johansen test - to be implemented
                logger.warning("Johansen method not fully implemented yet")
                
                # For now, just use Engle-Granger and note the method as Johansen
                eg_result = coint(log_price1, log_price2)
                
                results['t_statistic'] = eg_result[0]
                results['p_value'] = eg_result[1]
                results['is_cointegrated'] = results['p_value'] < 0.05
                
                # Placeholder for Johansen-specific results
                results['johansen_implemented'] = False
            
            else:
                logger.error(f"Unknown cointegration test method: {method}")
                return None
            
            logger.info(f"Cointegration test results for {symbol1}-{symbol2}: " + 
                      f"p-value={results['p_value']:.4f}, " + 
                      f"is_cointegrated={results['is_cointegrated']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing {symbol1}-{symbol2}: {e}")
            return None
    
    def find_pairs(self, lookback_days=252, p_value_threshold=0.05, min_half_life=1, max_half_life=30, 
                   min_correlation=0.5, max_pairs=None, method='engle-granger'):
        """
        Find all cointegrated pairs among the loaded symbols.
        
        Parameters:
        -----------
        lookback_days : int, optional
            Number of days to look back for cointegration test
        p_value_threshold : float, optional
            P-value threshold for cointegration test
        min_half_life : float, optional
            Minimum half-life in days
        max_half_life : float, optional
            Maximum half-life in days
        min_correlation : float, optional
            Minimum correlation between symbols
        max_pairs : int, optional
            Maximum number of pairs to test
        method : str, optional
            Cointegration test method ('engle-granger' or 'johansen')
            
        Returns:
        --------
        list
            List of cointegrated pairs
        """
        logger.info(f"Finding cointegrated pairs among {len(self.symbols)} symbols")
        
        # Generate all possible pairs
        all_pairs = list(itertools.combinations(self.symbols, 2))
        
        logger.info(f"Total possible pairs: {len(all_pairs)}")
        
        # Limit number of pairs if requested
        if max_pairs is not None and len(all_pairs) > max_pairs:
            all_pairs = all_pairs[:max_pairs]
            logger.info(f"Limited to testing {max_pairs} pairs")
        
        results = []
        
        # Test each pair
        for symbol1, symbol2 in all_pairs:
            try:
                pair_result = self.test_cointegration(symbol1, symbol2, lookback_days, method)
                
                if pair_result is not None:
                    # Check if pair passes filters
                    if pair_result['is_cointegrated']:
                        # Check half-life
                        half_life = pair_result.get('half_life', np.nan)
                        if half_life is not None and np.isfinite(half_life):
                            if half_life < min_half_life or half_life > max_half_life:
                                continue
                        
                        # Check correlation
                        correlation = pair_result.get('correlation', 0)
                        if correlation < min_correlation:
                            continue
                        
                        results.append(pair_result)
                        logger.info(f"Found cointegrated pair: {symbol1}-{symbol2}")
            
            except Exception as e:
                logger.error(f"Error testing {symbol1}-{symbol2}: {e}")
        
        logger.info(f"Found {len(results)} cointegrated pairs")
        
        # Sort by p-value
        results.sort(key=lambda x: x['p_value'])
        
        return results
    
    def save_results(self, results, filename=None):
        """
        Save cointegration results to a file.
        
        Parameters:
        -----------
        results : list
            List of cointegration test results
        filename : str, optional
            Name of the file to save results
            
        Returns:
        --------
        str
            Path to the saved file
        """
        if not results:
            logger.warning("No results to save")
            return None
        
        if filename is None:
            filename = f"cointegration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            # Create a new dictionary with serializable values
            serializable_result = {}
            for key, value in result.items():
                # Handle numpy types
                if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    serializable_result[key] = value.item()
                # Handle boolean values
                elif isinstance(value, bool):
                    serializable_result[key] = int(value)
                # Handle dictionary with numpy values
                elif isinstance(value, dict):
                    serializable_result[key] = {k: v.item() if isinstance(v, (np.int64, np.int32, np.float64, np.float32)) else v for k, v in value.items()}
                # Handle other types
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        # Save results as JSON
        json_path = os.path.join(self.output_dir, filename.replace('.csv', '.json'))
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved cointegration results to {json_path}")
        
        # Save as CSV as well (more convenient for analysis)
        csv_path = os.path.join(self.output_dir, filename.replace('.json', '.csv'))
        
        # Convert to DataFrame for CSV output
        df = pd.DataFrame(results)
        
        # If critical_values is in the results, expand it
        if 'critical_values' in df.columns:
            df['critical_value_1pct'] = df['critical_values'].apply(lambda x: x['1%'] if isinstance(x, dict) else None)
            df['critical_value_5pct'] = df['critical_values'].apply(lambda x: x['5%'] if isinstance(x, dict) else None)
            df['critical_value_10pct'] = df['critical_values'].apply(lambda x: x['10%'] if isinstance(x, dict) else None)
            df.drop('critical_values', axis=1, inplace=True)
        
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved cointegration results to {csv_path}")
        
        return json_path


# Example usage
if __name__ == "__main__":
    # Set input and output directories
    data_dir = r"data\processed"
    output_dir = r"data\results"
    
    # Create pairs finder
    finder = PairsFinder(data_dir, output_dir)
    
    # Load data
    finder.load_data(resample_to='1D')
    
    # Find cointegrated pairs
    results = finder.find_pairs(
        lookback_days=252,
        p_value_threshold=0.05,
        min_half_life=1,
        max_half_life=30,
        min_correlation=0.5,
        max_pairs=100,
        method='engle-granger'
    )
    
    # Save results
    finder.save_results(results, filename="cointegration_results.json")
    
    print(f"Found {len(results)} cointegrated pairs")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['symbol1']}-{result['symbol2']}: " + 
              f"p-value={result['p_value']:.4f}, " + 
              f"half-life={result.get('half_life', 'N/A'):.2f} days") 