"""
Script to analyze the cointegrated pairs in detail.

This will load the processed data for the cointegrated pairs and perform deeper analysis
on their relationship, spread properties, and trading potential.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class PairsAnalyzer:
    """
    Analyze cointegrated pairs in detail.
    """
    
    def __init__(self, data_dir, output_dir):
        """
        Initialize the PairsAnalyzer with input and output directories.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing processed data files
        output_dir : str
            Directory to save analysis results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store data by symbol
        self.data = {}
        
        # List of cointegrated pairs to analyze
        self.pairs = []
    
    def load_data(self, symbols=None, resample_to='1D'):
        """
        Load data for selected symbols.
        
        Parameters:
        -----------
        symbols : list, optional
            List of symbols to load data for. If None, load all available symbols.
        resample_to : str, optional
            Resample frequency (e.g., '1D', '1H', etc.)
            
        Returns:
        --------
        dict
            Dictionary of loaded data by symbol
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        # Get list of processed files
        all_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('_processed.parquet'):
                symbol = file.split('_')[0]
                if symbols is None or symbol in symbols:
                    all_files.append(file)
        
        logger.info(f"Found {len(all_files)} processed data files for selected symbols")
        
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
                    df_resampled = df_resampled.dropna()
                    self.data[symbol] = df_resampled
                else:
                    self.data[symbol] = df['close']
                
                logger.info(f"Loaded {len(self.data[symbol])} data points for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        logger.info(f"Loaded data for {len(self.data)} symbols")
        return self.data
    
    def load_pairs(self, pairs_file):
        """
        Load cointegrated pairs from a JSON file.
        
        Parameters:
        -----------
        pairs_file : str
            Path to the JSON file containing cointegrated pairs
            
        Returns:
        --------
        list
            List of cointegrated pairs
        """
        try:
            with open(pairs_file, 'r') as f:
                self.pairs = json.load(f)
            
            logger.info(f"Loaded {len(self.pairs)} cointegrated pairs from {pairs_file}")
            
            # Extract unique symbols from pairs
            symbols = set()
            for pair in self.pairs:
                symbols.add(pair['symbol1'])
                symbols.add(pair['symbol2'])
            
            logger.info(f"Found {len(symbols)} unique symbols in the pairs")
            
            # Load data for these symbols
            self.load_data(symbols)
            
            return self.pairs
            
        except Exception as e:
            logger.error(f"Error loading pairs from {pairs_file}: {e}")
            return []
    
    def create_manual_pairs_list(self):
        """
        Create a manual list of the pairs we identified previously.
        
        Returns:
        --------
        list
            List of manually defined cointegrated pairs
        """
        # These are the pairs we identified from our previous cointegration testing
        self.pairs = [
            {'symbol1': 'ALI', 'symbol2': 'ES', 'p_value': 0.0124},
            {'symbol1': 'ALI', 'symbol2': 'NQ', 'p_value': 0.0135},
            {'symbol1': 'ALI', 'symbol2': 'SI', 'p_value': 0.0069},
            {'symbol1': 'ALI', 'symbol2': 'YM', 'p_value': 0.0254},
            {'symbol1': 'ALI', 'symbol2': 'ZF', 'p_value': 0.0497},
            {'symbol1': 'ALI', 'symbol2': 'ZN', 'p_value': 0.0497},
            {'symbol1': 'BFX', 'symbol2': 'HG', 'p_value': 0.0329},
            {'symbol1': 'BFX', 'symbol2': 'RTY', 'p_value': 0.0109},
            {'symbol1': 'BFX', 'symbol2': 'ZC', 'p_value': 0.0475},
            {'symbol1': 'BFX', 'symbol2': 'ZN', 'p_value': 0.0100},
            {'symbol1': 'BFX', 'symbol2': 'ZT', 'p_value': 0.0249},
            {'symbol1': 'BFX', 'symbol2': 'ZW', 'p_value': 0.0305}
        ]
        
        # Extract unique symbols from pairs
        symbols = set()
        for pair in self.pairs:
            symbols.add(pair['symbol1'])
            symbols.add(pair['symbol2'])
        
        logger.info(f"Created list of {len(self.pairs)} cointegrated pairs with {len(symbols)} unique symbols")
        
        # Load data for these symbols
        self.load_data(symbols)
        
        return self.pairs
    
    def estimate_hedge_ratio(self, symbol1, symbol2, lookback_days=252):
        """
        Estimate the hedge ratio between two symbols using OLS regression on log prices.
        
        Parameters:
        -----------
        symbol1 : str
            First symbol (dependent variable)
        symbol2 : str
            Second symbol (independent variable)
        lookback_days : int, optional
            Number of days to look back for regression
            
        Returns:
        --------
        float
            Estimated hedge ratio
        """
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
        
        try:
            # Convert to log prices
            log_price1 = np.log(price1)
            log_price2 = np.log(price2)
            
            # Estimate hedge ratio using OLS
            X = sm.add_constant(log_price2)
            model = sm.OLS(log_price1, X).fit()
            hedge_ratio = model.params[1]
            
            return hedge_ratio
            
        except Exception as e:
            logger.error(f"Error estimating hedge ratio for {symbol1}-{symbol2}: {e}")
            return None
    
    def calculate_spread(self, symbol1, symbol2, hedge_ratio=None, lookback_days=252):
        """
        Calculate the spread between two symbols.
        
        Parameters:
        -----------
        symbol1 : str
            First symbol
        symbol2 : str
            Second symbol
        hedge_ratio : float, optional
            Hedge ratio to use. If None, estimate using OLS.
        lookback_days : int, optional
            Number of days to look back for spread calculation
            
        Returns:
        --------
        pd.Series
            Spread series
        float
            Hedge ratio used
        """
        # Get data for both symbols
        if symbol1 not in self.data or symbol2 not in self.data:
            logger.warning(f"Data not available for {symbol1} or {symbol2}")
            return None, None
        
        # Get price series
        price1 = self.data[symbol1]
        price2 = self.data[symbol2]
        
        # Ensure both series have the same index
        common_idx = price1.index.intersection(price2.index)
        if len(common_idx) < lookback_days:
            logger.warning(f"Not enough common data points for {symbol1} and {symbol2}")
            return None, None
        
        price1 = price1[common_idx]
        price2 = price2[common_idx]
        
        # Use only the most recent lookback_days
        if len(price1) > lookback_days:
            price1 = price1[-lookback_days:]
            price2 = price2[-lookback_days:]
        
        try:
            # Convert to log prices
            log_price1 = np.log(price1)
            log_price2 = np.log(price2)
            
            # Estimate hedge ratio if not provided
            if hedge_ratio is None:
                X = sm.add_constant(log_price2)
                model = sm.OLS(log_price1, X).fit()
                hedge_ratio = model.params[1]
            
            # Calculate spread
            spread = log_price1 - hedge_ratio * log_price2
            
            return spread, hedge_ratio
            
        except Exception as e:
            logger.error(f"Error calculating spread for {symbol1}-{symbol2}: {e}")
            return None, None
    
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
            half_life = -np.log(2) / res.params.iloc[1]
            
            # Return half-life if it's positive and finite
            if np.isfinite(half_life) and half_life > 0:
                return half_life
            else:
                return np.nan
        except Exception as e:
            logger.warning(f"Error calculating half-life: {e}")
            return np.nan
    
    def analyze_pair(self, symbol1, symbol2, lookback_days=252):
        """
        Analyze a pair of symbols in detail.
        
        Parameters:
        -----------
        symbol1 : str
            First symbol
        symbol2 : str
            Second symbol
        lookback_days : int, optional
            Number of days to look back for analysis
            
        Returns:
        --------
        dict
            Analysis results
        """
        logger.info(f"Analyzing pair {symbol1}-{symbol2}")
        
        # Calculate spread and hedge ratio
        spread, hedge_ratio = self.calculate_spread(symbol1, symbol2, lookback_days=lookback_days)
        
        if spread is None or hedge_ratio is None:
            logger.warning(f"Could not calculate spread for {symbol1}-{symbol2}")
            return None
        
        # Calculate half-life
        half_life = self.calculate_half_life(spread)
        
        # Calculate spread statistics
        spread_mean = spread.mean()
        spread_std = spread.std()
        spread_min = spread.min()
        spread_max = spread.max()
        
        # Calculate z-scores
        z_score = (spread - spread_mean) / spread_std
        
        # Calculate correlation
        if symbol1 in self.data and symbol2 in self.data:
            price1 = self.data[symbol1]
            price2 = self.data[symbol2]
            
            # Ensure both series have the same index
            common_idx = price1.index.intersection(price2.index)
            price1 = price1[common_idx]
            price2 = price2[common_idx]
            
            # Use only the most recent lookback_days
            if len(price1) > lookback_days:
                price1 = price1[-lookback_days:]
                price2 = price2[-lookback_days:]
            
            # Calculate log returns
            log_returns1 = np.log(price1 / price1.shift(1)).dropna()
            log_returns2 = np.log(price2 / price2.shift(1)).dropna()
            
            # Ensure both series have the same index
            common_idx = log_returns1.index.intersection(log_returns2.index)
            log_returns1 = log_returns1[common_idx]
            log_returns2 = log_returns2[common_idx]
            
            # Calculate correlation
            correlation = log_returns1.corr(log_returns2)
        else:
            correlation = np.nan
        
        # Calculate potential trades
        z_entry = 2.0
        z_exit = 0.5
        
        long_entries = (z_score < -z_entry).astype(int)
        long_exits = (z_score > -z_exit).astype(int)
        
        short_entries = (z_score > z_entry).astype(int)
        short_exits = (z_score < z_exit).astype(int)
        
        # Calculate number of potential trades
        long_trades = 0
        short_trades = 0
        
        in_long = False
        in_short = False
        
        for i in range(1, len(z_score)):
            # Long trades
            if not in_long and long_entries.iloc[i]:
                in_long = True
            elif in_long and long_exits.iloc[i]:
                in_long = False
                long_trades += 1
            
            # Short trades
            if not in_short and short_entries.iloc[i]:
                in_short = True
            elif in_short and short_exits.iloc[i]:
                in_short = False
                short_trades += 1
        
        total_trades = long_trades + short_trades
        
        # Create analysis results
        results = {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'hedge_ratio': hedge_ratio,
            'half_life': half_life,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_min': spread_min,
            'spread_max': spread_max,
            'correlation': correlation,
            'total_trades': total_trades,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'lookback_days': lookback_days,
            'start_date': spread.index[0].strftime('%Y-%m-%d'),
            'end_date': spread.index[-1].strftime('%Y-%m-%d')
        }
        
        # Save the spread and z-score for plotting
        results['spread'] = spread
        results['z_score'] = z_score
        
        logger.info(f"Analysis results for {symbol1}-{symbol2}: " + 
                  f"hedge_ratio={hedge_ratio:.4f}, " +
                  f"half_life={half_life:.2f} days, " +
                  f"correlation={correlation:.4f}, " +
                  f"trades={total_trades}")
        
        return results
    
    def analyze_all_pairs(self, lookback_days=252):
        """
        Analyze all cointegrated pairs.
        
        Parameters:
        -----------
        lookback_days : int, optional
            Number of days to look back for analysis
            
        Returns:
        --------
        list
            List of analysis results for all pairs
        """
        logger.info(f"Analyzing {len(self.pairs)} cointegrated pairs")
        
        results = []
        
        for pair in self.pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            
            # Check if both symbols exist in data
            if symbol1 not in self.data or symbol2 not in self.data:
                logger.warning(f"Data not available for {symbol1} or {symbol2}, skipping")
                continue
                
            # Analyze pair
            pair_result = self.analyze_pair(symbol1, symbol2, lookback_days)
            
            if pair_result is not None:
                results.append(pair_result)
        
        logger.info(f"Analyzed {len(results)} pairs successfully")
        
        # Sort by half-life
        results.sort(key=lambda x: x['half_life'])
        
        return results
    
    def plot_pair(self, pair_result, output_file=None):
        """
        Plot the spread and z-score for a pair.
        
        Parameters:
        -----------
        pair_result : dict
            Analysis results for a pair
        output_file : str, optional
            Path to save the plot. If None, display the plot.
            
        Returns:
        --------
        None
        """
        symbol1 = pair_result['symbol1']
        symbol2 = pair_result['symbol2']
        spread = pair_result['spread']
        z_score = pair_result['z_score']
        
        # Create a plot with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot spread
        ax1.plot(spread, label=f'Spread ({symbol1} - {pair_result["hedge_ratio"]:.4f} * {symbol2})')
        ax1.axhline(y=pair_result['spread_mean'], color='r', linestyle='-', label='Mean')
        ax1.axhline(y=pair_result['spread_mean'] + pair_result['spread_std'], color='g', linestyle='--', label='Mean + 1 Std')
        ax1.axhline(y=pair_result['spread_mean'] - pair_result['spread_std'], color='g', linestyle='--', label='Mean - 1 Std')
        ax1.set_title(f'Spread between {symbol1} and {symbol2}')
        ax1.set_ylabel('Spread')
        ax1.legend()
        ax1.grid(True)
        
        # Plot z-score
        ax2.plot(z_score, label='Z-Score')
        ax2.axhline(y=0, color='r', linestyle='-')
        ax2.axhline(y=1, color='g', linestyle='--')
        ax2.axhline(y=-1, color='g', linestyle='--')
        ax2.axhline(y=2, color='y', linestyle='--')
        ax2.axhline(y=-2, color='y', linestyle='--')
        ax2.set_title(f'Z-Score (Half-Life: {pair_result["half_life"]:.2f} days)')
        ax2.set_ylabel('Z-Score')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True)
        
        # Set title
        fig.suptitle(f'{symbol1}-{symbol2} Pair Analysis\n' +
                    f'Hedge Ratio: {pair_result["hedge_ratio"]:.4f}, Half-Life: {pair_result["half_life"]:.2f} days, ' +
                    f'Correlation: {pair_result["correlation"]:.4f}, Trades: {pair_result["total_trades"]}',
                    fontsize=14)
        
        plt.tight_layout()
        
        # Save or display plot
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def save_results(self, results, filename=None):
        """
        Save analysis results to a file.
        
        Parameters:
        -----------
        results : list
            List of analysis results for all pairs
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
            filename = f"pairs_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create a copy of results without the Series objects (spread and z_score)
        serializable_results = []
        for result in results:
            serializable_result = {k: v for k, v in result.items() if k not in ['spread', 'z_score']}
            serializable_results.append(serializable_result)
        
        # Save results as JSON
        json_path = os.path.join(self.output_dir, filename.replace('.csv', '.json'))
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved analysis results to {json_path}")
        
        # Save as CSV as well (more convenient for analysis)
        csv_path = os.path.join(self.output_dir, filename.replace('.json', '.csv'))
        
        # Convert to DataFrame for CSV output
        df = pd.DataFrame(serializable_results)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved analysis results to {csv_path}")
        
        return json_path
    
    def generate_plots(self, results, output_dir=None):
        """
        Generate plots for all pairs.
        
        Parameters:
        -----------
        results : list
            List of analysis results for all pairs
        output_dir : str, optional
            Directory to save the plots. If None, use self.output_dir.
            
        Returns:
        --------
        list
            List of paths to the saved plots
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'plots')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Generating plots for {len(results)} pairs")
        
        plot_paths = []
        
        for result in results:
            symbol1 = result['symbol1']
            symbol2 = result['symbol2']
            
            # Create output file path
            output_file = os.path.join(output_dir, f"{symbol1}_{symbol2}_pair_analysis.png")
            
            # Plot pair
            self.plot_pair(result, output_file)
            
            plot_paths.append(output_file)
        
        logger.info(f"Generated {len(plot_paths)} plots")
        
        return plot_paths

if __name__ == "__main__":
    # Set input and output directories
    data_dir = r"data\processed"
    output_dir = r"data\results"
    
    # Create analyzer
    analyzer = PairsAnalyzer(data_dir, output_dir)
    
    # Load pairs from JSON file or create manual list
    pairs = analyzer.create_manual_pairs_list()
    
    # Analyze all pairs
    results = analyzer.analyze_all_pairs(lookback_days=252)
    
    # Save results
    analyzer.save_results(results, filename="pairs_analysis.json")
    
    # Generate plots
    analyzer.generate_plots(results)
    
    # Print results
    print(f"Analyzed {len(results)} pairs")
    print("Top 5 pairs by half-life:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['symbol1']}-{result['symbol2']}: " + 
              f"half-life={result['half_life']:.2f} days, " + 
              f"hedge_ratio={result['hedge_ratio']:.4f}, " + 
              f"correlation={result['correlation']:.4f}, " + 
              f"trades={result['total_trades']}") 