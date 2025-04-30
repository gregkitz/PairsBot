import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from ..cointegration.cointegration_tests import test_cointegration, test_pairs_universe, calculate_half_life


class PairFinder:
    """
    Class to identify suitable pairs for statistical arbitrage trading.
    Analyzes multiple instruments to find cointegrated pairs that meet trading criteria.
    """
    
    def __init__(self, min_correlation=0.7, max_half_life=20, min_half_life=1, 
                min_cointegration_pct=0.6, train_test_split=0.7):
        """
        Initialize the pair finder.
        
        Parameters:
        -----------
        min_correlation : float
            Minimum correlation threshold for considering a pair
        max_half_life : float
            Maximum half-life for mean reversion (in days)
        min_half_life : float
            Minimum half-life for mean reversion (in days)
        min_cointegration_pct : float
            Minimum percentage of time the pair should be cointegrated
        train_test_split : float
            Proportion of data to use for training vs validation
        """
        self.min_correlation = min_correlation
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.min_cointegration_pct = min_cointegration_pct
        self.train_test_split = train_test_split
    
    def find_pairs(self, price_data, pairs_to_test=None, use_log_prices=True):
        """
        Find pairs suitable for trading from price data.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            DataFrame with price data, each column is a different instrument
        pairs_to_test : list of tuples, optional
            List of ticker pairs to test, if None, test all combinations
        use_log_prices : bool
            Whether to use log prices for analysis
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with pair analysis results, sorted by suitability
        """
        # Generate all pairs if not provided
        if pairs_to_test is None:
            pairs_to_test = list(itertools.combinations(price_data.columns, 2))
        
        results = []
        
        # Test each pair
        for ticker1, ticker2 in pairs_to_test:
            if ticker1 not in price_data.columns or ticker2 not in price_data.columns:
                continue
            
            # Calculate correlation
            if use_log_prices:
                correlation = np.log(price_data[ticker1]).corr(np.log(price_data[ticker2]))
            else:
                correlation = price_data[ticker1].corr(price_data[ticker2])
            
            # Skip pairs with low correlation
            if correlation < self.min_correlation:
                continue
            
            # Run cointegration test
            coint_result = test_cointegration(
                price_data[ticker1], 
                price_data[ticker2],
                test_type='both',
                train_test_split=self.train_test_split,
                use_log_prices=use_log_prices
            )
            
            if coint_result is None or coint_result['overall'] is None:
                continue
            
            # Check if cointegrated in both training and validation
            if not (coint_result['overall']['is_cointegrated_training'] and 
                    coint_result['overall']['is_cointegrated_validation']):
                continue
            
            # Check half-life constraints
            hl_train = coint_result['overall']['half_life_training']
            hl_valid = coint_result['overall']['half_life_validation']
            
            if not (self.min_half_life <= hl_train <= self.max_half_life and 
                    self.min_half_life <= hl_valid <= self.max_half_life):
                continue
            
            # Calculate percentage of time the pair is cointegrated
            training_df = coint_result['training']
            if 'is_cointegrated_eg' not in training_df.columns:
                continue
                
            cointegration_pct = training_df['is_cointegrated_eg'].mean()
            
            if cointegration_pct < self.min_cointegration_pct:
                continue
            
            # If pair passes all filters, add to results
            result_item = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'correlation': correlation,
                'hedge_ratio': coint_result['training']['hedge_ratio'].iloc[-1] if 'hedge_ratio' in coint_result['training'] else None,
                'half_life_train': hl_train,
                'half_life_valid': hl_valid,
                'p_value_train': coint_result['training']['p_value'].iloc[-1] if 'p_value' in coint_result['training'] else None,
                'p_value_valid': coint_result['validation']['valid_p_value'],
                'cointegration_pct': cointegration_pct,
                'johansen_stat': coint_result['training']['johansen_stat'].iloc[-1] if 'johansen_stat' in coint_result['training'] else None,
                'pair_score': self._calculate_pair_score(correlation, hl_train, hl_valid, 
                                                        coint_result['training']['p_value'].iloc[-1] if 'p_value' in coint_result['training'] else 1,
                                                        coint_result['validation']['valid_p_value'],
                                                        cointegration_pct)
            }
            
            results.append(result_item)
        
        # Convert to DataFrame and sort by pair score
        if not results:
            return pd.DataFrame()
            
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('pair_score', ascending=False)
        
        return df_results
    
    def _calculate_pair_score(self, correlation, half_life_train, half_life_valid, 
                             p_value_train, p_value_valid, cointegration_pct):
        """
        Calculate a score for ranking pairs. Higher is better.
        
        Parameters:
        -----------
        correlation : float
            Correlation between the two instruments
        half_life_train : float
            Half-life in training set
        half_life_valid : float
            Half-life in validation set
        p_value_train : float
            P-value from cointegration test in training set
        p_value_valid : float
            P-value from cointegration test in validation set
        cointegration_pct : float
            Percentage of time the pair is cointegrated
            
        Returns:
        --------
        float
            Score for the pair
        """
        # Ideal half-life is in the middle of our range
        ideal_half_life = (self.min_half_life + self.max_half_life) / 2
        
        # Calculate distances from ideal
        hl_train_distance = 1 - min(abs(half_life_train - ideal_half_life) / ideal_half_life, 1)
        hl_valid_distance = 1 - min(abs(half_life_valid - ideal_half_life) / ideal_half_life, 1)
        
        # Combine into score (higher is better)
        p_value_score = 2 - (p_value_train + p_value_valid)  # Lower p-values are better
        
        score = (
            correlation * 0.25 +                   # Reward high correlation
            hl_train_distance * 0.2 +              # Reward half-life close to ideal in training
            hl_valid_distance * 0.2 +              # Reward half-life close to ideal in validation
            p_value_score * 0.2 +                  # Reward low p-values
            cointegration_pct * 0.15               # Reward consistent cointegration
        )
        
        return score
    
    def analyze_pair(self, price1, price2, name1=None, name2=None, use_log_prices=True):
        """
        Perform detailed analysis of a specific pair.
        
        Parameters:
        -----------
        price1 : pandas.Series
            Price series for first instrument
        price2 : pandas.Series
            Price series for second instrument
        name1 : str, optional
            Name of first instrument
        name2 : str, optional
            Name of second instrument
        use_log_prices : bool
            Whether to use log prices
            
        Returns:
        --------
        dict
            Dictionary with detailed analysis results
        """
        # Use series names if names not provided
        if name1 is None and hasattr(price1, 'name'):
            name1 = price1.name
        if name2 is None and hasattr(price2, 'name'):
            name2 = price2.name
        
        # Default names if still None
        name1 = name1 or 'Asset 1'
        name2 = name2 or 'Asset 2'
        
        # Calculate correlation
        if use_log_prices:
            correlation = np.log(price1).corr(np.log(price2))
        else:
            correlation = price1.corr(price2)
        
        # Perform cointegration test
        coint_result = test_cointegration(
            price1, price2,
            test_type='both',
            train_test_split=self.train_test_split,
            use_log_prices=use_log_prices
        )
        
        # Generate statistics
        stats = {
            'pair_names': (name1, name2),
            'correlation': correlation,
            'cointegration_test': coint_result
        }
        
        # Add additional statistics
        if coint_result and 'overall' in coint_result and coint_result['overall']:
            # Extract last hedge ratio
            if 'training' in coint_result and 'hedge_ratio' in coint_result['training']:
                hedge_ratio = coint_result['training']['hedge_ratio'].iloc[-1]
            else:
                # Fallback to OLS calculation
                if use_log_prices:
                    x = np.log(price2)
                    y = np.log(price1)
                else:
                    x = price2
                    y = price1
                
                import statsmodels.api as sm
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                hedge_ratio = model.params[1]
            
            stats['hedge_ratio'] = hedge_ratio
            
            # Calculate spread
            if use_log_prices:
                spread = np.log(price1) - hedge_ratio * np.log(price2)
            else:
                spread = price1 - hedge_ratio * price2
            
            stats['spread'] = spread
            stats['spread_mean'] = spread.mean()
            stats['spread_std'] = spread.std()
            stats['spread_current'] = spread.iloc[-1]
            stats['spread_zscore'] = (spread.iloc[-1] - spread.mean()) / spread.std()
            
            # Calculate half-life
            stats['half_life'] = calculate_half_life(spread)
        
        return stats
    
    def plot_pair_analysis(self, price1, price2, name1=None, name2=None, use_log_prices=True):
        """
        Generate plots for pair analysis.
        
        Parameters:
        -----------
        price1 : pandas.Series
            Price series for first instrument
        price2 : pandas.Series
            Price series for second instrument
        name1 : str, optional
            Name of first instrument
        name2 : str, optional
            Name of second instrument
        use_log_prices : bool
            Whether to use log prices
            
        Returns:
        --------
        tuple
            (figure, axes)
        """
        # Use series names if names not provided
        if name1 is None and hasattr(price1, 'name'):
            name1 = price1.name
        if name2 is None and hasattr(price2, 'name'):
            name2 = price2.name
        
        # Default names if still None
        name1 = name1 or 'Asset 1'
        name2 = name2 or 'Asset 2'
        
        # Analyze the pair
        analysis = self.analyze_pair(price1, price2, name1, name2, use_log_prices)
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Plot original price series (normalized)
        ax1 = axes[0]
        norm_price1 = price1 / price1.iloc[0]
        norm_price2 = price2 / price2.iloc[0]
        
        ax1.plot(norm_price1.index, norm_price1, label=name1)
        ax1.plot(norm_price2.index, norm_price2, label=name2)
        ax1.set_title(f"Normalized Prices: {name1} vs {name2} (Correlation: {analysis['correlation']:.2f})")
        ax1.legend()
        ax1.grid(True)
        
        # Plot spread
        if 'spread' in analysis:
            ax2 = axes[1]
            spread = analysis['spread']
            
            ax2.plot(spread.index, spread)
            ax2.set_title(f"Spread (Half-Life: {analysis['half_life']:.2f} periods)")
            ax2.axhline(y=analysis['spread_mean'], color='r', linestyle='-', alpha=0.3)
            ax2.axhline(y=analysis['spread_mean'] + analysis['spread_std'], color='g', linestyle='--', alpha=0.3)
            ax2.axhline(y=analysis['spread_mean'] - analysis['spread_std'], color='g', linestyle='--', alpha=0.3)
            ax2.axhline(y=analysis['spread_mean'] + 2*analysis['spread_std'], color='y', linestyle='--', alpha=0.3)
            ax2.axhline(y=analysis['spread_mean'] - 2*analysis['spread_std'], color='y', linestyle='--', alpha=0.3)
            ax2.grid(True)
            
            # Plot z-score
            ax3 = axes[2]
            zscore = (spread - spread.rolling(window=20).mean()) / spread.rolling(window=20).std()
            
            ax3.plot(zscore.index, zscore)
            ax3.set_title("Z-Score (Rolling 20 periods)")
            ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax3.axhline(y=1, color='g', linestyle='--', alpha=0.3)
            ax3.axhline(y=-1, color='g', linestyle='--', alpha=0.3)
            ax3.axhline(y=2, color='y', linestyle='--', alpha=0.3)
            ax3.axhline(y=-2, color='y', linestyle='--', alpha=0.3)
            ax3.grid(True)
        
        plt.tight_layout()
        return fig, axes 

    def find_cointegrated_pairs(self, price_data, use_log_prices=True):
        """
        Find cointegrated pairs from price data.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            DataFrame with price data, each column is a different instrument
        use_log_prices : bool
            Whether to use log prices for analysis
            
        Returns:
        --------
        list
            List of dictionaries with pair analysis results
        """
        print(f"Finding cointegrated pairs using find_pairs method...")
        
        # Use existing find_pairs method
        pairs_df = self.find_pairs(price_data, use_log_prices=use_log_prices)
        
        if pairs_df.empty:
            print("No cointegrated pairs found.")
            return []
        
        # Convert DataFrame to list of dictionaries
        pairs_list = []
        for _, row in pairs_df.iterrows():
            pair_dict = {
                'ticker1': row['ticker1'],
                'ticker2': row['ticker2'],
                'correlation': row['correlation'],
                'hedge_ratio': row['hedge_ratio'],
                'half_life': row['half_life_train'],  # Use training half-life
                'p_value': row['p_value_train']       # Use training p-value
            }
            pairs_list.append(pair_dict)
            
            print(f"Found cointegrated pair: {row['ticker1']}-{row['ticker2']} " +
                  f"(p-value: {row['p_value_train']:.4f}, half-life: {row['half_life_train']:.2f} days)")
        
        return pairs_list
    
    def save_pairs_to_json(self, pairs, output_file):
        """
        Save pairs to JSON file.
        
        Parameters:
        -----------
        pairs : list
            List of dictionaries with pair analysis results
        output_file : str
            Path to output file
            
        Returns:
        --------
        None
        """
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create output structure
        output = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'min_correlation': self.min_correlation,
            'min_half_life': self.min_half_life,
            'max_half_life': self.max_half_life,
            'pairs': pairs
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2) 