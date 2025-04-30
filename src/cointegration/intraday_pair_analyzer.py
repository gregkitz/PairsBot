"""
Intraday Pair Analyzer Module

This module provides specialized functionality for analyzing cointegration relationships
at intraday timeframes, with specific adaptations for high-frequency data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
import json

from src.cointegration.cointegration_tests import calculate_half_life, test_cointegration
from src.data_processor.intraday_processor import IntradayDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntradayPairAnalyzer:
    """
    Analyze cointegration relationships at intraday timeframes.
    """
    
    def __init__(self, data_dir="data/processed", output_dir="output/intraday_analysis"):
        """
        Initialize the IntradayPairAnalyzer.
        
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
        
        # Initialize data processor
        self.data_processor = IntradayDataProcessor(data_dir)
        
        # Cache for cointegration test results
        self.cointegration_cache = {}
    
    def analyze_pair_stationarity(self, pair_id, start_date, end_date, timeframe="5min"):
        """
        Analyze the stationarity of a pair's spread.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier (e.g., "ES_NQ")
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        timeframe : str
            Timeframe for analysis
            
        Returns:
        --------
        dict
            Dictionary with stationarity analysis results
        """
        logger.info(f"Analyzing stationarity for pair {pair_id}")
        
        # Load pair data
        pair_data = self.data_processor.load_pair_data(pair_id, start_date, end_date, timeframe)
        
        if pair_data is None or pair_data['prices'] is None:
            logger.error(f"Failed to load data for pair {pair_id}")
            return None
        
        # Extract symbols
        symbol1, symbol2 = pair_id.split('_')
        
        # Extract price data
        price_data = pair_data['prices']
        
        # Test stationarity for each symbol
        results = {}
        
        # ADF test for each symbol
        for symbol, prices in price_data.items():
            adf_result = adfuller(prices.dropna())
            results[symbol] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
        
        # Test for cointegration
        test_result = test_cointegration(
            price_data[symbol1], 
            price_data[symbol2], 
            window=min(50, len(price_data) // 4),  # Shorter window for intraday
            test_type='both'
        )
        
        results['cointegration'] = test_result
        
        return results
    
    def analyze_pair_stability(self, pair_id, start_date, end_date, timeframe="5min", window_minutes=120):
        """
        Analyze the stability of a pair's cointegration relationship throughout the trading day.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier (e.g., "ES_NQ")
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        timeframe : str
            Timeframe for analysis
        window_minutes : int
            Rolling window size in minutes
            
        Returns:
        --------
        dict
            Dictionary with stability analysis results
        """
        logger.info(f"Analyzing stability for pair {pair_id}")
        
        # Load pair data
        pair_data = self.data_processor.load_pair_data(pair_id, start_date, end_date, timeframe)
        
        if pair_data is None or pair_data['prices'] is None:
            logger.error(f"Failed to load data for pair {pair_id}")
            return None
        
        # Extract symbols
        symbol1, symbol2 = pair_id.split('_')
        
        # Extract price data
        price_data = pair_data['prices']
        
        # Convert window_minutes to number of periods
        window = window_minutes // int(timeframe[:-3])
        
        # Calculate rolling p-values for cointegration
        symbol1_prices = price_data[symbol1]
        symbol2_prices = price_data[symbol2]
        
        dates = []
        p_values = []
        hedge_ratios = []
        half_lives = []
        
        for i in range(window, len(price_data)):
            try:
                # Get window of data
                y_window = symbol1_prices.iloc[i-window:i]
                x_window = symbol2_prices.iloc[i-window:i]
                
                # Calculate regression for this window
                X = sm.add_constant(x_window)
                model = sm.OLS(y_window, X).fit()
                
                # Get hedge ratio
                hedge_ratio = model.params[1]
                
                # Calculate residuals
                residuals = y_window - (model.params[0] + hedge_ratio * x_window)
                
                # ADF test on residuals
                adf_result = adfuller(residuals)
                
                # Calculate half-life
                half_life = calculate_half_life(residuals)
                
                # Store results
                dates.append(price_data.index[i])
                p_values.append(adf_result[1])
                hedge_ratios.append(hedge_ratio)
                half_lives.append(half_life)
            except Exception as e:
                logger.warning(f"Error in window {i}: {e}")
                continue
        
        # Create results dataframe
        stability_df = pd.DataFrame({
            'timestamp': dates,
            'p_value': p_values,
            'hedge_ratio': hedge_ratios,
            'half_life': half_lives,
            'is_cointegrated': [p < 0.05 for p in p_values]
        })
        
        if stability_df.empty:
            logger.error("Failed to calculate stability metrics")
            return None
        
        # Set timestamp as index
        stability_df.set_index('timestamp', inplace=True)
        
        # Calculate time-of-day stability
        tod_stability = stability_df.groupby(stability_df.index.hour).agg({
            'p_value': 'mean',
            'is_cointegrated': 'mean',
            'half_life': 'median',
            'hedge_ratio': ['mean', 'std']
        })
        
        # Calculate overall stability metrics
        stability_metrics = {
            'cointegration_frequency': stability_df['is_cointegrated'].mean(),
            'median_half_life': stability_df['half_life'].median(),
            'hedge_ratio_mean': stability_df['hedge_ratio'].mean(),
            'hedge_ratio_std': stability_df['hedge_ratio'].std(),
            'hedge_ratio_stability': 1 - (stability_df['hedge_ratio'].std() / abs(stability_df['hedge_ratio'].mean())) if abs(stability_df['hedge_ratio'].mean()) > 0 else 0,
            'tod_stability': tod_stability.to_dict()
        }
        
        return {
            'stability_metrics': stability_metrics,
            'stability_df': stability_df
        }
    
    def calculate_time_based_metrics(self, pair_id, start_date, end_date, timeframe="5min"):
        """
        Calculate time-of-day specific metrics for a pair.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier (e.g., "ES_NQ")
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        timeframe : str
            Timeframe for analysis
            
        Returns:
        --------
        dict
            Dictionary with time-based metrics
        """
        logger.info(f"Calculating time-based metrics for pair {pair_id}")
        
        # Load pair data
        pair_data = self.data_processor.load_pair_data(pair_id, start_date, end_date, timeframe)
        
        if pair_data is None or pair_data['prices'] is None:
            logger.error(f"Failed to load data for pair {pair_id}")
            return None
        
        # Extract symbols
        symbol1, symbol2 = pair_id.split('_')
        
        # Get spread using a fixed hedge ratio for simplicity
        prices = pair_data['prices']
        hedge_ratio = 1.0  # Simplified for analysis
        
        spread = prices[symbol1] - (hedge_ratio * prices[symbol2])
        
        # Calculate time-of-day metrics
        spread_df = pd.DataFrame({
            'spread': spread,
            'hour': spread.index.hour,
            'minute': spread.index.minute
        })
        
        # Spread change by hour
        spread_df['spread_change'] = spread_df['spread'].diff()
        spread_volatility = spread_df.groupby('hour')['spread_change'].agg(['std', 'count'])
        spread_volatility.columns = ['volatility', 'count']
        
        # Mean reversion strength by hour
        mean_reversion = []
        
        for hour in range(9, 17):  # Market hours
            hour_data = spread_df[spread_df['hour'] == hour]['spread']
            
            if len(hour_data) > 30:  # Minimum data points
                half_life = calculate_half_life(hour_data)
                mean_reversion.append({'hour': hour, 'half_life': half_life})
        
        mr_df = pd.DataFrame(mean_reversion)
        if not mr_df.empty:
            mr_df.set_index('hour', inplace=True)
        
        # Correlations by hour
        hourly_correlations = []
        
        for hour in range(9, 17):  # Market hours
            hour_prices = pair_data['prices'][pair_data['prices'].index.hour == hour]
            
            if len(hour_prices) > 30:  # Minimum data points
                corr = hour_prices[symbol1].corr(hour_prices[symbol2])
                hourly_correlations.append({'hour': hour, 'correlation': corr})
        
        corr_df = pd.DataFrame(hourly_correlations)
        if not corr_df.empty:
            corr_df.set_index('hour', inplace=True)
        
        return {
            'spread_volatility': spread_volatility,
            'mean_reversion': mr_df,
            'correlations': corr_df
        }
    
    def visualize_intraday_stability(self, pair_id, stability_result, output_file=None):
        """
        Visualize the intraday stability of a pair.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier
        stability_result : dict
            Result from analyze_pair_stability
        output_file : str, optional
            Path to save the visualization
            
        Returns:
        --------
        str
            Path to the saved visualization
        """
        if stability_result is None:
            logger.error("No stability results to visualize")
            return None
        
        stability_df = stability_result['stability_df']
        metrics = stability_result['stability_metrics']
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: P-values over time
        axes[0].plot(stability_df.index, stability_df['p_value'], label='P-value')
        axes[0].axhline(y=0.05, color='r', linestyle='--', label='5% Threshold')
        axes[0].set_ylabel('P-value')
        axes[0].set_title(f'Cointegration P-values - {pair_id}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Hedge ratio over time
        axes[1].plot(stability_df.index, stability_df['hedge_ratio'], label='Hedge Ratio')
        axes[1].axhline(y=metrics['hedge_ratio_mean'], color='g', linestyle='--', label='Mean')
        axes[1].set_ylabel('Hedge Ratio')
        axes[1].set_title(f'Hedge Ratio Stability - {pair_id}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Half-life over time
        axes[2].plot(stability_df.index, stability_df['half_life'], label='Half-life')
        axes[2].axhline(y=metrics['median_half_life'], color='g', linestyle='--', label='Median')
        axes[2].set_ylabel('Half-life (periods)')
        axes[2].set_title(f'Mean Reversion Speed - {pair_id}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Add text with summary statistics
        summary_text = (
            f"Cointegration Frequency: {metrics['cointegration_frequency']:.2%}\n"
            f"Median Half-life: {metrics['median_half_life']:.2f} periods\n"
            f"Hedge Ratio: {metrics['hedge_ratio_mean']:.4f} ± {metrics['hedge_ratio_std']:.4f}\n"
            f"Hedge Ratio Stability: {metrics['hedge_ratio_stability']:.2%}"
        )
        
        plt.figtext(0.02, 0.01, summary_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # Save or display plot
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{pair_id}_stability.png")
        
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def visualize_time_based_metrics(self, pair_id, time_metrics, output_file=None):
        """
        Visualize time-of-day metrics for a pair.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier
        time_metrics : dict
            Result from calculate_time_based_metrics
        output_file : str, optional
            Path to save the visualization
            
        Returns:
        --------
        str
            Path to the saved visualization
        """
        if time_metrics is None:
            logger.error("No time metrics to visualize")
            return None
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Spread volatility by hour
        if 'spread_volatility' in time_metrics and not time_metrics['spread_volatility'].empty:
            time_metrics['spread_volatility']['volatility'].plot(ax=axes[0], kind='bar', color='skyblue')
            axes[0].set_ylabel('Spread Volatility')
            axes[0].set_title(f'Spread Volatility by Hour - {pair_id}')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Mean reversion strength by hour
        if 'mean_reversion' in time_metrics and not time_metrics['mean_reversion'].empty:
            time_metrics['mean_reversion']['half_life'].plot(ax=axes[1], kind='bar', color='lightgreen')
            axes[1].set_ylabel('Half-life (periods)')
            axes[1].set_title(f'Mean Reversion Strength by Hour - {pair_id}')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Correlations by hour
        if 'correlations' in time_metrics and not time_metrics['correlations'].empty:
            time_metrics['correlations']['correlation'].plot(ax=axes[2], kind='bar', color='salmon')
            axes[2].set_ylabel('Correlation')
            axes[2].set_title(f'Correlation by Hour - {pair_id}')
            axes[2].set_ylim(min(0.5, time_metrics['correlations']['correlation'].min()), 1.0)
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or display plot
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{pair_id}_time_metrics.png")
        
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def run_comprehensive_analysis(self, pair_id, start_date, end_date, timeframe="5min"):
        """
        Run a comprehensive analysis of an intraday pair.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        timeframe : str
            Timeframe for analysis
            
        Returns:
        --------
        dict
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Running comprehensive analysis for pair {pair_id}")
        
        # Run all analyses
        stationarity = self.analyze_pair_stationarity(pair_id, start_date, end_date, timeframe)
        stability = self.analyze_pair_stability(pair_id, start_date, end_date, timeframe)
        time_metrics = self.calculate_time_based_metrics(pair_id, start_date, end_date, timeframe)
        
        # Create visualizations
        if stability is not None:
            stability_viz = self.visualize_intraday_stability(pair_id, stability)
        else:
            stability_viz = None
        
        if time_metrics is not None:
            time_viz = self.visualize_time_based_metrics(pair_id, time_metrics)
        else:
            time_viz = None
        
        # Combine results
        results = {
            'pair_id': pair_id,
            'period': {
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe
            },
            'stationarity': stationarity,
            'stability': stability['stability_metrics'] if stability is not None else None,
            'time_metrics': time_metrics,
            'visualizations': {
                'stability': stability_viz,
                'time_metrics': time_viz
            }
        }
        
        # Save results to file
        output_file = os.path.join(self.output_dir, f"{pair_id}_analysis.json")
        
        # Convert to JSON-serializable format
        serializable_results = self._make_json_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return results
    
    def _make_json_serializable(self, data):
        """Make data JSON serializable by converting NumPy and pandas types."""
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (np.ndarray, pd.Series)):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict()
        elif isinstance(data, np.number):
            return float(data)
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        else:
            return data 