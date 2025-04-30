"""
Test script for the IntradayPairAnalyzer.

This script demonstrates the analysis of intraday pairs for cointegration relationships
and visualizes various stability metrics.
"""

import os
import logging
import pandas as pd
from src.cointegration import IntradayPairAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_stationarity_analysis():
    """Test stationarity analysis for a pair."""
    data_dir = "data/processed"
    output_dir = "output/intraday_analysis"
    
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize analyzer
    analyzer = IntradayPairAnalyzer(data_dir, output_dir)
    
    # Get available symbols from data processor
    symbols = analyzer.data_processor.get_available_symbols()
    logger.info(f"Available symbols: {symbols}")
    
    if len(symbols) < 2:
        logger.error("Not enough symbols available for pair analysis")
        return
    
    # Create a test pair
    pair_id = f"{symbols[0]}_{symbols[1]}"
    logger.info(f"Testing pair: {pair_id}")
    
    # Set test date range
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Run stationarity analysis
    stationarity_result = analyzer.analyze_pair_stationarity(pair_id, start_date, end_date)
    
    if stationarity_result is None:
        logger.error("Failed to analyze pair stationarity")
        return
    
    # Log results
    for symbol, stats in stationarity_result.items():
        if symbol != 'cointegration':
            logger.info(f"{symbol} stationarity: p-value = {stats.get('p_value', 'N/A')}")
    
    cointegration = stationarity_result.get('cointegration', {})
    logger.info(f"Cointegration: ADF p-value = {cointegration.get('p_value', 'N/A')}")

def test_stability_analysis():
    """Test stability analysis for a pair."""
    data_dir = "data/processed"
    output_dir = "output/intraday_analysis"
    
    # Initialize analyzer
    analyzer = IntradayPairAnalyzer(data_dir, output_dir)
    
    # Get available symbols
    symbols = analyzer.data_processor.get_available_symbols()
    
    if len(symbols) < 2:
        logger.error("Not enough symbols available for pair analysis")
        return
    
    # Create a test pair
    pair_id = f"{symbols[0]}_{symbols[1]}"
    logger.info(f"Testing pair stability: {pair_id}")
    
    # Set test date range
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Run stability analysis with shorter window for testing
    stability_result = analyzer.analyze_pair_stability(
        pair_id, start_date, end_date, window_minutes=60
    )
    
    if stability_result is None:
        logger.error("Failed to analyze pair stability")
        return
    
    # Log results
    metrics = stability_result['stability_metrics']
    logger.info(f"Cointegration frequency: {metrics['cointegration_frequency']:.2%}")
    logger.info(f"Median half-life: {metrics['median_half_life']:.2f} periods")
    logger.info(f"Hedge ratio stability: {metrics['hedge_ratio_stability']:.2%}")
    
    # Visualize stability
    viz_path = analyzer.visualize_intraday_stability(pair_id, stability_result)
    logger.info(f"Stability visualization saved to: {viz_path}")

def test_time_based_metrics():
    """Test time-based metrics for a pair."""
    data_dir = "data/processed"
    output_dir = "output/intraday_analysis"
    
    # Initialize analyzer
    analyzer = IntradayPairAnalyzer(data_dir, output_dir)
    
    # Get available symbols
    symbols = analyzer.data_processor.get_available_symbols()
    
    if len(symbols) < 2:
        logger.error("Not enough symbols available for pair analysis")
        return
    
    # Create a test pair
    pair_id = f"{symbols[0]}_{symbols[1]}"
    logger.info(f"Testing time-based metrics: {pair_id}")
    
    # Set test date range
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Calculate time-based metrics
    time_metrics = analyzer.calculate_time_based_metrics(pair_id, start_date, end_date)
    
    if time_metrics is None:
        logger.error("Failed to calculate time-based metrics")
        return
    
    # Log results
    if 'spread_volatility' in time_metrics and not time_metrics['spread_volatility'].empty:
        logger.info(f"Spread volatility by hour:\n{time_metrics['spread_volatility']['volatility']}")
    
    if 'mean_reversion' in time_metrics and not time_metrics['mean_reversion'].empty:
        logger.info(f"Mean reversion by hour:\n{time_metrics['mean_reversion']['half_life']}")
    
    # Visualize time metrics
    viz_path = analyzer.visualize_time_based_metrics(pair_id, time_metrics)
    logger.info(f"Time metrics visualization saved to: {viz_path}")

def test_comprehensive_analysis():
    """Test comprehensive analysis for a pair."""
    data_dir = "data/processed"
    output_dir = "output/intraday_analysis"
    
    # Initialize analyzer
    analyzer = IntradayPairAnalyzer(data_dir, output_dir)
    
    # Get available symbols
    symbols = analyzer.data_processor.get_available_symbols()
    
    if len(symbols) < 2:
        logger.error("Not enough symbols available for pair analysis")
        return
    
    # Create a test pair
    pair_id = f"{symbols[0]}_{symbols[1]}"
    logger.info(f"Testing comprehensive analysis: {pair_id}")
    
    # Set test date range
    start_date = "2023-01-01"
    end_date = "2023-01-10"  # Shorter period for testing
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(pair_id, start_date, end_date)
    
    if results is None:
        logger.error("Failed to run comprehensive analysis")
        return
    
    # Log results
    logger.info(f"Comprehensive analysis completed for {pair_id}")
    logger.info(f"Results saved to: {os.path.join(output_dir, f'{pair_id}_analysis.json')}")
    
    if 'visualizations' in results:
        for viz_type, path in results['visualizations'].items():
            if path:
                logger.info(f"{viz_type} visualization saved to: {path}")

if __name__ == "__main__":
    logger.info("Testing IntradayPairAnalyzer")
    
    # Test stationarity analysis
    test_stationarity_analysis()
    
    # Test stability analysis
    test_stability_analysis()
    
    # Test time-based metrics
    test_time_based_metrics()
    
    # Test comprehensive analysis
    test_comprehensive_analysis()
    
    logger.info("Tests completed") 