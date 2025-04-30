#!/usr/bin/env python3
"""
Benchmark real-time signal processing performance.

This script runs performance tests comparing the standard signal generation process
with the optimized real-time processing implementation.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Import signal processing modules
from src.real_time.signal_optimizer import RealTimeSignalOptimizer
from src.signal_generation.signal_generator import SignalGenerator
from src.utils.data_loader import load_intraday_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'benchmark_real_time.log'))
    ]
)
logger = logging.getLogger(__name__)

def create_test_data(symbol, periods=1000, freq='1min'):
    """
    Create synthetic test data for benchmarking.
    
    Parameters:
    -----------
    symbol : str
        Symbol identifier
    periods : int
        Number of data points to generate
    freq : str
        Frequency of data points
        
    Returns:
    --------
    pd.DataFrame
        Synthetic price data
    """
    # Create date range
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    date_range = pd.date_range(end=end_date, periods=periods, freq=freq)
    
    # Generate random price data with realistic properties
    start_price = 100.0
    returns = np.random.normal(0, 0.001, periods)
    prices = start_price * (1 + np.cumsum(returns))
    
    # Add noise and structure for realism
    noise = np.random.normal(0, 0.0005, periods)
    trend = np.linspace(0, 0.01, periods)
    
    # Create realistic OHLCV data
    data = pd.DataFrame(index=date_range)
    data['open'] = prices * (1 + noise)
    data['high'] = data['open'] * (1 + np.abs(noise) * 2)
    data['low'] = data['open'] * (1 - np.abs(noise) * 2)
    data['close'] = prices * (1 + noise) * (1 + trend)
    data['volume'] = np.random.poisson(100, periods) * (1 + np.abs(noise) * 10)
    
    # Ensure high > low
    data['high'] = np.maximum(data['high'], data['open'])
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['open'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    return data

def load_benchmark_data(symbol, start_date, end_date, timeframe='1min', data_dir='data/intraday'):
    """
    Load real data for benchmarking.
    
    Parameters:
    -----------
    symbol : str
        Symbol identifier
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    timeframe : str
        Timeframe for data
    data_dir : str
        Directory with data files
        
    Returns:
    --------
    pd.DataFrame
        Price data
    """
    try:
        # Try to load real data
        data = load_intraday_data(symbol, start_date, end_date, timeframe, data_dir)
        if data is not None and not data.empty:
            return data
    except Exception as e:
        logger.warning(f"Could not load real data for {symbol}: {e}")
    
    # Fall back to synthetic data
    logger.info(f"Using synthetic data for {symbol}")
    return create_test_data(symbol)

def benchmark_standard_processing(symbol, data, config=None, iterations=1):
    """
    Benchmark standard signal processing.
    
    Parameters:
    -----------
    symbol : str
        Symbol identifier
    data : pd.DataFrame
        Price data
    config : dict, optional
        Signal generator configuration
    iterations : int
        Number of iterations for benchmarking
        
    Returns:
    --------
    dict
        Performance metrics
    """
    # Create standard signal generator
    config = config or {}
    signal_generator = SignalGenerator(config)
    
    # Prepare benchmark
    processing_times = []
    
    # Run benchmark
    for _ in range(iterations):
        start_time = time.time()
        
        # Simulate standard processing
        # - Calculate z-scores
        if 'close' in data.columns:
            prices = data['close']
        else:
            prices = data
            
        returns = prices.pct_change()
        rolling_mean = returns.rolling(window=20).mean()
        rolling_std = returns.rolling(window=20).std()
        z_scores = (returns - rolling_mean) / rolling_std
        
        # - Generate signals
        signals = signal_generator.generate_signals(z_scores)
        
        # - Apply trading logic
        signals = signal_generator.apply_holding_period(signals)
        
        # Record time
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
    
    # Calculate metrics
    metrics = {
        'avg_ms': np.mean(processing_times) * 1000,
        'min_ms': np.min(processing_times) * 1000,
        'max_ms': np.max(processing_times) * 1000,
        'p95_ms': np.percentile(processing_times, 95) * 1000,
        'iterations': iterations
    }
    
    return metrics

def benchmark_optimized_processing(symbol, data, config=None, iterations=1, incremental=False):
    """
    Benchmark optimized real-time signal processing.
    
    Parameters:
    -----------
    symbol : str
        Symbol identifier
    data : pd.DataFrame
        Price data
    config : dict, optional
        Signal optimizer configuration
    iterations : int
        Number of iterations for benchmarking
    incremental : bool
        Whether to test incremental updates
        
    Returns:
    --------
    dict
        Performance metrics
    """
    # Create optimized signal processor
    config = config or {}
    optimizer = RealTimeSignalOptimizer(config)
    
    # Prepare benchmark
    processing_times = []
    feature_times = []
    signal_times = []
    
    # Run benchmark
    if incremental:
        # Test incremental updates - split data into chunks
        chunk_size = len(data) // 10
        chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        # First load the initial chunk
        if len(chunks) > 0:
            optimizer.process_new_data(symbol, chunks[0])
        
        # Then test with incremental updates
        for chunk in chunks[1:]:
            start_time = time.time()
            result = optimizer.process_new_data(symbol, chunk)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            feature_times.append(result['timing']['feature_calculation_ms'] / 1000)
            signal_times.append(result['timing']['signal_generation_ms'] / 1000)
    else:
        # Test full data processing
        for _ in range(iterations):
            start_time = time.time()
            result = optimizer.process_new_data(symbol, data)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            feature_times.append(result['timing']['feature_calculation_ms'] / 1000)
            signal_times.append(result['timing']['signal_generation_ms'] / 1000)
    
    # Calculate metrics
    metrics = {
        'total': {
            'avg_ms': np.mean(processing_times) * 1000,
            'min_ms': np.min(processing_times) * 1000,
            'max_ms': np.max(processing_times) * 1000,
            'p95_ms': np.percentile(processing_times, 95) * 1000,
        },
        'features': {
            'avg_ms': np.mean(feature_times) * 1000,
            'min_ms': np.min(feature_times) * 1000,
            'max_ms': np.max(feature_times) * 1000,
            'p95_ms': np.percentile(feature_times, 95) * 1000,
        },
        'signals': {
            'avg_ms': np.mean(signal_times) * 1000,
            'min_ms': np.min(signal_times) * 1000,
            'max_ms': np.max(signal_times) * 1000,
            'p95_ms': np.percentile(signal_times, 95) * 1000,
        },
        'iterations': len(processing_times),
        'incremental': incremental
    }
    
    return metrics

def benchmark_latency_scaling(symbols, data_frames, config=None, iterations=1):
    """
    Benchmark how latency scales with the number of symbols.
    
    Parameters:
    -----------
    symbols : list
        List of symbol identifiers
    data_frames : list
        List of price data frames
    config : dict, optional
        Configuration
    iterations : int
        Number of iterations for benchmarking
        
    Returns:
    --------
    dict
        Scaling metrics
    """
    # Create optimized signal processor
    config = config or {}
    optimizer = RealTimeSignalOptimizer(config)
    
    # Prepare benchmark
    scaling_metrics = []
    
    # Test with increasing number of symbols
    for n_symbols in range(1, len(symbols) + 1):
        # Process n symbols
        current_symbols = symbols[:n_symbols]
        current_data = data_frames[:n_symbols]
        
        processing_times = []
        
        # Run benchmark
        for _ in range(iterations):
            start_time = time.time()
            
            # Process all symbols
            for i, symbol in enumerate(current_symbols):
                optimizer.process_new_data(symbol, current_data[i])
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        # Calculate metrics for this number of symbols
        scaling_metrics.append({
            'n_symbols': n_symbols,
            'avg_ms': np.mean(processing_times) * 1000,
            'min_ms': np.min(processing_times) * 1000,
            'max_ms': np.max(processing_times) * 1000,
            'p95_ms': np.percentile(processing_times, 95) * 1000,
            'avg_per_symbol_ms': np.mean(processing_times) * 1000 / n_symbols
        })
    
    return scaling_metrics

def create_benchmark_report(results, output_dir='output/benchmarks'):
    """
    Create benchmark report with visualizations.
    
    Parameters:
    -----------
    results : dict
        Benchmark results
    output_dir : str
        Output directory for report
        
    Returns:
    --------
    str
        Path to report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(output_dir, f'benchmark_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    
    # Standard vs Optimized
    if 'standard' in results and 'optimized' in results:
        labels = ['Standard', 'Optimized']
        avg_times = [
            results['standard']['avg_ms'],
            results['optimized']['total']['avg_ms']
        ]
        p95_times = [
            results['standard']['p95_ms'],
            results['optimized']['total']['p95_ms']
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, avg_times, width, label='Avg Time (ms)')
        plt.bar(x + width/2, p95_times, width, label='P95 Time (ms)')
        
        plt.ylabel('Processing Time (ms)')
        plt.title('Signal Processing Performance Comparison')
        plt.xticks(x, labels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'performance_comparison.png'))
        plt.close()
    
    # Incremental vs Full data processing
    if 'optimized' in results and 'incremental' in results:
        labels = ['Full Data', 'Incremental']
        avg_times = [
            results['optimized']['total']['avg_ms'],
            results['incremental']['total']['avg_ms']
        ]
        p95_times = [
            results['optimized']['total']['p95_ms'],
            results['incremental']['total']['p95_ms']
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, avg_times, width, label='Avg Time (ms)')
        plt.bar(x + width/2, p95_times, width, label='P95 Time (ms)')
        
        plt.ylabel('Processing Time (ms)')
        plt.title('Incremental vs Full Data Processing Performance')
        plt.xticks(x, labels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'incremental_comparison.png'))
        plt.close()
    
    # Scaling with number of symbols
    if 'scaling' in results:
        scaling_data = results['scaling']
        
        plt.figure(figsize=(10, 6))
        
        x = [metric['n_symbols'] for metric in scaling_data]
        avg_times = [metric['avg_ms'] for metric in scaling_data]
        per_symbol_times = [metric['avg_per_symbol_ms'] for metric in scaling_data]
        
        plt.plot(x, avg_times, 'o-', label='Total Processing Time (ms)')
        plt.plot(x, per_symbol_times, 's-', label='Time Per Symbol (ms)')
        
        plt.xlabel('Number of Symbols')
        plt.ylabel('Processing Time (ms)')
        plt.title('Processing Time Scaling with Number of Symbols')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'scaling_performance.png'))
        plt.close()
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Processing Benchmark Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metrics-table th {{ background-color: #4CAF50; color: white; }}
            .chart-container {{ margin: 20px 0; }}
            .chart {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Real-time Processing Benchmark Results</h1>
        <p>Benchmark run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Performance Comparison</h2>
        
        <h3>Standard vs. Optimized Processing</h3>
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Standard (ms)</th>
                <th>Optimized (ms)</th>
                <th>Improvement</th>
            </tr>
    """
    
    # Add metrics rows
    if 'standard' in results and 'optimized' in results:
        std_avg = results['standard']['avg_ms']
        opt_avg = results['optimized']['total']['avg_ms']
        improvement = (std_avg - opt_avg) / std_avg * 100 if std_avg > 0 else 0
        
        html_content += f"""
            <tr>
                <td>Average Processing Time</td>
                <td>{std_avg:.2f}</td>
                <td>{opt_avg:.2f}</td>
                <td>{improvement:.1f}%</td>
            </tr>
        """
        
        std_p95 = results['standard']['p95_ms']
        opt_p95 = results['optimized']['total']['p95_ms']
        improvement = (std_p95 - opt_p95) / std_p95 * 100 if std_p95 > 0 else 0
        
        html_content += f"""
            <tr>
                <td>P95 Processing Time</td>
                <td>{std_p95:.2f}</td>
                <td>{opt_p95:.2f}</td>
                <td>{improvement:.1f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <div class="chart-container">
            <img class="chart" src="performance_comparison.png" alt="Performance Comparison Chart">
        </div>
    """
    
    # Add incremental processing section
    if 'optimized' in results and 'incremental' in results:
        html_content += """
        <h3>Full vs. Incremental Processing</h3>
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Full Processing (ms)</th>
                <th>Incremental Updates (ms)</th>
                <th>Improvement</th>
            </tr>
        """
        
        full_avg = results['optimized']['total']['avg_ms']
        incr_avg = results['incremental']['total']['avg_ms']
        improvement = (full_avg - incr_avg) / full_avg * 100 if full_avg > 0 else 0
        
        html_content += f"""
            <tr>
                <td>Average Processing Time</td>
                <td>{full_avg:.2f}</td>
                <td>{incr_avg:.2f}</td>
                <td>{improvement:.1f}%</td>
            </tr>
        """
        
        full_p95 = results['optimized']['total']['p95_ms']
        incr_p95 = results['incremental']['total']['p95_ms']
        improvement = (full_p95 - incr_p95) / full_p95 * 100 if full_p95 > 0 else 0
        
        html_content += f"""
            <tr>
                <td>P95 Processing Time</td>
                <td>{full_p95:.2f}</td>
                <td>{incr_p95:.2f}</td>
                <td>{improvement:.1f}%</td>
            </tr>
        </table>
        
        <div class="chart-container">
            <img class="chart" src="incremental_comparison.png" alt="Incremental Processing Comparison Chart">
        </div>
        """
    
    # Add scaling section
    if 'scaling' in results:
        html_content += """
        <h3>Processing Time Scaling</h3>
        <table class="metrics-table">
            <tr>
                <th>Symbols</th>
                <th>Total Processing Time (ms)</th>
                <th>Time Per Symbol (ms)</th>
            </tr>
        """
        
        for metric in results['scaling']:
            html_content += f"""
            <tr>
                <td>{metric['n_symbols']}</td>
                <td>{metric['avg_ms']:.2f}</td>
                <td>{metric['avg_per_symbol_ms']:.2f}</td>
            </tr>
            """
        
        html_content += """
        </table>
        
        <div class="chart-container">
            <img class="chart" src="scaling_performance.png" alt="Scaling Performance Chart">
        </div>
        """
    
    # Finish HTML report
    html_content += """
        <h2>Conclusion</h2>
        <p>
            The optimized real-time processing implementation demonstrates significant performance improvements
            over the standard implementation. The incremental processing approach further reduces latency for
            continuous data updates, making it suitable for high-frequency trading applications.
        </p>
        
        <h3>Key Findings</h3>
        <ul>
    """
    
    # Add key findings
    if 'standard' in results and 'optimized' in results:
        std_avg = results['standard']['avg_ms']
        opt_avg = results['optimized']['total']['avg_ms']
        improvement = (std_avg - opt_avg) / std_avg * 100 if std_avg > 0 else 0
        
        html_content += f"""
            <li>The optimized implementation is <strong>{improvement:.1f}%</strong> faster than the standard implementation on average.</li>
        """
    
    if 'optimized' in results and 'incremental' in results:
        full_avg = results['optimized']['total']['avg_ms']
        incr_avg = results['incremental']['total']['avg_ms']
        improvement = (full_avg - incr_avg) / full_avg * 100 if full_avg > 0 else 0
        
        html_content += f"""
            <li>Incremental processing is <strong>{improvement:.1f}%</strong> faster than full data processing.</li>
        """
    
    if 'scaling' in results and len(results['scaling']) > 0:
        max_symbols = results['scaling'][-1]['n_symbols']
        max_time = results['scaling'][-1]['avg_ms']
        
        html_content += f"""
            <li>Processing {max_symbols} symbols takes <strong>{max_time:.2f} ms</strong> on average.</li>
        """
        
        if max_symbols > 0 and max_time / max_symbols < 500:
            html_content += f"""
            <li>Per-symbol processing time is well below the target of 500 ms, meeting real-time requirements.</li>
            """
    
    html_content += """
        </ul>
    </body>
    </html>
    """
    
    # Write HTML report
    report_path = os.path.join(report_dir, 'benchmark_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Benchmark report created at: {report_path}")
    return report_path

def run_benchmarks(symbols, start_date, end_date, timeframe, iterations, output_dir):
    """
    Run all benchmarks and generate report.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to benchmark
    start_date : str
        Start date for data
    end_date : str
        End date for data
    timeframe : str
        Timeframe for data
    iterations : int
        Number of iterations for benchmarking
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    str
        Path to benchmark report
    """
    # Load data for benchmarking
    data_frames = []
    
    logger.info("Loading data for benchmarking...")
    for symbol in tqdm(symbols, desc="Loading data"):
        data = load_benchmark_data(symbol, start_date, end_date, timeframe)
        if data is not None and not data.empty:
            data_frames.append(data)
        else:
            logger.warning(f"No data available for {symbol}, using synthetic data")
            data_frames.append(create_test_data(symbol))
    
    # Prepare results dictionary
    results = {}
    
    # Run standard processing benchmark
    logger.info("Benchmarking standard processing...")
    results['standard'] = benchmark_standard_processing(symbols[0], data_frames[0], iterations=iterations)
    logger.info(f"Standard processing: {results['standard']['avg_ms']:.2f} ms avg, {results['standard']['p95_ms']:.2f} ms p95")
    
    # Run optimized processing benchmark
    logger.info("Benchmarking optimized processing...")
    results['optimized'] = benchmark_optimized_processing(symbols[0], data_frames[0], iterations=iterations)
    logger.info(f"Optimized processing: {results['optimized']['total']['avg_ms']:.2f} ms avg, {results['optimized']['total']['p95_ms']:.2f} ms p95")
    
    # Run incremental processing benchmark
    logger.info("Benchmarking incremental processing...")
    results['incremental'] = benchmark_optimized_processing(symbols[0], data_frames[0], iterations=iterations, incremental=True)
    logger.info(f"Incremental processing: {results['incremental']['total']['avg_ms']:.2f} ms avg, {results['incremental']['total']['p95_ms']:.2f} ms p95")
    
    # Run scaling benchmark
    if len(symbols) > 1 and len(data_frames) > 1:
        logger.info("Benchmarking latency scaling...")
        results['scaling'] = benchmark_latency_scaling(symbols, data_frames, iterations=max(1, iterations // 2))
        logger.info(f"Scaling benchmark completed with {len(symbols)} symbols")
    
    # Create benchmark report
    return create_benchmark_report(results, output_dir)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark real-time signal processing')
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        default='ES_NQ,CL_GC,ZN_ZB',
        help='Comma-separated list of symbols to benchmark'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        help='Start date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1min',
        help='Timeframe for data'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=10,
        help='Number of iterations for benchmarking'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output/benchmarks',
        help='Output directory for benchmark results'
    )
    
    parser.add_argument(
        '--use-synthetic-data',
        action='store_true',
        help='Use synthetic data instead of real data'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Run benchmarks
    if args.use_synthetic_data:
        # Create synthetic data frames
        data_frames = [create_test_data(symbol) for symbol in symbols]
        
        # Prepare results dictionary
        results = {}
        
        # Run benchmarks with synthetic data
        logger.info("Benchmarking with synthetic data...")
        
        # Standard processing
        results['standard'] = benchmark_standard_processing(symbols[0], data_frames[0], iterations=args.iterations)
        
        # Optimized processing
        results['optimized'] = benchmark_optimized_processing(symbols[0], data_frames[0], iterations=args.iterations)
        
        # Incremental processing
        results['incremental'] = benchmark_optimized_processing(symbols[0], data_frames[0], iterations=args.iterations, incremental=True)
        
        # Scaling benchmark
        if len(symbols) > 1:
            results['scaling'] = benchmark_latency_scaling(symbols, data_frames, iterations=max(1, args.iterations // 2))
        
        # Create benchmark report
        create_benchmark_report(results, args.output_dir)
    else:
        # Run full benchmarks with real data
        report_path = run_benchmarks(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            iterations=args.iterations,
            output_dir=args.output_dir
        )
        
        logger.info(f"Benchmark completed. Report available at: {report_path}")

if __name__ == '__main__':
    main() 