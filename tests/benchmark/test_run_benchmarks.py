#!/usr/bin/env python3
"""
Run all benchmark tests for the system.

This script runs all benchmark tests and compiles a summary report.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import benchmark modules
try:
    from tests.benchmark.test_data_processing_benchmark import run_data_processing_benchmarks
    from tests.benchmark.test_backtest_engine_benchmark import run_backtest_benchmarks
    from tests.benchmark.test_optimization_benchmark import run_all_benchmarks as run_optimization_benchmarks
    from src.utils import create_directory
except ImportError as e:
    print(f"Error importing benchmark modules: {e}")
    sys.exit(1)
    
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run system benchmark tests")
    parser.add_argument('--tests', nargs='+', choices=['data', 'backtest', 'optimization', 'all'], default=['all'],
                       help="Tests to run: 'data', 'backtest', 'optimization', or 'all'")
    parser.add_argument('--output', default='benchmark_results', 
                       help="Directory to save benchmark results")
    parser.add_argument('--quick', action='store_true',
                       help="Run quick benchmarks with fewer iterations")
    return parser.parse_args()

def run_benchmarks(args):
    """
    Run selected benchmark tests.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Setup
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    create_directory(output_dir)
    
    # Results to collect
    results = {
        "timestamp": timestamp,
        "runtime": {},
        "benchmarks": {}
    }
    
    # Run data processing benchmarks
    if 'all' in args.tests or 'data' in args.tests:
        print("\n=== Running Data Processing Benchmarks ===\n")
        data_start = time.time()
        
        try:
            data_results = run_data_processing_benchmarks()
            results["benchmarks"]["data_processing"] = data_results
            
            data_time = time.time() - data_start
            results["runtime"]["data_processing"] = data_time
            print(f"\nData processing benchmarks completed in {data_time:.2f} seconds\n")
            
        except Exception as e:
            print(f"Error running data processing benchmarks: {e}")
            import traceback
            traceback.print_exc()
            results["benchmarks"]["data_processing"] = {"error": str(e)}
    
    # Run backtest engine benchmarks
    if 'all' in args.tests or 'backtest' in args.tests:
        print("\n=== Running Backtest Engine Benchmarks ===\n")
        backtest_start = time.time()
        
        try:
            backtest_results = run_backtest_benchmarks()
            results["benchmarks"]["backtest_engine"] = backtest_results
            
            backtest_time = time.time() - backtest_start
            results["runtime"]["backtest_engine"] = backtest_time
            print(f"\nBacktest engine benchmarks completed in {backtest_time:.2f} seconds\n")
            
        except Exception as e:
            print(f"Error running backtest engine benchmarks: {e}")
            import traceback
            traceback.print_exc()
            results["benchmarks"]["backtest_engine"] = {"error": str(e)}
    
    # Run optimization benchmarks
    if 'all' in args.tests or 'optimization' in args.tests:
        print("\n=== Running Optimization Algorithm Benchmarks ===\n")
        optimization_start = time.time()
        
        try:
            optimization_results = run_optimization_benchmarks()
            results["benchmarks"]["optimization"] = optimization_results
            
            optimization_time = time.time() - optimization_start
            results["runtime"]["optimization"] = optimization_time
            print(f"\nOptimization benchmarks completed in {optimization_time:.2f} seconds\n")
            
        except Exception as e:
            print(f"Error running optimization benchmarks: {e}")
            import traceback
            traceback.print_exc()
            results["benchmarks"]["optimization"] = {"error": str(e)}
    
    # Compile summary report
    total_time = time.time() - start_time
    results["runtime"]["total"] = total_time
    
    print("\n=== Benchmark Summary ===\n")
    print(f"Total benchmark runtime: {total_time:.2f} seconds")
    for benchmark, runtime in results["runtime"].items():
        if benchmark != "total":
            print(f"  {benchmark}: {runtime:.2f} seconds")
    
    # Save results
    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_dir}")
    
    return results

def render_html_report(results, output_dir):
    """
    Generate HTML report of benchmark results.
    
    Parameters:
    -----------
    results : dict
        Benchmark results
    output_dir : Path
        Directory to save the report
    """
    # Create plots directory in the output directory
    plots_dir = output_dir / "plots"
    create_directory(plots_dir)
    
    # Copy plots from the benchmark_results/plots directory if it exists
    source_plots_dir = Path("benchmark_results/plots")
    if source_plots_dir.exists():
        import shutil
        for plot_file in source_plots_dir.glob("*.png"):
            shutil.copy(plot_file, plots_dir)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmark Results - {results['timestamp']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .summary {{ background-color: #f5f5f5; }}
            .plot {{ margin: 10px 0; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Benchmark Results</h1>
        <div class="section summary">
            <h2>Summary</h2>
            <p>Timestamp: {results['timestamp']}</p>
            <p>Total Runtime: {results['runtime'].get('total', 0):.2f} seconds</p>
            <table>
                <tr><th>Benchmark</th><th>Runtime (seconds)</th></tr>
    """
    
    # Add runtime for each benchmark
    for benchmark, runtime in results["runtime"].items():
        if benchmark != "total":
            html_content += f'<tr><td>{benchmark}</td><td>{runtime:.2f}</td></tr>\n'
    
    html_content += """
            </table>
        </div>
    """
    
    # Add benchmark results
    for benchmark_name, benchmark_results in results["benchmarks"].items():
        html_content += f"""
        <div class="section">
            <h2>{benchmark_name} Benchmark</h2>
        """
        
        if isinstance(benchmark_results, list):
            # Format as table
            html_content += """
            <table>
                <tr><th>Function</th><th>Mean Time (s)</th><th>Median Time (s)</th><th>Min Time (s)</th><th>Max Time (s)</th></tr>
            """
            
            for result in benchmark_results:
                html_content += f"""
                <tr>
                    <td>{result.get('function', 'N/A')}</td>
                    <td>{result.get('mean_time', 0):.4f}</td>
                    <td>{result.get('median_time', 0):.4f}</td>
                    <td>{result.get('min_time', 0):.4f}</td>
                    <td>{result.get('max_time', 0):.4f}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
        elif isinstance(benchmark_results, dict) and "error" in benchmark_results:
            # Show error
            html_content += f"""
            <p style="color: red;">Error: {benchmark_results['error']}</p>
            """
        else:
            # Show as JSON
            html_content += f"""
            <pre>{json.dumps(benchmark_results, indent=2)}</pre>
            """
        
        html_content += """
        </div>
        """
    
    # Add plots section if any plots exist
    plot_files = list(plots_dir.glob("*.png"))
    if plot_files:
        html_content += """
        <div class="section">
            <h2>Plots</h2>
        """
        
        for plot_file in plot_files:
            name = plot_file.stem.replace("_", " ").title()
            html_content += f"""
            <div class="plot">
                <h3>{name}</h3>
                <img src="plots/{plot_file.name}" alt="{name}">
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_dir / "benchmark_report.html", "w") as f:
        f.write(html_content)
    
    print(f"HTML report generated at {output_dir / 'benchmark_report.html'}")

if __name__ == "__main__":
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set environment variable for quick mode if enabled
        if args.quick:
            os.environ["BENCHMARK_QUICK_MODE"] = "1"
            print("Running in quick mode with fewer iterations")
        
        # Run benchmarks
        results = run_benchmarks(args)
        
        # Generate HTML report
        render_html_report(results, Path(args.output) / results["timestamp"])
        
    except KeyboardInterrupt:
        print("\nBenchmark run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 