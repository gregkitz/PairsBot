# Example Notebooks and Tutorials

This directory contains example notebooks and tutorials for various components of the trading system. These examples demonstrate how to use the system's features and can serve as starting points for your own implementations.

## Available Examples

### Pair Trading Strategy Examples

- **[zscore_strategy_tutorial.ipynb](zscore_strategy_tutorial.ipynb)**: A comprehensive tutorial on using the ZScoreStrategyBacktest class to implement and backtest a pairs trading strategy. Includes parameter optimization, performance analysis, and visualization.

- **[zscore_strategy_distributed.ipynb](zscore_strategy_distributed.ipynb)**: Demonstrates how to use the z-score strategy with the distributed task system via Celery. Shows how to submit backtests and parameter optimizations as asynchronous tasks, monitor their progress, and analyze results.

## Running the Examples

To run these examples, you'll need:

1. A Python environment with the required dependencies installed
2. The trading system codebase accessible in your Python path

You can run the notebooks using Jupyter:

```bash
jupyter notebook zscore_strategy_tutorial.ipynb
```

Or JupyterLab:

```bash
jupyter lab
```

### For Distributed Examples

To run the distributed examples, you'll need:

1. Redis server running (for Celery task queue)
2. Celery workers running
3. The FastAPI server running

Start the services with:

```bash
# Start Redis (if not already running)
docker run -d -p 6379:6379 redis

# Start Celery workers
celery -A src.tasks.celery_app worker -l info -Q default,backtest,optimize,train,zscore

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Example Data

The examples use both synthetic and real market data:

- Synthetic data is generated within the notebooks for demonstration purposes
- Real data examples use historical price data for liquid futures contracts

## Recommended Learning Path

If you're new to the system, we recommend following this learning path:

1. Start with the z-score strategy tutorial to understand the basics of pair trading
2. Explore the cointegration testing examples to learn about pair selection
3. Review the backtest analysis examples to understand performance evaluation
4. Study the parameter optimization examples to learn how to fine-tune strategies
5. Explore the distributed processing examples to scale up your analysis

## Contributing Examples

If you develop an example that might be helpful to others, please consider contributing it:

1. Make sure your notebook includes clear explanations and documentation
2. Clean the notebook output before submission
3. Add appropriate references and citations for methods used
4. Submit a pull request with your example

## Upcoming Examples

We plan to add more examples in the future, including:

- Cointegration analysis and pair selection techniques
- Machine learning enhanced pairs trading
- Portfolio-level backtesting and optimization
- Market regime detection and adaptive strategies 