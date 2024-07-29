# Investment Strategy Backtesting Tool

This repository contains a Python-based tool for backtesting investment strategies and analyzing their performance. It consists of two main components: a backtesting module and a performance analysis module.

## Files

1. `backtest.py`: Contains the `Backtest` class for running historical simulations of investment strategies.
2. `performance.py`: Contains the `Performance` class for analyzing and visualizing backtest results.

## Requirements

- cuDF (RAPIDS)
- NumPy
- Matplotlib
- Seaborn
- Tabulate

## Usage

### Backtesting

The `Backtest` class in `backtest.py` allows you to run historical simulations of your investment strategy:

```python
from backtest import Backtest

# Initialize with historical data, model, and model parameters
backtest = Backtest(historical_data, model, model_params)

# Run the backtest for a specific time period
results = backtest.test(start_date, end_date)
```

### Performance Analysis

The `Performance` class in `performance.py` provides tools for analyzing and visualizing backtest results:

```python
from performance import Performance

# Initialize with backtest results and benchmark data
performance = Performance(backtest_data, benchmark_data)

# Generate a performance chart
performance.chart()

# Display a performance metrics table
performance.table()

# Get portfolio metrics as a dictionary
metrics = performance.portfolio_metrics()
```

### Features

- Backtesting of investment strategies using historical data
- Performance comparison against a benchmark
- Calculation of key performance metrics (e.g., returns, volatility, Sharpe ratio, beta)
- Visualization of cumulative returns
- Tabular display of performance metrics

### Note

This tool uses cuDF from the RAPIDS suite for GPU-accelerated data processing. Ensure you have a CUDA-capable GPU and the necessary RAPIDS dependencies installed. If you don't have access to a GPU just use plain pandas instead.