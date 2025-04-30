#!/bin/bash

# Script to run the tests for the Intraday Statistical Arbitrage System

# Install test requirements if not already installed
pip install -r requirements-test.txt

# Set environment variable to indicate we're in test mode
export QUANT_TRADER_ENV="test"

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/unit -v

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/integration -v

# Run all tests with coverage
echo "Running all tests with coverage..."
python -m pytest --cov=src tests/ 