"""
Execution Module

This module contains implementations of execution algorithms and transaction cost models
for trading operations, focusing on realistic modeling of market execution.

Components:
- Transaction cost models (exchange fees, slippage, market impact)
- Execution algorithms (VWAP, TWAP, adaptive execution)
- Execution simulation for backtesting
"""

from src.execution.transaction_costs import (
    ExchangeType, SlippageModel, ExchangeFeeConfig, ExchangeFeeModel,
    FixedSlippageModel, VolumeBasedSlippageModel, VolatilityAdjustedSlippageModel,
    MarketImpactModel, CustomSlippageModel, TransactionCostAnalyzer,
    create_futures_exchange_fee_model, create_stock_exchange_fee_model, create_crypto_exchange_fee_model
)

from src.execution.execution_algorithms import (
    ExecutionType, ExecutionConfig, TWAPConfig, VWAPConfig, AdaptiveConfig,
    ExecutionAlgorithm, MarketExecutionAlgorithm, TWAPExecutionAlgorithm,
    VWAPExecutionAlgorithm, AdaptiveExecutionAlgorithm, IcebergExecutionAlgorithm,
    ExecutionSimulator
)

__all__ = [
    # Transaction costs module
    'ExchangeType', 'SlippageModel', 'ExchangeFeeConfig', 'ExchangeFeeModel',
    'FixedSlippageModel', 'VolumeBasedSlippageModel', 'VolatilityAdjustedSlippageModel',
    'MarketImpactModel', 'CustomSlippageModel', 'TransactionCostAnalyzer',
    'create_futures_exchange_fee_model', 'create_stock_exchange_fee_model', 'create_crypto_exchange_fee_model',
    
    # Execution algorithms module
    'ExecutionType', 'ExecutionConfig', 'TWAPConfig', 'VWAPConfig', 'AdaptiveConfig',
    'ExecutionAlgorithm', 'MarketExecutionAlgorithm', 'TWAPExecutionAlgorithm',
    'VWAPExecutionAlgorithm', 'AdaptiveExecutionAlgorithm', 'IcebergExecutionAlgorithm',
    'ExecutionSimulator'
] 