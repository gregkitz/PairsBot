# Quant-Trader Codebase Summary


## Summary Statistics

- Python Files: 209
- Modules: 209
- Classes: 116
- Functions: 329
- Components: 166

## Component Overview

### asset_classes (11)

- **__init__** - Asset Classes Module for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Futures Asset Class for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/futures/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Cryptocurrency Asset Classes Module for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/cryptocurrencies/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Fixed Income Asset Classes Module for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/fixed_income/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Equity Asset Classes Module for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/equities/__init__.py`
  - Classes: 0, Functions: 0

- **base** - Base Asset Classes for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/base.py`
  - Classes: 2, Functions: 0

- **crypto_asset** - Cryptocurrency Asset Implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/cryptocurrencies/crypto_asset.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.asset_classes.base.Asset, src.asset_classes.base.AssetClass, src.connectors.ib.IBConnector

- **equity_asset** - Equity Asset Implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/equities/equity_asset.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.asset_classes.base.Asset, src.asset_classes.base.AssetClass, src.connectors.ib.IBConnector

- **factory** - Factory Module for Asset Classes.
  - Path: `src/asset_classes/factory.py`
  - Classes: 0, Functions: 3

- **fixed_income_asset** - Fixed Income Asset Implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/fixed_income/fixed_income_asset.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.asset_classes.base.Asset, src.asset_classes.base.AssetClass, src.connectors.ib.IBConnector

- **futures_asset** - Futures Asset Implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/asset_classes/futures/futures_asset.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.asset_classes.base.Asset, src.asset_classes.base.AssetClass, src.connectors.ib.IBConnector

### backtest (4)

- **__init__** - Backtesting module for simulating and evaluating trading strategies.
  - Path: `src/backtest/__init__.py`
  - Classes: 0, Functions: 0

- **backtest_engine** - No description
  - Path: `src/backtest/backtest_engine.py`
  - Classes: 1, Functions: 0

- **intraday_backtest_engine** - Intraday backtesting engine with realistic constraints and transaction cost modeling.
  - Path: `src/backtest/intraday_backtest_engine.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.backtest.backtest_engine.BacktestEngine

- **intraday_performance_visualization** - Visualization module for intraday backtest performance analysis.
  - Path: `src/backtest/intraday_performance_visualization.py`
  - Classes: 0, Functions: 7

### codebase_review (1)

- **codebase_analyzer** - Codebase Analyzer
  - Path: `codebase_review/codebase_analyzer.py`
  - Classes: 1, Functions: 1

### cointegration (5)

- **__init__** - Cointegration testing and pair selection module.
  - Path: `src/cointegration/__init__.py`
  - Classes: 0, Functions: 0

- **cointegration_tests** - No description
  - Path: `src/cointegration/cointegration_tests.py`
  - Classes: 0, Functions: 4

- **intraday_pair_analyzer** - Intraday Pair Analyzer Module
  - Path: `src/cointegration/intraday_pair_analyzer.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.cointegration.cointegration_tests.calculate_half_life, src.cointegration.cointegration_tests.test_cointegration, src.data_processor.intraday_processor.IntradayDataProcessor

- **pair_finder** - No description
  - Path: `src/cointegration/pair_finder.py`
  - Classes: 1, Functions: 0

- **pairs_finder** - Cointegration Pairs Finder Module
  - Path: `src/cointegration/pairs_finder.py`
  - Classes: 1, Functions: 0

### config (2)

- **__init__** - Configuration management module.
  - Path: `src/config/__init__.py`
  - Classes: 0, Functions: 0

- **configuration** - No description
  - Path: `src/config/configuration.py`
  - Classes: 2, Functions: 0

### connectors (4)

- **__init__** - Interactive Brokers (IB) Connector Module for the Intraday Statistical Arbitrage System.
  - Path: `src/connectors/ib/__init__.py`
  - Classes: 0, Functions: 0

- **fixed_connector** - Fixed version of the Interactive Brokers (IB) Connector for testing.
  - Path: `src/connectors/ib/fixed_connector.py`
  - Classes: 1, Functions: 0

- **ib_connector** - Interactive Brokers (IB) Connector for the Intraday Statistical Arbitrage System.
  - Path: `src/connectors/ib/ib_connector.py`
  - Classes: 1, Functions: 0

- **ib_utils** - Utility functions for working with Interactive Brokers (IB) contracts.
  - Path: `src/connectors/ib/ib_utils.py`
  - Classes: 0, Functions: 4

### data_processing (1)

- **data_processor** - No description
  - Path: `src/data_processing/data_processor.py`
  - Classes: 1, Functions: 0
  - Dependencies: data.data_loader.load_data

### data_processor (4)

- **__init__** - Data processing module for loading and managing market data.
  - Path: `src/data_processor/__init__.py`
  - Classes: 0, Functions: 0

- **data_processor** - No description
  - Path: `src/data_processor/data_processor.py`
  - Classes: 1, Functions: 0

- **futures_processor** - Futures Data Processor Module
  - Path: `src/data_processor/futures_processor.py`
  - Classes: 1, Functions: 0

- **intraday_processor** - Intraday Data Processor Module
  - Path: `src/data_processor/intraday_processor.py`
  - Classes: 1, Functions: 0

### examples (3)

- **run_enhanced_intraday_backtest** - Example script demonstrating how to use the IntradayBacktestEngine with performance visualization.
  - Path: `examples/run_enhanced_intraday_backtest.py`
  - Classes: 0, Functions: 5
  - Dependencies: src.backtest.intraday_backtest_engine.IntradayBacktestEngine, src.backtest.intraday_performance_visualization.create_intraday_performance_dashboard, src.backtest.intraday_performance_visualization.save_performance_dashboard, src.ml_enhancements.regime_detection.market_regime_classifier.MarketRegimeClassifier

- **run_ml_integrated_intraday** - Example script demonstrating the integrated ML system for intraday trading.
  - Path: `examples/run_ml_integrated_intraday.py`
  - Classes: 0, Functions: 6
  - Dependencies: src.ml_enhancements.intraday_integration.IntradayMLSystem, src.ml_enhancements.intraday_integration.integrate_with_paper_trading, src.ml_enhancements.intraday_signals.IntradaySignalProcessor, src.ml_enhancements.regime_detection.market_regime_classifier.MarketRegimeClassifier

- **use_strategy_variants** - Example script demonstrating how to use the different strategy variants.
  - Path: `examples/use_strategy_variants.py`
  - Classes: 0, Functions: 5
  - Dependencies: src.backtest_engine.BacktestEngine, src.pairs_trading_strategy.PairsTradingStrategy, src.strategy_variants.MLSignalStrategy, src.strategy_variants.TimeSeriesStrategy, src.visualization.plot_equity_curve
    and 4 more...

### live_trading (7)

- **__init__** - Live Trading Module for the Intraday Statistical Arbitrage System.
  - Path: `src/live_trading/__init__.py`
  - Classes: 0, Functions: 0

- **example** - Example script for using the LiveTrader with Interactive Brokers.
  - Path: `src/live_trading/example.py`
  - Classes: 0, Functions: 8
  - Dependencies: src.connectors.ib.contract_to_symbol, src.connectors.ib.symbol_to_contract, src.live_trading.LiveTrader, src.pairs_trading_strategy.PairsTradingStrategy

- **example_with_monitoring** - Example script for using the LiveTrader with TradingMonitor.
  - Path: `src/live_trading/example_with_monitoring.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.connectors.ib.contract_to_symbol, src.connectors.ib.symbol_to_contract, src.live_trading.LiveTrader, src.live_trading.monitoring.TradingMonitor, src.pairs_trading_strategy.PairsTradingStrategy

- **example_with_position_tracking** - Example script for using the LiveTrader with PositionTracker.
  - Path: `src/live_trading/example_with_position_tracking.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.connectors.ib.contract_to_symbol, src.connectors.ib.symbol_to_contract, src.live_trading.LiveTrader, src.live_trading.PositionTracker, src.pairs_trading_strategy.PairsTradingStrategy

- **live_trader** - Live Trader for the Intraday Statistical Arbitrage System.
  - Path: `src/live_trading/live_trader.py`
  - Classes: 1, Functions: 0

- **monitoring** - Monitoring module for the Intraday Statistical Arbitrage System.
  - Path: `src/live_trading/monitoring.py`
  - Classes: 1, Functions: 0

- **position_tracker** - Position Tracking module for the Intraday Statistical Arbitrage System.
  - Path: `src/live_trading/position_tracker.py`
  - Classes: 1, Functions: 0

### ml_enhancements (18)

- **__init__** - Machine Learning Enhancements for the Intraday Statistical Arbitrage System.
  - Path: `src/ml_enhancements/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Spread Prediction Models for Pairs Trading.
  - Path: `src/ml_enhancements/spread_prediction/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Regime Detection for Pairs Trading.
  - Path: `src/ml_enhancements/regime_detection/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Feature Engineering Module for ML-based pairs trading.
  - Path: `src/ml_enhancements/feature_engineering/__init__.py`
  - Classes: 0, Functions: 0

- **advanced_features** - Advanced Feature Engineering Module
  - Path: `src/ml_enhancements/feature_engineering/advanced_features.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.ml_enhancements.feature_engineering.intraday_features.IntradayFeatureEngineering

- **data_augmentation** - Time Series Data Augmentation Module
  - Path: `src/ml_enhancements/data_augmentation.py`
  - Classes: 1, Functions: 0

- **ensemble_models** - Ensemble Models Module
  - Path: `src/ml_enhancements/ensemble_models.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.ml_enhancements.training_utils.WalkForwardValidator

- **feature_generator** - Feature Generator for Pairs Trading.
  - Path: `src/ml_enhancements/feature_engineering/feature_generator.py`
  - Classes: 1, Functions: 0

- **intraday_features** - Intraday Feature Engineering Module
  - Path: `src/ml_enhancements/feature_engineering/intraday_features.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.ml_enhancements.feature_engineering.feature_generator.FeatureGenerator

- **intraday_integration** - Intraday ML Integration Module
  - Path: `src/ml_enhancements/intraday_integration.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.backtest.intraday_backtest_engine.IntradayBacktestEngine, src.backtest.intraday_performance_visualization.create_intraday_performance_dashboard, src.backtest.intraday_performance_visualization.save_performance_dashboard, src.ml_enhancements.intraday_signals.IntradaySignalEnhancer, src.ml_enhancements.intraday_signals.IntradaySignalProcessor
    and 1 more...

- **intraday_model_trainer** - Intraday Model Trainer Module
  - Path: `src/ml_enhancements/intraday_model_trainer.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.ml_enhancements.training_utils.WalkForwardValidator

- **intraday_signals** - Intraday Signal Enhancement Module
  - Path: `src/ml_enhancements/intraday_signals.py`
  - Classes: 2, Functions: 0

- **market_regime_classifier** - Market Regime Classifier Module
  - Path: `src/ml_enhancements/regime_detection/market_regime_classifier.py`
  - Classes: 1, Functions: 0

- **model_factory** - Model Factory for Spread Prediction.
  - Path: `src/ml_enhancements/spread_prediction/model_factory.py`
  - Classes: 0, Functions: 2

- **model_retraining** - Automated Model Retraining Module
  - Path: `src/ml_enhancements/model_retraining.py`
  - Classes: 1, Functions: 2

- **regime_detector** - Regime Detector for Pairs Trading.
  - Path: `src/ml_enhancements/regime_detection/regime_detector.py`
  - Classes: 1, Functions: 0

- **spread_predictor** - Spread Predictor for Pairs Trading.
  - Path: `src/ml_enhancements/spread_prediction/spread_predictor.py`
  - Classes: 1, Functions: 0

- **training_utils** - ML Training Utilities Module
  - Path: `src/ml_enhancements/training_utils.py`
  - Classes: 1, Functions: 3

### optimization (10)

- **__init__** - Parameter Optimization Framework for the Intraday Statistical Arbitrage System.
  - Path: `src/optimization/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Genetic Algorithm optimization for pairs trading strategy parameters.
  - Path: `src/optimization/genetic_algorithm/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Grid Search optimization for pairs trading strategy parameters.
  - Path: `src/optimization/grid_search/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Walk-Forward Testing framework for pairs trading strategy.
  - Path: `src/optimization/walk_forward/__init__.py`
  - Classes: 0, Functions: 0

- **adaptive_parameter_manager** - Adaptive Parameter Manager Module
  - Path: `src/optimization/adaptive_parameter_manager.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.ml_enhancements.regime_detection.market_regime_classifier.MarketRegimeClassifier

- **genetic_algorithm** - Genetic Algorithm Optimization for Pairs Trading Strategy.
  - Path: `src/optimization/genetic_algorithm/genetic_algorithm.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.optimization.parameter_space.ParameterSpace, src.optimization.parameter_space.create_default_parameter_space, src.pairs_trading_strategy.PairsTradingStrategy

- **grid_search** - Grid Search Optimization for Pairs Trading Strategy.
  - Path: `src/optimization/grid_search/grid_search.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.optimization.parameter_space.ParameterSpace, src.optimization.parameter_space.create_default_parameter_space, src.pairs_trading_strategy.PairsTradingStrategy

- **intraday_parameter_optimizer** - Intraday Parameter Optimizer Module
  - Path: `src/optimization/intraday_parameter_optimizer.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.backtest.intraday_backtest_engine.IntradayBacktestEngine, src.ml_enhancements.regime_detection.market_regime_classifier.MarketRegimeClassifier, src.optimization.grid_search.grid_search.GridSearchOptimizer, src.optimization.parameter_space.ParameterSpace, src.optimization.parameter_space.create_default_parameter_space

- **parameter_space** - Parameter Space definitions for optimization.
  - Path: `src/optimization/parameter_space.py`
  - Classes: 1, Functions: 1

- **walk_forward** - Walk-Forward Testing Framework for Pairs Trading Strategy.
  - Path: `src/optimization/walk_forward/walk_forward.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.optimization.genetic_algorithm.genetic_algorithm.GeneticOptimizer, src.optimization.grid_search.grid_search.GridSearchOptimizer, src.optimization.parameter_space.ParameterSpace, src.optimization.parameter_space.create_default_parameter_space, src.pairs_trading_strategy.PairsTradingStrategy

### pair_trading (2)

- **__init__** - Pair Trading module for the Intraday Statistical Arbitrage System.
  - Path: `src/pair_trading/__init__.py`
  - Classes: 0, Functions: 0

- **pair_finder** - Pair finder module for the Intraday Statistical Arbitrage System.
  - Path: `src/pair_trading/pair_finder.py`
  - Classes: 1, Functions: 0

### paper_trading (4)

- **__init__** - Paper Trading Module for the Intraday Statistical Arbitrage System.
  - Path: `src/paper_trading/__init__.py`
  - Classes: 0, Functions: 0

- **example** - Example script for using the PaperTrader with Interactive Brokers.
  - Path: `src/paper_trading/example.py`
  - Classes: 0, Functions: 7
  - Dependencies: src.paper_trading.PaperTrader

- **intraday_ml_paper_trader** - Intraday ML-Enhanced Paper Trader
  - Path: `src/paper_trading/intraday_ml_paper_trader.py`
  - Classes: 1, Functions: 1

- **paper_trader** - Paper Trader for the Intraday Statistical Arbitrage System.
  - Path: `src/paper_trading/paper_trader.py`
  - Classes: 1, Functions: 0

### performance (11)

- **__init__** - Performance Enhancement Module for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Efficient Data Structures for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/data_structures/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Parallel Processing Module for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/parallel/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Caching Module for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/caching/__init__.py`
  - Classes: 0, Functions: 0

- **compressed_ohlc** - Compressed OHLC Data Structure for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/data_structures/compressed_ohlc.py`
  - Classes: 1, Functions: 0

- **data_cache** - Data Cache for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/caching/data_cache.py`
  - Classes: 1, Functions: 0

- **function_cache** - Function Caching for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/caching/function_cache.py`
  - Classes: 1, Functions: 4

- **parallel_executor** - Parallel Executor for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/parallel/parallel_executor.py`
  - Classes: 1, Functions: 1

- **rolling_window** - Rolling Window Data Structures for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/data_structures/rolling_window.py`
  - Classes: 2, Functions: 0

- **sparse_time_series** - Sparse Time Series Data Structure for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/data_structures/sparse_time_series.py`
  - Classes: 3, Functions: 0

- **task_pool** - Task Pool for the Intraday Statistical Arbitrage System.
  - Path: `src/performance/parallel/task_pool.py`
  - Classes: 4, Functions: 0

### portfolio (1)

- **pairs_portfolio** - Pairs Trading Portfolio Module
  - Path: `src/portfolio/pairs_portfolio.py`
  - Classes: 1, Functions: 0

### reporting (4)

- **__init__** - Reporting Module for the Intraday Statistical Arbitrage System.
  - Path: `src/reporting/__init__.py`
  - Classes: 0, Functions: 0

- **example** - Example usage of the reporting module for generating backtest reports.
  - Path: `src/reporting/example.py`
  - Classes: 0, Functions: 2
  - Dependencies: src.reporting.BacktestReportGenerator, src.reporting.calculate_performance_metrics

- **metrics** - Performance Metrics for the Intraday Statistical Arbitrage System.
  - Path: `src/reporting/metrics.py`
  - Classes: 0, Functions: 6

- **report_generator** - Backtest Report Generator for the Intraday Statistical Arbitrage System.
  - Path: `src/reporting/report_generator.py`
  - Classes: 1, Functions: 0

### risk_management (3)

- **__init__** - Risk Management module for the Intraday Statistical Arbitrage System.
  - Path: `src/risk_management/__init__.py`
  - Classes: 0, Functions: 0

- **position_sizer** - Position sizer module for the Intraday Statistical Arbitrage System.
  - Path: `src/risk_management/position_sizer.py`
  - Classes: 1, Functions: 0

- **risk_manager** - Risk manager module for the Intraday Statistical Arbitrage System.
  - Path: `src/risk_management/risk_manager.py`
  - Classes: 1, Functions: 0

### scripts (1)

- **generate_test_data** - Test data generation script for the Intraday Statistical Arbitrage System.
  - Path: `scripts/generate_test_data.py`
  - Classes: 0, Functions: 4
  - Dependencies: src.utils.create_directory

### signal_generation (4)

- **__init__** - Signal Generation module for the Intraday Statistical Arbitrage System.
  - Path: `src/signal_generation/__init__.py`
  - Classes: 0, Functions: 0

- **pairs_signal_generator** - Pairs Trading Signal Generator Module
  - Path: `src/signal_generation/pairs_signal_generator.py`
  - Classes: 1, Functions: 0

- **signal_generator** - Signal generator module for the Intraday Statistical Arbitrage System.
  - Path: `src/signal_generation/signal_generator.py`
  - Classes: 1, Functions: 0

- **spread_calculator** - No description
  - Path: `src/signal_generation/spread_calculator.py`
  - Classes: 1, Functions: 0

### signals (2)

- **__init__** - Signal generation module for generating trading signals.
  - Path: `src/signals/__init__.py`
  - Classes: 0, Functions: 0

- **signal_generator** - No description
  - Path: `src/signals/signal_generator.py`
  - Classes: 2, Functions: 0

### spread_analytics (2)

- **__init__** - Spread Analytics module for the Intraday Statistical Arbitrage System.
  - Path: `src/spread_analytics/__init__.py`
  - Classes: 0, Functions: 0

- **spread_analyzer** - Spread analyzer module for the Intraday Statistical Arbitrage System.
  - Path: `src/spread_analytics/spread_analyzer.py`
  - Classes: 1, Functions: 0

### src (4)

- **__init__** - Quant Trader package initialization.
  - Path: `src/__init__.py`
  - Classes: 0, Functions: 0

- **main** - Main module for the Intraday Statistical Arbitrage System.
  - Path: `src/main.py`
  - Classes: 0, Functions: 5
  - Dependencies: src.backtest.BacktestEngine, src.cointegration.PairFinder, src.signals.SignalGenerator, src.signals.SignalType, src.utils.create_directory
    and 4 more...

- **pairs_trading_strategy** - No description
  - Path: `src/pairs_trading_strategy.py`
  - Classes: 1, Functions: 1
  - Dependencies: src.data_processing.data_processor.DataProcessor, src.risk_management.position_sizer.PositionSizer, src.risk_management.risk_manager.RiskManager, src.signal_generation.signal_generator.SignalGenerator, src.signal_generation.spread_calculator.SpreadCalculator
    and 1 more...

- **test_pair_selection** - No description
  - Path: `src/test_pair_selection.py`
  - Classes: 0, Functions: 1
  - Dependencies: src.cointegration.cointegration_tests.rolling_cointegration, src.cointegration.cointegration_tests.test_cointegration, src.cointegration.pair_finder.PairFinder, src.data_processing.data_processor.DataProcessor

### strategies (3)

- **__init__** - Strategies module for the Intraday Statistical Arbitrage System.
  - Path: `src/strategies/__init__.py`
  - Classes: 0, Functions: 0

- **base** - Base Strategy class for the Intraday Statistical Arbitrage System.
  - Path: `src/strategies/base.py`
  - Classes: 1, Functions: 0

- **multi_pair_portfolio** - Multi-Pair Portfolio Strategy Implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/strategies/multi_pair_portfolio.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.pair_trading.pair_finder.PairFinder, src.risk_management.position_sizer.PositionSizer, src.risk_management.risk_manager.RiskManager, src.signal_generation.signal_generator.SignalGenerator, src.strategies.base.Strategy
    and 2 more...

### strategy_variants (6)

- **__init__** - Strategy Variants Module for the Intraday Statistical Arbitrage System.
  - Path: `src/strategy_variants/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Machine Learning Signal Generation Module for the Intraday Statistical Arbitrage System.
  - Path: `src/strategy_variants/ml_signals/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Time Series Strategy Module for the Intraday Statistical Arbitrage System.
  - Path: `src/strategy_variants/time_series/__init__.py`
  - Classes: 0, Functions: 0

- **factory** - Strategy Factory Module for the Intraday Statistical Arbitrage System.
  - Path: `src/strategy_variants/factory.py`
  - Classes: 0, Functions: 1
  - Dependencies: src.pairs_trading_strategy.PairsTradingStrategy

- **ml_signal_strategy** - Machine Learning Signal Generation Strategy for the Intraday Statistical Arbitrage System.
  - Path: `src/strategy_variants/ml_signals/ml_signal_strategy.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.data_processor.data_processor.DataProcessor, src.pairs_trading_strategy.PairsTradingStrategy, src.risk_management.risk_manager.RiskManager, src.signal_generation.signal_generator.SignalGenerator, src.spread_analytics.spread_analyzer.SpreadAnalyzer

- **time_series_strategy** - Time Series Strategy Implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/strategy_variants/time_series/time_series_strategy.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.data_processor.data_processor.DataProcessor, src.pairs_trading_strategy.PairsTradingStrategy, src.risk_management.risk_manager.RiskManager, src.signal_generation.signal_generator.SignalGenerator, src.spread_analytics.spread_analyzer.SpreadAnalyzer

### tests (31)

- **__init__** - Tests for the Intraday Statistical Arbitrage System.
  - Path: `tests/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Unit tests for the Intraday Statistical Arbitrage System.
  - Path: `tests/unit/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Unit tests for the asset classes module.
  - Path: `tests/unit/asset_classes/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Unit tests for the cryptocurrency asset classes.
  - Path: `tests/unit/asset_classes/cryptocurrencies/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Unit tests for the fixed income asset classes.
  - Path: `tests/unit/asset_classes/fixed_income/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Unit tests for the equity asset classes.
  - Path: `tests/unit/asset_classes/equities/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Unit tests for the strategies module.
  - Path: `tests/unit/strategies/__init__.py`
  - Classes: 0, Functions: 0

- **mock_exchange** - Mock Exchange for testing purposes.
  - Path: `tests/mocks/mock_exchange.py`
  - Classes: 4, Functions: 0

- **mock_ib_connector** - Mock IB Connector for testing purposes.
  - Path: `tests/mocks/mock_ib_connector.py`
  - Classes: 1, Functions: 2

- **mock_market_data** - Mock Market Data Provider for testing purposes.
  - Path: `tests/mocks/mock_market_data.py`
  - Classes: 1, Functions: 0

- **test_base** - Unit tests for the base asset and asset class implementations.
  - Path: `tests/unit/asset_classes/test_base.py`
  - Classes: 4, Functions: 0
  - Dependencies: src.asset_classes.base.Asset, src.asset_classes.base.AssetClass

- **test_cointegration_tests** - Unit tests for the cointegration_tests module.
  - Path: `tests/unit/cointegration/test_cointegration_tests.py`
  - Classes: 1, Functions: 2
  - Dependencies: src.cointegration.cointegration_tests.calculate_half_life, src.cointegration.cointegration_tests.engle_granger_test, src.cointegration.cointegration_tests.johansen_test, src.cointegration.cointegration_tests.test_cointegration, src.cointegration.cointegration_tests.test_pairs_universe

- **test_crypto_asset** - Unit tests for the cryptocurrency asset and asset class implementations.
  - Path: `tests/unit/asset_classes/cryptocurrencies/test_crypto_asset.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.asset_classes.cryptocurrencies.crypto_asset.CryptoAsset, src.asset_classes.cryptocurrencies.crypto_asset.CryptoAssetClass, src.connectors.ib.IBConnector

- **test_data_flow** - Integration tests for the data flow between components.
  - Path: `tests/integration/test_data_flow.py`
  - Classes: 1, Functions: 8
  - Dependencies: src.cointegration.pair_finder.PairFinder, src.data_processor.data_processor.DataProcessor, src.signal_generation.signal_generator.SignalGenerator, src.spread_analytics.spread_analyzer.SpreadAnalyzer, tests.mocks.mock_ib_connector.MockIBConnector

- **test_data_processor** - Unit tests for the DataProcessor class.
  - Path: `tests/unit/data_processor/test_data_processor.py`
  - Classes: 1, Functions: 3
  - Dependencies: src.data_processor.data_processor.DataProcessor, tests.mocks.mock_ib_connector.MockIBConnector

- **test_equity_asset** - Unit tests for the equity asset and asset class implementations.
  - Path: `tests/unit/asset_classes/equities/test_equity_asset.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.asset_classes.equities.equity_asset.EquityAsset, src.asset_classes.equities.equity_asset.EquityAssetClass, src.connectors.ib.IBConnector

- **test_factory** - Unit tests for the asset factory implementation.
  - Path: `tests/unit/asset_classes/test_factory.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.asset_classes.equities.equity_asset.EquityAsset, src.asset_classes.equities.equity_asset.EquityAssetClass, src.asset_classes.factory.get_asset_class, src.asset_classes.futures.futures_asset.FuturesAsset, src.asset_classes.futures.futures_asset.FuturesAssetClass
    and 7 more...

- **test_fixed_income_asset** - Unit tests for the fixed income asset and asset class implementations.
  - Path: `tests/unit/asset_classes/fixed_income/test_fixed_income_asset.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.asset_classes.fixed_income.fixed_income_asset.FixedIncomeAsset, src.asset_classes.fixed_income.fixed_income_asset.FixedIncomeAssetClass, src.connectors.ib.IBConnector

- **test_futures_asset** - Unit tests for the FuturesAsset class.
  - Path: `tests/unit/asset_classes/futures/test_futures_asset.py`
  - Classes: 1, Functions: 2
  - Dependencies: src.asset_classes.futures.futures_asset.FuturesAsset, tests.mocks.mock_ib_connector.MockIBConnector

- **test_ib_integration** - Integration tests for Interactive Brokers TWS connectivity.
  - Path: `tests/integration/test_ib_integration.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.asset_classes.factory.create_asset, src.connectors.ib.ib_connector.IBConnector

- **test_intraday_backtest_engine** - Tests for the IntradayBacktestEngine class.
  - Path: `tests/test_intraday_backtest_engine.py`
  - Classes: 1, Functions: 0
  - Dependencies: src.backtest.intraday_backtest_engine.IntradayBacktestEngine

- **test_intraday_ml_integration** - Integration test for the intraday ML system.
  - Path: `tests/integration/test_intraday_ml_integration.py`
  - Classes: 0, Functions: 5
  - Dependencies: src.backtest.intraday_backtest_engine.IntradayBacktestEngine, src.ml_enhancements.feature_engineering.advanced_features.AdvancedFeatureEngineering, src.ml_enhancements.intraday_integration.IntradayMLSystem, src.ml_enhancements.regime_detection.market_regime_classifier.MarketRegimeClassifier, src.optimization.adaptive_parameter_manager.AdaptiveParameterManager

- **test_intraday_ml_system_integration** - Integration tests for the Intraday ML System focused on pipeline integrity,
  - Path: `tests/integration/test_intraday_ml_system_integration.py`
  - Classes: 0, Functions: 7
  - Dependencies: src.backtest.intraday_backtest_engine.IntradayBacktestEngine, src.ml_enhancements.intraday_integration.IntradayMLSystem, src.ml_enhancements.regime_detection.market_regime_classifier.MarketRegimeClassifier, src.optimization.adaptive_parameter_manager.AdaptiveParameterManager, src.paper_trading.intraday_ml_paper_trader.IntradayMLPaperTrader
    and 1 more...

- **test_multi_asset_integration** - Integration tests for multi-asset class support.
  - Path: `tests/integration/test_multi_asset_integration.py`
  - Classes: 1, Functions: 4
  - Dependencies: src.asset_classes.base.AssetClass, src.asset_classes.futures.futures_asset.FuturesAsset, src.asset_classes.futures.futures_asset.FuturesAssetClass, src.data_processor.data_processor.DataProcessor, src.pairs_trading_strategy.PairsTradingStrategy
    and 5 more...

- **test_multi_pair_portfolio** - Unit tests for the multi-pair portfolio strategy implementation.
  - Path: `tests/unit/strategies/test_multi_pair_portfolio.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.pair_trading.pair_finder.PairFinder, src.risk_management.position_sizer.PositionSizer, src.risk_management.risk_manager.RiskManager, src.signal_generation.signal_generator.SignalGenerator, src.strategies.multi_pair_portfolio.MultiPairPortfolio
    and 2 more...

- **test_multi_pair_portfolio** - Unit tests for the multi-pair portfolio strategy.
  - Path: `tests/unit/strategy_variants/test_multi_pair_portfolio.py`
  - Classes: 2, Functions: 0
  - Dependencies: src.pair_trading.pair_finder.PairFinder, src.risk_management.position_sizer.PositionSizer, src.risk_management.risk_manager.RiskManager, src.signal_generation.signal_generator.SignalGenerator, src.strategies.multi_pair_portfolio.MultiPairPortfolio
    and 2 more...

- **test_paper_trader** - Unit tests for the PaperTrader class.
  - Path: `tests/unit/paper_trading/test_paper_trader.py`
  - Classes: 1, Functions: 3
  - Dependencies: src.paper_trading.paper_trader.PaperTrader, tests.mocks.mock_ib_connector.MockIBConnector

- **test_risk_manager** - Unit tests for the RiskManager class.
  - Path: `tests/unit/risk_management/test_risk_manager.py`
  - Classes: 1, Functions: 3
  - Dependencies: src.risk_management.risk_manager.RiskManager

- **test_signal_generator** - Unit tests for the SignalGenerator class.
  - Path: `tests/unit/signal_generation/test_signal_generator.py`
  - Classes: 1, Functions: 4
  - Dependencies: src.signal_generation.signal_generator.SignalGenerator

- **test_spread_analyzer** - Unit tests for the SpreadAnalyzer class.
  - Path: `tests/unit/spread_analytics/test_spread_analyzer.py`
  - Classes: 1, Functions: 3
  - Dependencies: src.spread_analytics.spread_analyzer.SpreadAnalyzer

- **test_strategy_execution** - Integration tests for the strategy execution flow.
  - Path: `tests/integration/test_strategy_execution.py`
  - Classes: 1, Functions: 3
  - Dependencies: src.data_processor.data_processor.DataProcessor, src.pairs_trading_strategy.PairsTradingStrategy, src.signal_generation.signal_generator.SignalGenerator, src.spread_analytics.spread_analyzer.SpreadAnalyzer, tests.mocks.mock_ib_connector.MockIBConnector

### utils (2)

- **__init__** - Utility functions for common operations.
  - Path: `src/utils/__init__.py`
  - Classes: 0, Functions: 0

- **utils** - No description
  - Path: `src/utils/utils.py`
  - Classes: 0, Functions: 7

### visualization (2)

- **__init__** - Visualization module for generating plots and charts.
  - Path: `src/visualization/__init__.py`
  - Classes: 0, Functions: 0

- **plotting** - No description
  - Path: `src/visualization/plotting.py`
  - Classes: 0, Functions: 5

### web_interface (14)

- **__init__** - Web Interface Module for the Intraday Statistical Arbitrage System.
  - Path: `src/web_interface/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Configuration blueprint for the web interface.
  - Path: `src/web_interface/blueprints/configuration/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Authentication blueprint for the web interface.
  - Path: `src/web_interface/blueprints/auth/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Dashboard blueprint for the web interface.
  - Path: `src/web_interface/blueprints/dashboard/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - API blueprint for the web interface.
  - Path: `src/web_interface/blueprints/api/__init__.py`
  - Classes: 0, Functions: 0

- **__init__** - Strategy blueprint for the web interface.
  - Path: `src/web_interface/blueprints/strategy/__init__.py`
  - Classes: 0, Functions: 0

- **app** - Flask application module for the web interface.
  - Path: `src/web_interface/app.py`
  - Classes: 0, Functions: 1

- **models** - Models for the web interface.
  - Path: `src/web_interface/models.py`
  - Classes: 1, Functions: 0

- **routes** - Routes for the configuration blueprint.
  - Path: `src/web_interface/blueprints/configuration/routes.py`
  - Classes: 0, Functions: 5

- **routes** - Routes for the authentication blueprint.
  - Path: `src/web_interface/blueprints/auth/routes.py`
  - Classes: 0, Functions: 7

- **routes** - Routes for the dashboard blueprint.
  - Path: `src/web_interface/blueprints/dashboard/routes.py`
  - Classes: 0, Functions: 6

- **routes** - Routes for the API blueprint.
  - Path: `src/web_interface/blueprints/api/routes.py`
  - Classes: 0, Functions: 10

- **routes** - Routes for the strategy blueprint.
  - Path: `src/web_interface/blueprints/strategy/routes.py`
  - Classes: 0, Functions: 9

- **utils** - Utility functions for the web interface.
  - Path: `src/web_interface/utils.py`
  - Classes: 0, Functions: 6

## Key Classes

### asset_classes (10)

- **Asset** - Base class for all asset types.
  - Path: `src/asset_classes/base.py`
  - Inherits from: abc.ABC
  - Methods: 7

- **AssetClass** - Base class for all asset classes.
  - Path: `src/asset_classes/base.py`
  - Inherits from: abc.ABC
  - Methods: 9

- **CryptoAsset** - Cryptocurrency asset implementation for trading digital currencies.
  - Path: `src/asset_classes/cryptocurrencies/crypto_asset.py`
  - Inherits from: Asset
  - Methods: 8

- **CryptoAssetClass** - Cryptocurrency asset class implementation.
  - Path: `src/asset_classes/cryptocurrencies/crypto_asset.py`
  - Inherits from: AssetClass
  - Methods: 7

- **EquityAsset** - Equity asset implementation for trading stocks and ETFs.
  - Path: `src/asset_classes/equities/equity_asset.py`
  - Inherits from: Asset
  - Methods: 8

- **EquityAssetClass** - Equity asset class implementation.
  - Path: `src/asset_classes/equities/equity_asset.py`
  - Inherits from: AssetClass
  - Methods: 8

- **FixedIncomeAsset** - Fixed Income asset implementation for trading bonds and other debt instruments.
  - Path: `src/asset_classes/fixed_income/fixed_income_asset.py`
  - Inherits from: Asset
  - Methods: 7

- **FixedIncomeAssetClass** - Fixed Income asset class implementation.
  - Path: `src/asset_classes/fixed_income/fixed_income_asset.py`
  - Inherits from: AssetClass
  - Methods: 8

- **FuturesAsset** - Futures asset implementation for trading futures contracts.
  - Path: `src/asset_classes/futures/futures_asset.py`
  - Inherits from: Asset
  - Methods: 8

- **FuturesAssetClass** - Futures asset class implementation.
  - Path: `src/asset_classes/futures/futures_asset.py`
  - Inherits from: AssetClass
  - Methods: 7

### backtest (2)

- **BacktestEngine** - Backtesting engine for pairs trading strategies.
  - Path: `src/backtest/backtest_engine.py`
  - Methods: 15

- **IntradayBacktestEngine** - Specialized backtesting engine for intraday trading strategies.
  - Path: `src/backtest/intraday_backtest_engine.py`
  - Inherits from: BacktestEngine
  - Methods: 12

### codebase_review (1)

- **CodebaseAnalyzer** - Analyzes the codebase and generates reports.
  - Path: `codebase_review/codebase_analyzer.py`
  - Methods: 11

### cointegration (2)

- **IntradayPairAnalyzer** - Analyze cointegration relationships at intraday timeframes.
  - Path: `src/cointegration/intraday_pair_analyzer.py`
  - Methods: 8

- **PairsFinder** - Identify cointegrated pairs of financial instruments.
  - Path: `src/cointegration/pairs_finder.py`
  - Methods: 6

### config (2)

- **ConfigurationManager** - Manages configuration settings for the pairs trading system.
  - Path: `src/config/configuration.py`
  - Methods: 9

- **TimeFilter** - Enum for time-based filtering
  - Path: `src/config/configuration.py`
  - Inherits from: Enum
  - Methods: 0

### connectors (2)

- **FixedIBConnector** - Fixed version of the IBConnector for testing integration with TWS.
  - Path: `src/connectors/ib/fixed_connector.py`
  - Methods: 9

- **IBConnector** - Interactive Brokers (IB) Connector for the Intraday Statistical Arbitrage System.
  - Path: `src/connectors/ib/ib_connector.py`
  - Methods: 32

### data_processor (3)

- **DataProcessor** - Processes and manages data for the pairs trading system.
  - Path: `src/data_processor/data_processor.py`
  - Methods: 12

- **FuturesDataProcessor** - Process futures data files into a standardized format for analysis.
  - Path: `src/data_processor/futures_processor.py`
  - Methods: 5

- **IntradayDataProcessor** - Specialized processor for intraday data with optimizations for pairs trading.
  - Path: `src/data_processor/intraday_processor.py`
  - Methods: 8

### live_trading (5)

- **LiveTrader** - Live Trading implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/live_trading/live_trader.py`
  - Methods: 30

- **MonitoredLiveTrader** - Combined LiveTrader and TradingMonitor for a full monitored live trading setup.
  - Path: `src/live_trading/example_with_monitoring.py`
  - Methods: 15

- **PositionTracker** - Position tracking system that provides:
  - Path: `src/live_trading/position_tracker.py`
  - Methods: 24

- **TrackedLiveTrader** - LiveTrader extended with PositionTracker for comprehensive position management.
  - Path: `src/live_trading/example_with_position_tracking.py`
  - Methods: 15

- **TradingMonitor** - Trading monitoring system that provides:
  - Path: `src/live_trading/monitoring.py`
  - Methods: 21

### ml_enhancements (14)

- **AdvancedFeatureEngineering** - Advanced feature engineering for intraday trading.
  - Path: `src/ml_enhancements/feature_engineering/advanced_features.py`
  - Methods: 9

- **EnsembleModelFactory** - Factory class for creating and managing ensemble models.
  - Path: `src/ml_enhancements/ensemble_models.py`
  - Methods: 13

- **FeatureGenerator** - Feature Generator for Pairs Trading.
  - Path: `src/ml_enhancements/feature_engineering/feature_generator.py`
  - Methods: 10

- **IntradayFeatureEngineering** - Generate and analyze features specifically for intraday trading strategies.
  - Path: `src/ml_enhancements/feature_engineering/intraday_features.py`
  - Methods: 10

- **IntradayMLSystem** - Integrated ML system for intraday trading.
  - Path: `src/ml_enhancements/intraday_integration.py`
  - Methods: 10

- **IntradayModelTrainer** - Trainer for machine learning models focused on intraday trading signals.
  - Path: `src/ml_enhancements/intraday_model_trainer.py`
  - Methods: 8

- **IntradaySignalEnhancer** - Enhances statistical arbitrage signals with machine learning for intraday trading.
  - Path: `src/ml_enhancements/intraday_signals.py`
  - Methods: 15

- **IntradaySignalProcessor** - Process and apply intraday signal enhancements to a pairs trading strategy.
  - Path: `src/ml_enhancements/intraday_signals.py`
  - Methods: 3

- **MarketRegimeClassifier** - Detect market regimes to adapt trading parameters.
  - Path: `src/ml_enhancements/regime_detection/market_regime_classifier.py`
  - Methods: 9

- **ModelRetrainingManager** - Manages automated retraining of machine learning models for intraday trading.
  - Path: `src/ml_enhancements/model_retraining.py`
  - Methods: 13

- **RegimeDetector** - Regime Detector for Pairs Trading.
  - Path: `src/ml_enhancements/regime_detection/regime_detector.py`
  - Methods: 8

- **SpreadPredictor** - Spread Predictor for Pairs Trading.
  - Path: `src/ml_enhancements/spread_prediction/spread_predictor.py`
  - Methods: 9

- **TimeSeriesAugmentation** - Time series data augmentation techniques for financial data.
  - Path: `src/ml_enhancements/data_augmentation.py`
  - Methods: 10

- **WalkForwardValidator** - Implements walk-forward validation for time series data.
  - Path: `src/ml_enhancements/training_utils.py`
  - Methods: 4

### optimization (6)

- **AdaptiveParameterManager** - Manages and applies regime-specific parameters for intraday trading.
  - Path: `src/optimization/adaptive_parameter_manager.py`
  - Methods: 12

- **GeneticOptimizer** - Genetic Algorithm Optimizer for Pairs Trading Strategy.
  - Path: `src/optimization/genetic_algorithm/genetic_algorithm.py`
  - Methods: 11

- **GridSearchOptimizer** - Grid Search Optimizer for Pairs Trading Strategy.
  - Path: `src/optimization/grid_search/grid_search.py`
  - Methods: 6

- **IntradayParameterOptimizer** - Intraday Parameter Optimizer with regime-specific parameter optimization.
  - Path: `src/optimization/intraday_parameter_optimizer.py`
  - Methods: 8

- **ParameterSpace** - Defines the parameter space for optimization.
  - Path: `src/optimization/parameter_space.py`
  - Methods: 6

- **WalkForwardTester** - Walk-Forward Testing Framework for Pairs Trading Strategy.
  - Path: `src/optimization/walk_forward/walk_forward.py`
  - Methods: 9

### pair_trading (1)

- **PairFinder** - Pair finder class for identifying cointegrated pairs of assets.
  - Path: `src/pair_trading/pair_finder.py`
  - Methods: 2

### paper_trading (2)

- **IntradayMLPaperTrader** - Paper trading implementation with ML enhancements for intraday trading.
  - Path: `src/paper_trading/intraday_ml_paper_trader.py`
  - Methods: 30

- **PaperTrader** - Paper Trading implementation for the Intraday Statistical Arbitrage System.
  - Path: `src/paper_trading/paper_trader.py`
  - Methods: 28

### performance (13)

- **CompressedOHLC** - Memory-efficient compressed OHLC (Open, High, Low, Close) data structure.
  - Path: `src/performance/data_structures/compressed_ohlc.py`
  - Methods: 19

- **DataCache** - Data Cache for storing and retrieving frequently used data.
  - Path: `src/performance/caching/data_cache.py`
  - Methods: 17

- **ExponentialRollingWindow** - Exponential Rolling Window for time series data.
  - Path: `src/performance/data_structures/rolling_window.py`
  - Methods: 9

- **ParallelExecutor** - Parallel Executor for running tasks in parallel.
  - Path: `src/performance/parallel/parallel_executor.py`
  - Methods: 3

- **RollingWindow** - Efficient Rolling Window for time series data.
  - Path: `src/performance/data_structures/rolling_window.py`
  - Methods: 19

- **SparseTimeSeries** - Memory-efficient sparse time series optimized for financial data.
  - Path: `src/performance/data_structures/sparse_time_series.py`
  - Methods: 19

- **SparseTimeSeriesEWM** - Helper class for exponentially weighted operations on SparseTimeSeries.
  - Path: `src/performance/data_structures/sparse_time_series.py`
  - Methods: 5

- **SparseTimeSeriesRolling** - Helper class for rolling window operations on SparseTimeSeries.
  - Path: `src/performance/data_structures/sparse_time_series.py`
  - Methods: 9

- **Task** - Task metadata for the task pool.
  - Path: `src/performance/parallel/task_pool.py`
  - Methods: 0

- **TaskPool** - Task Pool for managing long-running parallel tasks.
  - Path: `src/performance/parallel/task_pool.py`
  - Methods: 10

- **TaskPriority** - Priority levels for tasks.
  - Path: `src/performance/parallel/task_pool.py`
  - Inherits from: Enum
  - Methods: 0

- **TaskStatus** - Status of a task in the task pool.
  - Path: `src/performance/parallel/task_pool.py`
  - Inherits from: Enum
  - Methods: 0

- **_TimedCache** - Internal class for implementing timed caching.
  - Path: `src/performance/caching/function_cache.py`
  - Methods: 4

### portfolio (1)

- **PairsPortfolio** - Manage a portfolio of pairs trades.
  - Path: `src/portfolio/pairs_portfolio.py`
  - Methods: 11

### reporting (1)

- **BacktestReportGenerator** - Backtest Report Generator for the Intraday Statistical Arbitrage System.
  - Path: `src/reporting/report_generator.py`
  - Methods: 12

### risk_management (2)

- **PositionSizer** - Position sizer class for calculating appropriate position sizes.
  - Path: `src/risk_management/position_sizer.py`
  - Methods: 5

- **RiskManager** - Risk manager class for controlling risk in trading strategies.
  - Path: `src/risk_management/risk_manager.py`
  - Methods: 2

### signal_generation (2)

- **PairsSignalGenerator** - Generate trading signals for pairs trading strategies.
  - Path: `src/signal_generation/pairs_signal_generator.py`
  - Methods: 10

- **SpreadCalculator** - Class for calculating and normalizing spreads between two assets.
  - Path: `src/signal_generation/spread_calculator.py`
  - Methods: 9

### signals (2)

- **SignalGenerator** - Generates trading signals based on spread analysis.
  - Path: `src/signals/signal_generator.py`
  - Methods: 11

- **SignalType** - Enum for signal types.
  - Path: `src/signals/signal_generator.py`
  - Inherits from: Enum
  - Methods: 0

### spread_analytics (1)

- **SpreadAnalyzer** - Spread analyzer class for analyzing spreads between assets.
  - Path: `src/spread_analytics/spread_analyzer.py`
  - Methods: 14

### src (1)

- **PairsTradingStrategy** - Main class for the pairs trading strategy. Integrates all components:
  - Path: `src/pairs_trading_strategy.py`
  - Methods: 10

### strategies (2)

- **MultiPairPortfolio** - Multi-pair portfolio strategy that manages multiple pairs simultaneously.
  - Path: `src/strategies/multi_pair_portfolio.py`
  - Inherits from: Strategy
  - Methods: 9

- **Strategy** - Base class for all trading strategies.
  - Path: `src/strategies/base.py`
  - Inherits from: abc.ABC
  - Methods: 5

### strategy_variants (2)

- **MLSignalStrategy** - Machine Learning Signal Strategy for pairs trading that extends the base
  - Path: `src/strategy_variants/ml_signals/ml_signal_strategy.py`
  - Inherits from: PairsTradingStrategy
  - Methods: 14

- **TimeSeriesStrategy** - Time Series Strategy for pairs trading that extends the base
  - Path: `src/strategy_variants/time_series/time_series_strategy.py`
  - Inherits from: PairsTradingStrategy
  - Methods: 11

### tests (31)

- **ConcreteAsset** - Concrete implementation of Asset for testing.
  - Path: `tests/unit/asset_classes/test_base.py`
  - Inherits from: Asset
  - Methods: 4

- **ConcreteAssetClass** - Concrete implementation of AssetClass for testing.
  - Path: `tests/unit/asset_classes/test_base.py`
  - Inherits from: AssetClass
  - Methods: 4

- **MockAsset** - Mock asset implementation for testing.
  - Path: `tests/unit/strategy_variants/test_multi_pair_portfolio.py`
  - Inherits from: Asset
  - Methods: 4

- **MockExchange** - Mock implementation of an exchange for testing purposes.
  - Path: `tests/mocks/mock_exchange.py`
  - Methods: 18

- **MockIBConnector** - Mock implementation of IBConnector for testing purposes.
  - Path: `tests/mocks/mock_ib_connector.py`
  - Methods: 14

- **MockMarketDataProvider** - Mock implementation of a market data provider for testing purposes.
  - Path: `tests/mocks/mock_market_data.py`
  - Methods: 7

- **OrderSide** - Order sides supported by the mock exchange.
  - Path: `tests/mocks/mock_exchange.py`
  - Inherits from: Enum
  - Methods: 0

- **OrderStatus** - Order statuses supported by the mock exchange.
  - Path: `tests/mocks/mock_exchange.py`
  - Inherits from: Enum
  - Methods: 0

- **OrderType** - Order types supported by the mock exchange.
  - Path: `tests/mocks/mock_exchange.py`
  - Inherits from: Enum
  - Methods: 0

- **TestAsset** - Test case for the Asset base class.
  - Path: `tests/unit/asset_classes/test_base.py`
  - Inherits from: unittest.TestCase
  - Methods: 3

- **TestAssetClass** - Test case for the AssetClass base class.
  - Path: `tests/unit/asset_classes/test_base.py`
  - Inherits from: unittest.TestCase
  - Methods: 6

- **TestAssetFactory** - Test cases for asset factory functions.
  - Path: `tests/unit/asset_classes/test_factory.py`
  - Inherits from: unittest.TestCase
  - Methods: 12

- **TestCointegrationTests** - Test class for cointegration testing functions.
  - Path: `tests/unit/cointegration/test_cointegration_tests.py`
  - Methods: 5

- **TestCryptoAsset** - Test cases for the CryptoAsset class.
  - Path: `tests/unit/asset_classes/cryptocurrencies/test_crypto_asset.py`
  - Inherits from: unittest.TestCase
  - Methods: 17

- **TestCryptoAssetClass** - Test cases for the CryptoAssetClass class.
  - Path: `tests/unit/asset_classes/cryptocurrencies/test_crypto_asset.py`
  - Inherits from: unittest.TestCase
  - Methods: 10

- **TestDataFlow** - Integration tests for the data flow between components.
  - Path: `tests/integration/test_data_flow.py`
  - Methods: 5

- **TestDataProcessor** - Test class for DataProcessor.
  - Path: `tests/unit/data_processor/test_data_processor.py`
  - Methods: 9

- **TestEquityAsset** - Test cases for the EquityAsset class.
  - Path: `tests/unit/asset_classes/equities/test_equity_asset.py`
  - Inherits from: unittest.TestCase
  - Methods: 10

- **TestEquityAssetClass** - Test cases for the EquityAssetClass class.
  - Path: `tests/unit/asset_classes/equities/test_equity_asset.py`
  - Inherits from: unittest.TestCase
  - Methods: 14

- **TestFixedIncomeAsset** - Test cases for the FixedIncomeAsset class.
  - Path: `tests/unit/asset_classes/fixed_income/test_fixed_income_asset.py`
  - Inherits from: unittest.TestCase
  - Methods: 16

- **TestFixedIncomeAssetClass** - Test cases for the FixedIncomeAssetClass class.
  - Path: `tests/unit/asset_classes/fixed_income/test_fixed_income_asset.py`
  - Inherits from: unittest.TestCase
  - Methods: 14

- **TestFuturesAsset** - Test class for FuturesAsset.
  - Path: `tests/unit/asset_classes/futures/test_futures_asset.py`
  - Methods: 8

- **TestIBIntegration** - Integration tests with Interactive Brokers TWS.
  - Path: `tests/integration/test_ib_integration.py`
  - Inherits from: unittest.TestCase
  - Methods: 9

- **TestIntradayBacktestEngine** - Test cases for the IntradayBacktestEngine class.
  - Path: `tests/test_intraday_backtest_engine.py`
  - Inherits from: unittest.TestCase
  - Methods: 11

- **TestMultiAssetIntegration** - Integration tests for multi-asset support.
  - Path: `tests/integration/test_multi_asset_integration.py`
  - Methods: 4

- **TestMultiPairPortfolio** - Test cases for MultiPairPortfolio strategy.
  - Path: `tests/unit/strategy_variants/test_multi_pair_portfolio.py`
  - Inherits from: unittest.TestCase
  - Methods: 8

- **TestPaperTrader** - Test class for PaperTrader.
  - Path: `tests/unit/paper_trading/test_paper_trader.py`
  - Methods: 9

- **TestRiskManager** - Test class for RiskManager.
  - Path: `tests/unit/risk_management/test_risk_manager.py`
  - Methods: 11

- **TestSignalGenerator** - Test class for SignalGenerator.
  - Path: `tests/unit/signal_generation/test_signal_generator.py`
  - Methods: 8

- **TestSpreadAnalyzer** - Test class for SpreadAnalyzer.
  - Path: `tests/unit/spread_analytics/test_spread_analyzer.py`
  - Methods: 9

- **TestStrategyExecution** - Integration tests for the strategy execution flow.
  - Path: `tests/integration/test_strategy_execution.py`
  - Methods: 2

### web_interface (1)

- **User** - User model for authentication.
  - Path: `src/web_interface/models.py`
  - Inherits from: UserMixin
  - Methods: 7
