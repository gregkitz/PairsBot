# Quant-Trader Component Inventory

## Core Components

### Data Processing
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| DataLoader | `src/data_processor/data_loader.py` | Loads historical data from various sources | pandas, numpy | Implemented |
| DataProcessor | `src/data_processor/data_processor.py` | Processes and normalizes financial time series | pandas, numpy | Implemented |
| IntraDayProcessor | `src/data_processor/intraday_processor.py` | Specialized processing for intraday data | pandas, numpy, DataProcessor | Implemented |
| FeatureCalculator | `src/data_processor/feature_calculator.py` | Calculates standard technical indicators | pandas, numpy, ta-lib | Implemented |
| DataValidator | `src/data_management/data_validator.py` | Validates data quality and integrity | pandas | Implemented |

### Asset Classes
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| AssetBase | `src/asset_classes/asset_base.py` | Base class for all asset types | None | Implemented |
| FuturesContract | `src/asset_classes/futures_contract.py` | Represents a futures contract | AssetBase | Implemented |
| FuturesPair | `src/asset_classes/futures_pair.py` | Represents a pair of futures contracts | FuturesContract | Implemented |
| EquityAsset | `src/asset_classes/equity_asset.py` | Represents an equity instrument | AssetBase | Implemented |
| CryptoAsset | `src/asset_classes/crypto_asset.py` | Represents a cryptocurrency | AssetBase | Implemented |

### Pair Trading
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| PairsTradingStrategy | `src/pairs_trading_strategy.py` | Core implementation of pairs trading logic | numpy, pandas, SpreadAnalyzer, SignalGenerator | Implemented |
| PairSelector | `src/cointegration/pair_selector.py` | Identifies and ranks cointegrated pairs | numpy, pandas, CointegrationTests | Implemented |
| SpreadCalculator | `src/spread_analytics/spread_calculator.py` | Calculates spread between two assets | numpy, pandas | Implemented |
| PairPerformanceAnalyzer | `src/backtest/pair_performance.py` | Analyzes performance of trading pairs | numpy, pandas | Implemented |
| IntraDayPairAnalyzer | `src/cointegration/intraday_pair_analyzer.py` | Specialized pair analysis for intraday | pandas, numpy, CointegrationTests | Implemented |

### Cointegration
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| CointegrationTests | `src/cointegration/cointegration_tests.py` | Implements Engle-Granger and Johansen tests | statsmodels, numpy, pandas | Implemented |
| HalfLifeCalculator | `src/cointegration/half_life.py` | Calculates mean-reversion speed | numpy, pandas | Implemented |
| RollingCointegration | `src/cointegration/rolling_analysis.py` | Performs rolling window cointegration analysis | numpy, pandas, CointegrationTests | Implemented |
| PairStabilityAnalyzer | `src/cointegration/pair_stability.py` | Analyzes stability of cointegration relationship | numpy, pandas | Implemented |
| CointegrationBreakdownDetector | `src/ml_enhancements/regime_detection/regime_detector.py` | Detects breakdowns in cointegration | numpy, pandas, scikit-learn | Implemented |

### Signal Generation
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| SignalGenerator | `src/signal_generation/signal_generator.py` | Generates trading signals based on statistical thresholds | numpy, pandas | Implemented |
| ZScoreSignalGenerator | `src/signal_generation/zscore_signals.py` | Specialized signal generation using z-scores | numpy, pandas, SignalGenerator | Implemented |
| FilterFactory | `src/signal_generation/filter_factory.py` | Creates filters for signal confirmation | numpy, pandas | Implemented |
| SignalCombiner | `src/signal_generation/signal_combiner.py` | Combines signals from multiple sources | numpy, pandas | Implemented |
| EnhancedSignalGenerator | `src/ml_enhancements/intraday_signals.py` | ML-enhanced signal generation | numpy, pandas, scikit-learn | Implemented |

### Portfolio Management
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| PortfolioManager | `src/portfolio/portfolio_manager.py` | Manages overall portfolio allocation | numpy, pandas, RiskManager | Implemented |
| PortfolioOptimizer | `src/portfolio/portfolio_optimizer.py` | Optimizes capital allocation across pairs | numpy, scipy | Implemented |
| PositionTracker | `src/portfolio/position_tracker.py` | Tracks open positions and their P&L | pandas | Implemented |
| RebalanceManager | `src/portfolio/rebalance_manager.py` | Manages portfolio rebalancing | numpy, pandas | Implemented |
| PortfolioConstraints | `src/portfolio/portfolio_constraints.py` | Enforces portfolio constraints | numpy | Implemented |

### Risk Management
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| RiskManager | `src/risk_management/risk_manager.py` | Manages overall risk exposure | numpy, pandas | Implemented |
| PositionSizer | `src/risk_management/position_sizer.py` | Calculates appropriate position sizes | numpy | Implemented |
| AdaptivePositionSizing | `src/risk_management/adaptive_position_sizing.py` | Volatility-based position sizing | numpy, pandas | Implemented |
| CorrelationManager | `src/risk_management/correlation_manager.py` | Manages correlation risk | numpy, pandas | Implemented |
| DrawdownManager | `src/risk_management/drawdown_manager.py` | Manages drawdown limits | numpy, pandas | Implemented |
| RiskMetrics | `src/risk_management/risk_metrics.py` | Calculates risk metrics | numpy, pandas | Implemented |

## ML Enhancements

### Feature Engineering
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| FeatureGenerator | `src/ml_enhancements/feature_engineering/feature_generator.py` | Base class for feature generation | numpy, pandas | Implemented |
| AdvancedFeatureEngineering | `src/ml_enhancements/feature_engineering/advanced_features.py` | Advanced ML-specific features | numpy, pandas, FeatureGenerator | Implemented |
| IntradayFeatureEngineering | `src/ml_enhancements/feature_engineering/intraday_features.py` | Intraday-specific features | numpy, pandas, FeatureGenerator | Implemented |
| FeatureImportanceAnalyzer | `src/ml_enhancements/feature_engineering/feature_importance.py` | Analyzes feature importance | scikit-learn, pandas | Partially Implemented |
| FeatureSelector | `src/ml_enhancements/feature_engineering/feature_selector.py` | Selects optimal features | scikit-learn, pandas | Partially Implemented |

### Regime Detection
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| RegimeDetector | `src/ml_enhancements/regime_detection/regime_detector.py` | Detects market regimes and cointegration breakdowns | numpy, pandas, scikit-learn | Implemented |
| MarketRegimeClassifier | `src/ml_enhancements/regime_detection/market_regime_classifier.py` | Classifies market regimes | numpy, pandas, scikit-learn | Implemented |
| RegimeBasedAdapter | `src/ml_enhancements/regime_detection/regime_adapter.py` | Adapts strategy parameters based on regime | numpy, pandas | Implemented |
| VolatilityRegimeDetector | `src/ml_enhancements/regime_detection/volatility_regime.py` | Specialized detection for volatility regimes | numpy, pandas | Implemented |
| RegimeTransitionPredictor | `src/ml_enhancements/regime_detection/transition_predictor.py` | Predicts regime transitions | scikit-learn, pandas | Partially Implemented |

### Spread Prediction
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| SpreadPredictor | `src/ml_enhancements/spread_prediction/spread_predictor.py` | Predicts future spread movements | scikit-learn, numpy, pandas | Implemented |
| ModelFactory | `src/ml_enhancements/spread_prediction/model_factory.py` | Creates prediction models | scikit-learn | Implemented |
| SpreadFeatureExtractor | `src/ml_enhancements/spread_prediction/feature_extractor.py` | Extracts features for spread prediction | numpy, pandas | Implemented |
| PredictionEvaluator | `src/ml_enhancements/spread_prediction/prediction_evaluator.py` | Evaluates prediction quality | scikit-learn, numpy | Implemented |
| EnsemblePredictor | `src/ml_enhancements/ensemble_models.py` | Ensemble prediction methods | scikit-learn | Implemented |

## Testing Infrastructure

### Test Fixtures
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| DataFixtures | `tests/fixtures/data_fixtures.py` | Provides test data | pandas, numpy | Implemented |
| PairFixtures | `tests/fixtures/pair_fixtures.py` | Provides test pairs | pandas | Implemented |
| ModelFixtures | `tests/fixtures/model_fixtures.py` | Provides test models | scikit-learn | Implemented |
| ConfigFixtures | `tests/fixtures/config_fixtures.py` | Provides test configurations | None | Implemented |
| MockDataGenerator | `tests/mocks/mock_data.py` | Generates synthetic test data | numpy, pandas | Implemented |

### Unit Tests
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| CointegrationTests | `tests/unit/cointegration/` | Tests cointegration functionality | pytest, numpy | Implemented |
| SignalGenerationTests | `tests/unit/signal_generation/` | Tests signal generation | pytest, pandas | Implemented |
| RiskManagementTests | `tests/unit/risk_management/` | Tests risk management | pytest, numpy | Implemented |
| DataProcessorTests | `tests/unit/data_processor/` | Tests data processing | pytest, pandas | Implemented |
| MLComponentTests | `tests/unit/ml_enhancements/` | Tests ML components | pytest, scikit-learn | Partially Implemented |

### Integration Tests
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| StrategyExecutionTests | `tests/integration/test_strategy_execution.py` | Tests end-to-end strategy execution | pytest, pandas | Implemented |
| MLIntegrationTests | `tests/integration/test_intraday_ml_integration.py` | Tests ML enhancement integration | pytest, pandas, scikit-learn | Implemented |
| DataFlowTests | `tests/integration/test_data_flow.py` | Tests data flow between components | pytest, pandas | Implemented |
| IBIntegrationTests | `tests/integration/test_ib_integration.py` | Tests Interactive Brokers integration | pytest, ib_insync | Implemented |
| MultiAssetTests | `tests/integration/test_multi_asset_integration.py` | Tests multi-asset strategies | pytest, pandas | Implemented |

## Execution

### Backtesting
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| BacktestEngine | `src/backtest/backtest_engine.py` | Core backtesting functionality | numpy, pandas | Implemented |
| IntradayBacktestEngine | `src/backtest/intraday_backtest_engine.py` | Specialized intraday backtesting | BacktestEngine, pandas | Implemented |
| PortfolioBacktest | `src/backtest/portfolio_backtest.py` | Multi-pair portfolio backtesting | BacktestEngine, pandas | Implemented |
| PerformanceAnalyzer | `src/backtest/performance_analyzer.py` | Analyzes backtest results | numpy, pandas | Implemented |
| TransactionCostModel | `src/backtest/transaction_cost_model.py` | Models trading costs | pandas | Implemented |

### Paper Trading
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| PaperTrader | `src/paper_trading/paper_trader.py` | Simulates live trading | pandas, numpy | Implemented |
| IntradayMLPaperTrader | `src/paper_trading/intraday_ml_paper_trader.py` | Intraday ML-enhanced paper trading | IntradayMLSystem, pandas | Implemented |
| PaperTradingMonitor | `src/paper_trading/paper_trading_monitor.py` | Monitors paper trading performance | pandas | Implemented |
| PaperTradingReporter | `src/paper_trading/paper_trading_reporter.py` | Generates paper trading reports | pandas, matplotlib | Implemented |
| VirtualOrderBook | `src/paper_trading/virtual_order_book.py` | Simulates order book | pandas | Implemented |

### Live Trading
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| IBConnector | `src/connectors/ib/ib_connector.py` | Connects to Interactive Brokers | ib_insync | Implemented |
| OrderManager | `src/execution/order_manager.py` | Manages order lifecycle | pandas | Implemented |
| ExecutionStrategy | `src/execution/execution_strategy.py` | Implements execution algorithms | numpy | Partially Implemented |
| OrderRouter | `src/execution/order_router.py` | Routes orders to appropriate venues | IBConnector | Partially Implemented |
| ExecutionAnalyzer | `src/execution/execution_analyzer.py` | Analyzes execution quality | pandas | Partially Implemented |

## Optimization

### Parameter Optimization
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| GridSearch | `src/optimization/grid_search.py` | Parameter grid search | numpy, pandas | Implemented |
| AdaptiveParameterManager | `src/optimization/adaptive_parameter_manager.py` | Adapts parameters by regime | pandas, MarketRegimeClassifier | Implemented |
| EvolutionaryOptimizer | `src/optimization/evolutionary_optimizer.py` | Genetic algorithm optimization | numpy, pandas | Implemented |
| ParameterSensitivityAnalyzer | `src/optimization/parameter_sensitivity.py` | Analyzes parameter sensitivity | numpy, pandas | Implemented |
| IntradayParameterOptimizer | `src/optimization/intraday_parameter_optimizer.py` | Intraday-specific parameter optimization | pandas, MarketRegimeClassifier | Implemented |

### Execution Optimization
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| SlippageOptimizer | `src/execution/slippage_optimizer.py` | Minimizes slippage | numpy, pandas | Partially Implemented |
| ExecutionScheduler | `src/execution/execution_scheduler.py` | Schedules order execution optimally | numpy, pandas | Partially Implemented |
| MarketImpactModel | `src/execution/market_impact_model.py` | Models market impact | numpy | Partially Implemented |
| LiquidityAnalyzer | `src/execution/liquidity_analyzer.py` | Analyzes market liquidity | pandas | Partially Implemented |
| OrderSplitter | `src/execution/order_splitter.py` | Splits large orders | numpy | Partially Implemented |

## Infrastructure

### Connectors
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| IBConnector | `src/connectors/ib/ib_connector.py` | Interactive Brokers connector | ib_insync | Implemented |
| DataProviderConnector | `src/connectors/data_provider_connector.py` | Connects to data providers | pandas, requests | Implemented |
| DatabaseConnector | `src/connectors/database_connector.py` | Database connection management | SQLAlchemy | Implemented |
| WebsocketConnector | `src/connectors/websocket_connector.py` | Websocket connections for real-time data | websockets | Partially Implemented |
| FIXConnector | `src/connectors/fix_connector.py` | FIX protocol connectivity | quickfix | Partially Implemented |

### Web Interface
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| APIServer | `api.py` | FastAPI server for remote control | FastAPI, celery | Implemented |
| DashboardApp | `src/web/dashboard_app.py` | Web dashboard for monitoring | Flask, Dash | Implemented |
| TaskManager | `tasks.py` | Background task management | Celery, Redis | Implemented |
| AuthenticationService | `src/web/auth_service.py` | User authentication | Flask-Login | Partially Implemented |
| StrategyController | `src/web/strategy_controller.py` | Web control of strategies | Flask | Implemented |

### Reporting
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| PerformanceDashboard | `src/visualization/dashboard.py` | Interactive performance dashboard | Dash, plotly | Implemented |
| TradeAnalyzer | `src/visualization/trade_analyzer.py` | Analyzes and reports on trades | pandas, matplotlib | Implemented |
| DrawdownAnalyzer | `src/visualization/drawdown_analyzer.py` | Analyzes drawdown periods | pandas, matplotlib | Implemented |
| ReportGenerator | `src/reporting/report_generator.py` | Generates performance reports | pandas, jinja2 | Implemented |
| RiskReporter | `src/reporting/risk_reporter.py` | Reports on risk exposure | pandas, matplotlib | Implemented |
| ReportingFramework | `src/reporting/reporting_framework.py` | Framework for generating all system reports | pandas, jinja2, matplotlib | Implemented |
| ScheduledReporter | `src/reporting/scheduled_reporter.py` | Handles scheduling and distribution of reports | pandas, schedule | Implemented |

## Utilities and Helpers

### Visualization
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| PerformanceVisualizer | `src/visualization/performance_visualizer.py` | Visualizes backtest performance | matplotlib, plotly | Implemented |
| SpreadVisualizer | `src/visualization/spread_visualizer.py` | Visualizes spread dynamics | matplotlib | Implemented |
| SignalVisualizer | `src/visualization/signal_visualizer.py` | Visualizes trading signals | matplotlib | Implemented |
| RegimeVisualizer | `src/visualization/regime_visualizer.py` | Visualizes market regimes | matplotlib, plotly | Partially Implemented |
| CorrelationVisualizer | `src/visualization/correlation_visualizer.py` | Visualizes asset correlations | seaborn | Implemented |

### Configuration
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| ConfigManager | `src/config/config_manager.py` | Manages system configuration | pyyaml | Implemented |
| StrategyConfig | `src/config/strategy_config.py` | Strategy-specific configuration | pyyaml | Implemented |
| ParameterConfig | `src/config/parameter_config.py` | Parameter configuration | pyyaml | Implemented |
| ValidationSchema | `src/config/validation_schema.py` | Configuration validation schemas | jsonschema | Implemented |
| EnvironmentConfig | `src/config/environment_config.py` | Environment-specific configuration | pyyaml, os | Implemented |

### Utils
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| TimeSeriesUtils | `src/utils/time_series_utils.py` | Time series manipulation utilities | pandas | Implemented |
| StatisticalUtils | `src/utils/statistical_utils.py` | Statistical calculation utilities | numpy, scipy | Implemented |
| DateTimeUtils | `src/utils/datetime_utils.py` | Date and time utilities | pandas, datetime | Implemented |
| LoggingUtils | `src/utils/logging_utils.py` | Logging utilities | logging | Implemented |
| ParallelProcessor | `src/infrastructure/parallel_processor.py` | Parallel processing utilities | joblib, multiprocessing | Implemented |

### Automation
| Component | Location | Purpose | Dependencies | Status |
|-----------|----------|---------|--------------|--------|
| TaskOrchestrator | `src/automation/task_orchestrator.py` | Coordinates task execution with retry capabilities | schedule, logging | Implemented |
| MasterOrchestration | `src/automation/master_orchestration.py` | Manages all automated processes | TaskOrchestrator, ConfigManager | Implemented |
| WindowsSchedulerSetup | `src/automation/windows_scheduler_setup.py` | Configures Windows Task Scheduler | subprocess | Implemented |
| AutomationConfig | `src/config/automation_config.yaml` | Configuration for automation system | None | Implemented |
| AutomationMonitor | `src/automation/automation_monitor.py` | Monitors automation tasks and system health | logging, smtplib | Implemented |
| SystemStartup | `src/automation/system_startup.py` | Handles system startup and recovery | subprocess | Implemented |
| SystemShutdown | `src/automation/system_shutdown.py` | Handles system shutdown and cleanup | psutil | Implemented |

## Notes:
- **Status**: Implemented, Partially Implemented, Needs Refactoring, Deprecated, etc.
- **Dependencies**: List of major dependencies, both internal and external 