# Codebase Audit Summary

## Large Files

| File | Lines | Excess |
|------|-------|--------|
| src\paper_trading\intraday_ml_paper_trader.py | 1332 | +832 |
| src\ml_enhancements\intraday_signals.py | 1242 | +742 |
| src\paper_trading\paper_trader.py | 1100 | +600 |
| src\live_trading\position_tracker.py | 989 | +489 |
| src\connectors\ib\ib_connector.py | 986 | +486 |
| src\live_trading\live_trader.py | 814 | +314 |
| src\live_trading\monitoring.py | 804 | +304 |
| src\ml_enhancements\model_retraining.py | 741 | +241 |
| src\strategy_variants\ml_signals\ml_signal_strategy.py | 670 | +170 |
| src\diagnostics\benchmark_real_time.py | 669 | +169 |

## Complex Functions

| Function | File | Lines | Complexity | Parameters |
|----------|------|-------|------------|------------|
| IntradaySignalEnhancer.enhance_signals | src\ml_enhancements\intraday_signals.py:948 | 231 | 26 | 5 |
| run_backtest | src\main.py:74 | 241 | 25 | 3 |
| IntradaySignalEnhancer.apply_intraday_adaptations | src\ml_enhancements\intraday_signals.py:1180 | 181 | 23 | 6 |
| PositionTracker.close_pair_position | src\live_trading\position_tracker.py:536 | 132 | 20 | 8 |
| FeatureGenerator._add_statistical_features | src\ml_enhancements\feature_engineering\feature_generator.py:211 | 102 | 20 | 2 |
| IntradaySignalEnhancer.load_models | src\ml_enhancements\intraday_signals.py:599 | 84 | 18 | 1 |
| MarketRegimeClassifier.calculate_features | src\ml_enhancements\regime_detection\market_regime_classifier.py:65 | 117 | 18 | 3 |
| CompressedOHLC.add_data | src\performance\data_structures\compressed_ohlc.py:71 | 89 | 18 | 2 |
| _create_summary_metrics | src\reporting\metrics.py:547 | 92 | 18 | 1 |
| process_signals | src\signal_generation\signal_processor.py:380 | 94 | 18 | 10 |

## Similar Function Pairs

| Function 1 | Function 2 | Similarity | Lines |
|------------|------------|------------|-------|
| FixedIBConnector.is_connected (src\connectors\ib\fixed_connector.py) | IBConnector.is_connected (src\connectors\ib\ib_connector.py) | 1.00 | 10/10 |
| LiveTrader._setup_auto_shutdown (src\live_trading\live_trader.py) | PaperTrader._setup_auto_shutdown (src\paper_trading\paper_trader.py) | 1.00 | 29/29 |
| IBConnector._notify_callbacks (src\connectors\ib\ib_connector.py) | PaperTrader._notify_callbacks (src\paper_trading\paper_trader.py) | 1.00 | 19/19 |
| GeneticOptimizer._evaluate_params (src\optimization\genetic_algorithm\genetic_algorithm.py) | GridSearchOptimizer._evaluate_params (src\optimization\grid_search\grid_search.py) | 0.96 | 63/63 |
| get_running_processes (src\web_interface\blueprints\api\routes.py) | get_running_processes (src\web_interface\blueprints\strategy\routes.py) | 0.96 | 30/28 |
| IBConnector.remove_callback (src\connectors\ib\ib_connector.py) | PaperTrader.remove_callback (src\paper_trading\paper_trader.py) | 0.90 | 25/25 |
| IBConnector.add_callback (src\connectors\ib\ib_connector.py) | PaperTrader.add_callback (src\paper_trading\paper_trader.py) | 0.89 | 22/22 |
| CryptoAsset.get_data (src\asset_classes\cryptocurrencies\crypto_asset.py) | EquityAsset.get_data (src\asset_classes\equities\equity_asset.py) | 0.86 | 29/29 |
| CryptoAsset.get_data (src\asset_classes\cryptocurrencies\crypto_asset.py) | FixedIncomeAsset.get_data (src\asset_classes\fixed_income\fixed_income_asset.py) | 0.86 | 29/29 |
| EquityAsset.get_data (src\asset_classes\equities\equity_asset.py) | FixedIncomeAsset.get_data (src\asset_classes\fixed_income\fixed_income_asset.py) | 0.86 | 29/29 |

## Import Graph Analysis

- Total modules: 149
- Connected components: 149
- Largest component size: 1
- Graph density: 0.0000
- Isolated modules: 149

### Isolated Modules

- src.main
- src.pairs_trading_strategy
- src.test_pair_selection
- src.__init__
- src.asset_classes.base
- src.asset_classes.factory
- src.asset_classes.__init__
- src.asset_classes.cryptocurrencies.crypto_asset
- src.asset_classes.cryptocurrencies.__init__
- src.asset_classes.equities.equity_asset
