import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import concurrent.futures
import json
import pickle
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Processes market data to generate trading signals using various strategies
    """
    def __init__(self, feature_dir: str = "data/features", signal_dir: str = "data/signals",
                 model_dir: str = "models"):
        """
        Initialize the SignalProcessor
        
        Parameters:
        -----------
        feature_dir : str
            Directory containing feature data
        signal_dir : str
            Directory to store generated signals
        model_dir : str
            Directory containing ML models for signal enhancement
        """
        self.feature_dir = Path(feature_dir)
        self.signal_dir = Path(signal_dir)
        self.model_dir = Path(model_dir)
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_signals(self, symbol: str, timeframe: str = "1day",
                         strategy: str = "combined",
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         use_ml: bool = False,
                         save_to_file: bool = True) -> pd.DataFrame:
        """
        Generate trading signals for a given symbol and timeframe
        
        Parameters:
        -----------
        symbol : str
            Symbol to generate signals for
        timeframe : str
            Timeframe of the data ('1min', '5min', '1hour', '1day')
        strategy : str
            Signal generation strategy ('trend', 'mean_reversion', 'combined')
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        use_ml : bool
            Whether to use ML models to enhance signals
        save_to_file : bool
            Whether to save the signals to a file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with generated signals
        """
        # Load feature data
        feature_file = self.feature_dir / symbol / timeframe / "features.csv"
        
        if not feature_file.exists():
            logger.error(f"Feature data file not found: {feature_file}")
            return pd.DataFrame()
        
        logger.info(f"Loading feature data for {symbol} {timeframe}")
        df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        
        # Filter by date if needed
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        if df.empty:
            logger.error(f"No feature data found for {symbol} {timeframe} in date range")
            return pd.DataFrame()
        
        # Generate signals based on strategy
        logger.info(f"Generating {strategy} signals for {symbol} {timeframe}")
        if strategy == "trend":
            signals_df = self._generate_trend_signals(df)
        elif strategy == "mean_reversion":
            signals_df = self._generate_mean_reversion_signals(df)
        elif strategy == "combined":
            signals_df = self._generate_combined_signals(df)
        else:
            logger.error(f"Unsupported strategy: {strategy}")
            return pd.DataFrame()
        
        # Enhance signals with ML if requested
        if use_ml:
            signals_df = self._enhance_signals_with_ml(signals_df, symbol, timeframe)
        
        # Save to file if requested
        if save_to_file:
            signal_dir = self.signal_dir / symbol / timeframe
            signal_dir.mkdir(parents=True, exist_ok=True)
            signal_file = signal_dir / f"{strategy}_signals.csv"
            signals_df.to_csv(signal_file)
            logger.info(f"Saved signals to {signal_file}")
            
        return signals_df
    
    def _generate_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trend-following signals
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with feature data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trend signals
        """
        result = df.copy()
        
        # Initialize signal columns
        result['trend_signal'] = 0  # -1 for sell, 0 for neutral, 1 for buy
        
        # Generate signals based on moving average crossovers
        result['ma_crossover_signal'] = 0
        result.loc[result['sma_20_50_cross'] == 1, 'ma_crossover_signal'] = 1
        result.loc[result['sma_20_50_cross'] == 0, 'ma_crossover_signal'] = -1
        
        # Generate signals based on MACD
        result['macd_signal'] = 0
        result.loc[result['macd_cross'] == 1, 'macd_signal'] = 1
        result.loc[result['macd_cross_below'] == 1, 'macd_signal'] = -1
        
        # Generate signals based on ADX and trend direction
        result['adx_signal'] = 0
        result.loc[(result['adx'] > 25) & (result['close_vs_sma_50'] > 0), 'adx_signal'] = 1
        result.loc[(result['adx'] > 25) & (result['close_vs_sma_50'] < 0), 'adx_signal'] = -1
        
        # Combine signals with weights
        result['trend_signal'] = (
            result['ma_crossover_signal'] * 0.4 +
            result['macd_signal'] * 0.3 +
            result['adx_signal'] * 0.3
        )
        
        # Threshold for final signal
        result['signal'] = 0
        result.loc[result['trend_signal'] > 0.3, 'signal'] = 1
        result.loc[result['trend_signal'] < -0.3, 'signal'] = -1
        
        # Calculate entry/exit points
        result['entry_price'] = np.nan
        result['exit_price'] = np.nan
        
        # Entry prices
        entry_mask = (result['signal'] != 0) & (result['signal'].shift(1) == 0)
        result.loc[entry_mask, 'entry_price'] = result.loc[entry_mask, 'close']
        
        # Exit prices
        exit_mask = (result['signal'] == 0) & (result['signal'].shift(1) != 0)
        result.loc[exit_mask, 'exit_price'] = result.loc[exit_mask, 'close']
        
        return result
    
    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean-reversion signals
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with feature data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with mean-reversion signals
        """
        result = df.copy()
        
        # Initialize signal columns
        result['mr_signal'] = 0  # -1 for sell, 0 for neutral, 1 for buy
        
        # Generate signals based on Bollinger Bands
        result['bb_signal'] = 0
        result.loc[result['bb_percent_b'] < 0.05, 'bb_signal'] = 1  # Oversold
        result.loc[result['bb_percent_b'] > 0.95, 'bb_signal'] = -1  # Overbought
        
        # Generate signals based on RSI
        result['rsi_signal'] = 0
        result.loc[result['rsi'] < 30, 'rsi_signal'] = 1  # Oversold
        result.loc[result['rsi'] > 70, 'rsi_signal'] = -1  # Overbought
        
        # Generate signals based on price deviation from SMA
        result['sma_dev_signal'] = 0
        result.loc[result['close_vs_sma_20'] < -0.05, 'sma_dev_signal'] = 1  # Price significantly below SMA
        result.loc[result['close_vs_sma_20'] > 0.05, 'sma_dev_signal'] = -1  # Price significantly above SMA
        
        # Combine signals with weights
        result['mr_signal'] = (
            result['bb_signal'] * 0.4 +
            result['rsi_signal'] * 0.4 +
            result['sma_dev_signal'] * 0.2
        )
        
        # Threshold for final signal
        result['signal'] = 0
        result.loc[result['mr_signal'] > 0.3, 'signal'] = 1
        result.loc[result['mr_signal'] < -0.3, 'signal'] = -1
        
        # Calculate entry/exit points
        result['entry_price'] = np.nan
        result['exit_price'] = np.nan
        
        # Entry prices
        entry_mask = (result['signal'] != 0) & (result['signal'].shift(1) == 0)
        result.loc[entry_mask, 'entry_price'] = result.loc[entry_mask, 'close']
        
        # Exit prices (mean reversion typically exits when price returns to mean)
        exit_mask_long = (result['signal'] == 1) & (result['close_vs_sma_20'] >= -0.01)
        exit_mask_short = (result['signal'] == -1) & (result['close_vs_sma_20'] <= 0.01)
        result.loc[exit_mask_long | exit_mask_short, 'exit_price'] = result.loc[exit_mask_long | exit_mask_short, 'close']
        result.loc[exit_mask_long | exit_mask_short, 'signal'] = 0
        
        return result
    
    def _generate_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined signals using both trend and mean-reversion
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with feature data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with combined signals
        """
        # Generate trend and mean-reversion signals separately
        trend_signals = self._generate_trend_signals(df)
        mr_signals = self._generate_mean_reversion_signals(df)
        
        # Combine results
        result = df.copy()
        result['trend_signal'] = trend_signals['trend_signal']
        result['mr_signal'] = mr_signals['mr_signal']
        
        # Determine market regime based on volatility and trend strength
        result['high_volatility'] = result['volatility_20'] > result['volatility_20'].rolling(20).mean()
        result['strong_trend'] = result['adx'] > 25
        
        # In high volatility + strong trend: use trend signals
        # In low volatility + weak trend: use mean-reversion signals
        # Otherwise: use a weighted combination
        
        result['signal'] = 0
        
        # High volatility + strong trend: trend signals
        mask = result['high_volatility'] & result['strong_trend']
        result.loc[mask & (trend_signals['signal'] == 1), 'signal'] = 1
        result.loc[mask & (trend_signals['signal'] == -1), 'signal'] = -1
        
        # Low volatility + weak trend: mean-reversion signals
        mask = (~result['high_volatility']) & (~result['strong_trend'])
        result.loc[mask & (mr_signals['signal'] == 1), 'signal'] = 1
        result.loc[mask & (mr_signals['signal'] == -1), 'signal'] = -1
        
        # Mixed conditions: weighted combination
        mask = (~result['high_volatility'] & result['strong_trend']) | (result['high_volatility'] & ~result['strong_trend'])
        result.loc[mask, 'signal'] = 0
        result.loc[mask & ((trend_signals['signal'] + mr_signals['signal']) >= 1.5), 'signal'] = 1
        result.loc[mask & ((trend_signals['signal'] + mr_signals['signal']) <= -1.5), 'signal'] = -1
        
        # Calculate entry/exit points
        result['entry_price'] = np.nan
        result['exit_price'] = np.nan
        
        # Entry prices
        entry_mask = (result['signal'] != 0) & (result['signal'].shift(1) == 0)
        result.loc[entry_mask, 'entry_price'] = result.loc[entry_mask, 'close']
        
        # Exit prices
        exit_mask = (result['signal'] == 0) & (result['signal'].shift(1) != 0)
        exit_mask_regime = (result['high_volatility'] != result['high_volatility'].shift(1)) | \
                           (result['strong_trend'] != result['strong_trend'].shift(1))
        result.loc[exit_mask | exit_mask_regime, 'exit_price'] = result.loc[exit_mask | exit_mask_regime, 'close']
        result.loc[exit_mask | exit_mask_regime, 'signal'] = 0
        
        return result
    
    def _enhance_signals_with_ml(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Enhance signals using ML models
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with signal data
        symbol : str
            Symbol for the signals
        timeframe : str
            Timeframe of the data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with ML-enhanced signals
        """
        # Check if ML model exists
        model_file = self.model_dir / symbol / timeframe / "signal_classifier.pkl"
        
        if not model_file.exists():
            logger.warning(f"ML model not found for {symbol} {timeframe}, using original signals")
            return df
        
        try:
            # Load model
            logger.info(f"Loading ML model for {symbol} {timeframe}")
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Prepare features for prediction
            X = df.drop(['open', 'high', 'low', 'close', 'volume', 'signal', 'entry_price', 'exit_price'], axis=1, errors='ignore')
            
            # Remove any remaining non-feature columns
            cols_to_drop = [col for col in X.columns if col.startswith('signal') or col.startswith('trend') or col.startswith('mr_')]
            X = X.drop(cols_to_drop, axis=1, errors='ignore')
            
            # Make predictions
            logger.info(f"Making ML predictions for {symbol} {timeframe}")
            result = df.copy()
            
            # Predict signal
            result['ml_signal'] = model.predict(X)
            
            # Combine original signal with ML signal
            # If they agree, keep the signal
            # If they disagree, reduce or eliminate the signal
            result['signal_original'] = result['signal'].copy()
            
            # Keep original signal only if ML agrees
            result.loc[result['signal'] != result['ml_signal'], 'signal'] = 0
            
            # Re-calculate entry/exit points
            result['entry_price'] = np.nan
            result['exit_price'] = np.nan
            
            # Entry prices
            entry_mask = (result['signal'] != 0) & (result['signal'].shift(1) == 0)
            result.loc[entry_mask, 'entry_price'] = result.loc[entry_mask, 'close']
            
            # Exit prices
            exit_mask = (result['signal'] == 0) & (result['signal'].shift(1) != 0)
            result.loc[exit_mask, 'exit_price'] = result.loc[exit_mask, 'close']
            
            logger.info(f"Successfully enhanced signals with ML for {symbol} {timeframe}")
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing signals with ML for {symbol} {timeframe}: {str(e)}")
            return df


def process_signals(symbols: Union[str, List[str]], 
                    timeframes: Union[str, List[str]] = "1day",
                    strategies: Union[str, List[str]] = "combined",
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    feature_dir: str = "data/features",
                    signal_dir: str = "data/signals",
                    model_dir: str = "models",
                    use_ml: bool = False,
                    parallel: bool = True) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Process signals for multiple symbols, timeframes, and strategies
    
    Parameters:
    -----------
    symbols : str or List[str]
        Symbol or list of symbols to process signals for
    timeframes : str or List[str]
        Timeframe or list of timeframes ('1min', '5min', '1hour', '1day')
    strategies : str or List[str]
        Strategy or list of strategies ('trend', 'mean_reversion', 'combined')
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    feature_dir : str
        Directory containing feature data
    signal_dir : str
        Directory to store generated signals
    model_dir : str
        Directory containing ML models
    use_ml : bool
        Whether to use ML models to enhance signals
    parallel : bool
        Whether to process data in parallel
        
    Returns:
    --------
    Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        Nested dictionary with symbol, timeframe, and strategy as keys, and signal DataFrames as values
    """
    # Convert single inputs to lists
    if isinstance(symbols, str):
        symbols = [symbols]
    if isinstance(timeframes, str):
        timeframes = [timeframes]
    if isinstance(strategies, str):
        strategies = [strategies]
        
    processor = SignalProcessor(feature_dir=feature_dir, signal_dir=signal_dir, model_dir=model_dir)
    results = {}
    
    total_tasks = len(symbols) * len(timeframes) * len(strategies)
    
    if parallel and total_tasks > 1:
        logger.info(f"Processing signals for {len(symbols)} symbols, {len(timeframes)} timeframes, {len(strategies)} strategies in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, total_tasks)) as executor:
            # Create processing tasks
            future_to_params = {}
            for symbol in symbols:
                results[symbol] = {}
                for tf in timeframes:
                    results[symbol][tf] = {}
                    for strategy in strategies:
                        future = executor.submit(
                            processor.generate_signals, symbol, tf, strategy, start_date, end_date, use_ml, True
                        )
                        future_to_params[future] = (symbol, tf, strategy)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_params):
                symbol, tf, strategy = future_to_params[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol][tf][strategy] = data
                except Exception as e:
                    logger.error(f"Error processing signals for {symbol} {tf} {strategy}: {str(e)}")
    else:
        logger.info(f"Processing signals for {len(symbols)} symbols, {len(timeframes)} timeframes, {len(strategies)} strategies sequentially")
        for symbol in symbols:
            results[symbol] = {}
            for tf in timeframes:
                results[symbol][tf] = {}
                for strategy in strategies:
                    try:
                        data = processor.generate_signals(symbol, tf, strategy, start_date, end_date, use_ml, True)
                        if data is not None and not data.empty:
                            results[symbol][tf][strategy] = data
                    except Exception as e:
                        logger.error(f"Error processing signals for {symbol} {tf} {strategy}: {str(e)}")
    
    logger.info(f"Completed signal processing for {len(symbols)} symbols")
    return results


def combine_pair_signals(symbol1: str, symbol2: str, timeframe: str,
                        strategy: str = "combined",
                        signal_dir: str = "data/signals") -> pd.DataFrame:
    """
    Combine signals for a pair of symbols
    
    Parameters:
    -----------
    symbol1 : str
        First symbol in the pair
    symbol2 : str
        Second symbol in the pair
    timeframe : str
        Timeframe of the data
    strategy : str
        Strategy used for signal generation
    signal_dir : str
        Directory containing signal data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with combined pair signals
    """
    # Load signal data for both symbols
    signal_file1 = Path(signal_dir) / symbol1 / timeframe / f"{strategy}_signals.csv"
    signal_file2 = Path(signal_dir) / symbol2 / timeframe / f"{strategy}_signals.csv"
    
    if not signal_file1.exists() or not signal_file2.exists():
        logger.error(f"Signal files not found for pair {symbol1}-{symbol2} {timeframe}")
        return pd.DataFrame()
    
    logger.info(f"Loading signal data for pair {symbol1}-{symbol2} {timeframe}")
    df1 = pd.read_csv(signal_file1, index_col=0, parse_dates=True)
    df2 = pd.read_csv(signal_file2, index_col=0, parse_dates=True)
    
    # Align the indices
    common_idx = df1.index.intersection(df2.index)
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]
    
    if df1.empty or df2.empty:
        logger.error(f"No common dates found for pair {symbol1}-{symbol2} {timeframe}")
        return pd.DataFrame()
    
    # Create pair signals DataFrame
    pair_df = pd.DataFrame(index=common_idx)
    pair_df[f'{symbol1}_signal'] = df1['signal']
    pair_df[f'{symbol2}_signal'] = df2['signal']
    pair_df[f'{symbol1}_close'] = df1['close']
    pair_df[f'{symbol2}_close'] = df2['close']
    
    # Generate pair signal (long symbol1, short symbol2) when they diverge
    pair_df['pair_signal'] = 0
    
    # Long symbol1, short symbol2
    divergence_long_short = (pair_df[f'{symbol1}_signal'] == 1) & (pair_df[f'{symbol2}_signal'] == -1)
    pair_df.loc[divergence_long_short, 'pair_signal'] = 1
    
    # Short symbol1, long symbol2
    divergence_short_long = (pair_df[f'{symbol1}_signal'] == -1) & (pair_df[f'{symbol2}_signal'] == 1)
    pair_df.loc[divergence_short_long, 'pair_signal'] = -1
    
    # Calculate entry/exit points
    pair_df['entry_price_ratio'] = np.nan
    pair_df['exit_price_ratio'] = np.nan
    
    # Entry prices (ratio of symbol1/symbol2)
    entry_mask = (pair_df['pair_signal'] != 0) & (pair_df['pair_signal'].shift(1) == 0)
    pair_df.loc[entry_mask, 'entry_price_ratio'] = pair_df.loc[entry_mask, f'{symbol1}_close'] / pair_df.loc[entry_mask, f'{symbol2}_close']
    
    # Exit prices (on signal reversal or when signals align)
    exit_mask = ((pair_df['pair_signal'] == 0) & (pair_df['pair_signal'].shift(1) != 0)) | \
                ((pair_df['pair_signal'] == 1) & (pair_df['pair_signal'].shift(1) == -1)) | \
                ((pair_df['pair_signal'] == -1) & (pair_df['pair_signal'].shift(1) == 1))
    pair_df.loc[exit_mask, 'exit_price_ratio'] = pair_df.loc[exit_mask, f'{symbol1}_close'] / pair_df.loc[exit_mask, f'{symbol2}_close']
    
    # Save pair signals
    pair_dir = Path(signal_dir) / f"{symbol1}_{symbol2}" / timeframe
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair_file = pair_dir / f"{strategy}_pair_signals.csv"
    pair_df.to_csv(pair_file)
    logger.info(f"Saved pair signals to {pair_file}")
    
    return pair_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process market signals")
    parser.add_argument("--symbols", required=True, help="Comma-separated list of symbols")
    parser.add_argument("--timeframes", default="1day", help="Comma-separated list of timeframes (1min,5min,1hour,1day)")
    parser.add_argument("--strategies", default="combined", help="Comma-separated list of strategies (trend,mean_reversion,combined)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--feature-dir", default="data/features", help="Feature directory")
    parser.add_argument("--signal-dir", default="data/signals", help="Signal directory")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--use-ml", action="store_true", help="Use ML models to enhance signals")
    parser.add_argument("--sequential", action="store_true", help="Process sequentially instead of in parallel")
    parser.add_argument("--pair-mode", action="store_true", help="Process signals for pairs of symbols")
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")
    strategies = args.strategies.split(",")
    
    # Process individual signals
    process_signals(
        symbols=symbols,
        timeframes=timeframes,
        strategies=strategies,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_dir=args.feature_dir,
        signal_dir=args.signal_dir,
        model_dir=args.model_dir,
        use_ml=args.use_ml,
        parallel=not args.sequential
    )
    
    # Process pair signals if requested
    if args.pair_mode and len(symbols) >= 2:
        logger.info(f"Processing pair signals for {len(symbols)} symbols")
        
        # Process all possible pairs
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                for tf in timeframes:
                    for strategy in strategies:
                        try:
                            combine_pair_signals(
                                symbol1=symbols[i],
                                symbol2=symbols[j],
                                timeframe=tf,
                                strategy=strategy,
                                signal_dir=args.signal_dir
                            )
                        except Exception as e:
                            logger.error(f"Error processing pair signals for {symbols[i]}-{symbols[j]} {tf} {strategy}: {str(e)}")
                            
        logger.info(f"Completed pair signal processing") 