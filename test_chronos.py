# test_enhanced_chronos.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import the EnhancedChronos class
from Code.data_processing.enhanced import EnhancedChronos

def test_enhanced_chronos():
    """Test the enhanced chronos system"""
    logger.info("Creating EnhancedChronos instance...")
    chronos = EnhancedChronos()
    
    # Test symbol (can be any cryptocurrency symbol)
    symbol = "BTCUSDT"
    
    try:
        # Test 1: Generate enhanced data
        logger.info(f"Generating enhanced data for {symbol}...")
        df = chronos.get_enhanced_data(symbol)
        
        if df is not None:
            logger.info(f"Successfully generated enhanced data with {len(df)} points")
            logger.info(f"Data columns: {df.columns.tolist()}")
            
            # Print some technical indicators
            if 'rsi_14' in df.columns:
                logger.info(f"Current RSI: {df['rsi_14'].iloc[-1]:.2f}")
            if 'ma_50' in df.columns and 'ma_200' in df.columns:
                logger.info(f"50-day MA: {df['ma_50'].iloc[-1]:.2f}")
                logger.info(f"200-day MA: {df['ma_200'].iloc[-1]:.2f}")
        else:
            logger.warning("Failed to generate enhanced data")
            return
        
        # Test 2: Generate forecast
        logger.info(f"Generating forecast for {symbol}...")
        forecast = chronos.generate_enhanced_forecast(symbol, days_ahead=7)
        
        if "error" not in forecast:
            logger.info(f"Successfully generated forecast")
            logger.info(f"Current price: ${forecast.get('current_price', 0):.2f}")
            
            # Print predictions
            predictions = forecast.get("predictions", [])
            for i, pred in enumerate(predictions):
                logger.info(f"Day {i+1} ({pred.get('date')}): ${pred.get('forecast_price', 0):.2f} ({pred.get('change_pct', 0):+.2f}%)")
            
            # Print plot path
            if "plot_path" in forecast:
                logger.info(f"Forecast plot saved to: {forecast.get('plot_path')}")
        else:
            logger.warning(f"Forecast error: {forecast.get('error')}")
        
        # Test 3: Detect market anomalies
        logger.info(f"Detecting market anomalies for {symbol}...")
        anomalies = chronos.detect_market_anomalies(df)
        
        if anomalies and anomalies.get("count", 0) > 0:
            logger.info(f"Detected {anomalies.get('count')} anomalies")
            for anomaly in anomalies.get("anomalies", [])[:3]:  # Show first 3 anomalies
                logger.info(f"Anomaly on {anomaly.get('date')}: {anomaly.get('type')} ({anomaly.get('severity')})")
        else:
            logger.info("No significant anomalies detected")
        
        # Test 4: Generate trading signals
        logger.info(f"Generating trading signals for {symbol}...")
        signals = chronos._generate_trading_signals(df)
        
        if signals and "signals" in signals:
            logger.info(f"Current trend: {signals.get('trend', 'unknown')}")
            logger.info(f"Detected {len(signals.get('signals', []))} trading signals")
            
            for signal in signals.get("signals", []):
                logger.info(f"Signal: {signal.get('indicator')} - {signal.get('signal')} - Action: {signal.get('action')}")
        else:
            logger.info("No trading signals detected")
        
        # Test 5: Run backtest (optional - can be slow)
        run_backtest = False  # Set to True to run backtest
        if run_backtest:
            logger.info(f"Running backtest for {symbol}...")
            backtest = chronos.backtest_forecast_accuracy(symbol, days_to_forecast=7, test_periods=4)
            
            if "error" not in backtest:
                logger.info(f"Backtest completed successfully")
                logger.info(f"Average accuracy: {backtest.get('avg_price_accuracy', 0):.2f}%")
                logger.info(f"Average MAPE: {backtest.get('avg_mape', 0):.2f}%")
            else:
                logger.warning(f"Backtest error: {backtest.get('error')}")
        
        logger.info("EnhancedChronos test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting EnhancedChronos test")
    test_enhanced_chronos()