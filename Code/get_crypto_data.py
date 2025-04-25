#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path setup
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from Sample_Data.vector_store.storage_manager import StorageManager
    STORAGE_AVAILABLE = True
    logger.info("Successfully imported StorageManager")
except ImportError:
    logger.warning("StorageManager not available - cannot retrieve data")
    STORAGE_AVAILABLE = False

def format_symbol_for_storage(symbol):
    """Format symbol for storage (e.g., BTC -> BTCUSD)"""
    # Remove any USD/USDT suffix if present
    base_symbol = symbol.replace("USD", "").replace("USDT", "")
    
    # Add USD suffix for consistency
    return f"{base_symbol}USD"

def get_crypto_data(symbol="BTC"):
    """
    Retrieve crypto data including:
    1. Latest forecast for the specified symbol
    2. Latest prices for major cryptocurrencies
    3. Historical time series data for the specified symbol
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC')
        
    Returns:
        dict: Results containing forecast, prices, and historical data
    """
    if not STORAGE_AVAILABLE:
        return {
            "success": False,
            "error": "StorageManager not available"
        }
    
    try:
        # Initialize storage manager
        storage_manager = StorageManager()
        result = {
            "success": True,
            "prices": [],
            "forecast": None,
            "historicalData": []
        }
        
        try:
            # 1. Get prices for major cryptocurrencies
            symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'DOTUSD', 'XRPUSD']
            
            for crypto_symbol in symbols:
                logger.info(f"Fetching price data for {crypto_symbol}")
                
                # Try to get market data
                market_data = storage_manager.retrieve_market_data(crypto_symbol, limit=1)
                
                if market_data and len(market_data) > 0:
                    logger.info(f"Found market data for {crypto_symbol}")
                    # Extract price information
                    price_data = {
                        "symbol": crypto_symbol,
                        "price": market_data[0].get("price", 0),
                        "price_change_24h": market_data[0].get("price_change_24h", 0),
                        "price_change_percentage_24h": market_data[0].get("price_change_24h", 0)
                    }
                    result["prices"].append(price_data)
                else:
                    logger.info(f"No market data found for {crypto_symbol}, trying time series data")
                    # Try to get time series data as fallback
                    time_series = storage_manager.retrieve_time_series(crypto_symbol, interval="1d", limit=2)
                    
                    if time_series and len(time_series) >= 2:
                        logger.info(f"Found time series data for {crypto_symbol}")
                        current_price = time_series[-1].get("close", 0)
                        previous_price = time_series[-2].get("close", 0)
                        
                        # Calculate percentage change
                        if previous_price > 0:
                            price_change = ((current_price - previous_price) / previous_price) * 100
                        else:
                            price_change = 0
                            
                        price_data = {
                            "symbol": crypto_symbol,
                            "price": current_price,
                            "price_change_24h": current_price - previous_price,
                            "price_change_percentage_24h": price_change
                        }
                        result["prices"].append(price_data)
                    else:
                        logger.warning(f"No price data found for {crypto_symbol}")
            
            # 2. Get forecast for selected symbol
            formatted_symbol = format_symbol_for_storage(symbol)
            logger.info(f"Looking up forecast for formatted symbol: {formatted_symbol}")
            
            forecasts = storage_manager.retrieve_latest_forecast(formatted_symbol, limit=1)
            
            if forecasts and len(forecasts) > 0:
                logger.info(f"Found forecast for {formatted_symbol}")
                result["forecast"] = forecasts[0]
            else:
                logger.warning(f"No forecast found for {formatted_symbol}")
            
            # 3. Get historical data for the selected symbol
            try:
                # Try to get 90 days of time series data
                time_series = storage_manager.retrieve_time_series(formatted_symbol, interval="1d", limit=90)
                if time_series and len(time_series) > 0:
                    logger.info(f"Found {len(time_series)} historical data points")
                    result["historicalData"] = [
                        {
                            "timestamp": item.get("timestamp"),
                            "price": item.get("close", item.get("price", 0))
                        }
                        for item in time_series if "timestamp" in item and ("close" in item or "price" in item)
                    ]
            except Exception as e:
                logger.warning(f"Failed to retrieve historical data: {e}")
            
            return result
            
        finally:
            # Close the storage manager connection
            storage_manager.close()
            
    except Exception as e:
        logger.error(f"Error retrieving crypto data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve cryptocurrency data")
    parser.add_argument("--symbol", type=str, default="BTC", help="Cryptocurrency symbol")
    
    args = parser.parse_args()
    
    result = get_crypto_data(args.symbol)
    print(json.dumps(result))