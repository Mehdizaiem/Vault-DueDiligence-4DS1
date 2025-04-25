#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Forecast Manager - Direct Weaviate connection
"""

import os
import json
import argparse
import logging
import weaviate
from weaviate.classes.query import Sort, Filter
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_weaviate_client():
    """Create and return a Weaviate client"""
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
    WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    try:
        client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                url=WEAVIATE_URL,
                grpc_port=WEAVIATE_GRPC_PORT
            )
        )
        
        # Explicitly connect
        client.connect()
        
        if not client.is_live():
            raise ConnectionError("Failed to connect to Weaviate")
            
        return client
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        raise

def normalize_symbol(symbol):
    """Normalize the symbol format to match the database"""
    # Remove underscore and USD/USDT suffix
    clean_symbol = symbol.upper().replace('_', '').replace('USDT', '').replace('USD', '')
    # Add USD suffix
    return f"{clean_symbol}USD"

def clean_forecast_data(forecast):
    """Clean and format forecast data for output"""
    clean_data = {}
    
    # Handle basic fields
    for key in ['symbol', 'model_name', 'model_type', 'trend', 'insight']:
        if key in forecast:
            clean_data[key] = str(forecast[key])
    
    # Handle numeric fields
    for key in ['current_price', 'final_forecast', 'change_pct', 'probability_increase', 'average_uncertainty', 'days_ahead']:
        if key in forecast:
            try:
                clean_data[key] = float(forecast[key])
            except (ValueError, TypeError):
                clean_data[key] = 0.0
    
    # Handle date fields
    if 'forecast_timestamp' in forecast:
        try:
            if isinstance(forecast['forecast_timestamp'], datetime):
                clean_data['forecast_timestamp'] = forecast['forecast_timestamp'].isoformat()
            else:
                clean_data['forecast_timestamp'] = str(forecast['forecast_timestamp'])
        except Exception:
            clean_data['forecast_timestamp'] = None
    
    # Handle arrays
    for key in ['forecast_dates', 'forecast_values', 'lower_bounds', 'upper_bounds']:
        if key in forecast:
            try:
                clean_data[key] = [float(x) if isinstance(x, (int, float)) else None for x in forecast[key]]
            except (ValueError, TypeError):
                clean_data[key] = []
    
    return clean_data

def get_latest_forecast(symbol='BTC_USD'):
    """Get the latest forecast from Weaviate"""
    client = None
    try:
        # Connect to Weaviate
        client = get_weaviate_client()
        logger.info("Connected to Weaviate")
        
        # Normalize symbol format to match database (BTCUSD)
        symbol = normalize_symbol(symbol)
        logger.info(f"Using normalized symbol: {symbol}")
        
        # Get the Forecast collection
        collection = client.collections.get("Forecast")
        
        # Query for specific symbol
        response = collection.query.fetch_objects(
            filters=Filter.by_property("symbol").equal(symbol),
            sort=Sort.by_property("forecast_timestamp", ascending=False),
            limit=1
        )
        
        if response.objects:
            raw_forecast = response.objects[0].properties
            forecast = clean_forecast_data(raw_forecast)
            forecast['id'] = str(response.objects[0].uuid)
            return forecast
        
        return {"error": f"No forecast found for {symbol}"}
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}
    finally:
        if client:
            client.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Simple Forecast Manager')
    parser.add_argument('--action', type=str, choices=['get_latest_forecast'], 
                        required=True, help='Action to perform')
    parser.add_argument('--symbol', type=str, default='BTC_USD', 
                        help='Cryptocurrency symbol')
    
    args = parser.parse_args()
    
    if args.action == 'get_latest_forecast':
        result = get_latest_forecast(args.symbol)
        # Format JSON output with indentation for readability
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()