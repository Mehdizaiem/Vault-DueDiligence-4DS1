import json
import argparse
import os
import sys
import logging
from typing import Dict, Any, List
from weaviate_client import get_weaviate_client
from weaviate.classes.query import Sort, Filter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format to avoid interfering with JSON output
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class ForecastManager:
    """
    Forecast manager for retrieving cryptocurrency price predictions from Weaviate
    """
    
    def __init__(self):
        """Initialize the forecast manager"""
        try:
            self.client = get_weaviate_client()
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise
    
    def list_all_forecasts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List all available forecasts.
        
        Args:
            limit: Maximum number of forecasts to retrieve
            
        Returns:
            List of forecast objects
        """
        try:
            # Get the Forecast collection
            collection = self.client.collections.get("Forecast")
            
            # Query for all forecasts, ordered by timestamp
            response = collection.query.fetch_objects(
                sort=Sort.by_property("forecast_timestamp", ascending=False),
                limit=limit
            )
            
            # Format results
            results = []
            for obj in response.objects:
                result = {
                    "id": str(obj.uuid),
                    **obj.properties
                }
                results.append(result)
            
            if not results:
                return []
                
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving forecasts: {e}")
            return []
    
    def get_latest_forecast(self, symbol: str = "BTC_USD", limit: int = 1) -> List[Dict[str, Any]]:
        """
        Get the latest forecast(s) for a symbol.
        
        Args:
            symbol: Cryptocurrency symbol
            limit: Maximum number of forecasts to retrieve
            
        Returns:
            List of forecast objects
        """
        try:
            # Get the Forecast collection
            collection = self.client.collections.get("Forecast")
            
            # Try different symbol formats
            symbol_formats = [
                symbol,  # Original format
                symbol.replace("_", ""),  # Remove underscore
                symbol.upper(),  # Uppercase
                symbol.lower(),  # Lowercase
                symbol.replace("_", "-"),  # Replace underscore with dash
            ]
            
            for sym in symbol_formats:
                # Query for forecasts for this symbol, ordered by timestamp
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal(sym),
                    sort=Sort.by_property("forecast_timestamp", ascending=False),
                    limit=limit
                )
                
                if response.objects:
                    # Format results
                    results = []
                    for obj in response.objects:
                        result = {
                            "timestamp": obj.properties.get("forecast_timestamp"),
                            "predicted_price": obj.properties.get("predicted_price"),
                            "confidence_interval_lower": obj.properties.get("confidence_interval_lower"),
                            "confidence_interval_upper": obj.properties.get("confidence_interval_upper"),
                            "model_name": obj.properties.get("model_name"),
                            "model_type": obj.properties.get("model_type"),
                            "days_ahead": obj.properties.get("days_ahead"),
                            "average_uncertainty": obj.properties.get("average_uncertainty")
                        }
                        results.append(result)
                    return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving forecasts: {e}")
            return []

def main():
    """CLI interface for the forecast manager"""
    parser = argparse.ArgumentParser(description='Forecast Manager CLI')
    parser.add_argument('--action', type=str, required=True, help='Action to perform')
    parser.add_argument('--symbol', type=str, default='BTC_USD', help='Cryptocurrency symbol')
    
    args = parser.parse_args()
    
    try:
        manager = ForecastManager()
        
        if args.action == 'get_latest_forecast':
            results = manager.get_latest_forecast(args.symbol)
            if results:
                # Print only the JSON data, no logging messages
                sys.stderr = open(os.devnull, 'w')  # Redirect stderr to prevent logging
                print(json.dumps(results[0]))  # Print first result for single forecast
                sys.stderr = sys.__stderr__  # Restore stderr
            else:
                print(json.dumps({"error": f"No forecast found for {args.symbol}"}))
        elif args.action == 'list_forecasts':
            results = manager.list_all_forecasts()
            if results:
                print(json.dumps(results))
            else:
                print(json.dumps({"error": "No forecasts found in the database"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()