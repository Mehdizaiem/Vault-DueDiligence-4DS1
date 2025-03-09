# crypto_data_diagnostic.py
import os
import sys
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diagnostic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def diagnose_time_series_data():
    """Check the time series data in Weaviate"""
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        client = get_weaviate_client()
        
        try:
            # Get CryptoTimeSeries collection
            collection = client.collections.get("CryptoTimeSeries")
            
            # Count all objects
            total_count = collection.query.aggregate.over_all().total_count
            logger.info(f"Total objects in CryptoTimeSeries: {total_count}")
            
            # Get all symbols with counts
            symbol_counts = collection.query.aggregate.over_all(
                group_by="symbol"
            )
            
            if hasattr(symbol_counts, 'groups'):
                print("\nSymbols in CryptoTimeSeries:")
                for group in symbol_counts.groups:
                    symbol = group.grouped_by.value
                    count = group.total_count
                    print(f"  {symbol}: {count} data points")
                
                # Get intervals
                interval_counts = collection.query.aggregate.over_all(
                    group_by="interval"
                )
                
                if hasattr(interval_counts, 'groups'):
                    print("\nIntervals in CryptoTimeSeries:")
                    for group in interval_counts.groups:
                        interval = group.grouped_by.value
                        count = group.total_count
                        print(f"  {interval}: {count} data points")
                
                # Get sample data for first symbol
                if symbol_counts.groups:
                    sample_symbol = symbol_counts.groups[0].grouped_by.value
                    
                    from weaviate.classes.query import Filter
                    sample_data = collection.query.fetch_objects(
                        filters=Filter.by_property("symbol").equal(sample_symbol),
                        limit=1
                    )
                    
                    if sample_data.objects:
                        print("\nSample data point:")
                        sample = sample_data.objects[0].properties
                        for key, value in sample.items():
                            print(f"  {key}: {value} ({type(value).__name__})")
            else:
                logger.warning("No symbols found in CryptoTimeSeries")
            
            # Check MarketMetrics collection too
            try:
                market_collection = client.collections.get("MarketMetrics")
                
                # Count all objects
                market_count = market_collection.query.aggregate.over_all().total_count
                logger.info(f"Total objects in MarketMetrics: {market_count}")
                
                # Get all symbols with counts
                market_symbols = market_collection.query.aggregate.over_all(
                    group_by="symbol"
                )
                
                if hasattr(market_symbols, 'groups'):
                    print("\nSymbols in MarketMetrics:")
                    for group in market_symbols.groups:
                        symbol = group.grouped_by.value
                        count = group.total_count
                        print(f"  {symbol}: {count} data points")
            except Exception as e:
                logger.error(f"Error checking MarketMetrics: {e}")
            
            return True
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error diagnosing time series data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def print_retrieval_test():
    """Test retrieving data for common symbols"""
    try:
        from Sample_Data.vector_store.storage_manager import StorageManager
        
        manager = StorageManager()
        
        try:
            symbols = ["BTCUSDT", "BTCUSD", "BTC", "ETHUSDT", "ETHBTC", "ETHUSD"]
            
            print("\nTesting data retrieval for common symbols:")
            for symbol in symbols:
                data = manager.retrieve_time_series(symbol)
                print(f"  {symbol}: {len(data)} data points")
                
                if data:
                    # Show first data point
                    first_point = data[0]
                    print(f"    First point: {first_point.get('timestamp')}, Close: {first_point.get('close')}")
            
            return True
        finally:
            manager.close()
    except Exception as e:
        logger.error(f"Error in retrieval test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("\n=== Crypto Data Diagnostic ===\n")
    diagnose_time_series_data()
    print("\n=== Retrieval Test ===\n")
    print_retrieval_test()