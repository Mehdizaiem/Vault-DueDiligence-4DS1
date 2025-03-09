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

def list_collections(client):
    """List all collections in Weaviate"""
    collections = client.collections.list_all()
    print("Available collections:")
    if collections:
        for collection_name in collections.keys():
            print(f"  - {collection_name}")
    else:
        print("  (None)")
    return list(collections.keys())

def diagnose_time_series_data():
    """Check the time series data in Weaviate"""
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        client = get_weaviate_client()
        logger.info("Weaviate client initialized successfully")
        
        try:
            # List collections first to confirm what's available
            available_collections = list_collections(client)
            
            # Use CryptoTimeSeries if available, else fallback to CryptoNewsSentiment
            collection_name = "CryptoTimeSeries" if "CryptoTimeSeries" in available_collections else "CryptoNewsSentiment"
            try:
                collection = client.collections.get(collection_name)
                logger.info(f"Retrieved {collection_name} collection")
            except Exception as e:
                logger.error(f"Failed to retrieve {collection_name}: {e}")
                print(f"Collection {collection_name} not found.")
                return False
            
            # Count all objects
            total_count_result = collection.aggregate.over_all(total_count=True)
            total_count = total_count_result.total_count
            logger.info(f"Total objects in {collection_name}: {total_count}")
            
            # Get all symbols with counts using group_by parameter
            from weaviate.classes.aggregate import GroupByAggregate
            symbol_counts = collection.aggregate.over_all(
                group_by=GroupByAggregate(prop="symbol"),
                total_count=True
            )
            
            if symbol_counts.groups:
                print(f"\nSymbols in {collection_name}:")
                for group in symbol_counts.groups:
                    symbol = group.grouped_by.value
                    count = group.total_count
                    print(f"  {symbol}: {count} data points")
                
                # Get intervals (if applicable)
                interval_counts = collection.aggregate.over_all(
                    group_by=GroupByAggregate(prop="interval"),
                    total_count=True
                )
                
                if interval_counts.groups:
                    print(f"\nIntervals in {collection_name}:")
                    for group in interval_counts.groups:
                        interval = group.grouped_by.value
                        count = group.total_count
                        print(f"  {interval}: {count} data points")
                
                # Get sample data for first symbol
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
                logger.warning(f"No symbols found in {collection_name} or empty result")
                print(f"No symbols found in {collection_name}")
            
            # Check MarketMetrics collection
            try:
                market_collection = client.collections.get("MarketMetrics")
                logger.info("Retrieved MarketMetrics collection")
                
                market_count_result = market_collection.aggregate.over_all(total_count=True)
                market_count = market_count_result.total_count
                logger.info(f"Total objects in MarketMetrics: {market_count}")
                
                market_symbols = market_collection.aggregate.over_all(
                    group_by=GroupByAggregate(prop="symbol"),
                    total_count=True
                )
                
                if market_symbols.groups:
                    print("\nSymbols in MarketMetrics:")
                    for group in market_symbols.groups:
                        symbol = group.grouped_by.value
                        count = group.total_count
                        print(f"  {symbol}: {count} data points")
                else:
                    logger.warning("No symbols found in MarketMetrics or empty result")
                    print("No symbols found in MarketMetrics")
            except Exception as e:
                logger.error(f"Error checking MarketMetrics: {e}")
            
            return True
        finally:
            client.close()
            logger.info("Weaviate client closed")
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
        logger.info("StorageManager initialized successfully")
        
        try:
            symbols = ["BTCUSDT", "BTCUSD", "BTC", "ETHUSDT", "ETHBTC", "ETHUSD"]
            
            print("\nTesting data retrieval for common symbols:")
            for symbol in symbols:
                data = manager.retrieve_time_series(symbol)
                print(f"  {symbol}: {len(data)} data points")
                
                if data:
                    first_point = data[0]
                    print(f"    First point: {first_point.get('timestamp')}, Close: {first_point.get('close')}")
            
            return True
        finally:
            manager.close()
            logger.info("StorageManager closed")
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