import logging
import sys
import os

# Add Sample_Data to path (from Code/data_processing, up to root, then into Sample_Data)
SAMPLE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Sample_Data'))
sys.path.append(SAMPLE_DATA_PATH)

# Now import from vector_store
from vector_store.weaviate_client import get_weaviate_client, check_weaviate_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_market_data():
    """Verify the loaded market data in Weaviate"""
    client = get_weaviate_client()
    
    try:
        # Get the MarketMetrics collection
        collection = client.collections.get("MarketMetrics")
        
        # Query some sample data
        response = collection.query.fetch_objects(
            limit=5,  # Get first 5 records
            include_vector=False
        )
        
        # Print total count
        total = collection.aggregate.over_all(group_by="source")
        logger.info(f"\nTotal records by source:")
        for group in total.groups:
            logger.info(f"Source: {group.grouped_by.value}, Count: {group.total_count}")
            
        # Print sample records
        logger.info("\nSample records:")
        for obj in response.objects:
            logger.info(f"\nSymbol: {obj.properties['symbol']}")
            # Handle None values before formatting as floats
            price = obj.properties.get('price', 0)
            if price is None:
                price = 0  # Default to 0 if None
            logger.info(f"Price: ${price:,.2f}")
            
            market_cap = obj.properties.get('market_cap', 0)
            if market_cap is None:
                market_cap = 0
            logger.info(f"Market Cap: ${market_cap:,.2f}")
            
            volume_24h = obj.properties.get('volume_24h', 0)
            if volume_24h is None:
                volume_24h = 0
            logger.info(f"24h Volume: ${volume_24h:,.2f}")
            
            price_change_24h = obj.properties.get('price_change_24h', 0)
            if price_change_24h is None:
                price_change_24h = 0
            logger.info(f"24h Change: {price_change_24h:,.2f}%")
            
            logger.info(f"Source: {obj.properties['source']}")
            logger.info(f"Timestamp: {obj.properties['timestamp']}")
            
    except Exception as e:
        logger.error(f"Error verifying data: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    verify_market_data()