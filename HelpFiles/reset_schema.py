# reset_schema.py
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def reset_crypto_time_series():
    """Delete and recreate the CryptoTimeSeries collection with proper schema"""
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        from Sample_Data.vector_store.schema_manager import create_crypto_time_series_schema
        
        client = get_weaviate_client()
        
        try:
            # Delete the collection if it exists
            logger.info("Deleting existing CryptoTimeSeries collection...")
            client.collections.delete("CryptoTimeSeries")
            logger.info("Successfully deleted CryptoTimeSeries collection")
        except Exception as e:
            logger.info(f"Collection may not exist or could not be deleted: {e}")
        
        # Create the collection with the fixed schema
        logger.info("Creating CryptoTimeSeries collection with correct schema...")
        collection = create_crypto_time_series_schema(client)
        
        logger.info("Schema reset complete!")
        client.close()
        return True
    except Exception as e:
        logger.error(f"Error resetting schema: {e}")
        return False

if __name__ == "__main__":
    reset_crypto_time_series()