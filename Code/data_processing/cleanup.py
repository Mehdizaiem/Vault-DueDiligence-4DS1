import sys
import os
import logging

# Add Sample_Data to path
SAMPLE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Sample_Data'))
sys.path.append(SAMPLE_DATA_PATH)

from vector_store.weaviate_client import get_weaviate_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_market_data():
    """Delete the MarketMetrics collection and recreate it"""
    client = get_weaviate_client()
    try:
        # Delete the MarketMetrics collection
        try:
            client.collections.delete("MarketMetrics")
            logger.info("✅ Successfully deleted MarketMetrics collection")
        except Exception as e:
            logger.info(f"Collection doesn't exist or already deleted: {e}")

        # Also cleanup CryptoNewsSentiment
        try:
            client.collections.delete("CryptoNewsSentiment")
            logger.info("✅ Successfully deleted CryptoNewsSentiment collection")
        except Exception as e:
            logger.info(f"Collection doesn't exist or already deleted: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        if client:
            client.close()
            logger.info("Weaviate client closed")

if __name__ == "__main__":
    cleanup_market_data()