import logging
from vector_store.weaviate_client import get_weaviate_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_collections():
    """Delete the CryptoDueDiligenceDocuments collection and prepare for fresh start"""
    client = get_weaviate_client()
    try:
        # Delete the CryptoDueDiligenceDocuments collection
        try:
            client.collections.delete("CryptoDueDiligenceDocuments")
            logger.info("✅ Successfully deleted CryptoDueDiligenceDocuments collection")
        except Exception as e:
            logger.info(f"Collection doesn't exist or already deleted: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        if client:
            client.close()
            logger.info("Weaviate client closed")

if __name__ == "__main__":
    cleanup_collections()