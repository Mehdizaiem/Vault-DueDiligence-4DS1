#!/usr/bin/env python
"""
Script to create all required Weaviate schemas including UserDocuments
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.storage_manager import StorageManager
from Sample_Data.vector_store.schema_manager import (
    create_crypto_due_diligence_schema,
    create_crypto_news_sentiment_schema,
    create_market_metrics_schema,
    create_crypto_time_series_schema,
    create_onchain_analytics_schema,
    create_user_documents_schema
)

def main():
    """Create all required schemas in Weaviate"""
    try:
        # Initialize storage manager
        storage_manager = StorageManager()
        
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            return 1
        
        # Create all schemas
        create_crypto_due_diligence_schema(storage_manager.client)
        create_crypto_news_sentiment_schema(storage_manager.client)
        create_market_metrics_schema(storage_manager.client)
        create_crypto_time_series_schema(storage_manager.client)
        create_onchain_analytics_schema(storage_manager.client)
        
        # Create UserDocuments schema
        create_user_documents_schema(storage_manager.client)
        
        logger.info("All schemas created successfully")
        return 0
    except Exception as e:
        logger.error(f"Error creating schemas: {e}")
        return 1
    finally:
        if 'storage_manager' in locals() and storage_manager:
            storage_manager.close()

if __name__ == "__main__":
    sys.exit(main())