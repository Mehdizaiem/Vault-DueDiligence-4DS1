#!/usr/bin/env python
import os
import sys
import logging
from datetime import datetime, timedelta
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def check_weaviate():
    """Check Weaviate connection and version"""
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        logger.info("Connecting to Weaviate...")
        client = get_weaviate_client()
        
        if client.is_live():
            # Instead of accessing meta.version directly, get the meta and check its structure first
            meta = client.get_meta()
            logger.info(f"✅ Connected to Weaviate successfully")
            logger.info(f"Meta information: {meta}")
            client.close()
            return True
        else:
            logger.error("❌ Weaviate is not responding")
            return False
    except Exception as e:
        logger.error(f"❌ Weaviate connection error: {e}")
        return False

def setup_schemas():
    """Set up required Weaviate schemas"""
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        from Sample_Data.vector_store.market_sentiment_schema import (
            create_sentiment_schema,
            create_forecast_schema,
            create_market_metrics_schema
        )
        
        logger.info("Setting up Weaviate schemas...")
        client = get_weaviate_client()
        
        try:
            create_sentiment_schema(client)
            logger.info("✅ CryptoNewsSentiment schema created")
            
            create_forecast_schema(client)
            logger.info("✅ CryptoForecasts schema created")
            
            create_market_metrics_schema(client)
            logger.info("✅ MarketMetrics schema created with sample data")
            
            return True
        finally:
            client.close()
    except Exception as e:
        logger.error(f"❌ Schema setup error: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_setup():
    """Validate the setup by querying data"""
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        logger.info("Validating setup...")
        client = get_weaviate_client()
        
        try:
            # Check collections individually instead of using list()
            required_collections = ["CryptoNewsSentiment", "CryptoForecasts", "MarketMetrics"]
            collection_exists = {}
            
            for collection_name in required_collections:
                try:
                    collection = client.collections.get(collection_name)
                    collection_exists[collection_name] = True
                    logger.info(f"Found collection: {collection_name}")
                except Exception as e:
                    collection_exists[collection_name] = False
                    logger.warning(f"Collection not found: {collection_name} - {str(e)}")
            
            logger.info(f"Collection status: {collection_exists}")
            
            missing_collections = [name for name, exists in collection_exists.items() if not exists]
            if missing_collections:
                logger.warning(f"❌ Missing collections: {missing_collections}")
                return False
                
            # Check for data in MarketMetrics
            try:
                market_collection = client.collections.get("MarketMetrics")
                market_count = market_collection.aggregate.over_all().total_count
                logger.info(f"MarketMetrics has {market_count} records")
                
                if market_count == 0:
                    logger.warning("❌ MarketMetrics has no data")
                    return False
            except Exception as e:
                logger.warning(f"Error checking MarketMetrics: {e}")
                
            logger.info("✅ Setup validation successful")
            return True
        finally:
            client.close()
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the setup process"""
    logger.info("Starting setup process...")
    
    # Step 1: Check Weaviate
    if not check_weaviate():
        logger.error("❌ Weaviate check failed. Please make sure Weaviate is running.")
        logger.error("Run 'docker-compose up -d' to start Weaviate")
        return False
        
    # Step 2: Set up schemas
    if not setup_schemas():
        logger.error("❌ Schema setup failed")
        return False
        
    # Step 3: Validate setup
    if not validate_setup():
        logger.error("❌ Setup validation failed")
        return False
        
    logger.info("✅ Setup completed successfully!")
    logger.info("You can now run the agentic_rag.py script:")
    logger.info("    python agentic_rag.py --interactive")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)