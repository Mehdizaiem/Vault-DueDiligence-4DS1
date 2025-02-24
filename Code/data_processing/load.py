from typing import List, Dict, Any
import sys
import os
# Add Sample_Data to path (from Code/data_processing, up to root, then into Sample_Data)
SAMPLE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Sample_Data'))
sys.path.append(SAMPLE_DATA_PATH)

# Now import from vector_store
from vector_store.weaviate_client import get_weaviate_client, check_weaviate_connection

import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_market_data(client):
    """Remove the MarketMetrics collection"""
    try:
        client.collections.delete("MarketMetrics")
        logger.info("✅ Cleaned up existing market data")
    except Exception as e:
        logger.info(f"No existing collection to clean up: {e}")

def create_market_schema(client):
    """Create MarketMetrics collection if it doesn't exist"""
    try:
        collection = client.collections.get("MarketMetrics")
        logger.info("Collection 'MarketMetrics' already exists")
        return collection
    except Exception:
        logger.info("Creating 'MarketMetrics' collection")
        try:
            collection = client.collections.create(
                name="MarketMetrics",
                vectorizer_config=None,  # No embeddings for market data
                properties=[
                    {
                        "name": "symbol",
                        "dataType": ["text"],
                        "description": "Trading symbol (e.g., BTCUSDT)"
                    },
                    {
                        "name": "source",
                        "dataType": ["text"],
                        "description": "Data source (e.g., binance, coingecko)"
                    },
                    {
                        "name": "price",
                        "dataType": ["number"],
                        "description": "Current price in USD"
                    },
                    {
                        "name": "market_cap",
                        "dataType": ["number"],
                        "description": "Market capitalization"
                    },
                    {
                        "name": "volume_24h",
                        "dataType": ["number"],
                        "description": "24h trading volume"
                    },
                    {
                        "name": "price_change_24h",
                        "dataType": ["number"],
                        "description": "24h price change percentage"
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Data timestamp"
                    }
                ]
            )
            logger.info("✅ Successfully created MarketMetrics collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create MarketMetrics collection: {str(e)}")
            raise

def load(data: List[Dict[str, Any]]) -> None:
    """Load transformed market data into Weaviate"""
    if not data:
        logger.warning("No data to load")
        return

    BATCH_SIZE = 100
    
    client = get_weaviate_client()
    try:
        # First, cleanup existing data
        cleanup_market_data(client)
        
        # Get or create the MarketMetrics collection
        collection = create_market_schema(client)
        
        # Process in batches
        total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
        processed = 0
        
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                objects_to_add = []
                for item in batch:
                    # Extract market data
                    market_data = item.get("data", {})
                    source = item.get("source", "unknown")
                    
                    # Log sample data for debugging
                    if batch_num == 1 and len(objects_to_add) == 0:
                        logger.info(f"Sample market data: {market_data}")
                    
                    # Prepare properties
                    properties = {
                        "symbol": market_data.get("symbol", "UNKNOWN"),
                        "source": source,
                        "price": float(market_data.get("price_usd", 0)),
                        "market_cap": float(market_data.get("market_cap_usd", 0)),
                        "volume_24h": float(market_data.get("volume_24h_usd", 0)),
                        "price_change_24h": float(market_data.get("price_change_24h", 0)),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Only add if we have valid price data
                    if properties["price"] > 0 or properties["market_cap"] > 0:
                        objects_to_add.append(properties)
                
                if objects_to_add:
                    # Use insert_many for batch processing in Weaviate v4
                    response = collection.data.insert_many(objects_to_add)
                    
                    if response.has_errors:
                        logger.error(f"Batch {batch_num} had errors: {response.errors}")
                    else:
                        processed += len(objects_to_add)
                        logger.info(f"✅ Batch {batch_num} processed: {len(objects_to_add)} valid records")
                else:
                    logger.warning(f"Batch {batch_num} had no valid records to add")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                continue
                
            # Small delay between batches
            time.sleep(0.1)
            
        logger.info(f"✅ Successfully processed {processed} out of {len(data)} records")
        
    except Exception as e:
        logger.error(f"❌ Error in load operation: {str(e)}")
        raise
    finally:
        client.close()