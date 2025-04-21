#!/usr/bin/env python
"""
Script to check UserQAHistory collection data in Weaviate.
"""
import os
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

def check_qa_history_collection():
    try:
        # Import Weaviate client and get connection
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        # Connect to Weaviate
        client = get_weaviate_client()
        logger.info("Connected to Weaviate")
        
        # Check if UserQAHistory collection exists
        try:
            collection = client.collections.get("UserQAHistory")
            logger.info("UserQAHistory collection exists")
        except Exception as e:
            logger.error(f"UserQAHistory collection does not exist: {e}")
            return
        
        # Get count of records
        count_result = collection.aggregate.over_all(total_count=True)
        total_count = count_result.total_count
        logger.info(f"Total records in UserQAHistory: {total_count}")
        
        if total_count == 0:
            logger.info("No records found in the collection")
            return
        
        # Get the most recent records
        from weaviate.classes.query import Sort
        response = collection.query.fetch_objects(
            limit=5,
            sort=Sort.by_property("timestamp", ascending=False)
        )
        
        # Display records
        logger.info(f"Showing {len(response.objects)} most recent records:")
        for i, obj in enumerate(response.objects):
            logger.info(f"\n--- Record {i+1} ---")
            logger.info(f"ID: {obj.uuid}")
            logger.info(f"User ID: {obj.properties.get('user_id', 'N/A')}")
            logger.info(f"Timestamp: {obj.properties.get('timestamp', 'N/A')}")
            logger.info(f"Question: {obj.properties.get('question', 'N/A')[:100]}...")
            logger.info(f"answer: {obj.properties.get('answer', 'N/A')[:5000]}...")
            # Show other important properties
            for prop in ["primary_category", "intent", "session_id", "feedback_rating"]:
                if prop in obj.properties:
                    logger.info(f"{prop.replace('_', ' ').title()}: {obj.properties[prop]}")
            
            # If crypto entities are present, show them
            if "crypto_entities" in obj.properties and obj.properties["crypto_entities"]:
                logger.info(f"Crypto Entities: {', '.join(obj.properties['crypto_entities'])}")
    
    except Exception as e:
        logger.error(f"Error checking QA history collection: {e}")
    finally:
        if 'client' in locals():
            client.close()
            logger.info("Weaviate connection closed")

if __name__ == "__main__":
    check_qa_history_collection()