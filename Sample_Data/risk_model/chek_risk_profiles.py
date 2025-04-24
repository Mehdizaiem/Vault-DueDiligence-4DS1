# File: check_riskprofiles.py (or similar name)

import os
import sys
import logging
from datetime import datetime, timezone
import weaviate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed (adjust if your script is elsewhere)
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
except ImportError as e:
     # Try adjusting path if running from a different directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Go up one level less
    if project_root not in sys.path:
         sys.path.append(project_root)
    # Try importing again
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    except ImportError:
         logger.error(f"Could not import get_weaviate_client. Ensure script is run from a location where Sample_Data is accessible or adjust sys.path: {e}")
         sys.exit(1)

COLLECTION_NAME = "RiskProfiles"

def check_collection():
    client = None
    try:
        logger.info(f"Attempting to connect to Weaviate...")
        client = get_weaviate_client()
        if not client or not client.is_live():
            logger.error("Failed to connect to Weaviate.")
            return

        logger.info("Successfully connected to Weaviate.")

        # 1. Check if the collection exists
        logger.info(f"Checking if collection '{COLLECTION_NAME}' exists...")
        all_collections = client.collections.list_all()
        collection_names = list(all_collections.keys()) # Get collection names

        if COLLECTION_NAME in collection_names:
            logger.info(f"✅ Collection '{COLLECTION_NAME}' found!")

            # 2. Get the collection object
            collection = client.collections.get(COLLECTION_NAME)

            # 3. Get the total count of objects
            # Note: Weaviate v4 aggregate syntax might differ slightly based on exact version
            try:
                 aggregate_result = collection.aggregate.over_all(total_count=True)
                 total_count = aggregate_result.total_count
                 logger.info(f"📊 Total objects found in '{COLLECTION_NAME}': {total_count}")
            except Exception as agg_e:
                 logger.warning(f"Could not get aggregate count for '{COLLECTION_NAME}': {agg_e}. Fetching objects directly.")
                 total_count = -1 # Indicate count unknown


            # 4. Fetch a few sample objects to verify data
            if total_count == 0:
                 logger.warning(f"Collection '{COLLECTION_NAME}' exists but is currently empty.")
            elif total_count > 0 or total_count == -1: # Proceed if count > 0 or unknown
                 logger.info(f"Fetching up to 5 sample objects from '{COLLECTION_NAME}'...")
                 try:
                     # Fetch most recently stored profiles first
                     response = collection.query.fetch_objects(
                         limit=5,
                         sort=weaviate.classes.query.Sort.by_property("analysis_timestamp", ascending=False) # Requires import Sort
                     )

                     if not response.objects:
                         logger.warning(f"Collection '{COLLECTION_NAME}' exists but no objects could be fetched (might be empty or query issue).")
                     else:
                         logger.info(f"🔍 Sample data from '{COLLECTION_NAME}':")
                         for i, obj in enumerate(response.objects):
                             print(f"--- Sample Object {i+1} ---")
                             # Print key properties
                             print(f"  UUID: {obj.uuid}")
                             print(f"  Symbol: {obj.properties.get('symbol')}")
                             print(f"  Score: {obj.properties.get('risk_score')}")
                             print(f"  Category: {obj.properties.get('risk_category')}")
                             print(f"  Timestamp: {obj.properties.get('analysis_timestamp')}")
                             print(f"  Factors: {obj.properties.get('risk_factors')}")
                             print(f"  Error: {obj.properties.get('calculation_error')}")
                             print("-" * 20)

                 except Exception as fetch_e:
                      logger.error(f"Error fetching objects from '{COLLECTION_NAME}': {fetch_e}", exc_info=True)

        else:
            logger.error(f"❌ Collection '{COLLECTION_NAME}' does NOT exist!")
            logger.error("Please ensure you have run the schema_manager.py script to create it.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if client:
            client.close()
            logger.info("Weaviate connection closed.")

if __name__ == "__main__":
    # Need to import Sort here if not imported globally
    from weaviate.classes.query import Sort
    check_collection()