# File: export_latest_weaviate_data.py

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import json

# --- Add project root to path ---
# Adjust this path if the script is located elsewhere relative to your project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # Assuming get_weaviate_client is in Sample_Data/vector_store/weaviate_client.py
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
except ImportError as e:
    print(f"Error importing Weaviate client. Make sure the path is correct: {e}")
    print("Attempting direct import...")
    try:
        # Try importing directly if the structure isn't as assumed
        from weaviate_client import get_weaviate_client
    except ImportError:
        print("Direct import also failed. Please check your project structure and PYTHONPATH.")
        sys.exit(1)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file

# --- Helper Function to Find Timestamp Field ---
def get_likely_timestamp_field(collection_name: str) -> str:
    """
    Guesses the most likely timestamp field name based on common patterns
    in your schemas. Adjust this dictionary as needed.
    """
    field_map = {
        "CryptoDueDiligenceDocuments": "date", # Or maybe 'upload_date' if more relevant for recency?
        "CryptoNewsSentiment": "date", # Or 'analyzed_at'?
        "UserDocuments": "upload_date", # This seems most appropriate for "last" user docs
        "MarketMetrics": "timestamp",
        "CryptoTimeSeries": "timestamp",
        "OnChainAnalytics": "analysis_timestamp", # Or 'last_activity'?
        "Forecast": "forecast_timestamp",
        "UserQAHistory": "timestamp",
        "RiskProfiles": "analysis_timestamp"
        # Add other collections if necessary
    }
    # Default guess if collection name isn't mapped
    return field_map.get(collection_name, "timestamp")

# --- Main Export Function ---
def export_last_10_items(output_filename: str = None):
    """
    Connects to Weaviate, retrieves the last 10 items from each collection,
    and writes their properties to a text file.
    """
    client = None
    if not output_filename:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"weaviate_latest_data_{timestamp_str}.txt"

    try:
        logger.info("Connecting to Weaviate...")
        client = get_weaviate_client()
        if not client.is_connected():
             logger.error("Failed to connect to Weaviate. Aborting.")
             return

        logger.info("Successfully connected to Weaviate.")

        logger.info("Fetching list of collections...")
        try:
            collections_list = client.collections.list_all() # Gets detailed info
            collection_names = [details.name for details in collections_list.values()]
            logger.info(f"Found collections: {collection_names}")
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return

        logger.info(f"Preparing to write data to {output_filename}...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"Weaviate Latest Data Export - {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

            for name in collection_names:
                logger.info(f"Processing collection: {name}")
                f.write(f"--- Collection: {name} ---\n\n")

                try:
                    collection = client.collections.get(name)
                    timestamp_field = get_likely_timestamp_field(name)
                    logger.info(f"Attempting to sort by field: '{timestamp_field}'")

                    # Import Sort class here
                    from weaviate.classes.query import Sort

                    try:
                        # Try fetching sorted by the likely timestamp field
                        response = collection.query.fetch_objects(
                            limit=10,
                            sort=Sort.by_property(timestamp_field, ascending=False)
                        )
                        logger.info(f"Successfully queried {name} sorted by '{timestamp_field}'.")

                    except Exception as sort_error:
                        logger.warning(f"Could not sort collection '{name}' by field '{timestamp_field}': {sort_error}")
                        logger.warning(f"Attempting to fetch last 10 items from '{name}' without specific sorting...")
                        # Fallback: fetch any 10 items if sorting fails
                        response = collection.query.fetch_objects(limit=10)


                    if not response.objects:
                        logger.info(f"No items found in collection: {name}")
                        f.write("  (No items found)\n\n")
                        continue

                    logger.info(f"Retrieved {len(response.objects)} items from {name}")
                    for i, item in enumerate(response.objects):
                        f.write(f"Item {i+1}:\n")
                        f.write(f"  UUID: {item.uuid}\n")
                        if item.properties:
                            for key, value in item.properties.items():
                                # Basic formatting for readability in TXT
                                value_str = str(value)
                                if len(value_str) > 150: # Truncate long text fields
                                     value_str = value_str[:150] + "..."
                                f.write(f"  {key}: {value_str}\n")
                        else:
                            f.write("  (No properties found)\n")
                        f.write("-" * 20 + "\n") # Separator between items

                except Exception as e:
                    logger.error(f"Failed to process collection {name}: {e}")
                    f.write(f"  (Error processing collection: {e})\n")

                f.write("\n") # Space after each collection

        logger.info(f"Successfully exported latest data to {output_filename}")

    except Exception as e:
        logger.error(f"An error occurred during the export process: {e}", exc_info=True)
    finally:
        if client and client.is_connected():
            logger.info("Closing Weaviate connection.")
            client.close()

# --- Run the Export ---
if __name__ == "__main__":
    export_last_10_items()


