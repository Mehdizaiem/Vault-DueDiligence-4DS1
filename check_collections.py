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

def check_collections():
    """Check all collections in Weaviate"""
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        client = get_weaviate_client()
        logger.info("Weaviate client is connected.")
        
        try:
            # List all collections
            collections = client.collections.list_all()
            
            if not collections:
                logger.info("No collections found in Weaviate.")
                print("No collections found in Weaviate.")
                return
            
            logger.info(f"Found {len(collections)} collections in Weaviate:")
            print(f"\nFound {len(collections)} collections in Weaviate:")
            
            # Define expected collections
            expected_collections = [
                "CryptoTimeSeries",
                "MarketMetrics", 
                "CryptoNewsSentiment",
                "CryptoDueDiligenceDocuments",
                "OnChainAnalytics"
            ]
            
            # Check which expected collections are missing
            missing_collections = [coll for coll in expected_collections if coll not in collections]
            
            if missing_collections:
                print(f"\n⚠️ Missing expected collections: {', '.join(missing_collections)}")
            else:
                print("\n✅ All expected collections are present!")
            
            for name, collection_info in collections.items():
                # Get collection details
                collection = client.collections.get(name)
                
                # Count objects
                count_result = collection.aggregate.over_all(total_count=True)
                total_count = count_result.total_count
                
                # Print collection info
                logger.info(f"🔹 {name}:")
                print(f"\n🔹 {name}:")
                
                # Safely access description - might be in different places depending on API version
                description = "N/A"
                if hasattr(collection_info, 'description'):
                    description = collection_info.description
                elif isinstance(collection_info, dict) and 'description' in collection_info:
                    description = collection_info['description']
                
                logger.info(f"  - Description: {description}")
                print(f"  - Description: {description}")
                logger.info(f"  - Object count: {total_count}")
                print(f"  - Object count: {total_count}")
                
                # Get collection properties (schema)
                properties = None
                if hasattr(collection_info, 'properties'):
                    properties = collection_info.properties
                elif isinstance(collection_info, dict) and 'properties' in collection_info:
                    properties = collection_info['properties']
                elif hasattr(collection, 'properties'):
                    # Try getting properties directly from collection
                    properties = collection.properties()
                
                if properties:
                    logger.info(f"  - Properties ({len(properties)}):")
                    print(f"  - Properties ({len(properties)}):")
                    
                    for prop in properties:
                        # Handle different property formats
                        if hasattr(prop, 'name') and hasattr(prop, 'data_type'):
                            prop_name = prop.name
                            prop_type = prop.data_type[0] if isinstance(prop.data_type, list) else prop.data_type
                        elif isinstance(prop, dict):
                            prop_name = prop.get('name', 'unknown')
                            prop_type = prop.get('dataType', ['unknown'])[0] if isinstance(prop.get('dataType'), list) else prop.get('dataType', 'unknown')
                        else:
                            prop_name = str(prop)
                            prop_type = "unknown"
                            
                        logger.info(f"    • {prop_name} ({prop_type})")
                        print(f"    • {prop_name} ({prop_type})")
                else:
                    logger.info("  - No properties found or unable to access them")
                    print("  - No properties found or unable to access them")
                
            return True
        finally:
            client.close()
            logger.info("Weaviate client closed.")
            
    except Exception as e:
        logger.error(f"Error checking collections: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("\n=== Weaviate Collections Diagnostic ===\n")
    check_collections()