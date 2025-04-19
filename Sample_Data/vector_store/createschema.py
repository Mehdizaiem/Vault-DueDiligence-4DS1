# setup_schemas.py
import logging
from storage_manager import StorageManager
from schema_manager import create_user_documents_schema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Initialize storage manager
    storage_manager = StorageManager()
    
    try:
        # Connect to Weaviate
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            return False
        
        # Create UserDocuments collection
        create_user_documents_schema(storage_manager.client)
        logger.info("UserDocuments schema created successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up schemas: {e}")
        return False
    finally:
        # Close connection
        storage_manager.close()

if __name__ == "__main__":
    success = main()
    print(f"Schema setup {'successful' if success else 'failed'}")