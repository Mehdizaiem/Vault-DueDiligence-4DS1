#!/usr/bin/env python
"""
Script to delete a user document from Weaviate
"""
import os
import sys
import json
import logging
import argparse
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.storage_manager import StorageManager

def delete_user_document(user_id, document_id):
    """
    Delete a user document from Weaviate.
    
    Args:
        user_id: User ID
        document_id: Document ID
        
    Returns:
        bool: Success status
    """
    try:
        # Initialize storage manager
        storage_manager = StorageManager()
        
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            return False
        
        client = storage_manager.client
        
        try:
            # Get UserDocuments collection
            collection = client.collections.get("UserDocuments")
        except Exception as e:
            logger.error(f"Error accessing UserDocuments collection: {e}")
            return False
        
        # Two options for document_id: it could be a UUID or a document title
        try:
            # First, try treating it as a UUID
            try:
                # Delete by UUID directly
                collection.data.delete_by_id(document_id)
                logger.info(f"Successfully deleted document with UUID {document_id}")
                return True
            except Exception:
                # If that fails, it might be a title or other identifier
                from weaviate.classes.query import Filter
                
                # Query for documents that match the user_id and have this document_id (title)
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("user_id").equal(user_id) &
                           (Filter.by_property("title").equal(document_id) |
                            Filter.by_id().equal(document_id))
                )
                
                if not response.objects:
                    logger.error(f"Document not found: {document_id}")
                    return False
                
                # Delete each matching document
                success = False
                for obj in response.objects:
                    try:
                        collection.data.delete_by_id(obj.uuid)
                        logger.info(f"Successfully deleted document with UUID {obj.uuid}")
                        success = True
                    except Exception as del_err:
                        logger.error(f"Error deleting document with UUID {obj.uuid}: {del_err}")
                
                return success
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            logger.error(traceback.format_exc())
            return False
        
    except Exception as e:
        logger.error(f"Error in delete_user_document: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        if 'storage_manager' in locals() and storage_manager:
            storage_manager.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Delete a user document from Weaviate')
    parser.add_argument('--user_id', required=True, help='User ID')
    parser.add_argument('--document_id', required=True, help='Document ID (UUID or title)')
    args = parser.parse_args()
    
    # Delete document
    success = delete_user_document(args.user_id, args.document_id)
    
    # Output as JSON
    result = {
        "success": success,
        "message": "Document deleted successfully" if success else "Failed to delete document"
    }
    print(json.dumps(result))
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())