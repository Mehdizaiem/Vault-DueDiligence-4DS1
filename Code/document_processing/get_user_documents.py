#!/usr/bin/env python
"""
Script to retrieve user documents from Weaviate
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.storage_manager import StorageManager
from weaviate.classes.query import Filter, Sort

def get_user_documents(user_id, limit=20, offset=0, sort_by='upload_date', sort_order='desc', status=None):
    """
    Retrieve user documents from Weaviate.
    
    Args:
        user_id: User ID
        limit: Maximum number of documents to retrieve
        offset: Offset for pagination
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
        status: Filter by processing status
        
    Returns:
        list: User documents
    """
    try:
        # Initialize storage manager
        storage_manager = StorageManager()
        
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            return []
        
        client = storage_manager.client
        
        try:
            # Get UserDocuments collection
            collection = client.collections.get("UserDocuments")
        except Exception as e:
            logger.error(f"Error accessing UserDocuments collection: {e}")
            return []
        
        # Build filter
        filters = Filter.by_property("user_id").equal(user_id)
        
        # Add status filter if provided
        if status:
            filters = filters & Filter.by_property("processing_status").equal(status)
        
        # Build sort
        ascending = sort_order.lower() == 'asc'
        sort = Sort.by_property(sort_by, ascending=ascending)
        
        # Execute query
        results = []
        try:
            response = collection.query.fetch_objects(
                filters=filters,
                limit=limit,
                offset=offset,
                sort=sort
            )
            
            # Process results
            for obj in response.objects:
                # Extract properties
                props = obj.properties
                
                # Convert datetime objects to strings
                for key, value in props.items():
                    if isinstance(value, datetime):
                        props[key] = value.isoformat()
                
                # Add document ID
                document = {
                    "id": str(obj.uuid),
                    **props
                }
                
                # Truncate content for list view
                if "content" in document and document["content"]:
                    document["content_preview"] = document["content"][:200]
                    del document["content"]
                
                results.append(document)
                
        except Exception as e:
            logger.error(f"Error querying UserDocuments: {e}")
            logger.error(traceback.format_exc())
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving user documents: {e}")
        logger.error(traceback.format_exc())
        return []
    finally:
        if 'storage_manager' in locals() and storage_manager:
            storage_manager.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Retrieve user documents from Weaviate')
    parser.add_argument('--user_id', required=True, help='User ID')
    parser.add_argument('--limit', type=int, default=20, help='Maximum number of documents to retrieve')
    parser.add_argument('--offset', type=int, default=0, help='Offset for pagination')
    parser.add_argument('--sort_by', default='upload_date', help='Field to sort by')
    parser.add_argument('--sort_order', default='desc', help='Sort order (asc or desc)')
    parser.add_argument('--status', help='Filter by processing status')
    args = parser.parse_args()
    
    # Retrieve documents
    documents = get_user_documents(
        args.user_id,
        args.limit,
        args.offset,
        args.sort_by,
        args.sort_order,
        args.status
    )
    
    # Output as JSON
    json.dump(documents, sys.stdout)

    
    return 0

if __name__ == "__main__":
    sys.exit(main())