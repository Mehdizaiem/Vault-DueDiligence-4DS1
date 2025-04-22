#!/usr/bin/env python
"""
Script to retrieve QA history for reporting purposes.
"""
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

# Import from your existing code
from Sample_Data.vector_store.storage_manager import StorageManager
from weaviate.classes.query import Filter, Sort

def main():
    """Parse arguments and retrieve QA history"""
    parser = argparse.ArgumentParser(description='Retrieve QA history from Weaviate')
    
    # Filter arguments
    parser.add_argument('--user_id', help='Filter by user ID')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    parser.add_argument('--start_date', help='Start date (ISO format)')
    parser.add_argument('--end_date', help='End date (ISO format)')
    parser.add_argument('--category', help='Filter by primary category')
    parser.add_argument('--entity', help='Filter by crypto entity')
    parser.add_argument('--session_id', help='Filter by session ID')
    parser.add_argument('--limit', type=int, default=100, help='Maximum number of records to return')
    
    args = parser.parse_args()
    
    # Initialize storage manager
    storage_manager = StorageManager()
    try:
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            print(json.dumps({"success": False, "error": "Failed to connect to Weaviate"}))
            return 1
        
        # Build filter
        filter_conditions = []
        
        # Date filter
        if args.start_date:
            filter_conditions.append(Filter.by_property("timestamp").greater_than(args.start_date))
        elif args.days:
            # Calculate date from days ago
            start_date = (datetime.now() - timedelta(days=args.days)).isoformat()
            filter_conditions.append(Filter.by_property("timestamp").greater_than(start_date))
        
        if args.end_date:
            filter_conditions.append(Filter.by_property("timestamp").less_than(args.end_date))
        
        # User filter
        if args.user_id:
            filter_conditions.append(Filter.by_property("user_id").equal(args.user_id))
        
        # Category filter
        if args.category:
            filter_conditions.append(Filter.by_property("primary_category").equal(args.category))
        
        # Entity filter
        if args.entity:
            filter_conditions.append(Filter.by_property("crypto_entities").contains_any([args.entity]))
        
        # Session filter
        if args.session_id:
            filter_conditions.append(Filter.by_property("session_id").equal(args.session_id))
        
        # Combine filters
        combined_filter = None
        for condition in filter_conditions:
            if combined_filter is None:
                combined_filter = condition
            else:
                combined_filter = combined_filter & condition
        
        # Get collection
        collection = storage_manager.client.collections.get("UserQAHistory")
        
        # Execute query
        response = collection.query.fetch_objects(
            filters=combined_filter,
            limit=args.limit,
            sort=Sort.by_property("timestamp", ascending=False)
        )
        
        # Format results
        results = []
        for obj in response.objects:
            results.append({
                "id": str(obj.uuid),
                **obj.properties
            })
        
        # Return results
        print(json.dumps({"success": True, "interactions": results}))
        return 0
    
    except Exception as e:
        logger.error(f"Error retrieving QA history: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        return 1
    finally:
        storage_manager.close()

if __name__ == "__main__":
    sys.exit(main())