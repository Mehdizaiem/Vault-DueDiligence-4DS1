#!/usr/bin/env python
"""
Script to store QA interactions in the Weaviate UserQAHistory collection.
"""
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

# Import from your existing code
from Sample_Data.vector_store.storage_manager import StorageManager

def main():
    """Parse arguments and store QA interaction"""
    parser = argparse.ArgumentParser(description='Store QA interaction in Weaviate')
    
    # Required arguments
    parser.add_argument('--user_id', required=True, help='ID of the user who asked the question')
    parser.add_argument('--question', required=True, help='The user\'s question')
    parser.add_argument('--answer', required=True, help='The AI\'s answer')
    
    # Optional arguments
    parser.add_argument('--session_id', help='Session ID for grouping related Q&A')
    parser.add_argument('--analysis', help='JSON string of query analysis results')
    parser.add_argument('--document_ids', help='JSON array of document IDs referenced')
    parser.add_argument('--feedback', help='JSON string of user feedback')
    parser.add_argument('--duration_ms', type=int, help='Time taken to generate the answer in ms')
    
    args = parser.parse_args()
    
    # Initialize storage manager
    storage_manager = StorageManager()
    try:
        # Parse complex arguments
        analysis = json.loads(args.analysis) if args.analysis else None
        document_ids = json.loads(args.document_ids) if args.document_ids else None
        feedback = json.loads(args.feedback) if args.feedback else None
        
        # Create missing schema if needed
        from Sample_Data.vector_store.schema_manager import create_user_qa_history_schema
        if storage_manager.connect():
            try:
                create_user_qa_history_schema(storage_manager.client)
            except Exception as schema_error:
                logger.warning(f"Error ensuring schema exists: {schema_error}")
        
        # Store the interaction
        success = storage_manager.store_qa_interaction(
            question=args.question,
            answer=args.answer,
            user_id=args.user_id,
            analysis=analysis,
            document_ids=document_ids,
            session_id=args.session_id,
            feedback=feedback,
            duration_ms=args.duration_ms
        )
        
        # Return result
        result = {"success": success}
        print(json.dumps(result))
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Error storing QA interaction: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        return 1
    finally:
        storage_manager.close()

if __name__ == "__main__":
    sys.exit(main())