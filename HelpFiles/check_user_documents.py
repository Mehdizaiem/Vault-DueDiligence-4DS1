#!/usr/bin/env python
"""
Improved script to check the contents of the UserDocuments collection in Weaviate.
This script provides better visualization of document content and detailed properties.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import Weaviate client
try:
    import weaviate
    from weaviate.classes.query import Filter
except ImportError:
    logger.error("Weaviate client not installed. Please install it using 'pip install weaviate-client'")
    sys.exit(1)

def get_weaviate_client():
    """Create and return a Weaviate client"""
    try:
        # Try to connect to local Weaviate instance
        client = weaviate.connect_to_local(
            port=9090, 
            grpc_port=50051
        )
        
        if client.is_ready():
            logger.info("Successfully connected to Weaviate ✅")
            return client
        else:
            logger.error("Weaviate is not ready")
            return None
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return None

def check_document_by_id(client, document_id: str) -> Optional[Dict]:
    """
    Check if a document with the given ID exists in UserDocuments collection
    
    Args:
        client: Weaviate client
        document_id: Document ID to check
        
    Returns:
        Document data if found, None otherwise
    """
    try:
        # Get the UserDocuments collection
        collection = client.collections.get("UserDocuments")
        
        # Try direct ID lookup first
        try:
            obj = collection.data.get_by_id(document_id)
            if obj:
                logger.info(f"Document found directly by UUID: {document_id}")
                return {
                    "id": document_id,
                    **obj.properties
                }
        except Exception as e:
            logger.warning(f"Direct ID lookup failed: {e}")
        
        # If direct ID lookup fails, try to find document where ID matches title or source
        response = collection.query.fetch_objects(
            filters=(Filter.by_property("title").equal(document_id) | 
                     Filter.by_property("source").equal(document_id)),
            limit=1
        )
        
        if response.objects:
            obj = response.objects[0]
            logger.info(f"Document found by property matching: {document_id}")
            return {
                "id": str(obj.uuid),
                **obj.properties
            }
        
        logger.warning(f"Document not found: {document_id}")
        return None
    except Exception as e:
        logger.error(f"Error checking document: {e}")
        return None

def list_user_documents(client, user_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    List documents in the UserDocuments collection
    
    Args:
        client: Weaviate client
        user_id: Filter by user ID (optional)
        limit: Maximum number of documents to retrieve
        
    Returns:
        List of documents
    """
    try:
        # Get the UserDocuments collection
        collection = client.collections.get("UserDocuments")
        
        # Build filter if user_id is provided
        if user_id:
            filters = Filter.by_property("user_id").equal(user_id)
            logger.info(f"Listing documents for user: {user_id}")
        else:
            filters = None
            logger.info("Listing all documents")
        
        # Query for documents
        response = collection.query.fetch_objects(
            filters=filters,
            limit=limit
        )
        
        # Format results
        results = []
        for obj in response.objects:
            results.append({
                "id": str(obj.uuid),
                **obj.properties
            })
        
        return results
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []

def format_content_preview(content, max_length=500, show_full=False):
    """Format content preview with better readability"""
    if not content:
        return "[EMPTY CONTENT]"
    
    if show_full:
        return content
        
    # Show a decent preview with newlines preserved
    if len(content) > max_length:
        preview = content[:max_length].replace('\r', '')
        # Try to end at a proper sentence
        last_period = preview.rfind('.')
        if last_period > max_length * 0.7:  # Only cut at sentence if we're showing most of the preview
            preview = preview[:last_period+1]
        return preview + "...\n[Content truncated - use --full-content to see all]"
    return content

def print_document_details(doc, show_full_content=False):
    """Print document details in a structured and readable format"""
    print("\n" + "=" * 80)
    print(f"📄 DOCUMENT: {doc.get('title', 'Untitled')}")
    print("=" * 80)
    
    # Basic information
    print("\n📋 BASIC INFORMATION:")
    print(f"  • ID: {doc.get('id', 'Unknown')}")
    print(f"  • Status: {doc.get('processing_status', 'Unknown')}")
    print(f"  • Upload Date: {doc.get('upload_date', 'Unknown')}")
    print(f"  • Public: {doc.get('is_public', False)}")
    
    # File information
    print("\n📁 FILE INFORMATION:")
    print(f"  • Document Type: {doc.get('document_type', 'Unknown')}")
    print(f"  • File Type: {doc.get('file_type', 'Unknown')}")
    
    # Size information
    file_size = doc.get('file_size', 0)
    size_str = f"{file_size:,} bytes"
    if file_size > 1024:
        size_str += f" ({file_size/1024:.1f} KB)"
    if file_size > 1024 * 1024:
        size_str += f" ({file_size/(1024*1024):.1f} MB)"
    print(f"  • File Size: {size_str}")
    
    # User information
    print("\n👤 USER INFORMATION:")
    print(f"  • User ID: {doc.get('user_id', 'Unknown')}")
    
    # Document metadata
    print("\n📊 DOCUMENT METADATA:")
    word_count = doc.get('word_count')
    if word_count:
        print(f"  • Word Count: {word_count:,}")
    
    sentence_count = doc.get('sentence_count')
    if sentence_count:
        print(f"  • Sentence Count: {sentence_count:,}")
    
    # Entity information
    entity_lists = {
        'crypto_entities': '💰 CRYPTOCURRENCY ENTITIES:',
        'person_entities': '👥 PERSON ENTITIES:',
        'org_entities': '🏢 ORGANIZATION ENTITIES:',
        'location_entities': '📍 LOCATION ENTITIES:',
        'risk_factors': '⚠️ RISK FACTORS:'
    }
    
    for entity_key, header in entity_lists.items():
        entities = doc.get(entity_key, [])
        if entities:
            print(f"\n{header}")
            for entity in entities:
                print(f"  • {entity}")
    
    # Notes
    notes = doc.get('notes')
    if notes:
        print("\n📝 NOTES:")
        print(f"  {notes}")
    
    # Content (the most important part)
    print("\n📃 DOCUMENT CONTENT:")
    content = doc.get('content', '')
    
    if content:
        content_length = len(content)
        print(f"  [Content length: {content_length:,} characters]")
        print("-" * 80)
        formatted_content = format_content_preview(content, show_full=show_full_content)
        print(formatted_content)
        print("-" * 80)
    else:
        print("  [NO CONTENT AVAILABLE]")
        print("-" * 80)
        print("  ⚠️ WARNING: Document has no content. This explains why Q&A doesn't work!")
        print("  Possible reasons:")
        print("  - Content extraction failed during document processing")
        print("  - The original document was empty or couldn't be parsed")
        print("  - There was an error saving the content to the database")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description='Check UserDocuments collection in Weaviate')
    parser.add_argument('--document_id', help='Check a specific document by ID')
    parser.add_argument('--user_id', help='Filter documents by user ID')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of documents to list')
    parser.add_argument('--full-content', action='store_true', help='Show full document content')
    args = parser.parse_args()
    
    # Get Weaviate client
    client = get_weaviate_client()
    if not client:
        sys.exit(1)
    
    try:
        # Check if UserDocuments collection exists
        try:
            collection = client.collections.get("UserDocuments")
            logger.info("UserDocuments collection exists ✅")
        except Exception as e:
            logger.error(f"UserDocuments collection does not exist: {e}")
            sys.exit(1)
        
        # Check specific document if ID provided
        if args.document_id:
            document = check_document_by_id(client, args.document_id)
            if document:
                print_document_details(document, args.full_content)
            else:
                print(f"\n⚠️ Document with ID '{args.document_id}' not found!")
                
                # List documents to help user find the right one
                print("\nAvailable documents:")
                documents = list_user_documents(client, args.user_id, args.limit)
                for i, doc in enumerate(documents):
                    print(f"{i+1}. {doc.get('title', 'Untitled')} (ID: {doc.get('id')})")
        else:
            # List documents
            documents = list_user_documents(client, args.user_id, args.limit)
            
            if documents:
                print(f"\n--- FOUND {len(documents)} DOCUMENTS ---")
                for doc in documents:
                    print_document_details(doc, args.full_content)
            else:
                print("\n⚠️ No documents found!")
                print("Possible reasons:")
                print("1. No documents have been uploaded")
                print("2. The UserDocuments collection exists but is empty")
                print("3. There's an issue with the Weaviate database")
                
    finally:
        # Close Weaviate client
        client.close()
        logger.info("Weaviate client closed")

if __name__ == "__main__":
    main()