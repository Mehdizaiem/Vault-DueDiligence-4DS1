#!/usr/bin/env python
"""
Script to process a user-uploaded document
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor
from Code.document_processing.document_tracker import DocumentTracker
from Sample_Data.vector_store.storage_manager import StorageManager
from Sample_Data.vector_store.embed import generate_mpnet_embedding
from weaviate.classes.config import DataType, Configure

# In process_user_document.py
def setup_user_documents_schema(storage_manager):
    """
    Set up the UserDocuments collection if it doesn't exist.
    """
    try:
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            return False
        
        # Import the schema creation function
        from Sample_Data.vector_store.schema_manager import create_user_documents_schema
        
        # Get the client from storage manager
        client = storage_manager.client
        
        # Create the schema
        create_user_documents_schema(client)
        
        logger.info("UserDocuments schema setup complete")
        return True
    except Exception as e:
        logger.error(f"Error setting up UserDocuments schema: {e}")
        return False

def get_document_type_from_extension(file_path):
    """
    Determine document type from file extension
    
    Args:
        file_path: Path to the document file
        
    Returns:
        str: Document type
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.pdf':
        return 'pdf'
    elif extension == '.docx':
        return 'docx'
    elif extension == '.txt':
        return 'text'
    else:
        return 'unknown'

def read_document_content(file_path):
    """
    Read document content based on file type
    
    Args:
        file_path: Path to the document file
        
    Returns:
        str: Document content
    """
    try:
        # Determine document type
        doc_type = get_document_type_from_extension(file_path)
        
        # Use the appropriate reader function
        if doc_type == 'pdf':
            from Code.document_processing.process_documents import read_pdf
            return read_pdf(file_path)
        elif doc_type == 'docx':
            from Code.document_processing.process_documents import read_docx
            return read_docx(file_path)
        elif doc_type == 'text':
            from Code.document_processing.process_documents import read_text_file
            return read_text_file(file_path)
        else:
            logger.error(f"Unsupported document type: {doc_type}")
            return None
    except Exception as e:
        logger.error(f"Error reading document content: {e}")
        logger.error(traceback.format_exc())
        return None

def process_document(file_path, processor):
    """
    Process document using the CryptoDocumentProcessor
    
    Args:
        file_path: Path to the document file
        processor: CryptoDocumentProcessor instance
        
    Returns:
        dict: Processed document data
    """
    try:
        # Read document content
        content = read_document_content(file_path)
        
        if content is None:
            logger.error(f"Failed to read content from {file_path}")
            return None
        
        # Process the document
        file_name = os.path.basename(file_path)
        result = processor.process_document(content, file_name)
        
        return result
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        logger.error(traceback.format_exc())
        return None

def store_user_document(storage_manager, processed_data, user_id, document_id, file_path, is_public, notes):
    """
    Store processed document in UserDocuments collection
    
    Args:
        storage_manager: StorageManager instance
        processed_data: Processed document data
        user_id: User ID
        document_id: Document ID
        file_path: Path to the document file
        is_public: Whether the document is publicly accessible
        notes: User-provided notes
        
    Returns:
        bool: Success status
    """
    try:
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            return False
        
        client = storage_manager.client
        
        # Get UserDocuments collection
        collection = client.collections.get("UserDocuments")
        
        # Get file metadata
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_type = get_document_type_from_extension(file_path)
        
        # Extract document content
        content = processed_data.get("metadata", {}).get("text", "")
        if not content:
            # Check multiple possible locations for content
            content = processed_data.get("content", "")
            if not content and "metadata" in processed_data:
                content = processed_data["metadata"].get("content", "")
            # Also check if raw content might be directly in processed_data
        if not content:
            content = read_document_content(file_path)  # Fallback to re-reading the file
        logger.info(f"Content length before storing: {len(content)} characters")
        logger.info(f"Content structure in processed_data: {json.dumps(list(processed_data.keys()))}")
        if "metadata" in processed_data:
            logger.info(f"Metadata keys: {json.dumps(list(processed_data['metadata'].keys()))}")
        # Generate embedding with MPNet
        vector = generate_mpnet_embedding(content)
        
        # Prepare properties
        properties = {
            "content": content,
            "title": os.path.basename(file_path),
            "document_type": processed_data.get("document_type", file_type),
            "source": os.path.basename(file_path),
            "user_id": user_id,
            "upload_date": datetime.now().isoformat(),
            "is_public": is_public,
            "file_size": file_size,
            "file_type": file_type,
            "processing_status": "completed",
            "notes": notes
        }
        
        # Add additional properties from processed data
        metadata = processed_data.get("metadata", {})
        entities = processed_data.get("entities", {})
        
        if metadata:
            properties["word_count"] = metadata.get("word_count", 0)
            properties["sentence_count"] = metadata.get("sentence_count", 0)
            
            # Handle dates properly
            if "document_date" in metadata:
                properties["date"] = metadata["document_date"]
        
        if entities:
            if "cryptocurrencies" in entities:
                properties["crypto_entities"] = entities["cryptocurrencies"]
            
            if "persons" in entities:
                properties["person_entities"] = entities["persons"]
                
            if "organizations" in entities:
                properties["org_entities"] = entities["organizations"]
                
            if "locations" in entities:
                properties["location_entities"] = entities["locations"]
                
            if "keywords" in entities:
                properties["keywords"] = entities["keywords"]
        
        # Add risk information
        risk_info = processed_data.get("risk_indicators", {})
        if risk_info and "risk_indicators" in risk_info:
            properties["risk_factors"] = risk_info["risk_indicators"]
        
        # Store the document
        collection.data.insert(properties=properties, vector=vector)
        
        logger.info(f"Successfully stored user document: {properties['title']}")
        return True
    except Exception as e:
        logger.error(f"Error storing user document: {e}")
        logger.error(traceback.format_exc())
        return False

def update_processing_status(storage_manager, user_id, document_id, status):
    """
    Update document processing status
    
    Args:
        storage_manager: StorageManager instance
        user_id: User ID
        document_id: Document ID
        status: Processing status
        
    Returns:
        bool: Success status
    """
    try:
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate")
            return False
        
        client = storage_manager.client
        
        # Get UserDocuments collection
        collection = client.collections.get("UserDocuments")
        
        # Find the document
        from weaviate.classes.query import Filter
        
        response = collection.query.fetch_objects(
            filters=Filter.by_property("user_id").equal(user_id) & 
                   Filter.by_property("title").equal(document_id)
        )
        
        if not response.objects:
            logger.error(f"Document not found: {document_id}")
            return False
        
        # Update the status
        for obj in response.objects:
            try:
                collection.data.update(
                    uuid=obj.uuid,
                    properties={
                        "processing_status": status,
                        "upload_date": datetime.now().isoformat()  # Update timestamp
                    }
                )
            except Exception as e:
                logger.error(f"Error updating document status: {e}")
                continue
        
        return True
    except Exception as e:
        logger.error(f"Error updating processing status: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process user-uploaded document')
    parser.add_argument('--file', required=True, help='Path to the document file')
    parser.add_argument('--user_id', required=True, help='User ID')
    parser.add_argument('--document_id', required=True, help='Document ID')
    parser.add_argument('--is_public', type=bool, default=False, help='Whether the document is publicly accessible')
    parser.add_argument('--notes', default='', help='User-provided notes')
    args = parser.parse_args()
    
    # Initialize storage manager
    storage_manager = StorageManager()
    
    try:
        # Set up schema if needed
        setup_user_documents_schema(storage_manager)
        
        # Update status to processing
        update_processing_status(storage_manager, args.user_id, args.document_id, "processing")
        
        # Initialize document processor
        processor = CryptoDocumentProcessor()
        
        # Process the document
        logger.info(f"Processing document: {args.file}")
        processed_data = process_document(args.file, processor)
        
        if processed_data is None:
            logger.error(f"Failed to process document: {args.file}")
            update_processing_status(storage_manager, args.user_id, args.document_id, "failed")
            return 1
        
        # Store the processed document
        success = store_user_document(
            storage_manager, 
            processed_data,
            args.user_id,
            args.document_id,
            args.file,
            args.is_public,
            args.notes
        )
        
        if not success:
            logger.error(f"Failed to store document: {args.file}")
            update_processing_status(storage_manager, args.user_id, args.document_id, "failed")
            return 1
        
        logger.info(f"Successfully processed and stored document: {args.file}")
        return 0
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        logger.error(traceback.format_exc())
        update_processing_status(storage_manager, args.user_id, args.document_id, "failed")
        return 1
    finally:
        # Close storage manager connection
        storage_manager.close()

if __name__ == "__main__":
    sys.exit(main())