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

# Modifications for process_user_document.py

def read_document_content(file_path):
    """
    Read document content based on file type with improved error handling
    
    Args:
        file_path: Path to the document file
        
    Returns:
        str: Document content
    """
    try:
        # Determine document type
        doc_type = get_document_type_from_extension(file_path)
        
        # Convert to Path object for compatibility
        from pathlib import Path
        file_path_obj = Path(file_path)
        
        # Import the read functions from process_documents
        from Code.document_processing.process_documents import read_pdf, read_docx, read_text_file
        
        # Use the appropriate reader function
        if doc_type == 'pdf':
            logger.info(f"Reading PDF file: {file_path}")
            # Modified PDF reading approach to avoid the 'str' object has no attribute 'name' error
            try:
                # First try with PyMuPDF for better results if available
                import fitz  # PyMuPDF
                
                logger.info("Using PyMuPDF for PDF extraction")
                with fitz.open(file_path) as pdf:
                    content = ""
                    for page in pdf:
                        content += page.get_text()
                
                logger.info(f"Successfully extracted {len(content)} characters from PDF using PyMuPDF")
                return content
            except ImportError:
                logger.info("PyMuPDF not available, falling back to PyPDF2")
                content = read_pdf(file_path_obj)
                logger.info(f"Extracted {len(content)} characters from PDF using PyPDF2")
                return content
            except Exception as pdf_err:
                logger.error(f"Error with PyMuPDF: {pdf_err}, falling back to PyPDF2")
                # Try the original PDF reader as fallback
                content = read_pdf(file_path_obj)
                logger.info(f"Extracted {len(content)} characters from PDF using PyPDF2")
                return content
                
        elif doc_type == 'docx':
            return read_docx(file_path_obj)
        elif doc_type == 'text':
            return read_text_file(file_path_obj)
        else:
            logger.error(f"Unsupported document type: {doc_type}")
            return None
    except Exception as e:
        logger.error(f"Error reading document content: {e}")
        logger.error(traceback.format_exc())
        return None

# In process_user_document.py, modify the process_document function:

def process_document(file_path, processor):
    """
    Process document using the CryptoDocumentProcessor with improved content handling
    
    Args:
        file_path: Path to the document file
        processor: CryptoDocumentProcessor instance
        
    Returns:
        dict: Processed document data
    """
    try:
        # Read document content
        content = read_document_content(file_path)
        
        # Log content extraction results
        if content:
            logger.info(f"Successfully extracted {len(content)} characters from {os.path.basename(file_path)}")
        else:
            logger.error(f"Failed to extract content from {file_path}")
            return None
        
        # Process the document
        file_name = os.path.basename(file_path)
        result = processor.process_document(content, file_name)
        
        if result:
            # Ensure content is stored in the result
            if "metadata" not in result:
                result["metadata"] = {}
            
            # Store the content in multiple locations to ensure it's available
            result["metadata"]["text"] = content
            result["metadata"]["content"] = content
            result["content"] = content
            
            logger.info(f"Document processed successfully with {len(result.keys())} keys in result")
            return result
        else:
            logger.error("Document processing returned None or empty result")
            return None
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        logger.error(traceback.format_exc())
        return None

def store_user_document(storage_manager, processed_data, user_id, document_id, file_path, is_public, notes):
    """
    Store processed document in UserDocuments collection with improved content handling
    
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
        
        # Extract document content - first check if it's in processed_data
        content = ""
        if processed_data:
            # Check multiple possible locations for content
            if "content" in processed_data:
                content = processed_data["content"]
                logger.info(f"Found content in processed_data.content: {len(content)} characters")
            elif "metadata" in processed_data and "text" in processed_data["metadata"]:
                content = processed_data["metadata"]["text"]
                logger.info(f"Found content in processed_data.metadata.text: {len(content)} characters")
            elif "metadata" in processed_data and "content" in processed_data["metadata"]:
                content = processed_data["metadata"]["content"]
                logger.info(f"Found content in processed_data.metadata.content: {len(content)} characters")
        
        # If we still don't have content, try to read it directly
        if not content:
            logger.warning("No content found in processed data, reading file directly")
            content = read_document_content(file_path)
            if content:
                logger.info(f"Read {len(content)} characters directly from file")
            else:
                logger.error("Failed to read content directly from file")
                content = "Error: Failed to extract content from this document."
        
        logger.info(f"Content length before storing: {len(content)} characters")
        
        # Generate embedding with MPNet
        vector = generate_mpnet_embedding(content)
        
        # Prepare properties
        properties = {
            "content": content,  # Store the content here
            "title": os.path.basename(file_path),
            "document_type": processed_data.get("document_type", file_type) if processed_data else file_type,
            "source": os.path.basename(file_path),
            "user_id": user_id,
            "upload_date": datetime.now().isoformat(),
            "is_public": is_public,
            "file_size": file_size,
            "file_type": file_type,
            "processing_status": "completed" if content else "failed",
            "notes": notes
        }
        
        # Add additional properties from processed data if available
        if processed_data:
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
        logger.info("Inserting document into Weaviate collection")
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