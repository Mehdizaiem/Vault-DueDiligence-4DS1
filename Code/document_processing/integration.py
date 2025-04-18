import logging
import hashlib
from typing import Dict, Any, Optional, Union
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor
from Code.document_processing.document_tracker import DocumentTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
doc_processor = None
doc_tracker = None

def get_document_processor(use_gpu=False):
    """Get or initialize the document processor"""
    global doc_processor
    if doc_processor is None:
        doc_processor = CryptoDocumentProcessor(use_gpu=use_gpu)
        logger.info("Document processor initialized")
    return doc_processor

def get_document_tracker(tracker_file=None):
    """Get or initialize the document tracker
    
    Args:
        tracker_file (str, optional): Path to the tracker file. If None, uses default location.
        
    Returns:
        DocumentTracker: The document tracker instance
    """
    global doc_tracker
    if doc_tracker is None:
        # Set default tracker file location if not specified
        if tracker_file is None:
            tracker_file = os.path.abspath(os.path.join(project_root, "Code", "document_processing", "output", "processed_documents.json"))
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
        
        doc_tracker = DocumentTracker(tracker_file=tracker_file)
        logger.info(f"Document tracker initialized with file: {os.path.abspath(tracker_file)}")
    return doc_tracker

def process_document_with_tracking(content: Union[str, bytes], filename: str, document_type: Optional[str] = None):
    """Process a document with change tracking
    
    Args:
        content (Union[str, bytes]): Document content
        filename (str): Document filename
        document_type (str, optional): Document type
        
    Returns:
        Dict: Processed document data, or None if no processing was needed
    """
    # Get processor and tracker
    processor = get_document_processor()
    tracker = get_document_tracker()
    
    # Determine if content is a file path
    is_file_path = isinstance(content, str) and os.path.exists(content)
    
    # Create identifier based on whether it's a file
    if is_file_path:
        # Use absolute path as identifier when content is a file path
        identifier = os.path.abspath(content)
        
        # Use the same file hash calculation method as the standalone script
        content_hash = tracker._compute_file_hash(identifier)
        
        # If we need to process, we'll need the actual content
        file_content = None
    else:
        # For non-file content, create a more consistent identifier
        # Instead of using hash(str(filename)) which can change between runs
        # Use a more stable hash based on the filename itself
        identifier = os.path.abspath(os.path.join(project_root, "Sample_Data", "raw_documents", filename))
        
        # Check if the file actually exists at this path - if so, use that path instead
        if os.path.exists(identifier):
            # This is actually a file in our raw_documents directory
            content_hash = tracker._compute_file_hash(identifier)
            file_content = content
        else:
            # This is truly content-only data, calculate hash based on content type
            if isinstance(content, bytes):
                content_hash = hashlib.sha256(content).hexdigest()
            else:
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            file_content = content
    
    try:
        # Check if we've processed this content before
        if identifier in tracker.document_records:
            record = tracker.document_records[identifier]
            if record["hash"] == content_hash and record["processed_success"]:
                logger.info(f"Document '{filename}' hasn't changed, skipping processing")
                return None  # No processing needed
        
        # For file paths, we need to read the content before processing
        if is_file_path:
            # Read the file based on type
            if filename.lower().endswith('.pdf'):
                from Code.document_processing.process_documents import read_pdf
                file_content = read_pdf(content)
            elif filename.lower().endswith('.docx'):
                from Code.document_processing.process_documents import read_docx
                file_content = read_docx(content)
            elif filename.lower().endswith('.txt'):
                from Code.document_processing.process_documents import read_text_file
                file_content = read_text_file(content)
                
            if file_content is None:
                logger.error(f"Could not extract text from {filename}")
                return None
        
        # Process the document with the text content
        logger.info(f"Processing document: {filename}")
        result = processor.process_document(text=file_content, filename=filename, document_type=document_type)
        
        # Update tracker directly without writing to a file
        tracker.document_records[identifier] = {
            "filename": filename,
            "mtime": datetime.now().timestamp(),
            "hash": content_hash,
            "last_processed": datetime.now().isoformat(),
            "processed_success": True
        }
        tracker._save_tracker()
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing document {filename} with tracking: {e}")
        return None

def store_processed_document(doc_data, storage_manager):
    """Store processed document data in Weaviate"""
    try:
        # If doc_data is None, it means no processing was needed (unchanged document)
        if doc_data is None:
            return True
            
        # Extract document metadata
        metadata = doc_data.get("metadata", {})
        document_type = doc_data.get("document_type", "other")
        
        # Prepare document for storage
        document = {
            "content": metadata.get("text", ""),
            "title": metadata.get("source", "Untitled Document"),
            "document_type": document_type,
            "source": metadata.get("source", "unknown"),
            "word_count": metadata.get("word_count", 0),
            "sentence_count": metadata.get("sentence_count", 0),
            "keywords": doc_data.get("entities", {}).get("keywords", []),
            "org_entities": doc_data.get("entities", {}).get("organizations", []),
            "person_entities": doc_data.get("entities", {}).get("persons", []),
            "location_entities": doc_data.get("entities", {}).get("locations", []),
            "extracted_risk_score": 50,  # Default value
        }
        
        # Store document in Weaviate
        success = storage_manager.store_due_diligence_document(document)
        return success
    except Exception as e:
        logger.error(f"Error storing processed document: {e}")
        return False