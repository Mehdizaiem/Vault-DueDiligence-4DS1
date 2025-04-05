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
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            tracker_file = os.path.join(output_dir, "processed_documents.json")
        
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
    
    # Create a temporary file path object for the tracker
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique key for this content
    content_key = f"content_{filename}_{hash(str(filename))}"
    
    try:
        # Skip the file writing step entirely - compute hash directly
        if isinstance(content, bytes):
            # For binary content
            content_hash = hashlib.sha256(content).hexdigest()
        else:
            # For text content
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Check if we've processed this content before
        if content_key in tracker.document_records:
            record = tracker.document_records[content_key]
            if record["hash"] == content_hash and record["processed_success"]:
                logger.info(f"Document '{filename}' hasn't changed, skipping processing")
                return None  # No processing needed
        
        # Process the document
        logger.info(f"Processing document: {filename}")
        result = processor.process_document(text=content, filename=filename, document_type=document_type)
        
        # Update tracker directly without writing to a file
        tracker.document_records[content_key] = {
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