import logging
import hashlib
from typing import Dict, Any, Optional, Union
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Debugging - print paths for troubleshooting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Project root: {project_root}")
logger.info(f"Python path: {sys.path}")

# Import local modules
from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor
from Code.document_processing.document_tracker import DocumentTracker

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
                # Import here to avoid circular imports
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
        logger.error(traceback.format_exc())
        return None
    
def find_similar_documents(query_text, documents, top_n=5):
    """Find similar documents using TF-IDF similarity.
    
    Args:
        query_text (str): Query text or document content
        documents (list): List of document dictionaries
        top_n (int): Number of results to return
        
    Returns:
        list: Similar documents with similarity scores
    """
    from Code.document_processing.document_similarity import DocumentSimilarityService
    
    similarity_service = DocumentSimilarityService()
    return similarity_service.find_similar_documents(query_text, documents, top_n=top_n)

def store_processed_document(doc_data, storage_manager):
    """Store processed document data in Weaviate"""
    try:
        # If doc_data is None, it means no processing was needed (unchanged document)
        if doc_data is None:
            return True
            
        # Extract document metadata
        metadata = doc_data.get("metadata", {})
        document_type = doc_data.get("document_type", "other")
        
        # IMPROVED CONTENT EXTRACTION - try multiple possible locations
        content = metadata.get("text", "")
        if not content:
            content = doc_data.get("content", "")
        if not content and "metadata" in doc_data:
            content = doc_data["metadata"].get("content", "")
        
        # Extract keywords from multiple possible locations with TF-IDF enhancement
        keywords = []
        
        # First try to get TF-IDF enhanced keywords from summary.main_topics
        if "summary" in doc_data and "main_topics" in doc_data["summary"]:
            keywords = doc_data["summary"]["main_topics"]
            logger.info(f"Using TF-IDF enhanced keywords: {keywords[:5]}")
        # Fall back to old keyword extraction method if no TF-IDF keywords
        elif "entities" in doc_data and "keywords" in doc_data["entities"]:
            keywords = doc_data["entities"]["keywords"]
            logger.info(f"Using entity-based keywords: {keywords[:5]}")
        
        # Extract crypto-specific data if available
        crypto_data = doc_data.get("crypto_specific", {})
        crypto_entities = []
        if crypto_data:
            if "blockchain_protocols" in crypto_data:
                crypto_entities.extend(crypto_data["blockchain_protocols"])
            if "token_standards" in crypto_data:
                crypto_entities.extend(crypto_data["token_standards"])
        
        # Calculate better risk score
        risk_score = 50  # Default neutral value
        if "risk_indicators" in doc_data and "risk_indicators" in doc_data["risk_indicators"]:
            # More risk indicators = higher risk score
            risk_indicators = doc_data["risk_indicators"]["risk_indicators"]
            risk_score += len(risk_indicators) * 5  # Each risk indicator adds 5 points
            
            # Cap at 100
            risk_score = min(100, risk_score)
        
        # Extract entities from doc_data
        entities = doc_data.get("entities", {})

        # Prepare document for storage
        document = {
            "content": content,  # Enhanced extraction
            "title": metadata.get("source", "Untitled Document"),
            "document_type": document_type,
            "source": metadata.get("source", "unknown"),
            "word_count": metadata.get("word_count", 0),
            "sentence_count": metadata.get("sentence_count", 0),
            "keywords": keywords,  # Now includes TF-IDF keywords if available
            "org_entities": entities.get("organizations", []),
            "person_entities": entities.get("persons", []),
            "location_entities": entities.get("locations", []),
            "crypto_entities": crypto_entities,
            "extracted_risk_score": risk_score,
            "extracted_date": metadata.get("document_date", datetime.now().isoformat())
        }
        
        # Add compliance data if available
        if "compliance_data" in doc_data:
            compliance = doc_data["compliance_data"]
            if "compliance_level" in compliance:
                document["compliance_level"] = compliance["compliance_level"]
            if "regulatory_requirements" in compliance:
                document["regulatory_requirements"] = compliance["regulatory_requirements"]
        
        # Add debug logging to verify document structure before storage
        logger.debug(f"Storing document with title: {document['title']}")
        logger.debug(f"Document keywords: {document['keywords']}")
        
        # Store document in Weaviate
        success = storage_manager.store_due_diligence_document(document)
        return success
    except Exception as e:
        logger.error(f"Error storing processed document: {e}")
        logger.error(traceback.format_exc())
        return False