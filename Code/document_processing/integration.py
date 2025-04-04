import logging
from typing import Dict, Any
import sys
import os
# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global document processor instance
doc_processor = None

def get_document_processor(use_gpu=False):
    """Get or initialize the document processor"""
    global doc_processor
    if doc_processor is None:
        doc_processor = CryptoDocumentProcessor(use_gpu=use_gpu)
        logger.info("Document processor initialized")
    return doc_processor

def store_processed_document(doc_data, storage_manager):
    """Store processed document data in Weaviate"""
    try:
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