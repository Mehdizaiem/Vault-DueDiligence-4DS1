# File path: Code/document_processing/document_similarity.py
import logging
from typing import List, Dict, Any, Optional
import os
import json

from Code.document_processing.document_processor import DocumentProcessor
from Code.document_processing.tf_idf_processor import TFIDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentSimilarityService:
    """
    Service for finding similar documents using TF-IDF similarity.
    """
    
    def __init__(self):
        """Initialize the document similarity service."""
        self.tfidf_processor = TFIDFProcessor()
        self.document_processor = DocumentProcessor()
        
        logger.info("Document similarity service initialized")
    
    def find_similar_documents(self, query: str, documents: List[Dict[str, Any]], 
                              content_field: str = "content", 
                              top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar documents to a query.
        
        Args:
            query (str): Query text or document content
            documents (List[Dict]): List of document dictionaries
            content_field (str): Field name containing document content
            top_n (int): Number of top results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        # Process the query if it's a file path
        if os.path.exists(query):
            try:
                with open(query, 'r', encoding='utf-8') as f:
                    query_text = f.read()
            except Exception as e:
                logger.error(f"Error reading query file: {e}")
                query_text = query
        else:
            query_text = query
        
        # Find similar documents
        similar_docs = self.tfidf_processor.find_similar_documents(
            query_text, documents, content_field, top_n
        )
        
        return similar_docs
    
    def compute_document_similarity(self, doc1: str, doc2: str) -> float:
        """
        Compute similarity between two documents.
        
        Args:
            doc1 (str): First document text or file path
            doc2 (str): Second document text or file path
            
        Returns:
            float: Similarity score (0-1)
        """
        # Process documents if they are file paths
        if os.path.exists(doc1):
            try:
                with open(doc1, 'r', encoding='utf-8') as f:
                    text1 = f.read()
            except Exception as e:
                logger.error(f"Error reading document 1: {e}")
                text1 = doc1
        else:
            text1 = doc1
            
        if os.path.exists(doc2):
            try:
                with open(doc2, 'r', encoding='utf-8') as f:
                    text2 = f.read()
            except Exception as e:
                logger.error(f"Error reading document 2: {e}")
                text2 = doc2
        else:
            text2 = doc2
        
        # Compute similarity
        similarity = self.tfidf_processor.compute_document_similarity(text1, text2)
        
        return similarity
    
    def batch_process_documents(self, documents: List[Dict[str, Any]], 
                               content_field: str = "content") -> None:
        """
        Process a batch of documents to update the TF-IDF corpus.
        
        Args:
            documents (List[Dict]): List of document dictionaries
            content_field (str): Field name containing document content
        """
        for doc in documents:
            content = doc.get(content_field, "")
            if content:
                self.tfidf_processor.update_corpus(content)
        
        logger.info(f"Processed {len(documents)} documents for TF-IDF corpus")