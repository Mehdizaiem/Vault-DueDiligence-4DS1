import re
import logging
import numpy as np
from typing import Dict, List, Any, Set, Optional
from collections import Counter
import math
from pathlib import Path
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TFIDFProcessor:
    """
    TF-IDF processor for enhancing document keyword extraction and similarity calculations.
    Works alongside the existing document processor without modifying its architecture.
    """
    
    def __init__(self, corpus_path: Optional[str] = None):
        """
        Initialize the TF-IDF processor.
        
        Args:
            corpus_path (str, optional): Path to save/load the document corpus data
        """
        self.document_count = 0
        self.term_document_freq = {}  # Dictionary of term -> number of documents containing term
        self.stopwords = self._load_stopwords()
        self.corpus_path = corpus_path or os.path.join(os.path.dirname(__file__), "tfidf_corpus.pkl")
        
        # Try to load existing corpus data
        self._load_corpus()
        
        logger.info("TF-IDF processor initialized")
    
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for TF-IDF processing"""
        return {
            'the', 'and', 'a', 'to', 'of', 'in', 'that', 'is', 'for', 'it', 'with', 'as', 
            'be', 'on', 'at', 'this', 'by', 'an', 'we', 'our', 'from', 'your', 'their',
            'has', 'have', 'had', 'not', 'but', 'what', 'all', 'were', 'when', 'who',
            'will', 'more', 'if', 'no', 'or', 'about', 'which', 'when', 'would', 'there',
            'can', 'also', 'use', 'any', 'some', 'they', 'than', 'then', 'these', 'such',
            'into', 'out', 'up', 'down', 'only', 'just', 'should', 'now', 'each', 'over',
            'very', 'may', 'one', 'like', 'other', 'how', 'its', 'his', 'her', 'them',
            'him', 'she', 'he', 'been', 'being', 'am', 'are', 'was', 'were', 'so', 'an'
        }
    
    def _load_corpus(self) -> bool:
        """
        Load corpus data from disk if available.
        
        Returns:
            bool: Success status
        """
        if os.path.exists(self.corpus_path):
            try:
                with open(self.corpus_path, 'rb') as f:
                    corpus_data = pickle.load(f)
                    self.document_count = corpus_data.get('document_count', 0)
                    self.term_document_freq = corpus_data.get('term_document_freq', {})
                logger.info(f"Loaded corpus data with {self.document_count} documents and {len(self.term_document_freq)} terms")
                return True
            except Exception as e:
                logger.error(f"Error loading corpus data: {e}")
                return False
        return False
    
    def _save_corpus(self) -> bool:
        """
        Save corpus data to disk.
        
        Returns:
            bool: Success status
        """
        try:
            corpus_data = {
                'document_count': self.document_count,
                'term_document_freq': self.term_document_freq
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.corpus_path)), exist_ok=True)
            
            with open(self.corpus_path, 'wb') as f:
                pickle.dump(corpus_data, f)
            logger.info(f"Saved corpus data with {self.document_count} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving corpus data: {e}")
            return False
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for TF-IDF computation.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of preprocessed tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]
        
        return tokens
    
    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """
        Extract n-grams from a list of tokens.
        
        Args:
            tokens (List[str]): Input tokens
            n (int): Size of n-grams
            
        Returns:
            List[str]: List of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def update_corpus(self, text: str) -> None:
        """
        Update the corpus with a new document.
        
        Args:
            text (str): Document text
        """
        tokens = self.preprocess_text(text)
        
        # Add bigrams
        bigrams = self.extract_ngrams(tokens, 2)
        
        # Combine tokens and bigrams
        all_terms = tokens + bigrams
        
        # Count unique terms in this document
        unique_terms = set(all_terms)
        
        # Update document count and term frequencies
        self.document_count += 1
        
        for term in unique_terms:
            self.term_document_freq[term] = self.term_document_freq.get(term, 0) + 1
        
        # Save corpus data periodically (every 10 documents)
        if self.document_count % 10 == 0:
            self._save_corpus()
    
    def compute_tf(self, text: str) -> Dict[str, float]:
        """
        Compute term frequency for a document.
        
        Args:
            text (str): Document text
            
        Returns:
            Dict[str, float]: Term frequency dictionary
        """
        tokens = self.preprocess_text(text)
        
        # Add bigrams
        bigrams = self.extract_ngrams(tokens, 2)
        
        # Combine tokens and bigrams
        all_terms = tokens + bigrams
        
        # If no terms, return empty dict
        if not all_terms:
            return {}
        
        # Count term occurrences
        term_counts = Counter(all_terms)
        
        # Compute term frequency
        total_terms = len(all_terms)
        tf = {term: count / total_terms for term, count in term_counts.items()}
        
        return tf
    
    def compute_idf(self, term: str) -> float:
        """
        Compute inverse document frequency for a term.
        
        Args:
            term (str): Input term
            
        Returns:
            float: IDF value
        """
        if self.document_count == 0:
            return 0.0
        
        # Get document frequency (add 1 for smoothing)
        doc_freq = self.term_document_freq.get(term, 0) + 1
        
        # Compute IDF
        idf = math.log((self.document_count + 1) / doc_freq)
        
        return idf
    
    def compute_tfidf(self, text: str) -> Dict[str, float]:
        """
        Compute TF-IDF scores for terms in a document.
        
        Args:
            text (str): Document text
            
        Returns:
            Dict[str, float]: TF-IDF scores dictionary
        """
        # Compute term frequency
        tf = self.compute_tf(text)
        
        # Compute TF-IDF scores
        tfidf = {}
        for term, tf_value in tf.items():
            idf = self.compute_idf(term)
            tfidf[term] = tf_value * idf
        
        return tfidf
    
    def extract_keywords(self, text: str, max_keywords: int = 10, crypto_boost: bool = True) -> List[str]:
        """
        Extract keywords from text using TF-IDF.
        
        Args:
            text (str): Document text
            max_keywords (int): Maximum number of keywords to extract
            crypto_boost (bool): Whether to boost crypto-related terms
            
        Returns:
            List[str]: Extracted keywords
        """
        # Skip short texts
        if not text or len(text) < 100:
            return []
        
        # Compute TF-IDF scores
        tfidf_scores = self.compute_tfidf(text)
        
        # If no scores, return empty list
        if not tfidf_scores:
            return []
        
        # Apply domain-specific boosting
        if crypto_boost:
            crypto_terms = {
                'bitcoin', 'ethereum', 'blockchain', 'crypto', 'token', 'defi', 'nft',
                'wallet', 'exchange', 'mining', 'node', 'ledger', 'smart contract',
                'decentralized', 'transaction', 'address', 'key', 'regulation', 'compliance',
                'consensus', 'hash', 'digital asset', 'whitepaper', 'ico', 'protocol'
            }
            
            # Boost crypto-related terms
            for term in list(tfidf_scores.keys()):
                if term.lower() in crypto_terms or any(crypto_term in term.lower() for crypto_term in crypto_terms):
                    tfidf_scores[term] *= 1.5  # 50% boost
        
        # Sort terms by TF-IDF scores
        sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top keywords
        keywords = [term for term, score in sorted_terms[:max_keywords]]
        
        return keywords
    
    def compute_document_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two documents using TF-IDF vectors.
        
        Args:
            text1 (str): First document text
            text2 (str): Second document text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Compute TF-IDF vectors
        tfidf1 = self.compute_tfidf(text1)
        tfidf2 = self.compute_tfidf(text2)
        
        # Get all terms
        all_terms = set(list(tfidf1.keys()) + list(tfidf2.keys()))
        
        # If no common terms, return 0
        if not all_terms:
            return 0.0
        
        # Compute dot product
        dot_product = sum(tfidf1.get(term, 0) * tfidf2.get(term, 0) for term in all_terms)
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(tfidf1.get(term, 0) ** 2 for term in all_terms))
        mag2 = math.sqrt(sum(tfidf2.get(term, 0) ** 2 for term in all_terms))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = dot_product / (mag1 * mag2)
        
        return similarity
    
    def find_similar_documents(self, query_text: str, documents: List[Dict[str, Any]], 
                              text_key: str = 'content', top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar documents to a query text.
        
        Args:
            query_text (str): Query text
            documents (List[Dict]): List of document dictionaries
            text_key (str): Key for accessing document text
            top_n (int): Number of top results to return
            
        Returns:
            List[Dict]: List of similar documents with similarity scores
        """
        # Compute similarities
        similarities = []
        for i, doc in enumerate(documents):
            text = doc.get(text_key, "")
            if text:
                similarity = self.compute_document_similarity(query_text, text)
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        results = []
        for i, similarity in similarities[:top_n]:
            doc = documents[i].copy()
            doc['similarity_score'] = similarity
            results.append(doc)
        
        return results