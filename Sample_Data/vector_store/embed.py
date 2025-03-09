# File path: Sample_Data/vector_store/embed.py
import datetime
import os
import numpy as np
import logging
from typing import Optional, Dict, List, Union, Any, Tuple
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
MPNET_DIMENSION = 768
FINBERT_DIMENSION = 768

# Initialize models
mpnet_model = None
finbert_model = None
finbert_tokenizer = None

def initialize_models():
    """Initialize all embedding models"""
    global mpnet_model, finbert_model, finbert_tokenizer
    
    try:
        logger.info(f"Loading all-MPNet model: {MPNET_MODEL_NAME}")
        mpnet_model = SentenceTransformer(MPNET_MODEL_NAME)
        
        logger.info(f"Loading FinBERT model: {FINBERT_MODEL_NAME}")
        finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        finbert_model = AutoModel.from_pretrained(FINBERT_MODEL_NAME)
        
        logger.info("All embedding models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing embedding models: {e}")
        return False

def generate_mpnet_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using all-MPNet.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector as a list of floats
    """
    global mpnet_model
    
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for embedding")
        return [0.0] * MPNET_DIMENSION
    
    if mpnet_model is None:
        initialize_models()
    
    try:
        # Truncate text to avoid potential memory issues
        text = text[:100000]  # Limit to 100K characters
        
        # Generate embeddings
        with torch.no_grad():  # Disable gradient calculation for inference
            embeddings = mpnet_model.encode(text, show_progress_bar=False)
        
        # Convert to list if it's a numpy array
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    except Exception as e:
        logger.error(f"Error generating MPNet embedding: {str(e)}")
        # Return a zero vector with the correct dimension in case of error
        return [0.0] * MPNET_DIMENSION

def generate_finbert_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using FinBERT.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector as a list of floats
    """
    global finbert_model, finbert_tokenizer
    
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for FinBERT embedding")
        return [0.0] * FINBERT_DIMENSION
    
    if finbert_model is None or finbert_tokenizer is None:
        initialize_models()
    
    try:
        # Truncate text if needed
        text = text[:500]  # FinBERT has a more limited context window
        
        # Tokenize and get input tensors
        inputs = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            
        # Use the CLS token embedding as the document representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Normalize the embedding
        embedding = cls_embedding[0]
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
            
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating FinBERT embedding: {str(e)}")
        return [0.0] * FINBERT_DIMENSION

def process_document_for_due_diligence(text: str, document_type: Optional[str] = None) -> Tuple[List[float], Dict[str, Any]]:
    """
    Process a document for the CryptoDueDiligenceDocuments collection.
    
    Args:
        text (str): The document text
        document_type (str, optional): Type of document
        
    Returns:
        tuple: (embedding, features) where embedding is the document vector
               and features is a dictionary of extracted features
    """
    # Generate embedding using all-MPNet
    embedding = generate_mpnet_embedding(text)
    
    # Extract features (simplified here, you might want to use your feature extractor)
    features = {
        "word_count": len(text.split()),
        "keywords": extract_keywords(text),
        "risk_score": calculate_risk_score(text)
    }
    
    return embedding, features

def process_news_for_sentiment(text: str, title: Optional[str] = None) -> Tuple[List[float], Dict[str, Any]]:
    """
    Process a news article for the CryptoNewsSentiment collection.
    
    Args:
        text (str): The article text
        title (str, optional): The article title
        
    Returns:
        tuple: (embedding, sentiment) where embedding is the document vector
               and sentiment is a dictionary with sentiment analysis results
    """
    # Combine title and text for better embedding if title is provided
    if title:
        combined_text = f"{title}\n\n{text}"
    else:
        combined_text = text
    
    # Generate embedding using FinBERT
    embedding = generate_finbert_embedding(combined_text)
    
    # Analyze sentiment using a simplified approach
    # In a real scenario, you should use the FinBERT model for sentiment analysis
    sentiment_score = calculate_sentiment(combined_text)
    if sentiment_score > 0.6:
        sentiment_label = "POSITIVE"
    elif sentiment_score < 0.4:
        sentiment_label = "NEGATIVE"
    else:
        sentiment_label = "NEUTRAL"
    
    sentiment = {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "analyzed_at": datetime.now().isoformat()
    }
    
    return embedding, sentiment

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simplified implementation).
    
    Args:
        text (str): The text to extract keywords from
        max_keywords (int): Maximum number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    # This is a placeholder implementation
    # In a real application, use a proper keyword extraction algorithm
    common_words = set(['the', 'and', 'a', 'to', 'of', 'in', 'that', 'is', 'for', 'it', 'with', 'as', 'be', 'on', 'at'])
    words = text.lower().split()
    # Filter out common words and count occurrences
    word_counts = {}
    for word in words:
        if len(word) > 3 and word not in common_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by count and get top keywords
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    return [keyword for keyword, _ in keywords]

def calculate_risk_score(text: str) -> float:
    """
    Calculate a risk score for a document (simplified implementation).
    
    Args:
        text (str): The document text
        
    Returns:
        float: Risk score (0-100)
    """
    # This is a placeholder implementation
    # In a real application, use a proper risk assessment algorithm
    text_lower = text.lower()
    
    # Risk categories with associated terms
    risk_categories = {
        "regulatory": ["regulatory", "regulation", "compliance", "legal", "jurisdiction", 
                     "restriction", "prohibited", "banned", "illegal", "unauthorized"],
        "financial": ["loss", "bankrupt", "insolvency", "debt", "liability", "expense", 
                    "cost", "inflation", "devaluation", "depreciation"],
        "technical": ["hack", "exploit", "vulnerability", "attack", "breach", "bug", 
                    "glitch", "failure", "malfunction", "error"],
        "operational": ["delay", "failure", "disruption", "interruption", "downtime", 
                      "outage", "maintenance", "discontinue", "cease", "halt"],
        "fraud": ["scam", "fraud", "phishing", "fake", "counterfeit", "impersonation",
                 "ponzi", "pyramid", "mlm", "money laundering"]
    }
    
    # Count risk terms by category
    risk_counts = {category: 0 for category in risk_categories}
    
    for category, terms in risk_categories.items():
        for term in terms:
            risk_counts[category] += text_lower.count(term)
    
    # Calculate overall risk score (weighted)
    weights = {
        "regulatory": 1.2,
        "financial": 1.0,
        "technical": 1.1,
        "operational": 0.9,
        "fraud": 1.3
    }
    
    total_weight = sum(weights.values())
    weighted_sum = sum(risk_counts[cat] * weights[cat] for cat in risk_counts)
    max_possible = 10 * total_weight  # Assuming max 10 mentions per category
    
    # Scale to 0-100
    risk_score = min(100, (weighted_sum / max_possible) * 100)
    
    return risk_score

def calculate_sentiment(text: str) -> float:
    """
    Calculate sentiment score (simplified implementation).
    
    Args:
        text (str): The text to analyze
        
    Returns:
        float: Sentiment score (0-1)
    """
    # This is a placeholder implementation
    # In a real application, use FinBERT for sentiment analysis
    text_lower = text.lower()
    
    # Simple dictionary-based approach
    positive_words = ["bullish", "growth", "profit", "gain", "surge", "rise", "positive", 
                     "promising", "success", "opportunity", "uptrend", "breakthrough", 
                     "milestone", "achievement", "progress", "improve", "advantage"]
    
    negative_words = ["bearish", "crash", "loss", "decline", "fall", "drop", "negative", 
                     "concern", "risk", "threat", "downtrend", "failure", "problem", 
                     "issue", "disaster", "crisis", "danger", "worry"]
    
    # Count positive and negative words
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)
    
    # Calculate sentiment score
    total_count = positive_count + negative_count
    if total_count == 0:
        return 0.5  # Neutral if no sentiment words found
    
    sentiment_score = positive_count / total_count
    return sentiment_score

# Initialize models on module import
initialize_models()

# Example usage
if __name__ == "__main__":
    # Test MPNet embedding
    sample_text = "This is a sample document about cryptocurrency markets. Bitcoin and Ethereum are popular cryptocurrencies."
    embedding = generate_mpnet_embedding(sample_text)
    
    logger.info(f"Generated MPNet embedding with {len(embedding)} dimensions")
    logger.info(f"First few embedding values: {embedding[:5]}")
    
    # Test FinBERT embedding
    news_text = "Bitcoin price surges to new all-time high as institutional adoption increases."
    finbert_embedding = generate_finbert_embedding(news_text)
    
    logger.info(f"Generated FinBERT embedding with {len(finbert_embedding)} dimensions")
    logger.info(f"First few FinBERT embedding values: {finbert_embedding[:5]}")