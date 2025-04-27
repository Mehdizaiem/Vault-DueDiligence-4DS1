# File path: Sample_Data/vector_store/embed.py
import datetime
import os
import numpy as np
import logging
from typing import Optional, Dict, List, Union, Any, Tuple
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
EMBEDDING_SERVICE_URL = "http://localhost:5000"
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

# Just add this line near the top of your file:
EMBEDDING_SERVICE_URL = "http://localhost:5000"

# And replace your existing embedding functions with these:
def generate_mpnet_embedding(text: str) -> List[float]:
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for embedding")
        return [0.0] * MPNET_DIMENSION
    
    try:
        response = requests.post(
            f"{EMBEDDING_SERVICE_URL}/generate_mpnet_embedding",
            json={"text": text},
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Error from embedding service: {response.text}")
            return [0.0] * MPNET_DIMENSION
            
        result = response.json()
        return result["embedding"]
    except Exception as e:
        logger.error(f"Error generating MPNet embedding: {str(e)}")
        return [0.0] * MPNET_DIMENSION

def generate_finbert_embedding(text: str) -> List[float]:
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for FinBERT embedding")
        return [0.0] * FINBERT_DIMENSION
    
    try:
        response = requests.post(
            f"{EMBEDDING_SERVICE_URL}/generate_finbert_embedding",
            json={"text": text},
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Error from embedding service: {response.text}")
            return [0.0] * FINBERT_DIMENSION
            
        result = response.json()
        return result["embedding"]
    except Exception as e:
        logger.error(f"Error generating FinBERT embedding: {str(e)}")
        return [0.0] * FINBERT_DIMENSION

# Remove the initialize_models function and any calls to it

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
    Extract keywords from text with improved implementation.
    
    Args:
        text (str): The text to extract keywords from
        max_keywords (int): Maximum number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    # Skip very short texts
    if not text or len(text) < 50:
        return []
        
    # Common stopwords to filter out
    stopwords = {'the', 'and', 'a', 'to', 'of', 'in', 'that', 'is', 'for', 'it', 'with', 'as', 
                'be', 'on', 'at', 'this', 'by', 'an', 'we', 'our', 'from', 'your', 'their',
                'has', 'have', 'had', 'not', 'but', 'what', 'all', 'were', 'when', 'who',
                'will', 'more', 'if', 'no', 'or', 'about', 'which', 'when', 'would', 'there',
                'can', 'also', 'use'}
                
    # Crypto-specific terms to prioritize
    crypto_terms = {'bitcoin', 'ethereum', 'blockchain', 'crypto', 'token', 'defi', 'nft',
                   'wallet', 'exchange', 'mining', 'node', 'ledger', 'smart contract',
                   'decentralized', 'transaction', 'address', 'key', 'regulation', 'compliance'}
    
    # Normalize text and split into words
    import re
    # Remove punctuation and normalize whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        # Skip very short words and stopwords
        if len(word) <= 3 or word in stopwords:
            continue
            
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Prioritize crypto terms by boosting their counts
    for word in list(word_counts.keys()):
        if word in crypto_terms:
            word_counts[word] *= 2  # Double the count for crypto-specific terms
    
    # Extract bigrams (two-word phrases) - often more meaningful than single words
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in stopwords and words[i+1] not in stopwords:
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)
    
    # Count bigram frequencies
    bigram_counts = {}
    for bigram in bigrams:
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    
    # Prioritize crypto-term bigrams
    for bigram in list(bigram_counts.keys()):
        if any(term in bigram for term in crypto_terms):
            bigram_counts[bigram] *= 2
    
    # Combine single words and bigrams, sorted by frequency
    all_terms = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    bigram_terms = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Prioritize important bigrams over common single words
    keywords = []
    
    # Add top bigrams first (they're often more meaningful)
    for term, _ in bigram_terms[:max_keywords // 2]:
        keywords.append(term)
    
    # Then add single words to fill the list
    for term, _ in all_terms:
        if len(keywords) >= max_keywords:
            break
        if term not in ' '.join(keywords):  # Avoid words that are already part of bigrams
            keywords.append(term)
    
    return keywords[:max_keywords]


def process_document(text, document_type=None):
    """
    Process a document to extract features and generate embedding.
    
    Args:
        text (str): The document text
        document_type (str, optional): Type of document
        
    Returns:
        tuple: (embedding, features) where embedding is the document vector
               and features is a dictionary of extracted features
    """
    # Generate embedding using all-MPNet
    embedding = generate_mpnet_embedding(text)
    
    # Extract features based on document type
    features = {
        "word_count": len(text.split()),
        "sentence_count": len([s for s in text.split('.') if s.strip()]),
        "keywords": extract_keywords(text),
        "risk_score": calculate_risk_score(text),
        "entities": extract_entities(text)
    }
    
    # Add document type specific features
    if document_type == "whitepaper":
        features.update({
            "has_tokenomics": "tokenomics" in text.lower(),
            "tech_score": calculate_tech_score(text),
            "has_roadmap": "roadmap" in text.lower(),
            "mentioned_blockchains": extract_blockchain_mentions(text)
        })
    
    elif document_type == "audit_report":
        features.update({
            "vulnerability_score": calculate_vulnerability_score(text),
            "critical_count": text.lower().count("critical vulnerability"),
            "high_count": text.lower().count("high severity"),
            "medium_count": text.lower().count("medium severity"),
            "low_count": text.lower().count("low severity"),
            "has_recommendations": "recommend" in text.lower()
        })
    
    elif document_type == "regulatory_filing":
        features.update({
            "mentioned_regulatory_bodies": extract_regulatory_bodies(text),
            "legal_score": calculate_legal_score(text),
            "has_penalties": any(word in text.lower() for word in ["penalty", "penalties", "fine", "sanction"])
        })
    
    elif document_type == "due_diligence_report":
        features.update({
            "assessment_score": calculate_assessment_score(text),
            "has_risk_assessment": "risk assessment" in text.lower(),
            "has_recommendations": "recommend" in text.lower()
        })
    
    return embedding, features

def extract_entities(text):
    """
    Extract named entities from text (simplified implementation).
    
    Args:
        text (str): The text to extract entities from
        
    Returns:
        dict: Dictionary of entity types and their values
    """
    # This is a placeholder implementation
    # In a real application, use a proper NER model like spaCy
    
    entities = {
        "ORG": [],
        "PERSON": [],
        "GPE": []  # Geo-Political Entities (locations)
    }
    
    # Simple rule-based approach (not effective but serves as placeholder)
    words = text.split()
    for i, word in enumerate(words):
        if word.startswith("Corp") or word.startswith("Inc") or word.startswith("LLC"):
            if i > 0:
                entities["ORG"].append(words[i-1] + " " + word)
        
        if word in ["Mr.", "Mrs.", "Ms.", "Dr."]:
            if i < len(words) - 1:
                entities["PERSON"].append(word + " " + words[i+1])
    
    # Remove duplicates
    for entity_type in entities:
        entities[entity_type] = list(set(entities[entity_type]))
    
    return entities

def calculate_tech_score(text):
    """
    Calculate a technical depth score for a document (simplified implementation).
    
    Args:
        text (str): The document text
        
    Returns:
        float: Technical depth score (0-100)
    """
    # This is a placeholder implementation
    # In a real application, use a more sophisticated approach
    
    text_lower = text.lower()
    
    tech_terms = [
        "algorithm", "architecture", "blockchain", "consensus", "cryptography", 
        "decentralized", "encryption", "hash", "implementation", "protocol", 
        "scalability", "security", "smart contract", "token", "transaction"
    ]
    
    # Count tech terms
    tech_count = sum(text_lower.count(term) for term in tech_terms)
    
    # Normalize to 0-100 scale
    max_expected = 50  # Arbitrary number for scaling
    tech_score = min(100, (tech_count / max_expected) * 100)
    
    return tech_score

def calculate_vulnerability_score(text):
    """
    Calculate a vulnerability mentions score (simplified implementation).
    
    Args:
        text (str): The document text
        
    Returns:
        float: Vulnerability score (0-100)
    """
    # This is a placeholder implementation
    
    text_lower = text.lower()
    
    vulnerability_terms = [
        "vulnerability", "exploit", "bug", "defect", "flaw", "issue",
        "critical", "high severity", "medium severity", "low severity",
        "security risk", "attack vector"
    ]
    
    # Count vulnerability terms
    vuln_count = sum(text_lower.count(term) for term in vulnerability_terms)
    
    # Normalize to 0-100 scale
    max_expected = 30  # Arbitrary number for scaling
    vuln_score = min(100, (vuln_count / max_expected) * 100)
    
    return vuln_score

def calculate_legal_score(text):
    """
    Calculate a legal terminology score (simplified implementation).
    
    Args:
        text (str): The document text
        
    Returns:
        float: Legal score (0-100)
    """
    # This is a placeholder implementation
    
    text_lower = text.lower()
    
    legal_terms = [
        "pursuant", "hereby", "regulation", "compliance", "jurisdiction",
        "statute", "provision", "aforementioned", "legal", "law",
        "liability", "enforcement", "violation", "clause", "penalty"
    ]
    
    # Count legal terms
    legal_count = sum(text_lower.count(term) for term in legal_terms)
    
    # Normalize to 0-100 scale
    max_expected = 40  # Arbitrary number for scaling
    legal_score = min(100, (legal_count / max_expected) * 100)
    
    return legal_score

def calculate_assessment_score(text):
    """
    Calculate an assessment terminology score (simplified implementation).
    
    Args:
        text (str): The document text
        
    Returns:
        float: Assessment score (0-100)
    """
    # This is a placeholder implementation
    
    text_lower = text.lower()
    
    assessment_terms = [
        "assessment", "evaluation", "analysis", "review", "finding",
        "recommendation", "conclusion", "scorecard", "rating", "grade",
        "performance", "measure", "metric", "quality", "criteria"
    ]
    
    # Count assessment terms
    assessment_count = sum(text_lower.count(term) for term in assessment_terms)
    
    # Normalize to 0-100 scale
    max_expected = 35  # Arbitrary number for scaling
    assessment_score = min(100, (assessment_count / max_expected) * 100)
    
    return assessment_score

def extract_blockchain_mentions(text):
    """
    Extract mentions of blockchain platforms (simplified implementation).
    
    Args:
        text (str): The document text
        
    Returns:
        list: List of mentioned blockchains
    """
    # This is a placeholder implementation
    
    text_lower = text.lower()
    
    blockchains = [
        "bitcoin", "ethereum", "binance", "solana", "cardano",
        "ripple", "polkadot", "avalanche", "cosmos", "polygon"
    ]
    
    # Find mentioned blockchains
    mentioned = [chain for chain in blockchains if chain in text_lower]
    
    return mentioned

def extract_regulatory_bodies(text):
    """
    Extract mentions of regulatory bodies (simplified implementation).
    
    Args:
        text (str): The document text
        
    Returns:
        list: List of mentioned regulatory bodies
    """
    # This is a placeholder implementation
    
    text_lower = text.lower()
    
    bodies = [
        "sec", "securities and exchange commission",
        "finra", "financial industry regulatory authority",
        "cftc", "commodity futures trading commission",
        "fca", "financial conduct authority",
        "mifid", "fsb", "financial stability board",
        "fatf", "financial action task force"
    ]
    
    # Find mentioned regulatory bodies
    mentioned = []
    for body in bodies:
        if body in text_lower:
            # Add the shorter version (like SEC instead of securities and exchange commission)
            if body in ["sec", "finra", "cftc", "fca", "mifid", "fsb", "fatf"]:
                mentioned.append(body.upper())
            else:
                # Find the acronym for longer names
                words = body.split()
                if len(words) > 1:
                    acronym = "".join(word[0] for word in words if word not in ["and", "of"])
                    mentioned.append(acronym.upper())
    
    return list(set(mentioned))  # Remove duplicates
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