import os
import numpy as np
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from feature_extraction import CryptoFeatureExtractor
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the embedding model name from environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# Initialize the model
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Loaded SentenceTransformer model: {EMBEDDING_MODEL}")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    raise

# Initialize the feature extractor with large model for better accuracy
feature_extractor = CryptoFeatureExtractor(model_size="large")

def generate_embedding(text):
    """
    Generate an embedding vector for the given text using SentenceTransformer.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector as a list of floats
    """
    try:
        # Truncate text to avoid potential memory issues
        text = text[:100000]  # Limit to 100K characters (matches feature extractor)
        
        # Generate embeddings
        embeddings = model.encode(text)
        
        # Convert to list if it's a numpy array
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

def generate_embedding_for_long_text(text):
    """
    Generate embeddings for longer documents by chunking and averaging.
    
    Args:
        text (str): The document text
        
    Returns:
        list: The averaged embedding vector as a list of floats
    """
    # Check if the text is short enough for direct embedding
    if len(text.split()) < 400:  # Rough estimate for 512 tokens
        return generate_embedding(text)
    
    # Split text into chunks with overlap
    max_words = 400  # Rough estimate for staying under token limits
    overlap = 50
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    logger.info(f"Split long document into {len(chunks)} chunks")
    
    # Generate embeddings for each chunk
    chunk_embeddings = []
    
    for chunk in chunks:
        embedding = generate_embedding(chunk)
        if embedding:  # Check that we got a valid embedding
            # Convert to numpy array for easier manipulation
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            chunk_embeddings.append(embedding)
    
    # Combine chunk embeddings (average pooling)
    if chunk_embeddings:
        combined_embedding = np.mean(chunk_embeddings, axis=0)
        return combined_embedding.tolist()
    else:
        logger.warning("Failed to generate any valid chunk embeddings")
        return []

def extract_features(text, document_type=None):
    """
    Extract features from text using the CryptoFeatureExtractor.
    
    Args:
        text (str): The text to extract features from
        document_type (str, optional): Type of document
        
    Returns:
        dict: Dictionary of extracted features
    """
    return feature_extractor.extract_features(text, document_type)

def process_document(text, document_type=None):
    """
    Process a document by generating its embedding and extracting features.
    
    Args:
        text (str): The document text
        document_type (str, optional): Type of document
        
    Returns:
        tuple: (embedding, features) where embedding is the document vector
               and features is a dictionary of extracted features
    """
    logger.info(f"Processing document of type: {document_type or 'unknown'}")
    
    # Generate embedding
    embedding = generate_embedding_for_long_text(text)
    
    # Extract features using the same text
    features = extract_features(text, document_type)
    
    return embedding, features

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample document about cryptocurrency markets. Bitcoin and Ethereum are popular cryptocurrencies."
    embedding, features = process_document(sample_text, "news_article")
    
    logger.info(f"Generated embedding with {len(embedding)} dimensions")
    logger.info(f"First few embedding values: {embedding[:5]}")
    logger.info("Extracted features:")
    for key, value in features.items():
        logger.info(f"{key}: {value}")