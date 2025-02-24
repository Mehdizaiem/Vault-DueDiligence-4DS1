import os
from transformers import BertTokenizer, BertModel
import torch
from dotenv import load_dotenv
from feature_extraction import CryptoFeatureExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the embedding model name from environment variables
# Default to 'bert-base-uncased' if not specified
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bert-base-uncased")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(EMBEDDING_MODEL)
model = BertModel.from_pretrained(EMBEDDING_MODEL)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Initialize the feature extractor
feature_extractor = CryptoFeatureExtractor()

def generate_embedding(text):
    """
    Generate an embedding vector for the given text using BERT.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector as a list of floats
    """
    # Tokenize the text and prepare for model
    inputs = tokenizer(text, 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt")
    
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():  # Disable gradient calculations
        outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the document embedding
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
        
        # Convert to list and move back to CPU if necessary
        if device.type == "cuda":
            embeddings = embeddings.cpu()
            
        return embeddings.numpy().tolist()

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
        tuple: (embedding, features) where embedding is a list of floats
               and features is a dictionary of extracted features
    """
    logger.info(f"Processing document of type: {document_type or 'unknown'}")
    
    # Generate embedding
    embedding = generate_embedding(text)
    
    # Extract features
    features = extract_features(text, document_type)
    
    return embedding, features

if __name__ == "__main__":
    # Test the embedding and feature extraction
    test_text = "This is a test document about cryptocurrency regulations."
    
    embedding, features = process_document(test_text, "regulatory_filing")
    
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First few values: {embedding[:5]}")
    
    print("\nExtracted features:")
    for key, value in features.items():
        print(f"{key}: {value}")