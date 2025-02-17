import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the embedding model name from environment variables
# Default to 'sentence-transformers/all-MiniLM-L6-v2' if not specified
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Initialize the model
model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embedding(text):
    """
    Generate an embedding vector for the given text.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector as a list of floats
    """
    return model.encode(text).tolist()

if __name__ == "__main__":
    # Test the embedding generation
    test_text = "This is a test document about cryptocurrency regulations."
    embedding = generate_embedding(test_text)
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First few values: {embedding[:5]}")