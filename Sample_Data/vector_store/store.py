import weaviate
import os
from dotenv import load_dotenv
from vector_store.embed import generate_embedding

def create_schema(client):
    """Create the LegalDocument collection if it doesn't exist"""
    # Check if collection exists
    try:
        client.collections.get("LegalDocument")
        print("Collection 'LegalDocument' already exists")
    except weaviate.exceptions.WeaviateGRPCError:
        # Create collection with HuggingFace vectorizer configuration
        print("Creating 'LegalDocument' collection")
        client.collections.create(
            name="LegalDocument",
            description="Legal and fraud-related documents",
            properties=[
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document text"
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                    "description": "File name"
                }
            ],
            vectorizer_config=None  # We'll provide vectors directly
        )

def store_document(client, text, filename):
    """
    Store a document in Weaviate with its embedding.
    
    Args:
        client (weaviate.WeaviateClient): The active Weaviate client
        text (str): The document text
        filename (str): The source filename
    """
    # Ensure the schema exists
    create_schema(client)
    
    # Get the collection
    collection = client.collections.get("LegalDocument")
    
    # Generate embedding using HuggingFace model
    vector = generate_embedding(text)
    
    # Insert the object
    collection.data.insert(
        properties={
            "content": text,
            "source": filename
        },
        vector=vector
    )

if __name__ == "__main__":
    # Test storing a document
    from vector_store.weaviate_client import get_weaviate_client
    
    client = get_weaviate_client()
    sample_text = "This is a test document about crypto regulations."
    store_document(client, sample_text, "test_doc.txt")
    print("Test document stored successfully")