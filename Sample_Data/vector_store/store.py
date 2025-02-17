import weaviate
import os
from dotenv import load_dotenv
from vector_store.embed import generate_embedding

# Load environment variables
load_dotenv()

# Get Weaviate connection details
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# Ensure URL has http:// or https:// prefix
if not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"http://{WEAVIATE_URL}"

# Initialize Weaviate client with v4 syntax
client = weaviate.WeaviateClient(
    connection_params=weaviate.connect.ConnectionParams.from_url(
        url=WEAVIATE_URL,
        grpc_port=WEAVIATE_GRPC_PORT
    )
)

def create_schema():
    """Create the LegalDocument collection if it doesn't exist"""
    # Check if collection exists
    try:
        client.collections.get("LegalDocument")
        print("Collection 'LegalDocument' already exists")
    except weaviate.exceptions.WeaviateGRPCError:
        # Create collection
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
            vectorizer_config=None  # No auto-vectorization
        )

def store_document(text, filename):
    """
    Store a document in Weaviate with its embedding.
    
    Args:
        text (str): The document text
        filename (str): The source filename
    """
    # Ensure the schema exists
    create_schema()
    
    # Get the collection
    collection = client.collections.get("LegalDocument")
    
    # Generate embedding
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
    sample_text = "This is a test document about crypto regulations."
    store_document(sample_text, "test_doc.txt")
    print("Test document stored successfully")