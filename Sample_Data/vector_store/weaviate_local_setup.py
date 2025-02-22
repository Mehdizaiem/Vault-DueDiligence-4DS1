import os
import logging
import json
from pathlib import Path
import weaviate
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local Weaviate connection details
WEAVIATE_URL = "http://localhost:8080"

class LocalEmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with a smaller, faster model that works well for semantic search."""
        self.model = None
        self.model_name = model_name
        
    def load_model(self):
        """Load the model only when needed."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
            
    def get_embedding(self, text):
        """Generate embedding for the given text."""
        model = self.load_model()
        embedding = model.encode(text)
        return embedding.tolist()

def connect_to_weaviate():
    """Connect to local Weaviate instance."""
    try:
        client = weaviate.Client(
            url=WEAVIATE_URL,
        )
        logger.info("Successfully connected to local Weaviate instance")
        return client
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise

def create_schema(client):
    """Create the CryptoDocument schema in Weaviate."""
    schema = {
        "classes": [
            {
                "class": "CryptoDocument",
                "description": "A document related to crypto due diligence",
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "description": "The content of the document",
                        "dataType": ["text"]
                    },
                    {
                        "name": "source_file",
                        "description": "Source file name",
                        "dataType": ["string"]
                    },
                    {
                        "name": "document_type",
                        "description": "Type of document",
                        "dataType": ["string"]
                    }
                ]
            }
        ]
    }
    
    # Create schema
    try:
        client.schema.create(schema)
        logger.info("Schema created successfully")
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        raise

def import_data(client, embeddings_dir):
    """Import data from embeddings files into Weaviate."""
    embedding_model = LocalEmbeddingModel()
    
    # Get all embedding files
    embeddings_path = Path(embeddings_dir)
    embedding_files = list(embeddings_path.glob("*_with_embeddings.json"))
    
    logger.info(f"Found {len(embedding_files)} embedding files to import")
    
    total_chunks = 0
    
    for file_path in embedding_files:
        try:
            logger.info(f"Processing {file_path.name}")
            
            # Load embeddings file
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
            
            # Import each chunk
            for chunk in chunks:
                try:
                    # Extract document metadata
                    content = chunk["text"]
                    metadata = chunk.get("metadata", {})
                    source_file = metadata.get("source", "unknown")
                    document_type = metadata.get("type", "unknown")
                    
                    # Get embedding
                    embedding = chunk.get("embedding")
                    
                    # If embedding is missing, generate it
                    if not embedding:
                        embedding = embedding_model.get_embedding(content)
                    
                    # Create object in Weaviate
                    client.data_object.create(
                        class_name="CryptoDocument",
                        data_object={
                            "content": content,
                            "source_file": source_file,
                            "document_type": document_type
                        },
                        vector=embedding
                    )
                    
                    total_chunks += 1
                    
                except Exception as e:
                    logger.error(f"Error importing chunk: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {str(e)}")
            continue
    
    logger.info(f"Successfully imported {total_chunks} chunks into Weaviate")

def main():
    """Main function to set up local Weaviate instance."""
    try:
        # Connect to Weaviate
        client = connect_to_weaviate()
        
        # Check if schema already exists
        try:
            exists = client.schema.exists("CryptoDocument")
        except:
            exists = False
        
        if not exists:
            # Create schema
            create_schema(client)
            
            # Import data
            import_data(client, "Sample_Data/processed/embeddings")
            
            logger.info("Local Weaviate setup completed successfully!")
        else:
            logger.info("Schema already exists. Skipping setup.")
        
    except Exception as e:
        logger.error(f"Setup error: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()