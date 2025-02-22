import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

WEAVIATE_URL = "https://nc9dgaitpc5mcgrx952a.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "eE90sruF7UQxXi66mcV6bQJCvFeweGLwzkoE"

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in environment variables")

def connect_to_weaviate():
    """Connect to Weaviate Cloud using v4 client."""
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        )
        logger.info("Successfully connected to Weaviate")
        return client
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise

def create_collection(client):
    """Create collection for crypto documents."""
    crypto_collection_config = {
        "properties": {
            "content": {"type": "text"},
            "source_file": {"type": "text"},
            "chunk_id": {"type": "int"},
            "document_type": {"type": "text"},
        },
        "vectorizer": "none"  # No vectorizer since we're providing custom embeddings
    }
    
    if not client.collections.exists("CryptoDocument"):
        client.collections.create("CryptoDocument", crypto_collection_config)
        logger.info("Collection created successfully")
    else:
        logger.info("Collection already exists")

def import_embeddings(client):
    """Import embeddings from processed files."""
    embeddings_dir = Path("../processed/embeddings")
    
    if not embeddings_dir.exists():
        logger.error(f"Error: Directory not found: {embeddings_dir.absolute()}")
        return
    
    files = list(embeddings_dir.glob("*_chunks_with_embeddings.json"))
    if not files:
        logger.warning(f"No embedding files found in {embeddings_dir.absolute()}")
        return

    collection = client.collections.get("CryptoDocument")
    
    for embedding_file in files:
        logger.info(f"Processing: {embedding_file}")
        
        try:
            with open(embedding_file, 'r', encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            logger.error(f"Error reading file {embedding_file}: {e}")
            continue
        
        base_filename = embedding_file.stem.split("_chunks_with_embeddings")[0]
        
        document_type = "other"
        if "Cryptocurrency-Payments" in str(embedding_file):
            document_type = "agreement"
        elif "SOANA" in str(embedding_file) or "CONSIL" in str(embedding_file):
            document_type = "regulation"
        elif "Updated-Guidance" in str(embedding_file):
            document_type = "guidance"

        logger.info(f"Importing {len(chunks)} chunks from {base_filename} (type: {document_type})")
        
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            for j, chunk in enumerate(batch_chunks):
                if "embedding" not in chunk:
                    logger.warning(f"Warning: No embedding found for chunk {i+j}")
                    continue
                
                properties = {
                    "content": chunk["text"],
                    "source_file": base_filename,
                    "chunk_id": i+j,
                    "document_type": document_type
                }
                
                vector = chunk["embedding"]
                
                try:
                    collection.data.insert(
                        properties=properties,
                        vector=vector
                    )
                except Exception as e:
                    logger.error(f"Error adding object {i+j}: {e}")
        
        logger.info(f"Completed import of {base_filename}")

def main():
    try:
        client = connect_to_weaviate()
        create_collection(client)
        import_embeddings(client)
        
        collection = client.collections.get("CryptoDocument")
        count_result = collection.aggregate.over_all(total_count=True)
        logger.info(f"Total objects in database: {count_result.total_count}")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()
