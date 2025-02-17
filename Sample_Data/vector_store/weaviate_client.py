import weaviate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Weaviate connection details
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# Ensure URL has http:// or https:// prefix
if not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"http://{WEAVIATE_URL}"

def get_weaviate_client():
    """Create and return a Weaviate client using v4 syntax"""
    client = weaviate.WeaviateClient(
        connection_params=weaviate.connect.ConnectionParams.from_url(
            url=WEAVIATE_URL,
            grpc_port=WEAVIATE_GRPC_PORT
        )
    )
    return client

if __name__ == "__main__":
    client = get_weaviate_client()
    
    if client.is_ready():
        print("Connected to Weaviate successfully!")
    else:
        raise Exception("Failed to connect to Weaviate.")