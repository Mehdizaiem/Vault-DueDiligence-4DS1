import weaviate
import os
import logging
from dotenv import load_dotenv
from weaviate.connect import ConnectionParams
from weaviate.classes.init import AdditionalConfig, Timeout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_weaviate_client():
    """Create and return a Weaviate v4 client with improved error handling."""
    
    # Load environment variables
    load_dotenv()
    
    # Get Weaviate connection details
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
    WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    try:
        # ✅ Correct way to connect to Weaviate (v4)
        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_url(
                url=WEAVIATE_URL,
                grpc_port=WEAVIATE_GRPC_PORT
            ),
            additional_config=AdditionalConfig(
                timeout=Timeout(connect=60, query=120, init=60)
            ),
            skip_init_checks=True  # ✅ Avoids gRPC health check failures
        )

        # 🔄 Ensure connection is open before returning
        client.connect()

        if not client.is_live():
            raise ConnectionError(f"❌ Weaviate at {WEAVIATE_URL} is not live. Check if it's running.")

        logger.info(f"✅ Connected to Weaviate at {WEAVIATE_URL}.")
        return client

    except Exception as e:
        logger.error(f"❌ Final connection attempt failed: {e}")
        raise ConnectionError("Failed to connect to Weaviate. Ensure it's running and accessible.") from e
