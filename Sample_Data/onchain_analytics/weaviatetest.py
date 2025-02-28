import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directories to path
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
logger.info(f"Parent path: {parent_path}")
sys.path.append(parent_path)

# Try imports
try:
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    logger.info("Successfully imported get_weaviate_client")
    
    # Try to connect
    client = get_weaviate_client()
    if client.is_live():
        logger.info("Successfully connected to Weaviate")
    else:
        logger.error("Failed to connect to Weaviate")
except Exception as e:
    logger.error(f"Error: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())