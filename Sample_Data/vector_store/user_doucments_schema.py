"""
User Documents Collection Schema for Weaviate
This defines the schema for storing user-uploaded documents in Weaviate
"""
import logging
from weaviate.classes.config import DataType, Configure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_user_documents_schema(client):
    """
    Create the UserDocuments collection if it doesn't exist.
    
    Args:
        client: Weaviate client
        
    Returns:
        The created or existing collection
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("UserDocuments")
        logger.info("UserDocuments collection already exists")
        return collection
    except Exception:
        logger.info("Creating UserDocuments collection")
        
        try:
            # Create the collection with all-MPNet vectorizer
            collection = client.collections.create(
                name="UserDocuments",
                description="Collection for user-uploaded due diligence documents",
                vectorizer_config=Configure.Vectorizer.none(),  # We'll provide vectors directly
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=Configure.VectorIndex.Distance.cosine
                ),
                properties=[
                    # Basic document properties
                    {
                        "name": "content",
                        "data_type": DataType.TEXT,
                        "description": "Extracted text from the document"
                    },
                    {
                        "name": "source",
                        "data_type": DataType.TEXT,
                        "description": "Original file name or source"
                    },
                    {
                        "name": "document_type",
                        "data_type": DataType.TEXT,
                        "description": "Type of document"
                    },
                    {
                        "name": "title",
                        "data_type": DataType.TEXT,
                        "description": "Title or name of the document"
                    },
                    {
                        "name": "date",
                        "data_type": DataType.DATE,
                        "description": "Creation or publication date"
                    },
                    # User tracking
                    {
                        "name": "user_id",
                        "data_type": DataType.TEXT,
                        "description": "ID of the user who uploaded the document"
                    },
                    {
                        "name": "upload_date",
                        "data_type": DataType.DATE,
                        "description": "Date when document was uploaded" 
                    },
                    {
                        "name": "is_public",
                        "data_type": DataType.BOOLEAN,
                        "description": "Whether the document is publicly accessible"
                    },
                    # Additional metadata fields
                    {
                        "name": "author_issuer",
                        "data_type": DataType.TEXT,
                        "description": "Author, issuer, or organization responsible"
                    },
                    {
                        "name": "category",
                        "data_type": DataType.TEXT,
                        "description": "Category (e.g., technical, legal, compliance, business)"
                    },
                    {
                        "name": "risk_score",
                        "data_type": DataType.NUMBER,
                        "description": "Risk assessment score (0-100)"
                    },
                    {
                        "name": "keywords",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Key terms extracted from the document"
                    },
                    # Extracted entities
                    {
                        "name": "org_entities",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Organization entities mentioned in the document"
                    },
                    {
                        "name": "person_entities",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Person entities mentioned in the document"
                    },
                    {
                        "name": "location_entities",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Location entities mentioned in the document"
                    },
                    {
                        "name": "file_size",
                        "data_type": DataType.INT,
                        "description": "File size in bytes"
                    },
                    {
                        "name": "file_type",
                        "data_type": DataType.TEXT,
                        "description": "File type (e.g., PDF, DOCX, TXT)"
                    },
                    {
                        "name": "processing_status",
                        "data_type": DataType.TEXT,
                        "description": "Status of document processing (pending, processing, completed, failed)"
                    },
                    {
                        "name": "notes",
                        "data_type": DataType.TEXT,
                        "description": "User-provided notes about the document"
                    },
                    {
                        "name": "crypto_entities",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Cryptocurrency entities mentioned in the document"
                    },
                    {
                        "name": "risk_factors",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Risk factors identified in the document"
                    }
                ]
            )
            
            logger.info("Successfully created UserDocuments collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create UserDocuments collection: {e}")
            raise