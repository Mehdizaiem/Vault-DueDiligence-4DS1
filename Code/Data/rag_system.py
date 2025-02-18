import os
import logging
import requests
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import weaviate
import weaviate.classes.config as wc
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#work on this
# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

def test_api_key(url: str, api_key: str) -> bool:
    """Test if the API key is valid by making a direct request."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.get(f"https://{url}/v1/meta", headers=headers)
        logger.info(f"API Test Status Code: {response.status_code}")
        logger.info(f"API Test Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API test failed: {str(e)}")
        return False

class DocumentProcessor:
    def __init__(self):
        # Load environment variables
        load_dotenv(PROJECT_ROOT / '.env.local')
        
        # Get Weaviate configuration
        self.http_host = "x6ap9jbpsrcsjxnugja.c0.europe-west3.gcp.weaviate.cloud"
        self.api_key = os.getenv('WEAVIATE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Missing WEAVIATE_API_KEY in environment variables")
            
        logger.info("Testing API key...")
        if not test_api_key(self.http_host, self.api_key):
            raise ValueError("Invalid API key or connection failed")

        try:
            # Initialize HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
            )
            logger.info("Successfully initialized embeddings model")
            
            # Initialize Weaviate client with v4 configuration
            self.client = weaviate.connect_to_wcs(
                cluster_url=f"https://{self.http_host}",
                auth_credentials=weaviate.auth.AuthApiKey(self.api_key)
            )
            
            logger.info("Successfully connected to Weaviate")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, directory_path: Path) -> List:
        """Load all PDF documents from the specified directory."""
        try:
            loader = DirectoryLoader(
                str(directory_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def process_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

            # --- DEBUGGING PRINTS START ---
            print("\n--- Sample Chunks after Splitting (process_documents) ---")
            for i, chunk in enumerate(chunks[:2]):  # Print content of the first 2 chunks
                print(f"\nChunk {i+1} Content Preview (process_documents): {chunk.page_content[:100]}...")
                print(f"Chunk {i+1} Metadata (process_documents): {chunk.metadata}")
            print("--- DEBUGGING PRINTS END ---\n")
            # --- DEBUGGING PRINTS END ---

            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def store_in_weaviate(self, chunks: List, class_name: str = "Document"):
        """Store document chunks in Weaviate."""
        try:
            # Get existing collections
            existing_collections = self.client.collections.list_all()

            # Create collection if it doesn't exist
            if class_name not in existing_collections:
                self.client.collections.create(
                    name=class_name,
                    properties=[
                        wc.Property(name="content", data_type=wc.DataType.TEXT),
                        wc.Property(name="metadata", data_type=wc.DataType.TEXT),
                        wc.Property(name="chunk_id", data_type=wc.DataType.INT),
                        wc.Property(name="doc_id", data_type=wc.DataType.TEXT),
                    ]
                )
                logger.info(f"Created collection: {class_name}")

            collection = self.client.collections.get(class_name)

            # Configure batch processing using dynamic batch sizing
            with collection.batch.dynamic() as batch:
                # Process documents in batches
                for i, chunk in enumerate(chunks):
                    # Get vector embeddings from HuggingFace
                    vector = self.embeddings.embed_query(chunk.page_content)

                    # Prepare properties
                    properties = {
                        "content": chunk.page_content,
                        "metadata": str(chunk.metadata),
                        "chunk_id": i,
                        "doc_id": str(chunk.metadata.get("source", ""))
                    }

                    # --- DEBUGGING PRINTS START ---
                    if (i + 1) % 100 == 0 or i < 2: # Print for the first 2 chunks and every 100th chunk
                        print("\n--- Debugging Chunk before Weaviate Storage ---")
                        print(f"Chunk {i+1} Content Preview (store_in_weaviate): {chunk.page_content[:100]}...")
                        print(f"Chunk {i+1} Metadata (store_in_weaviate): {chunk.metadata}")
                        print(f"Properties being stored (store_in_weaviate): {properties}")
                        print("--- DEBUGGING PRINTS END ---\n")
                    # --- DEBUGGING PRINTS END ---


                    batch.add_object(
                        properties=properties,
                        vector=vector
                    )

                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1} chunks")

            logger.info(f"Successfully stored {len(chunks)} chunks in Weaviate")
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise

    def query_documents(self, query_text: str, class_name: str = "Document", limit: int = 5):
        """Query documents from Weaviate, now fetching all objects for debugging."""
        try:
            # Get the collection
            collection = self.client.collections.get(class_name)

            # --- MODIFIED QUERY: Fetch all objects without near_vector ---
            response = collection.query.fetch_objects(
                limit=limit,
                return_properties=["content", "metadata", "chunk_id", "doc_id"]
            )
            # --- END MODIFIED QUERY ---


            if not response.objects:
                logger.warning("No results found in Weaviate collection (fetch_objects).")
                return []

            # --- DEBUGGING PRINTS START (query_documents) ---
            print("\n--- Debugging Results from query_documents (fetch_objects) ---")
            print(f"Number of objects retrieved: {len(response.objects)}")
            for i, result in enumerate(response.objects[:2]): # Print info for first 2 results
                print(f"\nResult {i+1} (query_documents - fetch_objects):")
                print(f"Document ID: {result.uuid}")
                print(f"Content Preview (query_documents - fetch_objects): {result.properties.get('content', 'No content available')[:100]}...")
                print(f"Metadata (query_documents - fetch_objects): {result.properties.get('metadata')}")
                print(f"Properties (query_documents - fetch_objects): {result.properties}")
            print("--- DEBUGGING PRINTS END (query_documents) ---\n")
            # --- DEBUGGING PRINTS END ---


            return response.objects
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            raise

    def search_documents(self, query_text: str, class_name: str = "Document", limit: int = 5, threshold: float = 0.0):
        """Extended search with threshold filtering and better result formatting."""
        try:
            results = self.query_documents(query_text, class_name, limit)
            
            formatted_results = []
            for result in results:
                # Skip results below threshold
                if result.metadata.score < threshold:
                    continue
                    
                formatted_results.append({
                    'content': result.properties['content'],
                    'doc_id': result.properties['doc_id'],
                    'chunk_id': result.properties['chunk_id'],
                    'score': result.metadata.score,
                    'metadata': result.properties['metadata']
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise

def main():
    try:
        processor = DocumentProcessor()
        raw_docs_path = PROJECT_ROOT / 'Sample_Data' / 'raw'

        # Load and process documents
        documents = processor.load_documents(raw_docs_path)
        chunks = processor.process_documents(documents)

        # Store in Weaviate
        processor.store_in_weaviate(chunks)

        processor = DocumentProcessor()
        query = "What are the main topics discussed in the documents?"
        search_results = processor.query_documents(query_text=query, limit=5)

        for i, result in enumerate(search_results, 1):
            print(f"\nResult {i}:")
            print(f"Document ID: {result.uuid}")
            print(f"Content: {result.properties.get('content', 'No content available')}")
            print("-" * 50)

        # Print formatted results
        print(f"\nSearch Results for: {query}")
        print("-" * 50)

        for i, result in enumerate(search_results, 1):
            relevance_score = result.metadata.score if result.metadata.score is not None else "N/A"
            print(f"\nResult {i}:")
            print(f"Document: {result.properties['doc_id']}")
            print(f"Chunk ID: {result.properties['chunk_id']}")
            print(f"Relevance Score: {relevance_score}") # Modified line
            print(f"Content Preview: {result.properties['content'][:200]}...")
            print("-" * 50)

            # --- DEBUGGING PRINTS START ---
            print("\n--- Debugging Result Object ---")
            print(f"Type of result: {type(result)}")
            print(f"Result object: {result}")
            print(f"Result.properties: {result.properties}")
            print(f"Result.metadata: {result.metadata}")
            print("--- DEBUGGING PRINTS END ---")


        logger.info("Document processing and search completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if 'processor' in locals():
            processor.client.close()

if __name__ == "__main__":
    main()