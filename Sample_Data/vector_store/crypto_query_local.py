import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Local Weaviate connection details
WEAVIATE_URL = "http://localhost:8080"  # Local Weaviate instance

def connect_to_weaviate():
    """Connect to local Weaviate instance using v4 client."""
    try:
        client = weaviate.Client(
            url=WEAVIATE_URL,
        )
        logger.info("Successfully connected to local Weaviate instance")
        return client
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise

# Rest of your code remains the same...
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

def keyword_search(client, query, limit=5):
    """Search for documents using BM25 keyword search."""
    try:
        collection = client.collections.get("CryptoDocument")
        results = collection.query.bm25(
            query=query,
            properties=["content"],
            limit=limit
        ).with_all_attributes().do()
        
        return results.objects
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []

def semantic_search(client, query, embedding_model, limit=5):
    """Search for relevant documents using vector search."""
    try:
        # Generate query embedding
        query_vector = embedding_model.get_embedding(query)
        
        # Get collection
        collection = client.collections.get("CryptoDocument")
        
        # Perform vector search
        results = collection.query.near_vector(
            vector=query_vector,
            limit=limit
        ).with_additional(["distance"]).with_all_attributes().do()
        
        return results.objects
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def generate_simple_answer(query, docs):
    """Generate a simple answer based on retrieved documents without using LLM."""
    if not docs:
        return "No relevant documents found."
    
    # Extract relevant sections
    answer_parts = []
    sources = []
    
    for i, doc in enumerate(docs):
        props = doc.properties
        content = props['content']
        source = props['source_file']
        
        # Simple extractive approach: find sentences containing query terms
        query_terms = set(query.lower().split())
        sentences = content.split('. ')
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_terms = set(sentence.lower().split())
            # Check if any query term is in the sentence
            if query_terms.intersection(sentence_terms):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            answer_parts.append('. '.join(relevant_sentences))
            sources.append(source)
    
    if answer_parts:
        answer = "\n\n".join(answer_parts)
        source_info = "\n\nInformation based on: " + ", ".join(sources)
        return answer + source_info
    else:
        # If no specific sentences match, return the most relevant document
        return f"Based on the documents, here's the most relevant information:\n\n{docs[0].properties['content']}\n\nSource: {docs[0].properties['source_file']}"

def explore_database(client):
    """Display statistics about the database."""
    try:
        collection = client.collections.get("CryptoDocument")
        
        # Get total count
        count_result = collection.aggregate.over_all(total_count=True)
        total_count = count_result.total_count
        print(f"\nTotal documents in database: {total_count}")
        
        # Get document types
        doc_types = collection.aggregate.group_by(
            properties=["document_type"],
            objects_per_group=1
        ).with_additional("group_by_count").do()
        
        print("\nDocument types:")
        for group in doc_types.groups:
            doc_type = group.group_by["document_type"]
            count = group.additional["group_by_count"]
            print(f"- {doc_type}: {count} documents")
        
        # Get source files
        sources = collection.aggregate.group_by(
            properties=["source_file"],
            objects_per_group=1
        ).with_additional("group_by_count").do()
        
        print("\nSource files:")
        for group in sources.groups:
            source = group.group_by["source_file"]
            count = group.additional["group_by_count"]
            print(f"- {source}: {count} chunks")
            
    except Exception as e:
        logger.error(f"Error exploring database: {e}")

def main():
    """Main function with interactive query loop."""
    client = None
    embedding_model = None
    
    try:
        # Connect to Weaviate
        client = connect_to_weaviate()
        
        # Initialize embedding model
        embedding_model = LocalEmbeddingModel()
        
        print("\n=== Crypto Due Diligence Assistant ===")
        print("Commands:")
        print("  'stats' - Show database statistics")
        print("  'kw:your query' - Keyword search")
        print("  'vs:your query' - Vector search")
        print("  'exit' - Exit program")
        print("\nDefault search: Keyword search")
        
        while True:
            query = input("\nEnter query: ").strip()
            
            if query.lower() == 'exit':
                break
                
            if query.lower() == 'stats':
                explore_database(client)
                continue
                
            # Determine search type
            search_type = "keyword" # Default to keyword search since it's faster
            if query.lower().startswith("kw:"):
                search_type = "keyword"
                query = query[3:].strip()
            elif query.lower().startswith("vs:"):
                search_type = "vector"
                query = query[3:].strip()
                
            if not query:
                print("Please enter a valid query")
                continue
                
            # Perform search
            print(f"Performing {search_type} search for: '{query}'")
            if search_type == "keyword":
                results = keyword_search(client, query)
            else:
                results = semantic_search(client, query, embedding_model)
                
            if not results:
                print("No results found")
                continue
                
            # Generate answer
            answer = generate_simple_answer(query, results)
            
            # Display results
            print("\n=== Answer ===")
            print(answer)
            
            print("\n=== Sources ===")
            for i, doc in enumerate(results):
                distance = doc.additional.get("distance", 0) if hasattr(doc, "additional") else 0
                
                print(f"{i+1}. {doc.properties['source_file']}")
                print(f"   Type: {doc.properties['document_type']}")
                if hasattr(doc, "additional") and "distance" in doc.additional:
                    print(f"   Relevance: {1 - doc.additional['distance']:.4f}")
                print(f"   Preview: {doc.properties['content'][:100]}...\n")
                
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    main()