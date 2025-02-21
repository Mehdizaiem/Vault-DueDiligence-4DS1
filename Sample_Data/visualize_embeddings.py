import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from sklearn.manifold import TSNE
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv

# Function to get Weaviate client
def get_weaviate_client():
    """Create and return a Weaviate client with improved error handling and connection options"""
    load_dotenv()
    
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
    WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    try:
        client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                url=WEAVIATE_URL,
                grpc_port=WEAVIATE_GRPC_PORT
            ),
            additional_config=AdditionalConfig(
                timeout=Timeout(connect=60, query=120, init=60)
            )
        )
        client.connect()
        if not client.is_live():
            raise ConnectionError("Failed to connect to Weaviate. Check server status.")
        print("Weaviate client is connected.")
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {str(e)}")
        raise

# Function to fetch embeddings
def fetch_all_embeddings(client):
    """Fetch all documents with their embeddings from Weaviate."""
    try:
        collection = client.collections.get("LegalDocument")
        
        # Fetch all objects with vectors
        response = collection.query.fetch_objects(
            include_vector=True,
            limit=1000
        )
        
        documents = []
        for obj in response.objects:
            # Debug: Print the vector type to understand its structure
            vector = obj.vector
            if isinstance(vector, dict):
                # If vector is a dict, assume default vector is under 'default' key or handle accordingly
                vector = vector.get("default", [])
            elif not isinstance(vector, (list, np.ndarray)):
                print(f"Unexpected vector format for {obj.properties.get('source')}: {type(vector)}")
                continue
            
            doc = {
                "content": obj.properties.get("content", ""),
                "source": obj.properties.get("source", "Unknown"),
                "vector": vector
            }
            documents.append(doc)
        
        print(f"Fetched {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"Error fetching embeddings: {str(e)}")
        return []

# Function to reduce dimensions
def reduce_dimensions(vectors):
    """Reduce high-dimensional vectors to 2D using t-SNE."""
    vectors_array = np.array(vectors, dtype=float)  # Ensure float type
    if len(vectors_array) < 2:
        raise ValueError("Need at least 2 vectors for t-SNE reduction.")
    perplexity = min(30, len(vectors) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_vectors = tsne.fit_transform(vectors_array)
    return reduced_vectors

# Function to visualize with Plotly
def visualize_embeddings():
    """Visualize embedded data in a 2D scatter plot with Plotly."""
    client = get_weaviate_client()
    try:
        documents = fetch_all_embeddings(client)
        if not documents:
            print("No data to visualize.")
            return
        
        # Extract vectors and metadata
        vectors = [doc["vector"] for doc in documents]
        sources = [doc["source"] for doc in documents]
        contents = [doc["content"][:50] + "..." if len(doc["content"]) > 50 else doc["content"] for doc in documents]
        
        # Reduce dimensions
        reduced_vectors = reduce_dimensions(vectors)
        
        # Create interactive scatter plot
        fig = px.scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            hover_data={"Source": sources, "Content": contents},
            labels={"x": "Dimension 1", "y": "Dimension 2"},
            title="2D Visualization of LegalDocument Embeddings"
        )
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        fig.update_layout(showlegend=True)
        
        # Show the plot
        fig.show()
        
        # Save to HTML
        fig.write_html("embeddings_visualization.html")
        print("Visualization saved as 'embeddings_visualization.html'.")
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
    finally:
        client.close()
        print("Weaviate client connection closed.")

if __name__ == "__main__":
    visualize_embeddings()