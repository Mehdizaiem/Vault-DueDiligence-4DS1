import numpy as np
import psycopg2
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from db_config import connect_db

# Load embedding model and LLM
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

# Load FAISS Index
FAISS_INDEX_PATH = "C:/faiss_index/doc_index"
index = faiss.read_index(FAISS_INDEX_PATH)

def search_similar_chunks(query_text, top_k=3):
    """Retrieve most relevant document chunks using FAISS."""
    query_embedding = embedding_model.encode(query_text).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document_name, chunk_text FROM document_chunks
                WHERE id = ANY(%s);
            """, (indices[0].tolist(),))
            results = cur.fetchall()

    return results

def generate_rag_response(query_text):
    """Use LLM to generate structured responses from retrieved chunks."""
    relevant_chunks = search_similar_chunks(query_text)

    if not relevant_chunks:
        return "No relevant information found."

    context = "\n".join([chunk[1] for chunk in relevant_chunks])

    prompt = f"""
    Based on the following document excerpts:

    {context}

    Answer the query: "{query_text}" in a structured format.
    """

    response = llm_pipeline(prompt, max_length=1024, truncation=True)
    return response[0]["generated_text"]

if __name__ == "__main__":
    query = input("Enter your query: ")
    answer = generate_rag_response(query)
    print("\n📌 Structured Response:\n", answer)
