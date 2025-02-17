import os
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import pdf2image
import faiss
from sentence_transformers import SentenceTransformer
import psycopg2
from db_config import connect_db

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding Dimension:", embedding_model.get_sentence_embedding_dimension())


# FAISS Index (for fast similarity search)
FAISS_INDEX_PATH = r"C:\Users\Nessrine\Vault-DueDiligence-4DS1\Sample_Data\Faiss_index"
# Ensure FAISS directory exists
faiss_dir = os.path.dirname(FAISS_INDEX_PATH)
if not os.path.exists(faiss_dir):
    os.makedirs(faiss_dir)  # Create the directory if it doesn't exist

# Delete existing FAISS index (to prevent corruption issues)
'''if os.path.exists(FAISS_INDEX_PATH):
    os.remove(FAISS_INDEX_PATH)  # Remove old index before writing a new one'''
index = faiss.IndexFlatL2(384)  # Ensure this matches your embedding dimension

# Folder Paths
DOCUMENTS_FOLDER = r"C:\Users\Nessrine\Vault-DueDiligence-4DS1\Sample_Data\documents"


def extract_text_from_pdf(pdf_path):
    """Extract text from PDFs using OCR if needed."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])

    # Use OCR if text extraction fails
    if len(text.strip()) < 10:
        images = pdf2image.convert_from_path(pdf_path)
        text = "\n".join(pytesseract.image_to_string(np.array(img)) for img in images)

    return text.strip()

def chunk_text(text, chunk_size=300):
    """Split text into chunks of a given size (number of words)."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_chunk_data(document_name, chunks):
    """Store chunked text in PostgreSQL, and embeddings in FAISS."""
    conn = connect_db()
    cur = conn.cursor()

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk)

        # Ensure the embedding dimension matches FAISS
        embedding_dim = embedding.shape[0]
        faiss_dim = index.d  # Get FAISS index dimension

        if embedding_dim != faiss_dim:
            print(f"⚠️ Dimension Mismatch: FAISS expects {faiss_dim}, but got {embedding_dim}. Skipping.")
            continue  # Skip this chunk to avoid errors

        # Insert chunk metadata into PostgreSQL
        cur.execute("""
            INSERT INTO document_chunks (document_name, chunk_id, chunk_text)
            VALUES (%s, %s, %s);
        """, (document_name, i, chunk))

        # Add embedding to FAISS Index
        index.add(np.array([embedding], dtype=np.float32))

    conn.commit()
    cur.close()
    conn.close()


def process_documents():
    """Process all PDFs, extract chunks, and store embeddings."""
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            print(f"Processing {filename}...")

            extracted_text = extract_text_from_pdf(file_path)
            chunks = chunk_text(extracted_text)

            store_chunk_data(filename, chunks)

            print(f"✅ Processed and stored chunks for: {filename}")

    # Save FAISS Index
    faiss.write_index(index, FAISS_INDEX_PATH)

if __name__ == "__main__":
    process_documents()
