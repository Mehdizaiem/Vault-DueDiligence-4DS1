import os
import glob
from docx import Document
from pypdf import PdfReader
from dotenv import load_dotenv
from vector_store.store import store_document

# Load environment variables
load_dotenv()

# Get DATA_PATH, with a sensible default
DATA_PATH = os.getenv("DATA_PATH", os.path.join("raw_documents"))

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_pdf(file_path):
    """Read PDF files using pypdf instead of PyPDF2"""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ''  # Handles None returns
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def load_documents():
    """Loads all documents from the specified folder."""
    # Create absolute path if relative path is given
    abs_data_path = DATA_PATH if os.path.isabs(DATA_PATH) else os.path.abspath(DATA_PATH)
    
    # Check if directory exists
    if not os.path.isdir(abs_data_path):
        print(f"WARNING: Directory not found: {abs_data_path}")
        print(f"Current working directory: {os.getcwd()}")
        return []
    
    # Get all files in the directory
    file_paths = glob.glob(os.path.join(abs_data_path, "*"))
    print(f"Found {len(file_paths)} files in {abs_data_path}")
    
    documents = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.docx'):
                content = read_docx(file_path)
            elif file_path.endswith('.pdf'):
                content = read_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            documents.append({
                "filename": os.path.basename(file_path),
                "content": content
            })
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return documents

def ingest_documents(client):
    """Load documents and store them in Weaviate."""
    documents = load_documents()
    print(f"Ingesting {len(documents)} documents into Weaviate...")

    for doc in documents:
        try:
            store_document(client, doc["content"], doc["filename"])
            print(f"Stored document: {doc['filename']}")
        except Exception as e:
            print(f"Error storing document {doc['filename']}: {e}")

    print(f"Successfully ingested {len(documents)} documents.")

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    
    # Uncomment to ingest documents to Weaviate
    #ingest_documents()