import os
import re
import csv
from datetime import datetime
import weaviate
from docx import Document
from pypdf import PdfReader
from vector_store.embed import generate_embedding

CSV_OUTPUT = "extracted_features.csv"

def extract_metadata(filename):
    """Extract metadata such as document type, date, and inferred category."""
    metadata = {
        "filename": filename,
        "document_type": infer_document_type(filename),
        "date": infer_date_from_filename(filename),
        "category": classify_category(filename),
    }
    return metadata

def infer_document_type(filename):
    """Infer document type based on filename keywords."""
    filename_lower = filename.lower()
    if "indictment" in filename_lower or "usa_v_" in filename_lower:
        return "regulatory_filing"
    elif "whitepaper" in filename_lower:
        return "whitepaper"
    elif "agreement" in filename_lower or "contract" in filename_lower:
        return "contract"
    elif "audit" in filename_lower:
        return "audit_report"
    else:
        return "general_document"

def infer_date_from_filename(filename):
    """Extracts a date if present in the filename (format YYYY-MM-DD)."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else None

def classify_category(filename):
    """Classify document into predefined categories (legal, compliance, business, etc.)."""
    if any(keyword in filename.lower() for keyword in ["regulation", "compliance", "fraud"]):
        return "legal"
    elif "investment" in filename.lower() or "fund" in filename.lower():
        return "business"
    elif "audit" in filename.lower():
        return "technical"
    else:
        return "general"

def read_document(file_path):
    """Extract text from different document types (PDF, DOCX, TXT)."""
    if file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".pdf"):
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def ingest_documents(client, data_path="raw_documents"):
    """Load documents, extract features, store them in Weaviate, and save metadata to CSV."""
    files = os.listdir(data_path)
    print(f"Found {len(files)} files. Extracting features...")

    # Open CSV file to write metadata
    with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Document Type", "Date", "Category", "Content Preview"])  # Header

        for filename in files:
            file_path = os.path.join(data_path, filename)
            text = read_document(file_path)
            if not text:
                print(f"Skipping empty document: {filename}")
                continue

            # Extract metadata
            metadata = extract_metadata(filename)

            # Generate embedding vector
            vector = generate_embedding(text)

            # Store in Weaviate
            client.collections.get("CryptoDueDiligenceDocuments").data.insert(
                properties={
                    "content": text,
                    "source": filename,
                    "document_type": metadata["document_type"],
                    "date": metadata["date"],
                    "category": metadata["category"],
                },
                vector=vector
            )

            # Save metadata to CSV
            writer.writerow([filename, metadata["document_type"], metadata["date"], metadata["category"], text[:300]])
            print(f"Stored {filename} in Weaviate and saved metadata to CSV.")

    print(f"✅ Extracted features saved to {CSV_OUTPUT}")
