import os
import re
import csv
import json
import logging
from datetime import datetime
import weaviate
from docx import Document
from pypdf import PdfReader
from vector_store.weaviate_client import get_weaviate_client  # ✅ Correct Import Path
from vector_store.embed import generate_embedding
from weaviate.exceptions import WeaviateBaseError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output file names
CSV_OUTPUT = "extracted_features.csv"
JSON_OUTPUT = "extracted_features.json"

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
    filename_lower = filename.lower()
    if any(keyword in filename_lower for keyword in ["regulation", "compliance", "fraud"]):
        return "legal"
    elif "investment" in filename_lower or "fund" in filename_lower:
        return "business"
    elif "audit" in filename_lower:
        return "technical"
    else:
        return "general"

def assess_risk(metadata):
    """Assigns a risk score (0-100) based on document type and content."""
    risk_score = 0

    # High-risk document types
    if metadata["document_type"] in ["regulatory_filing", "audit_report"]:
        risk_score += 40  # Regulatory filings & audits usually indicate compliance risks

    # Look for high-risk keywords
    high_risk_words = ["fraud", "scam", "money laundering", "ponzi", "illegal", "regulatory violation", "SEC investigation"]
    if any(word in metadata["content"].lower() for word in high_risk_words):
        risk_score += 30  # Increase risk score for dangerous words

    # If the document discusses a lawsuit or investigation, add more risk points
    legal_risk_words = ["lawsuit", "charges filed", "court case", "criminal", "legal action"]
    if any(word in metadata["content"].lower() for word in legal_risk_words):
        risk_score += 25

    # Cap risk score at 100
    return min(risk_score, 100)

def extract_metadata(filename, content):
    """Extract metadata such as document type, date, category, and risk score."""
    metadata = {
        "filename": filename,
        "document_type": infer_document_type(filename),
        "date": infer_date_from_filename(filename),
        "category": classify_category(filename),
        "content": content[:500],  # ✅ Store only the first 500 characters for preview
        "source": filename,  # ✅ Ensure source is stored
    }

    # Apply risk assessment before storing metadata
    metadata["risk_score"] = assess_risk(metadata)
    
    return metadata

def read_document(file_path):
    """Extract text from different document types (PDF, DOCX, TXT)."""
    try:
        if file_path.endswith(".docx"):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith(".pdf"):
            text = ""
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        logger.error(f"❌ Error reading {file_path}: {e}")
        return ""

def ingest_documents(client, data_path="raw_documents"):
    """Load documents, extract features, store them in Weaviate, and save metadata to CSV & JSON."""
    files = os.listdir(data_path)
    logger.info(f"📂 Found {len(files)} files. Extracting features...")

    extracted_data = []
    
    with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Document Type", "Date", "Category", "Risk Score", "Content Preview"])

        for filename in files:
            file_path = os.path.join(data_path, filename)
            text = read_document(file_path)
            if not text:
                logger.warning(f"⚠️ Skipping empty document: {filename}")
                continue

            vector = generate_embedding(text)
            metadata = extract_metadata(filename, text)

            # Store in Weaviate
            try:
                collection = client.collections.get("CryptoDueDiligenceDocuments")
                collection.data.insert(
                    properties=metadata,
                    vector=vector
                )
                logger.info(f"✅ Successfully stored '{filename}' in Weaviate")
            except WeaviateBaseError as e:
                logger.error(f"❌ Failed to insert '{filename}' in Weaviate: {str(e)}")

            extracted_data.append(metadata)
            writer.writerow([
                metadata["filename"],
                metadata["document_type"],
                metadata["date"],
                metadata["category"],
                metadata["risk_score"],  # ✅ Added risk score
                metadata["content"]
            ])

    with open(JSON_OUTPUT, "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, indent=4)

    logger.info("✅ Extraction complete! Metadata saved to CSV and JSON.")

if __name__ == "__main__":
    client = get_weaviate_client()  # ✅ Use correct Weaviate import
    ingest_documents(client)
    client.close()
