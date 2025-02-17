import os
import json
import re
import pytesseract
import numpy as np
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from Loaders.database import create_document_data_table, insert_document_data
from transformers import pipeline

# Provide the full path to the specific file
SAMPLE_FILE_PATH = r"C:\Users\Nessrine\Downloads\Document sans titre.pdf"

# Load Hugging Face token from the local cache
HF_TOKEN = "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"  # Use the token you authenticated with

llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

def extract_text_from_pdf(pdf_path):
    """Extract text from both digital PDFs and scanned PDFs using OCR if needed."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])

    # If text extraction fails, use OCR
    if len(text.strip()) < 10:
        images = convert_from_path(pdf_path)
        text = "\n".join(pytesseract.image_to_string(np.array(img)) for img in images)

    return text.strip()

def extract_structured_data_llm(raw_text):
    """Use an LLM to extract structured contract details in a JSON format."""
    
    prompt = f"""
    Extract structured details from the following contract text like detect names or states location contact and search in the text near these words to find the corresponding informations :

    {raw_text[:3000]}  # Truncate input for efficiency

    Return JSON with these fields:
    {{
        "buyer_name": "John",
        "buyer_last_name": "Doe",
        "buyer_company": "ABC Corp",
        "seller_name": "Alice",
        "seller_last_name": "Smith",
        "seller_company": "Crypto Ltd",
        "agreement": "Cryptocurrency Purchase Agreement",
        "date": "2024-02-10"
    }}
    """

    response = llm_pipeline(prompt, max_length=1024, truncation=True)

    try:
        structured_data = json.loads(response[0]["generated_text"])  # Convert LLM output to JSON
    except json.JSONDecodeError:
        structured_data = {"error": "Failed to parse JSON", "raw_output": response[0]["generated_text"]}

    return structured_data



def process_document(pdf_path):
    """Extract structured details dynamically and store them in the database."""
    extracted_text = extract_text_from_pdf(pdf_path)
    structured_data = extract_structured_data_llm(extracted_text)
    create_document_data_table()
    insert_document_data(structured_data)
    print(f"✅ Successfully Processed: {pdf_path}")

if __name__ == "__main__":
    process_document(SAMPLE_FILE_PATH)
