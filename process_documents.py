import os
import logging
from pathlib import Path
import docx
import pypdf
from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor
from Code.document_processing.integration import store_processed_document
from Sample_Data.vector_store.storage_manager import StorageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_docx(file_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {e}")
        return None

def read_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(file_path, 'rb') as file:  # Open in binary mode
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        return None

def read_text_file(file_path):
    """Read a plain text file with error handling for different encodings."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return None
    
    logger.error(f"Failed to read {file_path} with any of the tried encodings")
    return None

def main():
    # Initialize document processor
    processor = CryptoDocumentProcessor()
    
    # Initialize storage manager
    storage_manager = StorageManager()
    storage_manager.setup_schemas()
    
    # Directory containing documents
    data_dir = Path("Sample_Data/raw_documents")
    
    # Find all document files
    doc_files = []
    for ext in [".txt", ".docx", ".pdf"]:
        doc_files.extend(list(data_dir.glob(f"*{ext}")))
    
    logger.info(f"Found {len(doc_files)} documents to process")
    
    # Process each document
    for doc_file in doc_files:
        try:
            # Read document content based on file type
            text = None
            if doc_file.suffix.lower() == '.pdf':
                text = read_pdf(doc_file)
            elif doc_file.suffix.lower() == '.docx':
                text = read_docx(doc_file)
            elif doc_file.suffix.lower() == '.txt':
                text = read_text_file(doc_file)
            
            if text is None:
                logger.warning(f"⚠️ Could not extract text from {doc_file.name}")
                continue
                
            # Process document
            logger.info(f"Processing {doc_file.name}...")
            result = processor.process_document(text=text, filename=doc_file.name)
            
            # Store result in Weaviate
            success = store_processed_document(result, storage_manager)
            
            if success:
                logger.info(f"✅ Successfully processed: {doc_file.name}")
            else:
                logger.warning(f"⚠️ Failed to store: {doc_file.name}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {doc_file.name}: {e}")
    
    # Export entity graph
    processor.export_graph("entity_relationships.gexf")
    logger.info("Processing complete - check entity_relationships.gexf for visualization")
    
    # Close the storage manager connection
    storage_manager.close()

if __name__ == "__main__":
    main()