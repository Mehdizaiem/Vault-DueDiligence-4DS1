import sys
from pathlib import Path
import os

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parents[2]  # Navigate up to Vault-DueDiligence-4DS1/
sys.path.append(str(project_root))

import argparse
import logging
import json
from docx import Document
import weaviate
from weaviate.connect import ConnectionParams
from crypto_document_processor import CryptoDocumentProcessor
# Direct import without using the project name
# This should work regardless of the folder name
sys.path.append(str(project_root / "Sample_Data" / "vector_store"))
try:
    from storage_manager import StorageManager
except ImportError:
    # Fallback for direct import
    raise ImportError("Could not import StorageManager. Please make sure the Sample_Data/vector_store directory exists and contains storage_manager.py")
from integration import store_processed_document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_pdf(file_path: Path) -> str:
    """
    Read text from a .pdf file.
    
    Args:
        file_path (Path): Path to the .pdf file
        
    Returns:
        str: Extracted text
    """
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            logger.debug(f"Extracted {len(text)} characters from {file_path.name}")
            return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {str(e)}")
        return ""

def read_text_file(file_path: Path) -> str:
    """
    Read text from a .txt file.
    
    Args:
        file_path (Path): Path to the .txt file
        
    Returns:
        str: Extracted text
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            text = f.read()
            logger.debug(f"Extracted {len(text)} characters from {file_path.name}")
            return text
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {str(e)}")
        return ""

def read_docx(file_path: Path) -> str:
    """
    Read text from a .docx file.
    
    Args:
        file_path (Path): Path to the .docx file
        
    Returns:
        str: Extracted text
    """
    try:
        doc = Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        logger.debug(f"Extracted {len(text)} characters from {file_path.name}")
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX {file_path}: {str(e)}")
        return ""

def read_document(file_path: Path) -> str:
    """
    Read text from a document based on its file type.
    
    Args:
        file_path (Path): Path to the file
        
    Returns:
        str: Extracted text
    """
    if file_path.suffix.lower() == '.docx':
        return read_docx(file_path)
    elif file_path.suffix.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_path.suffix.lower() == '.txt':
        return read_text_file(file_path)
    else:
        logger.error(f"Unsupported file type: {file_path}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Process documents for crypto due diligence")
    parser.add_argument("--force-all", action="store_true", help="Force reprocessing of all documents")
    parser.add_argument("--filter", type=str, help="Process only the specified document (filename)")
    args = parser.parse_args()

    # Debug: Print current path and project root
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Sample_Data path: {os.path.join(project_root, 'Sample_Data')}")
    logger.info(f"Sample_Data exists: {os.path.exists(os.path.join(project_root, 'Sample_Data'))}")
    
    sample_data_init = os.path.join(project_root, 'Sample_Data', '__init__.py')
    if not os.path.exists(sample_data_init):
        logger.info(f"Creating Sample_Data/__init__.py")
        os.makedirs(os.path.dirname(sample_data_init), exist_ok=True)
        with open(sample_data_init, 'w') as f:
            f.write("# This file makes Sample_Data a package\n")

    vector_store_init = os.path.join(project_root, 'Sample_Data', 'vector_store', '__init__.py')
    if not os.path.exists(vector_store_init):
        logger.info(f"Creating Sample_Data/vector_store/__init__.py")
        os.makedirs(os.path.dirname(vector_store_init), exist_ok=True)
        with open(vector_store_init, 'w') as f:
            f.write("# This file makes vector_store a package\n")

    # Initialize Weaviate v4 client
    try:
        client = weaviate.connect_to_local(host="localhost", port=9090)
        logger.info("Weaviate client is connected.")
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {str(e)}")
        return

    # Initialize StorageManager for Weaviate storage
    storage_manager = StorageManager()
    try:
        if not storage_manager.connect():
            logger.error("Failed to connect to Weaviate for storage")
            return

        try:
            # Verify Weaviate connection
            if not client.is_ready():
                logger.error("Weaviate server is not ready")
                return

            # Initialize processor
            processor = CryptoDocumentProcessor(use_gpu=False)

            # Document tracker
            tracker_file = Path("Code/document_processing/output/processed_documents.json")
            processed = {}
            if tracker_file.exists() and not args.force_all:
                with tracker_file.open("r") as f:
                    processed = json.load(f)
                logger.info(f"Tracker file exists with {len(processed)} processed documents")
            else:
                logger.info("Tracker file does not exist or force-all enabled, starting fresh")

            # Document directory
            doc_dir = project_root / "Sample_Data" / "raw_documents"
            logger.info(f"Resolved document directory: {doc_dir.resolve()}")
            logger.info(f"Document directory exists: {doc_dir.exists()}")
            
            document_files = []
            for ext in ["*.docx", "*.pdf", "*.txt"]:
                document_files.extend(doc_dir.glob(ext))
            logger.info(f"Found documents: {[f.name for f in document_files]}")
            
            # Filter documents if specified
            if args.filter:
                document_files = [f for f in document_files if f.name.lower() == args.filter.lower()]
                if not document_files:
                    logger.error(f"No document found matching {args.filter}")
                    return
            
            logger.info(f"Processing {len(document_files)} documents")

            # Process each document
            for i, doc_file in enumerate(document_files, 1):
                if not args.force_all and doc_file.name in processed and processed[doc_file.name]["status"] == "success":
                    logger.info(f"Skipping already processed {doc_file.name} ({i}/{len(document_files)})")
                    continue

                logger.info(f"Processing {doc_file.name}... ({i}/{len(document_files)})")

                try:
                    text = read_document(doc_file)
                    if not text:
                        logger.warning(f"No text extracted from {doc_file.name}")
                        processed[doc_file.name] = {"status": "failed", "error": "No text extracted"}
                        continue
                    result = processor.process_document(text=text, filename=doc_file.name)
                    # Store in Weaviate
                    success = store_processed_document(result, storage_manager)
                    if success:
                        logger.info(f"Stored {doc_file.name} in Weaviate")
                    else:
                        logger.error(f"Failed to store {doc_file.name} in Weaviate")
                    processed[doc_file.name] = {"status": "success", "result": result}
                    logger.debug(f"Processed {doc_file.name} successfully")
                except Exception as e:
                    logger.error(f"❌ Error processing {doc_file.name}: {str(e)}")
                    processed[doc_file.name] = {"status": "failed", "error": str(e)}

                try:
                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
                    with tracker_file.open("w") as f:
                        json.dump(processed, f, indent=2)
                    logger.info(f"Successfully saved tracking data to {tracker_file}")
                except Exception as e:
                    logger.error(f"Error saving tracker file: {str(e)}")

                logger.info(f"Remaining documents to process: {len(document_files) - i}")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            storage_manager.close()
            client.close()
            logger.info("Weaviate client connection closed")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        storage_manager.close()
        client.close()
        logger.info("Weaviate client connection closed")

if __name__ == "__main__":
    main()