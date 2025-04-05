import logging
from pathlib import Path
import docx
import pypdf
import sys
import os
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor
from Code.document_processing.integration import store_processed_document
from Code.document_processing.document_tracker import DocumentTracker
from Sample_Data.vector_store.storage_manager import StorageManager


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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process cryptocurrency documents')
    parser.add_argument('--force-all', action='store_true', 
                        help='Force processing of all documents, ignoring tracking data')
    parser.add_argument('--tracker-file', type=str, default=None,
                        help='Path to the document tracker file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Custom data directory path instead of default')
    args = parser.parse_args()
    
    # Initialize document processor
    processor = CryptoDocumentProcessor()
    
    # Initialize storage manager
    storage_manager = StorageManager()
    storage_manager.setup_schemas()
    
    # Get script directory and set up output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set tracker file location
    if args.tracker_file:
        tracker_file = args.tracker_file
    else:
        tracker_file = os.path.join(output_dir, "processed_documents.json")
    
    # Initialize document tracker with output directory path
    tracker = DocumentTracker(tracker_file=tracker_file)
    logger.info(f"Document tracker file location: {os.path.abspath(tracker.tracker_file)}")
    
    # Directory containing documents
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(project_root) / "Sample_Data" / "raw_documents"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Log to see exactly where the system is looking for documents
    logger.info(f"Looking for documents in: {data_dir.absolute()}")
    
    # Check for document changes
    if args.force_all:
        # If forcing all, get all document files
        doc_files = []
        for ext in [".txt", ".docx", ".pdf"]:
            doc_files.extend(list(data_dir.glob(f"*{ext}")))
        files_to_process = doc_files
        logger.info(f"Force processing all {len(files_to_process)} documents")
    else:
        # Get new and modified files
        new_files, modified_files, unchanged_files = tracker.check_document_changes(data_dir)
        files_to_process = new_files + modified_files
        
        logger.info(f"Found {len(files_to_process)} documents to process "
                   f"({len(new_files)} new, {len(modified_files)} modified)")
        
    if not files_to_process:
        logger.info("No new or changed documents to process.")
        
    # Process documents that need processing
    processed_count = 0
    error_count = 0
    
    for doc_file in files_to_process:
        try:
            logger.info(f"Processing {doc_file.name}...")
            
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
                tracker.update_document_record(doc_file, processed_success=False)
                error_count += 1
                continue
                
            # Process document
            result = processor.process_document(text=text, filename=doc_file.name)
            
            # Store result in Weaviate
            success = store_processed_document(result, storage_manager)
            
            if success:
                logger.info(f"✅ Successfully processed: {doc_file.name}")
                tracker.update_document_record(doc_file, processed_success=True)
                processed_count += 1
            else:
                logger.warning(f"⚠️ Failed to store: {doc_file.name}")
                tracker.update_document_record(doc_file, processed_success=False)
                error_count += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing {doc_file.name}: {e}")
            tracker.update_document_record(doc_file, processed_success=False)
            error_count += 1
    
    # Clean up deleted file records
    deleted_count = tracker.clean_deleted_records(data_dir)
    if deleted_count > 0:
        logger.info(f"Cleaned {deleted_count} records of deleted files")
    
    # Export entity graph
    output_file = os.path.join(output_dir, "entity_relationships.gexf")
    processor.export_graph(output_file)
    
    # Log summary
    logger.info("Processing complete:")
    logger.info(f"  - Total documents: {len(files_to_process)}")
    logger.info(f"  - Successfully processed: {processed_count}")
    logger.info(f"  - Errors: {error_count}")
    logger.info(f"  - Entity relationships exported to: {output_file}")
    
    # Close the storage manager connection
    storage_manager.close()

if __name__ == "__main__":
    main()