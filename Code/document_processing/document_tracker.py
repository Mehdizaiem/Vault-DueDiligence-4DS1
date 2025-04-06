import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentTracker:
    """
    Tracks document processing status to avoid reprocessing unchanged documents.
    
    This class maintains a record of processed documents with their file paths, 
    modification timestamps, and content hashes to determine if reprocessing is needed.
    """
    
    def __init__(self, tracker_file: str = "processed_documents.json"):
        """
        Initialize the document tracker.
        
        Args:
            tracker_file (str): Path to the JSON file that stores document tracking info
        """
        self.tracker_file = tracker_file
        self.document_records = self._load_tracker()
        
    def _load_tracker(self) -> Dict[str, Dict]:
        """
        Load document tracking data from file.
        
        Returns:
            Dict[str, Dict]: Dictionary mapping file paths to document metadata
        """
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading document tracker: {e}")
                return {}
        return {}
    
    def _save_tracker(self) -> bool:
        """
        Save document tracking data to file.
        
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(self.tracker_file))
            os.makedirs(directory, exist_ok=True)
            
            # Save with temporary file approach to avoid corruption
            temp_file = f"{self.tracker_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.document_records, f, indent=2)
            
            # Replace the original file with the temp file
            if os.path.exists(temp_file):
                if os.path.exists(self.tracker_file):
                    os.remove(self.tracker_file)
                os.rename(temp_file, self.tracker_file)
                
            # Verify file was saved
            if os.path.exists(self.tracker_file):
                logger.info(f"Successfully saved tracking data to {self.tracker_file}")
                return True
            else:
                logger.error(f"Failed to save tracking data: file not found after save")
                return False
        except Exception as e:
            logger.error(f"Error saving document tracker: {e}")
            return False
    
    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """
        Compute SHA-256 hash of file contents.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Optional[str]: Hex digest of file hash, or None if error
        """
        try:
            sha256_hash = hashlib.sha256()
            
            with open(file_path, "rb") as f:
                # Read and update hash in chunks for memory efficiency
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
                    
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return None
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def compute_content_hash(self, content):
        """Compute hash of content which may be string or bytes"""
        sha256_hash = hashlib.sha256()
        if isinstance(content, str):
            sha256_hash.update(content.encode('utf-8'))
        else:
            sha256_hash.update(content)  # Assume bytes
        return sha256_hash.hexdigest()
    
    def check_document_changes(self, data_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Check for new, modified, and unchanged documents.
        
        Args:
            data_dir (Path): Directory containing documents
            
        Returns:
            Tuple[List[Path], List[Path], List[Path]]: Lists of new, modified, and unchanged document paths
        """
        # Find all document files
        doc_files = []
        for ext in [".txt", ".docx", ".pdf"]:
            doc_files.extend(list(data_dir.glob(f"*{ext}")))
        
        # Convert paths to strings for dictionary keys
        current_files = {str(doc_file): doc_file for doc_file in doc_files}
        
        # Track new, modified, and unchanged files
        new_files = []
        modified_files = []
        unchanged_files = []
        
        # Check each current file against our records
        for file_path_str, file_path in current_files.items():
            # Get file stats
            file_stats = os.stat(file_path)
            file_mtime = file_stats.st_mtime
            
            if file_path_str not in self.document_records:
                # New file
                new_files.append(file_path)
            else:
                # Existing file - check if modified
                record = self.document_records[file_path_str]
                
                # First check modification time for efficiency
                if file_mtime > record["mtime"]:
                    # Modification time changed, verify with hash
                    current_hash = self._compute_file_hash(file_path_str)
                    
                    if current_hash != record["hash"]:
                        # Content actually changed
                        modified_files.append(file_path)
                    else:
                        # Only metadata changed, content is the same
                        unchanged_files.append(file_path)
                else:
                    # No change in modification time
                    unchanged_files.append(file_path)
        
        # Identify deleted files (in records but not in current files)
        deleted_files = set(self.document_records.keys()) - set(current_files.keys())
        
        # Log summary
        logger.info(f"Document check: {len(new_files)} new, {len(modified_files)} modified, "
                   f"{len(unchanged_files)} unchanged, {len(deleted_files)} deleted")
        
        return new_files, modified_files, unchanged_files
    
    def check_document_content_changed(self, content: str, identifier: str) -> bool:
        """
        Check if document content has changed since last processing.
        
        Args:
            content (str): Document content
            identifier (str): Unique identifier for this document
            
        Returns:
            bool: True if content has changed or is new, False if unchanged
        """
        # Compute hash of content
        if os.path.exists(content):
            # If content is a file path, use file hash
            content_hash = self._compute_file_hash(content)
        else:
            # For content string, use content hash
            content_hash = self.compute_content_hash(content)
        
        # Check if we have processed this document before
        if identifier in self.document_records:
            record = self.document_records[identifier]
            # Check if hash matches (unchanged)
            if record["hash"] == content_hash:
                return False
        
        # Document is new or has changed
        return True
    
    def update_document_record(self, file_path: Path, processed_success: bool = True) -> None:
        """
        Update the tracking record for a processed document.
        
        Args:
            file_path (Path): Path to the document
            processed_success (bool): Whether processing was successful
        """
        file_path_str = os.path.abspath(str(file_path))
        
        try:
            # Get file stats
            file_stats = os.stat(file_path)
            file_mtime = file_stats.st_mtime
            
            # Compute file hash
            file_hash = self._compute_file_hash(file_path_str)
            
            if file_hash:
                # Update record
                self.document_records[file_path_str] = {
                    "filename": file_path.name,
                    "mtime": file_mtime,
                    "hash": file_hash,
                    "last_processed": self._get_current_timestamp(),
                    "processed_success": processed_success
                }
                
                # Save tracker after each update
                self._save_tracker()
                
        except Exception as e:
            logger.error(f"Error updating document record for {file_path}: {e}")
    
    def update_content_record(self, content: str, identifier: str, filename: str, processed_success: bool = True) -> None:
        """
        Update the tracking record for a processed content string.
        
        Args:
            content (str): Document content
            identifier (str): Unique identifier for this document
            filename (str): Original filename
            processed_success (bool): Whether processing was successful
        """
        try:
            # Compute content hash
            content_hash = self.compute_content_hash(content)
            
            # Update record
            self.document_records[identifier] = {
                "filename": filename,
                "mtime": datetime.now().timestamp(),
                "hash": content_hash,
                "last_processed": self._get_current_timestamp(),
                "processed_success": processed_success
            }
            
            # Save tracker after each update
            self._save_tracker()
                
        except Exception as e:
            logger.error(f"Error updating content record for {filename}: {e}")

    def normalize_path(self, path: str) -> str:
        """
        Normalize file paths to ensure consistent tracking across systems.
        
        Args:
            path (str): File path
            
        Returns:
            str: Normalized path
        """
        try:
            # Convert to absolute path and normalize
            abs_path = os.path.abspath(path)
            # Ensure consistent path separators
            normalized = os.path.normpath(abs_path)
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing path {path}: {e}")
            return path
    
    def clean_deleted_records(self, data_dir: Path) -> int:
        """
        Remove records for documents that no longer exist.
        
        Args:
            data_dir (Path): Directory containing documents
            
        Returns:
            int: Number of records cleaned
        """
        # Get current files
        current_files = set()
        for ext in [".txt", ".docx", ".pdf"]:
            for doc_file in data_dir.glob(f"*{ext}"):
                current_files.add(str(doc_file))
        
        # Find deleted files (only considering file paths, not content identifiers)
        deleted_records = set()
        for key in self.document_records.keys():
            # Only consider keys that look like file paths
            if os.path.sep in key and not key.startswith("content_"):
                if key not in current_files:
                    deleted_records.add(key)
        
        # Remove deleted records
        for file_path in deleted_records:
            del self.document_records[file_path]
        
        # Save changes
        if deleted_records:
            self._save_tracker()
        
        return len(deleted_records)