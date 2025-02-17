import os
from typing import List, Dict
from pathlib import Path
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkProcessor:
    def __init__(self, processed_dir: str = "Sample_Data/processed"):
        """Initialize the chunk processor."""
        self.processed_dir = Path(processed_dir)
        self.text_dir = self.processed_dir / "text"
        self.chunks_dir = self.processed_dir / "chunks"
        self.metadata_dir = self.processed_dir / "metadata"
        
        # Create directories if they don't exist
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def read_document(self, file_path: Path) -> str:
        """Read a document from the processed text directory."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return ""

    def create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into chunks with metadata."""
        chunks = self.text_splitter.create_documents([text], [metadata])
        return [{"text": chunk.page_content, "metadata": chunk.metadata} 
                for chunk in chunks]

    def process_documents(self):
        """Process all documents in the text directory."""
        print("\nStarting document chunking process...")
        
        for text_file in self.text_dir.glob("*.txt"):
            print(f"\nProcessing: {text_file.name}")
            
            # Read the document
            text = self.read_document(text_file)
            if not text:
                continue
                
            # Create metadata
            metadata = {
                "source": text_file.stem,
                "file_path": str(text_file),
                "document_type": "pdf"  # You can enhance this based on file types
            }
            
            # Create chunks
            chunks = self.create_chunks(text, metadata)
            print(f"Created {len(chunks)} chunks")
            
            # Save chunks
            chunk_file = self.chunks_dir / f"{text_file.stem}_chunks.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"Saved chunks to: {chunk_file}")
            
            # Save metadata
            metadata_file = self.metadata_dir / f"{text_file.stem}_metadata.json"
            metadata["num_chunks"] = len(chunks)
            metadata["chunk_size"] = 1000
            metadata["chunk_overlap"] = 200
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Saved metadata to: {metadata_file}")

def main():
    """Main function to test the chunk processor."""
    processor = ChunkProcessor()
    processor.process_documents()
    print("\nChunking process completed!")

if __name__ == "__main__":
    main()