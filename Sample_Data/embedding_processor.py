import os
from pathlib import Path
import json
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingProcessor:
    def __init__(self, processed_dir: str = "Sample_Data/processed"):
        """Initialize the embedding processor."""
        self.processed_dir = Path(processed_dir)
        self.chunks_dir = self.processed_dir / "chunks"
        self.embeddings_dir = self.processed_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    def load_chunks(self, chunk_file: Path) -> List[Dict]:
        """Load chunks from a JSON file."""
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {chunk_file}: {str(e)}")
            return []

    def process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks."""
        try:
            texts = [chunk["text"] for chunk in chunks]
            print(f"Generating embeddings for {len(texts)} chunks...")
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            print(f"Successfully generated {len(embeddings)} embeddings")
            
            # Add embeddings to chunk data
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
                
            return chunks
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return []

    def save_embeddings(self, processed_chunks: List[Dict], output_file: Path):
        """Save processed chunks with embeddings."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved embeddings to {output_file}")
        except Exception as e:
            print(f"Error saving embeddings to {output_file}: {str(e)}")

    def process_all_documents(self):
        """Process all chunk files and generate embeddings."""
        print("\nStarting embedding generation process...")
        
        chunk_files = list(self.chunks_dir.glob("*.json"))
        print(f"Found {len(chunk_files)} chunk files to process")
        
        for chunk_file in chunk_files:
            print(f"\nProcessing: {chunk_file.name}")
            
            # Load chunks
            chunks = self.load_chunks(chunk_file)
            if not chunks:
                continue
            print(f"Loaded {len(chunks)} chunks")
            
            # Generate embeddings
            processed_chunks = self.process_chunks(chunks)
            if not processed_chunks:
                continue
            print(f"Generated embeddings for {len(processed_chunks)} chunks")
            
            # Save results
            output_file = self.embeddings_dir / f"{chunk_file.stem}_with_embeddings.json"
            self.save_embeddings(processed_chunks, output_file)

def main():
    """Main function to run the embedding processor."""
    processor = EmbeddingProcessor()
    processor.process_all_documents()
    print("\nEmbedding generation completed!")

if __name__ == "__main__":
    main()