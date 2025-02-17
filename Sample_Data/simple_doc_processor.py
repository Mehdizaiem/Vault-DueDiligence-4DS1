import os
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader

class SimpleDocumentProcessor:
    def __init__(self, raw_dir: str = "Sample_Data/raw"):
        """Initialize the document processor with the raw documents directory."""
        self.raw_dir = Path(raw_dir)

    def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        print(f"\nAttempting to read: {file_path}")
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                return ""
                
            # Check file size
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")
            
            reader = PdfReader(file_path)
            print(f"Number of pages: {len(reader.pages)}")
            
            text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    print(f"Successfully extracted text from page {i+1}")
                except Exception as e:
                    print(f"Error extracting text from page {i+1}: {str(e)}")
            
            if not text.strip():
                print("Warning: No text was extracted from the PDF")
            else:
                print(f"Successfully extracted {len(text)} characters")
            
            return text
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return ""

    def process_directory(self, subdir: str) -> Dict[str, str]:
        """Process all PDFs in a specific subdirectory."""
        dir_path = self.raw_dir / subdir
        print(f"\nSearching in directory: {dir_path}")
        
        documents = {}
        
        # List all PDF files in directory
        pdf_files = list(dir_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for file_path in pdf_files:
            print(f"\nProcessing file: {file_path.name}")
            try:
                text = self.read_pdf(str(file_path))
                if text:
                    documents[file_path.name] = text
                    print(f"Successfully processed: {file_path.name}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                
        return documents

def main():
    """Main function to test the document processor."""
    processor = SimpleDocumentProcessor()
    
    # Process each subdirectory
    subdirs = ['agreements', 'regulations']
    all_documents = {}
    
    print("Starting document processing...")
    for subdir in subdirs:
        print(f"\nProcessing directory: {subdir}")
        documents = processor.process_directory(subdir)
        all_documents.update(documents)
    
    # Print summary of processed documents
    print("\nProcessed Documents Summary:")
    if not all_documents:
        print("No documents were successfully processed!")
    else:
        for doc_name, content in all_documents.items():
            print(f"{doc_name}: {len(content)} characters")
            # Save processed text to file
            output_dir = Path("Sample_Data/processed/text")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{doc_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Saved processed text to: {output_file}")

if __name__ == "__main__":
    main()