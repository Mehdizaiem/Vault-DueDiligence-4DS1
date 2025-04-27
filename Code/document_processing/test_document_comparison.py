# test_document_comparison.py
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Code.document_processing.document_similarity import DocumentSimilarityService

# Initialize service
similarity_service = DocumentSimilarityService()

# Compare two documents (provide file paths as arguments)
if len(sys.argv) >= 3:
    doc1_path = sys.argv[1]
    doc2_path = sys.argv[2]
    
    # Calculate similarity
    similarity = similarity_service.compute_document_similarity(doc1_path, doc2_path)
    
    print(f"Similarity between documents: {similarity:.4f}")
else:
    print("Usage: python test_document_comparison.py <doc1_path> <doc2_path>")
