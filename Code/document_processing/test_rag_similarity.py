# test_rag_similarity.py
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import directly from root (not from Code.document_processing)
from agentic_rag import CryptoDueDiligenceSystem

# Initialize the system
system = CryptoDueDiligenceSystem()
system.initialize()

# Find similar documents to a query
query = "Regulatory compliance requirements for cryptocurrency exchanges"
similar_docs = system.find_similar_documents(query, top_n=3)

# Print results
print(f"Query: {query}")
print("\nSimilar documents:")
for i, doc in enumerate(similar_docs, 1):
    print(f"\n{i}. {doc.get('title', 'Untitled')}")
    print(f"   Similarity: {doc.get('similarity_score', 0):.4f}")
    content = doc.get('content', '')
    print(f"   Preview: {content[:150]}...")