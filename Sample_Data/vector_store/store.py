import weaviate
from vector_store.embed import generate_embedding
from weaviate.exceptions import WeaviateBaseError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_schema(client):
    """Create the CryptoDueDiligenceDocuments collection if it doesn't exist"""
    try:
        client.collections.get("CryptoDueDiligenceDocuments")
        logger.info("Collection 'CryptoDueDiligenceDocuments' already exists")
    except WeaviateBaseError:
        logger.info("Creating 'CryptoDueDiligenceDocuments' collection")
        try:
            client.collections.create(
                name="CryptoDueDiligenceDocuments",
                description="Collection for all documents related to crypto fund due diligence, including whitepapers, audit reports, regulatory filings, due diligence reports, and project documentation",
                properties=[
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Extracted text from the document"
                    },
                    {
                        "name": "source",
                        "dataType": ["text"],
                        "description": "Original file name or source"
                    },
                    {
                        "name": "document_type",
                        "dataType": ["text"],
                        "description": "Type of document (e.g., whitepaper, audit_report, regulatory_filing, due_diligence_report, project_documentation)"
                    },
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Title or name of the document"
                    },
                    {
                        "name": "crypto_fund",
                        "dataType": ["text"],
                        "description": "Associated crypto fund or asset (optional)"
                    },
                    {
                        "name": "date",
                        "dataType": ["date"],
                        "description": "Creation, publication, or issuance date"
                    },
                    {
                        "name": "author_issuer",
                        "dataType": ["text"],
                        "description": "Author, issuer, or organization responsible"
                    },
                    {
                        "name": "category",
                        "dataType": ["text"],
                        "description": "Broad category (e.g., technical, legal, compliance, business)"
                    },
                    {
                        "name": "status",
                        "dataType": ["text"],
                        "description": "Status of the document or report (optional, e.g., completed, in_progress, pending, draft)"
                    },
                    {
                        "name": "risk_score",
                        "dataType": ["number"],
                        "description": "Numerical risk assessment score (0-100, optional)"
                    },
                    {
                        "name": "authority",
                        "dataType": ["text"],
                        "description": "Regulatory body or authority (optional, e.g., SEC, FINRA)"
                    },
                    {
                        "name": "jurisdiction",
                        "dataType": ["text"],
                        "description": "Geographic or legal jurisdiction (optional, e.g., US, EU)"
                    }
                ],
                vectorizer_config=None  # We'll provide vectors directly
            )
            logger.info("Successfully created 'CryptoDueDiligenceDocuments' collection")
        except WeaviateBaseError as e:
            logger.error(f"Failed to create 'CryptoDueDiligenceDocuments' collection: {str(e)}")
            raise

def infer_document_type(filename):
    """Infer document type based on filename patterns"""
    filename_lower = filename.lower()
    if "indictment" in filename_lower or "usa_v_" in filename_lower:
        return "regulatory_filing"
    elif "whitepaper" in filename_lower:
        return "whitepaper"
    elif "agreement" in filename_lower or "contract" in filename_lower:
        return "project_documentation"
    elif "due_diligence" in filename_lower or "due-diligence" in filename_lower:
        return "due_diligence_report"
    elif "audit" in filename_lower:
        return "audit_report"
    else:
        return "project_documentation"  # Default fallback

def store_document(client, text, filename, document_type=None, title=None, date=None, crypto_fund=None, 
                  author_issuer=None, category=None, status=None, risk_score=None, authority=None, 
                  jurisdiction=None):
    """
    Store a document in Weaviate with its embedding.
    
    Args:
        client (weaviate.WeaviateClient): The active Weaviate client
        text (str): The document text
        filename (str): The source filename
        document_type (str, optional): Type of document (will be inferred if not provided)
        title (str, optional): Title or name of the document (will be inferred if not provided)
        date (str, optional): Creation, publication, or issuance date (YYYY-MM-DD format)
        crypto_fund (str, optional): Associated crypto fund or asset
        author_issuer (str, optional): Author, issuer, or organization responsible
        category (str, optional): Broad category (e.g., technical, legal, compliance, business)
        status (str, optional): Status of the document or report
        risk_score (float, optional): Numerical risk assessment score (0-100)
        authority (str, optional): Regulatory body or authority
        jurisdiction (str, optional): Geographic or legal jurisdiction
    """
    # Ensure the schema exists
    create_schema(client)
    
    # Get the collection
    collection = client.collections.get("CryptoDueDiligenceDocuments")
    
    # Generate embedding using HuggingFace model
    vector = generate_embedding(text)
    
    # Infer document_type if not provided
    if not document_type:
        document_type = infer_document_type(filename)
    
    # Use filename as title if not provided, removing extension
    if not title:
        title = filename.rsplit('.', 1)[0]
    
    # Prepare properties with required fields
    properties = {
        "content": text,
        "source": filename,
        "document_type": document_type,
        "title": title
    }
    
    # Add optional properties if provided
    if date:
        properties["date"] = date
    if crypto_fund:
        properties["crypto_fund"] = crypto_fund
    if author_issuer:
        properties["author_issuer"] = author_issuer
    if category:
        properties["category"] = category
    if status:
        properties["status"] = status
    if risk_score is not None:
        properties["risk_score"] = risk_score
    if authority:
        properties["authority"] = authority
    if jurisdiction:
        properties["jurisdiction"] = jurisdiction
    
    try:
        # Insert the object
        collection.data.insert(
            properties=properties,
            vector=vector
        )
        logger.info(f"Document '{title}' stored successfully in 'CryptoDueDiligenceDocuments'")
    except WeaviateBaseError as e:
        logger.error(f"Error storing document '{title}': {str(e)}")
        raise

if __name__ == "__main__":
    from vector_store.weaviate_client import get_weaviate_client
    
    try:
        # Initialize Weaviate client
        client = get_weaviate_client()
        
        # Example: Store a sample document
        sample_text = "UNITED STATES DISTRICT COURT DISTRICT OF OREGON PORTLAND DIVISION UNITED STATES OF AMERICA v. VLADIMIR OKHOTNIKOV, a/k/a 'LADO,' OLENA OBLAMSKA, a/k/a 'LOLA FERRARI,' MIKHAIL SERGEEV, a/k/a 'MIKE MOONEY,' 'GLEB,' 'GLEB MILLION,' and SERGEY MASLAKOV, Defendants. 3:23-cr-57 Indictment 18 U.S.C. §§ 1349, 981..."
        store_document(
            client=client,
            text=sample_text,
            filename="2023-cr-57_indictment.pdf",
            document_type="regulatory_filing",
            title="U.S. v. Okhotnikov Indictment",
            date="2023-02-22",
            author_issuer="U.S. District Court, District of Oregon",
            category="legal",
            authority="U.S. Department of Justice",
            jurisdiction="US"
        )
        
        print("Test document stored successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        client.close()