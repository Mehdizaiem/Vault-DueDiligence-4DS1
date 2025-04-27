# File: score_user_documents.py

import logging
from uuid import uuid4
from datetime import datetime

from storage_manager import StorageManager
from Code.document_processing.document_risk_model import assess_document_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize storage
    storage = StorageManager()
    if not storage.connect():
        logger.error("❌ Failed to connect to Weaviate.")
        return

    client = storage.client
    try:
        documents = client.collections.get("UserDocuments").query.fetch_objects(limit=1000).objects
        logger.info(f"📄 Fetched {len(documents)} user documents.")

        risk_collection = client.collections.get("user_doc_risk")

        for doc in documents:
            content = doc.properties.get("content")
            if not content:
                logger.warning(f"⚠️ Skipping empty document: {doc.uuid}")
                continue

            score, category, keywords = assess_document_risk(content)

            logger.info(f"→ {doc.properties.get('title', 'Untitled')} | Score: {score:.2f} | Category: {category}")

            risk_obj = {
                "document_id": str(doc.uuid),
                "user_id": doc.properties.get("user_id", "unknown"),
                "title": doc.properties.get("title", "Untitled"),
                "upload_date": doc.properties.get("upload_date", str(datetime.utcnow())),
                "risk_score": score,
                "risk_category": category,
                "risk_factors": keywords
            }

            risk_collection.data.insert_many([risk_obj])


        logger.info("✅ Finished processing all documents.")

    except Exception as e:
        logger.error(f"💥 Error during processing: {e}")
    finally:
        storage.close()

if __name__ == "__main__":
    main()
