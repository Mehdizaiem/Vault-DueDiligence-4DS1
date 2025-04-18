import logging
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
from weaviate.collections.classes.config import Property, DataType
import time

logger = logging.getLogger(__name__)

def create_risk_schema(client):
    """Create RiskProfiles schema in Weaviate"""
    try:
        client.collections.create(
            name="RiskProfiles",
            vectorizer_config=None,
            properties=[
                Property(name="fund_id", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.DATE),
                Property(name="regulatory", data_type=DataType.NUMBER),
                Property(name="market", data_type=DataType.NUMBER),
                Property(name="technical", data_type=DataType.NUMBER),
                Property(name="operational", data_type=DataType.NUMBER),
                Property(name="fraud", data_type=DataType.NUMBER),
                Property(name="overall_score", data_type=DataType.NUMBER),
                Property(name="alerts", data_type=DataType.TEXT_ARRAY),
            ]
        )
        logger.info("✅ Created RiskProfiles schema")
    except Exception as e:
        logger.warning(f"Schema exists or failed to create: {e}")

def store_profiles(profiles):
    client = get_weaviate_client()
    create_risk_schema(client)

    try:
        # Retry logic to wait for schema to be available
        for _ in range(5):
            try:
                collection = client.collections.get("RiskProfiles")
                break
            except Exception:
                time.sleep(1)
        else:
            logger.error("❌ Could not find RiskProfiles collection after retries")
            return

        inserted = collection.data.insert_many(profiles)
        if inserted.has_errors:
            logger.error(f"Insertion errors: {inserted.errors}")
        else:
            logger.info(f"✅ Stored {len(profiles)} profiles")
    finally:
        client.close()