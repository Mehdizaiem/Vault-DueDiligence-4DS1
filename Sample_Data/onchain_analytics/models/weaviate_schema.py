# Sample_Data/onchain_analytics/models/weaviate_schema.py
import logging
import sys
import os

# Add parent directory to path to import weaviate_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from Sample_Data.vector_store.weaviate_client import get_weaviate_client

logger = logging.getLogger(__name__)

def create_onchain_schema(client=None):
    """Create or update OnChainAnalytics collection in Weaviate."""
    close_client = False
    try:
        if client is None:
            client = get_weaviate_client()
            close_client = True
        
        # Check if collection exists
        try:
            client.collections.get("OnChainAnalytics")
            logger.info("Collection 'OnChainAnalytics' already exists")
        except Exception:
            logger.info("Creating 'OnChainAnalytics' collection")
            try:
                client.collections.create(
                    name="OnChainAnalytics",
                    description="Blockchain wallet analytics and transaction data",
                    properties=[
                        {
                            "name": "address",
                            "dataType": ["text"],
                            "description": "Wallet or contract address"
                        },
                        {
                            "name": "blockchain",
                            "dataType": ["text"],
                            "description": "Blockchain network (ethereum, binance, etc.)"
                        },
                        {
                            "name": "entity_type",
                            "dataType": ["text"],
                            "description": "Type of entity (wallet, contract, token)"
                        },
                        {
                            "name": "transaction_count",
                            "dataType": ["int"],
                            "description": "Total number of transactions"
                        },
                        {
                            "name": "token_transaction_count",
                            "dataType": ["int"],
                            "description": "Total number of token transactions"
                        },
                        {
                            "name": "total_received",
                            "dataType": ["number"],
                            "description": "Total value received in native currency"
                        },
                        {
                            "name": "total_sent",
                            "dataType": ["number"],
                            "description": "Total value sent in native currency"
                        },
                        {
                            "name": "balance",
                            "dataType": ["number"],
                            "description": "Current balance in native currency"
                        },
                        {
                            "name": "first_activity",
                            "dataType": ["date"],
                            "description": "Timestamp of first activity"
                        },
                        {
                            "name": "last_activity",
                            "dataType": ["date"],
                            "description": "Timestamp of most recent activity"
                        },
                        {
                            "name": "active_days",
                            "dataType": ["int"],
                            "description": "Number of days between first and last activity"
                        },
                        {
                            "name": "unique_interactions",
                            "dataType": ["int"],
                            "description": "Number of unique addresses interacted with"
                        },
                        {
                            "name": "contract_interactions",
                            "dataType": ["int"],
                            "description": "Number of contract interactions"
                        },
                        {
                            "name": "tokens",
                            "dataType": ["text[]"],
                            "description": "Token symbols held by this address"
                        },
                        {
                            "name": "risk_score",
                            "dataType": ["number"],
                            "description": "Risk assessment score (0-100)"
                        },
                        {
                            "name": "risk_level",
                            "dataType": ["text"],
                            "description": "Risk level category"
                        },
                        {
                            "name": "risk_factors",
                            "dataType": ["text[]"],
                            "description": "Identified risk factors"
                        },
                        {
                            "name": "related_fund",
                            "dataType": ["text"],
                            "description": "Related crypto fund name if applicable"
                        },
                        {
                            "name": "analysis_timestamp",
                            "dataType": ["date"],
                            "description": "When this analysis was performed"
                        }
                    ],
                    vectorizer_config=None  # We'll provide vectors directly
                )
                logger.info("Successfully created 'OnChainAnalytics' collection")
            except Exception as e:
                logger.error(f"Failed to create 'OnChainAnalytics' collection: {str(e)}")
                raise
        
        return True
    except Exception as e:
        logger.error(f"Error setting up OnChainAnalytics schema: {str(e)}")
        return False
    finally:
        if close_client and client:
            client.close()