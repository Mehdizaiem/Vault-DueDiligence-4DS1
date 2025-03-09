# Sample_Data/onchain_analytics/models/weaviate_storage.py
import logging
from typing import Dict, Any
from datetime import datetime
import sys
import os

# Add path to find vector_store
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(parent_path)

# Then import using the full path
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
# In weaviate_storage.py
from Sample_Data.vector_store.embed import generate_mpnet_embedding
from .weaviate_schema import create_onchain_schema

logger = logging.getLogger(__name__)

def store_wallet_analysis(wallet_data: Dict[str, Any], related_fund: str = None) -> bool:
    """Store wallet analysis data in Weaviate."""
    client = get_weaviate_client()
    
    try:
        # Ensure schema exists
        create_onchain_schema(client)
        
        # Get collection
        collection = client.collections.get("OnChainAnalytics")
        
        # Extract data
        address = wallet_data.get("address")
        blockchain = wallet_data.get("blockchain", "ethereum")
        analytics = wallet_data.get("analytics", {})
        risk_assessment = wallet_data.get("risk_assessment", {})
        
        # Prepare text for embedding
        description = f"""
        Wallet address {address} on {blockchain} blockchain. 
        This wallet has {analytics.get('transaction_count', 0)} transactions and 
        {analytics.get('token_transaction_count', 0)} token transfers.
        It has interacted with {analytics.get('unique_interactions_count', 0)} unique addresses
        and {analytics.get('unique_contracts_count', 0)} smart contracts.
        The wallet has a risk score of {risk_assessment.get('risk_score', 50)}.
        """
        
        # Generate embedding
        vector = generate_mpnet_embedding(description)
        
        # Convert timestamps to ISO format
        first_tx = analytics.get("first_transaction_timestamp")
        last_tx = analytics.get("latest_transaction_timestamp")
        
        first_date = datetime.fromtimestamp(first_tx).isoformat() if first_tx else None
        last_date = datetime.fromtimestamp(last_tx).isoformat() if last_tx else None
        
        # Prepare token list
        tokens = list(analytics.get("tokens", {}).keys())
        
        # Prepare properties
        properties = {
            "address": address,
            "blockchain": blockchain,
            "entity_type": "wallet",
            "transaction_count": analytics.get("transaction_count", 0),
            "token_transaction_count": analytics.get("token_transaction_count", 0),
            "total_received": analytics.get("total_received_eth", 0),
            "total_sent": analytics.get("total_sent_eth", 0),
            "balance": analytics.get("balance_eth", 0),
            "first_activity": first_date,
            "last_activity": last_date,
            "active_days": int(analytics.get("account_age_days", 0)),
            "unique_interactions": analytics.get("unique_interactions_count", 0),
            "contract_interactions": analytics.get("contract_interaction_count", 0),
            "tokens": tokens,
            "risk_score": risk_assessment.get("risk_score", 50),
            "risk_level": risk_assessment.get("risk_level", "Medium"),
            "risk_factors": risk_assessment.get("risk_factors", []),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Add related fund if provided
        if related_fund:
            properties["related_fund"] = related_fund
        
        # Store in Weaviate
        collection.data.insert(
            properties=properties,
            vector=vector
        )
        
        logger.info(f"Stored wallet analysis for {address} in Weaviate")
        return True
        
    except Exception as e:
        logger.error(f"Error storing wallet analysis: {str(e)}")
        return False
    finally:
        client.close()