# Code/data_processing/onchain_manager.py
import os
import sys
import logging
from typing import Dict, Any, Optional

# Ensure paths are properly set up
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from Sample_Data.onchain_analytics.analyzers.wallet_analyzer import WalletAnalyzer
from Sample_Data.onchain_analytics.models.weaviate_storage import store_wallet_analysis
from Sample_Data.onchain_analytics.models.weaviate_schema import create_onchain_schema
from Sample_Data.vector_store.embed import generate_mpnet_embedding as generate_embedding
logger = logging.getLogger(__name__)

class OnChainManager:
    """Manager for on-chain analytics, interfaces with existing functionality"""
    
    def __init__(self, storage_manager=None):
        """Initialize the on-chain manager"""
        self.storage_manager = storage_manager
        self.wallet_analyzer = WalletAnalyzer()
        
        # Ensure schema exists
        self._setup_schema()
    
    def _setup_schema(self):
        """Set up the OnChainAnalytics schema if it doesn't exist"""
        try:
            success = create_onchain_schema()
            if success:
                logger.info("OnChainAnalytics schema created or verified successfully")
            else:
                logger.error("Failed to create OnChainAnalytics schema")
        except Exception as e:
            logger.error(f"Error setting up OnChainAnalytics schema: {e}")
    
    def analyze_and_store(self, address: str, blockchain: str = "ethereum", 
                         related_fund: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a blockchain address and store the results.
        
        Args:
            address (str): Blockchain address
            blockchain (str): Blockchain name (default: ethereum)
            related_fund (str, optional): Related fund name
            
        Returns:
            Dict: Analysis results
        """
        logger.info(f"Analyzing {blockchain} address: {address}")
        
        try:
            # Use existing wallet analyzer
            analysis = self.wallet_analyzer.analyze_ethereum_wallet(address)
            
            if "error" in analysis:
                logger.warning(f"Analysis error for {address}: {analysis['error']}")
                return analysis
            
            # Store in Weaviate if analysis successful
            success = store_wallet_analysis(analysis, related_fund)
            
            if success:
                logger.info(f"Analysis for {address} stored successfully")
            else:
                logger.warning(f"Failed to store analysis for {address}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {address}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"address": address, "error": str(e)}