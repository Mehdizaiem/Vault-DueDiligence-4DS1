"""
Due Diligence Coordinator Module

This module coordinates the entire due diligence process by orchestrating
the data retrieval, analysis, and synthesis components.
"""

import os
import logging
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DueDiligenceCoordinator:
    """
    Orchestrates the entire due diligence process from document analysis to report generation.
    
    This class coordinates all the components of the system:
    - Document analysis to extract key fund information
    - Data retrieval from various collections
    - Market and on-chain analysis
    - Risk assessment
    - Report generation
    """
    
    def __init__(self, storage_manager=None, document_analyzer=None, data_retriever=None):
        """
        Initialize the coordinator with necessary components.
        
        Args:
            storage_manager: StorageManager instance for data access
            document_analyzer: DocumentAnalyzer instance for document processing
            data_retriever: DataRetriever instance for collection data
        """
        from analysis.document_analyzer import DocumentAnalyzer
        from data.retriever import DataRetriever
        from analysis.market_analyzer import MarketAnalyzer
        from analysis.onchain_analyzer import OnChainAnalyzer
        from analysis.risk_analyzer import RiskAnalyzer
        from analysis.compliance_analyzer import ComplianceAnalyzer
        
        # Initialize storage manager if not provided
        if storage_manager is None:
            from Sample_Data.vector_store.storage_manager import StorageManager
            self.storage = StorageManager()
            self.storage.connect()
        else:
            self.storage = storage_manager
            
        # Initialize document analyzer if not provided
        self.document_analyzer = document_analyzer or DocumentAnalyzer()
        
        # Initialize data retriever if not provided
        self.data_retriever = data_retriever or DataRetriever(self.storage)
        
        # Initialize analyzers
        self.market_analyzer = MarketAnalyzer(self.data_retriever)
        self.onchain_analyzer = OnChainAnalyzer(self.data_retriever)
        self.risk_analyzer = RiskAnalyzer(self.data_retriever)
        self.compliance_analyzer = ComplianceAnalyzer(self.data_retriever)
        
        logger.info("Due Diligence Coordinator initialized")
    
    def process_document(self, document_id: str) -> Dict[str, Any]:
        """
        Process a document to extract and enrich fund information for due diligence.
        
        Args:
            document_id: ID of the document to process
            
        Returns:
            Dict containing all synthesized due diligence data
        """
        logger.info(f"Starting due diligence process for document: {document_id}")
        
        try:
            # Step 1: Retrieve the document
            document = self._get_document(document_id)
            if not document:
                return {"error": f"Document not found: {document_id}"}
            
            # Log basic document info
            logger.info(f"Processing document: {document.get('title', 'Untitled')} ({document_id})")
            
            # Step 2: Analyze the document to extract fund information
            document_content = document.get('content', '')
            if not document_content:
                return {"error": f"Document has no content: {document_id}"}
            
            # Perform document analysis
            analysis_results = self.document_analyzer.analyze_document(document_content)
            logger.info(f"Document analysis completed with {analysis_results['analysis_confidence']}% confidence")
            
            # Step 3: Process the extracted information
            due_diligence_data = self._process_analysis_results(analysis_results)
            
            # Step 4: Generate unique job ID for this due diligence process
            job_id = str(uuid.uuid4())
            due_diligence_data["job_id"] = job_id
            due_diligence_data["document_id"] = document_id
            due_diligence_data["processed_at"] = datetime.now().isoformat()
            
            # Return the synthesized data
            return due_diligence_data
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to process document: {str(e)}"}
    
    def _get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from storage.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document data or None if not found
        """
        try:
            # Try direct retrieval from UserDocuments collection
            from UserQ_A.DocumentAnalyzer import DocumentAnalyzer as UserDocAnalyzer
            doc_analyzer = UserDocAnalyzer()
            document = doc_analyzer.get_document_by_id(document_id)
            doc_analyzer.close()
            
            if document:
                return document
                
            # Fallback to basic retrieval
            collections = ["UserDocuments", "CryptoDueDiligenceDocuments"]
            for collection_name in collections:
                results = self.storage.retrieve_documents(
                    query="",
                    collection_name=collection_name,
                    limit=1,
                    document_id=document_id
                )
                if results and len(results) > 0:
                    return results[0]
                    
            # Document not found
            logger.error(f"Document not found: {document_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _process_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and enrich the document analysis results.
        
        Args:
            analysis_results: Results from document analysis
            
        Returns:
            Enriched due diligence data
        """
        # Extract key information
        fund_info = analysis_results.get("fund_info", {})
        wallet_data = analysis_results.get("wallet_data", [])
        portfolio_data = analysis_results.get("portfolio_data", {})
        
        # Get crypto entities mentioned (from fund name, portfolio, etc.)
        crypto_entities = self._extract_crypto_entities(fund_info, portfolio_data)
        
        # Get wallet addresses
        wallet_addresses = [wallet.get("address") for wallet in wallet_data]
        
        # Enrich with market data if crypto entities are found
        market_analysis = {}
        if crypto_entities:
            logger.info(f"Enriching with market data for: {', '.join(crypto_entities)}")
            market_analysis = self.market_analyzer.analyze_market_data(crypto_entities)
        
        # Enrich with on-chain data if wallet addresses are found
        onchain_analysis = {}
        if wallet_addresses:
            logger.info(f"Enriching with on-chain data for {len(wallet_addresses)} wallets")
            onchain_analysis = self.onchain_analyzer.analyze_wallets(wallet_addresses)
        
        # Perform risk assessment
        risk_assessment = self.risk_analyzer.assess_risk(
            fund_info=fund_info,
            wallet_data=wallet_data,
            portfolio_data=portfolio_data,
            market_analysis=market_analysis,
            onchain_analysis=onchain_analysis
        )
        
        # Perform compliance analysis
        compliance_analysis = self.compliance_analyzer.analyze_compliance(
            fund_info=fund_info,
            document_analysis=analysis_results,
            risk_assessment=risk_assessment
        )
        
        # Combine all data into a comprehensive due diligence dataset
        due_diligence_data = {
            "fund_info": fund_info,
            "wallet_data": wallet_data,
            "portfolio_data": portfolio_data,
            "market_analysis": market_analysis,
            "onchain_analysis": onchain_analysis,
            "risk_assessment": risk_assessment,
            "compliance_analysis": compliance_analysis,
            "team_data": analysis_results.get("team_data", {}),
            "analysis_confidence": analysis_results.get("analysis_confidence", 0)
        }
        
        return due_diligence_data
    
    def _extract_crypto_entities(self, fund_info: Dict[str, Any], portfolio_data: Dict[str, float]) -> List[str]:
        """
        Extract cryptocurrency entities from fund information and portfolio.
        
        Args:
            fund_info: Fund information
            portfolio_data: Portfolio allocation data
            
        Returns:
            List of cryptocurrency entities
        """
        crypto_entities = set()
        
        # Extract from fund name
        fund_name = fund_info.get("fund_name", "").lower()
        common_cryptos = ["bitcoin", "ethereum", "eth", "btc", "solana", "sol", "cardano", "ada"]
        for crypto in common_cryptos:
            if crypto in fund_name:
                crypto_entities.add(crypto)
        
        # Extract from portfolio
        for asset_name in portfolio_data.keys():
            asset_lower = asset_name.lower()
            
            # Check for direct mentions of cryptocurrencies
            if any(crypto in asset_lower for crypto in common_cryptos):
                for crypto in common_cryptos:
                    if crypto in asset_lower:
                        crypto_entities.add(crypto)
                        break
            
            # Check for key terms that suggest crypto assets
            crypto_keywords = ["token", "coin", "crypto", "staking", "chain"]
            if any(keyword in asset_lower for keyword in crypto_keywords):
                # Try to extract the asset name
                parts = asset_name.split()
                if parts and len(parts[0]) >= 2:  # At least 2 chars to avoid false positives
                    crypto_entities.add(parts[0].lower())
        
        return list(crypto_entities)
        
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'storage') and self.storage:
            self.storage.close()
        logger.info("Due Diligence Coordinator resources cleaned up")