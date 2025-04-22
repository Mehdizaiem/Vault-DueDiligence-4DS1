"""
Data Integration Module

This module coordinates the integration of various data sources and analysis components
for the crypto fund due diligence system. It manages the flow of data between components
and ensures that all relevant information is retrieved and processed.
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary components
from Sample_Data.vector_store.storage_manager import StorageManager
from data.retriever import DataRetriever
from analysis.document_analyzer import DocumentAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from analysis.onchain_analyzer import OnChainAnalyzer
from analysis.risk_analyzer import RiskAnalyzer
from models.fund import CryptoFund

class DataIntegrator:
    """
    Manages the integration of different data sources and analysis components
    to provide comprehensive due diligence assessments.
    """
    
    def __init__(self):
        """Initialize the data integrator with required components"""
        # Initialize storage and data components
        self.storage = StorageManager()
        self.retriever = DataRetriever(self.storage)
        
        # Initialize analysis components
        self.document_analyzer = DocumentAnalyzer()
        self.market_analyzer = MarketAnalyzer(self.retriever)
        self.onchain_analyzer = OnChainAnalyzer(self.retriever)
        self.risk_analyzer = RiskAnalyzer(self.retriever)
        
        # Connect to storage
        self.storage.connect()
    
    def analyze_fund_document(self, document_id: str) -> Dict[str, Any]:
        """
        Analyze a fund document to extract and integrate all relevant data.
        
        Args:
            document_id: ID of the document to analyze
            
        Returns:
            Dict with integrated analysis results
        """
        logger.info(f"Starting comprehensive analysis of document {document_id}")
        
        # Initialize result structure
        result = {
            "document_id": document_id,
            "analysis_date": datetime.now().isoformat(),
            "status": "in_progress",
            "fund_data": {},
            "market_analysis": {},
            "wallet_analysis": {},
            "risk_assessment": {},
            "additional_context": {},
            "errors": []
        }
        
        try:
            # Step 1: Retrieve the document
            document = self.retriever.get_document_by_id(document_id)
            
            if not document:
                error_msg = f"Document with ID {document_id} not found"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["status"] = "failed"
                return result
            
            # Step 2: Analyze the document content
            document_content = document.get("content", "")
            document_analysis = self.document_analyzer.analyze_document(document_content)
            
            # Store basic document info and analysis
            result["fund_data"] = document_analysis
            result["document_info"] = {
                "title": document.get("title", "Untitled Document"),
                "upload_date": document.get("upload_date", "Unknown"),
                "file_type": document.get("file_type", "Unknown"),
                "file_size": document.get("file_size", 0)
            }
            
            # Step 3: Extract entities for further analysis
            crypto_entities = self.retriever.extract_crypto_entities(document_content)
            ethereum_addresses = self.retriever.extract_ethereum_addresses(document_content)
            
            logger.info(f"Extracted {len(crypto_entities)} crypto entities and {len(ethereum_addresses)} Ethereum addresses")
            
            # Step 4: Analyze market data for extracted crypto entities
            if crypto_entities:
                try:
                    market_analysis = self.market_analyzer.analyze_market_data(crypto_entities)
                    result["market_analysis"] = market_analysis
                except Exception as market_error:
                    error_msg = f"Error analyzing market data: {str(market_error)}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
            
            # Step 5: Analyze on-chain data for extracted Ethereum addresses
            if ethereum_addresses:
                try:
                    wallet_analysis = self.onchain_analyzer.analyze_wallets(ethereum_addresses)
                    result["wallet_analysis"] = wallet_analysis
                except Exception as onchain_error:
                    error_msg = f"Error analyzing on-chain data: {str(onchain_error)}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
            
            # Step 6: Perform risk assessment
            try:
                risk_assessment = self.risk_analyzer.analyze_fund_risks(
                    result["fund_data"],
                    result.get("wallet_analysis", {}),
                    result.get("market_analysis", {})
                )
                result["risk_assessment"] = risk_assessment
            except Exception as risk_error:
                error_msg = f"Error performing risk assessment: {str(risk_error)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
            
            # Step 7: Fetch additional context from other collections
            # Get related regulatory documents
            try:
                fund_name = document_analysis.get("fund_info", {}).get("fund_name", "")
                if fund_name:
                    regulatory_documents = self.retriever.get_regulatory_documents(
                        query=f"crypto fund regulation compliance {fund_name}",
                        limit=3
                    )
                    result["additional_context"]["regulatory_documents"] = regulatory_documents
            except Exception as reg_error:
                logger.error(f"Error fetching regulatory documents: {str(reg_error)}")
            
            # Get news sentiment for relevant crypto assets
            try:
                sentiment_data = {}
                for entity in crypto_entities:
                    sentiment = self.retriever.get_sentiment_analysis(entity)
                    if sentiment:
                        sentiment_data[entity] = sentiment
                
                if sentiment_data:
                    result["additional_context"]["sentiment_data"] = sentiment_data
            except Exception as sentiment_error:
                logger.error(f"Error fetching sentiment data: {str(sentiment_error)}")
            
            # Step 8: Create a CryptoFund model to represent the fund
            try:
                fund_model = CryptoFund.from_analysis_result(result)
                result["fund_model"] = fund_model.to_dict()
            except Exception as model_error:
                error_msg = f"Error creating fund model: {str(model_error)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
            
            # Set status to completed
            result["status"] = "completed" if not result["errors"] else "completed_with_errors"
            logger.info(f"Analysis completed for document {document_id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error during analysis: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            result["status"] = "failed"
            return result
        
    def analyze_fund_by_name(self, fund_name: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a fund by name by searching for relevant documents.
        
        Args:
            fund_name: Name of the fund to analyze
            user_id: Optional user ID to filter documents
            
        Returns:
            Dict with integrated analysis results
        """
        logger.info(f"Starting fund analysis by name: {fund_name}")
        
        # Search for documents related to the fund
        documents = self.retriever.get_user_documents(query=fund_name, user_id=user_id, limit=1)
        
        if not documents:
            logger.warning(f"No documents found for fund name: {fund_name}")
            return {
                "status": "failed",
                "errors": [f"No documents found for fund name: {fund_name}"]
            }
        
        # Take the most relevant document
        document = documents[0]
        document_id = document.get("id")
        
        # Analyze the document
        return self.analyze_fund_document(document_id)
    
    def enrich_with_portfolio_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich the analysis results with detailed portfolio data.
        
        Args:
            result: Initial analysis results
            
        Returns:
            Enriched analysis results
        """
        portfolio_data = result.get("fund_data", {}).get("portfolio_data", {})
        
        if not portfolio_data:
            logger.warning("No portfolio data to enrich")
            return result
        
        # Initialize enriched portfolio data
        enriched_portfolio = {}
        
        # For each asset in the portfolio, fetch additional data
        for asset, allocation in portfolio_data.items():
            # Skip assets with zero or very small allocation
            if allocation < 0.001:
                continue
                
            asset_data = {
                "allocation": allocation,
                "value": 0.0  # Will be calculated if AUM is known
            }
            
            # Try to match asset to known cryptocurrencies
            crypto_entities = self.retriever.extract_crypto_entities(asset)
            if crypto_entities:
                crypto = crypto_entities[0]  # Take the first match
                
                # Get current market data
                market_data = self.retriever.get_market_data(
                    self._convert_crypto_to_symbol(crypto), 
                    limit=1
                )
                
                if market_data and len(market_data) > 0:
                    asset_data["market_data"] = market_data[0]
                
                # Get sentiment data
                sentiment = self.retriever.get_sentiment_analysis(crypto)
                if sentiment:
                    asset_data["sentiment"] = sentiment
            
            # Add to enriched portfolio
            enriched_portfolio[asset] = asset_data
        
        # Calculate values if AUM is known
        fund_aum = result.get("fund_data", {}).get("fund_info", {}).get("aum")
        if fund_aum:
            try:
                fund_aum_float = float(fund_aum)
                for asset, data in enriched_portfolio.items():
                    data["value"] = fund_aum_float * data["allocation"]
            except (ValueError, TypeError):
                logger.warning(f"Could not convert AUM to float: {fund_aum}")
        
        # Update the result
        result["enriched_portfolio"] = enriched_portfolio
        
        return result
    
    def _convert_crypto_to_symbol(self, crypto: str) -> str:
        """
        Convert cryptocurrency name to trading symbol.
        
        Args:
            crypto: Cryptocurrency name
            
        Returns:
            Trading symbol
        """
        name_to_symbol = {
            "bitcoin": "BTCUSDT",
            "ethereum": "ETHUSDT",
            "solana": "SOLUSDT",
            "binance": "BNBUSDT",
            "cardano": "ADAUSDT",
            "ripple": "XRPUSDT",
            "polkadot": "DOTUSDT",
            "dogecoin": "DOGEUSDT",
            "avalanche": "AVAXUSDT",
            "polygon": "MATICUSDT"
        }
        
        return name_to_symbol.get(crypto.lower(), f"{crypto.upper()}USDT")
    
    def close(self):
        """Close connections"""
        if hasattr(self, 'storage') and self.storage:
            self.storage.close()