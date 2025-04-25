# Path: processing/coordinator.py
import os
import logging
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import json

# Assuming paths are correctly set up (may need adjustment based on final structure)
# If running main.py, paths should be set there. Direct imports assume correct PYTHONPATH.
from analysis.document_analyzer import DocumentAnalyzer
from data.retriever import DataRetriever
from analysis.market_analyzer import MarketAnalyzer
from analysis.onchain_analyzer import OnChainAnalyzer
from analysis.risk_analyzer import RiskAnalyzer
from analysis.compliance_analyzer import ComplianceAnalyzer
from Sample_Data.vector_store.storage_manager import StorageManager
from UserQ_A.DocumentAnalyzer import DocumentAnalyzer as UserDocAnalyzer # Check if this path/import is correct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

class DueDiligenceCoordinator:
    def __init__(self, storage_manager=None, document_analyzer=None, data_retriever=None):
        # Initialize storage manager if not provided
        if storage_manager is None:
            try:
                self.storage = StorageManager()
                if not self.storage.connect():
                     logger.warning("Coordinator failed to connect to Weaviate on init.")
            except ImportError as e:
                 logger.error(f"Failed to import StorageManager: {e}. Ensure Sample_Data is in Python path.")
                 raise
            except Exception as e:
                 logger.error(f"Failed to initialize StorageManager: {e}", exc_info=True)
                 # Decide if we should raise or continue degraded
                 raise # Raising is safer for now
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
        logger.info(f"Starting due diligence process for document: {document_id}")

        try:
            document = self._get_document(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                return {"error": f"Document not found: {document_id}"}

            logger.info(f"Processing document: {document.get('title', 'Untitled')} ({document_id})")

            document_content = document.get('content', '')
            if not document_content:
                 logger.error(f"Document has no content: {document_id}")
                 return {"error": f"Document has no content: {document_id}"}

            analysis_results = self.document_analyzer.analyze_document(document_content)
            logger.info(f"Document analysis completed with {analysis_results.get('analysis_confidence', 'N/A')}% confidence")

            due_diligence_data = self._process_analysis_results(analysis_results)

            # Add job tracking and metadata
            job_id = str(uuid.uuid4())
            due_diligence_data["job_id"] = job_id
            due_diligence_data["document_id"] = document_id
            due_diligence_data["processed_at"] = datetime.now().isoformat()
            due_diligence_data["document_title"] = document.get('title', 'Untitled')

            logger.info(f"Due diligence process completed for document {document_id} (Job ID: {job_id})")
            return due_diligence_data

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
            return {"error": f"Failed to process document: {str(e)}"}

    def _get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"Attempting to retrieve document with ID: {document_id}")
        document = None
        try:
            # Try UserQ_A first if it exists and is correctly imported
            if 'UserDocAnalyzer' in globals():
                doc_analyzer = UserDocAnalyzer()
                document = doc_analyzer.get_document_by_id(document_id)
                doc_analyzer.close()
                if document:
                    logger.debug(f"Document {document_id} found via UserDocAnalyzer.")
                    return document
                else:
                     logger.debug(f"Document {document_id} not found via UserDocAnalyzer.")
            else:
                 logger.warning("UserDocAnalyzer not available, skipping.")

            # Fallback to storage manager retrieval (assuming retriever has get_document_by_id)
            logger.debug(f"Trying StorageManager via DataRetriever for {document_id}.")
            document = self.data_retriever.get_document_by_id(document_id) # Use the retriever method
            if document:
                logger.debug(f"Document {document_id} found via Retriever.")
                return document

            logger.warning(f"Document {document_id} not found in any searched source.")
            return None

        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}", exc_info=True)
            return None # Return None on error

    def _process_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Processing and enriching document analysis results...")
        # Extract base data from initial analysis
        fund_info = analysis_results.get("fund_info", {})
        wallet_data_raw = analysis_results.get("wallet_data", []) # Raw data from DocumentAnalyzer
        portfolio_data = analysis_results.get("portfolio_data", {})
        compliance_data_raw = analysis_results.get("compliance_data", {})
        team_data = analysis_results.get("team_data", {})

        crypto_entities = self._extract_crypto_entities(fund_info, portfolio_data)
        # Ensure wallet addresses are extracted correctly from wallet_data_raw structure
        wallet_addresses = [w.get("address") for w in wallet_data_raw if isinstance(w, dict) and w.get("address")]

        # --- Enrichment ---
        market_analysis = {}
        if crypto_entities:
            logger.info(f"Enriching with market data for: {', '.join(crypto_entities)}")
            try: market_analysis = self.market_analyzer.analyze_market_data(crypto_entities)
            except Exception as e: logger.error(f"Market Analysis failed: {e}", exc_info=True); market_analysis = {"error": str(e)}

        onchain_analysis = {}
        if wallet_addresses:
            logger.info(f"Enriching with on-chain data for {len(wallet_addresses)} wallets")
            try: onchain_analysis = self.onchain_analyzer.analyze_wallets(wallet_addresses)
            except Exception as e: logger.error(f"OnChain Analysis failed: {e}", exc_info=True); onchain_analysis = {"error": str(e)}

        # --- Risk Analysis ---
        risk_assessment = {}
        logger.info("Starting risk assessment")
        try:
            # Pass all relevant structured data to risk analyzer
            risk_assessment = self.risk_analyzer.analyze_fund_risks(
                fund_data=analysis_results, # Pass the full initial analysis results
                wallet_analysis=onchain_analysis,
                market_analysis=market_analysis
            )
            logger.info("Risk assessment complete")
        except Exception as e: logger.error(f"Risk Analysis failed: {e}", exc_info=True); risk_assessment = {"error": str(e)}

        # --- Compliance Analysis ---
        compliance_analysis_result = {}
        logger.info("Starting compliance analysis")
        try:
             # Pass the original analysis_results which contains compliance_data etc.
             compliance_analysis_result = self.compliance_analyzer.analyze_compliance(
                 fund_data=analysis_results
             )
             logger.info("Compliance analysis complete")
        except Exception as e: logger.error(f"Compliance Analysis failed: {e}", exc_info=True); compliance_analysis_result = {"error": str(e)}

        # --- Structure data for LLM Prompts (Robust Extraction) ---
        risk_factors_raw = risk_assessment.get("risk_factors", [])
        if not isinstance(risk_factors_raw, list): risk_factors_raw = []

        top_risk_factors_list = []
        for f in risk_factors_raw[:3]: # Process top 3 raw factors
            if isinstance(f, dict) and f.get("factor"):
                # Format factor with level for better context
                factor_text = f.get("factor")
                level = f.get("risk_level", 0)
                level_cat = self._map_level_to_category(level) # Helper needed or use RiskAnalyzer's logic
                top_risk_factors_list.append(f"{factor_text} (Risk Level: {level_cat})")
            elif isinstance(f, str): # Fallback for string factors
                 top_risk_factors_list.append(f)
                 logger.warning(f"Found string risk factor '{f}' instead of dict. Using string directly.")

        risk_summary_for_prompt = {
            "overall_risk_score": risk_assessment.get("overall_risk_score"),
            "risk_level": risk_assessment.get("risk_level"),
            "top_risk_factors": top_risk_factors_list
        }
        compliance_summary_for_prompt = {
            "overall_compliance_score": compliance_analysis_result.get("overall_compliance_score"),
            "compliance_level": compliance_analysis_result.get("compliance_level"),
            "top_compliance_gaps": compliance_analysis_result.get("compliance_gaps", [])[:3]
        }
        key_points_for_prompt = { "strengths": [], "concerns": [] } # Actual points generated later

        # --- Final Output Dictionary ---
        due_diligence_data = {
            # Raw/Initial Analysis Sections
            "fund_info": fund_info,
            "wallet_data": wallet_data_raw, # Keep raw data from document analysis
            "portfolio_data": portfolio_data,
            "team_data": team_data,
            "compliance_data_raw": compliance_data_raw, # Keep raw compliance data too
            # Enriched Analysis Sections
            "market_analysis": market_analysis,
            "onchain_analysis": onchain_analysis, # Include full onchain results
            "risk_assessment": risk_assessment, # Include full risk assessment
            "compliance_analysis": compliance_analysis_result, # Include full compliance analysis
            # Metadata
            "analysis_confidence": analysis_results.get("analysis_confidence", 0),
            # Structured Summaries for LLM Prompting
            "llm_prompt_data": {
                "fund_info": {"name": fund_info.get("fund_name"), "aum": fund_info.get("aum"), "strategy": fund_info.get("strategy")},
                "risk_summary": risk_summary_for_prompt,
                "compliance_summary": compliance_summary_for_prompt,
                "key_points": key_points_for_prompt # Placeholder for points to be generated
            }
        }
        logger.debug("Finished processing analysis results.")
        return due_diligence_data

    def _extract_crypto_entities(self, fund_info: Dict[str, Any], portfolio_data: Dict[str, float]) -> List[str]:
        # (Keep previous robust logic for extracting entities)
        crypto_entities = set()
        fund_name = fund_info.get("fund_name", "").lower()
        strategy = fund_info.get("strategy", "").lower()
        common_cryptos = ["bitcoin", "ethereum", "eth", "btc", "solana", "sol", "cardano", "ada", "binance", "bnb", "ripple", "xrp", "polkadot", "dot"] # Expanded list

        # Check fund name and strategy
        for crypto in common_cryptos:
            standard_name = self._standardize_crypto_name(crypto)
            if crypto in fund_name or crypto in strategy:
                crypto_entities.add(standard_name)

        # Check portfolio assets
        for asset_name in portfolio_data.keys():
            asset_lower = asset_name.lower()
            for crypto in common_cryptos:
                if crypto in asset_lower:
                    standard_name = self._standardize_crypto_name(crypto)
                    crypto_entities.add(standard_name)
                    break # Move to next asset once a match is found

        return list(crypto_entities)

    def _standardize_crypto_name(self, name: str) -> str:
        """Helper to map variations to standard names."""
        name = name.lower()
        if name in ["eth", "ether"]: return "ethereum"
        if name in ["btc", "xbt"]: return "bitcoin"
        if name == "sol": return "solana"
        if name == "bnb": return "binance"
        if name == "ada": return "cardano"
        if name == "xrp": return "ripple"
        if name == "dot": return "polkadot"
        # Add more mappings as needed
        return name

    def _map_level_to_category(self, level: float) -> str:
         """Helper to map a 0-10 risk level score to a category"""
         # This mirrors the logic likely in RiskAnalyzer._determine_risk_level but applied here
         if level < 2.0: return "Very Low"
         elif level < 4.0: return "Low"
         elif level < 6.0: return "Medium"
         elif level < 8.0: return "High"
         else: return "Very High"


    def close(self):
        if hasattr(self, 'storage') and self.storage:
            try:
                self.storage.close()
                logger.info("Storage manager closed by coordinator.")
            except Exception as e:
                logger.error(f"Error closing storage manager via coordinator: {e}")
        logger.info("Due Diligence Coordinator resources released.")