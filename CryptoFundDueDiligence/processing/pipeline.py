"""
Due Diligence Pipeline Module

This module defines the processing pipeline for crypto fund due diligence.
It manages the workflow from document upload to analysis and data enrichment.
"""

import os
import logging
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DueDiligencePipeline:
    """
    Pipeline for processing crypto fund due diligence.
    
    This class manages the workflow of due diligence analysis:
    1. Document ingestion and initial processing
    2. Data enrichment from various sources
    3. Analysis execution in parallel where possible
    4. Result aggregation and validation
    """
    
    def __init__(self, coordinator):
        """
        Initialize the pipeline with a coordinator.
        
        Args:
            coordinator: DueDiligenceCoordinator instance
        """
        self.coordinator = coordinator
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def process_document_async(self, document_id: str) -> Dict[str, Any]:
        """
        Process a document asynchronously with parallel data enrichment.
        
        Args:
            document_id: ID of the document to process
            
        Returns:
            Dict containing all synthesized due diligence data
        """
        logger.info(f"Starting asynchronous processing pipeline for document: {document_id}")
        
        try:
            # Step 1: Get document and perform initial analysis
            # This is done in the coordinator and cannot be parallelized
            initial_results = await self._run_in_executor(
                self.coordinator.process_document, 
                document_id
            )
            
            if "error" in initial_results:
                return initial_results
            
            # Step 2: Enrich with additional data in parallel
            enriched_results = await self._enrich_data_parallel(initial_results)
            
            # Step 3: Validate and finalize results
            final_results = self._validate_results(enriched_results)
            
            logger.info(f"Pipeline processing completed for document: {document_id}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in processing pipeline for document {document_id}: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Pipeline processing failed: {str(e)}"}
    
    async def _run_in_executor(self, func, *args):
        """Run a blocking function in the executor."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, func, *args
        )
    
    async def _enrich_data_parallel(self, initial_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich the initial results with additional data in parallel.
        
        Args:
            initial_results: Results from initial document processing
            
        Returns:
            Enriched results with additional data
        """
        # Extract key information for enrichment
        fund_info = initial_results.get("fund_info", {})
        wallet_data = initial_results.get("wallet_data", [])
        portfolio_data = initial_results.get("portfolio_data", {})
        
        # Define enrichment tasks that can run in parallel
        enrichment_tasks = []
        
        # Task 1: Get historical market data if not already present
        if "market_analysis" in initial_results and not initial_results["market_analysis"].get("historical_data"):
            crypto_entities = self._extract_crypto_entities(fund_info, portfolio_data)
            if crypto_entities:
                enrichment_tasks.append(self._get_historical_market_data(crypto_entities))
        
        # Task 2: Get additional on-chain data if not already present
        if "onchain_analysis" in initial_results and not initial_results["onchain_analysis"].get("transaction_history"):
            wallet_addresses = [wallet.get("address") for wallet in wallet_data]
            if wallet_addresses:
                enrichment_tasks.append(self._get_transaction_history(wallet_addresses))
        
        # Task 3: Get regulatory information based on fund jurisdictions
        if "compliance_analysis" in initial_results:
            jurisdictions = self._extract_jurisdictions(initial_results)
            if jurisdictions:
                enrichment_tasks.append(self._get_regulatory_information(jurisdictions))
        
        # Execute all enrichment tasks in parallel
        if enrichment_tasks:
            logger.info(f"Running {len(enrichment_tasks)} enrichment tasks in parallel")
            enrichment_results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)
            
            # Process the results and update the initial data
            for i, result in enumerate(enrichment_results):
                if isinstance(result, Exception):
                    logger.error(f"Enrichment task {i} failed: {result}")
                else:
                    self._update_results_with_enrichment(initial_results, result)
        
        return initial_results
    
    async def _get_historical_market_data(self, crypto_entities: List[str]) -> Dict[str, Any]:
        """
        Get historical market data for crypto entities.
        
        Args:
            crypto_entities: List of cryptocurrency entities
            
        Returns:
            Historical market data
        """
        return await self._run_in_executor(
            self._fetch_historical_market_data,
            crypto_entities
        )
    
    def _fetch_historical_market_data(self, crypto_entities: List[str]) -> Dict[str, Any]:
        """
        Fetch historical market data (executor function).
        
        Args:
            crypto_entities: List of cryptocurrency entities
            
        Returns:
            Historical market data
        """
        from analysis.market_analyzer import MarketAnalyzer
        
        market_analyzer = MarketAnalyzer(self.coordinator.data_retriever)
        historical_data = market_analyzer.get_historical_data(crypto_entities)
        
        return {
            "type": "historical_market_data",
            "data": historical_data
        }
    
    async def _get_transaction_history(self, wallet_addresses: List[str]) -> Dict[str, Any]:
        """
        Get transaction history for wallet addresses.
        
        Args:
            wallet_addresses: List of wallet addresses
            
        Returns:
            Transaction history data
        """
        return await self._run_in_executor(
            self._fetch_transaction_history,
            wallet_addresses
        )
    
    def _fetch_transaction_history(self, wallet_addresses: List[str]) -> Dict[str, Any]:
        """
        Fetch transaction history (executor function).
        
        Args:
            wallet_addresses: List of wallet addresses
            
        Returns:
            Transaction history data
        """
        from analysis.onchain_analyzer import OnChainAnalyzer
        
        onchain_analyzer = OnChainAnalyzer(self.coordinator.data_retriever)
        transaction_history = onchain_analyzer.get_transaction_history(wallet_addresses)
        
        return {
            "type": "transaction_history",
            "data": transaction_history
        }
    
    async def _get_regulatory_information(self, jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Get regulatory information for jurisdictions.
        
        Args:
            jurisdictions: List of jurisdictions
            
        Returns:
            Regulatory information
        """
        return await self._run_in_executor(
            self._fetch_regulatory_information,
            jurisdictions
        )
    
    def _fetch_regulatory_information(self, jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Fetch regulatory information (executor function).
        
        Args:
            jurisdictions: List of jurisdictions
            
        Returns:
            Regulatory information
        """
        from analysis.compliance_analyzer import ComplianceAnalyzer
        
        compliance_analyzer = ComplianceAnalyzer(self.coordinator.data_retriever)
        regulatory_info = compliance_analyzer.get_regulatory_information(jurisdictions)
        
        return {
            "type": "regulatory_information",
            "data": regulatory_info
        }
    
    def _update_results_with_enrichment(self, results: Dict[str, Any], enrichment_result: Dict[str, Any]) -> None:
        """
        Update the results with enrichment data.
        
        Args:
            results: The results to update
            enrichment_result: The enrichment data to add
        """
        if not isinstance(enrichment_result, dict) or "type" not in enrichment_result:
            return
        
        enrichment_type = enrichment_result.get("type")
        enrichment_data = enrichment_result.get("data", {})
        
        if enrichment_type == "historical_market_data":
            if "market_analysis" in results:
                results["market_analysis"]["historical_data"] = enrichment_data
            else:
                results["market_analysis"] = {"historical_data": enrichment_data}
                
        elif enrichment_type == "transaction_history":
            if "onchain_analysis" in results:
                results["onchain_analysis"]["transaction_history"] = enrichment_data
            else:
                results["onchain_analysis"] = {"transaction_history": enrichment_data}
                
        elif enrichment_type == "regulatory_information":
            if "compliance_analysis" in results:
                results["compliance_analysis"]["regulatory_information"] = enrichment_data
            else:
                results["compliance_analysis"] = {"regulatory_information": enrichment_data}
    
    def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and finalize the enriched results.
        
        Args:
            results: The enriched results to validate
            
        Returns:
            Validated and finalized results
        """
        # Check for missing critical information
        missing_fields = []
        
        if not results.get("fund_info", {}).get("fund_name"):
            missing_fields.append("Fund Name")
            
        if not results.get("fund_info", {}).get("aum"):
            missing_fields.append("AUM")
            
        if not results.get("wallet_data"):
            missing_fields.append("Wallet Data")
            
        # Add validation status
        results["validation"] = {
            "is_valid": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "warnings": self._generate_warnings(results),
            "validation_timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def _generate_warnings(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate warnings for potential issues in the results.
        
        Args:
            results: The results to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for low analysis confidence
        confidence = results.get("analysis_confidence", 0)
        if confidence < 70:
            warnings.append(f"Low analysis confidence: {confidence}%")
            
        # Check for high risk scores
        risk_assessment = results.get("risk_assessment", {})
        risk_factors = risk_assessment.get("risk_factors", [])
        
        high_risk_factors = [
            factor for factor in risk_factors 
            if factor.get("rating", 0) > 7  # High risk threshold
        ]
        
        if high_risk_factors:
            factor_names = [factor.get("factor", "Unknown") for factor in high_risk_factors]
            warnings.append(f"High risk factors detected: {', '.join(factor_names)}")
            
        # Check for regulatory issues
        compliance_analysis = results.get("compliance_analysis", {})
        regulatory_issues = compliance_analysis.get("regulatory_issues", [])
        
        if regulatory_issues:
            warnings.append(f"Regulatory issues detected: {len(regulatory_issues)} issues found")
            
        # Check wallet concentration
        wallet_data = results.get("wallet_data", [])
        if wallet_data and len(wallet_data) == 1:
            warnings.append("Single wallet concentration risk: Fund uses only one wallet")
            
        return warnings
    
    def _extract_crypto_entities(self, fund_info: Dict[str, Any], portfolio_data: Dict[str, float]) -> List[str]:
        """
        Extract cryptocurrency entities from fund information and portfolio.
        
        Args:
            fund_info: Fund information
            portfolio_data: Portfolio allocation data
            
        Returns:
            List of cryptocurrency entities
        """
        # This is a delegate to the coordinator's method
        return self.coordinator._extract_crypto_entities(fund_info, portfolio_data)
    
    def _extract_jurisdictions(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract jurisdictions from results.
        
        Args:
            results: Analysis results
            
        Returns:
            List of jurisdictions
        """
        jurisdictions = set()
        
        # Check compliance data
        compliance_data = results.get("compliance_analysis", {})
        if "regulatory_status" in compliance_data:
            for status in compliance_data["regulatory_status"]:
                for jurisdiction in ["US", "USA", "UK", "EU", "Cayman", "Singapore", "Switzerland", "Hong Kong"]:
                    if jurisdiction in status:
                        jurisdictions.add(jurisdiction)
        
        # Check fund info
        fund_info = results.get("fund_info", {})
        for field in ["jurisdiction", "registration"]:
            if field in fund_info:
                value = fund_info[field]
                for jurisdiction in ["US", "USA", "UK", "EU", "Cayman", "Singapore", "Switzerland", "Hong Kong"]:
                    if jurisdiction in value:
                        jurisdictions.add(jurisdiction)
        
        return list(jurisdictions)
    
    def process_document(self, document_id: str) -> Dict[str, Any]:
        """
        Process a document synchronously (for backward compatibility).
        
        Args:
            document_id: ID of the document to process
            
        Returns:
            Dict containing all synthesized due diligence data
        """
        logger.info(f"Starting synchronous processing pipeline for document: {document_id}")
        
        # Create an event loop if necessary
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except (RuntimeError, ValueError):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the async processing
        try:
            results = loop.run_until_complete(self.process_document_async(document_id))
            return results
        except Exception as e:
            logger.error(f"Error in synchronous processing pipeline: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Pipeline processing failed: {str(e)}"}
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("Due Diligence Pipeline resources cleaned up")