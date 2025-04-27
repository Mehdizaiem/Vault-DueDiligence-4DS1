import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
import traceback
import json
from pathlib import Path
import sys

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import base document processor
from Code.document_processing.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDocumentProcessor(DocumentProcessor):
    """
    Specialized document processor for cryptocurrency and blockchain documents.
    Enhances the base DocumentProcessor with crypto-specific processing.
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the crypto document processor.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration
        """
        # Initialize the base processor
        super().__init__(use_gpu=use_gpu)
        
        # Add the sentencizer component to fix the sentence boundaries issue
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')
            logger.info("Added sentencizer to the NLP pipeline")
        
        # Load crypto-specific resources
        self._load_crypto_resources()
        
        logger.info("Initialized CryptoDocumentProcessor")
    
    def _load_crypto_resources(self):
        """Load cryptocurrency-specific resources and patterns"""
        # Crypto categorization
        self.crypto_categories = {
            "stablecoins": [
                "tether", "usdt", "usd coin", "usdc", "binance usd", "busd", "dai", 
                "terra usd", "ust", "true usd", "tusd", "paxos standard", "pax",
                "frax", "fei", "nusd", "susd", "eusd"
            ],
            "defi_protocols": [
                "uniswap", "curve", "aave", "compound", "maker", "synthetix", "yearn",
                "balancer", "sushiswap", "pancakeswap", "1inch", "bancor", "kyber",
                "dydx", "instadapp", "convex", "alchemix", "rari capital", "olympus dao"
            ],
            "token_standards": [
                "erc-20", "erc20", "erc-721", "erc721", "erc-1155", "erc1155",
                "bep-20", "bep20", "bep-721", "bep721", "trc-20", "trc20"
            ],
            "scaling_solutions": [
                "layer 2", "l2", "optimism", "arbitrum", "zk-rollups", "zk rollups",
                "optimistic rollups", "polygon", "matic", "plasma", "sidechains",
                "lightning network", "state channels", "loopring", "immutable x", "starknet"
            ],
            "blockchain_protocols": [
                "proof of work", "pow", "proof of stake", "pos", "proof of authority", "poa",
                "delegated proof of stake", "dpos", "practical byzantine fault tolerance", "pbft",
                "sharding", "consensus algorithm", "smart contract", "evm", "solidity",
                "web3", "decentralized", "distributed ledger"
            ],
            "regulatory_terms": [
                "kyc", "know your customer", "aml", "anti-money laundering", "cft",
                "counter financing of terrorism", "travel rule", "fatf", "compliance",
                "regulatory", "sec", "cftc", "finra", "mifid", "gdpr", "dora", "mica"
            ]
        }
        
        # Risk factors particularly important in crypto due diligence
        self.crypto_risk_factors = [
            "private key management", "key management practices", "custody solution",
            "cold storage", "hot wallet", "smart contract vulnerabilities", "audit",
            "third-party audit", "security audit", "code review", "code audit",
            "flash loan", "reentrancy attack", "frontrunning", "oracle manipulation",
            "market manipulation", "wash trading", "pump and dump", "rug pull", 
            "exit scam", "team anonymity", "anonymous team", "regulatory risk",
            "jurisdictional risk", "security breach", "hack", "exploit",
            "liquidity risk", "impermanent loss", "token distribution", "token inflation",
            "centralization risk", "governance risk", "excessive token allocation",
            "vesting schedule", "emission rate", "unaudited code", "admin keys",
            "backdoor", "upgradability risk", "fork risk", "chain re-org", "51% attack"
        ]
        
        # Common compliance requirements in crypto
        self.compliance_terms = [
            "registration", "licensing", "securities", "commodity", "utility token",
            "security token", "money services business", "msb", "money transmitter",
            "virtual asset service provider", "vasp", "custody", "investor protection",
            "accredited investor", "qualified investor", "disclosure requirements", 
            "reporting requirements", "safe harbor", "legal opinion", "jurisdictional",
            "cross-border", "sanctions", "ofac", "restricted territories", "travel rule"
        ]
    
    def process_document(self, text: str, filename: Optional[str] = None, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with enhanced crypto-specific features.
        
        Args:
            text (str): Document text
            filename (str, optional): Source filename
            document_type (str, optional): Document type
            
        Returns:
            Dict[str, Any]: Processed document data
        """
        # First, process with the base processor
        try:
            base_result = super().process_document(text, filename, document_type)
            
            # Now enhance with crypto-specific processing
            crypto_specific = self._extract_crypto_specific(text, base_result)
            base_result["crypto_specific"] = crypto_specific
            
            # Add compliance and regulatory data
            compliance_data = self._extract_compliance_data(text, base_result)
            base_result["compliance_data"] = compliance_data
            
            # Add enhanced risk indicators
            risk_indicators = self._extract_crypto_risk_indicators(text, base_result)
            base_result["risk_indicators"] = risk_indicators
            
            # Add source info if provided
            if filename:
                base_result["source"] = filename
            
            return base_result
            
        except Exception as e:
            logger.error(f"Error in crypto-specific document processing: {e}")
            logger.error(traceback.format_exc())
            
            # Try to return a minimal result
            return {
                "metadata": {"source": filename, "text": text[:5000] if text else ""},
                "document_type": document_type or "unknown",
                "error": str(e)
            }
    
    def _extract_crypto_specific(self, text: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract crypto-specific information from the document.
        
        Args:
            text (str): Document text
            base_result (Dict[str, Any]): Base processing result
            
        Returns:
            Dict[str, Any]: Crypto-specific data
        """
        doc_text = text.lower()
        
        # Extract crypto categories
        results = {}
        for category, terms in self.crypto_categories.items():
            found_terms = []
            for term in terms:
                if term.lower() in doc_text:
                    found_terms.append(term)
            
            if found_terms:
                results[category] = found_terms
        
        # Extract blockchain protocols mentioned (beyond those in categories)
        blockchain_protocols = []
        blockchain_entities = base_result.get("entities", {}).get("blockchain_terms", [])
        crypto_entities = base_result.get("entities", {}).get("cryptocurrencies", [])
        
        if blockchain_entities:
            blockchain_protocols.extend(blockchain_entities)
        
        if crypto_entities:
            blockchain_protocols.extend([e for e in crypto_entities if e.lower() not in [p.lower() for p in blockchain_protocols]])
        
        if blockchain_protocols:
            results["blockchain_protocols"] = blockchain_protocols
        
        return results
    
    def _extract_compliance_data(self, text: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract compliance-related data from the document.
        
        Args:
            text (str): Document text
            base_result (Dict[str, Any]): Base processing result
            
        Returns:
            Dict[str, Any]: Compliance data
        """
        doc_text = text.lower()
        
        # Initialize results
        compliance_data = {
            "compliance_terms": [],
            "compliance_level": "unknown",
            "regulatory_requirements": [],
            "jurisdictions": []
        }
        
        # Check for compliance terms
        for term in self.compliance_terms:
            if term.lower() in doc_text:
                compliance_data["compliance_terms"].append(term)
        
        # Get jurisdictions from base processing
        jurisdictions = base_result.get("regulatory_mentions", {}).get("jurisdictions", [])
        if jurisdictions:
            compliance_data["jurisdictions"] = jurisdictions
        
        # Get regulatory requirements from base processing
        requirements = base_result.get("regulatory_mentions", {}).get("compliance_requirements", [])
        if requirements:
            compliance_data["regulatory_requirements"] = requirements
        
        # Determine compliance level
        if len(compliance_data["compliance_terms"]) > 10 or len(compliance_data["regulatory_requirements"]) > 5:
            compliance_data["compliance_level"] = "high"
        elif len(compliance_data["compliance_terms"]) > 5 or len(compliance_data["regulatory_requirements"]) > 2:
            compliance_data["compliance_level"] = "medium"
        elif len(compliance_data["compliance_terms"]) > 0 or len(compliance_data["regulatory_requirements"]) > 0:
            compliance_data["compliance_level"] = "low"
        
        return compliance_data
    
    def _extract_crypto_risk_indicators(self, text: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract crypto-specific risk indicators from the document.
        
        Args:
            text (str): Document text
            base_result (Dict[str, Any]): Base processing result
            
        Returns:
            Dict[str, Any]: Enhanced risk indicators
        """
        doc_text = text.lower()
        
        # Get base risk indicators
        base_risks = base_result.get("risk_indicators", {})
        
        # Initialize enhanced risk indicators
        risk_indicators = {
            "risk_indicators": base_risks.get("risk_indicators", []),
            "high_risk_activities": base_risks.get("high_risk_activities", []),
            "crypto_specific_risks": [],
            "security_considerations": [],
            "risk_level": "unknown"
        }
        
        # Check for crypto-specific risk factors
        for risk_factor in self.crypto_risk_factors:
            if risk_factor.lower() in doc_text:
                risk_indicators["crypto_specific_risks"].append(risk_factor)
        
        # Check for security considerations specifically
        security_terms = ["security", "secure", "vulnerability", "vulnerabilities", 
                         "audit", "audited", "safety", "protection", "safeguard",
                         "backup", "backups", "encrypted", "encryption", "firewall"]
        
        for term in security_terms:
            if term.lower() in doc_text:
                # Try to get surrounding context
                idx = doc_text.find(term.lower())
                if idx >= 0:
                    start = max(0, idx - 100)
                    end = min(len(doc_text), idx + 100)
                    context = doc_text[start:end]
                    risk_indicators["security_considerations"].append({"term": term, "context": context})
        
        # Determine overall risk level
        total_risks = (len(risk_indicators["risk_indicators"]) + 
                     len(risk_indicators["high_risk_activities"]) * 2 + 
                     len(risk_indicators["crypto_specific_risks"]))
        
        if total_risks > 10:
            risk_indicators["risk_level"] = "high"
        elif total_risks > 5:
            risk_indicators["risk_level"] = "medium"
        elif total_risks > 0:
            risk_indicators["risk_level"] = "low"
        else:
            risk_indicators["risk_level"] = "minimal"
        
        return risk_indicators