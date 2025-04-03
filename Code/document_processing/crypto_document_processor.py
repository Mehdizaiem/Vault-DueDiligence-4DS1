from Code.document_processing.document_processor import DocumentProcessor
import re
import logging
from typing import Dict, List, Any, Optional
import spacy
from spacy.matcher import PhraseMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDocumentProcessor(DocumentProcessor):
    """
    Specialized DocumentProcessor for cryptocurrency and blockchain documents
    with enhanced extraction capabilities for crypto-specific information
    """
    
    def __init__(self, use_gpu: bool = False):
        """Initialize with parent class and add specialized extractors"""
        super().__init__(use_gpu=use_gpu)
        
        # Initialize specialized crypto patterns
        self._init_specialized_crypto_patterns()
        self._init_compliance_patterns()
        
    def _init_specialized_crypto_patterns(self):
        """Initialize specialized patterns for crypto extraction"""
        # Blockchain protocols and consensus mechanisms
        self.blockchain_protocols = [
            "Proof of Work", "PoW", "Proof of Stake", "PoS", "Delegated Proof of Stake", "DPoS",
            "Proof of Authority", "PoA", "Proof of Space", "PoSpace", "Proof of Capacity", "PoC",
            "Practical Byzantine Fault Tolerance", "PBFT", "Tendermint", "Avalanche consensus",
            "Directed Acyclic Graph", "DAG", "Proof of History", "PoH", "Proof of Elapsed Time",
            "DPOS", "Nominated Proof of Stake", "NPoS", "Proof of Importance", "PoI"
        ]
        
        # Token standards
        self.token_standards = [
            "ERC-20", "ERC20", "ERC-721", "ERC721", "ERC-1155", "ERC1155",
            "BEP-20", "BEP20", "BEP-721", "BEP721", "BEP-1155", "BEP1155",
            "TRC-20", "TRC20", "TRC-721", "TRC721", "TRC-1155", "TRC1155",
            "SPL Token", "FA 1.2", "FA 2.0", "NEP-5", "NEP5"
        ]
        
        # Crypto exchanges
        self.crypto_exchanges = [
            "Binance", "Coinbase", "Kraken", "Gemini", "FTX", "BitFinex", "Bitstamp",
            "OKEx", "Huobi", "KuCoin", "BitMEX", "Bybit", "Gate.io", "Bittrex",
            "Poloniex", "Coincheck", "CoinDCX", "WazirX", "CoinSwitch", "Uniswap",
            "SushiSwap", "PancakeSwap", "dYdX", "1inch", "SundaeSwap", "TraderJoe"
        ]
        
        # Create a matcher for blockchain protocols
        self.protocol_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        protocol_patterns = [self.nlp.make_doc(text) for text in self.blockchain_protocols]
        self.protocol_matcher.add("BLOCKCHAIN_PROTOCOLS", protocol_patterns)
        
        # Create a matcher for token standards
        self.token_standard_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        token_patterns = [self.nlp.make_doc(text) for text in self.token_standards]
        self.token_standard_matcher.add("TOKEN_STANDARDS", token_patterns)
        
        # Create a matcher for crypto exchanges
        self.exchange_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        exchange_patterns = [self.nlp.make_doc(text) for text in self.crypto_exchanges]
        self.exchange_matcher.add("CRYPTO_EXCHANGES", exchange_patterns)
        
        # Regular expressions for token economics and metrics
        self.supply_pattern = re.compile(r'(?:total|max|circulating|current)\s+supply\s+(?:is|of|:)?\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|billion|trillion|M|B|T)?', re.IGNORECASE)
        self.ico_price_pattern = re.compile(r'(?:ico|initial|token|sale)\s+price\s+(?:is|of|:)?\s+\$?(\d+(?:,\d+)*(?:\.\d+)?)', re.IGNORECASE)
        self.gas_fee_pattern = re.compile(r'gas\s+(?:price|fee|cost)\s+(?:is|of|:)?\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(gwei|eth|btc|sat)', re.IGNORECASE)
    
    def _init_compliance_patterns(self):
        """Initialize patterns specific to compliance and legal requirements"""
        # KYC/AML terminology
        self.compliance_terms = [
            "know your customer", "KYC", "anti-money laundering", "AML", 
            "customer due diligence", "CDD", "enhanced due diligence", "EDD",
            "suspicious activity report", "SAR", "suspicious transaction report", "STR",
            "travel rule", "FATF Recommendation 16", "sanctions screening",
            "politically exposed person", "PEP", "beneficial owner", "BO",
            "transaction monitoring", "risk-based approach", "RBA", "sanctions compliance",
            "OFAC", "compliance program", "compliance officer", "CCO", "compliance policy"
        ]
        
        # Create a matcher for compliance terms
        self.compliance_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        compliance_patterns = [self.nlp.make_doc(text) for text in self.compliance_terms]
        self.compliance_matcher.add("COMPLIANCE_TERMS", compliance_patterns)
        
        # Smart contract risks and vulnerabilities
        self.smart_contract_risks = [
            "reentrancy", "front-running", "oracle manipulation", "flash loan attack",
            "integer overflow", "integer underflow", "DoS attack", "denial of service",
            "timestamp dependence", "block hash dependence", "tx.origin authentication",
            "unchecked return values", "race condition", "price manipulation", "sandwich attack",
            "rugpull", "rug pull", "exit scam", "backdoor", "honeypot", "gas griefing",
            "signature replay", "ERC20 approve race", "solidity version", "unauthorized access",
            "storage collision", "arithmetic overflow", "logic error", "gas limit", "out of gas"
        ]
        
        # Create a matcher for smart contract risks
        self.risk_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        risk_patterns = [self.nlp.make_doc(text) for text in self.smart_contract_risks]
        self.risk_matcher.add("SMART_CONTRACT_RISKS", risk_patterns)
    
    def process_document(self, text: str, filename: Optional[str] = None, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with crypto-specific enhancements.
        
        Args:
            text (str): The document text
            filename (str, optional): Source filename
            document_type (str, optional): Document type if known
            
        Returns:
            Dict[str, Any]: Extracted features and metadata with crypto-specific additions
        """
        # Process using the base implementation
        result = super().process_document(text, filename, document_type)
        
        # Apply crypto-specific enhancements
        result["crypto_specific"] = self._extract_crypto_specific_data(text, self.nlp(text))
        
        # Extract compliance information
        result["compliance_data"] = self._extract_compliance_data(text, self.nlp(text))
        
        return result
    
    def _extract_crypto_specific_data(self, text: str, doc: Any) -> Dict[str, Any]:
        """
        Extract crypto-specific data from the document.
        
        Args:
            text (str): Document text
            doc: spaCy processed document
            
        Returns:
            Dict[str, Any]: Crypto-specific data
        """
        crypto_data = {
            "blockchain_protocols": [],
            "token_standards": [],
            "exchanges_mentioned": [],
            "token_economics": {},
            "wallet_addresses": []
        }
        
        # Extract blockchain protocols
        protocol_matches = self.protocol_matcher(doc)
        for match_id, start, end in protocol_matches:
            span = doc[start:end]
            crypto_data["blockchain_protocols"].append(span.text)
        
        # Extract token standards
        token_matches = self.token_standard_matcher(doc)
        for match_id, start, end in token_matches:
            span = doc[start:end]
            crypto_data["token_standards"].append(span.text)
        
        # Extract crypto exchanges
        exchange_matches = self.exchange_matcher(doc)
        for match_id, start, end in exchange_matches:
            span = doc[start:end]
            crypto_data["exchanges_mentioned"].append(span.text)
        
        # Extract token economics data
        token_economics = {}
        
        # Extract supply information
        supply_matches = self.supply_pattern.findall(text)
        if supply_matches:
            token_economics["supply"] = supply_matches[0]
        
        # Extract ICO price
        ico_price_matches = self.ico_price_pattern.findall(text)
        if ico_price_matches:
            token_economics["ico_price"] = ico_price_matches[0]
        
        # Extract gas fees
        gas_fee_matches = self.gas_fee_pattern.findall(text)
        if gas_fee_matches:
            token_economics["gas_fee"] = gas_fee_matches[0][0]
            token_economics["gas_fee_unit"] = gas_fee_matches[0][1]
        
        crypto_data["token_economics"] = token_economics
        
        # Extract wallet addresses (reuse from parent class)
        btc_addresses = self.btc_address_pattern.findall(text)
        eth_addresses = self.eth_address_pattern.findall(text)
        crypto_data["wallet_addresses"] = list(set(btc_addresses + eth_addresses))
        
        # Remove duplicates
        for key in ["blockchain_protocols", "token_standards", "exchanges_mentioned"]:
            crypto_data[key] = list(set(crypto_data[key]))
        
        return crypto_data
    
    def _extract_compliance_data(self, text: str, doc: Any) -> Dict[str, Any]:
        """
        Extract compliance-related data from the document.
        
        Args:
            text (str): Document text
            doc: spaCy processed document
            
        Returns:
            Dict[str, Any]: Compliance-related data
        """
        compliance_data = {
            "kyc_aml_mentions": [],
            "regulatory_requirements": [],
            "smart_contract_risks": [],
            "compliance_level": "unknown"
        }
        
        # Extract KYC/AML mentions
        compliance_matches = self.compliance_matcher(doc)
        for match_id, start, end in compliance_matches:
            span = doc[start:end]
            sent = span.sent.text
            compliance_data["kyc_aml_mentions"].append({
                "term": span.text,
                "context": sent
            })
        
        # Extract smart contract risks
        risk_matches = self.risk_matcher(doc)
        for match_id, start, end in risk_matches:
            span = doc[start:end]
            sent = span.sent.text
            compliance_data["smart_contract_risks"].append({
                "risk": span.text,
                "context": sent
            })
        
        # Extract regulatory requirements
        # Look for sentences with requirement language
        requirement_markers = ["must", "shall", "required", "mandatory", "obligation", "comply", "compliance"]
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(marker in sent_lower for marker in requirement_markers):
                # Check if the sentence also mentions regulation
                if any(reg_term in sent_lower for reg_term in ["regulation", "law", "compliance", "kyc", "aml"]):
                    compliance_data["regulatory_requirements"].append(sent.text)
        
        # Attempt to determine compliance level based on mentions
        if len(compliance_data["kyc_aml_mentions"]) > 5:
            compliance_data["compliance_level"] = "high"
        elif len(compliance_data["kyc_aml_mentions"]) > 2:
            compliance_data["compliance_level"] = "medium"
        elif len(compliance_data["kyc_aml_mentions"]) > 0:
            compliance_data["compliance_level"] = "low"
        
        return compliance_data
    
    def analyze_token_distribution(self, text: str) -> Dict[str, Any]:
        """
        Analyze token distribution information in the document.
        
        Args:
            text (str): Document text
            
        Returns:
            Dict[str, Any]: Token distribution analysis
        """
        distribution = {
            "categories": [],
            "percentages": [],
            "vesting_periods": []
        }
        
        # Look for token distribution sections
        distribution_section = ""
        doc_sections = text.split("\n\n")
        
        for section in doc_sections:
            if "token distribution" in section.lower() or "token allocation" in section.lower():
                distribution_section = section
                break
        
        if not distribution_section:
            return distribution
        
        # Extract allocation categories and percentages
        allocation_pattern = re.compile(r'(\w+(?:\s+\w+)*)\s*(?:-|:)\s*(\d+(?:\.\d+)?)%', re.IGNORECASE)
        allocations = allocation_pattern.findall(distribution_section)
        
        for category, percentage in allocations:
            distribution["categories"].append(category.strip())
            distribution["percentages"].append(float(percentage))
        
        # Look for vesting information
        vesting_pattern = re.compile(r'(?:vesting|lock(?:ed|ing))\s+(?:period|schedule)?\s+(?:of|is|:)?\s+(\d+)\s+(month|year|day)s?', re.IGNORECASE)
        vesting_matches = vesting_pattern.findall(distribution_section)
        
        for period, unit in vesting_matches:
            distribution["vesting_periods"].append(f"{period} {unit}{'s' if int(period) > 1 and not unit.endswith('s') else ''}")
        
        return distribution
    
    def analyze_smart_contract_security(self, text: str) -> Dict[str, Any]:
        """
        Analyze smart contract security information in the document.
        
        Args:
            text (str): Document text
            
        Returns:
            Dict[str, Any]: Smart contract security analysis
        """
        security = {
            "audits_mentioned": [],
            "vulnerabilities": [],
            "security_measures": [],
            "security_score": 0
        }
        
        # Look for audit mentions
        audit_pattern = re.compile(r'(?:audit(?:ed)?|security review)\s+by\s+([A-Za-z0-9\s]+)', re.IGNORECASE)
        audit_matches = audit_pattern.findall(text)
        
        for audit in audit_matches:
            security["audits_mentioned"].append(audit.strip())
        
        # Look for security measures
        security_measures = [
            "multi-signature", "multisig", "time lock", "timelock", "freeze function",
            "pause function", "circuit breaker", "formal verification", "automated testing",
            "unit tests", "role-based access", "privilege separation", "proxy pattern"
        ]
        
        for measure in security_measures:
            if measure in text.lower():
                security["security_measures"].append(measure)
        
        # Count vulnerabilities (reuse smart contract risks)
        doc = self.nlp(text)
        risk_matches = self.risk_matcher(doc)
        for match_id, start, end in risk_matches:
            span = doc[start:end]
            security["vulnerabilities"].append(span.text)
        
        # Calculate basic security score
        security_score = 0
        security_score += len(security["audits_mentioned"]) * 20  # Up to 60 points for audits
        security_score += len(security["security_measures"]) * 5   # Up to 40 points for security measures
        security_score -= len(security["vulnerabilities"]) * 10    # Deductions for vulnerabilities
        
        # Normalize score
        security["security_score"] = max(0, min(100, security_score))
        
        return security