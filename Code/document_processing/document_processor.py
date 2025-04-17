import re
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from datetime import datetime
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import networkx as nx
from collections import Counter, defaultdict
from pathlib import Path
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Enhanced DocumentProcessor class for crypto documents that:
    - Extracts financial metrics and key entities
    - Identifies regulatory mentions and compliance flags
    - Detects document type through classification
    - Builds relationship graphs between entities
    - Extracts key insights through summarization
    """

    # Document type labels
    DOC_TYPE_WHITEPAPER = "whitepaper"
    DOC_TYPE_AGREEMENT = "agreement"
    DOC_TYPE_REGULATION = "regulation"
    DOC_TYPE_LEGAL_CASE = "legal_case"
    DOC_TYPE_FINANCIAL_ANALYSIS = "financial_analysis"
    DOC_TYPE_RISK_ASSESSMENT = "risk_assessment"
    DOC_TYPE_OTHER = "other"

    def __init__(self, use_gpu: bool = False):
        """
        Initialize the DocumentProcessor with NLP models and detection patterns.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration for NLP tasks
        """
        # Load NLP models
        try:
            # Try loading the large model first for better entity recognition
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded en_core_web_lg NLP model")
        except OSError:
            try:
                # Fall back to medium model
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded en_core_web_md NLP model")
            except OSError:
                # Fall back to small model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded en_core_web_sm NLP model")
        
        # Configure GPU usage if available
        if use_gpu and spacy.prefer_gpu():
            self.nlp.to_gpu()
            logger.info("Using GPU acceleration for NLP tasks")
        
        # Initialize entity extraction patterns
        self._init_cryptocurrency_patterns()
        self._init_regulatory_patterns()
        self._init_financial_patterns()
        self._init_risk_indicators()
        
        # Initialize document classification patterns
        self._init_document_classifiers()
        
        # Initialize relationship extraction
        self.entity_graph = nx.DiGraph()
    
    def _init_cryptocurrency_patterns(self):
        """Initialize patterns for cryptocurrency entity recognition"""
        # Common cryptocurrencies and blockchain terms
        self.crypto_terms = [
            "Bitcoin", "BTC", "Ethereum", "ETH", "Ripple", "XRP", "Litecoin", "LTC",
            "Bitcoin Cash", "BCH", "Cardano", "ADA", "Polkadot", "DOT", "Solana", "SOL",
            "Binance Coin", "BNB", "blockchain", "smart contract", "token", "wallet",
            "private key", "public key", "mining", "miner", "hash", "block", "node",
            "address", "transaction", "VASP", "exchange", "cryptocurrency", "crypto",
            "virtual asset", "digital asset", "coin", "altcoin", "stablecoin", "ICO",
            "initial coin offering", "NFT", "non-fungible token", "DeFi", "decentralized finance",
            "DAO", "decentralized autonomous organization", "Web3"
        ]
        
        # Create a phrase matcher for cryptocurrencies
        self.crypto_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(text) for text in self.crypto_terms]
        self.crypto_matcher.add("CRYPTO_TERMS", patterns)
        
        # Regex patterns for crypto addresses and hashes
        self.btc_address_pattern = re.compile(r'\b(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b')
        self.eth_address_pattern = re.compile(r'\b0x[a-fA-F0-9]{40}\b')
        self.txid_pattern = re.compile(r'\b[a-fA-F0-9]{64}\b')  # Transaction hash pattern
    
    def _init_regulatory_patterns(self):
        """Initialize patterns for regulatory entities and terms"""
        # Regulatory bodies and frameworks
        self.regulatory_bodies = [
            "SEC", "Securities and Exchange Commission",
            "CFTC", "Commodity Futures Trading Commission",
            "FinCEN", "Financial Crimes Enforcement Network",
            "FATF", "Financial Action Task Force",
            "OCC", "Office of the Comptroller of the Currency",
            "FINRA", "Financial Industry Regulatory Authority",
            "FCA", "Financial Conduct Authority",
            "BaFin", "Federal Financial Supervisory Authority",
            "MAS", "Monetary Authority of Singapore",
            "ESMA", "European Securities and Markets Authority",
            "FSB", "Financial Stability Board",
            "OFAC", "Office of Foreign Assets Control",
            "NYSDFS", "New York State Department of Financial Services",
            "FINCEN", "Treasury", "IRS", "Internal Revenue Service"
        ]
        
        # Regulatory terms
        self.regulatory_terms = [
            "KYC", "know your customer", "AML", "anti-money laundering", "CTF", 
            "counter-terrorist financing", "BSA", "Bank Secrecy Act",
            "compliance", "regulation", "directive", "sanction", "fine", "penalty",
            "enforcement", "violation", "ruling", "license", "registration",
            "money transmitter", "money services business", "MSB", 
            "travel rule", "suspicious activity", "STR", "suspicious transaction report",
            "SAR", "suspicious activity report", "CDD", "customer due diligence",
            "EDD", "enhanced due diligence", "CFT", "combating financing of terrorism"
        ]
        
        # Create a phrase matcher for regulatory terms
        self.regulatory_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        reg_patterns = [self.nlp.make_doc(text) for text in self.regulatory_bodies + self.regulatory_terms]
        self.regulatory_matcher.add("REGULATORY_TERMS", reg_patterns)
    
    def _init_financial_patterns(self):
        """Initialize patterns for financial metrics and terms"""
        # Financial metrics and terms
        self.financial_terms = [
            "market cap", "market capitalization", "volume", "liquidity", "volatility",
            "price", "exchange rate", "trading volume", "transaction volume", "daily volume",
            "TVL", "total value locked", "revenue", "profit", "loss", "fee", "commission",
            "return on investment", "ROI", "APY", "annual percentage yield",
            "stake", "staking", "yield", "dividend", "interest", "APR", 
            "annual percentage rate", "gas fee", "network fee", "miner fee"
        ]
        
        # Create a phrase matcher for financial terms
        self.financial_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        fin_patterns = [self.nlp.make_doc(text) for text in self.financial_terms]
        self.financial_matcher.add("FINANCIAL_TERMS", fin_patterns)
        
        # Regex patterns for monetary values
        self.usd_pattern = re.compile(r'\$\s*\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s*(?:USD|dollars|US dollars)', re.IGNORECASE)
        self.crypto_value_pattern = re.compile(r'\d+(?:\.\d+)?\s*(?:BTC|ETH|XRP|LTC|BCH|ADA|DOT|SOL|BNB)', re.IGNORECASE)
        
        # Amount extraction patterns
        self.amount_pattern = re.compile(r'([0-9]+(?:[,.][0-9]+)*(?:\.[0-9]+)?)\s*(?:million|billion|trillion)?', re.IGNORECASE)
        self.million_billion_pattern = re.compile(r'million|billion|trillion', re.IGNORECASE)
    
    def _init_risk_indicators(self):
        """Initialize patterns for risk indicators in crypto documents"""
        # ML/TF risk indicators based on FATF red flags
        self.risk_indicators = [
            # Transaction patterns
            "multiple transactions", "structuring", "smurfing", "layering", "mixing", "tumbling",
            "obfuscate", "anonymous", "privacy coin", "mixer", "tumbler", "P2P platform",
            "darknet", "dark web", "suspicious", "unusual pattern", "shell company",
            
            # High-risk activities
            "unregistered", "unlicensed", "sanctioned", "fraud", "scam", "ponzi", "pyramid scheme",
            "ransomware", "extortion", "hacked", "stolen funds", "illicit", "illegal", "criminal",
            
            # Red flag behaviors
            "refused to provide", "false identification", "forged documents", "front company",
            "no economic purpose", "high-risk jurisdiction", "tax haven", "non-cooperative",
            
            # Other indicators
            "politically exposed person", "PEP", "high-risk customer", "high-risk country",
            "sanctioned country", "non-compliant", "terrorism", "terrorist financing",
            "drug trafficking", "human trafficking", "arms trafficking"
        ]
        
        # Create a phrase matcher for risk indicators
        self.risk_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        risk_patterns = [self.nlp.make_doc(text) for text in self.risk_indicators]
        self.risk_matcher.add("RISK_INDICATORS", risk_patterns)

    def _init_document_classifiers(self):
        """Initialize patterns for document classification"""
        # Document type classification patterns
        self.doc_type_patterns = {
            self.DOC_TYPE_WHITEPAPER: [
                "whitepaper", "white paper", "technical paper", "vision", "roadmap", 
                "tokenomics", "utility", "use case", "ecosystem", "architecture"
            ],
            self.DOC_TYPE_AGREEMENT: [
                "agreement", "contract", "terms", "conditions", "legal agreement",
                "token sale agreement", "purchase agreement", "service agreement",
                "license agreement", "user agreement", "smart contract agreement"
            ],
            self.DOC_TYPE_REGULATION: [
                "guidance", "advisory", "regulation", "compliance", "framework",
                "policy", "guideline", "rule", "directive", "regulatory", "regulator"
            ],
            self.DOC_TYPE_LEGAL_CASE: [
                "case", "court", "litigation", "plaintiff", "defendant", "judge",
                "ruling", "opinion", "decision", "versus", "v.", "appeal", "trial",
                "indictment", "prosecution", "criminal", "civil action"
            ],
            self.DOC_TYPE_FINANCIAL_ANALYSIS: [
                "analysis", "report", "research", "market research", "investment",
                "projection", "forecast", "outlook", "performance", "trend",
                "financial report", "quarterly report", "annual report"
            ],
            self.DOC_TYPE_RISK_ASSESSMENT: [
                "risk assessment", "vulnerability", "threat", "risk factor",
                "red flag", "indicator", "warning sign", "suspicious", "monitor"
            ]
        }
        
        # Create a phrase matcher for each document type
        self.doc_type_matchers = {}
        for doc_type, terms in self.doc_type_patterns.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(text) for text in terms]
            matcher.add(f"DOC_TYPE_{doc_type.upper()}", patterns)
            self.doc_type_matchers[doc_type] = matcher

    def process_document(self, text: str, filename: Optional[str] = None, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document to extract features, entities, and generate embedding.
        
        Args:
            text (str): The document text
            filename (str, optional): Source filename
            document_type (str, optional): Document type if known
            
        Returns:
            Dict[str, Any]: Extracted features and metadata
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text provided for processing")
            return {}
        
        # Truncate very long documents for NLP processing
        # We'll use the first and last parts which often contain key information
        max_nlp_length = 100000  # spaCy can handle documents up to ~1M chars but gets slow
        if len(text) > max_nlp_length:
            logger.info(f"Document length ({len(text)} chars) exceeds NLP processing limit, truncating")
            first_part = text[:max_nlp_length // 2]
            last_part = text[-max_nlp_length // 2:]
            nlp_text = first_part + " " + last_part
        else:
            nlp_text = text
        
        # Process with NLP
        doc = self.nlp(nlp_text)
        
        # Classify document type if not provided
        if not document_type:
            document_type = self._classify_document_type(doc, filename)
            
        # Extract basic metadata
        metadata = self._extract_metadata(text, doc, filename)
        
        # Extract entities based on document type
        entities = self._extract_entities(doc, document_type)
        
        # Extract financial metrics
        financial_metrics = self._extract_financial_metrics(text, doc)
        
        # Extract regulatory mentions
        regulatory_mentions = self._extract_regulatory_mentions(doc)
        
        # Extract risk indicators
        risk_indicators = self._extract_risk_indicators(doc)
        
        # Build relationship graph
        relationships = self._build_relationship_graph(doc, entities)
        
        # Generate summary
        summary = self._generate_summary(doc, document_type)
        
        # Compile results
        result = {
            "metadata": metadata,
            "document_type": document_type,
            "entities": entities,
            "financial_metrics": financial_metrics,
            "regulatory_mentions": regulatory_mentions,
            "risk_indicators": risk_indicators,
            "relationships": relationships,
            "summary": summary
        }
        
        return result
    
    def _classify_document_type(self, doc: Any, filename: Optional[str] = None) -> str:
        """
        Classify the document type based on content patterns.
        
        Args:
            doc: spaCy processed document
            filename (str, optional): Source filename
            
        Returns:
            str: Classified document type
        """
        # Convert doc to string for matching
        doc_text = doc.text.lower()
        
        # Use filename hints if available
        if filename:
            filename_lower = filename.lower()
            if "whitepaper" in filename_lower or "white-paper" in filename_lower:
                return self.DOC_TYPE_WHITEPAPER
            elif "agreement" in filename_lower or "contract" in filename_lower:
                return self.DOC_TYPE_AGREEMENT
            elif "regulation" in filename_lower or "guidance" in filename_lower:
                return self.DOC_TYPE_REGULATION
            elif "case" in filename_lower or "court" in filename_lower or "v." in filename_lower:
                return self.DOC_TYPE_LEGAL_CASE
            elif "analysis" in filename_lower or "report" in filename_lower:
                return self.DOC_TYPE_FINANCIAL_ANALYSIS
            elif "risk" in filename_lower or "assessment" in filename_lower:
                return self.DOC_TYPE_RISK_ASSESSMENT
        
        # Count pattern matches for each document type
        type_scores = {doc_type: 0 for doc_type in self.doc_type_patterns.keys()}
        
        for doc_type, matcher in self.doc_type_matchers.items():
            matches = matcher(doc)
            type_scores[doc_type] = len(matches)
        
        # Additional document-specific markers
        if "plaintiff" in doc_text or "defendant" in doc_text or "court" in doc_text:
            type_scores[self.DOC_TYPE_LEGAL_CASE] += 5
        
        if "token sale" in doc_text or "purchase agreement" in doc_text:
            type_scores[self.DOC_TYPE_AGREEMENT] += 5
            
        if "this whitepaper" in doc_text or "tokenomics" in doc_text:
            type_scores[self.DOC_TYPE_WHITEPAPER] += 5
            
        if "market analysis" in doc_text or "price prediction" in doc_text:
            type_scores[self.DOC_TYPE_FINANCIAL_ANALYSIS] += 5
            
        if "red flag indicators" in doc_text or "suspicious activity" in doc_text:
            type_scores[self.DOC_TYPE_RISK_ASSESSMENT] += 5
            
        if "regulatory framework" in doc_text or "guidance" in doc_text:
            type_scores[self.DOC_TYPE_REGULATION] += 5
        
        # Get the type with the highest score
        if max(type_scores.values()) > 0:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        # Default to other if no strong match
        return self.DOC_TYPE_OTHER
    
    def _extract_metadata(self, text: str, doc: Any, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract basic metadata from the document.
        
        Args:
            text (str): Full text of the document
            doc: spaCy processed document
            filename (str, optional): Source filename
            
        Returns:
            Dict[str, Any]: Document metadata
        """
        # Calculate word count based on full text (not just NLP-processed portion)
        word_count = len(text.split())
        
        # Calculate sentence count from NLP doc
        sentence_count = len(list(doc.sents))
        
        # Extract dates mentioned in the document
        dates = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append(ent.text)
        
        # Extract potential document date (typically near the beginning)
        doc_date = None
        if dates:
            doc_date = dates[0]  # Simplistic approach - take first date
        
        # Extract potential authors or organizations
        authors = []
        organizations = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                authors.append(ent.text)
            elif ent.label_ == "ORG":
                organizations.append(ent.text)
        
        # Count unique authors and organizations
        unique_authors = list(set(authors))[:10]  # Limit to top 10
        unique_organizations = list(set(organizations))[:10]  # Limit to top 10
        
        # Calculate reading time (assuming average 200 words per minute)
        reading_time_minutes = word_count / 200
        
        # Determine language (simplified - assuming English)
        language = "en"
        
        # Source info
        source = filename if filename else "unknown"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "reading_time_minutes": reading_time_minutes,
            "document_date": doc_date,
            "language": language,
            "source": source,
            "potential_authors": unique_authors,
            "potential_organizations": unique_organizations
        }
    
    def _extract_entities(self, doc: Any, document_type: str) -> Dict[str, List[str]]:
        """
        Extract entities based on document type.
        
        Args:
            doc: spaCy processed document
            document_type (str): Type of document
            
        Returns:
            Dict[str, List[str]]: Extracted entities by category
        """
        # Initialize entities dictionary
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "cryptocurrencies": [],
            "blockchain_terms": [],
            "defi_protocols": [],
            "token_standards": [],
            "nft_terms": [],
            "stablecoins": [],
            "scaling_solutions": [],
            "financial_instruments": [],
            "crypto_addresses": [],
            "dates": [],
            "monetary_values": []
        }
        
        # Extract named entities from NLP processing
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["persons"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "GPE" or ent.label_ == "LOC":
                entities["locations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["monetary_values"].append(ent.text)
        
        # Extract cryptocurrency terms using custom matcher
        crypto_matches = self.crypto_matcher(doc)
        for match_id, start, end in crypto_matches:
            span = doc[start:end]
            if any(term in span.text.lower() for term in ["bitcoin", "ethereum", "ripple", "litecoin", "cardano", "binance", "solana", "polkadot"]):
                entities["cryptocurrencies"].append(span.text)
            else:
                entities["blockchain_terms"].append(span.text)
        
        # Extract crypto addresses from full text
        doc_text = doc.text
        btc_addresses = self.btc_address_pattern.findall(doc_text)
        eth_addresses = self.eth_address_pattern.findall(doc_text)
        txids = self.txid_pattern.findall(doc_text)
        
        # Combine and deduplicate
        crypto_addresses = list(set(btc_addresses + eth_addresses))
        if crypto_addresses:
            entities["crypto_addresses"] = crypto_addresses
        
        # Add transaction IDs if relevant to document type
        if document_type in [self.DOC_TYPE_LEGAL_CASE, self.DOC_TYPE_RISK_ASSESSMENT]:
            entities["transaction_ids"] = list(set(txids))
        
        # Deduplicate all entity lists
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_financial_metrics(self, text: str, doc: Any) -> Dict[str, Any]:
        """
        Extract financial metrics from the document.
        
        Args:
            text (str): Full document text
            doc: spaCy processed document
            
        Returns:
            Dict[str, Any]: Extracted financial metrics
        """
        metrics = {
            "mentioned_amounts": [],
            "mentioned_cryptocurrencies": [],
            "market_values": {},
            "fees": [],
            "transaction_volumes": []
        }
        
        # Find all USD amounts
        usd_amounts = self.usd_pattern.findall(text)
        
        # Find all crypto amounts
        crypto_amounts = self.crypto_value_pattern.findall(text)
        
        # Extract money entities from NLP
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                metrics["mentioned_amounts"].append(ent.text)
        
        # Match financial terms
        financial_matches = self.financial_matcher(doc)
        
        # Process structured financial information
        for match_id, start, end in financial_matches:
            span = doc[start:end]
            term = span.text.lower()
            
            # Look for amounts near financial terms
            sent = span.sent
            sent_text = sent.text
            
            # Check different financial terms
            if "market cap" in term or "market capitalization" in term:
                # Look for amounts in the same sentence
                usd_in_sent = self.usd_pattern.findall(sent_text)
                if usd_in_sent:
                    metrics["market_values"]["market_cap"] = usd_in_sent[0]
            
            elif "volume" in term:
                usd_in_sent = self.usd_pattern.findall(sent_text)
                if usd_in_sent:
                    metrics["transaction_volumes"].append({"description": term, "amount": usd_in_sent[0]})
            
            elif "fee" in term:
                usd_in_sent = self.usd_pattern.findall(sent_text)
                if usd_in_sent:
                    metrics["fees"].append({"description": term, "amount": usd_in_sent[0]})
        
        # Process crypto amounts
        metrics["crypto_amounts"] = crypto_amounts
        
        # Extract potential amounts with million/billion/trillion
        big_numbers = []
        for sent in doc.sents:
            sent_text = sent.text
            amount_matches = self.amount_pattern.findall(sent_text)
            
            for amount in amount_matches:
                # Check if followed by million/billion/trillion
                # This is a simplistic approach and may need refinement
                if self.million_billion_pattern.search(sent_text):
                    big_numbers.append(f"{amount} {self.million_billion_pattern.search(sent_text).group(0)}")
        
        metrics["large_amounts"] = list(set(big_numbers))
        
        return metrics
    
    def _extract_regulatory_mentions(self, doc: Any) -> Dict[str, Any]:
        """
        Extract regulatory mentions from the document.
        
        Args:
            doc: spaCy processed document
            
        Returns:
            Dict[str, Any]: Extracted regulatory information
        """
        regulatory_info = {
            "regulatory_bodies": [],
            "regulatory_terms": [],
            "compliance_requirements": [],
            "jurisdictions": []
        }
        
        # Match regulatory terms and bodies
        regulatory_matches = self.regulatory_matcher(doc)
        
        for match_id, start, end in regulatory_matches:
            span = doc[start:end]
            term = span.text
            
            # Categorize the term
            term_lower = term.lower()
            
            if any(body.lower() in term_lower for body in self.regulatory_bodies):
                regulatory_info["regulatory_bodies"].append(term)
            else:
                regulatory_info["regulatory_terms"].append(term)
            
            # Extract compliance requirements from sentences mentioning regulations
            sent = span.sent.text
            if "must" in sent or "required" in sent or "shall" in sent or "mandatory" in sent:
                regulatory_info["compliance_requirements"].append(sent)
        
        # Extract potential jurisdictions
        jurisdictions = []
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Countries, cities, states
                jurisdictions.append(ent.text)
        
        regulatory_info["jurisdictions"] = list(set(jurisdictions))
        
        # Deduplicate
        for key in regulatory_info:
            regulatory_info[key] = list(set(regulatory_info[key]))
        
        return regulatory_info
    
    def _extract_risk_indicators(self, doc: Any) -> Dict[str, Any]:
        """
        Extract risk indicators from the document.
        
        Args:
            doc: spaCy processed document
            
        Returns:
            Dict[str, Any]: Extracted risk indicators
        """
        risk_info = {
            "risk_indicators": [],
            "high_risk_activities": [],
            "suspicious_patterns": [],
            "mentioned_jurisdictions": []
        }
        
        # Match risk terms
        risk_matches = self.risk_matcher(doc)
        
        for match_id, start, end in risk_matches:
            span = doc[start:end]
            term = span.text
            risk_info["risk_indicators"].append(term)
            
            # Categorize the indicator
            term_lower = term.lower()
            
            # High-risk activities
            if any(activity in term_lower for activity in ["fraud", "scam", "ponzi", "ransomware", "extortion", "illicit", "illegal"]):
                risk_info["high_risk_activities"].append(term)
            
            # Suspicious patterns
            if any(pattern in term_lower for pattern in ["structuring", "smurfing", "layering", "mixing", "tumbling", "suspicious", "unusual"]):
                # Get the sentence for context
                context = span.sent.text
                risk_info["suspicious_patterns"].append({"term": term, "context": context})
        
        # Extract jurisdictions that might be high-risk
        high_risk_contexts = []
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(term in sent_lower for term in ["high-risk", "high risk", "sanctioned", "non-cooperative"]):
                for ent in sent.ents:
                    if ent.label_ == "GPE":  # Country, state, city
                        risk_info["mentioned_jurisdictions"].append({"jurisdiction": ent.text, "context": sent.text})
        
        # Deduplicate lists
        risk_info["risk_indicators"] = list(set(risk_info["risk_indicators"]))
        risk_info["high_risk_activities"] = list(set(risk_info["high_risk_activities"]))
        
        return risk_info
    
    def _build_relationship_graph(self, doc: Any, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Build a relationship graph between entities.
        
        Args:
            doc: spaCy processed document
            entities: Previously extracted entities
            
        Returns:
            Dict[str, Any]: Entity relationships
        """
        relationships = {
            "entity_pairs": [],
            "connection_strength": {}
        }
        
        # Create a temporary graph for this document
        temp_graph = nx.DiGraph()
        
        # Extract persons and organizations
        persons = entities.get("persons", [])
        organizations = entities.get("organizations", [])
        cryptocurrencies = entities.get("cryptocurrencies", [])
        
        # Add all main entities to the graph
        all_key_entities = persons + organizations + cryptocurrencies
        for entity in all_key_entities:
            temp_graph.add_node(entity)
        
        # Find co-occurrences in sentences
        for sent in doc.sents:
            sent_entities = []
            
            # Find entities in this sentence
            for ent in sent.ents:
                if ent.text in all_key_entities:
                    sent_entities.append(ent.text)
            
            # Add edges between all entities in this sentence
            for i, entity1 in enumerate(sent_entities):
                for entity2 in sent_entities[i+1:]:
                    if entity1 != entity2:
                        # Check if this edge already exists
                        if temp_graph.has_edge(entity1, entity2):
                            # Increment weight
                            temp_graph[entity1][entity2]["weight"] += 1
                        else:
                            # Create new edge with weight 1
                            temp_graph.add_edge(entity1, entity2, weight=1)
        
        # Extract the strongest relationships
        strong_relationships = []
        for u, v, data in temp_graph.edges(data=True):
            if data["weight"] > 1:  # Consider relationships that occur more than once
                strong_relationships.append({
                    "source": u,
                    "target": v,
                    "strength": data["weight"]
                })
        
        # Sort by strength
        strong_relationships.sort(key=lambda x: x["strength"], reverse=True)
        
        # Add to the global entity graph
        for rel in strong_relationships:
            source = rel["source"]
            target = rel["target"]
            strength = rel["strength"]
            
            # Update the global graph
            if not self.entity_graph.has_node(source):
                self.entity_graph.add_node(source)
            if not self.entity_graph.has_node(target):
                self.entity_graph.add_node(target)
            
            if self.entity_graph.has_edge(source, target):
                self.entity_graph[source][target]["weight"] += strength
            else:
                self.entity_graph.add_edge(source, target, weight=strength)
        
        # Return the relationships found in this document
        relationships["entity_pairs"] = strong_relationships
        
        # Calculate overall connection strength for each entity (centrality)
        centrality = {}
        if len(temp_graph.nodes()) > 0:
            degree_centrality = nx.degree_centrality(temp_graph)
            for entity, centrality_score in degree_centrality.items():
                centrality[entity] = centrality_score
        
        relationships["connection_strength"] = centrality
        
        return relationships
    
    def _generate_summary(self, doc: Any, document_type: str) -> Dict[str, Any]:
        """
        Extract key insights through summarization.
        
        Args:
            doc: spaCy processed document
            document_type (str): Type of document
            
        Returns:
            Dict[str, Any]: Document summary and key insights
        """
        summary = {
            "key_sentences": [],
            "key_insights": [],
            "main_topics": [],
            "document_purpose": ""
        }
        
        # Extract most important sentences based on entity density and sentence position
        important_sentences = []
        
        # Give importance to first few and last few sentences in the document
        sent_list = list(doc.sents)
        if len(sent_list) > 2:
            # Add first three sentences
            important_sentences.extend(sent_list[:3])
            
            # Add last two sentences
            important_sentences.extend(sent_list[-2:])
        else:
            important_sentences.extend(sent_list)
        
        # Find sentences with high entity density or important keywords
        for sent in doc.sents:
            # Count entities in the sentence
            entity_count = sum(1 for ent in sent.ents)
            
            # Check for important keywords based on document type
            if document_type == self.DOC_TYPE_WHITEPAPER:
                if any(keyword in sent.text.lower() for keyword in ["purpose", "vision", "mission", "objective", "goal", "aims to", "designed to"]):
                    important_sentences.append(sent)
            
            elif document_type == self.DOC_TYPE_AGREEMENT:
                if any(keyword in sent.text.lower() for keyword in ["agree", "terms", "condition", "party", "obligation", "shall", "must", "covenant"]):
                    important_sentences.append(sent)
            
            elif document_type == self.DOC_TYPE_REGULATION:
                if any(keyword in sent.text.lower() for keyword in ["require", "prohibit", "must", "shall", "regulate", "compliance", "subject to"]):
                    important_sentences.append(sent)
            
            elif document_type == self.DOC_TYPE_LEGAL_CASE:
                if any(keyword in sent.text.lower() for keyword in ["court", "ruled", "decision", "judgment", "opinion", "held", "concluded"]):
                    important_sentences.append(sent)
            
            elif document_type == self.DOC_TYPE_FINANCIAL_ANALYSIS:
                if any(keyword in sent.text.lower() for keyword in ["analysis", "shows", "indicates", "concludes", "predicts", "projects", "estimates"]):
                    important_sentences.append(sent)
            
            elif document_type == self.DOC_TYPE_RISK_ASSESSMENT:
                if any(keyword in sent.text.lower() for keyword in ["risk", "warning", "indicator", "flag", "suspicious", "monitor", "detect"]):
                    important_sentences.append(sent)
            
            # If sentence has many entities, it's probably important
            elif entity_count >= 3:
                important_sentences.append(sent)
        
        # Remove duplicates and sort by position in document
        unique_important_sentences = []
        sent_positions = {}
        
        for i, sent in enumerate(doc.sents):
            sent_positions[sent.text] = i
        
        for sent in important_sentences:
            if sent.text not in [s.text for s in unique_important_sentences]:
                unique_important_sentences.append(sent)
        
        unique_important_sentences.sort(key=lambda sent: sent_positions.get(sent.text, 0))
        
        # Limit to top sentences
        top_sentences = unique_important_sentences[:5]  # Limit to 5 sentences
        summary["key_sentences"] = [sent.text for sent in top_sentences]
        
        # Extract key insights based on document type (using generic approach for now)
        insights = []
        
        # Look for sentences with insight markers based on document type
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            
            # Common insight markers across document types
            if any(marker in sent_lower for marker in ["key", "important", "significant", "critical", "essential"]):
                insights.append(sent.text)
            
            # Document-specific markers
            if document_type == self.DOC_TYPE_WHITEPAPER:
                if any(marker in sent_lower for marker in ["provide", "enable", "solution", "problem", "innovation"]):
                    insights.append(sent.text)
            
            elif document_type == self.DOC_TYPE_AGREEMENT:
                if any(marker in sent_lower for marker in ["obligation", "right", "must", "shall", "agree to"]):
                    insights.append(sent.text)
            
            elif document_type == self.DOC_TYPE_REGULATION:
                if any(marker in sent_lower for marker in ["require", "regulate", "compliance", "prohibited"]):
                    insights.append(sent.text)
            
            elif document_type == self.DOC_TYPE_LEGAL_CASE:
                if any(marker in sent_lower for marker in ["court held", "ruling", "decided", "judgment", "conclusion"]):
                    insights.append(sent.text)
            
            elif document_type == self.DOC_TYPE_FINANCIAL_ANALYSIS:
                if any(marker in sent_lower for marker in ["analysis shows", "we predict", "expected to", "forecasted"]):
                    insights.append(sent.text)
            
            elif document_type == self.DOC_TYPE_RISK_ASSESSMENT:
                if any(marker in sent_lower for marker in ["risk indicator", "red flag", "warning sign", "suspicious"]):
                    insights.append(sent.text)
        
        # Deduplicate and limit insights
        unique_insights = list(set(insights))
        summary["key_insights"] = unique_insights[:7]  # Limit to 7 insights
        
        # Extract main topics (naive approach - most common noun phrases)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        noun_phrase_counter = Counter(noun_phrases)
        most_common_phrases = noun_phrase_counter.most_common(10)
        
        summary["main_topics"] = [phrase for phrase, count in most_common_phrases if count > 1]
        
        # Determine document purpose based on type and content
        if document_type == self.DOC_TYPE_WHITEPAPER:
            summary["document_purpose"] = "Describes a cryptocurrency or blockchain project's concept, technology, and value proposition"
        elif document_type == self.DOC_TYPE_AGREEMENT:
            summary["document_purpose"] = "Establishes legal terms and conditions for a cryptocurrency transaction or relationship"
        elif document_type == self.DOC_TYPE_REGULATION:
            summary["document_purpose"] = "Provides regulatory guidance or requirements for virtual assets"
        elif document_type == self.DOC_TYPE_LEGAL_CASE:
            summary["document_purpose"] = "Documents a legal proceeding involving cryptocurrencies or blockchain"
        elif document_type == self.DOC_TYPE_FINANCIAL_ANALYSIS:
            summary["document_purpose"] = "Analyzes cryptocurrency market trends, performance, or investment opportunities"
        elif document_type == self.DOC_TYPE_RISK_ASSESSMENT:
            summary["document_purpose"] = "Identifies money laundering, fraud, or other risks related to virtual assets"
        else:
            summary["document_purpose"] = "Provides information related to cryptocurrencies or blockchain technology"
        
        return summary
    
    def export_graph(self, output_file: str, format: str = 'gexf') -> bool:
        """
        Export the relationship graph to a file format that can be visualized.
        
        Args:
            output_file: Path to output file
            format: Output format (gexf, graphml, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format == 'gexf':
                nx.write_gexf(self.entity_graph, output_file)
            elif format == 'graphml':
                nx.write_graphml(self.entity_graph, output_file)
            else:
                logger.error(f"Unsupported graph format: {format}")
                return False
                
            logger.info(f"Exported relationship graph to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting relationship graph: {str(e)}")
            return False