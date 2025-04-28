import os
import re
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
from Code.document_processing.tf_idf_processor import TFIDFProcessor

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
            self.nlp = spacy.load("en_core_web_md")  # Use medium model for better performance
            logger.info("Loaded en_core_web_md NLP model")
        except OSError:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded en_core_web_sm NLP model")
        
        # Add sentencizer to the pipeline to fix the sentence boundaries issue
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')
            logger.info("Added sentencizer to the NLP pipeline")
        
        # Configure GPU usage if available
        if use_gpu and spacy.prefer_gpu():
            logger.info("Using GPU acceleration for NLP tasks")

        # Initialize TF-IDF processor
        self.tfidf_processor = TFIDFProcessor()
        
        # Initialize entity extraction patterns
        self._init_cryptocurrency_patterns()
        self._init_regulatory_patterns()
        self._init_financial_patterns()
        self._init_risk_indicators()
        
        # Initialize document classification patterns
        self._init_document_classifiers()
        
        # Initialize relationship extraction
        self.entity_graph = nx.DiGraph()

    def map_document_type_for_storage(self, internal_type: str) -> str:
        """
        Map internal document type constants to consistent storage types.
        
        Args:
            internal_type (str): Internal document type constant
            
        Returns:
            str: Storage-compatible document type
        """
        type_map = {
            "whitepaper": "whitepaper",
            "agreement": "project_documentation",
            "regulation": "regulatory_filing",
            "legal_case": "regulatory_filing",
            "financial_analysis": "due_diligence_report",
            "risk_assessment": "due_diligence_report",
            "other": "project_documentation"
        }
        normalized_type = internal_type.lower() if internal_type else "other"
        mapped_type = type_map.get(normalized_type, "project_documentation")
        logger.debug(f"Mapping document type: {internal_type} -> {mapped_type}")
        return mapped_type

    def _init_cryptocurrency_patterns(self):
        """Initialize patterns for cryptocurrency entity recognition"""
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
        self.crypto_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(text) for text in self.crypto_terms]
        self.crypto_matcher.add("CRYPTO_TERMS", patterns)
        self.btc_address_pattern = re.compile(r'\b(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b')
        self.eth_address_pattern = re.compile(r'\b0x[a-fA-F0-9]{40}\b')
        self.txid_pattern = re.compile(r'\b[a-fA-F0-9]{64}\b')

    def _init_regulatory_patterns(self):
        """Initialize patterns for regulatory entities and terms"""
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
        self.regulatory_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        reg_patterns = [self.nlp.make_doc(text) for text in self.regulatory_bodies + self.regulatory_terms]
        self.regulatory_matcher.add("REGULATORY_TERMS", reg_patterns)

    def _init_financial_patterns(self):
        """Initialize patterns for financial metrics and terms"""
        self.financial_terms = [
            "market cap", "market capitalization", "volume", "liquidity", "volatility",
            "price", "exchange rate", "trading volume", "transaction volume", "daily volume",
            "TVL", "total value locked", "revenue", "profit", "loss", "fee", "commission",
            "return on investment", "ROI", "APY", "annual percentage yield",
            "stake", "staking", "yield", "dividend", "interest", "APR", 
            "annual percentage rate", "gas fee", "network fee", "miner fee"
        ]
        self.financial_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        fin_patterns = [self.nlp.make_doc(text) for text in self.financial_terms]
        self.financial_matcher.add("FINANCIAL_TERMS", fin_patterns)
        self.usd_pattern = re.compile(r'\$\s*\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s*(?:USD|dollars|US dollars)', re.IGNORECASE)
        self.crypto_value_pattern = re.compile(r'\d+(?:\.\d+)?\s*(?:BTC|ETH|XRP|LTC|BCH|ADA|DOT|SOL|BNB)', re.IGNORECASE)
        self.amount_pattern = re.compile(r'([0-9]+(?:[,.][0-9]+)*(?:\.[0-9]+)?)\s*(?:million|billion|trillion)?', re.IGNORECASE)
        self.million_billion_pattern = re.compile(r'million|billion|trillion', re.IGNORECASE)

    def _init_risk_indicators(self):
        """Initialize patterns for risk indicators in crypto documents"""
        self.risk_indicators = [
            "multiple transactions", "structuring", "smurfing", "layering", "mixing", "tumbling",
            "obfuscate", "anonymous", "privacy coin", "mixer", "tumbler", "P2P platform",
            "darknet", "dark web", "suspicious", "unusual pattern", "shell company",
            "unregistered", "unlicensed", "sanctioned", "fraud", "scam", "ponzi", "pyramid scheme",
            "ransomware", "extortion", "hacked", "stolen funds", "illicit", "illegal", "criminal",
            "refused to provide", "false identification", "forged documents", "front company",
            "no economic purpose", "high-risk jurisdiction", "tax haven", "non-cooperative",
            "politically exposed person", "PEP", "high-risk customer", "high-risk country",
            "sanctioned country", "non-compliant", "terrorism", "terrorist financing",
            "drug trafficking", "human trafficking", "arms trafficking"
        ]
        self.risk_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        risk_patterns = [self.nlp.make_doc(text) for text in self.risk_indicators]
        self.risk_matcher.add("RISK_INDICATORS", risk_patterns)

    def _init_document_classifiers(self):
        """Initialize patterns for document classification"""
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
        max_nlp_length = 50000  # Reduced for better performance
        if len(text) > max_nlp_length:
            logger.info(f"Document length ({len(text)} chars) exceeds NLP processing limit, truncating")
            first_part = text[:max_nlp_length // 2]
            last_part = text[-max_nlp_length // 2:]
            nlp_text = first_part + " " + last_part
        else:
            nlp_text = text
        
        # Process with NLP, using only necessary components
        # Note: We're now NOT disabling the parser since we need it for sentence boundaries
        try:
            doc = self.nlp(nlp_text)
        except Exception as e:
            logger.error(f"Error in NLP processing: {e}")
            # Fallback approach - use sentencizer directly
            doc = self.nlp.make_doc(nlp_text)
            self.nlp.get_pipe('sentencizer')(doc)
        
        # Classify document type if not provided
        if not document_type:
            document_type = self._classify_document_type(doc, filename)

        # Map to standard storage type
        storage_document_type = self.map_document_type_for_storage(document_type)
            
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
            "document_type": storage_document_type,
            "original_document_type": document_type,
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
        Classify the document type based on content patterns with TF-IDF enhancement.
        
        Args:
            doc: spaCy processed document
            filename (str, optional): Source filename
            
        Returns:
            str: Classified document type
        """
        doc_text = doc.text.lower()
        
        # Check filename first (keep existing code)
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
        
        # Extract TF-IDF keywords
        tfidf_keywords = self.tfidf_processor.extract_keywords(doc_text, max_keywords=20)
        
        # Initialize scores
        type_scores = {doc_type: 0 for doc_type in self.doc_type_patterns.keys()}
        
        # Score document types based on TF-IDF keywords
        for keyword in tfidf_keywords:
            for doc_type, terms in self.doc_type_patterns.items():
                if any(term.lower() in keyword.lower() for term in terms):
                    type_scores[doc_type] += 1
        
        # Also use pattern matching scores (keep existing code)
        for doc_type, matcher in self.doc_type_matchers.items():
            matches = matcher(doc)
            type_scores[doc_type] += len(matches)
        
        # Add specific checks (keep existing code)
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
        
        if max(type_scores.values()) > 0:
            return max(type_scores.items(), key=lambda x: x[1])[0]
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
        word_count = len(text.split())
        
        # Safely get sentence count - check if sents attribute is available
        try:
            sentence_count = len(list(doc.sents))
        except ValueError:
            # If sentences aren't available, estimate by splitting on periods
            sentence_count = len([s for s in text.split('.') if s.strip()])
        
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        doc_date = dates[0] if dates else None
        authors = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        unique_authors = list(set(authors))[:10]
        unique_organizations = list(set(organizations))[:10]
        reading_time_minutes = word_count / 200
        language = "en"
        source = filename if filename else "unknown"
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "reading_time_minutes": reading_time_minutes,
            "document_date": doc_date,
            "language": language,
            "source": source,
            "potential_authors": unique_authors,
            "potential_organizations": unique_organizations,
            "text": text[:5000]  # Store a preview of the text
        }

    def _extract_entities(self, doc: Any, document_type: str) -> Dict[str, List[str]]:
        """
        Extract entities based on document type efficiently.
        
        Args:
            doc: spaCy processed document
            document_type (str): Type of document
            
        Returns:
            Dict[str, List[str]]: Extracted entities by category
        """
        logger.debug("Starting entity extraction")
        entities = {
            "persons": set(),
            "organizations": set(),
            "locations": set(),
            "cryptocurrencies": set(),
            "blockchain_terms": set(),
            "defi_protocols": set(),
            "token_standards": set(),
            "nft_terms": set(),
            "stablecoins": set(),
            "scaling_solutions": set(),
            "financial_instruments": set(),
            "crypto_addresses": set(),
            "dates": set(),
            "monetary_values": set()
        }
        
        # Process in chunks
        chunk_size = 5000
        doc_text = doc.text
        chunks = [doc_text[i:i + chunk_size] for i in range(0, len(doc_text), chunk_size)]
        total_chunks = len(chunks)
        
        # Use a safer approach that doesn't rely on sent boundaries for entity extraction
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.debug(f"Processing entity chunk {i+1}/{total_chunks}")
            
            # Process the chunk with minimal pipeline components
            chunk_doc = self.nlp.make_doc(chunk)
            # Run the entity recognizer directly
            self.nlp.get_pipe('ner')(chunk_doc)
            
            # Extract named entities
            for ent in chunk_doc.ents:
                if ent.label_ == "PERSON":
                    entities["persons"].add(ent.text)
                elif ent.label_ == "ORG":
                    entities["organizations"].add(ent.text)
                elif ent.label_ == "GPE" or ent.label_ == "LOC":
                    entities["locations"].add(ent.text)
                elif ent.label_ == "DATE":
                    entities["dates"].add(ent.text)
                elif ent.label_ == "MONEY":
                    entities["monetary_values"].add(ent.text)
            
            # Extract cryptocurrency terms
            crypto_matches = self.crypto_matcher(chunk_doc)
            for match_id, start, end in crypto_matches:
                span = chunk_doc[start:end]
                if any(term in span.text.lower() for term in ["bitcoin", "ethereum", "ripple", "litecoin", "cardano", "binance", "solana", "polkadot"]):
                    entities["cryptocurrencies"].add(span.text)
                else:
                    entities["blockchain_terms"].add(span.text)
        
        # Extract crypto addresses from full text
        btc_addresses = self.btc_address_pattern.findall(doc_text)
        eth_addresses = self.eth_address_pattern.findall(doc_text)
        txids = self.txid_pattern.findall(doc_text)
        crypto_addresses = set(btc_addresses + eth_addresses)
        if crypto_addresses:
            entities["crypto_addresses"] = crypto_addresses
        
        if document_type in [self.DOC_TYPE_LEGAL_CASE, self.DOC_TYPE_RISK_ASSESSMENT]:
            entities["transaction_ids"] = set(txids)
        
        # Convert sets to lists
        entities = {key: list(value) for key, value in entities.items()}
        
        logger.debug("Completed entity extraction")
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
        
        # Use regex patterns directly on text for better reliability
        usd_amounts = self.usd_pattern.findall(text)
        crypto_amounts = self.crypto_value_pattern.findall(text)
        
        # Extract monetary entities from spaCy
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                metrics["mentioned_amounts"].append(ent.text)
        
        # Try to extract financial terms using phrasematcher
        financial_matches = self.financial_matcher(doc)
        
        # Safely process sentences if available
        try:
            for match_id, start, end in financial_matches:
                span = doc[start:end]
                term = span.text.lower()
                
                # Try to get sentence, but handle the case where sentences aren't available
                try:
                    sent = span.sent
                    sent_text = sent.text
                except (AttributeError, ValueError):
                    # Approximate sentence by taking surrounding context
                    start_idx = max(0, span.start_char - 100)
                    end_idx = min(len(text), span.end_char + 100)
                    sent_text = text[start_idx:end_idx]
                
                usd_in_sent = self.usd_pattern.findall(sent_text)
                if "market cap" in term or "market capitalization" in term:
                    if usd_in_sent:
                        metrics["market_values"]["market_cap"] = usd_in_sent[0]
                elif "volume" in term:
                    if usd_in_sent:
                        metrics["transaction_volumes"].append({"description": term, "amount": usd_in_sent[0]})
                elif "fee" in term:
                    if usd_in_sent:
                        metrics["fees"].append({"description": term, "amount": usd_in_sent[0]})
        except Exception as e:
            logger.warning(f"Error processing financial metrics: {e}")
            
        metrics["crypto_amounts"] = crypto_amounts
        
        # Extract large numbers from text directly with regex
        big_numbers = []
        amount_matches = self.amount_pattern.findall(text)
        million_billion_matches = self.million_billion_pattern.findall(text)
        
        if amount_matches and million_billion_matches:
            for i in range(min(len(amount_matches), len(million_billion_matches))):
                big_numbers.append(f"{amount_matches[i]} {million_billion_matches[i]}")
        
        metrics["large_amounts"] = list(set(big_numbers))
        return metrics

    def _extract_regulatory_mentions(self, doc: Any) -> Dict[str, Any]:
        """
        Extract regulatory mentions from the document efficiently.
        
        Args:
            doc: spaCy processed document
            
        Returns:
            Dict[str, Any]: Extracted regulatory information
        """
        logger.debug("Starting regulatory mentions extraction")
        regulatory_info = {
            "regulatory_bodies": set(),
            "regulatory_terms": set(),
            "compliance_requirements": set(),
            "jurisdictions": set()
        }
        
        # Process text directly to reduce dependency on sentence boundaries
        doc_text = doc.text
        
        # Extract regulatory bodies and terms using the matcher
        regulatory_matches = self.regulatory_matcher(doc)
        for match_id, start, end in regulatory_matches:
            term = doc[start:end].text
            term_lower = term.lower()
            if any(body.lower() in term_lower for body in self.regulatory_bodies):
                regulatory_info["regulatory_bodies"].add(term)
            else:
                regulatory_info["regulatory_terms"].add(term)
            
            # Try to get sentence context, but handle the case where sentences aren't available
            try:
                sent = doc[start:end].sent.text
                if any(keyword in sent.lower() for keyword in ["must", "required", "shall"]):
                    regulatory_info["compliance_requirements"].add(sent)
            except (AttributeError, ValueError):
                # If sentence boundaries aren't available, use regex to approximate
                for match in re.finditer(r'[^.!?]*(?:must|required|shall)[^.!?]*[.!?]', doc_text, re.IGNORECASE):
                    regulatory_info["compliance_requirements"].add(match.group(0).strip())
        
        # Extract jurisdictions from entities
        for ent in doc.ents:
            if ent.label_ == "GPE":
                regulatory_info["jurisdictions"].add(ent.text)
        
        # Convert sets to lists
        regulatory_info = {key: list(value) for key, value in regulatory_info.items()}
        
        logger.debug("Completed regulatory mentions extraction")
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
        
        doc_text = doc.text
        
        # Extract risk indicators using the matcher
        risk_matches = self.risk_matcher(doc)
        for match_id, start, end in risk_matches:
            span = doc[start:end]
            term = span.text
            risk_info["risk_indicators"].append(term)
            term_lower = term.lower()
            
            if any(activity in term_lower for activity in ["fraud", "scam", "ponzi", "ransomware", "extortion", "illicit", "illegal"]):
                risk_info["high_risk_activities"].append(term)
                
            if any(pattern in term_lower for pattern in ["structuring", "smurfing", "layering", "mixing", "tumbling", "suspicious", "unusual"]):
                # Try to get sentence context if available
                try:
                    context = span.sent.text
                except (AttributeError, ValueError):
                    # If sentence boundaries aren't available, use regex to approximate
                    start_idx = max(0, span.start_char - 100)
                    end_idx = min(len(doc_text), span.end_char + 100)
                    context = doc_text[start_idx:end_idx]
                
                risk_info["suspicious_patterns"].append({"term": term, "context": context})
        
        # Extract high-risk jurisdictions through regex patterns rather than sentence iteration
        high_risk_terms = ["high-risk", "high risk", "sanctioned", "non-cooperative"]
        for term in high_risk_terms:
            for match in re.finditer(r'[^.!?]*' + re.escape(term) + r'[^.!?]*[.!?]', doc_text, re.IGNORECASE):
                context = match.group(0).strip()
                # Look for country/jurisdiction names within this context
                for ent in doc.ents:
                    if ent.label_ == "GPE" and ent.text in context:
                        risk_info["mentioned_jurisdictions"].append({"jurisdiction": ent.text, "context": context})
        
        # Deduplicate results
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
        
        doc_text = doc.text
        temp_graph = nx.DiGraph()
        
        # Get all key entities
        persons = entities.get("persons", [])
        organizations = entities.get("organizations", [])
        cryptocurrencies = entities.get("cryptocurrencies", [])
        all_key_entities = persons + organizations + cryptocurrencies
        
        # Add nodes to graph
        for entity in all_key_entities:
            temp_graph.add_node(entity)
        
        # Connect entities based on co-occurrence in context windows
        # This approach doesn't rely on sentence boundaries
        context_window = 150  # characters
        
        # Create a mapping of entity to its positions in the text
        entity_positions = {}
        for entity in all_key_entities:
            # Find all occurrences of this entity in the text
            for match in re.finditer(re.escape(entity), doc_text, re.IGNORECASE):
                if entity not in entity_positions:
                    entity_positions[entity] = []
                entity_positions[entity].append((match.start(), match.end()))
        
        # Connect entities that appear close to each other
        for entity1, positions1 in entity_positions.items():
            for pos1_start, pos1_end in positions1:
                # Define the context window around this entity
                context_start = max(0, pos1_start - context_window)
                context_end = min(len(doc_text), pos1_end + context_window)
                
                # Check which other entities appear in this context window
                for entity2, positions2 in entity_positions.items():
                    if entity1 == entity2:
                        continue  # Skip self
                        
                    for pos2_start, pos2_end in positions2:
                        # Check if the second entity is within the context window of the first
                        if (pos2_start >= context_start and pos2_start <= context_end) or \
                           (pos2_end >= context_start and pos2_end <= context_end):
                            # Entities co-occur in this context window
                            if temp_graph.has_edge(entity1, entity2):
                                temp_graph[entity1][entity2]["weight"] += 1
                            else:
                                temp_graph.add_edge(entity1, entity2, weight=1)
        
        # Extract strong relationships
        strong_relationships = []
        for u, v, data in temp_graph.edges(data=True):
            if data["weight"] > 1:
                strong_relationships.append({
                    "source": u,
                    "target": v,
                    "strength": data["weight"]
                })
        
        # Sort by strength
        strong_relationships.sort(key=lambda x: x["strength"], reverse=True)
        
        # Add to global entity graph
        for rel in strong_relationships:
            source = rel["source"]
            target = rel["target"]
            strength = rel["strength"]
            
            # Add nodes if they don't exist
            if not self.entity_graph.has_node(source):
                self.entity_graph.add_node(source)
            if not self.entity_graph.has_node(target):
                self.entity_graph.add_node(target)
                
            # Add or update edge
            if self.entity_graph.has_edge(source, target):
                self.entity_graph[source][target]["weight"] += strength
            else:
                self.entity_graph.add_edge(source, target, weight=strength)
        
        # Store results
        relationships["entity_pairs"] = strong_relationships
        
        # Calculate centrality
        centrality = {}
        if len(temp_graph.nodes()) > 0:
            degree_centrality = nx.degree_centrality(temp_graph)
            for entity, centrality_score in degree_centrality.items():
                centrality[entity] = centrality_score
                
        relationships["connection_strength"] = centrality
        return relationships

    def _generate_summary(self, doc: Any, document_type: str) -> Dict[str, Any]:
        """
        Extract key insights through summarization with TF-IDF enhancement.
        
        Args:
            doc: spaCy processed document
            document_type (str): Type of document
            
        Returns:
            Dict[str, Any]: Document summary and key insights
        """
        logger.debug(f"Generating summary for document type: {document_type}")
        
        doc_text = doc.text
        
        summary = {
            "key_sentences": [],
            "key_insights": [],
            "main_topics": [],
            "document_purpose": ""
        }
        
        # Update TF-IDF corpus with this document
        self.tfidf_processor.update_corpus(doc_text)
        
        # Extract keywords using TF-IDF
        tfidf_keywords = self.tfidf_processor.extract_keywords(doc_text, max_keywords=15)
        
        # Use regex to split text into sentences, avoiding dependency on spaCy's sentence segmentation
        sentence_pattern = re.compile(r'[^.!?]+[.!?]')
        sentences = [match.group(0).strip() for match in sentence_pattern.finditer(doc_text)]
        
        # If no sentences were found (unlikely), fall back to simple splitting
        if not sentences:
            sentences = [s.strip() + '.' for s in doc_text.split('.') if s.strip()]
        
        # Important sentences are often at the beginning and end of documents
        important_sentences = []
        if len(sentences) > 3:
            important_sentences.extend(sentences[:3])  # First 3 sentences
            important_sentences.extend(sentences[-2:])  # Last 2 sentences
        else:
            important_sentences.extend(sentences)  # All sentences if less than 3
        
        # Add sentences containing key phrases based on document type
        keyword_patterns = []
        
        if document_type == self.DOC_TYPE_WHITEPAPER:
            keyword_patterns.extend([
                r'[^.!?]*\b(?:purpose|vision|mission|objective|goal|aims to|designed to)\b[^.!?]*[.!?]'
            ])
        elif document_type == self.DOC_TYPE_AGREEMENT:
            keyword_patterns.extend([
                r'[^.!?]*\b(?:agree|terms|condition|party|obligation|shall|must|covenant)\b[^.!?]*[.!?]'
            ])
        elif document_type == self.DOC_TYPE_REGULATION:
            keyword_patterns.extend([
                r'[^.!?]*\b(?:require|prohibit|must|shall|regulate|compliance|subject to)\b[^.!?]*[.!?]'
            ])
        elif document_type == self.DOC_TYPE_LEGAL_CASE:
            keyword_patterns.extend([
                r'[^.!?]*\b(?:court|ruled|decision|judgment|opinion|held|concluded)\b[^.!?]*[.!?]'
            ])
        elif document_type == self.DOC_TYPE_FINANCIAL_ANALYSIS:
            keyword_patterns.extend([
                r'[^.!?]*\b(?:analysis|shows|indicates|concludes|predicts|projects|estimates)\b[^.!?]*[.!?]'
            ])
        elif document_type == self.DOC_TYPE_RISK_ASSESSMENT:
            keyword_patterns.extend([
                r'[^.!?]*\b(?:risk|warning|indicator|flag|suspicious|monitor|detect)\b[^.!?]*[.!?]'
            ])
        
        # Generic patterns for important sentences (those with multiple entities or key terms)
        keyword_patterns.extend([
            r'[^.!?]*\b(?:key|important|significant|critical|essential)\b[^.!?]*[.!?]',
            r'[^.!?]*\b(?:conclusion|summary|in summary|to summarize|in conclusion)\b[^.!?]*[.!?]'
        ])
        
        # Find sentences matching patterns
        for pattern in keyword_patterns:
            for match in re.finditer(pattern, doc_text, re.IGNORECASE):
                important_sentences.append(match.group(0).strip())
        
        # Add sentences that contain TF-IDF keywords
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in tfidf_keywords[:5]):
                important_sentences.append(sentence)
        
        # Score sentences based on TF-IDF keyword presence
        scored_sentences = []
        for sentence in important_sentences:
            # Count keyword occurrences in sentence
            keyword_count = sum(1 for keyword in tfidf_keywords if keyword.lower() in sentence.lower())
            scored_sentences.append((sentence, keyword_count))
        
        # Sort by score and deduplicate
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        unique_important_sentences = []
        seen_sentences = set()
        for sentence, _ in scored_sentences:
            # Use a simple deduplication approach
            simplified = re.sub(r'\s+', ' ', sentence.lower())
            if simplified not in seen_sentences:
                unique_important_sentences.append(sentence)
                seen_sentences.add(simplified)
        
        # Use the top sentences as key sentences
        summary["key_sentences"] = unique_important_sentences[:5]
        
        # Extract key insights using TF-IDF important sentences
        insights = unique_important_sentences[:7]  # Top sentences are likely key insights
        
        # Add additional insights using the same approach as key sentences but with different patterns
        insight_patterns = [
            r'[^.!?]*\b(?:provide|enable|solution|problem|innovation)\b[^.!?]*[.!?]',
            r'[^.!?]*\b(?:obligation|right|must|shall|agree to)\b[^.!?]*[.!?]',
            r'[^.!?]*\b(?:require|regulate|compliance|prohibited)\b[^.!?]*[.!?]',
            r'[^.!?]*\b(?:court held|ruling|decided|judgment|conclusion)\b[^.!?]*[.!?]',
            r'[^.!?]*\b(?:analysis shows|we predict|expected to|forecasted)\b[^.!?]*[.!?]',
            r'[^.!?]*\b(?:risk indicator|red flag|warning sign|suspicious)\b[^.!?]*[.!?]'
        ]
        
        for pattern in insight_patterns:
            for match in re.finditer(pattern, doc_text, re.IGNORECASE):
                insights.append(match.group(0).strip())
        
        # Deduplicate insights
        unique_insights = []
        seen_insights = set()
        for insight in insights:
            simplified = re.sub(r'\s+', ' ', insight.lower())
            if simplified not in seen_insights:
                unique_insights.append(insight)
                seen_insights.add(simplified)
        
        summary["key_insights"] = unique_insights[:7]
        
        # Use TF-IDF keywords as main topics
        summary["main_topics"] = tfidf_keywords[:10]
        
        # Set document purpose based on document type (keep existing code)
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
                
        logger.debug(f"Extracted insights: {summary['key_insights']}")
        return summary
    
    def find_similar_documents(self, query_text: str, documents: List[Dict[str, Any]], 
                            text_key: str = 'content', top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar documents to a query text using TF-IDF similarity.
        
        Args:
            query_text (str): Query text
            documents (List[Dict]): List of document dictionaries
            text_key (str): Key for accessing document text
            top_n (int): Number of top results to return
            
        Returns:
            List[Dict]: List of similar documents with similarity scores
        """
        return self.tfidf_processor.find_similar_documents(query_text, documents, text_key, top_n)

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
            # Make sure the output directory exists
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
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