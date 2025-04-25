#!/usr/bin/env python
"""
Document Analyzer Module - Enhances the Q&A system by providing deep analysis of user documents
and connecting them with relevant data from Weaviate collections.
"""

import os
import sys
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import from your existing code
from Sample_Data.vector_store.storage_manager import StorageManager
from Sample_Data.vector_store.embed import generate_mpnet_embedding

class DocumentAnalyzer:
    """
    Advanced document analyzer that:
    1. Retrieves document content from Weaviate
    2. Performs deep analysis on document content
    3. Connects document topics with other relevant information in collections
    4. Provides comprehensive context for the Q&A system
    """
    
    def __init__(self):
        """Initialize the document analyzer"""
        self.storage = StorageManager()
        
    def connect(self):
        """Connect to storage"""
        return self.storage.connect()
    
    def close(self):
        """Close storage connection"""
        if self.storage:
            self.storage.close()
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """
        Retrieve full document by ID directly from Weaviate.
        
        Args:
            document_id: The document ID to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            if not self.connect():
                logger.error("Failed to connect to Weaviate")
                return None
                
            logger.info(f"Attempting to retrieve document with ID: {document_id}")
            
            # Get collections to search in
            collections = ["UserDocuments"]
            
            from weaviate.classes.query import Filter
            
            for collection_name in collections:
                try:
                    collection = self.storage.client.collections.get(collection_name)
                    
                    # Try looking up by ID first
                    try:
                        # Log attempt to retrieve by ID
                        logger.info(f"Attempting direct ID lookup for: {document_id}")
                        object_by_id = collection.data.get_by_id(document_id)
                        
                        if object_by_id:
                            logger.info(f"Successfully found document by direct ID lookup")
                            document_data = {
                                "id": document_id,
                                "collection": collection_name,
                                **object_by_id.properties
                            }
                            
                            # Log content length to verify content is retrieved
                            content_length = len(document_data.get("content", ""))
                            logger.info(f"Retrieved document content length: {content_length}")
                            
                            return document_data
                    except Exception as e:
                        logger.warning(f"Error getting document by direct ID: {e}")
                    
                    # Try filter-based lookup with UUID
                    try:
                        logger.info(f"Attempting filter-based UUID lookup for: {document_id}")
                        response = collection.query.fetch_objects(
                            filters=Filter.by_id().equal(document_id),
                            limit=1
                        )
                        
                        if response.objects:
                            obj = response.objects[0]
                            logger.info(f"Found document via UUID filter")
                            document_data = {
                                "id": str(obj.uuid),
                                "collection": collection_name,
                                **obj.properties
                            }
                            
                            # Log content length to verify content is retrieved
                            content_length = len(document_data.get("content", ""))
                            logger.info(f"Retrieved document content length: {content_length}")
                            
                            return document_data
                    except Exception as e:
                        logger.warning(f"Error with UUID filter lookup: {e}")
                    
                    # Try filter by title or source
                    try:
                        logger.info(f"Attempting property filter lookup (title/source)")
                        response = collection.query.fetch_objects(
                            filters=(Filter.by_property("title").equal(document_id) | 
                                    Filter.by_property("source").equal(document_id)),
                            limit=1
                        )
                        
                        if response.objects:
                            obj = response.objects[0]
                            logger.info(f"Found document via property filter")
                            document_data = {
                                "id": str(obj.uuid),
                                "collection": collection_name,
                                **obj.properties
                            }
                            
                            # Log content length to verify content is retrieved
                            content_length = len(document_data.get("content", ""))
                            logger.info(f"Retrieved document content length: {content_length}")
                            
                            return document_data
                    except Exception as e:
                        logger.warning(f"Error with property filter lookup: {e}")
                    
                    # Last resort: list all documents to find a match
                    try:
                        logger.info("Last resort: listing all documents to find a match")
                        all_docs = collection.query.fetch_objects(limit=20)
                        
                        logger.info(f"Found {len(all_docs.objects)} documents in collection")
                        
                        # Log all document IDs for debugging
                        for doc in all_docs.objects:
                            logger.info(f"Document in collection: {doc.uuid}")
                            
                            # Check if this document matches our target ID
                            if str(doc.uuid) == document_id:
                                logger.info(f"Found matching document: {doc.uuid}")
                                document_data = {
                                    "id": str(doc.uuid),
                                    "collection": collection_name,
                                    **doc.properties
                                }
                                
                                # Log content length to verify content is retrieved
                                content_length = len(document_data.get("content", ""))
                                logger.info(f"Retrieved document content length: {content_length}")
                                
                                return document_data
                    except Exception as e:
                        logger.warning(f"Error listing all documents: {e}")
                        
                except Exception as e:
                    logger.error(f"Error querying {collection_name}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            logger.warning(f"Document not found after all attempts: {document_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def analyze_document(self, document: Dict) -> Dict:
        """
        Perform deep analysis on document content.
        
        Args:
            document: The document data from Weaviate
            
        Returns:
            Analysis results
        """
        analysis = {
            "document_info": self._extract_document_info(document),
            "content_summary": self._summarize_content(document),
            "entities": self._extract_entities(document),
            "key_topics": self._extract_key_topics(document),
            "risk_assessment": self._analyze_risk_factors(document),
            "dates_mentioned": self._extract_dates(document),
            "funds_mentioned": self._extract_funds(document),
            "crypto_assets": self._extract_crypto_assets(document)
        }
        
        return analysis
    
    def find_related_information(self, document: Dict, analysis: Dict) -> Dict:
        """
        Find related information in other collections based on document analysis.
        
        Args:
            document: The document data
            analysis: Document analysis results
            
        Returns:
            Related information from collections
        """
        related_info = {}
        
        try:
            # Find related market data for mentioned crypto assets
            if analysis["crypto_assets"]:
                related_info["market_data"] = self._find_related_market_data(analysis["crypto_assets"])
            
            # Find related sentiment data
            if analysis["crypto_assets"]:
                related_info["sentiment_data"] = self._find_related_sentiment_data(analysis["crypto_assets"])
            
            # Find related risk information
            if analysis["risk_assessment"]:
                related_info["risk_data"] = self._find_related_risk_data(analysis["risk_assessment"])
            
            # Find related documents
            if analysis["key_topics"]:
                related_info["related_documents"] = self._find_related_documents(document, analysis["key_topics"])
                
            # Check for regulatory information if it seems like a regulatory document
            if "regulations" in analysis["key_topics"] or "compliance" in analysis["key_topics"]:
                related_info["regulatory_info"] = self._find_regulatory_information(analysis["key_topics"])
            
            return related_info
            
        except Exception as e:
            logger.error(f"Error finding related information: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def get_comprehensive_context(self, document_id: str, question: str) -> Dict:
        """
        Get comprehensive context about a document for answering a question.
        
        Args:
            document_id: The document ID
            question: The user's question
            
        Returns:
            Comprehensive context for answering the question
        """
        try:
            # Retrieve the document
            logger.info(f"Getting comprehensive context for document: {document_id}")
            document = self.get_document_by_id(document_id)
            
            if not document:
                logger.error(f"Document not found: {document_id}")
                return {
                    "error": f"Document not found: {document_id}",
                    "context": "No document information available."
                }
            
            # Log document properties to verify retrieval
            logger.info(f"Retrieved document: {document.get('title', 'Untitled')}")
            
            # Check if content exists and log its length
            content = document.get('content', '')
            logger.info(f"Document content length: {len(content)}")
            
            if not content:
                logger.warning("Document content is empty!")
            
            # Perform document analysis
            analysis = self.analyze_document(document)
            
            # Find related information from collections
            related_info = self.find_related_information(document, analysis)
            
            # Extract relevant parts based on the question
            relevant_parts = self._extract_relevant_parts(document, analysis, related_info, question)
            
            # Format the context
            context = self._format_context(document, analysis, related_info, relevant_parts)
            
            return {
                "document": document,
                "analysis": analysis,
                "related_info": related_info,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive context: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Error analyzing document: {str(e)}",
                "context": "An error occurred while analyzing the document."
            }
    
    def _extract_document_info(self, document: Dict) -> Dict:
        """Extract basic document information"""
        doc_info = {
            "title": document.get("title", "Untitled Document"),
            "document_type": document.get("document_type", "Unknown"),
            "upload_date": document.get("upload_date", "Unknown"),
            "file_type": document.get("file_type", "Unknown"),
            "file_size": document.get("file_size", 0),
            "word_count": document.get("word_count", 0),
            "sentence_count": document.get("sentence_count", 0),
            "is_public": document.get("is_public", False),
            "processing_status": document.get("processing_status", "Unknown")
        }
        
        # Add date if available
        if "date" in document:
            doc_info["document_date"] = document["date"]
        
        return doc_info
    
    def _summarize_content(self, document: Dict) -> str:
        """Generate a summary of document content"""
        content = document.get("content", "")
        
        if not content:
            return "No content available for summarization."
        
        # Simple summary approach - extract first paragraph and key sentences
        paragraphs = content.split('\n\n')
        first_paragraph = paragraphs[0] if paragraphs else ""
        
        # Extract some key sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        important_sentences = []
        
        # Look for sentences with important keywords
        important_keywords = ["key", "important", "critical", "significant", "risk", 
                            "opportunity", "strategy", "recommend", "conclusion", 
                            "summary", "result", "finding", "analysis"]
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                important_sentences.append(sentence)
                
                # Limit to 5 important sentences
                if len(important_sentences) >= 5:
                    break
        
        # Combine first paragraph and important sentences
        summary = first_paragraph
        
        if important_sentences:
            summary += "\n\nKey points:\n- " + "\n- ".join(important_sentences)
        
        return summary
    
    def _extract_entities(self, document: Dict) -> Dict:
        """Extract entities from document"""
        entities = {}
        
        # Use pre-extracted entities if available
        if "org_entities" in document:
            entities["organizations"] = document["org_entities"]
        
        if "person_entities" in document:
            entities["persons"] = document["person_entities"]
        
        if "location_entities" in document:
            entities["locations"] = document["location_entities"]
        
        if "crypto_entities" in document:
            entities["cryptocurrencies"] = document["crypto_entities"]
        
        # If no entities are available, try to extract from content
        if not entities and "content" in document:
            entities = self._extract_entities_from_text(document["content"])
        
        return entities
    
    def _extract_entities_from_text(self, text: str) -> Dict:
        """Extract entities from text using simple regex patterns"""
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "cryptocurrencies": []
        }
        
        # Simple pattern matching for cryptocurrencies
        crypto_patterns = [
            r'\b(bitcoin|btc|ethereum|eth|solana|sol|binance|bnb|cardano|ada|ripple|xrp)\b',
            r'\b[A-Z]{3,5}\b'  # Potential crypto symbols
        ]
        
        for pattern in crypto_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                crypto = match.group(0)
                if crypto not in entities["cryptocurrencies"]:
                    entities["cryptocurrencies"].append(crypto)
        
        # Basic org pattern (may need improvement)
        org_patterns = [
            r'\b([A-Z][a-z]+\s+(?:Inc|LLC|Corp|Corporation|Foundation|Association|Fund|Capital))\b',
            r'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)+)\s+(?:Inc|LLC|Corp|Foundation)\b'
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                org = match.group(0)
                if org not in entities["organizations"]:
                    entities["organizations"].append(org)
        
        return entities
    
    def _extract_key_topics(self, document: Dict) -> List[str]:
        """Extract key topics from document"""
        # Use pre-extracted keywords if available
        if "keywords" in document and document["keywords"]:
            return document["keywords"]
        
        # Otherwise extract from content
        topics = []
        content = document.get("content", "").lower()
        
        # Define common crypto and finance related topics
        common_topics = [
            "blockchain", "cryptocurrency", "token", "risk", "investment", "regulation",
            "compliance", "security", "investor", "trading", "exchange", "liquidity",
            "market", "volatility", "governance", "defi", "nft", "staking", "mining",
            "wallet", "smart contract", "consensus", "protocol", "yield", "apy", "apr",
            "tokenomics", "ico", "airdrop", "whitepaper", "roadmap", "audit", "hash",
            "kyc", "aml", "tax", "legal", "jurisdiction", "license", "custody"
        ]
        
        for topic in common_topics:
            if topic in content:
                topics.append(topic)
        
        return topics
    
    def _analyze_risk_factors(self, document: Dict) -> List[str]:
        """Analyze risk factors in the document"""
        # Use pre-extracted risk factors if available
        if "risk_factors" in document and document["risk_factors"]:
            return document["risk_factors"]
        
        risk_factors = []
        content = document.get("content", "").lower()
        
        # Define common risk-related terms and patterns
        risk_terms = [
            "risk", "threat", "vulnerability", "exposure", "danger", "hazard",
            "uncertainty", "volatility", "concern", "issue", "problem", "challenge",
            "liability", "loss", "penalty", "fine", "sanction", "regulatory action",
            "lawsuit", "litigation", "compliance", "breach", "violation", "failure",
            "security", "hack", "attack", "exploit", "fraud", "scam", "theft"
        ]
        
        # Check for risk sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sentence in sentences:
            if any(term in sentence for term in risk_terms):
                # Clean up the sentence and add to risk factors
                risk_sentence = sentence.strip().capitalize()
                if risk_sentence and len(risk_sentence) > 10:  # Avoid very short matches
                    risk_factors.append(risk_sentence)
        
        # Limit to top 5 most descriptive risk factors
        risk_factors.sort(key=len, reverse=True)
        return risk_factors[:5]
    
    def _extract_dates(self, document: Dict) -> List[str]:
        """Extract dates mentioned in the document"""
        dates = []
        content = document.get("content", "")
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
            r'\bQ[1-4] \d{4}\b',  # Q1 2023
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b'  # Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                date = match.group(0)
                if date not in dates:
                    dates.append(date)
        
        return dates
    
    def _extract_funds(self, document: Dict) -> List[str]:
        """Extract fund names mentioned in the document"""
        funds = []
        content = document.get("content", "")
        
        # Fund name patterns
        fund_patterns = [
            r'\b[A-Z][a-zA-Z\s]+ (?:Fund|Capital|Partners|Investments|Group|Trust|ETF)\b',
            r'\b[A-Z][a-zA-Z\s]+ (?:LP|LLC|Inc|Limited)\b',
            r'\b(?:Crypto|Digital|Asset|Blockchain)[a-zA-Z\s]+ (?:Fund|Capital|Partners|Investments)\b'
        ]
        
        for pattern in fund_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                fund = match.group(0)
                if fund not in funds:
                    funds.append(fund)
        
        return funds
    
    def _extract_crypto_assets(self, document: Dict) -> List[str]:
        """Extract cryptocurrency assets mentioned in the document"""
        # Use pre-extracted crypto entities if available
        if "crypto_entities" in document and document["crypto_entities"]:
            return document["crypto_entities"]
        
        assets = []
        content = document.get("content", "").lower()
        
        # Common cryptocurrencies
        common_cryptos = {
            "bitcoin": ["bitcoin", "btc", "xbt"],
            "ethereum": ["ethereum", "eth", "ether"],
            "cardano": ["cardano", "ada"],
            "solana": ["solana", "sol"],
            "binance": ["binance coin", "bnb"],
            "ripple": ["ripple", "xrp"],
            "polkadot": ["polkadot", "dot"],
            "dogecoin": ["dogecoin", "doge"],
            "tether": ["tether", "usdt"],
            "usd coin": ["usd coin", "usdc"]
        }
        
        for crypto, variants in common_cryptos.items():
            if any(variant in content for variant in variants):
                assets.append(crypto)
        
        return assets
    
    def _find_related_market_data(self, crypto_assets: List[str]) -> List[Dict]:
        """Find related market data for mentioned crypto assets"""
        market_data = []
        
        for asset in crypto_assets:
            # Map asset name to symbol
            symbol_map = {
                "bitcoin": "BTCUSDT",
                "ethereum": "ETHUSDT",
                "cardano": "ADAUSDT",
                "solana": "SOLUSDT",
                "binance": "BNBUSDT",
                "ripple": "XRPUSDT",
                "polkadot": "DOTUSDT",
                "dogecoin": "DOGEUSDT"
            }
            
            symbol = symbol_map.get(asset.lower(), f"{asset.upper()}USDT")
            
            try:
                # Get market data from MarketMetrics
                data = self.storage.retrieve_market_data(symbol, limit=1)
                if data:
                    market_data.append({
                        "asset": asset,
                        "symbol": symbol,
                        "data": data[0] if isinstance(data, list) else data
                    })
            except Exception as e:
                logger.warning(f"Error retrieving market data for {asset}: {e}")
        
        return market_data
    
    def _find_related_sentiment_data(self, crypto_assets: List[str]) -> List[Dict]:
        """Find related sentiment data for mentioned crypto assets"""
        sentiment_data = []
        
        for asset in crypto_assets:
            try:
                # Get sentiment data
                data = self.storage.get_sentiment_stats(asset)
                if data and not isinstance(data, str) and "error" not in data:
                    sentiment_data.append({
                        "asset": asset,
                        "data": data
                    })
            except Exception as e:
                logger.warning(f"Error retrieving sentiment data for {asset}: {e}")
        
        return sentiment_data
    
    def _find_related_risk_data(self, risk_factors: List[str]) -> List[Dict]:
        """Find related risk information based on identified risk factors"""
        risk_data = []
        
        # Extract key risk terms
        risk_terms = []
        for factor in risk_factors:
            # Extract significant words from risk factors
            words = re.findall(r'\b[a-zA-Z]{4,}\b', factor.lower())
            for word in words:
                if word not in ["risk", "factor", "potential", "possible", "there"]:
                    risk_terms.append(word)
        
        if not risk_terms:
            return risk_data
        
        try:
            # Search for related risk information in CryptoDueDiligenceDocuments
            query = " ".join(risk_terms[:5]) + " risk factors"  # Limit to top 5 terms
            
            documents = self.storage.retrieve_documents(
                query=query,
                collection_name="CryptoDueDiligenceDocuments",
                limit=3
            )
            
            if documents:
                for doc in documents:
                    # Extract relevant parts about risk
                    content = doc.get("content", "")
                    risk_excerpts = []
                    
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    for sentence in sentences:
                        if any(term in sentence.lower() for term in risk_terms):
                            risk_excerpts.append(sentence.strip())
                    
                    if risk_excerpts:
                        risk_data.append({
                            "source": doc.get("title", "Unknown document"),
                            "excerpts": risk_excerpts[:3]  # Limit to 3 excerpts
                        })
        except Exception as e:
            logger.warning(f"Error finding related risk data: {e}")
        
        return risk_data
    
    def _find_related_documents(self, document: Dict, topics: List[str]) -> List[Dict]:
        """Find related documents based on key topics"""
        related_docs = []
        
        if not topics:
            return related_docs
        
        try:
            # Build a query from the top topics
            query = " ".join(topics[:5])  # Limit to top 5 topics
            
            # Get document ID to exclude current document
            current_id = document.get("id", "")
            
            # Query for similar documents
            similar_docs = self.storage.retrieve_documents(
                query=query,
                collection_name="UserDocuments",
                limit=3
            )
            
            if similar_docs:
                for doc in similar_docs:
                    # Skip the current document
                    if doc.get("id", "") == current_id:
                        continue
                    
                    related_docs.append({
                        "id": doc.get("id", ""),
                        "title": doc.get("title", "Untitled Document"),
                        "document_type": doc.get("document_type", "Unknown"),
                        "upload_date": doc.get("upload_date", "Unknown"),
                        "similarity_topics": [topic for topic in topics if topic in doc.get("content", "").lower()]
                    })
        except Exception as e:
            logger.warning(f"Error finding related documents: {e}")
        
        return related_docs
    
    def _find_regulatory_information(self, topics: List[str]) -> Dict:
        """Find relevant regulatory information based on topics"""
        regulatory_info = {
            "regulations": [],
            "compliance_notes": []
        }
        
        # Regulatory-related topics
        regulatory_topics = ["regulation", "compliance", "legal", "kyc", "aml", "license", "jurisdiction"]
        
        # Filter for regulatory topics
        relevant_topics = [topic for topic in topics if any(reg_topic in topic for reg_topic in regulatory_topics)]
        
        if not relevant_topics:
            return regulatory_info
        
        try:
            # Build a query from the regulatory topics
            query = " ".join(relevant_topics) + " cryptocurrency regulations compliance"
            
            # Search for regulatory information
            docs = self.storage.retrieve_documents(
                query=query,
                collection_name="CryptoDueDiligenceDocuments",
                limit=2
            )
            
            if docs:
                for doc in docs:
                    content = doc.get("content", "")
                    
                    # Extract relevant regulatory sentences
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    for sentence in sentences:
                        if any(topic in sentence.lower() for topic in relevant_topics):
                            if len(sentence) > 30:  # Avoid very short matches
                                regulatory_info["regulations"].append({
                                    "source": doc.get("title", "Unknown document"),
                                    "text": sentence.strip()
                                })
                    
                    # Limit to top 5 regulations
                    regulatory_info["regulations"] = regulatory_info["regulations"][:5]
        except Exception as e:
            logger.warning(f"Error finding regulatory information: {e}")
        
        return regulatory_info
    
    def _extract_relevant_parts(self, document: Dict, analysis: Dict, related_info: Dict, question: str) -> Dict:
        """Extract parts of document and analysis relevant to the question"""
        question_lower = question.lower()
        relevant_parts = {
            "document_sections": [],
            "entities": [],
            "risk_factors": [],
            "market_data": [],
            "sentiment_data": [],
            "regulatory_info": []
        }
        
        # Define categories of questions
        categories = {
            "content": ["what", "content", "about", "summary", "overview", "contain", "say", "mention", "describe", "explain"],
            "risk": ["risk", "danger", "concern", "issue", "problem", "threat", "vulnerability", "exposure"],
            "date": ["when", "date", "time", "period", "year", "month", "day"],
            "entity": ["who", "person", "organization", "company", "fund", "entity", "involved"],
            "crypto": ["cryptocurrency", "crypto", "bitcoin", "ethereum", "blockchain", "token", "asset"],
            "regulation": ["regulation", "compliance", "legal", "law", "requirement", "rule", "policy"],
            "market": ["market", "price", "value", "trend", "performance", "trade", "exchange"]
        }
        
        # Determine question categories
        question_categories = []
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                question_categories.append(category)
        
        # If no categories detected, default to content
        if not question_categories:
            question_categories = ["content"]
        
        # Extract document content sections based on question
        content = document.get("content", "")
        if content:
            if "content" in question_categories:
                # If asking about content, include summary and first paragraph
                relevant_parts["document_sections"].append({
                    "type": "summary",
                    "text": analysis["content_summary"]
                })
                
                # Add first paragraph
                paragraphs = content.split('\n\n')
                if paragraphs:
                    relevant_parts["document_sections"].append({
                        "type": "introduction",
                        "text": paragraphs[0]
                    })
            
            # If asking about specific topics, extract relevant sections
            for category in question_categories:
                if category == "risk" and analysis["risk_assessment"]:
                    relevant_parts["risk_factors"] = analysis["risk_assessment"]
                
                if category == "entity" and analysis["entities"]:
                    for entity_type, entities in analysis["entities"].items():
                        if entities:
                            relevant_parts["entities"].append({
                                "type": entity_type,
                                "items": entities
                            })
                
                if category == "date" and analysis["dates_mentioned"]:
                    relevant_parts["document_sections"].append({
                        "type": "dates",
                        "text": f"Dates mentioned in the document: {', '.join(analysis['dates_mentioned'])}"
                    })
                
                if category == "crypto" and analysis["crypto_assets"]:
                    # Include crypto assets
                    relevant_parts["document_sections"].append({
                        "type": "crypto_assets",
                        "text": f"Cryptocurrencies mentioned: {', '.join(analysis['crypto_assets'])}"
                    })
                    
                    # Include related market data
                    if "market_data" in related_info and related_info["market_data"]:
                        relevant_parts["market_data"] = related_info["market_data"]
                
                if category == "market" and "market_data" in related_info:
                    relevant_parts["market_data"] = related_info["market_data"]
                
                if category == "regulation" and "regulatory_info" in related_info:
                    relevant_parts["regulatory_info"] = related_info["regulatory_info"]
        
        # Look for specific question keywords in content
        question_words = re.findall(r'\b[a-zA-Z]{4,}\b', question_lower)
        question_words = [w for w in question_words if w not in ["what", "when", "where", "who", "why", "how", "does", "is", "are", "should", "could", "would", "will", "about", "there", "this", "that", "these", "those", "with", "from", "document", "goal", "risk", "market", "content", "analysis"]]
        
        if question_words and content:
            # Search for paragraphs containing question keywords
            paragraphs = content.split('\n\n')
            for paragraph in paragraphs:
                if any(word in paragraph.lower() for word in question_words):
                    relevant_parts["document_sections"].append({
                        "type": "relevant_paragraph",
                        "text": paragraph.strip()
                    })
                    
                    # Limit to 3 relevant paragraphs
                    if len([s for s in relevant_parts["document_sections"] if s["type"] == "relevant_paragraph"]) >= 3:
                        break
        
        # Include sentiment data if relevant
        if (("sentiment" in question_lower or "market" in question_categories or "crypto" in question_categories) and
            "sentiment_data" in related_info and related_info["sentiment_data"]):
            relevant_parts["sentiment_data"] = related_info["sentiment_data"]
        
        return relevant_parts
    
    def _format_context(self, document: Dict, analysis: Dict, related_info: Dict, relevant_parts: Dict) -> str:
        """Format the context for the LLM"""
        context = []
        
        # Document metadata
        context.append(f"DOCUMENT INFORMATION:")
        context.append(f"Title: {analysis['document_info'].get('title', 'Untitled Document')}")
        context.append(f"Document Type: {analysis['document_info'].get('document_type', 'Unknown')}")
        if 'document_date' in analysis['document_info']:
            context.append(f"Date: {analysis['document_info']['document_date']}")
        context.append("")
        
        # Add document sections
        if relevant_parts["document_sections"]:
            context.append("DOCUMENT CONTENT:")
            for section in relevant_parts["document_sections"]:
                if section["type"] == "summary":
                    context.append("Document Summary:")
                    context.append(section["text"])
                    context.append("")
                elif section["type"] == "introduction":
                    context.append("Document Introduction:")
                    context.append(section["text"])
                    context.append("")
                elif section["type"] == "relevant_paragraph":
                    context.append("Relevant Content:")
                    context.append(section["text"])
                    context.append("")
                elif section["type"] == "dates":
                    context.append(section["text"])
                    context.append("")
                elif section["type"] == "crypto_assets":
                    context.append(section["text"])
                    context.append("")
        
        # Add entities
        if relevant_parts["entities"]:
            context.append("ENTITIES MENTIONED:")
            for entity_group in relevant_parts["entities"]:
                context.append(f"{entity_group['type'].capitalize()}: {', '.join(entity_group['items'])}")
            context.append("")
        
        # Add risk factors
        if relevant_parts["risk_factors"]:
            context.append("RISK FACTORS:")
            for risk in relevant_parts["risk_factors"]:
                context.append(f"- {risk}")
            context.append("")
        
        # Add market data
        if relevant_parts["market_data"]:
            context.append("RELEVANT MARKET DATA:")
            for market_item in relevant_parts["market_data"]:
                asset = market_item["asset"].upper()
                data = market_item["data"]
                context.append(f"{asset} Market Data:")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key not in ["id", "_id", "timestamp", "vector"]:
                            context.append(f"- {key}: {value}")
                context.append("")
        
        # Add sentiment data
        if relevant_parts["sentiment_data"]:
            context.append("SENTIMENT ANALYSIS:")
            for sentiment_item in relevant_parts["sentiment_data"]:
                asset = sentiment_item["asset"].upper()
                data = sentiment_item["data"]
                context.append(f"{asset} Sentiment:")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key in ["overall_sentiment", "positive", "negative", "neutral"]:
                            context.append(f"- {key}: {value}")
                context.append("")
        
        # Add regulatory information
        if relevant_parts["regulatory_info"] and "regulations" in relevant_parts["regulatory_info"]:
            context.append("REGULATORY INFORMATION:")
            for reg in relevant_parts["regulatory_info"]["regulations"]:
                context.append(f"- {reg['text']} (Source: {reg['source']})")
            context.append("")
        
        # Add key topics
        if analysis["key_topics"]:
            context.append(f"KEY TOPICS: {', '.join(analysis['key_topics'])}")
            context.append("")
        
        # Add related documents
        if "related_documents" in related_info and related_info["related_documents"]:
            context.append("RELATED DOCUMENTS:")
            for doc in related_info["related_documents"]:
                context.append(f"- {doc['title']} (Related topics: {', '.join(doc['similarity_topics'])})")
            context.append("")
        
        # Combine all context
        return "\n".join(context)