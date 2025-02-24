import re
import spacy
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoFeatureExtractor:
    """
    Extract features from cryptocurrency-related documents for due diligence.
    Works with the existing Weaviate 4 setup and BERT embeddings.
    """
    
    def __init__(self):
        """Initialize the feature extractor with necessary models."""
        try:
            # Load spaCy model for NER and text processing
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # TF-IDF for keyword extraction
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        
        logger.info("CryptoFeatureExtractor initialized successfully")
    
    def extract_features(self, text: str, document_type: str = None) -> Dict[str, Any]:
        """
        Extract features from document text based on document type.
        
        Args:
            text (str): The document text
            document_type (str, optional): Type of document (e.g., whitepaper, audit_report)
            
        Returns:
            Dict[str, Any]: Dictionary of extracted features
        """
        if not text:
            logger.warning("Empty text provided for feature extraction")
            return {}
        
        # Process with spaCy for base analysis
        doc = self.nlp(text[:100000])  # Limit to 100K chars to avoid memory issues
        
        # Extract common features for all document types
        features = self._extract_common_features(text, doc)
        
        # Extract type-specific features
        if document_type:
            type_features = self._extract_type_specific_features(text, doc, document_type)
            features.update(type_features)
        
        return features
    
    def _extract_common_features(self, text: str, doc) -> Dict[str, Any]:
        """Extract features common to all document types."""
        # Basic text statistics
        word_count = len(text.split())
        sentence_count = len(list(doc.sents))
        
        # Named entities
        entities = {}
        for entity_type in ["ORG", "PERSON", "GPE", "MONEY", "DATE"]:
            entities[entity_type] = [ent.text for ent in doc.ents if ent.label_ == entity_type]
        
        # Keywords using TF-IDF
        try:
            # Fit TF-IDF on the text
            self.tfidf.fit([text])
            
            # Transform the text to get feature importance
            tfidf_vector = self.tfidf.transform([text])
            
            # Get feature names and their scores
            feature_names = self.tfidf.get_feature_names_out()
            scores = tfidf_vector.toarray()[0]
            
            # Sort by importance and get top keywords
            keywords = [feature_names[i] for i in scores.argsort()[::-1][:10]]
        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            keywords = []
        
        # Detect risk language
        risk_terms = ["risk", "risks", "warning", "caution", "danger", "hazard", 
                      "threat", "vulnerability", "exposure", "liability"]
        risk_score = sum(1 for term in risk_terms if term in text.lower()) / len(risk_terms)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "entities": entities,
            "keywords": keywords,
            "risk_score": min(risk_score * 100, 100)  # Scale to 0-100
        }
    
    def _extract_type_specific_features(self, text: str, doc, document_type: str) -> Dict[str, Any]:
        """Extract features specific to document type."""
        text_lower = text.lower()
        
        if document_type == "whitepaper":
            return self._extract_whitepaper_features(text_lower)
        elif document_type == "audit_report":
            return self._extract_audit_features(text_lower)
        elif document_type == "regulatory_filing":
            return self._extract_regulatory_features(text_lower)
        elif document_type == "due_diligence_report":
            return self._extract_due_diligence_features(text_lower)
        else:
            return {}  # No specific features for unknown document types
    
    def _extract_whitepaper_features(self, text: str) -> Dict[str, Any]:
        """Extract features specific to whitepapers."""
        # Check for tokenomics section
        has_tokenomics = any(term in text for term in ["tokenomics", "token economics", "token distribution"])
        
        # Check for technology description
        tech_terms = ["blockchain", "consensus", "algorithm", "protocol", "smart contract", 
                      "token", "cryptocurrency", "crypto", "decentralized", "distributed"]
        tech_score = sum(1 for term in tech_terms if term in text) / len(tech_terms)
        
        # Check for roadmap
        has_roadmap = any(term in text for term in ["roadmap", "timeline", "milestones", "development plan"])
        
        # Extract blockchain mentions
        blockchain_terms = ["ethereum", "bitcoin", "solana", "binance smart chain", "polygon", 
                           "avalanche", "cardano", "polkadot", "cosmos", "arbitrum"]
        mentioned_blockchains = [term for term in blockchain_terms if term in text]
        
        return {
            "has_tokenomics": has_tokenomics,
            "tech_score": min(tech_score * 100, 100),  # Scale to 0-100
            "has_roadmap": has_roadmap,
            "mentioned_blockchains": mentioned_blockchains
        }
    
    def _extract_audit_features(self, text: str) -> Dict[str, Any]:
        """Extract features specific to audit reports."""
        # Check for vulnerability mentions
        vulnerability_terms = ["vulnerability", "exploit", "bug", "issue", "flaw", 
                              "weakness", "security", "attack", "breach"]
        vulnerability_score = sum(1 for term in vulnerability_terms if term in text) / len(vulnerability_terms)
        
        # Detect severity levels
        severity_patterns = {
            "critical": r"critical\s+(?:issue|vulnerability|finding|bug)",
            "high": r"high\s+(?:issue|vulnerability|finding|bug)",
            "medium": r"medium\s+(?:issue|vulnerability|finding|bug)",
            "low": r"low\s+(?:issue|vulnerability|finding|bug)"
        }
        
        severity_counts = {}
        for level, pattern in severity_patterns.items():
            severity_counts[f"{level}_count"] = len(re.findall(pattern, text))
        
        # Check for recommendations
        has_recommendations = any(term in text for term in ["recommend", "suggestion", "advise", "proposal"])
        
        return {
            "vulnerability_score": min(vulnerability_score * 100, 100),  # Scale to 0-100
            **severity_counts,
            "has_recommendations": has_recommendations
        }
    
    def _extract_regulatory_features(self, text: str) -> Dict[str, Any]:
        """Extract features specific to regulatory filings."""
        # Detect regulatory bodies
        regulatory_bodies = ["sec", "cftc", "finra", "fca", "msa", "bafin", "fsa", "finma", "asic"]
        mentioned_bodies = [body for body in regulatory_bodies if body in text]
        
        # Check for legal terminology
        legal_terms = ["compliance", "regulation", "law", "statute", "jurisdiction", 
                      "legal", "enforce", "violation", "prohibited", "requirement"]
        legal_score = sum(1 for term in legal_terms if term in text) / len(legal_terms)
        
        # Check for penalties or sanctions
        penalty_terms = ["penalty", "fine", "sanction", "charge", "prosecution", 
                        "conviction", "sentence", "ban", "prohibition", "cease and desist"]
        has_penalties = any(term in text for term in penalty_terms)
        
        return {
            "mentioned_regulatory_bodies": mentioned_bodies,
            "legal_score": min(legal_score * 100, 100),  # Scale to 0-100
            "has_penalties": has_penalties
        }
    
    def _extract_due_diligence_features(self, text: str) -> Dict[str, Any]:
        """Extract features specific to due diligence reports."""
        # Check for assessment terminology
        assessment_terms = ["assessment", "evaluation", "analysis", "review", "examination", 
                           "investigation", "verification", "validation", "audit", "inspection"]
        assessment_score = sum(1 for term in assessment_terms if term in text) / len(assessment_terms)
        
        # Check for risk assessment
        risk_assessment_terms = ["risk assessment", "risk analysis", "risk evaluation", 
                               "risk profile", "risk factor", "risk level"]
        has_risk_assessment = any(term in text for term in risk_assessment_terms)
        
        # Check for recommendations
        has_recommendations = any(term in text for term in ["recommend", "suggestion", "advise", "proposal"])
        
        return {
            "assessment_score": min(assessment_score * 100, 100),  # Scale to 0-100
            "has_risk_assessment": has_risk_assessment,
            "has_recommendations": has_recommendations
        }

# For testing
if __name__ == "__main__":
    extractor = CryptoFeatureExtractor()
    
    # Example text
    sample_text = """
    This whitepaper introduces our new blockchain protocol that aims to revolutionize
    the DeFi space. Our tokenomics model includes a total supply of 1 billion tokens,
    with 20% allocated to the team, 30% for public sale, and 50% for ecosystem development.
    
    The technology is built on Ethereum and uses a novel consensus algorithm that improves
    transaction throughput while maintaining security. Our roadmap includes mainnet launch
    in Q2 2023, followed by partnership integrations in Q3.
    
    Risk factors include regulatory uncertainty and competition from established protocols.
    """
    
    # Extract features
    features = extractor.extract_features(sample_text, "whitepaper")
    
    # Print features
    for key, value in features.items():
        print(f"{key}: {value}")