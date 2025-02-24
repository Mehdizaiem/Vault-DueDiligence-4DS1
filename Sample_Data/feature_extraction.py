import re
import spacy
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob  # For sentiment analysis
from langdetect import detect  # For language detection
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoFeatureExtractor:
    """
    Enhanced feature extractor for cryptocurrency due diligence documents.
    Integrates with embedding generation pipeline and extracts high-value features.
    """

    def __init__(self, model_size: str = "large"):
        """Initialize with configurable spaCy model size."""
        self.nlp_cache = {}  # Cache for spaCy docs
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        
        # Load spaCy model
        model_map = {"small": "en_core_web_sm", "medium": "en_core_web_md", "large": "en_core_web_lg"}
        spacy_model = model_map.get(model_size, "en_core_web_lg")
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"Downloading spaCy model '{spacy_model}'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        # Add custom crypto patterns
        self._add_crypto_patterns()
        logger.info("CryptoFeatureExtractor initialized successfully")

    def _add_crypto_patterns(self):
        """Add custom entity patterns for crypto-specific terms."""
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "CRYPTO", "pattern": [{"LOWER": {"IN": ["bitcoin", "ethereum", "solana", "bnb", "ada"]}}]},
            {"label": "TOKEN", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}$"}}]}
        ]
        ruler.add_patterns(patterns)

    def _chunk_text(self, text: str, max_chars: int = 100000) -> List[str]:
        """Chunk text into manageable pieces by sentence boundaries."""
        if len(text) <= max_chars:
            return [text]
        
        doc = self.nlp(text[:max_chars * 2])
        chunks = []
        current_chunk = ""
        
        for sent in doc.sents:
            if len(current_chunk) + len(sent.text) <= max_chars:
                current_chunk += sent.text + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent.text + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Text chunked into {len(chunks)} parts")
        return chunks

    def extract_features(self, text: str, document_type: str = None) -> Dict[str, Any]:
        """
        Extract features from text, designed to work with embedding pipeline.
        """
        if not text:
            logger.warning("Empty text provided for feature extraction")
            return {}
        
        # Detect language
        lang = detect(text[:500])
        if lang != "en":
            logger.warning(f"Non-English text detected (lang: {lang}). Features may be limited.")
        
        # Chunk text if necessary (aligned with embedding chunking)
        chunks = self._chunk_text(text, max_chars=100000)  # Matches your embedding limit
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_chunk, chunk, document_type) 
                      for chunk in chunks]
            features_list = [future.result() for future in futures]
        
        return self._aggregate_features(features_list)

    def _process_chunk(self, text: str, document_type: str) -> Dict[str, Any]:
        """Process a single chunk of text."""
        doc = self._get_cached_doc(text)
        features = self._extract_common_features(text, doc)
        
        if document_type:
            type_features = self._extract_type_specific_features(text, doc, document_type)
            features.update(type_features)
        
        return features

    def _get_cached_doc(self, text: str) -> spacy.tokens.Doc:
        """Retrieve or create cached spaCy Doc."""
        text_hash = hash(text[:1000])
        if text_hash not in self.nlp_cache:
            self.nlp_cache[text_hash] = self.nlp(text[:100000])
        return self.nlp_cache[text_hash]

    def _aggregate_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate features from multiple chunks."""
        if not features_list:
            return {}
        
        aggregated = features_list[0].copy()
        if len(features_list) == 1:
            return aggregated
        
        # Numeric aggregation
        for key in ["word_count", "sentence_count", "risk_score", "compliance_score", "sentiment_score"]:
            if key in aggregated:
                aggregated[key] = sum(f.get(key, 0) for f in features_list) / len(features_list)
        
        # List/dict aggregation
        for key in ["entities", "keywords", "crypto_mentions", "risk_factors", "compliance_mentions"]:
            if key == "entities":
                for ent_type in aggregated[key]:
                    combined = set()
                    for f in features_list:
                        combined.update(f[key][ent_type])
                    aggregated[key][ent_type] = list(combined)
            elif key == "crypto_mentions":
                combined = {}
                for f in features_list:
                    for crypto, count in f[key].items():
                        combined[crypto] = combined.get(crypto, 0) + count
                aggregated[key] = combined
            elif key in aggregated:
                combined = set()
                for f in features_list:
                    if isinstance(f[key], list):
                        combined.update(f[key])
                aggregated[key] = list(combined)
        
        return aggregated

    def _extract_common_features(self, text: str, doc) -> Dict[str, Any]:
        """Extract common features with sentiment analysis."""
        word_count = len(text.split())
        sentence_count = len(list(doc.sents))
        
        crypto_mentions = self._extract_crypto_mentions(text)
        entities = self._extract_entities(doc)
        
        try:
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            keywords = [feature_names[i] for i in scores.argsort()[::-1][:10]]
        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            keywords = []

        risk_assessment = self._perform_risk_assessment(text)
        compliance_info = self._extract_compliance_info(text)
        
        sentiment = TextBlob(text[:1000]).sentiment
        sentiment_score = sentiment.polarity

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "entities": entities,
            "keywords": keywords,
            "crypto_mentions": crypto_mentions,
            "risk_score": risk_assessment["risk_score"],
            "risk_factors": risk_assessment["risk_factors"],
            "compliance_score": compliance_info["compliance_score"],
            "compliance_mentions": compliance_info["compliance_mentions"],
            "sentiment_score": sentiment_score
        }

    def _extract_crypto_mentions(self, text: str) -> Dict[str, int]:
        """Extract cryptocurrency mentions."""
        text_lower = text.lower()
        crypto_terms = {
            "bitcoin": ["bitcoin", "btc", "satoshi"],
            "ethereum": ["ethereum", "eth", "ether", "erc20", "erc721", "solidity"],
            "solana": ["solana", "sol"],
            "binance": ["binance", "bnb", "bsc"],
            "cardano": ["cardano", "ada"]
        }
        mentions = {}
        for crypto, terms in crypto_terms.items():
            count = sum(text_lower.count(term) for term in terms)
            if count > 0:
                mentions[crypto] = count
        return mentions

    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract entities with custom crypto recognition."""
        entities = {
            "organizations": [], "people": [], "locations": [], 
            "money": [], "dates": [], "crypto": [], "tokens": [], "other": []
        }
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "PERSON":
                entities["people"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["money"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "CRYPTO":
                entities["crypto"].append(ent.text)
            elif ent.label_ == "TOKEN":
                entities["tokens"].append(ent.text)
            else:
                entities["other"].append(f"{ent.text} ({ent.label_})")
        
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        return entities

    def _perform_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Perform sentiment-weighted risk assessment."""
        text_lower = text.lower()
        risk_categories = {
            "regulatory": ["regulatory", "compliance", "legal"],
            "financial": ["loss", "bankrupt", "debt"],
            "technical": ["hack", "vulnerability", "bug"],
            "fraud": ["scam", "fraud", "phishing"]
        }
        
        risk_counts = {cat: 0 for cat in risk_categories}
        risk_mentions = {cat: [] for cat in risk_categories}
        
        for cat, terms in risk_categories.items():
            for term in terms:
                count = text_lower.count(term)
                if count > 0:
                    risk_counts[cat] += count
                    risk_mentions[cat].append(term)
        
        total_risk = sum(risk_counts.values())
        sentiment = TextBlob(text[:1000]).sentiment.polarity
        risk_score = min(100, (total_risk * (1 - sentiment)) * 10)
        
        risk_factors = [{"category": cat, "terms": terms, "count": risk_counts[cat]}
                       for cat, terms in risk_mentions.items() if terms]
        risk_factors.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors[:3],
            "risk_distribution": risk_counts
        }

    def _extract_compliance_info(self, text: str) -> Dict[str, Any]:
        """Extract compliance info with sentiment adjustment."""
        text_lower = text.lower()
        compliance_topics = {
            "kyc_aml": ["kyc", "aml", "identity verification"],
            "data_privacy": ["gdpr", "privacy", "data protection"]
        }
        
        compliance_counts = {cat: 0 for cat in compliance_topics}
        compliance_mentions = []
        for cat, terms in compliance_topics.items():
            for term in terms:
                if term in text_lower:
                    compliance_counts[cat] += text_lower.count(term)
                    compliance_mentions.append(term)
        
        total_compliance = sum(compliance_counts.values())
        sentiment = TextBlob(text[:1000]).sentiment.polarity
        compliance_score = min(100, (total_compliance * (1 + sentiment)) * 5)
        
        return {
            "compliance_score": compliance_score,
            "compliance_mentions": list(set(compliance_mentions))
        }

    def _extract_type_specific_features(self, text: str, doc, document_type: str) -> Dict[str, Any]:
        """Extract type-specific features."""
        text_lower = text.lower()
        if document_type == "whitepaper":
            tokenomics = self._extract_tokenomics_details(text_lower)
            return {"tokenomics_details": tokenomics}
        elif document_type == "regulatory_filing":
            regulatory = self._extract_regulatory_features(text_lower)
            return {"regulatory_features": regulatory}
        return {}

    def _extract_tokenomics_details(self, text: str) -> Dict[str, Any]:
        """Extract enhanced tokenomics details."""
        has_tokenomics = any(term in text for term in ["tokenomics", "token distribution"])
        supply_pattern = r'(?:total|max)\s+supply[^\d]*(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*(?:million|billion|trillion|m|b|t)?'
        supply_match = re.search(supply_pattern, text, re.IGNORECASE)
        
        token_supply = supply_match.group(1) if supply_match else None
        supply_unit = supply_match.group(0).split()[-1] if supply_match and len(supply_match.groups()) > 1 else None
        
        vesting_pattern = r'vesting\s+(?:schedule|period)\s*[^\d]*(\d+\s*(?:month|year)s?)'
        vesting = re.search(vesting_pattern, text, re.IGNORECASE)
        
        return {
            "has_tokenomics": has_tokenomics,
            "token_supply": token_supply,
            "supply_unit": supply_unit,
            "vesting_schedule": vesting.group(1) if vesting else None
        }

    def _extract_regulatory_features(self, text: str) -> Dict[str, Any]:
        """Extract basic regulatory features (expand as needed)."""
        regulatory_bodies = ["sec", "cftc", "finra", "fca"]
        mentions = [body for body in regulatory_bodies if body in text]
        return {
            "regulatory_mentions": mentions,
            "has_regulation": len(mentions) > 0
        }

if __name__ == "__main__":
    # For testing with your pipeline, assuming you call it from process_document
    extractor = CryptoFeatureExtractor()
    # No sample text; rely on your process_document.py to provide input
    logger.info("CryptoFeatureExtractor ready for pipeline integration")