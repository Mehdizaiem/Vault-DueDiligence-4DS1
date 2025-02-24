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
    Enhanced feature extractor for cryptocurrency due diligence documents.
    Focuses on extracting high-value features for risk assessment, compliance, 
    and technical evaluation of crypto assets and funds.
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
        
        logger.info("Enhanced CryptoFeatureExtractor initialized successfully")
    
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
        
        # Extract cryptocurrency mentions
        crypto_mentions = self._extract_crypto_mentions(text)
        
        # Named entities with better organization
        entities = self._extract_entities(doc)
        
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
        
        # Enhanced risk assessment
        risk_assessment = self._perform_risk_assessment(text)
        
        # Detect compliance mentions
        compliance_info = self._extract_compliance_info(text)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "entities": entities,
            "keywords": keywords,
            "crypto_mentions": crypto_mentions,
            "risk_score": risk_assessment["risk_score"],
            "risk_factors": risk_assessment["risk_factors"],
            "compliance_score": compliance_info["compliance_score"],
            "compliance_mentions": compliance_info["compliance_mentions"]
        }
    
    def _extract_type_specific_features(self, text: str, doc, document_type: str) -> Dict[str, Any]:
        """Extract features specific to document type."""
        text_lower = text.lower()
        
        if document_type == "whitepaper":
            return self._extract_whitepaper_features(text_lower, doc)
        elif document_type == "audit_report":
            return self._extract_audit_features(text_lower, doc)
        elif document_type == "regulatory_filing":
            return self._extract_regulatory_features(text_lower, doc)
        elif document_type == "due_diligence_report":
            return self._extract_due_diligence_features(text_lower, doc)
        elif document_type == "project_documentation":
            return self._extract_project_documentation_features(text_lower, doc)
        else:
            return {}  # No specific features for unknown document types
    
    def _extract_crypto_mentions(self, text: str) -> Dict[str, int]:
        """Extract cryptocurrency and blockchain mentions with counts."""
        text_lower = text.lower()
        
        # Major cryptocurrencies and blockchains
        crypto_terms = {
            "bitcoin": ["bitcoin", "btc", "satoshi"],
            "ethereum": ["ethereum", "eth", "ether", "erc20", "erc721", "solidity"],
            "solana": ["solana", "sol"],
            "binance": ["binance", "bnb", "bsc", "binance smart chain"],
            "cardano": ["cardano", "ada"],
            "polkadot": ["polkadot", "dot"],
            "ripple": ["ripple", "xrp"],
            "avalanche": ["avalanche", "avax"],
            "polygon": ["polygon", "matic"],
            "cosmos": ["cosmos", "atom"],
            "chainlink": ["chainlink", "link"],
            "tron": ["tron", "trx"],
            "stellar": ["stellar", "xlm"],
            "defi": ["defi", "decentralized finance", "yield farming", "liquidity mining", "amm"],
            "nft": ["nft", "non-fungible token", "non fungible token"],
            "dao": ["dao", "decentralized autonomous organization"],
            "stablecoin": ["stablecoin", "usdt", "usdc", "dai", "busd", "tether"]
        }
        
        # Count mentions
        mentions = {}
        for crypto, terms in crypto_terms.items():
            count = sum(text_lower.count(term) for term in terms)
            if count > 0:
                mentions[crypto] = count
                
        return mentions
    
    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract and organize named entities from the document."""
        # Organize by entity type
        entities = {
            "organizations": [],
            "people": [],
            "locations": [],
            "money": [],
            "dates": [],
            "other": []
        }
        
        # Process entities
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "PERSON":
                entities["people"].append(ent.text)
            elif ent.label_ == "GPE" or ent.label_ == "LOC":
                entities["locations"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["money"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            else:
                entities["other"].append(f"{ent.text} ({ent.label_})")
        
        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
            
        return entities
    
    def _perform_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Perform enhanced risk assessment on the text."""
        text_lower = text.lower()
        
        # Risk categories with associated terms
        risk_categories = {
            "regulatory": ["regulatory", "regulation", "compliance", "legal", "jurisdiction", 
                         "restriction", "prohibited", "banned", "illegal", "unauthorized"],
            "financial": ["loss", "bankrupt", "insolvency", "debt", "liability", "expense", 
                        "cost", "inflation", "devaluation", "depreciation"],
            "technical": ["hack", "exploit", "vulnerability", "attack", "breach", "bug", 
                        "glitch", "failure", "malfunction", "error"],
            "operational": ["delay", "failure", "disruption", "interruption", "downtime", 
                          "outage", "maintenance", "discontinue", "cease", "halt"],
            "market": ["volatility", "fluctuation", "crash", "bear", "dump", "sell-off", 
                      "competition", "rival", "alternative", "substitute"],
            "fraud": ["scam", "fraud", "phishing", "fake", "counterfeit", "impersonation",
                     "ponzi", "pyramid", "mlm", "money laundering"]
        }
        
        # Count risk terms by category
        risk_counts = {category: 0 for category in risk_categories}
        risk_mentions = {category: [] for category in risk_categories}
        
        # Extract risk contexts (sentences containing risk terms)
        risk_contexts = []
        sentences = [sent.text for sent in self.nlp(text).sents]
        
        for category, terms in risk_categories.items():
            for term in terms:
                count = text_lower.count(term)
                if count > 0:
                    risk_counts[category] += count
                    risk_mentions[category].append(term)
                    
                    # Find sentences containing this risk term
                    for sentence in sentences:
                        if term in sentence.lower() and len(sentence.split()) > 5:  # Avoid fragments
                            risk_contexts.append({
                                "category": category,
                                "term": term,
                                "context": sentence.strip()
                            })
        
        # Calculate overall risk score (weighted)
        weights = {
            "regulatory": 1.2,
            "financial": 1.0,
            "technical": 1.1,
            "operational": 0.9,
            "market": 0.8,
            "fraud": 1.3
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(risk_counts[cat] * weights[cat] for cat in risk_counts)
        max_possible = 10 * total_weight  # Assuming max 10 mentions per category
        
        # Scale to 0-100
        risk_score = min(100, (weighted_sum / max_possible) * 100)
        
        # Get top risk factors
        risk_factors = []
        for category, terms in risk_mentions.items():
            if terms:
                risk_factors.append({
                    "category": category,
                    "terms": terms,
                    "count": risk_counts[category]
                })
        
        # Sort risk factors by count
        risk_factors.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors[:3],  # Top 3 risk categories
            "risk_distribution": risk_counts,
            "risk_contexts": risk_contexts[:5]  # Top 5 contextual examples
        }
    
    def _extract_compliance_info(self, text: str) -> Dict[str, Any]:
        """Extract compliance-related information."""
        text_lower = text.lower()
        
        # Compliance topics
        compliance_topics = {
            "kyc_aml": ["kyc", "know your customer", "aml", "anti-money laundering", "identity verification", 
                       "customer identification", "cip", "customer due diligence", "cdd"],
            "data_privacy": ["gdpr", "ccpa", "data protection", "privacy policy", "data privacy", 
                           "personal data", "data subject", "data rights"],
            "tax": ["tax", "taxation", "irs", "income tax", "capital gains", "tax reporting", 
                   "tax compliance", "tax liability", "fatca", "withholding"],
            "securities": ["security", "securities", "investment contract", "howey test", "sec", 
                         "securities and exchange commission", "regulated security"],
            "licensing": ["license", "licensed", "registration", "registered", "msa", "money services business", 
                         "money transmitter", "bitlicense", "financial conduct authority"]
        }
        
        # Count compliance terms by category
        compliance_counts = {category: 0 for category in compliance_topics}
        compliance_mentions = []
        
        for category, terms in compliance_topics.items():
            for term in terms:
                if term in text_lower:
                    compliance_counts[category] += text_lower.count(term)
                    compliance_mentions.append(term)
        
        # Calculate compliance score
        total_compliance_terms = sum(compliance_counts.values())
        max_expected = 20  # Expected reasonable max number of compliance terms
        compliance_score = min(100, (total_compliance_terms / max_expected) * 100)
        
        return {
            "compliance_score": compliance_score,
            "compliance_distribution": compliance_counts,
            "compliance_mentions": list(set(compliance_mentions))  # Unique mentions
        }
    
    def _extract_whitepaper_features(self, text: str, doc) -> Dict[str, Any]:
        """Extract features specific to whitepapers with enhanced detail."""
        # Tokenomics information
        tokenomics = self._extract_tokenomics_details(text)
        
        # Technology stack assessment
        tech_stack = self._extract_tech_stack(text)
        
        # Team information
        team_info = self._extract_team_info(text, doc)
        
        # Roadmap details
        roadmap = self._extract_roadmap_details(text, doc)
        
        # Use case analysis
        use_cases = self._extract_use_cases(text)
        
        return {
            "has_tokenomics": tokenomics["has_tokenomics"],
            "tokenomics_details": tokenomics,
            "tech_score": tech_stack["tech_score"],
            "tech_stack": tech_stack,
            "has_team_info": len(team_info["team_members"]) > 0,
            "team_info": team_info,
            "has_roadmap": roadmap["has_roadmap"],
            "roadmap_details": roadmap,
            "use_cases": use_cases
        }
    
    def _extract_tokenomics_details(self, text: str) -> Dict[str, Any]:
        """Extract detailed tokenomics information."""
        has_tokenomics = any(term in text for term in ["tokenomics", "token economics", "token distribution", "token allocation"])
        
        # Try to extract token supply information
        supply_pattern = r'(?:total|max|initial|circulating)\s+(?:supply|token)\s*(?:of|:|\s+is)?\s*(?:is|are|will be)?\s*(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*(?:million|billion|trillion|m|b|t)?'
        supply_match = re.search(supply_pattern, text, re.IGNORECASE)
        
        token_supply = None
        supply_unit = None
        if supply_match:
            token_supply = supply_match.group(1)
            # Check if there's a unit specified
            unit_match = re.search(r'(\d+)\s+(million|billion|trillion|m|b|t)', supply_match.group(0), re.IGNORECASE)
            if unit_match:
                supply_unit = unit_match.group(2).lower()
                
        # Try to extract allocation percentages
        allocation_pattern = r'(\d+(?:\.\d+)?)%\s+(?:allocated|allocation|for|to)\s+(?:the)?\s*([a-z\s]+)'
        allocation_matches = re.findall(allocation_pattern, text, re.IGNORECASE)
        
        allocations = {}
        for match in allocation_matches:
            percentage = float(match[0])
            category = match[1].strip().lower()
            allocations[category] = percentage
            
        return {
            "has_tokenomics": has_tokenomics,
            "token_supply": token_supply,
            "supply_unit": supply_unit,
            "allocations": allocations,
            "allocation_detected": len(allocations) > 0
        }
        
    def _extract_tech_stack(self, text: str) -> Dict[str, Any]:
        """Extract information about the technical stack."""
        # Technical terminology
        tech_terms = {
            "blockchain": ["blockchain", "distributed ledger", "dlt"],
            "consensus": ["consensus", "proof of work", "pow", "proof of stake", "pos", "delegated proof of stake", "dpos"],
            "smart_contracts": ["smart contract", "solidity", "contract code", "self-executing"],
            "scaling": ["layer 2", "l2", "scaling", "sharding", "sidechain", "rollup", "optimistic", "zk"],
            "privacy": ["privacy", "zero knowledge", "zk", "confidential", "anonymous", "private"],
            "interoperability": ["interoperability", "cross-chain", "bridge", "atomic swap"],
            "governance": ["governance", "voting", "proposal"]
        }
        
        # Count technical terms by category
        tech_counts = {category: 0 for category in tech_terms}
        tech_mentions = {category: [] for category in tech_terms}
        
        for category, terms in tech_terms.items():
            for term in terms:
                if term in text:
                    tech_counts[category] += text.count(term)
                    tech_mentions[category].append(term)
        
        # Calculate technical score
        total_tech_categories = sum(1 for category in tech_counts if tech_counts[category] > 0)
        tech_score = (total_tech_categories / len(tech_terms)) * 100
        
        # Extract specific blockchain platforms
        blockchain_platforms = ["ethereum", "bitcoin", "solana", "polkadot", "avalanche", "cosmos", 
                               "binance smart chain", "bsc", "polygon", "cardano", "tezos", "algorand"]
        mentioned_blockchains = [chain for chain in blockchain_platforms if chain in text]
        
        return {
            "tech_score": tech_score,
            "tech_distribution": tech_counts,
            "tech_mentions": tech_mentions,
            "mentioned_blockchains": mentioned_blockchains,
            "primary_blockchain": mentioned_blockchains[0] if mentioned_blockchains else None
        }
    
    def _extract_team_info(self, text: str, doc) -> Dict[str, Any]:
        """Extract information about the team."""
        # Check for team section
        has_team_section = any(term in text for term in ["team", "our team", "founding team", "core team", "team members"])
        
        # Extract people names that appear near "team" mentions
        team_members = []
        team_contexts = []
        
        # First try to find titles/roles with names
        role_patterns = [
            r'((?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+)\s*(?:,|is|as)?\s+(?:the|our)?\s*([A-Za-z\s]+(?:Chief|Director|Head|Lead|Manager|Engineer|Developer|Founder|CEO|CTO|CFO|COO))',
            r'(?:Chief|Director|Head|Lead|Manager|Engineer|Developer|Founder|CEO|CTO|CFO|COO)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})'
        ]
        
        for pattern in role_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    name = match[0].strip()
                    role = match[1].strip()
                else:
                    name = match.strip()
                    role = "Unknown"
                    
                if name and len(name.split()) >= 2:  # Ensure it's likely a full name
                    team_members.append({"name": name, "role": role})
        
        # If no structured info found, use NER to find PERSON entities near "team"
        if not team_members:
            sentences = list(doc.sents)
            for i, sent in enumerate(sentences):
                if "team" in sent.text.lower():
                    # Check current sentence and surrounding sentences
                    context_sents = [sent.text]
                    if i > 0:
                        context_sents.append(sentences[i-1].text)
                    if i < len(sentences) - 1:
                        context_sents.append(sentences[i+1].text)
                    
                    context = " ".join(context_sents)
                    context_doc = self.nlp(context)
                    
                    for ent in context_doc.ents:
                        if ent.label_ == "PERSON":
                            team_members.append({"name": ent.text, "role": "Unknown"})
                            team_contexts.append(context)
        
        # Remove duplicates while preserving order
        unique_members = []
        seen = set()
        for member in team_members:
            if member["name"] not in seen:
                seen.add(member["name"])
                unique_members.append(member)
        
        # Extract LinkedIn or social media mentions
        social_media_pattern = r'(?:linkedin|twitter|github)\.com/(?:in/)?([a-z0-9_-]+)'
        social_media_handles = re.findall(social_media_pattern, text, re.IGNORECASE)
        
        return {
            "has_team_section": has_team_section,
            "team_members": unique_members,
            "team_size": len(unique_members),
            "social_media_handles": social_media_handles,
            "team_contexts": team_contexts[:3]  # Up to 3 context examples
        }
    
    def _extract_roadmap_details(self, text: str, doc) -> Dict[str, Any]:
        """Extract detailed roadmap information."""
        has_roadmap = any(term in text for term in ["roadmap", "timeline", "milestones", "development plan", "project timeline"])
        
        # Try to find date-specific milestones
        milestone_pattern = r'(?:Q[1-4]|[A-Z][a-z]+)\s+\d{4}\s*[:-]?\s*([^,.;]+)'
        milestone_matches = re.findall(milestone_pattern, text)
        
        milestones = []
        for match in milestone_matches:
            milestone = match.strip()
            # Only include if milestone is reasonably long and not just a date
            if len(milestone) > 10 and not re.match(r'^Q[1-4]\s+\d{4}$', milestone):
                milestones.append(milestone)
        
        # Extract the roadmap time horizon
        years_mentioned = re.findall(r'\b(20\d\d)\b', text)
        time_horizon = None
        if years_mentioned:
            years = [int(year) for year in years_mentioned]
            if years:
                current_year = min(years)
                last_year = max(years)
                if last_year > current_year:
                    time_horizon = last_year - current_year
        
        return {
            "has_roadmap": has_roadmap,
            "milestones": milestones,
            "milestone_count": len(milestones),
            "time_horizon_years": time_horizon
        }
    
    def _extract_use_cases(self, text: str) -> Dict[str, Any]:
        """Extract use cases mentioned in the document."""
        use_case_patterns = [
            r'use cases?[:\s]+([^.]+)',
            r'can be used for[:\s]+([^.]+)',
            r'applications?[:\s]+([^.]+)',
            r'designed for[:\s]+([^.]+)'
        ]
        
        use_cases = []
        for pattern in use_case_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                use_case = match.strip()
                if len(use_case) > 10:  # Filter out very short matches
                    use_cases.append(use_case)
        
        # Look for industry-specific keywords
        industry_keywords = {
            "finance": ["payment", "remittance", "lending", "borrowing", "trading", "exchange", "defi"],
            "gaming": ["gaming", "game", "metaverse", "virtual world", "play to earn", "p2e"],
            "supply_chain": ["supply chain", "logistics", "tracking", "provenance", "authenticity"],
            "identity": ["identity", "authentication", "verification", "credential", "kyc"],
            "healthcare": ["health", "medical", "patient", "clinical", "healthcare"],
            "real_estate": ["real estate", "property", "land", "deed", "title"],
            "content": ["content", "media", "music", "art", "creator", "royalty"],
            "governance": ["governance", "voting", "dao", "decision making"]
        }
        
        industries = []
        for industry, terms in industry_keywords.items():
            if any(term in text.lower() for term in terms):
                industries.append(industry)
        
        return {
            "use_cases": use_cases,
            "industries": industries
        }
    
    def _extract_audit_features(self, text: str, doc) -> Dict[str, Any]:
        """Extract features specific to audit reports with enhanced detail."""
        # Vulnerability assessment
        vulnerability_assessment = self._extract_vulnerabilities(text)
        
        # Code quality assessment
        code_quality = self._extract_code_quality(text)
        
        # Audit scope
        audit_scope = self._extract_audit_scope(text)
        
        # Recommendations
        recommendations = self._extract_recommendations(text)
        
        # Overall security rating
        if vulnerability_assessment["critical_count"] > 0:
            security_rating = "Critical"
        elif vulnerability_assessment["high_count"] > 0:
            security_rating = "High Risk"
        elif vulnerability_assessment["medium_count"] > 0:
            security_rating = "Medium Risk"
        elif vulnerability_assessment["low_count"] > 0:
            security_rating = "Low Risk"
        else:
            security_rating = "Secure"
            
        return {
            "vulnerability_score": vulnerability_assessment["vulnerability_score"],
            "vulnerability_assessment": vulnerability_assessment,
            "code_quality": code_quality,
            "audit_scope": audit_scope,
            "recommendations": recommendations,
            "security_rating": security_rating
        }
    
    def _extract_vulnerabilities(self, text: str) -> Dict[str, Any]:
        """Extract detailed vulnerability information from audit reports."""
        # Detect severity levels with counts
        severity_patterns = {
            "critical": r'critical[\s-]+(?:issue|finding|vulnerability|severity)',
            "high": r'high[\s-]+(?:issue|finding|vulnerability|severity)',
            "medium": r'medium[\s-]+(?:issue|finding|vulnerability|severity)',
            "low": r'low[\s-]+(?:issue|finding|vulnerability|severity)',
            "informational": r'informational[\s-]+(?:issue|finding|note|severity)'
        }
        
        severity_counts = {}
        for level, pattern in severity_patterns.items():
            severity_counts[f"{level}_count"] = len(re.findall(pattern, text, re.IGNORECASE))
        
        # Calculate vulnerability score based on severity counts
        weights = {"critical_count": 10, "high_count": 5, "medium_count": 2, "low_count": 1, "informational_count": 0}
        weighted_sum = sum(severity_counts[k] * weights[k] for k in severity_counts)
        max_score = 20  # Reasonable maximum expected
        vulnerability_score = min(100, (weighted_sum / max_score) * 100)
        
        # Extract common vulnerability types
        vulnerability_types = {
            "reentrancy": ["reentrancy", "re-entrancy"],
            "overflow": ["overflow", "underflow", "integer overflow"],
            "access_control": ["access control", "authorization", "permission"],
            "front_running": ["front running", "front-running", "transaction ordering"],
            "logic_error": ["logic error", "logical flaw", "business logic"],
            "gas_optimization": ["gas optimization", "gas usage", "gas consumption"]
        }
        
        detected_vulnerabilities = []
        for vuln_type, terms in vulnerability_types.items():
            if any(term in text for term in terms):
                detected_vulnerabilities.append(vuln_type)
        
        return {
            **severity_counts,
            "vulnerability_score": vulnerability_score,
            "detected_vulnerabilities": detected_vulnerabilities
        }
    
    def _extract_code_quality(self, text: str) -> Dict[str, Any]:
        """Extract code quality assessment from audit reports."""
        # Code quality terms
        quality_terms = {
            "positive": ["well structured", "well-structured", "clean code", "well documented", "well-documented", 
                        "maintainable", "efficient", "optimized", "best practices", "high quality"],
            "negative": ["poorly structured", "poorly-structured", "messy code", "poorly documented", "poorly-documented",
                        "hard to maintain", "inefficient", "unoptimized", "bad practices", "low quality"]
        }
        
        # Count positive and negative mentions
        quality_counts = {"positive": 0, "negative": 0}
        for sentiment, terms in quality_terms.items():
            for term in terms:
                quality_counts[sentiment] += text.count(term)
                
        # Calculate code quality score
        total_mentions = quality_counts["positive"] + quality_counts["negative"]
        code_quality_score = 50  # Default neutral
        if total_mentions > 0:
            code_quality_score = min(100, max(0, 50 + (quality_counts["positive"] - quality_counts["negative"]) * 10))
            
        return {
            "code_quality_score": code_quality_score,
            "quality_mentions": quality_counts
        }
    
    def _extract_audit_scope(self, text: str) -> Dict[str, Any]:
        """Extract audit scope information."""
        # Check for scope section
        has_scope_section = any(term in text for term in ["scope", "audit scope", "scope of audit", "in scope"])
        
        # Try to extract contract names or files
        contract_pattern = r'(?:contract|file|module)[\s:]+([A-Za-z0-9_]+\.?[A-Za-z0-9_]*)'
        contracts = re.findall(contract_pattern, text, re.IGNORECASE)
        
        # Try to extract commit hash
        commit_pattern = r'(?:commit|hash)[:=\s]+([a-f0-9]{7,40})'
        commit_hash = re.search(commit_pattern, text, re.IGNORECASE)
        
        # Version information
        version_pattern = r'[vV]ersion[:=\s]+([0-9]+(?:\.[0-9]+)*)'
        version = re.search(version_pattern, text, re.IGNORECASE)
        
        return {
            "has_scope_section": has_scope_section,
            "contracts": list(set(contracts)),
            "commit_hash": commit_hash.group(1) if commit_hash else None,
            "version": version.group(1) if version else None
        }
    
    def _extract_recommendations(self, text: str) -> Dict[str, Any]:
        """Extract recommendations from audit reports."""
        # Check for recommendations section
        has_recommendations_section = any(term in text for term in ["recommendation", "recommendations", "suggested changes", "mitigation"])
        
        # Try to extract specific recommendations
        recommendation_patterns = [
            r'recommend(?:ation)?[:\s]+([^.]+)',
            r'suggest(?:ion|ed)?[:\s]+([^.]+)',
            r'should(?:\sbe)?[:\s]+([^.]+)',
            r'mitigation[:\s]+([^.]+)'
        ]
        
        recommendations = []
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10:  # Filter out very short matches
                    recommendations.append(match.strip())
        
        return {
            "has_recommendations": has_recommendations_section,
            "recommendations": recommendations[:5]  # Top 5 recommendations
        }
    
    def _extract_regulatory_features(self, text: str, doc) -> Dict[str, Any]:
        """Extract features specific to regulatory filings with enhanced detail."""
        # Regulatory bodies and frameworks
        regulatory_info = self._extract_regulatory_bodies(text)
        
        # Legal actions
        legal_actions = self._extract_legal_actions(text, doc)
        
        # Jurisdictions
        jurisdictions = self._extract_jurisdictions(text)
        
        # Compliance requirements
        compliance_requirements = self._extract_compliance_requirements(text)
        
        # Penalties or remedies
        penalties = self._extract_penalties(text)
        
        return {
            "regulatory_info": regulatory_info,
            "legal_actions": legal_actions,
            "jurisdictions": jurisdictions,
            "compliance_requirements": compliance_requirements,
            "penalties": penalties,
            "legal_score": regulatory_info["legal_score"]
        }
    
    def _extract_regulatory_bodies(self, text: str) -> Dict[str, Any]:
        """Extract mentions of regulatory bodies and frameworks."""
        # Regulatory bodies
        regulatory_bodies = {
            "us": ["sec", "securities and exchange commission", "cftc", "commodity futures trading commission", 
                  "finra", "financial industry regulatory authority", "fincen", "financial crimes enforcement network"],
            "uk": ["fca", "financial conduct authority", "pra", "prudential regulation authority"],
            "eu": ["esma", "european securities and markets authority", "eba", "european banking authority"],
            "global": ["fatf", "financial action task force", "iosco", "international organization of securities commissions"],
            "asia": ["mas", "monetary authority of singapore", "sfc", "securities and futures commission", 
                   "fsa", "financial services agency", "jfsa"]
        }
        
        # Count regulatory body mentions by region
        body_counts = {region: 0 for region in regulatory_bodies}
        mentioned_bodies = []
        
        for region, bodies in regulatory_bodies.items():
            for body in bodies:
                if body in text.lower():
                    body_counts[region] += text.lower().count(body)
                    mentioned_bodies.append(body)
        
        # Calculate legal score
        total_mentions = sum(body_counts.values())
        legal_score = min(100, total_mentions * 10)  # Scale up to 100
        
        # Try to identify primary regulatory body
        primary_body = None
        if mentioned_bodies:
            body_mentions = {}
            for body in mentioned_bodies:
                body_mentions[body] = text.lower().count(body)
            
            primary_body = max(body_mentions, key=body_mentions.get)
            
        return {
            "mentioned_regulatory_bodies": mentioned_bodies,
            "regulatory_body_distribution": body_counts,
            "legal_score": legal_score,
            "primary_regulatory_body": primary_body
        }
    
    def _extract_legal_actions(self, text: str, doc) -> Dict[str, Any]:
        """Extract information about legal actions."""
        # Legal action terminology
        legal_action_terms = ["complaint", "indictment", "lawsuit", "litigation", "prosecution", "investigation", 
                            "subpoena", "cease and desist", "injunction", "allegation", "charge"]
        
        # Check for legal actions
        has_legal_action = any(term in text.lower() for term in legal_action_terms)
        
        # Try to extract parties involved
        parties = set()
        
        # Look for "v." or "vs." patterns
        versus_pattern = r'([A-Z][a-zA-Z\s,\.]+)\s+v(?:s)?\.?\s+([A-Z][a-zA-Z\s,\.]+)'
        versus_matches = re.findall(versus_pattern, text)
        
        for match in versus_matches:
            for party in match:
                if len(party) > 3:  # Filter out very short matches
                    parties.add(party.strip())
        
        # Extract dates related to legal actions
        date_pattern = r'(?:filed|issued|dated|commenced|initiated)\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        return {
            "has_legal_action": has_legal_action,
            "legal_action_type": [term for term in legal_action_terms if term in text.lower()],
            "parties": list(parties),
            "dates": dates
        }
    
    def _extract_jurisdictions(self, text: str) -> Dict[str, Any]:
        """Extract jurisdictions mentioned in the document."""
        # Jurisdictions
        jurisdictions = {
            "us": ["united states", "u.s.", "usa", "america"],
            "uk": ["united kingdom", "u.k.", "britain", "england"],
            "eu": ["european union", "eu", "europe"],
            "asia": ["china", "japan", "singapore", "hong kong", "south korea"],
            "other": ["canada", "australia", "brazil", "switzerland", "uae", "dubai"]
        }
        
        # Count jurisdiction mentions
        jurisdiction_counts = {region: 0 for region in jurisdictions}
        mentioned_jurisdictions = []
        
        for region, places in jurisdictions.items():
            for place in places:
                if place.lower() in text.lower():
                    jurisdiction_counts[region] += text.lower().count(place.lower())
                    mentioned_jurisdictions.append(place)
        
        # Try to identify primary jurisdiction
        primary_jurisdiction = None
        if mentioned_jurisdictions:
            jurisdiction_mentions = {}
            for jurisdiction in mentioned_jurisdictions:
                jurisdiction_mentions[jurisdiction] = text.lower().count(jurisdiction.lower())
            
            primary_jurisdiction = max(jurisdiction_mentions, key=jurisdiction_mentions.get)
            
        return {
            "mentioned_jurisdictions": mentioned_jurisdictions,
            "jurisdiction_distribution": jurisdiction_counts,
            "primary_jurisdiction": primary_jurisdiction
        }
    
    def _extract_compliance_requirements(self, text: str) -> Dict[str, Any]:
        """Extract compliance requirements mentioned in the document."""
        # Compliance requirement terminology
        requirement_terms = ["must", "required", "shall", "mandatory", "obligation", "comply", "compliance", "adhere"]
        
        # Check if the document contains compliance requirements
        has_requirements = any(term in text.lower() for term in requirement_terms)
        
        # Try to extract specific requirements
        requirement_patterns = [
            r'must\s+([^.;:]+)',
            r'required to\s+([^.;:]+)',
            r'shall\s+([^.;:]+)',
            r'obligated to\s+([^.;:]+)'
        ]
        
        requirements = []
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10:  # Filter out very short matches
                    requirements.append(match.strip())
        
        return {
            "has_compliance_requirements": has_requirements,
            "requirements": requirements[:5]  # Top 5 requirements
        }
    
    def _extract_penalties(self, text: str) -> Dict[str, Any]:
        """Extract information about penalties or remedies."""
        # Penalty terminology
        penalty_terms = ["penalty", "fine", "sanction", "forfeit", "disgorgement", "cease and desist", 
                        "injunction", "ban", "prohibition", "barred"]
        
        # Check if the document mentions penalties
        has_penalties = any(term in text.lower() for term in penalty_terms)
        
        # Try to extract monetary penalties
        money_pattern = r'(?:penalt|fine|sanction)(?:y|ies)?\s+of\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|b)?'
        money_matches = re.findall(money_pattern, text, re.IGNORECASE)
        
        monetary_penalties = []
        for match in money_matches:
            monetary_penalties.append(match)
        
        return {
            "has_penalties": has_penalties,
            "penalty_types": [term for term in penalty_terms if term in text.lower()],
            "monetary_penalties": monetary_penalties
        }
    
    def _extract_due_diligence_features(self, text: str, doc) -> Dict[str, Any]:
        """Extract features specific to due diligence reports with enhanced detail."""
        # Risk assessment
        risk_assessment = self._extract_dd_risk_assessment(text)
        
        # Financial analysis
        financial_analysis = self._extract_financial_analysis(text)
        
        # Operational assessment
        operational_assessment = self._extract_operational_assessment(text)
        
        # Compliance assessment
        compliance_assessment = self._extract_compliance_assessment(text)
        
        # Recommendations
        recommendations = self._extract_dd_recommendations(text)
        
        return {
            "risk_assessment": risk_assessment,
            "financial_analysis": financial_analysis,
            "operational_assessment": operational_assessment,
            "compliance_assessment": compliance_assessment,
            "recommendations": recommendations,
            "assessment_score": risk_assessment["assessment_score"]
        }
    
    def _extract_dd_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Extract risk assessment information from due diligence reports."""
        # Risk assessment terminology
        risk_terms = ["risk assessment", "risk analysis", "risk evaluation", "risk factor", "risk profile"]
        
        # Check if the document contains a risk assessment
        has_risk_assessment = any(term in text.lower() for term in risk_terms)
        
        # Risk level mentions
        risk_levels = {
            "high": ["high risk", "significant risk", "material risk", "critical risk"],
            "medium": ["medium risk", "moderate risk", "average risk"],
            "low": ["low risk", "minimal risk", "minor risk", "negligible risk"]
        }
        
        # Count risk level mentions
        risk_level_counts = {level: 0 for level in risk_levels}
        for level, terms in risk_levels.items():
            for term in terms:
                risk_level_counts[level] += text.lower().count(term)
        
        # Calculate assessment score
        assessment_terms = ["assessment", "evaluation", "analysis", "review", "examination", 
                           "investigation", "verification", "validation", "audit", "inspection"]
        assessment_count = sum(text.lower().count(term) for term in assessment_terms)
        assessment_score = min(100, assessment_count * 5)  # Scale up to 100
        
        # Extract specific risk factors
        risk_factor_pattern = r'risk factors?(?:\sinclude)?(?:\:|\s+are)?([^.]*(?:\.[^.]*){0,2})'
        risk_factor_matches = re.findall(risk_factor_pattern, text, re.IGNORECASE)
        
        risk_factors = []
        for match in risk_factor_matches:
            if len(match) > 20:  # Filter out very short matches
                risk_factors.append(match.strip())
        
        return {
            "has_risk_assessment": has_risk_assessment,
            "risk_level_distribution": risk_level_counts,
            "assessment_score": assessment_score,
            "risk_factors": risk_factors[:3]  # Top 3 risk factors
        }
    
    def _extract_financial_analysis(self, text: str) -> Dict[str, Any]:
        """Extract financial analysis information from due diligence reports."""
        # Financial terminology
        financial_terms = ["financial", "balance sheet", "income statement", "cash flow", "revenue", 
                         "profit", "margin", "ebitda", "roi", "return on investment"]
        
        # Check if the document contains financial analysis
        has_financial_analysis = any(term in text.lower() for term in financial_terms)
        
        # Try to extract financial metrics
        metrics = {
            "revenue": r'revenue\s*(?:of|:|\s+is|\s+was)?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|b)?',
            "profit": r'(?:profit|net income)\s*(?:of|:|\s+is|\s+was)?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|b)?',
            "margin": r'margin\s*(?:of|:|\s+is|\s+was)?\s*(\d{1,3}(?:\.\d+)?)%',
            "burn_rate": r'burn\s*rate\s*(?:of|:|\s+is|\s+was)?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|b)?',
            "valuation": r'valuation\s*(?:of|:|\s+is|\s+was)?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|b)?'
        }
        
        extracted_metrics = {}
        for metric, pattern in metrics.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted_metrics[metric] = match.group(1)
        
        return {
            "has_financial_analysis": has_financial_analysis,
            "financial_metrics": extracted_metrics
        }
    
    def _extract_operational_assessment(self, text: str) -> Dict[str, Any]:
        """Extract operational assessment information from due diligence reports."""
        # Operational terminology
        operational_terms = ["operational", "operation", "business model", "process", "procedure", 
                            "workflow", "management", "governance", "infrastructure"]
        
        # Check if the document contains operational assessment
        has_operational_assessment = any(term in text.lower() for term in operational_terms)
        
        # Try to extract operational strengths and weaknesses
        strength_pattern = r'(?:strength|strong|advantage|positive|excellent)\s*(?::|include|is|are)?\s*([^.;:]+)'
        weakness_pattern = r'(?:weakness|weak|disadvantage|negative|poor|concern)\s*(?::|include|is|are)?\s*([^.;:]+)'
        
        strengths = []
        for match in re.findall(strength_pattern, text, re.IGNORECASE):
            if len(match) > 10:  # Filter out very short matches
                strengths.append(match.strip())
        
        weaknesses = []
        for match in re.findall(weakness_pattern, text, re.IGNORECASE):
            if len(match) > 10:  # Filter out very short matches
                weaknesses.append(match.strip())
        
        return {
            "has_operational_assessment": has_operational_assessment,
            "strengths": strengths[:3],  # Top 3 strengths
            "weaknesses": weaknesses[:3]  # Top 3 weaknesses
        }
    
    def _extract_compliance_assessment(self, text: str) -> Dict[str, Any]:
        """Extract compliance assessment information from due diligence reports."""
        # Compliance terminology
        compliance_terms = ["compliance", "compliant", "regulatory", "regulation", "law", "legal", 
                          "requirement", "standard", "policy", "procedure"]
        
        # Check if the document contains compliance assessment
        has_compliance_assessment = any(term in text.lower() for term in compliance_terms)
        
        # Try to extract compliance status
        compliant_pattern = r'(?:is|are|fully|deemed|considered|found to be)\s+(?:compliant|in compliance)'
        non_compliant_pattern = r'(?:is|are|not|deemed|considered|found to be)\s+(?:non-compliant|not compliant|not in compliance)'
        
        compliance_status = None
        if re.search(compliant_pattern, text, re.IGNORECASE) and not re.search(non_compliant_pattern, text, re.IGNORECASE):
            compliance_status = "Compliant"
        elif re.search(non_compliant_pattern, text, re.IGNORECASE):
            compliance_status = "Non-Compliant"
        elif has_compliance_assessment:
            compliance_status = "Partially Compliant"
        
        # Try to extract compliance issues
        issue_pattern = r'(?:compliance|regulatory)\s+(?:issue|concern|gap|finding)\s*(?::|include|is|are)?\s*([^.;:]+)'
        
        issues = []
        for match in re.findall(issue_pattern, text, re.IGNORECASE):
            if len(match) > 10:  # Filter out very short matches
                issues.append(match.strip())
        
        return {
            "has_compliance_assessment": has_compliance_assessment,
            "compliance_status": compliance_status,
            "compliance_issues": issues[:3]  # Top 3 issues
        }
    
    def _extract_dd_recommendations(self, text: str) -> Dict[str, Any]:
        """Extract recommendations from due diligence reports."""
        # Recommendation terminology
        recommendation_terms = ["recommend", "suggestion", "advise", "proposal", "action item"]
        
        # Check if the document contains recommendations
        has_recommendations = any(term in text.lower() for term in recommendation_terms)
        
        # Try to extract specific recommendations
        recommendation_pattern = r'(?:recommend|suggest|advise|propose)\s+(?:to|that)?\s*([^.;:]+)'
        
        recommendations = []
        for match in re.findall(recommendation_pattern, text, re.IGNORECASE):
            if len(match) > 10:  # Filter out very short matches
                recommendations.append(match.strip())
        
        return {
            "has_recommendations": has_recommendations,
            "recommendations": recommendations[:5]  # Top 5 recommendations
        }
        
    def _extract_project_documentation_features(self, text: str, doc) -> Dict[str, Any]:
        """Extract features specific to project documentation."""
        # Technical specifications
        technical_specs = self._extract_technical_specs(text)
        
        # Project timeline
        project_timeline = self._extract_project_timeline(text)
        
        # Architecture information
        architecture_info = self._extract_architecture_info(text)
        
        # API documentation
        api_info = self._extract_api_info(text)
        
        return {
            "technical_specs": technical_specs,
            "project_timeline": project_timeline,
            "architecture_info": architecture_info,
            "api_info": api_info,
        }
    
    def _extract_technical_specs(self, text: str) -> Dict[str, Any]:
        """Extract technical specifications from project documentation."""
        # Technical terminology
        technical_terms = ["specification", "requirement", "feature", "functionality", "technical", 
                         "architecture", "design", "implementation", "interface", "protocol"]
        
        # Check if the document contains technical specifications
        has_technical_specs = any(term in text.lower() for term in technical_terms)
        
        # Try to extract specific specifications
        spec_pattern = r'(?:specification|requirement|feature)\s*(?::|include|is|are)?\s*([^.;:]+)'
        
        specs = []
        for match in re.findall(spec_pattern, text, re.IGNORECASE):
            if len(match) > 10:  # Filter out very short matches
                specs.append(match.strip())
        
        # Programming languages mentioned
        languages = ["javascript", "python", "java", "c++", "c#", "go", "rust", "solidity", "ruby", "php", "swift"]
        mentioned_languages = [lang for lang in languages if lang in text.lower()]
        
        return {
            "has_technical_specs": has_technical_specs,
            "specifications": specs[:5],  # Top 5 specifications
            "programming_languages": mentioned_languages
        }
    
    def _extract_project_timeline(self, text: str) -> Dict[str, Any]:
        """Extract project timeline information from project documentation."""
        # Timeline terminology
        timeline_terms = ["timeline", "schedule", "deadline", "milestone", "phase", "stage", "sprint"]
        
        # Check if the document contains timeline information
        has_timeline = any(term in text.lower() for term in timeline_terms)
        
        # Try to extract date-related information
        date_pattern = r'(?:by|on|before|after)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        # Try to extract project phases
        phase_pattern = r'(?:phase|stage|sprint)\s+(\d+)'
        phases = re.findall(phase_pattern, text, re.IGNORECASE)
        
        return {
            "has_timeline": has_timeline,
            "mentioned_dates": dates,
            "project_phases": [int(phase) for phase in phases]
        }
    
    def _extract_architecture_info(self, text: str) -> Dict[str, Any]:
        """Extract architecture information from project documentation."""
        # Architecture terminology
        architecture_terms = ["architecture", "component", "module", "system", "design", "structure", 
                            "pattern", "framework", "infrastructure", "platform"]
        
        # Check if the document contains architecture information
        has_architecture_info = any(term in text.lower() for term in architecture_terms)
        
        # Try to extract components
        component_pattern = r'(?:component|module)\s+(?:called|named)?\s*[\'"]?([A-Za-z0-9_]+)[\'"]?'
        components = re.findall(component_pattern, text, re.IGNORECASE)
        
        # Architecture patterns mentioned
        patterns = ["mvc", "model-view-controller", "microservice", "client-server", "layered", 
                  "event-driven", "rest", "graphql", "pub-sub", "publisher-subscriber"]
        mentioned_patterns = [pattern for pattern in patterns if pattern in text.lower()]
        
        return {
            "has_architecture_info": has_architecture_info,
            "components": components,
            "architecture_patterns": mentioned_patterns
        }
    
    def _extract_api_info(self, text: str) -> Dict[str, Any]:
        """Extract API information from project documentation."""
        # API terminology
        api_terms = ["api", "endpoint", "interface", "rest", "graphql", "json", "xml", "http", 
                    "get", "post", "put", "delete", "request", "response"]
        
        # Check if the document contains API information
        has_api_info = any(term in text.lower() for term in api_terms)
        
        # Try to extract endpoints
        endpoint_pattern = r'(?:endpoint|url|uri|path)\s*(?::|is|at)?\s*[\'"]?(/[a-zA-Z0-9/_-]+)[\'"]?'
        endpoints = re.findall(endpoint_pattern, text, re.IGNORECASE)
        
        # Data formats mentioned
        formats = ["json", "xml", "yaml", "protobuf", "grpc"]
        mentioned_formats = [format for format in formats if format in text.lower()]
        
        return {
            "has_api_info": has_api_info,
            "endpoints": endpoints,
            "data_formats": mentioned_formats
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