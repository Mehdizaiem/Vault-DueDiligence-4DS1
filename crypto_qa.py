#!/usr/bin/env python
import os
import logging
import sys
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import requests
import traceback
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_qa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import from your existing code
from Sample_Data.vector_store.storage_manager import StorageManager
# Reuse agentic_rag.py functions
from agentic_rag import CryptoDueDiligenceSystem

def load_env_from_local():
    """Load environment variables from .env.local file"""
    env_file = Path('.env.local')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value
    else:
        logger.warning(".env.local file not found. Make sure to set GROQ_API_KEY environment variable.")

# Load environment variables
load_env_from_local()

class QueryAnalyzer:
    """
    Advanced query analyzer that performs deep analysis of questions
    to determine intent, entities, and appropriate retrieval strategies.
    """
    
    def __init__(self):
        self.crypto_entities = {
            "bitcoin": ["bitcoin", "btc", "xbt", "satoshi", "nakamoto"],
            "ethereum": ["ethereum", "eth", "vitalik", "buterin", "ether"],
            "solana": ["solana", "sol"],
            "binance": ["binance", "bnb", "binance coin"],
            "cardano": ["cardano", "ada", "hoskinson"],
            "ripple": ["ripple", "xrp"],
            "polkadot": ["polkadot", "dot"],
            "dogecoin": ["dogecoin", "doge"],
            "avalanche": ["avalanche", "avax"],
            "polygon": ["polygon", "matic"],
            "tether": ["tether", "usdt"],
            "usd coin": ["usd coin", "usdc"],
        }
        
        # Initialize category keywords
        self._initialize_categories()
        
        # Regular expressions for entity extraction
        self.date_regex = re.compile(r'\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE)
        self.number_regex = re.compile(r'\b\d+(?:\.\d+)?\b')
        self.percentage_regex = re.compile(r'\b\d+(?:\.\d+)?%\b')
        self.address_regex = re.compile(r'\b0x[a-fA-F0-9]{40}\b')  # Ethereum address format
        
    def _initialize_categories(self):
        """Initialize the categories and their keywords."""
        self.categories = {
            "legal_regulatory": [
                "legal", "regulatory", "compliance", "jurisdiction", "license", 
                "permit", "registration", "law", "regulation", "kyc", "aml",
                "securities", "tax", "gdpr", "privacy", "intellectual property",
                "trademark", "copyright", "patent", "sanctions", "sec", "cftc",
                "finra", "fincen", "howey", "security", "commodity", "fca", "mifid",
                "gdpr", "data protection", "legal opinion", "counsel", "attorney",
                "lawyer", "court", "lawsuit", "litigation", "illegal", "legality"
            ],
            "team_background": [
                "team", "founder", "ceo", "cto", "background", "experience",
                "expertise", "track record", "advisor", "management", "leadership",
                "qualification", "doxxed", "anonymous", "succession", "board",
                "director", "executive", "employee", "staff", "developer", "engineer",
                "previous", "history", "reputation", "scandal", "controversy", 
                "professional", "education", "degree", "linkedin", "github",
                "profile", "resume", "curriculum vitae", "cv", "biography", "bio",
                "worked", "employment", "hired", "fired", "quit", "resigned"
            ],
            "technical": [
                "security", "blockchain", "smart contract", "code", "audit", 
                "protocol", "architecture", "bug", "vulnerability", "hack",
                "encryption", "consensus", "node", "scalability", "infrastructure",
                "algorithm", "cryptography", "hash", "mining", "staking", "proof",
                "pow", "pos", "wallet", "key", "private key", "public key",
                "custody", "cold storage", "hot wallet", "mainnet", "testnet",
                "network", "bandwidth", "throughput", "latency", "transaction",
                "block", "chain", "fork", "upgrade", "rollback", "gas", "fee",
                "layer", "layer-1", "layer-2", "sidechain", "rollup", "zkrollup",
                "optimistic rollup", "sharding", "virtual machine", "evm",
                "wasm", "solidity", "rust", "github", "repository", "commit"
            ],
            "financial": [
                "token", "tokenomics", "price", "market cap", "volume", "liquidity",
                "revenue", "profit", "funding", "investment", "treasury", "burn",
                "supply", "distribution", "allocation", "vesting", "inflation",
                "deflation", "circulating", "maximum", "total", "issuance", "mint",
                "airdrop", "presale", "ico", "ido", "sto", "ieo", "private sale",
                "seed round", "venture capital", "vc", "investor", "angel",
                "institutional", "retail", "whale", "dump", "pump", "dilution",
                "staking", "yield", "apr", "apy", "roi", "return", "interest",
                "borrow", "lend", "loan", "collateral", "leverage", "margin",
                "derivative", "futures", "options", "swap", "trade", "order book"
            ],
            "market_price": [
                "price", "value", "worth", "cost", "trading at", "exchange rate",
                "drop", "fall", "decrease", "down", "bear", "bull", "bearish", "bullish",
                "rise", "increase", "up", "trend", "movement", "performance",
                "ath", "all-time high", "all time high", "low", "high", "bottom", "top",
                "resistance", "support", "breakout", "correction", "rebound", "crash",
                "surge", "soar", "plummet", "tank", "rally", "dump", "pump", "volatile",
                "volatility", "stable", "momentum", "reversal", "consolidation",
                "accumulation", "distribution", "volume", "buy", "sell", "bid", "ask"
            ],
            "market_analysis": [
                "market", "competitor", "adoption", "user", "growth", "trend",
                "industry", "sector", "demand", "supply", "competition", "advantage",
                "differentiator", "unique", "positioning", "target", "analysis",
                "metric", "statistic", "data", "research", "report", "study",
                "survey", "analyst", "prediction", "forecast", "projection",
                "estimate", "user base", "customer", "client", "adoption curve",
                "penetration", "market share", "dominance", "moat", "barrier",
                "first mover", "network effect", "ecosystem", "competitor"
            ],
            "governance": [
                "governance", "dao", "voting", "proposal", "stake", "decision",
                "decentralization", "centralization", "control", "community",
                "power", "authority", "participation", "token holder", "incentive",
                "mechanism", "treasury", "fund", "allocation", "grant", "bounty",
                "reward", "contributor", "maintainer", "moderator", "validator",
                "validator set", "consensus", "quorum", "veto", "approve", "reject",
                "democratic", "plutocratic", "meritocratic", "representation",
                "delegate", "proxy", "quadratic", "holographic", "conviction"
            ],
            "risk": [
                "risk", "threat", "vulnerability", "exploit", "attack", "rug pull",
                "scam", "fraud", "manipulation", "security", "volatility", "downturn",
                "bear market", "crash", "regulation", "legal", "compliance", 
                "liability", "exposure", "danger", "hazard", "peril", "downside",
                "bankruptcy", "insolvency", "default", "failure", "collapse",
                "hacked", "breach", "stolen", "loss", "mitigation", "preventative",
                "contingency", "backup", "insurance", "protection", "safeguard"
            ],
            "sentiment": [
                "sentiment", "feeling", "opinion", "attitude", "perception", "view",
                "outlook", "bullish", "bearish", "positive", "negative", "neutral",
                "optimistic", "pessimistic", "confidence", "fear", "greed", 
                "uncertainty", "doubt", "belief", "trust", "distrust", "skeptical",
                "hopeful", "excited", "worried", "concerned", "enthusiastic",
                "popularity", "awareness", "reputation", "social media", "twitter", 
                "reddit", "telegram", "discord", "news", "articles", "press"
            ],
            "on_chain": [
                "on chain", "onchain", "chain data", "blockchain data", "transaction",
                "address", "wallet", "balance", "transfer", "movement", "inflow",
                "outflow", "exchange", "deposit", "withdrawal", "activity", "dormant",
                "active", "holder", "holding", "whale", "distribution", "concentration",
                "hodl", "hodler", "analysis", "analytic", "metric", "indicator", 
                "signal", "glassnode", "intotheblock", "santiment", "block explorer"
            ],
            "defi": [
                "defi", "decentralized finance", "lending", "borrowing", "yield",
                "farming", "harvesting", "liquidity", "pool", "amm", "automated market maker",
                "dex", "decentralized exchange", "swap", "stake", "unstake", "lock",
                "tvl", "total value locked", "collateral", "liquidation", "loan",
                "apy", "apr", "interest", "curve", "uniswap", "sushiswap", "compound",
                "aave", "maker", "dai", "stablecoin", "pegged", "algorithmic stablecoin"
            ],
            "nft": [
                "nft", "non-fungible token", "collectible", "art", "artist", "creator",
                "royalty", "mint", "marketplace", "opensea", "rarible", "foundation",
                "auction", "bid", "floor price", "rarity", "trait", "attribute",
                "metadata", "token uri", "erc-721", "erc721", "erc-1155", "erc1155",
                "jpeg", "png", "media", "digital asset", "ownership", "provenance"
            ],
            "macroeconomic": [
                "macro", "macroeconomic", "economy", "economic", "inflation", "deflation",
                "interest rate", "fed", "federal reserve", "central bank", "monetary policy",
                "fiscal policy", "recession", "depression", "growth", "gdp", "unemployment",
                "employment", "labor", "supply chain", "shortage", "surplus", "geopolitical",
                "regulation", "policy", "government", "politics", "election", "war", "conflict",
                "sanction", "embargo", "trade", "tariff", "global", "international"
            ],
            "comparative": [
                "compare", "comparison", "versus", "vs", "difference", "similar", "better",
                "worse", "advantage", "disadvantage", "pro", "con", "benefit", "drawback",
                "strength", "weakness", "opportunity", "threat", "swot", "competitive",
                "alternative", "rival", "competitor", "peer", "benchmark", "measure",
                "metric", "ratio", "relative", "absolute", "performance", "outperform",
                "underperform", "match", "exceed", "fall short", "rank", "rating"
            ],
            "due_diligence": [
                "due diligence", "dd", "assessment", "evaluate", "investigation", "verify",
                "verification", "review", "audit", "check", "research", "background check",
                "vetting", "examination", "analyze", "scrutiny", "scrutinize", "thorough",
                "comprehensive", "deep dive", "in-depth", "detailed", "meticulous",
                "careful", "rigorous", "standards", "criteria", "framework", "checklist",
                "questionnaire", "inquiry", "probe", "fact-finding", "fact check"
            ]
        }
        
    def analyze(self, question: str) -> Dict:
        """
        Perform comprehensive analysis of the question.
        
        Args:
            question (str): The user's question
            
        Returns:
            Dict: Analysis results containing category, entities, intent, etc.
        """
        question_lower = question.lower()
        
        # Perform entity extraction
        crypto_entities = self._extract_crypto_entities(question_lower)
        dates = self._extract_dates(question_lower)
        numbers = self._extract_numbers(question_lower)
        addresses = self._extract_addresses(question_lower)
        
        # Determine primary and secondary categories
        category_scores = self._score_categories(question_lower)
        primary_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "general"
        
        # Get secondary categories (any with a score > 0)
        secondary_categories = [cat for cat, score in category_scores.items() 
                               if score > 0 and cat != primary_category]
        
        # Determine temporality (past, present, future)
        temporality = self._determine_temporality(question_lower)
        
        # Determine if it's a comparison question
        is_comparison = self._is_comparison_question(question_lower)
        compared_entities = self._extract_comparison_entities(question_lower) if is_comparison else []
        
        # Determine the intent of the question
        intent = self._determine_intent(question_lower)
        
        # Identify if specific time period is mentioned
        time_period = self._extract_time_period(question_lower)
        
        # Identify potential collections to search
        collections_to_search = self._determine_collections(primary_category, crypto_entities, intent, addresses)
        
        # Combine all analysis results
        analysis = {
            "primary_category": primary_category,
            "secondary_categories": secondary_categories,
            "crypto_entities": crypto_entities,
            "dates": dates,
            "numbers": numbers,
            "addresses": addresses,
            "temporality": temporality,
            "is_comparison": is_comparison,
            "compared_entities": compared_entities,
            "intent": intent,
            "time_period": time_period,
            "collections_to_search": collections_to_search,
            "complexity": self._determine_complexity(question_lower),
            "question_type": self._determine_question_type(question_lower)
        }
        
        logger.info(f"Query analysis: {json.dumps(analysis, default=str)}")
        return analysis
    
    def _score_categories(self, question: str) -> Dict[str, int]:
        """Score each category based on keyword matches."""
        category_scores = {}
        
        for category, keywords in self.categories.items():
            # Count occurrences of each keyword
            score = sum(1 for keyword in keywords if keyword in question)
            
            # Add score if above 0
            if score > 0:
                category_scores[category] = score
        
        return category_scores
    
    def _extract_crypto_entities(self, question: str) -> List[str]:
        """Extract cryptocurrency entities from the question."""
        found_entities = []
        
        # Check for crypto names and symbols
        for entity, variations in self.crypto_entities.items():
            if any(f" {var} " in f" {question} " or 
                   question.startswith(f"{var} ") or 
                   question.endswith(f" {var}") or
                   question == var 
                   for var in variations):
                found_entities.append(entity)
        
        return found_entities
    
    def _extract_dates(self, question: str) -> List[str]:
        """Extract date references from the question."""
        return self.date_regex.findall(question)
    
    def _extract_numbers(self, question: str) -> List[str]:
        """Extract numeric values from the question."""
        return self.number_regex.findall(question)
    
    def _extract_addresses(self, question: str) -> List[str]:
        """Extract blockchain addresses from the question."""
        return self.address_regex.findall(question)
    
    def _determine_temporality(self, question: str) -> str:
        """Determine if the question is about past, present, or future."""
        # Past tense indicators
        past_indicators = ["was", "were", "did", "had", "happened", "occurred", "previous", 
                          "historically", "before", "earlier", "last", "past", "history"]
        
        # Future tense indicators
        future_indicators = ["will", "going to", "shall", "may", "might", "could", "would", 
                            "predict", "forecast", "projection", "expect", "anticipate", 
                            "future", "soon", "upcoming", "next", "tomorrow"]
        
        # Check for past tense
        if any(indicator in question for indicator in past_indicators):
            return "past"
        
        # Check for future tense
        if any(indicator in question for indicator in future_indicators):
            return "future"
        
        # Default to present
        return "present"
    
    def _is_comparison_question(self, question: str) -> bool:
        """Determine if this is a comparison question."""
        comparison_indicators = ["compare", "comparison", "versus", "vs", "vs.", "difference", 
                                "differences", "similar", "similarities", "better", "worse", 
                                "best", "worst", "stronger", "weaker", "faster", "slower", 
                                "higher", "lower", "more", "less", "or", "rather than", 
                                "compared to", "against", "between"]
        
        return any(f" {indicator} " in f" {question} " for indicator in comparison_indicators)
    
    def _extract_comparison_entities(self, question: str) -> List[str]:
        """Extract entities being compared in a comparison question."""
        compared_entities = []
        
        # First check for cryptocurrency entities
        all_variations = []
        for entity, variations in self.crypto_entities.items():
            for var in variations:
                all_variations.append((var, entity))
        
        # Sort by length (longest first) to avoid partial matches
        all_variations.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Find all occurrences
        for variation, entity in all_variations:
            if variation in question and entity not in compared_entities:
                compared_entities.append(entity)
        
        return compared_entities
    
    def _determine_intent(self, question: str) -> str:
        """Determine the intent of the question."""
        # Define intent keywords
        intent_keywords = {
            "information": ["what", "explain", "describe", "tell", "information", "details", "overview"],
            "analysis": ["analyze", "assessment", "evaluate", "review", "pros", "cons", "swot", "strength", "weakness"],
            "price": ["price", "value", "worth", "cost", "trading", "exchange rate", "market value"],
            "prediction": ["predict", "forecast", "projection", "future", "will", "expect", "anticipate"],
            "recommendation": ["recommend", "suggest", "should", "best", "advise", "advice", "opinion"],
            "instruction": ["how", "steps", "guide", "tutorial", "instructions", "process", "procedure"],
            "comparison": ["compare", "comparison", "versus", "vs", "difference", "similar", "better", "worse"],
            "risk_assessment": ["risk", "exposure", "vulnerability", "danger", "threat", "potential downside"],
            "due_diligence": ["due diligence", "dd", "thorough investigation", "comprehensive evaluation"]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the highest scoring intent, or "information" as default
        return max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else "information"
    
    def _extract_time_period(self, question: str) -> str:
        """Extract time period from the question."""
        time_periods = {
            "today": ["today", "now", "current", "currently", "present"],
            "yesterday": ["yesterday"],
            "this_week": ["this week", "current week", "past week"],
            "last_week": ["last week", "previous week"],
            "this_month": ["this month", "current month"],
            "last_month": ["last month", "previous month"],
            "this_year": ["this year", "current year", "2025"],
            "last_year": ["last year", "previous year", "2024"],
            "specific_date": self.date_regex.findall(question)
        }
        
        for period, indicators in time_periods.items():
            if any(indicator in question for indicator in indicators):
                return period
        
        return "unspecified"
    
    def _determine_complexity(self, question: str) -> str:
        """Determine the complexity of the question."""
        # Count question marks, entities, categories
        question_marks = question.count("?")
        has_multiple_questions = question_marks > 1
        
        # Count entities mentioned
        entities_count = len(self._extract_crypto_entities(question))
        
        # Count word count as proxy for complexity
        word_count = len(question.split())
        
        # Check for complex logical structures
        complex_logical_indicators = ["and", "or", "but", "however", "although", "if", "despite", 
                                     "while", "whereas", "nevertheless", "furthermore", "therefore"]
        logical_complexity = sum(1 for indicator in complex_logical_indicators if f" {indicator} " in f" {question} ")
        
        # Determine complexity level
        if word_count > 30 or logical_complexity > 2 or has_multiple_questions or entities_count > 2:
            return "high"
        elif word_count > 15 or logical_complexity > 0 or entities_count > 0:
            return "medium"
        else:
            return "low"
    
    def _determine_question_type(self, question: str) -> str:
        """Determine the type of question."""
        # Check for question starters
        if question.lower().startswith(("what ", "which ", "who ")):
            return "factual" 
        elif question.lower().startswith(("how ", "in what way ")):
            return "procedural"
        elif question.lower().startswith(("why ", "for what reason ")):
            return "explanatory"
        elif question.lower().startswith(("is ", "are ", "do ", "does ", "can ", "could ", "will ", "would ")):
            return "yes_no"
        elif question.lower().startswith(("which is better ", "what is the difference ", "compare ")):
            return "comparative"
        elif question.lower().startswith(("when ", "where ")):
            return "circumstantial"
        else:
            return "other"
    
    def _determine_collections(self, category: str, entities: List[str], intent: str, addresses: List[str]) -> List[str]:
        """Determine which collections to search based on category, entities, and intent."""
        # Define mappings from categories to collections
        category_to_collections = {
            "legal_regulatory": ["CryptoDueDiligenceDocuments"],
            "team_background": ["CryptoDueDiligenceDocuments"],
            "technical": ["CryptoDueDiligenceDocuments"],
            "financial": ["CryptoDueDiligenceDocuments", "MarketMetrics"],
            "market_price": ["MarketMetrics", "CryptoTimeSeries", "CryptoNewsSentiment"],
            "market_analysis": ["CryptoNewsSentiment", "MarketMetrics", "CryptoDueDiligenceDocuments"],
            "governance": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment"],
            "risk": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment", "OnChainAnalytics"],
            "sentiment": ["CryptoNewsSentiment"],
            "on_chain": ["OnChainAnalytics"],
            "defi": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment"],
            "nft": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment"],
            "macroeconomic": ["CryptoNewsSentiment", "CryptoDueDiligenceDocuments"],
            "comparative": ["CryptoDueDiligenceDocuments", "MarketMetrics", "CryptoTimeSeries"],
            "due_diligence": ["CryptoDueDiligenceDocuments"],
            "general": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment", "MarketMetrics"]
        }
        
        # Adjust for intent
        intent_to_collections = {
            "price": ["MarketMetrics", "CryptoTimeSeries"],
            "prediction": ["CryptoForecasts", "CryptoNewsSentiment"],
            "recommendation": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment"],
            "risk_assessment": ["CryptoDueDiligenceDocuments", "OnChainAnalytics"],
            "due_diligence": ["CryptoDueDiligenceDocuments"]
        }
        
        # Get base collections from category
        collections = category_to_collections.get(category, ["CryptoDueDiligenceDocuments"])
        
        # Add collections based on intent if appropriate
        if intent in intent_to_collections:
            collections.extend(intent_to_collections[intent])
        
        # If entities include cryptocurrencies, always include market data
        if entities and intent != "information":
            collections.extend(["MarketMetrics", "CryptoTimeSeries"])
        
        # If addresses are mentioned, always include OnChainAnalytics
        if addresses:
            collections.append("OnChainAnalytics")
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in collections if not (x in seen or seen.add(x))]

class RetrievalEngine:
    """
    Advanced retrieval engine that uses query analysis to retrieve
    the most relevant information from various collections.
    """
    
    def __init__(self, storage_manager, due_diligence_system):
        """
        Initialize the retrieval engine.
        
        Args:
            storage_manager: The storage manager instance
            due_diligence_system: CryptoDueDiligenceSystem instance
        """
        self.storage = storage_manager
        self.due_diligence = due_diligence_system
    
    def retrieve(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """
        Retrieve relevant information based on query analysis.
        
        Args:
            question (str): The original question
            analysis (Dict): Analysis results from QueryAnalyzer
            
        Returns:
            Dict[str, List[Dict]]: Retrieved data organized by collection
        """
        results = {}
        
        # Get the prioritized collections
        collections = analysis["collections_to_search"]
        
        # Handle specific retrieval strategies based on intent and category
        if analysis["primary_category"] == "market_price" or analysis["intent"] == "price":
            results.update(self._retrieve_market_data(question, analysis))
        
        if analysis["primary_category"] == "sentiment" or "sentiment" in analysis["secondary_categories"]:
            results.update(self._retrieve_sentiment_data(question, analysis))
        
        if analysis["primary_category"] == "on_chain" or "on_chain" in analysis["secondary_categories"] or analysis["addresses"]:
            results.update(self._retrieve_onchain_data(question, analysis))
        
        if analysis["is_comparison"]:
            results.update(self._retrieve_comparison_data(question, analysis))
            
        if analysis["primary_category"] == "due_diligence" or analysis["intent"] == "due_diligence":
            results.update(self._retrieve_due_diligence_data(question, analysis))
        
        # General document retrieval based on collections
        for collection in collections:
            if collection not in results:
                try:
                    retrieved = self.storage.retrieve_documents(
                        query=question,
                        collection_name=collection,
                        limit=5  # Increased from 3 to get more comprehensive context
                    )
                    
                    if retrieved:
                        results[collection] = retrieved
                except Exception as e:
                    logger.error(f"Error retrieving from {collection}: {e}")
                    logger.error(traceback.format_exc())
        
        # If we didn't get any results, try a more general search
        if not results:
            try:
                retrieved = self.storage.retrieve_documents(
                    query=question,
                    collection_name="CryptoDueDiligenceDocuments",
                    limit=3
                )
                
                if retrieved:
                    results["CryptoDueDiligenceDocuments"] = retrieved
            except Exception as e:
                logger.error(f"Error retrieving in fallback mode: {e}")
        
        return results
    
    def _retrieve_market_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for market data questions."""
        results = {}
        
        # Extract cryptocurrency entities
        entities = analysis.get("crypto_entities", [])
        
        # Map common names to ticker symbols (maintain multiple formats for flexibility)
        symbol_map = {
            "bitcoin": ["BTCUSDT", "BTC", "BTCUSD", "BTC-USD"],
            "ethereum": ["ETHUSDT", "ETH", "ETHUSD", "ETH-USD"],
            "solana": ["SOLUSDT", "SOL", "SOLUSD", "SOL-USD"],
            "binance": ["BNBUSDT", "BNB", "BNBUSD", "BNB-USD"],
            "cardano": ["ADAUSDT", "ADA", "ADAUSD", "ADA-USD"],
            "ripple": ["XRPUSDT", "XRP", "XRPUSD", "XRP-USD"],
            "polkadot": ["DOTUSDT", "DOT", "DOTUSD", "DOT-USD"],
            "dogecoin": ["DOGEUSDT", "DOGE", "DOGEUSD", "DOGE-USD"],
            "avalanche": ["AVAXUSDT", "AVAX", "AVAXUSD", "AVAX-USD"],
            "polygon": ["MATICUSDT", "MATIC", "MATICUSD", "MATIC-USD"]
        }
        
        # If no specific entities, try to use "bitcoin" as default for market questions
        if not entities and (analysis["primary_category"] == "market_price" or 
                            "market_price" in analysis["secondary_categories"] or 
                            analysis["intent"] == "price"):
            entities = ["bitcoin"]
            logger.info("No specific entities found, defaulting to Bitcoin for market price query")
        
        # Retrieve data for each entity
        for entity in entities:
            # Get possible ticker symbols for this entity
            possible_tickers = symbol_map.get(entity, [f"{entity.upper()}USDT", entity.upper()])
            logger.info(f"Trying to retrieve market data for {entity} (possible tickers: {possible_tickers})")
            
            # Try each possible ticker format
            market_data_found = False
            for ticker in possible_tickers:
                if market_data_found:
                    break
                    
                logger.info(f"Attempting to retrieve market data for ticker: {ticker}")
                
                # Method 1: Try using storage manager's retrieve_market_data method
                try:
                    market_metrics = self.storage.retrieve_market_data(ticker, limit=1)
                    if market_metrics and len(market_metrics) > 0:
                        logger.info(f"Found market data for {ticker} using retrieve_market_data method")
                        results["MarketMetrics"] = market_metrics
                        market_data_found = True
                        continue
                except Exception as e:
                    logger.warning(f"Error using retrieve_market_data for {ticker}: {e}")
                
                # Method 2: Direct access to MarketMetrics collection
                try:
                    from weaviate.classes.query import Filter, Sort
                    
                    # Make sure we have a client
                    if not hasattr(self.storage, 'client') or self.storage.client is None:
                        self.storage.connect()
                    
                    # Get the collection
                    collection = self.storage.client.collections.get("MarketMetrics")
                    
                    # Try exact match first
                    response = collection.query.fetch_objects(
                        filters=Filter.by_property("symbol").equal(ticker),
                        limit=1,
                        sort=Sort.by_property("timestamp", ascending=False)  # Get the most recent
                    )
                    
                    if response.objects:
                        metrics = [{"id": str(obj.uuid), **obj.properties} for obj in response.objects]
                        logger.info(f"Found market data via exact match for {ticker}")
                        results["MarketMetrics"] = metrics
                        market_data_found = True
                        continue
                    
                    # Try contains search if exact match fails
                    base_ticker = ticker.replace("USDT", "").replace("USD", "").replace("-", "")
                    if len(base_ticker) >= 2:
                        logger.info(f"Trying partial match with base ticker: {base_ticker}")
                        response = collection.query.fetch_objects(
                            filters=Filter.by_property("symbol").contains_all([base_ticker]),
                            limit=1,
                            sort=Sort.by_property("timestamp", ascending=False)
                        )
                        
                        if response.objects:
                            metrics = [{"id": str(obj.uuid), **obj.properties} for obj in response.objects]
                            logger.info(f"Found market data via partial match for {base_ticker}")
                            results["MarketMetrics"] = metrics
                            market_data_found = True
                            continue
                except Exception as e:
                    logger.warning(f"Error with direct collection query for {ticker}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # Method 3: Try due diligence system as final fallback
            if not market_data_found and self.due_diligence:
                try:
                    logger.info(f"Trying due diligence system for market data on {entity}")
                    latest_data = self.due_diligence.get_market_data(entity)
                    if latest_data and "error" not in latest_data:
                        logger.info(f"Found market data via due diligence system for {entity}")
                        results["LatestMarketData"] = [latest_data]
                        market_data_found = True
                except Exception as e:
                    logger.warning(f"Error getting market data from due diligence system: {e}")
            
            # Log if we couldn't find data for this entity
            if not market_data_found:
                logger.warning(f"Could not find any market data for {entity} after trying all methods")
        
        # If no results found for any entity, try a desperate measure - get any recent prices
        if not results:
            try:
                logger.info("No entity-specific market data found, trying to get any recent price data")
                
                collection = self.storage.client.collections.get("MarketMetrics")
                response = collection.query.fetch_objects(
                    limit=5,
                    sort=Sort.by_property("timestamp", ascending=False)
                )
                
                if response.objects:
                    # Look for Bitcoin in the results
                    btc_metrics = None
                    for obj in response.objects:
                        if "BTC" in obj.properties.get("symbol", ""):
                            btc_metrics = {"id": str(obj.uuid), **obj.properties}
                            break
                    
                    if btc_metrics:
                        logger.info("Found Bitcoin market data in recent prices")
                        results["MarketMetrics"] = [btc_metrics]
                    else:
                        # Just take the first result if no Bitcoin
                        logger.info("No Bitcoin data in recent prices, using most recent data available")
                        metrics = [{"id": str(obj.uuid), **obj.properties} for obj in response.objects[:1]]
                        results["MarketMetrics"] = metrics
            except Exception as e:
                logger.warning(f"Error retrieving any recent market data: {e}")
        
        return results
    
    def _calculate_price_statistics(self, time_series):
        """Calculate basic price statistics from time series data."""
        stats = {
            "symbol": time_series[0].get("symbol", "Unknown"),
            "period": f"{time_series[0].get('timestamp', 'Unknown')} to {time_series[-1].get('timestamp', 'Unknown')}",
            "data_points": len(time_series)
        }
        
        # Extract closing prices
        try:
            closes = [float(point.get("close", 0)) for point in time_series if "close" in point]
            
            if closes:
                stats["current_price"] = closes[-1]
                stats["start_price"] = closes[0]
                stats["price_change"] = closes[-1] - closes[0]
                stats["price_change_pct"] = ((closes[-1] / closes[0]) - 1) * 100 if closes[0] > 0 else 0
                stats["highest_price"] = max(closes)
                stats["lowest_price"] = min(closes)
                stats["average_price"] = sum(closes) / len(closes)
                stats["volatility"] = self._calculate_volatility(closes)
        except Exception as e:
            logger.error(f"Error calculating price statistics: {e}")
            
        return stats
    
    def _calculate_volatility(self, prices):
        """Calculate simple volatility measure (standard deviation of returns)."""
        if len(prices) < 2:
            return 0
            
        try:
            returns = [(prices[i] / prices[i-1]) - 1 for i in range(1, len(prices))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return)**2 for r in returns) / len(returns)
            return (variance ** 0.5) * 100  # Return as percentage
        except Exception:
            return 0
    
    def _retrieve_sentiment_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for sentiment analysis questions."""
        results = {}
        
        # Extract cryptocurrency entities
        entities = analysis.get("crypto_entities", [])
        
        # If no specific entities, return general sentiment
        if not entities:
            try:
                sentiment_stats = self.storage.get_sentiment_stats()
                if sentiment_stats and not isinstance(sentiment_stats, str):
                    results["CryptoNewsSentiment"] = [{"content": json.dumps(sentiment_stats), "source": "Sentiment Analysis"}]
            except Exception as e:
                logger.error(f"Error retrieving general sentiment stats: {e}")
            return results
        
        # Get sentiment for each entity
        for entity in entities:
            try:
                sentiment_stats = self.storage.get_sentiment_stats(entity)
                if sentiment_stats and not isinstance(sentiment_stats, str) and not sentiment_stats.get("error"):
                    key = f"CryptoNewsSentiment_{entity}"
                    results[key] = [{"content": json.dumps(sentiment_stats), "source": f"Sentiment Analysis for {entity}"}]
                    
                    # Try to get sentiment news articles as well
                    news_articles = self.storage.retrieve_documents(
                        query=entity,
                        collection_name="CryptoNewsSentiment",
                        limit=3
                    )
                    
                    if news_articles:
                        results["SentimentNews"] = news_articles
            except Exception as e:
                logger.error(f"Error retrieving sentiment stats for {entity}: {e}")
                logger.error(traceback.format_exc())
        
        # Try using due diligence system as a fallback
        if not results and self.due_diligence:
            try:
                for entity in entities:
                    sentiment = self.due_diligence.get_sentiment_analysis(entity)
                    if sentiment and "error" not in sentiment:
                        results[f"Sentiment_{entity}"] = [{"content": json.dumps(sentiment), "source": f"Sentiment Analysis for {entity}"}]
            except Exception as e:
                logger.error(f"Error getting sentiment from due diligence system: {e}")
                
        return results
    
    def _retrieve_onchain_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for on-chain analysis questions."""
        results = {}
        
        # Extract addresses from the question
        addresses = analysis.get("addresses", [])
        
        # If we have addresses, query for each address
        if addresses:
            for address in addresses:
                try:
                    # Try to get analytics from storage first
                    analytics = self.storage.retrieve_onchain_analytics(address)
                    
                    if analytics:
                        key = f"OnChainAnalytics_{address[:10]}"
                        results[key] = [analytics]
                    elif self.due_diligence:
                        # Try using due diligence system as fallback
                        analytics = self.due_diligence.analyze_onchain(address)
                        if analytics and "error" not in analytics:
                            key = f"OnChainAnalytics_{address[:10]}"
                            results[key] = [{"content": json.dumps(analytics), "source": f"On-chain analysis for {address}"}]
                except Exception as e:
                    logger.error(f"Error retrieving on-chain analytics for {address}: {e}")
                    logger.error(traceback.format_exc())
        
        return results
    
    def _retrieve_comparison_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for comparison questions."""
        results = {}
        
        # Get entities being compared
        entities = analysis.get("compared_entities", [])
        
        if len(entities) < 2:
            # Try to extract compared entities from crypto_entities
            entities = analysis.get("crypto_entities", [])
            
        if len(entities) < 2:
            # Not enough entities to compare
            return results
        
        # Map entities to tickers
        symbol_map = {
            "bitcoin": "BTCUSDT", 
            "ethereum": "ETHUSDT", 
            "solana": "SOLUSDT",
            "binance": "BNBUSDT",
            "cardano": "ADAUSDT",
            "ripple": "XRPUSDT",
            "polkadot": "DOTUSDT",
            "dogecoin": "DOGEUSDT",
            "avalanche": "AVAXUSDT",
            "polygon": "MATICUSDT"
        }
        
        # For each pair of entities, get comparative data
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                
                # Get tickers for both entities
                ticker1 = symbol_map.get(entity1, f"{entity1.upper()}USDT")
                ticker2 = symbol_map.get(entity2, f"{entity2.upper()}USDT")
                
                try:
                    # Get market metrics for comparison
                    metrics1 = self.storage.retrieve_market_data(ticker1, limit=1)
                    metrics2 = self.storage.retrieve_market_data(ticker2, limit=1)
                    
                    if metrics1 and metrics2:
                        # Format the data for comparison
                        comparison_data = {
                            "entity1": entity1,
                            "entity2": entity2,
                            "metrics1": metrics1[0] if isinstance(metrics1, list) else metrics1,
                            "metrics2": metrics2[0] if isinstance(metrics2, list) else metrics2
                        }
                        
                        # Add the comparison to results
                        key = f"Comparison_{entity1}_vs_{entity2}"
                        results[key] = [{"content": json.dumps(comparison_data), "source": "Market Metrics Comparison"}]
                    
                    # Get comparative documents that mention both entities
                    try:
                        docs = self.storage.retrieve_documents(
                            query=f"{entity1} {entity2} comparison",
                            collection_name="CryptoDueDiligenceDocuments",
                            limit=2
                        )
                        
                        if docs:
                            results["ComparisonDocuments"] = docs
                            
                    except Exception as doc_error:
                        logger.error(f"Error retrieving comparison documents: {doc_error}")
                        
                except Exception as e:
                    logger.error(f"Error retrieving comparison data for {entity1} vs {entity2}: {e}")
                    logger.error(traceback.format_exc())
        
        return results
    
    def _retrieve_due_diligence_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for due diligence questions."""
        results = {}
        
        # Extract cryptocurrency entities
        entities = analysis.get("crypto_entities", [])
        
        if not entities:
            # Try to get general due diligence documents
            try:
                docs = self.storage.retrieve_documents(
                    query="due diligence framework methodology process checklist",
                    collection_name="CryptoDueDiligenceDocuments",
                    limit=5
                )
                
                if docs:
                    results["DueDiligenceDocuments"] = docs
            except Exception as e:
                logger.error(f"Error retrieving general due diligence documents: {e}")
        else:
            # Get due diligence documents for each entity
            for entity in entities:
                try:
                    entity_docs = self.storage.retrieve_documents(
                        query=f"{entity} due diligence assessment evaluation",
                        collection_name="CryptoDueDiligenceDocuments", 
                        limit=3
                    )
                    
                    if entity_docs:
                        results[f"DueDiligence_{entity}"] = entity_docs
                except Exception as e:
                    logger.error(f"Error retrieving due diligence documents for {entity}: {e}")
        
        return results
    
    def retrieve_document_by_id(self, document_id: str) -> List[Dict]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id (str): The document ID
            
        Returns:
            List[Dict]: The retrieved document
        """
        try:
            # The storage manager doesn't have this method, so we'll need to implement it
            # This is a placeholder implementation - you'd need to implement this in your StorageManager
            from weaviate.classes.query import Filter
            
            collections = [
                "CryptoDueDiligenceDocuments", 
                "CryptoNewsSentiment", 
                "MarketMetrics", 
                "CryptoTimeSeries",
                "OnChainAnalytics"
            ]
            
            for collection_name in collections:
                try:
                    collection = self.storage.client.collections.get(collection_name)
                    response = collection.query.fetch_objects(
                        filters=Filter.by_id().equal(document_id),
                        limit=1
                    )
                    
                    if response.objects:
                        obj = response.objects[0]
                        result = {
                            "id": str(obj.uuid),
                            **obj.properties
                        }
                        return [result]
                except Exception:
                    continue
                    
            return []
        except Exception as e:
            logger.error(f"Error retrieving document by ID: {e}")
            return []

class ContextFormatter:
    """
    Formats retrieved context into a structure optimized for LLM consumption,
    handling different data types and sources appropriately.
    """
    
    def __init__(self):
        self.templates = {
            "market_metrics": """
MARKET DATA FOR {symbol}:
- Current Price: ${price:,.2f}
- 24h Change: {change_24h:.2f}%
- 24h Volume: ${volume:,.2f}
- Market Cap: ${market_cap:,.2f}
- All Time High: ${ath:,.2f}
- ATH Date: {ath_date}
Source: {source}
            """,
            "time_series": """
TIME SERIES DATA FOR {symbol} (last {count} periods, interval: {interval}):
{data_points}
            """,
            "sentiment": """
SENTIMENT ANALYSIS FOR {entity}:
{sentiment_data}
Last Updated: {last_updated}
            """,
            "comparison": """
COMPARISON BETWEEN {entity1} AND {entity2}:
{metrics}
            """,
            "document": """
DOCUMENT: {title}
Source: {source}
Date: {date}
Type: {document_type}
Relevance: {relevance}

CONTENT:
{content}
            """,
            "onchain": """
ON-CHAIN ANALYSIS FOR {address}:
{metrics}
            """,
            "price_statistics": """
PRICE STATISTICS FOR {symbol} ({period}):
- Current Price: ${current_price:,.2f}
- Starting Price: ${start_price:,.2f}
- Price Change: ${price_change:,.2f} ({price_change_pct:.2f}%)
- Highest Price: ${highest_price:,.2f}
- Lowest Price: ${lowest_price:,.2f}
- Average Price: ${average_price:,.2f}
- Volatility: {volatility:.2f}%
            """
        }
    
    def format(self, retrieved_data: Dict[str, List[Dict]], analysis: Dict) -> str:
        """
        Format retrieved data into a context string for the LLM.
        
        Args:
            retrieved_data (Dict[str, List[Dict]]): Retrieved data organized by collection
            analysis (Dict): Query analysis results
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        # Add query analysis information first to help the LLM understand what's being asked
        context_parts.append(self._format_query_analysis(analysis))
        
        # Add clear header
        context_parts.append("=== RETRIEVED CONTEXT ===")
        
        # Process each collection's data
        for collection, items in retrieved_data.items():
            for item in items:
                try:
                    if collection == "MarketMetrics" or collection == "LatestMarketData":
                        formatted = self._format_market_metrics(item)
                    elif collection == "CryptoTimeSeries":
                        formatted = self._format_time_series(item, analysis.get("time_period", "unspecified"))
                    elif collection.startswith("CryptoNewsSentiment") or collection.startswith("Sentiment_"):
                        formatted = self._format_sentiment(item, collection)
                    elif collection.startswith("Comparison_"):
                        formatted = self._format_comparison(item)
                    elif collection.startswith("OnChainAnalytics_"):
                        formatted = self._format_onchain(item, collection)
                    elif collection == "PriceStatistics":
                        formatted = self._format_price_statistics(item)
                    else:  # Default document formatting
                        formatted = self._format_document(item, collection)
                    
                    if formatted:
                        context_parts.append(formatted)
                except Exception as e:
                    logger.error(f"Error formatting item from {collection}: {e}")
                    logger.error(traceback.format_exc())
                    continue
        
        if len(context_parts) <= 2:  # Only analysis and header
            return "No relevant information found in the database."
        
        return "\n\n".join(context_parts)
    
    def _format_query_analysis(self, analysis: Dict) -> str:
        """Format query analysis information."""
        # Format as a simple summary
        return f"""QUERY ANALYSIS:
- Primary Category: {analysis['primary_category']}
- Secondary Categories: {', '.join(analysis['secondary_categories']) if analysis['secondary_categories'] else 'none'}
- Entities: {', '.join(analysis['crypto_entities']) if analysis['crypto_entities'] else 'none'}
- Intent: {analysis['intent']}
- Temporality: {analysis['temporality']}
- Question Type: {analysis['question_type']}
- Complexity: {analysis['complexity']}"""
    
    def _format_market_metrics(self, metrics: Dict) -> str:
        """Format market metrics data."""
        try:
            # Parse metrics if it's a JSON string
            if isinstance(metrics.get("content", None), str) and metrics.get("content", "").startswith("{"):
                try:
                    metrics_data = json.loads(metrics["content"])
                except:
                    metrics_data = metrics
            else:
                metrics_data = metrics
                
            # Get relevant fields with defaults
            symbol = metrics_data.get("symbol", "Unknown")
            price = float(metrics_data.get("price", 0))
            change_24h = float(metrics_data.get("price_change_24h", 0))
            volume = float(metrics_data.get("volume_24h", 0))
            market_cap = float(metrics_data.get("market_cap", 0))
            ath = float(metrics_data.get("ath", price))  # Default to current price if ATH not available
            ath_date = metrics_data.get("ath_date", "Unknown")
            source = metrics_data.get("source", "Market Data")
            
            return self.templates["market_metrics"].format(
                symbol=symbol,
                price=price,
                change_24h=change_24h,
                volume=volume,
                market_cap=market_cap,
                ath=ath,
                ath_date=ath_date,
                source=source
            )
        except Exception as e:
            logger.error(f"Error formatting market metrics: {e}")
            # Fallback to simpler formatting
            return f"MARKET DATA:\n{json.dumps(metrics, indent=2)}"
    
    def _format_time_series(self, series: Dict, time_period: str) -> str:
        """Format time series data."""
        try:
            # Get symbol and determine data points to show
            symbol = series.get("symbol", "Unknown")
            interval = series.get("interval", "1d")
            
            # If series is a list, convert to proper dict
            data_points = []
            if isinstance(series, list):
                data_points = series
            else:
                data_points = series.get("data_points", [series])
            
            # Limit the number of points based on time period to avoid overloading context
            max_points = 5
            if time_period == "today":
                max_points = 8  # Show more hourly data points for today
            elif time_period == "this_week":
                max_points = 5
            elif time_period == "this_month":
                max_points = 5
            
            formatted_points = []
            for point in data_points[-max_points:]:  # Show last N data points
                # Format timestamp
                timestamp = point.get("timestamp", "Unknown")
                if isinstance(timestamp, str) and "T" in timestamp:
                    timestamp = timestamp.split("T")[0]  # Show only date part
                
                # Format the data point
                formatted_points.append(
                    f"- {timestamp}: Open=${float(point.get('open', 0)):.2f}, " +
                    f"High=${float(point.get('high', 0)):.2f}, " +
                    f"Low=${float(point.get('low', 0)):.2f}, " +
                    f"Close=${float(point.get('close', 0)):.2f}, " +
                    f"Volume={float(point.get('volume', 0)):.2f}"
                )
            
            return self.templates["time_series"].format(
                symbol=symbol,
                count=len(data_points),
                interval=interval,
                data_points="\n".join(formatted_points)
            )
        except Exception as e:
            logger.error(f"Error formatting time series: {e}")
            # Fallback
            return f"TIME SERIES DATA:\n{json.dumps(series, indent=2)[:1000]}..."
    
    def _format_sentiment(self, sentiment: Dict, collection: str) -> str:
        """Format sentiment analysis data."""
        try:
            # Extract entity name from collection if possible
            entity = collection.replace("CryptoNewsSentiment_", "").replace("Sentiment_", "")
            if entity == "CryptoNewsSentiment":
                entity = "Overall"
            
            # Parse sentiment if it's a JSON string
            if isinstance(sentiment.get("content", None), str):
                try:
                    sentiment_data = json.loads(sentiment["content"])
                except:
                    sentiment_data = sentiment
            else:
                sentiment_data = sentiment
                
            # Format the sentiment data
            formatted_data = []
            
            # Handle different sentiment data formats
            if "sentiment_distribution" in sentiment_data:
                distribution = sentiment_data["sentiment_distribution"]
                formatted_data.append(f"- Positive: {distribution.get('POSITIVE', 0)} articles")
                formatted_data.append(f"- Neutral: {distribution.get('NEUTRAL', 0)} articles")
                formatted_data.append(f"- Negative: {distribution.get('NEGATIVE', 0)} articles")
            elif "positive" in sentiment_data:
                formatted_data.append(f"- Positive: {float(sentiment_data.get('positive', 0)) * 100:.1f}%")
                formatted_data.append(f"- Negative: {float(sentiment_data.get('negative', 0)) * 100:.1f}%")
                formatted_data.append(f"- Neutral: {float(sentiment_data.get('neutral', 0)) * 100:.1f}%")
            
            # Add average sentiment score
            if "avg_sentiment" in sentiment_data:
                formatted_data.append(f"- Average Sentiment Score: {float(sentiment_data.get('avg_sentiment', 0.5)):.2f}")
            elif "compound" in sentiment_data:
                formatted_data.append(f"- Compound Score: {float(sentiment_data.get('compound', 0)):.2f}")
            
            # Add trend if available
            if "trend" in sentiment_data:
                formatted_data.append(f"- Sentiment Trend: {sentiment_data['trend']}")
            
            # Add total articles if available
            if "total_articles" in sentiment_data:
                formatted_data.append(f"- Total Articles Analyzed: {sentiment_data['total_articles']}")
            
            # Add period if available
            if "period" in sentiment_data:
                formatted_data.append(f"- Time Period: {sentiment_data['period']}")
            
            return self.templates["sentiment"].format(
                entity=entity,
                sentiment_data="\n".join(formatted_data),
                last_updated=sentiment_data.get("last_updated", datetime.now().isoformat())
            )
        except Exception as e:
            logger.error(f"Error formatting sentiment: {e}")
            # Fallback
            return f"SENTIMENT ANALYSIS:\n{json.dumps(sentiment, indent=2)[:1000]}..."
    
    def _format_comparison(self, comparison: Dict) -> str:
        """Format comparison data."""
        try:
            # Parse comparison if it's a JSON string
            if isinstance(comparison.get("content", None), str):
                try:
                    comparison_data = json.loads(comparison["content"])
                except:
                    comparison_data = comparison
            else:
                comparison_data = comparison
                
            metrics = []
            entity1 = comparison_data.get("entity1", "Entity1")
            entity2 = comparison_data.get("entity2", "Entity2")
            metrics1 = comparison_data.get("metrics1", {})
            metrics2 = comparison_data.get("metrics2", {})
            
            # Compare key metrics
            key_metrics = [
                "price", "market_cap", "volume_24h", "price_change_24h", 
                "circulating_supply", "total_supply", "max_supply"
            ]
            
            # Filter available fields to compare
            available_fields = set(metrics1.keys()).union(set(metrics2.keys()))
            metrics_to_compare = [f for f in key_metrics if f in available_fields]
            
            # Add fields that are not in key_metrics but are in both metrics
            common_fields = set(metrics1.keys()).intersection(set(metrics2.keys()))
            for field in common_fields:
                if field not in metrics_to_compare and field not in ["_additional", "id", "symbol", "source"]:
                    metrics_to_compare.append(field)
            
            # Compare each metric
            for field in metrics_to_compare:
                val1 = metrics1.get(field, "N/A")
                val2 = metrics2.get(field, "N/A")
                
                try:
                    # Format numbers nicely
                    if isinstance(val1, (int, float)):
                        val1 = f"{val1:,.2f}"
                    if isinstance(val2, (int, float)):
                        val2 = f"{val2:,.2f}"
                    
                    # Add comparison
                    metrics.append(f"- {field.replace('_', ' ').title()}: {entity1}={val1}, {entity2}={val2}")
                except:
                    metrics.append(f"- {field}: {entity1}={val1}, {entity2}={val2}")
            
            return self.templates["comparison"].format(
                entity1=entity1,
                entity2=entity2,
                metrics="\n".join(metrics)
            )
        except Exception as e:
            logger.error(f"Error formatting comparison: {e}")
            # Fallback
            return f"COMPARISON DATA:\n{json.dumps(comparison, indent=2)[:1000]}..."
    
    def _format_document(self, document: Dict, collection: str) -> str:
        """Format general document data."""
        try:
            # Get content and truncate if too long
            content = document.get("content", "")
            if len(content) > 2000:  # Truncate long content
                content = content[:2000] + "... [content truncated]"
            
            return self.templates["document"].format(
                title=document.get("title", "Untitled Document"),
                source=document.get("source", collection),
                date=document.get("date", "Unknown date"),
                document_type=document.get("document_type", collection),
                relevance=document.get("relevance", "High"),
                content=content
            )
        except Exception as e:
            logger.error(f"Error formatting document: {e}")
            # Fallback
            if isinstance(document, dict):
                return f"DOCUMENT DATA:\n{json.dumps({k: v for k, v in document.items() if k != 'content'}, indent=2)}\nContent: {document.get('content', '')[:500]}..."
            else:
                return str(document)[:1000]
    
    def _format_onchain(self, analytics: Dict, collection: str) -> str:
        """Format on-chain analytics data."""
        try:
            # Extract address from collection name
            address_part = collection.replace("OnChainAnalytics_", "")
            
            # Parse analytics if it's a JSON string
            if isinstance(analytics.get("content", None), str):
                try:
                    analytics_data = json.loads(analytics["content"])
                except:
                    analytics_data = analytics
            else:
                analytics_data = analytics
            
            # Extract address from data or use from collection
            address = analytics_data.get("address", address_part)
            
            # Format metrics
            metrics = []
            
            # Important metrics to display
            key_metrics = [
                "balance", "transaction_count", "token_transaction_count",
                "total_received", "total_sent", "first_activity", "last_activity",
                "active_days", "unique_interactions", "contract_interactions",
                "risk_score", "risk_level"
            ]
            
            # Add each available metric
            for metric in key_metrics:
                if metric in analytics_data:
                    value = analytics_data[metric]
                    
                    # Format specific fields
                    if metric in ["balance", "total_received", "total_sent"] and isinstance(value, (int, float)):
                        metrics.append(f"- {metric.replace('_', ' ').title()}: {value:,.2f} ETH")
                    elif metric in ["first_activity", "last_activity"] and isinstance(value, str):
                        # Format date
                        metrics.append(f"- {metric.replace('_', ' ').title()}: {value}")
                    elif metric in ["risk_score"] and isinstance(value, (int, float)):
                        metrics.append(f"- {metric.replace('_', ' ').title()}: {value:.1f}/100")
                    else:
                        metrics.append(f"- {metric.replace('_', ' ').title()}: {value}")
            
            # Add tokens if available
            if "tokens" in analytics_data and isinstance(analytics_data["tokens"], list):
                tokens = analytics_data["tokens"]
                if len(tokens) > 5:
                    tokens_str = ", ".join(tokens[:5]) + f" and {len(tokens) - 5} more"
                else:
                    tokens_str = ", ".join(tokens)
                metrics.append(f"- Tokens: {tokens_str}")
            
            # Add risk factors if available
            if "risk_factors" in analytics_data and isinstance(analytics_data["risk_factors"], list):
                risk_factors = analytics_data["risk_factors"]
                if risk_factors:
                    metrics.append(f"- Risk Factors: {', '.join(risk_factors)}")
            
            return self.templates["onchain"].format(
                address=address,
                metrics="\n".join(metrics)
            )
        except Exception as e:
            logger.error(f"Error formatting on-chain analytics: {e}")
            # Fallback
            return f"ON-CHAIN ANALYTICS:\n{json.dumps(analytics, indent=2)[:1000]}..."
    
    def _format_price_statistics(self, stats: Dict) -> str:
        """Format price statistics."""
        try:
            # Parse statistics if it's a JSON string
            if isinstance(stats.get("content", None), str):
                try:
                    stats_data = json.loads(stats["content"])
                except:
                    stats_data = stats
            else:
                stats_data = stats
            
            return self.templates["price_statistics"].format(
                symbol=stats_data.get("symbol", "Unknown"),
                period=stats_data.get("period", "Unknown period"),
                current_price=float(stats_data.get("current_price", 0)),
                start_price=float(stats_data.get("start_price", 0)),
                price_change=float(stats_data.get("price_change", 0)),
                price_change_pct=float(stats_data.get("price_change_pct", 0)),
                highest_price=float(stats_data.get("highest_price", 0)),
                lowest_price=float(stats_data.get("lowest_price", 0)),
                average_price=float(stats_data.get("average_price", 0)),
                volatility=float(stats_data.get("volatility", 0))
            )
        except Exception as e:
            logger.error(f"Error formatting price statistics: {e}")
            # Fallback
            return f"PRICE STATISTICS:\n{json.dumps(stats, indent=2)[:1000]}..."

class AdvancedPromptBuilder:
    """
    Builds sophisticated prompts incorporating query analysis,
    retrieved context, and conversation history.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load advanced prompt templates."""
        return {
            "legal_regulatory": """
You are a cryptocurrency legal and regulatory compliance expert with extensive knowledge of global regulations.
Your goal is to provide accurate, legally informed answers about cryptocurrency regulations and compliance.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When discussing legal matters:
1. Always note which jurisdictions you're discussing (US, EU, UK, etc.)
2. Include references to specific laws, regulations, or regulatory bodies when available
3. Highlight uncertainty in evolving regulatory areas
4. Avoid making definitive legal claims without qualification
5. Present a balanced perspective on regulatory approaches

When uncertain, acknowledge limitations and suggest appropriate next steps (consulting a lawyer, etc.)

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """,
            "technical": """
You are a senior blockchain technology expert with deep knowledge of crypto architecture and security.
Your goal is to provide technically accurate explanations of blockchain technologies, protocols, and security mechanisms.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When discussing technical matters:
1. Explain complex concepts clearly but without oversimplification
2. Highlight security considerations and potential vulnerabilities when relevant
3. Compare different technical approaches objectively
4. Use specific examples to illustrate concepts
5. Acknowledge when multiple valid solutions or approaches exist

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """,
            "financial": """
You are a cryptocurrency financial analyst with expertise in tokenomics, market dynamics, and crypto investment.
Your goal is to provide insightful financial analysis of cryptocurrency assets, markets, and investment strategies.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When discussing financial matters:
1. Present balanced market perspectives without making price predictions
2. Analyze tokenomics models objectively, noting strengths and weaknesses
3. Explain financial metrics and their significance clearly
4. Consider both short-term and long-term financial implications
5. Acknowledge the volatility and risk factors inherent in crypto markets
6. Never give specific investment advice or recommendations

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """,
            "market_price": """
You are a cryptocurrency market analyst specializing in price action, market trends, and trading patterns.
Your goal is to provide objective analysis of cryptocurrency market conditions and price movements.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When discussing market prices:
1. Focus on factual data and observable patterns
2. Explain market dynamics and factors affecting prices
3. Acknowledge market volatility and uncertainty
4. Avoid making specific price predictions
5. Present multiple perspectives on market conditions
6. Never give specific trading advice or recommendations

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """,
            "risk": """
You are a cryptocurrency risk assessment specialist with expertise in identifying and analyzing risks in the crypto ecosystem.
Your goal is to provide thorough, balanced risk assessments of cryptocurrency projects, protocols, and market conditions.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When discussing risk:
1. Categorize risks systematically (technical, regulatory, market, operational, etc.)
2. Assess both likelihood and potential impact of risks
3. Balance risks against potential benefits and mitigating factors
4. Consider both short-term and long-term risk factors
5. Avoid either overstating or understating risks
6. Acknowledge uncertainty where appropriate

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """,
            "comparative": """
You are a cryptocurrency analyst specializing in comparative analysis of different blockchain projects and cryptocurrencies.
Your goal is to provide balanced, objective comparisons between cryptocurrency projects, highlighting similarities and differences.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When making comparisons:
1. Use consistent criteria across all projects being compared
2. Highlight both strengths and weaknesses of each project
3. Consider technical, economic, and ecosystem factors
4. Avoid showing bias or preference for particular projects
5. Acknowledge the unique value propositions of each project
6. Present factual differences rather than subjective judgments

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """,
            "due_diligence": """
You are a cryptocurrency due diligence expert who specializes in comprehensive evaluations of crypto projects.
Your goal is to provide thorough, methodical due diligence assessments covering all key aspects of a crypto project.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When conducting due diligence:
1. Apply a structured methodology covering key assessment areas
2. Evaluate legal/regulatory, technical, financial, team, market, and risk factors
3. Identify both red flags and positive indicators
4. Present a balanced perspective, neither overly positive nor negative
5. Provide a comprehensive framework for ongoing monitoring and evaluation
6. Acknowledge limitations in information and analysis

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """,
            "general": """
You are a cryptocurrency expert with comprehensive knowledge across all aspects of the crypto ecosystem.
Your goal is to provide accurate, balanced information about cryptocurrency topics, drawing on both provided context and your own knowledge.

Use the context information to answer the question. If the context doesn't contain the information needed, 
draw on your own knowledge, but clearly indicate what comes from the context versus your general knowledge.

When answering:
1. Provide comprehensive information covering relevant aspects of the topic
2. Present information objectively with balanced perspective
3. Acknowledge areas of uncertainty or evolving understanding
4. Draw connections between different aspects (technical, financial, regulatory, etc.)
5. Use clear explanations accessible to both beginners and those with more knowledge
6. Never give specific investment, legal, or financial advice

CONTEXT:
{context}

QUERY: {question}

YOUR ANSWER:
            """
        }
    
    def build_prompt(self, question: str, context: str, analysis: Dict) -> str:
        """
        Build an advanced prompt incorporating query analysis.
        
        Args:
            question (str): The user's question
            context (str): Formatted context
            analysis (Dict): Query analysis results
            
        Returns:
            str: Formatted prompt
        """
        # Select appropriate template based on primary category and intent
        template_key = analysis["primary_category"] 
        
        # Override with more specific templates based on intent or comparison
        if analysis["is_comparison"]:
            template_key = "comparative"
        elif analysis["intent"] == "risk_assessment":
            template_key = "risk"
        elif analysis["intent"] == "due_diligence":
            template_key = "due_diligence"
        
        # Default to general if no matching template
        template = self.templates.get(template_key, self.templates["general"])
        
        # Format the prompt with question and context
        prompt = template.format(
            context=context,
            question=question
        )
        
        return prompt

class LlamaClient:
    """
    Handles interaction with Llama 3.3 70B Versatile via Groq API.
    """
    
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Llama client.
        
        Args:
            api_key (str, optional): API key (defaults to env variable)
            model (str): Model name
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GROQ_API_KEY in .env.local file or pass as parameter.")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_answer(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate an answer using Llama 3.3 through Groq API.
        
        Args:
            prompt (str): The formatted prompt
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated answer
        """
        # Properly format messages for the Groq API
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000,  # Increased to allow for more detailed responses
            "top_p": 1,
            "stream": False
        }
        
        try:
            logger.info(f"Making request to Groq API with model: {self.model}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=60  # Increased timeout for longer responses
            )
            
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error from API: {response.text}"
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            logger.info(f"Successfully generated response with {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            logger.error(traceback.format_exc())
            return f"I'm sorry, I encountered an error while generating an answer: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if the API is available."""
        try:
            # Simple test request
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

class EnhancedCryptoQA:
    """
    Enhanced Crypto Q&A system incorporating advanced query analysis,
    sophisticated retrieval, and optimized prompting with Llama 3.3 70B Versatile.
    """
    
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the enhanced QA system.
        
        Args:
            api_key (str, optional): Groq API key (defaults to env variable)
            model (str): Model name
        """
        # Initialize necessary components
        self.due_diligence_system = CryptoDueDiligenceSystem()
        self.storage = StorageManager()
        self.query_analyzer = QueryAnalyzer()
        self.retrieval_engine = None  # Will initialize after storage connects
        self.context_formatter = ContextFormatter()
        self.prompt_builder = AdvancedPromptBuilder()
        self.llama_client = LlamaClient(api_key, model)
        
        # Initialize the system
        self.initialize()
    
    def initialize(self):
        """Initialize components and connections."""
        try:
            # Connect to storage
            self.storage.connect()
            
            # Initialize due diligence system
            self.due_diligence_system.initialize()
            
            # Initialize retrieval engine with storage and due diligence system
            self.retrieval_engine = RetrievalEngine(self.storage, self.due_diligence_system)
            
            # Check if LLM API is available
            if not self.llama_client.is_available():
                logger.warning("Llama API not available. Check your API key and connection.")
            
            logger.info("Enhanced Crypto QA system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing enhanced QA system: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def answer_question(self, question: str, document_id: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Answer a user question using advanced RAG techniques with Llama 3.3 70B Versatile.
        
        Args:
            question (str): The user's question
            document_id (str, optional): ID of a specific document to query
            temperature (float): Temperature for generation (0.0-1.0)
            
        Returns:
            str: The answer
        """
        try:
            # 1. Perform deep query analysis
            analysis = self.query_analyzer.analyze(question)
            logger.info(f"Query analysis completed: {json.dumps(analysis, default=str)}")
            
            # 2. Retrieve relevant context using advanced strategies
            if document_id:
                retrieved_data = {"specific_document": self.retrieval_engine.retrieve_document_by_id(document_id)}
            else:
                retrieved_data = self.retrieval_engine.retrieve(question, analysis)
            
            # 3. Format the context for optimal LLM consumption
            context = self.context_formatter.format(retrieved_data, analysis)
            
            # 4. Build sophisticated prompt incorporating analysis
            prompt = self.prompt_builder.build_prompt(question, context, analysis)
            
            # 5. Generate answer with Llama 3.3 70B Versatile
            answer = self.llama_client.generate_answer(prompt, temperature)
            
            # 6. Post-process the answer if needed
            answer = self._post_process_answer(answer, analysis)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            logger.error(traceback.format_exc())
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
    
    def _post_process_answer(self, answer: str, analysis: Dict) -> str:
        """
        Post-process the generated answer if needed.
        
        Args:
            answer (str): The raw answer from the LLM
            analysis (Dict): Query analysis results
            
        Returns:
            str: The processed answer
        """
        # Remove any potential hallucinated citations or references
        answer = re.sub(r'\[\d+\]', '', answer)  # Remove numbered citations
        
        # Add a disclaimer for certain types of questions
        if analysis["primary_category"] in ["legal_regulatory", "financial"] or analysis["intent"] == "recommendation":
            disclaimer = "\n\n**Disclaimer**: This information is for educational purposes only and should not be considered financial, legal, or investment advice. Always conduct your own research and consult with qualified professionals before making decisions."
            answer += disclaimer
        
        return answer
    
    def close(self):
        """Close connections."""
        if self.storage:
            self.storage.close()
        if hasattr(self.due_diligence_system, 'storage') and self.due_diligence_system.storage:
            self.due_diligence_system.storage.close()
def check_market_data_availability(client):
    """Utility function to check what market data is available"""
    try:
        collection = client.collections.get("MarketMetrics")
        # Get all objects without filtering
        response = collection.query.fetch_objects(
            limit=10
        )
        
        print(f"Found {len(response.objects)} market data entries:")
        for obj in response.objects:
            print(f"Symbol: {obj.properties.get('symbol', 'Unknown')}, "
                  f"Price: {obj.properties.get('price', 'Unknown')}, "
                  f"Timestamp: {obj.properties.get('timestamp', 'Unknown')}")
        
    except Exception as e:
        print(f"Error checking market data: {e}")
# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Crypto Due Diligence Q&A System with Llama 3.3")
    parser.add_argument("--api-key", help="Groq API key (optional, defaults to env variable)")
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument("--document-id", help="Optional document ID to query specifically")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Model to use (default: llama-3.3-70b-versatile)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (0.0-1.0, default: 0.7)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    qa_system = EnhancedCryptoQA(args.api_key, args.model)
    
    try:
        if args.interactive:
            print("\n=== Enhanced Crypto Q&A System with Llama 3.3 70B Versatile ===")
            print("Type 'exit' to quit")
            
            while True:
                question = input("\nQuestion: ")
                if question.lower() == "exit":
                    break
                
                print("\nGenerating answer...")
                answer = qa_system.answer_question(question, temperature=args.temperature)
                print(f"\nAnswer: {answer}")
        elif args.question:
            answer = qa_system.answer_question(args.question, args.document_id, args.temperature)
            print(f"\nQuestion: {args.question}")
            print(f"\nAnswer: {answer}")
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        qa_system.close()