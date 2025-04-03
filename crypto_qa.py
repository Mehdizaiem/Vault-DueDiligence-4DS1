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
# We'll use your agentic_rag.py functions for relevant features
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
        collections_to_search = self._determine_collections(primary_category, crypto_entities, intent)
        
        # Combine all analysis results
        analysis = {
            "primary_category": primary_category,
            "secondary_categories": secondary_categories,
            "crypto_entities": crypto_entities,
            "dates": dates,
            "numbers": numbers,
            "temporality": temporality,
            "is_comparison": is_comparison,
            "compared_entities": compared_entities,
            "intent": intent,
            "time_period": time_period,
            "collections_to_search": collections_to_search
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
                                "higher", "lower", "more", "less", "or", "rather than"]
        
        return any(indicator in question for indicator in comparison_indicators)
    
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
            "comparison": ["compare", "comparison", "versus", "vs", "difference", "similar", "better", "worse"]
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
    
    def _determine_collections(self, category: str, entities: List[str], intent: str) -> List[str]:
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
            "general": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment", "MarketMetrics"]
        }
        
        # Adjust for intent
        intent_to_collections = {
            "price": ["MarketMetrics", "CryptoTimeSeries"],
            "prediction": ["CryptoForecasts", "CryptoNewsSentiment"],
            "recommendation": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment"]
        }
        
        # Get base collections from category
        collections = category_to_collections.get(category, ["CryptoDueDiligenceDocuments"])
        
        # Add collections based on intent if appropriate
        if intent in intent_to_collections:
            collections.extend(intent_to_collections[intent])
        
        # If entities include cryptocurrencies, always include market data
        if entities and intent != "information":
            collections.extend(["MarketMetrics", "CryptoTimeSeries"])
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in collections if not (x in seen or seen.add(x))]

class RetrievalEngine:
    """
    Advanced retrieval engine that uses query analysis to retrieve
    the most relevant information from various collections.
    """
    
    def __init__(self, storage_manager):
        """
        Initialize the retrieval engine.
        
        Args:
            storage_manager: The storage manager instance
        """
        self.storage = storage_manager
    
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
        
        if analysis["primary_category"] == "on_chain" or "on_chain" in analysis["secondary_categories"]:
            results.update(self._retrieve_onchain_data(question, analysis))
        
        if analysis["is_comparison"]:
            results.update(self._retrieve_comparison_data(question, analysis))
        
        # General document retrieval based on collections
        for collection in collections:
            if collection not in results:
                try:
                    retrieved = self.storage.retrieve_documents(
                        query=question,
                        collection_name=collection,
                        limit=3
                    )
                    
                    if retrieved:
                        results[collection] = retrieved
                except Exception as e:
                    logger.error(f"Error retrieving from {collection}: {e}")
        
        return results
    
    def _retrieve_market_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for market data questions."""
        results = {}
        
        # Extract cryptocurrency entities
        entities = analysis.get("crypto_entities", [])
        
        # Map common names to ticker symbols
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
        
        # If no specific entities, try to use "bitcoin" as default for market questions
        if not entities and analysis["primary_category"] == "market_price":
            entities = ["bitcoin"]
        
        # Retrieve data for each entity
        for entity in entities:
            ticker = symbol_map.get(entity, f"{entity.upper()}USDT")
            
            # Time interval based on analysis
            interval = "1d"  # default
            time_period = analysis.get("time_period", "unspecified")
            
            if time_period in ["today", "yesterday"]:
                interval = "1h"
            elif time_period in ["this_week", "last_week"]:
                interval = "1d"
            elif time_period in ["this_month", "last_month"]:
                interval = "1d"
            
            # Limit based on time period
            limit = 7  # default
            
            if time_period == "today":
                limit = 24
            elif time_period == "this_week":
                limit = 7
            elif time_period == "this_month":
                limit = 30
            elif time_period == "this_year":
                limit = 365
            
            # Get market metrics
            try:
                market_metrics = self.storage.retrieve_market_data(ticker, limit=3)
                if market_metrics:
                    results["MarketMetrics"] = market_metrics
            except Exception as e:
                logger.error(f"Error retrieving market metrics for {ticker}: {e}")
            
            # Get time series data
            try:
                time_series = self.storage.retrieve_time_series(ticker, interval=interval, limit=limit)
                if time_series:
                    results["CryptoTimeSeries"] = time_series
            except Exception as e:
                logger.error(f"Error retrieving time series for {ticker}: {e}")
            
            # Get forecasts if available and intent is related to prediction
            if analysis["intent"] == "prediction" or analysis["temporality"] == "future":
                try:
                    # This assumes you have a retrieve_forecasts method
                    forecasts = self._retrieve_forecasts(ticker)
                    if forecasts:
                        results["CryptoForecasts"] = forecasts
                except Exception as e:
                    logger.error(f"Error retrieving forecasts for {ticker}: {e}")
        
        return results
    
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
                if sentiment_stats and not isinstance(sentiment_stats, str):
                    key = f"CryptoNewsSentiment_{entity}"
                    results[key] = [{"content": json.dumps(sentiment_stats), "source": f"Sentiment Analysis for {entity}"}]
            except Exception as e:
                logger.error(f"Error retrieving sentiment stats for {entity}: {e}")
        
        return results
    
    def _retrieve_onchain_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for on-chain analysis questions."""
        results = {}
        
        # Extract cryptocurrency entities and look for addresses
        entities = analysis.get("crypto_entities", [])
        
        # This is a placeholder - you would need to implement address extraction
        addresses = self._extract_addresses_from_question(question)
        
        # If we have addresses, query for each address
        if addresses:
            for address in addresses:
                try:
                    # This assumes you have a retrieve_onchain_analytics method
                    analytics = self._retrieve_onchain_analytics(address)
                    if analytics:
                        key = f"OnChainAnalytics_{address[:10]}"
                        results[key] = [analytics]
                except Exception as e:
                    logger.error(f"Error retrieving on-chain analytics for {address}: {e}")
        
        return results
    
    def _retrieve_comparison_data(self, question: str, analysis: Dict) -> Dict[str, List[Dict]]:
        """Specialized retrieval for comparison questions."""
        results = {}
        
        # Get entities being compared
        entities = analysis.get("compared_entities", [])
        
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
        
        # Retrieve comparison data for each entity pair
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                
                # Get market data for both entities
                ticker1 = symbol_map.get(entity1, f"{entity1.upper()}USDT")
                ticker2 = symbol_map.get(entity2, f"{entity2.upper()}USDT")
                
                try:
                    # Retrieve metrics for both entities
                    metrics1 = self.storage.retrieve_market_data(ticker1, limit=1)
                    metrics2 = self.storage.retrieve_market_data(ticker2, limit=1)
                    
                    if metrics1 and metrics2:
                        comparison_data = {
                            "entity1": entity1,
                            "entity2": entity2,
                            "metrics1": metrics1[0],
                            "metrics2": metrics2[0]
                        }
                        
                        key = f"Comparison_{entity1}_vs_{entity2}"
                        results[key] = [{"content": json.dumps(comparison_data), "source": "Market Metrics Comparison"}]
                except Exception as e:
                    logger.error(f"Error retrieving comparison data for {entity1} vs {entity2}: {e}")
        
        return results
    
    def _retrieve_forecasts(self, ticker: str) -> List[Dict]:
        """Retrieve forecast data for a given ticker (placeholder implementation)."""
        # This would be implemented based on your forecast data storage
        # For now, return an empty list
        return []
    
    def _retrieve_onchain_analytics(self, address: str) -> Dict:
        """Retrieve on-chain analytics for a given address (placeholder implementation)."""
        # This would be implemented based on your on-chain data storage
        # For now, return an empty dict
        return {}
    
    def _extract_addresses_from_question(self, question: str) -> List[str]:
        """Extract cryptocurrency addresses from the question (placeholder implementation)."""
        # This would use regex patterns to identify addresses for different chains
        # For now, return an empty list
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
            """,
            "time_series": """
TIME SERIES DATA FOR {symbol} (last {count} periods):
{data_points}
            """,
            "sentiment": """
SENTIMENT ANALYSIS FOR {entity}:
- Positive: {positive:.1f}%
- Negative: {negative:.1f}%
- Neutral: {neutral:.1f}%
- Compound Score: {compound:.2f}
- Last Updated: {last_updated}
            """,
            "comparison": """
COMPARISON BETWEEN {entity1} AND {entity2}:
{metrics}
            """,
            "document": """
DOCUMENT: {title}
Source: {source}
Date: {date}
Content: {content}
            """
        }
    
    def format(self, retrieved_data: Dict[str, List[Dict]]) -> str:
        """
        Format retrieved data into a context string for the LLM.
        
        Args:
            retrieved_data (Dict[str, List[Dict]]): Retrieved data organized by collection
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        for collection, items in retrieved_data.items():
            for item in items:
                try:
                    if collection == "MarketMetrics":
                        formatted = self._format_market_metrics(item)
                    elif collection == "CryptoTimeSeries":
                        formatted = self._format_time_series(item)
                    elif collection.startswith("CryptoNewsSentiment"):
                        formatted = self._format_sentiment(item, collection)
                    elif collection.startswith("Comparison"):
                        formatted = self._format_comparison(item)
                    else:  # Default document formatting
                        formatted = self._format_document(item)
                    
                    if formatted:
                        context_parts.append(formatted)
                except Exception as e:
                    logger.error(f"Error formatting item from {collection}: {e}")
                    continue
        
        if not context_parts:
            return "No relevant information found in the database."
        
        return "\n\n".join(context_parts)
    
    def _format_market_metrics(self, metrics: Dict) -> str:
        """Format market metrics data."""
        return self.templates["market_metrics"].format(
            symbol=metrics.get("symbol", "Unknown"),
            price=float(metrics.get("price", 0)),
            change_24h=float(metrics.get("change_24h", 0)),
            volume=float(metrics.get("volume_24h", 0)),
            market_cap=float(metrics.get("market_cap", 0)),
            ath=float(metrics.get("ath", 0)),
            ath_date=metrics.get("ath_date", "Unknown")
        )
    
    def _format_time_series(self, series: Dict) -> str:
        """Format time series data."""
        data_points = series.get("data_points", [])
        formatted_points = []
        
        for point in data_points[-5:]:  # Show last 5 data points
            formatted_points.append(
                f"- {point['timestamp']}: Open=${point['open']:.2f}, High=${point['high']:.2f}, "
                f"Low=${point['low']:.2f}, Close=${point['close']:.2f}, Volume={point['volume']:.2f}"
            )
        
        return self.templates["time_series"].format(
            symbol=series.get("symbol", "Unknown"),
            count=len(data_points),
            data_points="\n".join(formatted_points)
        )
    
    def _format_sentiment(self, sentiment: Dict, collection: str) -> str:
        """Format sentiment analysis data."""
        # Extract entity name from collection if possible
        entity = collection.replace("CryptoNewsSentiment_", "") if "_" in collection else "Overall"
        
        return self.templates["sentiment"].format(
            entity=entity,
            positive=float(sentiment.get("positive", 0)) * 100,
            negative=float(sentiment.get("negative", 0)) * 100,
            neutral=float(sentiment.get("neutral", 0)) * 100,
            compound=float(sentiment.get("compound", 0)),
            last_updated=sentiment.get("last_updated", "Unknown")
        )
    
    def _format_comparison(self, comparison: Dict) -> str:
        """Format comparison data."""
        metrics = []
        entity1 = comparison.get("entity1", "Entity1")
        entity2 = comparison.get("entity2", "Entity2")
        metrics1 = comparison.get("metrics1", {})
        metrics2 = comparison.get("metrics2", {})
        
        # Compare all available metrics
        for key in set(metrics1.keys()).union(set(metrics2.keys())):
            if key in ["_additional", "id"]:  # Skip metadata fields
                continue
                
            val1 = metrics1.get(key, "N/A")
            val2 = metrics2.get(key, "N/A")
            
            try:
                # Try to format numbers nicely
                if isinstance(val1, (int, float)):
                    val1 = f"{val1:,.2f}"
                if isinstance(val2, (int, float)):
                    val2 = f"{val2:,.2f}"
            except:
                pass
                
            metrics.append(f"- {key}: {entity1}={val1}, {entity2}={val2}")
        
        return self.templates["comparison"].format(
            entity1=entity1,
            entity2=entity2,
            metrics="\n".join(metrics)
        )
    
    def _format_document(self, document: Dict) -> str:
        """Format general document data."""
        content = document.get("content", "")
        if len(content) > 1500:  # Truncate long content
            content = content[:1500] + "..."
        
        return self.templates["document"].format(
            title=document.get("title", "Untitled Document"),
            source=document.get("source", "Unknown source"),
            date=document.get("date", "Unknown date"),
            content=content
        )

class AdvancedPromptBuilder:
    """
    Builds sophisticated prompts incorporating query analysis,
    retrieved context, and conversation history.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        self.query_analyzer = QueryAnalyzer()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load advanced prompt templates."""
        return {
            "legal_regulatory": """
You are an expert cryptocurrency legal and regulatory compliance analyst. 
Your task is to provide detailed, accurate answers to questions about crypto regulations.
Use the following guidelines:
1. Always cite specific laws, regulations, or documents when possible
2. Highlight jurisdictional differences when relevant
3. Explain complex legal concepts in simple terms
4. Clearly state when information is uncertain or varies by jurisdiction

CONTEXT:
{context}

QUESTION ANALYSIS:
- Primary Category: {primary_category}
- Secondary Categories: {secondary_categories}
- Entities: {entities}
- Intent: {intent}
- Temporality: {temporality}

QUESTION: {question}

ANSWER:
            """,
            "technical": """
You are a senior blockchain security engineer and technical analyst.
Your task is to provide expert-level technical analysis of cryptocurrency projects.
Use the following guidelines:
1. Explain technical concepts clearly but accurately
2. Highlight security considerations and risks
3. Compare different technical approaches when relevant
4. Provide specific examples from the context when possible
5. Clearly state when information is uncertain or unknown

CONTEXT:
{context}

QUESTION ANALYSIS:
- Primary Category: {primary_category}
- Secondary Categories: {secondary_categories}
- Entities: {entities}
- Intent: {intent}
- Temporality: {temporality}

QUESTION: {question}

ANSWER:
            """,
            # Additional templates for other categories...
            "general": """
You are a cryptocurrency due diligence expert with deep knowledge across all aspects
of crypto projects including legal, technical, financial, market, governance, and risk.
Your task is to provide comprehensive, balanced answers to cryptocurrency questions.
Use the following guidelines:
1. Structure your answer logically
2. Cover all relevant aspects mentioned in the context
3. Highlight uncertainties or missing information
4. Provide actionable insights when possible

CONTEXT:
{context}

QUESTION ANALYSIS:
- Primary Category: {primary_category}
- Secondary Categories: {secondary_categories}
- Entities: {entities}
- Intent: {intent}
- Temporality: {temporality}

QUESTION: {question}

ANSWER:
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
        # Get template based on primary category
        template = self.templates.get(analysis["primary_category"], self.templates["general"])
        
        # Format the prompt
        prompt = template.format(
            context=context,
            primary_category=analysis["primary_category"],
            secondary_categories=", ".join(analysis["secondary_categories"]),
            entities=", ".join(analysis["crypto_entities"]),
            intent=analysis["intent"],
            temporality=analysis["temporality"],
            question=question
        )
        
        return prompt
class MistralClient:
    """
    Handles interaction with Mistral API via Groq.
    """
    
    def __init__(self, api_key: str = None, model: str = "mistral-saba-24b"):
        """
        Initialize the Mistral client.
        
        Args:
            api_key (str, optional): API key (defaults to env variable)
            model (str): Model name - using mistral-saba-24b which is currently supported
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
    
    def generate_answer(self, prompt: str, temperature: float = 1.0) -> str:
        """
        Generate an answer using Mistral through Groq API.
        
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
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False
        }
        
        try:
            logger.info(f"Making request to Groq API with model: {self.model}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data
            )
            
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error from API: {response.text}"
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I'm sorry, I encountered an error while generating an answer: {str(e)}"
        
class EnhancedCryptoQA:
    """
    Enhanced Crypto Q&A system incorporating advanced query analysis,
    sophisticated retrieval, and optimized prompting.
    """
    
    def __init__(self, api_key: str = None, model: str = "mistral-saba-24b"):
        """
        Initialize the enhanced QA system.
        
        Args:
            api_key (str, optional): Mistral API key through Groq (defaults to env variable)
            model (str): Model name
        """
        self.storage = StorageManager()
        self.query_analyzer = QueryAnalyzer()
        self.retrieval_engine = RetrievalEngine(self.storage)
        self.context_formatter = ContextFormatter()
        self.prompt_builder = AdvancedPromptBuilder()
        self.mistral_client = MistralClient(api_key, model)
        
        # Connect to storage
        self.storage.connect()
    
    def answer_question(self, question: str, document_id: Optional[str] = None) -> str:
        """
        Answer a user question using advanced RAG techniques.
        
        Args:
            question (str): The user's question
            document_id (str, optional): ID of a specific document to query
            
        Returns:
            str: The answer
        """
        try:
            # 1. Perform deep query analysis
            analysis = self.query_analyzer.analyze(question)
            logger.info(f"Query analysis completed: {analysis}")
            
            # 2. Retrieve relevant context using advanced strategies
            if document_id:
                retrieved_data = {"specific_document": self.retrieval_engine.retrieve_document_by_id(document_id)}
            else:
                retrieved_data = self.retrieval_engine.retrieve(question, analysis)
            
            # 3. Format the context for optimal LLM consumption
            context = self.context_formatter.format(retrieved_data)
            
            # 4. Build sophisticated prompt incorporating analysis
            prompt = self.prompt_builder.build_prompt(question, context, analysis)
            
            # 5. Generate answer with Mistral
            answer = self.mistral_client.generate_answer(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
    
    def close(self):
        """Close connections."""
        self.storage.close()

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Crypto Due Diligence Q&A System")
    parser.add_argument("--api-key", help="Mistral API key via Groq (optional, defaults to env variable)")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--document-id", help="Optional document ID to query specifically")
    
    args = parser.parse_args()
    
    qa_system = EnhancedCryptoQA(args.api_key)
    
    try:
        answer = qa_system.answer_question(args.question, args.document_id)
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer: {answer}")
    finally:
        qa_system.close()