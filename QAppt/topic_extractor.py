from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
import spacy
import re
import logging
import traceback

logger = logging.getLogger(__name__)

class TopicExtractor:
    def __init__(self):
        """Initialize topic extractor"""
        self.topic_map = {
            "Bitcoin": ["bitcoin", "btc", "xbt"],
            "Ethereum": ["ethereum", "eth"],
            "BNB": ["bnb", "binance coin"],
            "Market Analysis": ["market", "price", "trading", "volume"],
            "Technical Analysis": ["technical", "chart", "pattern", "indicator"],
            "DeFi": ["defi", "yield", "liquidity", "swap"],
            "NFTs": ["nft", "non-fungible", "collectible"],
            "Regulation": ["regulation", "compliance", "legal"],
            "Technology": ["blockchain", "protocol", "smart contracts"]
        }

    def extract_topics(self, conversation_history: List[Dict]) -> Dict[str, List[str]]:
        """Extract topics from conversation history"""
        topics = {
            'main_topics': [],
            'technical_topics': [],
            'market_topics': [],
            'regulatory_topics': [],
            'entities': []
        }

        # Process each conversation entry
        for entry in conversation_history:
            # Combine question and answer text
            text = f"{entry['question']} {entry['answer']}".lower()
            
            # Extract topics based on keywords
            for topic, keywords in self.topic_map.items():
                if any(keyword in text for keyword in keywords):
                    category = self._categorize_topic(topic)
                    if category in topics:
                        topics[category].append(topic)
                        # Add cryptocurrencies to entities as well
                        if topic in ["Bitcoin", "Ethereum", "BNB"]:
                            topics['entities'].append(topic)

        # Remove duplicates while preserving order
        for category in topics:
            topics[category] = list(dict.fromkeys(topics[category]))

        return topics

    def _categorize_topic(self, topic: str) -> str:
        """Categorize topic into appropriate category"""
        if topic in ["Bitcoin", "Ethereum", "BNB"]:
            return "main_topics"
        elif topic in ["Technical Analysis", "Technology", "DeFi", "NFTs"]:
            return "technical_topics"
        elif topic in ["Market Analysis", "Trading", "Price"]:
            return "market_topics"
        elif topic in ["Regulation", "Compliance"]:
            return "regulatory_topics"
        else:
            return "main_topics"
