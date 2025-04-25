"""
Data Retriever Module

This module handles retrieval of data from all collections for the crypto fund due diligence analysis.
It provides a unified interface to access market data, on-chain analytics, regulatory documents,
and other information sources.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataRetriever:
    """
    Retrieves data from all available collections for comprehensive crypto fund analysis.
    Serves as a unified interface for accessing various data sources.
    """
    
    def __init__(self, storage_manager):
        """
        Initialize the data retriever with a storage manager.
        
        Args:
            storage_manager: The storage manager that connects to Weaviate
        """
        self.storage = storage_manager
        self.ensure_connection()
    
    def ensure_connection(self):
        """Ensure the storage manager is connected"""
        if not hasattr(self.storage, 'client') or self.storage.client is None:
            self.storage.connect()
    
    def get_market_data(self, symbol: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Get market data for a specific symbol.
        
        Args:
            symbol: Cryptocurrency trading symbol (e.g., BTCUSDT)
            limit: Maximum number of data points to retrieve
            
        Returns:
            List of market data points
        """
        try:
            self.ensure_connection()
            
            # Try to get data from MarketMetrics collection
            market_data = self.storage.retrieve_market_data(symbol, limit=limit)
            
            if market_data:
                logger.info(f"Retrieved {len(market_data)} market data points for {symbol}")
                return market_data
            else:
                logger.warning(f"No market data found for {symbol}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
            return []
    
    def get_historical_data(self, symbol: str, interval: str = "1d", limit: int = 90) -> List[Dict[str, Any]]:
        """
        Get historical time series data for a specific symbol.
        
        Args:
            symbol: Cryptocurrency trading symbol (e.g., BTCUSDT)
            interval: Time interval (e.g., 1h, 1d, 1w)
            limit: Maximum number of data points to retrieve
            
        Returns:
            List of historical data points
        """
        try:
            self.ensure_connection()
            
            # Try to get data from CryptoTimeSeries collection
            historical_data = self.storage.retrieve_time_series(symbol, interval=interval, limit=limit)
            
            if historical_data:
                logger.info(f"Retrieved {len(historical_data)} historical data points for {symbol} ({interval})")
                return historical_data
            else:
                logger.warning(f"No historical data found for {symbol} ({interval})")
                return []
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return []
    
    def get_onchain_analytics(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get on-chain analytics for a specific Ethereum address.
        
        Args:
            address: Ethereum wallet address
            
        Returns:
            On-chain analytics data or None if not found
        """
        try:
            self.ensure_connection()
            
            # Try to get data from OnChainAnalytics collection
            onchain_data = self.storage.retrieve_onchain_analytics(address)
            
            if onchain_data:
                logger.info(f"Retrieved on-chain analytics for {address}")
                return onchain_data
            else:
                logger.warning(f"No on-chain analytics found for {address}")
                
                # If not in collection, try to get data directly from blockchain
                # This could be implemented by calling the onchain_manager in your due diligence system
                # For now, we'll just return None if not in the collection
                return None
        except Exception as e:
            logger.error(f"Error retrieving on-chain analytics for {address}: {e}")
            return None
    
    def get_sentiment_analysis(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment analysis for a specific crypto asset.
        
        Args:
            asset: Cryptocurrency name or symbol
            
        Returns:
            Sentiment analysis data or None if not found
        """
        try:
            self.ensure_connection()
            
            # Try to get data from CryptoNewsSentiment collection
            sentiment_data = self.storage.get_sentiment_stats(asset, days=30)
            
            if sentiment_data and not isinstance(sentiment_data, str) and "error" not in sentiment_data:
                logger.info(f"Retrieved sentiment analysis for {asset}")
                return sentiment_data
            else:
                logger.warning(f"No sentiment analysis found for {asset}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving sentiment analysis for {asset}: {e}")
            return None
    
    def get_forecasts(self, symbol: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Get price forecasts for a specific symbol.
        
        Args:
            symbol: Cryptocurrency trading symbol (e.g., BTCUSDT)
            limit: Maximum number of forecasts to retrieve
            
        Returns:
            List of forecast data or empty list if not found
        """
        try:
            self.ensure_connection()
            
            # Try to get data from Forecast collection
            forecasts = self.storage.retrieve_latest_forecast(symbol, limit=limit)
            
            if forecasts:
                logger.info(f"Retrieved {len(forecasts)} forecasts for {symbol}")
                return forecasts
            else:
                logger.warning(f"No forecasts found for {symbol}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving forecasts for {symbol}: {e}")
            return []
    
    def get_regulatory_documents(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get regulatory documents related to a specific query.
        
        Args:
            query: Search query for regulatory documents
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of regulatory documents
        """
        try:
            self.ensure_connection()
            
            # Try to get data from CryptoDueDiligenceDocuments collection
            documents = self.storage.retrieve_documents(
                query=query,
                collection_name="CryptoDueDiligenceDocuments",
                limit=limit
            )
            
            if documents:
                logger.info(f"Retrieved {len(documents)} regulatory documents for '{query}'")
                return documents
            else:
                logger.warning(f"No regulatory documents found for '{query}'")
                return []
        except Exception as e:
            logger.error(f"Error retrieving regulatory documents for '{query}': {e}")
            return []
    
    def get_news_articles(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get news articles related to a specific query.
        
        Args:
            query: Search query for news articles
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of news articles
        """
        try:
            self.ensure_connection()
            
            # Try to get data from CryptoNewsSentiment collection
            articles = self.storage.retrieve_documents(
                query=query,
                collection_name="CryptoNewsSentiment",
                limit=limit
            )
            
            if articles:
                logger.info(f"Retrieved {len(articles)} news articles for '{query}'")
                return articles
            else:
                logger.warning(f"No news articles found for '{query}'")
                return []
        except Exception as e:
            logger.error(f"Error retrieving news articles for '{query}': {e}")
            return []
    
    def get_user_documents(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get user-uploaded documents related to a specific query.
        
        Args:
            query: Search query for user documents
            user_id: User ID to filter documents
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of user documents
        """
        try:
            self.ensure_connection()
            
            # Try to get data from UserDocuments collection
            documents = self.storage.retrieve_documents(
                query=query,
                collection_name="UserDocuments",
                limit=limit,
                user_id=user_id
            )
            
            if documents:
                logger.info(f"Retrieved {len(documents)} user documents for '{query}'")
                return documents
            else:
                logger.warning(f"No user documents found for '{query}'")
                return []
        except Exception as e:
            logger.error(f"Error retrieving user documents for '{query}': {e}")
            return []
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            self.ensure_connection()
            
            # Use the DocumentAnalyzer to get the document
            from UserQ_A.DocumentAnalyzer import DocumentAnalyzer
            
            analyzer = DocumentAnalyzer()
            document = analyzer.get_document_by_id(document_id)
            
            if document:
                logger.info(f"Retrieved document with ID {document_id}")
                return document
            else:
                logger.warning(f"No document found with ID {document_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving document with ID {document_id}: {e}")
            return None
    
    def search_all_collections(self, query: str, limit_per_collection: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all collections for a specific query.
        
        Args:
            query: Search query
            limit_per_collection: Maximum number of results per collection
            
        Returns:
            Dict mapping collection names to their search results
        """
        try:
            self.ensure_connection()
            
            # Define collections to search
            collections = [
                "CryptoDueDiligenceDocuments",
                "CryptoNewsSentiment",
                "UserDocuments"
            ]
            
            # Search each collection
            results = {}
            
            for collection in collections:
                try:
                    collection_results = self.storage.retrieve_documents(
                        query=query,
                        collection_name=collection,
                        limit=limit_per_collection
                    )
                    
                    if collection_results:
                        results[collection] = collection_results
                except Exception as collection_error:
                    logger.error(f"Error searching collection {collection}: {collection_error}")
            
            if results:
                total_results = sum(len(results[col]) for col in results)
                logger.info(f"Found {total_results} results across {len(results)} collections for '{query}'")
                return results
            else:
                logger.warning(f"No results found for '{query}' in any collection")
                return {}
        except Exception as e:
            logger.error(f"Error searching across collections for '{query}': {e}")
            return {}
    
    def extract_crypto_entities(self, content: str) -> List[str]:
        """
        Extract cryptocurrency entities from text content.
        This is a simple extraction method - in a real system you might use NER.
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of extracted cryptocurrency entities
        """
        crypto_keywords = {
            "bitcoin": ["bitcoin", "btc", "xbt"],
            "ethereum": ["ethereum", "eth", "ether"],
            "solana": ["solana", "sol"],
            "binance": ["binance", "bnb", "binance coin"],
            "cardano": ["cardano", "ada"],
            "ripple": ["ripple", "xrp"],
            "polkadot": ["polkadot", "dot"],
            "dogecoin": ["dogecoin", "doge"],
            "avalanche": ["avalanche", "avax"],
            "polygon": ["polygon", "matic"]
        }
        
        found_entities = []
        content_lower = content.lower()
        
        for entity, keywords in crypto_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                found_entities.append(entity)
        
        return found_entities
    
    def extract_ethereum_addresses(self, content: str) -> List[str]:
        """
        Extract Ethereum wallet addresses from text content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of extracted Ethereum addresses
        """
        import re
        
        # Pattern for Ethereum addresses (simple version)
        eth_address_pattern = r'0x[a-fA-F0-9]{40}\b'
        
        # Find all matches
        addresses = re.findall(eth_address_pattern, content)
        
        return addresses