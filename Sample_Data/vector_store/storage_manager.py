# File path: Sample_Data/vector_store/storage_manager.py
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from datetime import timedelta
from weaviate.classes.query import Sort
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Weaviate client and embedding functions
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
from Sample_Data.vector_store.embed import generate_mpnet_embedding, generate_finbert_embedding
from Sample_Data.vector_store.schema_manager import (
    create_crypto_due_diligence_schema,
    create_crypto_news_sentiment_schema,
    create_market_metrics_schema,
    create_crypto_time_series_schema,
    create_onchain_analytics_schema
)

class StorageManager:
    """
    Manages storage of crypto due diligence data in Weaviate collections.
    Handles storage strategies for different data types including embedding generation.
    """
    
    def __init__(self):
        """Initialize the storage manager"""
        self.client = None
    
    def connect(self):
        """Connect to Weaviate"""
        if self.client is None:
            self.client = get_weaviate_client()
            
        return self.client.is_live()
    
    def close(self):
        """Close the Weaviate connection"""
        if self.client:
            self.client.close()
            self.client = None
    
    def setup_schemas(self):
        """Set up all required schemas"""
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return False
            
        try:
            # Create all schemas
            create_crypto_due_diligence_schema(self.client)
            create_crypto_news_sentiment_schema(self.client)
            create_market_metrics_schema(self.client)
            create_crypto_time_series_schema(self.client)
            create_onchain_analytics_schema(self.client)
            
            logger.info("All schemas set up successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up schemas: {e}")
            return False
    
    def store_due_diligence_document(self, document: Dict) -> bool:
        """
        Store a due diligence document in the CryptoDueDiligenceDocuments collection
        
        Args:
            document (Dict): Document data including content, title, etc.
            
        Returns:
            bool: Success status
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return False
            
        try:
            # Get the collection
            collection = self.client.collections.get("CryptoDueDiligenceDocuments")
            
            # Extract document content
            content = document.get("content", "")
            document_type = document.get("document_type", "unknown")
            title = document.get("title", "Untitled")
            
            # Generate embedding with all-MPNet
            vector = generate_mpnet_embedding(content)
            
            # Prepare properties
            properties = {
                "content": content,
                "document_type": document_type,
                "title": title,
                "source": document.get("source", "unknown"),
            }
            
            # Add optional properties if present
            for field in ["date", "author_issuer", "category", "risk_score", "keywords"]:
                if field in document and document[field] is not None:
                    properties[field] = document[field]
            
            # Format date if it exists but isn't already in the right format
            if "date" in properties and not isinstance(properties["date"], str):
                try:
                    properties["date"] = properties["date"].isoformat()
                except (AttributeError, TypeError):
                    # Handle other date formats or issues
                    del properties["date"]
            
            # Store the document
            collection.data.insert(properties=properties, vector=vector)
            
            logger.info(f"Successfully stored document: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing due diligence document: {e}")
            return False
    
    def store_news_article(self, article: Dict) -> bool:
        """
        Store a news article in the CryptoNewsSentiment collection
        
        Args:
            article (Dict): News article data including content, title, sentiment, etc.
            
        Returns:
            bool: Success status
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return False
            
        try:
            # Get the collection
            collection = self.client.collections.get("CryptoNewsSentiment")
            
            # Extract article content
            content = article.get("content", "")
            title = article.get("title", "")
            
            # Generate embedding with FinBERT
            combined_text = f"{title}\n\n{content}" if title else content
            vector = generate_finbert_embedding(combined_text)
            
            # Prepare properties
            properties = {
                "content": content,
                "title": title,
                "source": article.get("source", "unknown"),
                "url": article.get("url", ""),
                "sentiment_label": article.get("sentiment_label", "NEUTRAL"),
                "sentiment_score": article.get("sentiment_score", 0.5),
                "analyzed_at": article.get("analyzed_at", datetime.now().isoformat())
            }
            
            # Add optional properties if present
            for field in ["date", "authors", "image_url", "related_assets"]:
                if field in article and article[field] is not None:
                    properties[field] = article[field]
            
            # Format date if needed
            if "date" in properties and not isinstance(properties["date"], str):
                try:
                    properties["date"] = properties["date"].isoformat()
                except (AttributeError, TypeError):
                    # Use current time as fallback
                    properties["date"] = datetime.now().isoformat()
            
            # Store the article
            collection.data.insert(properties=properties, vector=vector)
            
            logger.info(f"Successfully stored news article: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing news article: {e}")
            return False
    
    def store_market_data(self, market_data: Union[Dict, List[Dict]]) -> bool:
        """
        Store market data in the MarketMetrics collection
        
        Args:
            market_data (Dict or List[Dict]): Market data point(s)
            
        Returns:
            bool: Success status
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return False
            
        try:
            # Get the collection
            collection = self.client.collections.get("MarketMetrics")
            
            # Handle both single item and list
            if isinstance(market_data, dict):
                market_data = [market_data]
            
            # Process each data point
            success_count = 0
            for data in market_data:
                try:
                    # Prepare properties
                    properties = {
                        "symbol": data.get("symbol", "UNKNOWN"),
                        "source": data.get("source", "unknown"),
                        "price": data.get("price", 0.0),
                        "market_cap": data.get("market_cap", 0.0),
                        "volume_24h": data.get("volume_24h", 0.0),
                        "price_change_24h": data.get("price_change_24h", 0.0),
                        "timestamp": data.get("timestamp", datetime.now().isoformat())
                    }
                    
                    # Store the data point
                    collection.data.insert(properties=properties)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing market data point: {e}")
                    continue
            
            logger.info(f"Successfully stored {success_count}/{len(market_data)} market data points")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return False
    
    def store_time_series(self, time_series_data: List[Dict]) -> bool:
        """
        Store time series data in the CryptoTimeSeries collection with improved error handling
        
        Args:
            time_series_data (List[Dict]): Time series data points
            
        Returns:
            bool: Success status
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return False
            
        try:
            # Get the collection
            collection = self.client.collections.get("CryptoTimeSeries")
            
            # Process data points in batches
            batch_size = 100
            success_count = 0
            error_count = 0
            
            # Track data types for logging
            data_types = {}
            
            for i in range(0, len(time_series_data), batch_size):
                batch = time_series_data[i:i+batch_size]
                objects_to_insert = []
                
                for data in batch:
                    # Ensure all numeric fields are properly converted to float/int
                    try:
                        # Prepare properties with correct data types
                        properties = {
                            "symbol": str(data.get("symbol", "UNKNOWN")),
                            "exchange": str(data.get("exchange", "unknown")),
                            "timestamp": data.get("timestamp", datetime.now().isoformat()),
                            "open": float(data.get("open", 0.0)),
                            "high": float(data.get("high", 0.0)),
                            "low": float(data.get("low", 0.0)),
                            "close": float(data.get("close", 0.0)),
                            "volume": float(data.get("volume", 0.0)),
                            "interval": str(data.get("interval", "1d"))
                        }
                        
                        # For debugging, track data types
                        if len(data_types) == 0:
                            for key, value in properties.items():
                                data_types[key] = type(value).__name__
                        
                        objects_to_insert.append(properties)
                    except Exception as e:
                        logger.error(f"Error preparing data point: {e}")
                        error_count += 1
                        continue
                
                # Insert batch
                if objects_to_insert:
                    try:
                        response = collection.data.insert_many(objects_to_insert)
                        
                        # Check for errors
                        if hasattr(response, 'has_errors') and response.has_errors:
                            logger.error(f"Batch insert had errors: {response.errors}")
                            error_count += len(objects_to_insert)
                        else:
                            success_count += len(objects_to_insert)
                            logger.info(f"Inserted batch of {len(objects_to_insert)} time series points")
                    except Exception as e:
                        logger.error(f"Error inserting batch: {e}")
                        
                        # Try one by one in case batch insertion fails
                        partial_success = 0
                        for obj in objects_to_insert:
                            try:
                                collection.data.insert(properties=obj)
                                partial_success += 1
                            except Exception:
                                error_count += 1
                        
                        if partial_success > 0:
                            success_count += partial_success
                            logger.info(f"Inserted {partial_success}/{len(objects_to_insert)} time series points individually")
                        
            # Log data types for debugging
            logger.debug(f"Data types for time series data: {data_types}")
            
            logger.info(f"Successfully stored {success_count}/{len(time_series_data)} time series data points")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error storing time series data: {e}")
            return False
    
    def store_onchain_analytics(self, analytics_data: Dict) -> bool:
        """
        Store on-chain analytics data in the OnChainAnalytics collection
        
        Args:
            analytics_data (Dict): On-chain analytics data
            
        Returns:
            bool: Success status
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return False
            
        try:
            # Get the collection
            collection = self.client.collections.get("OnChainAnalytics")
            
            # Prepare properties
            properties = {
                "address": analytics_data.get("address", ""),
                "blockchain": analytics_data.get("blockchain", "ethereum"),
                "entity_type": analytics_data.get("entity_type", "wallet"),
                "transaction_count": analytics_data.get("transaction_count", 0),
                "token_transaction_count": analytics_data.get("token_transaction_count", 0),
                "total_received": analytics_data.get("total_received", 0.0),
                "total_sent": analytics_data.get("total_sent", 0.0),
                "balance": analytics_data.get("balance", 0.0),
                "analysis_timestamp": analytics_data.get("analysis_timestamp", datetime.now().isoformat())
            }
            
            # Add optional fields if present
            optional_fields = [
                "first_activity", "last_activity", "active_days", "unique_interactions",
                "contract_interactions", "risk_score", "risk_level", "risk_factors",
                "related_fund"
            ]
            
            for field in optional_fields:
                if field in analytics_data and analytics_data[field] is not None:
                    properties[field] = analytics_data[field]
            
            # Add tokens array if present
            if "tokens" in analytics_data and isinstance(analytics_data["tokens"], list):
                properties["tokens"] = analytics_data["tokens"]
            
            # Store the analytics data
            collection.data.insert(properties=properties)
            
            logger.info(f"Successfully stored on-chain analytics for {properties['address']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing on-chain analytics: {e}")
            return False
    
    def retrieve_documents(self, query: str, collection_name: str = "CryptoDueDiligenceDocuments", limit: int = 5) -> List[Dict]:
        """
        Retrieve documents from a collection based on a query
        
        Args:
            query (str): Search query
            collection_name (str): Collection to search
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Retrieved documents
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return []
            
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Generate query vector based on collection
            if collection_name == "CryptoNewsSentiment":
                query_vector = generate_finbert_embedding(query)
            else:
                query_vector = generate_mpnet_embedding(query)
            
            # Perform hybrid search (vector + BM25)
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=0.5,  # Balance between vector and keyword search
                limit=limit
            )
            
            # Format results
            results = []
            for obj in response.objects:
                result = {
                    "id": str(obj.uuid),
                    **obj.properties
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def retrieve_market_data(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve market data for a specific symbol
        
        Args:
            symbol (str): Cryptocurrency symbol
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Market data
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return []
            
        try:
            # Get the collection
            collection = self.client.collections.get("MarketMetrics")
            
            # Build filter
            from weaviate.classes.query import Filter
            response = collection.query.fetch_objects(
                filters=Filter.by_property("symbol").equal(symbol),
                limit=limit,
                sort=[{"path": ["timestamp"], "order": "desc"}]
            )
            
            # Format results
            results = []
            for obj in response.objects:
                result = {
                    "id": str(obj.uuid),
                    **obj.properties
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return []
    
    def retrieve_time_series(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict]:
        """
        Retrieve time series data for a specific symbol and interval with improved debugging.
        
        Args:
            symbol (str): Cryptocurrency symbol
            interval (str): Time interval
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Time series data
        """
        if not self.connect():
            logger.error(f"Failed to connect to Weaviate when retrieving data for {symbol}")
            return []
            
        try:
            # Get the collection
            collection = self.client.collections.get("CryptoTimeSeries")
            
            # Log the request
            logger.debug(f"Retrieving time series data for symbol={symbol}, interval={interval}, limit={limit}")
            
            # Normalize the symbol name for consistency
            if not symbol.upper().endswith("USDT") and not symbol.upper().endswith("USD"):
                symbol = f"{symbol.upper()}USDT"
            symbol = symbol.upper()
            
            # Also try with alternative symbol notation
            alt_symbol = symbol.replace("USDT", "USD") if "USDT" in symbol else symbol.replace("USD", "USDT")
            
            try:
                # First try to count how many records exist for this symbol
                from weaviate.classes.query import Filter
                
                # Create filter for symbol
                main_filter = Filter.by_property("symbol").equal(symbol)
                alt_filter = Filter.by_property("symbol").equal(alt_symbol)
                combined_filter = main_filter | alt_filter
                
                # Add interval filter if specified
                if interval:
                    interval_filter = Filter.by_property("interval").equal(interval)
                    combined_filter = combined_filter & interval_filter
                
                # Count matches
                count_result = collection.query.aggregate.with_where(
                    filter=combined_filter
                ).total_count
                
                logger.info(f"Found {count_result} time series data points for {symbol}/{alt_symbol}")
                
                # If no matches found, log more details
                if count_result == 0:
                    # Get all symbol entries to see what's available
                    all_symbols_result = collection.query.aggregate.over_all(
                        group_by="symbol"
                    )
                    
                    if all_symbols_result and hasattr(all_symbols_result, 'groups'):
                        available_symbols = [group.grouped_by.value for group in all_symbols_result.groups]
                        logger.info(f"Available symbols in CryptoTimeSeries: {available_symbols}")
                        
                        # Check if symbol might be formatted differently
                        symbol_base = symbol.replace("USDT", "").replace("USD", "")
                        similar_symbols = [s for s in available_symbols if symbol_base in s]
                        
                        if similar_symbols:
                            logger.info(f"Found similar symbols to {symbol}: {similar_symbols}")
                            
                            # Try with the first similar symbol
                            symbol = similar_symbols[0]
                            logger.info(f"Trying again with symbol: {symbol}")
                            main_filter = Filter.by_property("symbol").equal(symbol)
                            
                            if interval:
                                combined_filter = main_filter & Filter.by_property("interval").equal(interval)
                            else:
                                combined_filter = main_filter
                    else:
                        logger.warning("No symbols found in CryptoTimeSeries collection")
                        return []
                
                # Execute query with proper sort format
                from weaviate.classes.query import Sort
                
                response = collection.query.fetch_objects(
                    filters=combined_filter,
                    limit=limit,
                    sort=[Sort.by_property("timestamp", ascending=True)]
                )
                
                # Format results
                results = []
                for obj in response.objects:
                    result = {
                        "id": str(obj.uuid),
                        **obj.properties
                    }
                    results.append(result)
                
                logger.info(f"Retrieved {len(results)} time series data points for {symbol}")
                return results
                
            except Exception as e:
                logger.error(f"Error building query for time series: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving time series: {e}")
            return []
    
    def retrieve_onchain_analytics(self, address: str) -> Optional[Dict]:
        """
        Retrieve on-chain analytics for a specific address
        
        Args:
            address (str): Blockchain address
            
        Returns:
            Dict or None: On-chain analytics
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return None
            
        try:
            # Get the collection
            collection = self.client.collections.get("OnChainAnalytics")
            
            # Build filter
            from weaviate.classes.query import Filter
            response = collection.query.fetch_objects(
                filters=Filter.by_property("address").equal(address),
                limit=1,
                sort=[{"path": ["analysis_timestamp"], "order": "desc"}]
            )
            
            # Format result
            if response.objects:
                obj = response.objects[0]
                result = {
                    "id": str(obj.uuid),
                    **obj.properties
                }
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving on-chain analytics: {e}")
            return None
    
    def get_sentiment_stats(self, asset: Optional[str] = None, days: int = 7) -> Dict:
        """
        Get sentiment statistics for crypto news
        
        Args:
            asset (str, optional): Specific crypto asset (e.g., "bitcoin")
            days (int): Number of days to include
            
        Returns:
            Dict: Sentiment statistics
        """
        if not self.connect():
            logger.error("Failed to connect to Weaviate")
            return {"error": "Failed to connect to Weaviate"}
            
        try:
            # Get the collection
            collection = self.client.collections.get("CryptoNewsSentiment")
            
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Build filters
            from weaviate.classes.query import Filter
            
            date_filter = Filter.by_property("date").greater_than(from_date)
            
            if asset:
                # Filter by related assets
                asset_filter = Filter.by_property("related_assets").contains_any([asset.lower()])
                combined_filter = date_filter & asset_filter
            else:
                combined_filter = date_filter
            
            # Get articles matching filters
            response = collection.query.fetch_objects(
                filters=combined_filter,
                limit=1000  # Get enough articles for good statistics
            )
            
            # Extract sentiment data
            articles = []
            for obj in response.objects:
                articles.append({
                    "sentiment_label": obj.properties.get("sentiment_label", "NEUTRAL"),
                    "sentiment_score": obj.properties.get("sentiment_score", 0.5),
                    "date": obj.properties.get("date")
                })
            
            # Calculate statistics
            total_articles = len(articles)
            
            if total_articles == 0:
                return {
                    "total_articles": 0,
                    "avg_sentiment": 0.5,
                    "sentiment_distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
                }
            
            # Calculate average sentiment score
            avg_sentiment = sum(a["sentiment_score"] for a in articles) / total_articles
            
            # Count sentiment labels
            sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
            for article in articles:
                label = article["sentiment_label"]
                sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            
            # Calculate trend if possible
            if total_articles >= 10:
                # Sort by date
                sorted_articles = sorted(articles, key=lambda x: x.get("date", ""))
                
                # Split into first and second half
                half_idx = len(sorted_articles) // 2
                first_half = sorted_articles[:half_idx]
                second_half = sorted_articles[half_idx:]
                
                # Calculate averages
                first_half_avg = sum(a["sentiment_score"] for a in first_half) / len(first_half)
                second_half_avg = sum(a["sentiment_score"] for a in second_half) / len(second_half)
                
                # Determine trend
                if second_half_avg > first_half_avg + 0.1:
                    trend = "improving"
                elif second_half_avg < first_half_avg - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient data"
            
            # Return statistics
            return {
                "total_articles": total_articles,
                "avg_sentiment": avg_sentiment,
                "sentiment_distribution": sentiment_counts,
                "trend": trend,
                "period": f"last {days} days"
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment stats: {e}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    manager = StorageManager()
    
    try:
        # Setup schemas
        manager.setup_schemas()
        
        # Test storing a document
        document = {
            "content": "This is a test document about crypto investments.",
            "title": "Test Document",
            "document_type": "whitepaper",
            "source": "test.txt"
        }
        
        success = manager.store_due_diligence_document(document)
        print(f"Store document success: {success}")
        
        # Test document retrieval
        results = manager.retrieve_documents("crypto investments")
        print(f"Retrieved {len(results)} documents")
        
    finally:
        manager.close()