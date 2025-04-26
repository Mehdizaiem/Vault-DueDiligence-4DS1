import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

# Configure logging to stderr only
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Suppress httpx logging (Weaviate HTTP requests)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Custom JSON encoder to handle Weaviate UUIDs and other non-serializable types
class WeaviateJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Weaviate UUID objects
        if hasattr(obj, 'uuid_bytes') and callable(getattr(obj, 'uuid_bytes', None)):
            return str(obj)
        # Handle regular UUID objects
        if isinstance(obj, uuid.UUID):
            return str(obj)
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Let the base class handle anything else
        return super().default(obj)

def fetch_analytics_data():
    """
    Main function to fetch analytics data from Weaviate.
    Returns a JSON string with the data.
    """
    try:
        # Import Weaviate client
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        client = get_weaviate_client()
        logger.info("Weaviate client is connected.")
        
        if not client.is_live():
            logger.error("Failed to connect to Weaviate")
            return json.dumps({"error": "Weaviate connection failed"}, cls=WeaviateJSONEncoder)

        try:
            # Fetch data from Weaviate collections
            kpis = fetch_kpis_from_weaviate(client)
            asset_distribution = fetch_asset_distribution_from_weaviate(client)
            portfolio_performance = fetch_portfolio_performance_from_weaviate(client)
            recent_news = fetch_recent_news_from_weaviate(client)
            due_diligence_docs = fetch_due_diligence_from_weaviate(client)
            
            # Close the Weaviate client
            client.close()
            
            # Combine all data
            result = {
                "kpis": kpis,
                "asset_distribution": asset_distribution,
                "portfolio_performance": portfolio_performance,
                "recent_news": recent_news,
                "due_diligence": due_diligence_docs
            }
            
            # Use the custom JSON encoder to handle non-serializable types
            return json.dumps(result, cls=WeaviateJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error fetching data from Weaviate: {e}")
            client.close()
            return json.dumps({"error": str(e)}, cls=WeaviateJSONEncoder)
                
    except Exception as e:
        logger.error(f"Fatal error in fetch_analytics_data: {e}")
        return json.dumps({"error": str(e)}, cls=WeaviateJSONEncoder)

def fetch_kpis_from_weaviate(client) -> Dict:
    """
    Fetch KPI data from Weaviate collections using correct v4 API syntax.
    
    Args:
        client: Weaviate client
        
    Returns:
        Dict: KPI data
    """
    try:
        # For total market cap and asset count
        try:
            market_metrics = client.collections.get("MarketMetrics")
            
            # Get total market cap
            total_market_cap = 0
            try:
                # Calculate total market cap from all entries
                result = market_metrics.query.fetch_objects(
                    limit=100,
                    return_properties=["market_cap", "symbol"]
                )
                
                for obj in result.objects:
                    market_cap = obj.properties.get("market_cap", 0)
                    total_market_cap += market_cap
            except Exception as e:
                logger.error(f"Error calculating total market cap: {e}")
            
            # Format market cap as "$XM" string
            total_market_cap_str = f"${total_market_cap / 1_000_000:.1f}M"
            
            # Get asset count by counting distinct symbols
            asset_count = 0
            symbols = set()
            try:
                symbol_results = market_metrics.query.fetch_objects(
                    limit=100,
                    return_properties=["symbol"]
                )
                
                for obj in symbol_results.objects:
                    symbol = obj.properties.get("symbol")
                    if symbol:
                        symbols.add(symbol)
                
                asset_count = len(symbols)
            except Exception as e:
                logger.error(f"Error calculating asset count: {e}")
            
        except Exception as e:
            logger.error(f"Error fetching market metrics: {e}")
            total_market_cap_str = "$0M"
            asset_count = 0
        
        # For average price change
        try:
            avg_price_change = 0
            price_changes = []
            
            try:
                price_change_results = market_metrics.query.fetch_objects(
                    limit=100,
                    return_properties=["price_change_24h"]
                )
                
                for obj in price_change_results.objects:
                    price_change = obj.properties.get("price_change_24h", 0)
                    if price_change is not None:
                        price_changes.append(price_change)
                
                if price_changes:
                    avg_price_change = sum(price_changes) / len(price_changes)
            except Exception as e:
                logger.error(f"Error calculating average price change: {e}")
                
            avg_price_change_str = f"{avg_price_change:.1f}%"
            price_trend = "up" if avg_price_change >= 0 else "down"
            
        except Exception as e:
            logger.error(f"Error fetching price change data: {e}")
            avg_price_change_str = "0%"
            price_trend = "neutral"
            
        # For news sentiment
        try:
            news_sentiment = client.collections.get("CryptoNewsSentiment")
            
            sentiment_values = []
            try:
                sentiment_results = news_sentiment.query.fetch_objects(
                    limit=100,
                    return_properties=["sentiment_score"]
                )
                
                for obj in sentiment_results.objects:
                    score = obj.properties.get("sentiment_score")
                    if score is not None:
                        sentiment_values.append(score)
                        
            except Exception as e:
                logger.error(f"Error fetching sentiment scores: {e}")
            
            # Calculate average sentiment
            avg_sentiment = 0
            if sentiment_values:
                avg_sentiment = sum(sentiment_values) / len(sentiment_values)
                
            # Map sentiment to a percentage-like value
            sentiment_pct = (avg_sentiment + 1) / 2 * 100  # Assuming sentiment is between -1 and 1
            sentiment_str = f"{sentiment_pct:.1f}%"
            sentiment_trend = "up" if avg_sentiment > 0 else "down"
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            sentiment_str = "50%"
            sentiment_trend = "neutral"
        
        # Return KPI structure with all metrics
        return {
            "market_cap": {
                "value": total_market_cap_str,
                "change": "+5.3%",  # Placeholder
                "trend": "up"
            },
            "asset_count": {
                "value": str(asset_count),
                "change": "+2",  # Placeholder
                "trend": "up"
            },
            "price_change": {
                "value": avg_price_change_str,
                "change": avg_price_change_str,
                "trend": price_trend
            },
            "market_sentiment": {
                "value": sentiment_str,
                "change": "+3.2%",  # Placeholder
                "trend": sentiment_trend
            }
        }
        
    except Exception as e:
        logger.error(f"Error in fetch_kpis_from_weaviate: {e}")
        return {}

def fetch_asset_distribution_from_weaviate(client) -> List[Dict]:
    """
    Fetch asset distribution data using the correct v4 API syntax.
    
    Args:
        client: Weaviate client
        
    Returns:
        List[Dict]: Asset distribution data
    """
    try:
        # Get MarketMetrics collection
        market_metrics = client.collections.get("MarketMetrics")
        
        # Get all market metrics entries
        result = market_metrics.query.fetch_objects(
            limit=100,
            return_properties=["symbol", "market_cap"]
        )
        
        if not result.objects:
            logger.error("No asset distribution data found")
            return []
        
        # Manually calculate symbol market caps
        symbol_caps = {}
        total_market_cap = 0
        
        for obj in result.objects:
            props = obj.properties
            symbol = props.get("symbol")
            market_cap = props.get("market_cap", 0)
            
            if not symbol:
                continue
                
            if symbol not in symbol_caps:
                symbol_caps[symbol] = 0
            
            symbol_caps[symbol] += market_cap
            total_market_cap += market_cap
        
        # Convert to list and sort
        symbol_list = [(symbol, cap) for symbol, cap in symbol_caps.items()]
        symbol_list.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare distribution data
        colors = ['bg-orange-500', 'bg-indigo-500', 'bg-green-500', 'bg-purple-500', 'bg-blue-500', 'bg-red-500', 'bg-gray-500']
        result = []
        
        # Get top symbols plus "Other"
        top_count = min(6, len(symbol_list))  # Get top 6 or all if less
        
        for i, (symbol, market_cap) in enumerate(symbol_list):
            if i < top_count:
                # Clean up symbol name for display
                clean_symbol = symbol.replace("USDT", "").replace("USD", "")
                
                # Calculate percentage
                percentage = (market_cap / total_market_cap * 100) if total_market_cap > 0 else 0
                result.append({
                    "name": clean_symbol,
                    "value": round(percentage, 1),
                    "color": colors[i % len(colors)]
                })
            elif i == top_count:
                # Calculate "Other" category
                other_percentage = 100 - sum(item["value"] for item in result)
                if other_percentage > 0:
                    result.append({
                        "name": "Other Coins",
                        "value": round(other_percentage, 1),
                        "color": "bg-gray-500"
                    })
                break
        
        return result
        
    except Exception as e:
        logger.error(f"Error in fetch_asset_distribution_from_weaviate: {e}")
        return []

def fetch_portfolio_performance_from_weaviate(client, date_range="30d") -> List[Dict]:
    """
    Fetch cryptocurrency performance data using correct v4 API syntax and symbol names.
    
    Args:
        client: Weaviate client
        date_range: Time range (e.g., "30d")
        
    Returns:
        List[Dict]: Portfolio performance data
    """
    try:
        # Get CryptoTimeSeries collection
        time_series = client.collections.get("CryptoTimeSeries")
        
        # Calculate date range (choose a longer period to get more data points)
        days = int(date_range.replace('d', ''))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        from weaviate.classes.query import Filter, Sort
        
        # Get list of available symbols in the database
        available_symbols = []
        try:
            symbol_query = time_series.query.fetch_objects(
                limit=100,
                return_properties=["symbol"]
            )
            
            symbol_count = {}
            for obj in symbol_query.objects:
                symbol = obj.properties.get("symbol")
                if symbol:
                    symbol_count[symbol] = symbol_count.get(symbol, 0) + 1
            
            # Sort symbols by frequency (most data points first)
            sorted_symbols = sorted(symbol_count.items(), key=lambda x: x[1], reverse=True)
            available_symbols = [symbol for symbol, count in sorted_symbols[:5]]  # Take top 5 symbols
            
            logger.info(f"Available symbols in CryptoTimeSeries: {available_symbols}")
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
        
        performance_data = []
        
        # Query for each symbol
        for symbol in available_symbols:
            try:
                # Get time series data for this symbol
                response = time_series.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal(symbol),
                    sort=Sort.by_property("timestamp", ascending=True),
                    return_properties=["symbol", "timestamp", "close"],
                    limit=100
                )
                
                if not response.objects:
                    logger.warning(f"No data found for symbol {symbol}")
                    continue
                
                # Sample entries to reduce density (based on how many entries were returned)
                objects = response.objects
                sample_rate = max(1, len(objects) // 30)  # Aim for about 30 data points
                
                # Need to handle the case where timestamps aren't consistent
                data_by_timestamp = {}
                
                for obj in objects:
                    props = obj.properties
                    ts = props.get("timestamp")
                    close = props.get("close", 0)
                    
                    if ts and close is not None:
                        data_by_timestamp[ts] = (props.get("symbol", ""), close)
                
                # Sort by timestamp
                sorted_data = sorted(data_by_timestamp.items())
                
                # Get the first close price to calculate percentage changes
                if not sorted_data:
                    continue
                    
                base_symbol, base_price = sorted_data[0][1]
                if base_price == 0:
                    continue  # Skip if first price is zero to avoid division by zero
                
                # Sample at regular intervals
                for i in range(0, len(sorted_data), sample_rate):
                    ts, (symbol_name, close_price) = sorted_data[i]
                    
                    try:
                        # Calculate change percentage from the base price
                        change_pct = ((close_price - base_price) / base_price * 100)
                        
                        # Clean up symbol name for display
                        display_symbol = symbol_name.replace("USDUSD", "")
                        for base in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LINK", "TRX", "BCH"]:
                            if base in display_symbol:
                                display_symbol = base + "USDT"
                                break
                        
                        # Format timestamp into a standard format
                        try:
                            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            ts = dt.isoformat()
                        except:
                            pass  # Keep original format if parsing fails
                        
                        # Add to performance data
                        performance_data.append({
                            "symbol": display_symbol,
                            "timestamp": ts,
                            "change_pct": round(change_pct, 2)
                        })
                    except Exception as e:
                        logger.error(f"Error processing data point: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
        
        if performance_data:
            logger.info(f"Successfully retrieved {len(performance_data)} performance data points")
            return performance_data
        else:
            logger.error("No performance data found")
            return []
        
    except Exception as e:
        logger.error(f"Error in fetch_portfolio_performance_from_weaviate: {e}")
        return []

def fetch_recent_news_from_weaviate(client, limit=5) -> List[Dict]:
    """
    Fetch recent news with sentiment analysis from CryptoNewsSentiment collection.
    
    Args:
        client: Weaviate client
        limit: Number of news items to fetch
        
    Returns:
        List[Dict]: Recent news with sentiment
    """
    try:
        news = client.collections.get("CryptoNewsSentiment")
        
        from weaviate.classes.query import Sort
        
        # Get most recent news articles sorted by date
        response = news.query.fetch_objects(
            sort=Sort.by_property("date", ascending=False),
            return_properties=["title", "source", "date", "sentiment_score", "sentiment_label", "related_assets", "url"],
            limit=limit
        )
        
        news_items = []
        
        for obj in response.objects:
            props = obj.properties
            
            # Map sentiment score (-1 to 1) to a color class
            sentiment_score = props.get("sentiment_score", 0)
            if sentiment_score > 0.3:
                sentiment_color = "text-green-500"
            elif sentiment_score < -0.3:
                sentiment_color = "text-red-500"
            else:
                sentiment_color = "text-amber-500"
                
            news_items.append({
                "title": props.get("title", "Untitled"),
                "source": props.get("source", "Unknown Source"),
                "date": props.get("date", ""),
                "sentiment_score": round(sentiment_score, 2),
                "sentiment_label": props.get("sentiment_label", "Neutral"),
                "sentiment_color": sentiment_color,
                "related_assets": props.get("related_assets", []),
                "url": props.get("url", "#")
            })
        
        return news_items
        
    except Exception as e:
        logger.error(f"Error in fetch_recent_news_from_weaviate: {e}")
        return []

def fetch_due_diligence_from_weaviate(client, limit=4) -> List[Dict]:
    """
    Fetch due diligence documents from CryptoDueDiligenceDocuments collection.
    
    Args:
        client: Weaviate client
        limit: Number of documents to fetch
        
    Returns:
        List[Dict]: Due diligence document summaries
    """
    try:
        docs = client.collections.get("CryptoDueDiligenceDocuments")
        
        # Get due diligence documents
        response = docs.query.fetch_objects(
            return_properties=["title", "document_type", "source", "keywords"],
            limit=limit
        )
        
        documents = []
        
        for obj in response.objects:
            props = obj.properties
            
            # Get icon based on document type
            doc_type = props.get("document_type", "").lower()
            if "whitepaper" in doc_type:
                icon = "file-text"
            elif "report" in doc_type:
                icon = "clipboard-list"
            elif "analysis" in doc_type:
                icon = "chart-bar"
            elif "audit" in doc_type:
                icon = "shield-check"
            else:
                icon = "document"
                
            # Convert UUID to string to ensure it's JSON serializable
            doc_id = str(obj.uuid) if obj.uuid else ""
                
            documents.append({
                "title": props.get("title", "Untitled Document"),
                "document_type": props.get("document_type", "Document"),
                "source": props.get("source", "Unknown Source"),
                "keywords": props.get("keywords", []),
                "icon": icon,
                "id": doc_id
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"Error in fetch_due_diligence_from_weaviate: {e}")
        return []

if __name__ == "__main__":
    # When run directly, output the JSON to stdout
    result = fetch_analytics_data()
    print(result)