
import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from weaviate.exceptions import WeaviateBaseError

# --- Path Setup ---
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# --- Logging Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(current_path, '..', '..')))
from Code.data_processing.sentiment_analyzer import CryptoSentimentAnalyzer



analyzer = CryptoSentimentAnalyzer()

def store_sentiment_data(df: pd.DataFrame) -> bool:
    """Analyze and store sentiment data in Weaviate."""
    try:
        analyzed_df = analyzer.analyze_dataframe(df)
        success = analyzer.store_in_weaviate(analyzed_df)
        return success
    except Exception as e:
        logger.error(f"Failed to analyze/store sentiment data: {e}")
        return False

def get_sentiment_stats(symbol: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
    """Return sentiment stats for the last `days` for the given `symbol`."""
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    from weaviate.classes.query import Filter

    client = get_weaviate_client()
    try:
        collection = client.collections.get("CryptoNewsSentiment")
    except WeaviateBaseError:
        logger.warning("Collection not found.")
        return {
            "total_articles": 0,
            "sentiment_distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
            "avg_sentiment": 0.5,
            "error": "Collection not found"
        }

    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        filters = Filter.by_property("date").greater_than(start_date)
        if symbol:
            filters = filters & (
                Filter.by_property("content").contains_all([symbol.lower()]) |
                Filter.by_property("title").contains_all([symbol.lower()])
            )

        result = collection.query.fetch_objects(
            filters=filters,
            return_properties=["sentiment_label", "sentiment_score"],
            limit=100
        )

        sentiment_distribution = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        scores = []

        for obj in result.objects:
            label = obj.properties.get("sentiment_label", "NEUTRAL")
            score = obj.properties.get("sentiment_score", 0.5)
            sentiment_distribution[label] += 1
            scores.append(score)

        avg = sum(scores) / len(scores) if scores else 0.5
        return {
            "total_articles": len(scores),
            "sentiment_distribution": sentiment_distribution,
            "avg_sentiment": round(avg, 3),
            "timeframe": f"Last {days} days"
        }
    except Exception as e:
        logger.error(f"Error fetching sentiment stats: {e}")
        return {
            "total_articles": 0,
            "sentiment_distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
            "avg_sentiment": 0.5,
            "error": str(e)
        }
    finally:
        client.close()

def cleanup_sentiment_data() -> bool:
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    client = get_weaviate_client()
    try:
        client.collections.delete("CryptoNewsSentiment")
        logger.info("🗑️ Collection deleted successfully.")
        return True
    except Exception as e:
        logger.warning(f"Couldn't delete collection: {e}")
        return False
    finally:
        client.close()

# CLI entry point
if __name__ == "__main__":
    stats = get_sentiment_stats()
    print("📊 Sentiment Stats:")
    print(f"Total Articles: {stats['total_articles']}")
    print(f"Avg Sentiment: {stats['avg_sentiment']:.2f}")
    print("Distribution:")
    for k, v in stats["sentiment_distribution"].items():
        print(f"  {k}: {v}")
