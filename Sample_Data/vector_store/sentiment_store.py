import pandas as pd
import numpy as np
import logging
import os
import sys
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob

import os
import sys

# Dynamically set root
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Code.data_processing.sentiment_analyzer import CryptoSentimentAnalyzer


# --- Logging config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Using FinBERT for sentiment analysis")

# Initialize the analyzer
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
    from weaviate.exceptions import WeaviateBaseError

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
            label = obj.properties.get("sentiment_label", "NEUTRAL").upper()
            score = obj.properties.get("sentiment_score", 0.5)

            # Clamp scores to valid range [0.0, 1.0]
            score = max(0.0, min(1.0, score))

            # Count by label
            if label not in sentiment_distribution:
                label = "NEUTRAL"  # fallback
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
# Path setup before import
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.insert(0, project_root)

from Code.data_processing.sentiment_analyzer import CryptoSentimentAnalyzer


def main():
    analyzer = CryptoSentimentAnalyzer()
    stats = get_sentiment_stats()
    print("📊 Sentiment Stats:")
    print(f"Total Articles: {stats['total_articles']}")
    print(f"Avg Sentiment: {stats['avg_sentiment']:.2f}")
    print("Distribution:")
    for k, v in stats["sentiment_distribution"].items():
        print(f"  {k}: {v}")
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Required on Windows to avoid spawn issues

    analyzer = CryptoSentimentAnalyzer()

    default_file = os.path.join(project_root, "Sample_Data", "data_ingestion", "processed", "crypto_news.csv")

    if os.path.exists(default_file):
        results = analyzer.run(input_file=default_file)
        if not results.empty:
            print(f"Sentiment distribution: {results['sentiment_label'].value_counts()}")
    else:
        print(f"Default news file not found: {default_file}")
        print("Run the news_scraper.py script first to generate news data.")

