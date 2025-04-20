import logging
import weaviate

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_recent_sentiment_entries(limit=10):
    """
    Query recent sentiment analysis entries from Weaviate database.
    """
    try:
        client = weaviate.connect_to_local(port=9090, grpc_port=50051)

        if not client.is_ready():
            logger.error("Weaviate is not ready")
            return []

        logger.info("Weaviate is ready ✅")
        collection = client.collections.get("CryptoNewsSentiment")
        entries = collection.query.fetch_objects(limit=limit)

        results = []
        for entry in entries.objects:
            results.append({
                "title": entry.properties.get("title"),
                "date": entry.properties.get("date"),
                "authors": entry.properties.get("authors"),
                "sentiment_label": entry.properties.get("sentiment_label"),
                "sentiment_score": entry.properties.get("sentiment_score"),
                "url": entry.properties.get("url"),
                "content_preview": entry.properties.get("content")[:150] + "..." if entry.properties.get("content") else None,
                "aspect": entry.properties.get("aspect", "N/A")
            })


        return results

    except Exception as e:
        logger.error(f"Error querying Weaviate: {e}")
        return []

    finally:
        if "client" in locals():
            client.close()

if __name__ == "__main__":
    results = query_recent_sentiment_entries()
    for entry in results:
        print(f"\n📰 Title: {entry['title']}")
        print(f"🗓️  Date: {entry['date']}")
        print(f"✍️  Authors: {entry['authors']}")
        print(f"📊 Sentiment Label: {entry['sentiment_label']}")
        print(f"💯 Sentiment Score: {entry['sentiment_score']}")
        print(f"🔗 URL: {entry['url']}")
        print(f"🧠 Content Preview: {entry['content_preview']}")
        print(f"🔍 Aspect: {entry.get('aspect', 'Not available')}")