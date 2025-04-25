import weaviate

# Connect to local Weaviate instance
client = weaviate.connect_to_local(
    port=9090,
    grpc_port=50051,
)

if client.is_ready():
    print("Weaviate is ready ✅")

    # List all collections (classes)
    schema = client.collections.list_all()
    print("Collections (Classes) in schema:")
    for collection_name in schema:
        print(f"📁 {collection_name}")

    # Access the CryptoNewsSentiment collection
    collection = client.collections.get("CryptoNewsSentiment")

    # Fetch and display recent sentiment-analyzed entries
    print("\n📊 Recent Sentiment Analysis Entries:\n")
    entries = collection.query.fetch_objects(limit=10)

    for entry in entries.objects:
        print(f"📰 Title: {entry.properties.get('title')}")
        print(f"🗓️  Date: {entry.properties.get('date')}")
        print(f"✍️  Authors: {entry.properties.get('authors')}")
        print(f"📊 Sentiment Label: {entry.properties.get('sentiment_label')}")
        print(f"💯 Sentiment Score: {entry.properties.get('sentiment_score')}")
        print(f"🔗 URL: {entry.properties.get('url')}")
        content = entry.properties.get('content')
        if content:
            print("🧠 Content Preview:", content[:150] + "...\n")
        else:
            print("🧠 Content: None\n")

else:
    print("Weaviate is not ready ❌")

client.close()
