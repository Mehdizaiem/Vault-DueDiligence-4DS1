def clean_twitter_data(raw_tweets):
    cleaned_tweets = []
    for tweet in raw_tweets:
        cleaned_tweets.append({
            "tweet_id": tweet["tweet_id"],
            "user": tweet["user"],
            "text": tweet["text"].replace("\n", " ").strip(),
            "created_at": tweet["created_at"],
        })
    return cleaned_tweets
