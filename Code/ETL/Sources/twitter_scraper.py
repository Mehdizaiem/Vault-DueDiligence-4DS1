'''import tweepy
import logging
from Config.settings import TWITTER_API_CONFIG
from Utils.logger import log_info, log_error

def authenticate_twitter():
    """Authenticate with Twitter API using Tweepy"""
    auth = tweepy.OAuth1UserHandler(
        TWITTER_API_CONFIG["api_key"],
        TWITTER_API_CONFIG["api_secret_key"],
        TWITTER_API_CONFIG["access_token"],
        TWITTER_API_CONFIG["access_token_secret"]
    )
    return tweepy.API(auth, wait_on_rate_limit=True)

def fetch_tweets(keyword, count=50):
    try:
        api = authenticate_twitter()

        # Search for tweets containing the keyword (search API)
        tweets = api.search_tweets(q=keyword, count=count, lang="en", tweet_mode="extended")

        logging.info(f"Successfully fetched {len(tweets)} tweets related to keyword '{keyword}'")

        # Extract tweet data
        tweet_data = [{
            "tweet_id": tweet.id_str,
            "user": tweet.user.screen_name,
            "created_at": tweet.created_at,
            "text": tweet.full_text,
            "likes": tweet.favorite_count,
            "retweets": tweet.retweet_count
        } for tweet in tweets]

        return tweet_data

    except tweepy.TweepError as e:
        logging.error(f"Error fetching tweets: {e}")
        return []
'''
