# test_system.py
import os
import sys
from agentic_rag import AgenticRagSystem

def test_all_components():
    system = AgenticRagSystem()
    system.initialize()
    
    print("Testing news scraping...")
    news_df = system.scrape_news()
    
    print("Testing sentiment analysis...")
    sentiment_df = system.analyze_sentiment()
    
    print("Testing forecasting...")
    forecasts = system.update_forecasts()
    
    print("Testing search...")
    results = system.process_user_query("Bitcoin price trends")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_all_components()