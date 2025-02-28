import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
import json
from typing import Dict, List, Optional, Union

# Configure paths
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from project
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
from Sample_Data.vector_store.market_sentiment_schema import create_sentiment_schema

# Use TextBlob for sentiment analysis (faster and more reliable)
from textblob import TextBlob
logger.info("Using TextBlob for sentiment analysis")

class CryptoSentimentAnalyzer:
    """Analyzes sentiment of crypto news articles using TextBlob"""
    
    def __init__(self):
        """Initialize with TextBlob for sentiment analysis"""
        logger.info("Initializing sentiment analyzer with TextBlob")
        self.use_transformers = False
        
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis result with label and score
        """
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5}
            
        try:
            # Use TextBlob for sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # TextBlob polarity is between -1 and 1, convert to 0-1
            score = (polarity + 1) / 2
            
            # Determine label
            if score > 0.6:
                label = "POSITIVE"
            elif score < 0.4:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
                
            return {"label": label, "score": score}
                
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def analyze_dataframe(self, df: pd.DataFrame, content_column: str = "content") -> pd.DataFrame:
        """
        Analyze sentiment for all articles in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing articles
            content_column (str): Column name containing the article text
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {len(df)} articles")
        
        results = []
        for idx, row in df.iterrows():
            text = row[content_column]
            sentiment = self.analyze_text(text)
            
            results.append({
                **row.to_dict(),
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "analyzed_at": datetime.now().isoformat()
            })
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1} articles")
        
        return pd.DataFrame(results)
    
    def store_in_weaviate(self, df: pd.DataFrame) -> bool:
        """
        Store sentiment analysis results in Weaviate.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            bool: True if successful, False otherwise
        """
        client = get_weaviate_client()
        
        try:
            # Create schema if it doesn't exist
            create_sentiment_schema(client)
            
            # Get the collection
            collection = client.collections.get("CryptoNewsSentiment")
            
            # Store data in batches
            batch_size = 50
            total = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                objects = []
                
                for _, row in batch.iterrows():
                    try:
                        data = {
                            "source": row["source"],
                            "title": row["title"],
                            "url": row["url"],
                            "content": row["content"],
                            "sentiment_label": row["sentiment_label"],
                            "sentiment_score": float(row["sentiment_score"]),
                            "analyzed_at": row["analyzed_at"]
                        }
                        
                        # Handle optional fields
                        if "date" in row and row["date"]:
                            if isinstance(row["date"], str):
                                data["date"] = row["date"]
                            else:
                                data["date"] = row["date"].isoformat() if hasattr(row["date"], 'isoformat') else str(row["date"])
                        
                        if "authors" in row and isinstance(row["authors"], list):
                            data["authors"] = row["authors"]
                        elif "authors" in row and isinstance(row["authors"], str):
                            # Handle string representation of list
                            try:
                                authors = json.loads(row["authors"].replace("'", "\""))
                                data["authors"] = authors if isinstance(authors, list) else []
                            except:
                                data["authors"] = []
                        
                        if "image_url" in row:
                            data["image_url"] = row["image_url"]
                        
                        objects.append(data)
                    except Exception as e:
                        logger.error(f"Error preparing object for Weaviate: {e}")
                
                # Insert batch
                if objects:
                    response = collection.data.insert_many(objects)
                    total += len(objects)
                    logger.info(f"Stored {len(objects)} articles in Weaviate")
            
            logger.info(f"Successfully stored {total} articles with sentiment analysis in Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Error storing sentiment data in Weaviate: {e}")
            return False
        finally:
            if client:
                client.close()
    
    def run(self, input_file: Optional[str] = None, df: Optional[pd.DataFrame] = None, store_in_db: bool = True) -> pd.DataFrame:
        """
        Run analysis on CSV file or DataFrame of news articles.
        
        Args:
            input_file (str, optional): Path to CSV file with news articles
            df (pd.DataFrame, optional): DataFrame with news articles
            store_in_db (bool): Whether to store results in Weaviate
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        try:
            # Load data
            if df is not None:
                # Use provided DataFrame
                pass
            elif input_file is not None:
                # Load from CSV
                df = pd.read_csv(input_file)
                logger.info(f"Loaded {len(df)} articles from {input_file}")
            else:
                # Try default location
                default_file = os.path.join(project_root, "Sample_Data", "data_ingestion", "processed", "crypto_news.csv")
                if os.path.exists(default_file):
                    df = pd.read_csv(default_file)
                    logger.info(f"Loaded {len(df)} articles from {default_file}")
                else:
                    logger.error("No input data provided and default file not found")
                    return pd.DataFrame()
            
            # Analyze sentiment
            results_df = self.analyze_dataframe(df)
            
            # Save results to CSV
            output_dir = os.path.join(project_root, "Sample_Data", "data_ingestion", "processed")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"sentiment_analysis_{datetime.now().strftime('%Y%m%d')}.csv")
            results_df.to_csv(output_file, index=False)
            logger.info(f"Saved sentiment analysis results to {output_file}")
            
            # Store in Weaviate if requested
            if store_in_db:
                self.store_in_weaviate(results_df)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis run: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    analyzer = CryptoSentimentAnalyzer()
    
    # Try to find the default news CSV
    default_file = os.path.join(project_root, "Sample_Data", "data_ingestion", "processed", "crypto_news.csv")
    
    if os.path.exists(default_file):
        results = analyzer.run(input_file=default_file)
        if not results.empty:
            print(f"Sentiment distribution: {results['sentiment_label'].value_counts()}")
    else:
        print(f"Default news file not found: {default_file}")
        print("Run the news_scraper.py script first to generate news data.")