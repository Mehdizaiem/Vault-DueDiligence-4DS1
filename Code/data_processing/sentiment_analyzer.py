import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
import json
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)



def extract_aspect(text: str) -> str:
    """Simple rule-based aspect extraction"""
    text = text.lower()
    if any(word in text for word in ["sec", "regulation", "lawsuit", "ban", "legal", "policy"]):
        return "regulation"
    elif any(word in text for word in ["bitcoin", "ethereum", "solana", "altcoin", "crypto"]):
        return "cryptocurrency"
    elif any(word in text for word in ["bullish", "market", "sell-off", "bearish", "rally", "price"]):
        return "market"
    elif any(word in text for word in ["blockchain", "technology", "smart contract", "nft", "web3"]):
        return "technology"
    else:
        return "general"

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
logger.info("Using FinBERT for sentiment analysis")


class CryptoSentimentAnalyzer:
    """Analyzes sentiment of crypto news articles using TextBlob"""
    
    def __init__(self):
        """Initialize with TextBlob for sentiment analysis"""
        logger.info("Initializing sentiment analyzer with TextBlob")
        self.use_transformers = False
        
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            result = sentiment_pipeline(text[:512])[0]
            label = result["label"].upper()
            score = result["score"]

            # Scale to 0-1
            if "POS" in label:
                scaled = 0.6 + 0.4 * score
            elif "NEG" in label:
                scaled = 0.4 - 0.4 * score
            else:
                scaled = 0.5

            return {"label": label, "score": round(scaled, 3)}
        
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
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
            
            # Format date to RFC3339
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            aspect = extract_aspect(text)

            results.append({
                **row.to_dict(),
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "aspect": aspect,
                "analyzed_at": current_time
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
                        # Format current date in RFC3339 format
                        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                        
                        data = {
                            "source": row["source"],
                            "title": row["title"],
                            "url": row["url"],
                            "content": row["content"],
                            "sentiment_label": row["sentiment_label"],
                            "sentiment_score": float(row["sentiment_score"]),
                            "analyzed_at": current_time
                        }
                        
                        # Handle optional fields
                        if "date" in row and row["date"]:
                            # Format date to RFC3339
                            if isinstance(row["date"], datetime):
                                data["date"] = row["date"].strftime("%Y-%m-%dT%H:%M:%SZ")
                            elif isinstance(row["date"], str):
                                # Try to parse and format the string date
                                try:
                                    # Try multiple date formats
                                    try:
                                        parsed_date = datetime.fromisoformat(row["date"].replace('Z', '+00:00'))
                                    except:
                                        # Try alternate formats
                                        try:
                                            parsed_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S.%f")
                                        except:
                                            try:
                                                # Try timestamp format
                                                if str(row["date"]).isdigit() and len(str(row["date"])) > 10:
                                                    # Unix timestamp in milliseconds
                                                    parsed_date = datetime.fromtimestamp(int(row["date"])/1000)
                                                else:
                                                    # Default to current date
                                                    parsed_date = datetime.now()
                                            except:
                                                parsed_date = datetime.now()
                                    
                                    data["date"] = parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                                except:
                                    # Use current time as fallback
                                    data["date"] = current_time
                            else:
                                data["date"] = current_time
                        else:
                            data["date"] = current_time
                        
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
                    try:
                        response = collection.data.insert_many(objects)
                        total += len(objects)
                        logger.info(f"Stored {len(objects)} articles in Weaviate")
                    except Exception as e:
                        logger.error(f"Failed to insert batch: {e}")
                        # Try inserting one by one
                        successful = 0
                        for obj in objects:
                            try:
                                collection.data.insert(obj)
                                successful += 1
                            except Exception as individual_e:
                                logger.error(f"Failed to insert individual object: {individual_e}")
                        
                        if successful > 0:
                            total += successful
                            logger.info(f"Stored {successful} articles individually after batch failure")
            
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
    def aggregate_weighted_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate weighted sentiment by date and aspect.
        """
        if "sentiment_score" not in df or "aspect" not in df or "date" not in df:
            logger.warning("Missing columns for sentiment aggregation.")
            return pd.DataFrame()
        
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["weight"] = df["content"].str.len().fillna(100)  # fallback weight
        
        grouped = df.groupby(["date", "aspect"]).apply(
            lambda x: np.average(x["sentiment_score"], weights=x["weight"])
        ).reset_index(name="adjusted_score")
        
        return grouped
def plot_sentiment_trends(sentiment_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sentiment_df.empty or "date" not in sentiment_df or "aspect" not in sentiment_df:
        logger.warning("Invalid sentiment DataFrame for plotting")
        return
    
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    df_grouped = sentiment_df.groupby(["date", "aspect"])["sentiment_score"].mean().reset_index()
    df_pivot = df_grouped.pivot(index="date", columns="aspect", values="sentiment_score").fillna(0)
    df_pivot_rolled = df_pivot.rolling(window=7, min_periods=1).mean()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_pivot_rolled)
    plt.title("📈 7-Day Rolling Sentiment by Aspect")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="Aspect")
    plt.show()

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