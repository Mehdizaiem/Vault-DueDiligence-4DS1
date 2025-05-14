import os
import sys
import json
import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk import download, data

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path setup
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.weaviate_client import get_weaviate_client
from Sample_Data.vector_store.market_sentiment_schema import create_sentiment_schema


def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()


def extract_aspect(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["sec", "regulation", "lawsuit", "ban", "legal"]):
        return "REGULATION"
    if any(w in text for w in ["bitcoin", "ethereum", "solana", "crypto"]):
        return "CRYPTOCURRENCY"
    if any(w in text for w in ["bullish", "bearish", "price", "market"]):
        return "MARKET"
    if any(w in text for w in ["blockchain", "nft", "smart contract", "web3"]):
        return "TECHNOLOGY"
    return "GENERAL"


def format_date_rfc3339(date_value):
    """Format date to RFC3339 format required by Weaviate"""
    if not date_value:
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
    try:
        if isinstance(date_value, str):
            # Handle different string formats
            if "T" in date_value and date_value.endswith("Z"):
                # Already in RFC3339 format
                return date_value
            elif "T" in date_value:
                # ISO format without Z
                dt = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
            elif "-" in date_value and ":" in date_value:
                # Format like '2025-04-15 09:47:09.316529'
                if "." in date_value:
                    dt = datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    dt = datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S")
            elif "-" in date_value:
                # Just a date like '2025-04-15'
                dt = datetime.strptime(date_value, "%Y-%m-%d")
            else:
                # Unknown format, use current time
                dt = datetime.now()
        elif isinstance(date_value, (int, float)):
            # Handle timestamp
            if date_value > 1000000000000:  # Milliseconds
                dt = datetime.fromtimestamp(date_value/1000)
            else:  # Seconds
                dt = datetime.fromtimestamp(date_value)
        else:
            # Unknown type
            dt = datetime.now()
            
        # Format to RFC3339
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
    except Exception as e:
        logger.warning(f"Date formatting error: {e}")
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


class CryptoSentimentAnalyzer:
    def __init__(self):
        logger.info("Initializing FinBERT sentiment pipeline from Hugging Face...")
        try:
            # Load from Hugging Face instead of local path
            model_name = "yiyanghkust/finbert-tone"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            
            try:
                data.find("tokenizers/punkt")
            except LookupError:
                download("punkt")
            
            logger.info("FinBERT model loaded successfully from Hugging Face")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise

    def analyze_text(self, text: str) -> Dict:
        text = preprocess_text(text)
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 5]

        if not sentences:
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "explanation": "",
                "top_sentences": [],
                "aspect": "general"
            }

        # Process each sentence
        results = []
        for sentence in sentences:
            if len(sentence) > 512:
                sentence = sentence[:512]  # Truncate to fit model max length
            try:
                result = self.pipeline(sentence)[0]
                results.append({
                    "text": sentence,
                    "label": result["label"].upper(),
                    "score": float(result["score"])
                })
            except Exception as e:
                logger.warning(f"Error processing sentence: {str(e)[:100]}")
                continue

        if not results:
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "explanation": "",
                "top_sentences": [],
                "aspect": "general"
            }

        # Count labels
        label_counts = {"POSITIVE":.0, "NEGATIVE": 0, "NEUTRAL": 0}
        for result in results:
            label_counts[result["label"]] += 1

        # Determine final label by majority vote
        # Recompute label using weighted confidence instead of simple count
        confidence_by_label = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        for result in results:
            confidence_by_label[result["label"]] += result["score"]

        labels_sorted = sorted(confidence_by_label.items(), key=lambda x: x[1], reverse=True)
        if labels_sorted[0][1] - labels_sorted[1][1] < 0.05:
            final_label = "NEUTRAL"
        else:
            final_label = labels_sorted[0][0]

        matching_sentences = [r for r in results if r["label"] == final_label]
        avg_score = sum(r["score"] for r in matching_sentences) / len(matching_sentences) if matching_sentences else 0.5

        if final_label == "POSITIVE":
            scaled_score = round(0.6 + 0.4 * avg_score, 3)
        elif final_label == "NEGATIVE":
            scaled_score = round(0.0 + 0.4 * avg_score, 3)
        else:
            variance = np.var([r["score"] for r in matching_sentences]) if matching_sentences else 0.0
            scaled_score = round(0.45 + min(0.1, variance), 3)



        # Ensure score is within bounds
        scaled_score = max(0.0, min(1.0, round(scaled_score, 3)))
        
        # Find the strongest sentiment sentence for explanation
        if matching_sentences:
            explanation_sentence = max(matching_sentences, key=lambda x: x["score"])["text"]
        else:
            explanation_sentence = results[0]["text"] if results else ""
        
        # Get top sentences (highest score for each label)
        top_sentences = []
        for label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            label_results = [r for r in results if r["label"] == label]
            if label_results:
                top = max(label_results, key=lambda x: x["score"])
                top_sentences.append({
                    "text": top["text"], 
                    "label": label, 
                    "score": round(top["score"], 3)
                })
        
        # Sort by score in descending order
        top_sentences = sorted(top_sentences, key=lambda x: x["score"], reverse=True)
        
        return {
            "label": final_label,
            "score": scaled_score,
            "explanation": explanation_sentence,
            "top_sentences": top_sentences[:3],  # Return top 3 sentences
            "aspect": extract_aspect(text)
        }

    def analyze_dataframe(self, df: pd.DataFrame, content_column: str = "content") -> pd.DataFrame:
        logger.info(f"Analyzing sentiment for {len(df)} articles")
        results = []

        for idx, row in df.iterrows():
            try:
                text = str(row[content_column]) if pd.notna(row[content_column]) else ""
                if not text.strip():
                    logger.warning(f"Empty content in row {idx}")
                    continue
                    
                sentiment = self.analyze_text(text)
                aspect = sentiment.get("aspect", "general")
                current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Create result dictionary with proper error handling
                result_dict = {k: v for k, v in row.to_dict().items()}
                result_dict.update({
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": sentiment["score"],
                    "aspect": aspect,
                    "explanation": sentiment["explanation"],
                    "top_sentences": json.dumps(sentiment["top_sentences"]),
                    "analyzed_at": current_time,
                })
                
                results.append(result_dict)
                
                if idx > 0 and idx % 10 == 0:
                    logger.info(f"Processed {idx}/{len(df)} articles")
                    
            except Exception as e:
                logger.error(f"Error analyzing row {idx}: {str(e)[:100]}")
                continue

        if not results:
            logger.warning("No results were produced during analysis")
            return pd.DataFrame()
            
        return pd.DataFrame(results)

    def store_in_weaviate(self, df: pd.DataFrame) -> bool:
        client = get_weaviate_client()
        try:
            create_sentiment_schema(client)
            collection = client.collections.get("CryptoNewsSentiment")
            batch_size = 50
            total = 0

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                objects = []

                for _, row in batch.iterrows():
                    # Convert date format to RFC3339
                    analyzed_at = row.get("analyzed_at")
                    analyzed_at = format_date_rfc3339(analyzed_at)
                    
                    # Process the date field
                    date_value = row.get("date", analyzed_at)
                    date_value = format_date_rfc3339(date_value)

                    # Process authors
                    authors = []
                    if isinstance(row.get("authors"), str):
                        try:
                            authors = json.loads(row["authors"].replace("'", "\""))
                        except:
                            if "," in row.get("authors", ""):
                                authors = [a.strip() for a in row["authors"].split(",")]
                            else:
                                authors = [row["authors"]]
                    elif isinstance(row.get("authors"), list):
                        authors = row["authors"]

                    data_obj = {
                        "source": str(row.get("source", "")),
                        "title": str(row.get("title", "")),
                        "url": str(row.get("url", "")),
                        "content": str(row.get("content", "")),
                        "sentiment_label": str(row.get("sentiment_label", "NEUTRAL")),
                        "sentiment_score": float(row.get("sentiment_score", 0.5)),
                        "analyzed_at": analyzed_at,
                        "date": date_value,
                        "authors": authors,
                        "image_url": str(row.get("image_url", "")),
                        "aspect": str(row.get("aspect", "general"))
                    }

                    objects.append(data_obj)

                if objects:
                    try:
                        collection.data.insert_many(objects)
                        total += len(objects)
                        logger.info(f"Inserted batch of {len(objects)} objects")
                    except Exception as e:
                        logger.error(f"Batch insert error: {e}")
                        # Try inserting one by one to identify problematic records
                        for obj in objects:
                            try:
                                collection.data.insert(obj)
                                total += 1
                            except Exception as e:
                                logger.error(f"Individual insert error: {e}")

            logger.info(f"✅ Successfully stored {total} articles.")
            return True

        except Exception as e:
            logger.error(f"Storage error: {e}")
            return False
        finally:
            client.close()

    def run(self, input_file: Optional[str] = None, df: Optional[pd.DataFrame] = None, store_in_db: bool = True) -> pd.DataFrame:
        if df is None and input_file:
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} articles from {input_file}")
        elif df is None:
            logger.warning("No input provided.")
            return pd.DataFrame()

        results = self.analyze_dataframe(df)
        if store_in_db:
            self.store_in_weaviate(results)
        out_path = os.path.join(project_root, "Sample_Data", "data_ingestion", "processed", f"sentiment_{datetime.now().strftime('%Y%m%d')}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results.to_csv(out_path, index=False)
        logger.info(f"Results saved to: {out_path}")
        return results

if __name__ == "__main__":
    # Simple test code
    analyzer = CryptoSentimentAnalyzer()
    test_text = "Bitcoin prices have surged over 10% today, reaching new all-time highs. Analysts are bullish on future growth."
    result = analyzer.analyze_text(test_text)
    print(f"Sentiment: {result['label']}, Score: {result['score']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Aspect: {result['aspect']}")

