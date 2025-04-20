import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
from multiprocessing import freeze_support

# Path Setup
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, "..", ".."))
sys.path.insert(0, project_root)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Delayed import so it doesn't trigger on multiprocessing load
analyzer = None

def store_sentiment_data(df: pd.DataFrame) -> bool:
    try:
        analyzed_df = analyzer.analyze_dataframe(df)

        # Normalize scores
        scores = analyzed_df['sentiment_score'].values
        if len(scores) > 0:
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
                analyzed_df['sentiment_score'] = normalized.round(3)
                logger.info("✅ Sentiment scores normalized before storage")

        return analyzer.store_in_weaviate(analyzed_df)
    except Exception as e:
        logger.error(f"Failed to analyze/store sentiment data: {e}")
        return False

def fix_and_reload_data() -> bool:
    logger.info("Fixing and reloading data...")
    from pathlib import Path
    file_path = Path(project_root) / "Sample_Data" / "data_ingestion" / "processed" / "crypto_news.csv"
    if not file_path.exists():
        logger.error(f"Missing input CSV: {file_path}")
        return False
    df = pd.read_csv(file_path)
    return store_sentiment_data(df)

def main():
    global analyzer
    from Code.data_processing.sentiment_analyzer import CryptoSentimentAnalyzer
    analyzer = CryptoSentimentAnalyzer()

    default_file = os.path.join(project_root, "Sample_Data", "data_ingestion", "processed", "crypto_news.csv")
    print(f"📁 Looking for: {default_file}")

    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        logger.info("Running in fix mode - will reload all data with corrected date formats")
        if fix_and_reload_data():
            logger.info("✅ Successfully fixed and reloaded data")
        else:
            logger.error("❌ Failed to fix data")
        sys.exit(0)

    if os.path.exists(default_file):
        results = analyzer.run(input_file=default_file)
        if not results.empty:
            print(f"\n📊 Sentiment distribution: \n{results['sentiment_label'].value_counts()}")
            print("\n🧠 Judgment Sentences (Clean View):")
            for idx, row in results.iterrows():
                print(f"\n{idx+1}. 📰 Title: {row.get('title', '')[:100]}...")
                print(f"   🔍 Aspect: {row.get('aspect', 'N/A')}")
                print(f"   📊 Sentiment: {row['sentiment_label']} (score={row['sentiment_score']:.2f})")
                print(f"   💬 Explanation: {row['explanation']}")
                try:
                    top_sentences = json.loads(row["top_sentences"])
                    for s_idx, s in enumerate(top_sentences, start=1):
                        if s['text'] != row['explanation']:
                            scaled = (
                                0.6 + 0.4 * s['score'] if s['label'] == 'POSITIVE' else
                                0.4 * s['score'] if s['label'] == 'NEGATIVE' else
                                0.5
                            )
                            print(f"     {s_idx}. [{s['label']}] (confidence={s['score']:.2f}, scaled={scaled:.2f}): {s['text'][:120]}...")
                except:
                    print("     ⚠️ Error loading top sentences.")
        else:
            print("❌ Analysis returned no results.")
    else:
        print(f"❌ Default news file not found: {default_file}")
        print("⚠️ Run the news_scraper.py script first to generate news data.")

if __name__ == "__main__":
    freeze_support()
    main()
