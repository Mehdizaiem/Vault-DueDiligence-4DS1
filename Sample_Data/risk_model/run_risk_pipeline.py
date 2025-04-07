import sys
import os
from pathlib import Path
import logging

# Find the project root (parent of Sample_Data)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Go up three levels: risk -> Sample_Data -> root
sys.path.insert(0, str(project_root))

# Original imports will now work
from Sample_Data.risk_model.ml_risk_evaluator import MLRiskEvaluator  # Use our new ML evaluator
from Sample_Data.risk_model.store_risk_profiles import store_profiles
from Sample_Data.vector_store.weaviate_client import get_weaviate_client

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_latest_market_data():
    client = get_weaviate_client()
    try:
        collection = client.collections.get("MarketMetrics")
        response = collection.query.fetch_objects(limit=1000, include_vector=False)
        return [
            {
                "fund_id": obj.properties["symbol"],
                "market_data": {
                    "volume_24h": obj.properties.get("volume_24h", 0),
                    "price_change_24h": obj.properties.get("price_change_24h", 0),
                    "market_cap": obj.properties.get("market_cap", 0),
                    # Add additional metrics if available in your data
                    "volatility": obj.properties.get("volatility", 0.1),
                    "liquidity_ratio": obj.properties.get("liquidity_ratio", 1.0)
                }
            }
            for obj in response.objects
        ]
    finally:
        client.close()

if __name__ == "__main__":
    # Check if model exists, if not, train it first
    model_path = Path(__file__).parent / "models" / "risk_model.pkl"
    if not model_path.exists():
        logger.info("Pre-trained model not found. Training a new model first...")
        from Sample_Data.risk_model.train_risk_model import generate_synthetic_data, train_model
        df = generate_synthetic_data(n_samples=2000)
        train_model(df, model_path.parent)
    
    # Create ML-based evaluator with model
    evaluator = MLRiskEvaluator(model_path=str(model_path))
    
    # Fetch market data
    logger.info("Fetching latest market data...")
    raw_data = fetch_latest_market_data()
    
    # Process data and generate risk profiles
    logger.info(f"Evaluating risk for {len(raw_data)} funds...")
    all_profiles = []
    for item in raw_data:
        profile = evaluator.evaluate(item["fund_id"], item["market_data"])
        all_profiles.append(profile)
    
    # Store profiles in database
    logger.info(f"Storing {len(all_profiles)} risk profiles...")
    store_profiles(all_profiles)
    
    logger.info("Risk evaluation pipeline completed successfully")