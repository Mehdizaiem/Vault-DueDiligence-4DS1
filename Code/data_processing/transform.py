import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_binance_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Binance API data"""
    try:
        return {
            "symbol": data.get("symbol", ""),
            "price_usd": float(data.get("price", 0)),
            "market_cap_usd": float(data.get("quoteVolume", 0)),  # Quote volume as market cap
            "volume_24h_usd": float(data.get("volume", 0)),  # 24h volume
            "price_change_24h": float(data.get("priceChangePercent", 0))  # 24h price change
        }
    except Exception as e:
        logger.error(f"Error transforming Binance data: {str(e)}, Data: {data}")
        return {}

def transform(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform the raw API data into a standardized format."""
    if not data:
        logger.warning("No data to transform")
        return []

    transformed_data = []
    success_count = 0
    error_count = 0
    
    for item in data:
        try:
            source = item.get("source", "").lower()
            raw_data = item.get("data", {})

            # Only process Binance data
            if source != "binance":
                continue

            if not raw_data:
                logger.warning("Missing data in item")
                error_count += 1
                continue

            transformed = transform_binance_data(raw_data)

            # Only add if we have valid price data
            if transformed.get('price_usd', 0) > 0:
                transformed_data.append({
                    "source": source,
                    "data": transformed
                })
                success_count += 1
            else:
                error_count += 1

        except Exception as e:
            logger.error(f"Error transforming item: {str(e)}")
            error_count += 1
            continue

    logger.info(f"\nTransformation Summary:")
    logger.info(f"Successfully transformed Binance records: {success_count}")
    logger.info(f"Failed transformations: {error_count}")
    logger.info(f"Total processed: {success_count + error_count}")
    
    # Log sample data
    if transformed_data:
        sample = transformed_data[0]
        logger.info(f"\nSample transformed data:")
        logger.info(f"Symbol: {sample['data']['symbol']}")
        logger.info(f"Price: ${sample['data']['price_usd']:,.2f}")
        logger.info(f"24h Change: {sample['data']['price_change_24h']}%")
    
    return transformed_data

if __name__ == "__main__":
    # Test data
    test_data = [
        {
            "source": "binance",
            "data": {
                "symbol": "BTCUSDT",
                "price": "50000",
                "volume": "1000000",
                "priceChangePercent": "-2.5"
            }
        }
    ]
    
    result = transform(test_data)
    print("Test transformation result:", result)