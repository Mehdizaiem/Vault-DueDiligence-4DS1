import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from typing import List, Dict, Any

def transform(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform the raw API data into a standardized format."""
    if not data:
        return []

    transformed_data = []
    
    for item in data:
        source = item.get("source")
        item_data = item.get("data", {})

        if not source or not isinstance(item_data, dict):
            print(f"Invalid or missing data for source: {source}")
            continue

        # Clean and convert data values
        cleaned_data = {
            "source": source,
            "data": {
                k: (float(v) if isinstance(v, (str, int, float)) and str(v).replace('.', '').isdigit() else v)
                for k, v in item_data.items() if v is not None  # Exclude None values
            }
        }
        transformed_data.append(cleaned_data)
    
    return transformed_data