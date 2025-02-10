import sys
import os
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Code.data_acquisition.blockchain_collectors.api_collectors import fetch_all

def extract() -> List[Dict[str, Any]]:
    """Extract data from all configured APIs."""
    print("Starting data extraction...")
    data = fetch_all()
    print(f"Extracted data from {len(data)} sources")
    return data