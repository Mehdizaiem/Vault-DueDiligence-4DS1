from typing import List, Dict, Any
from collections import defaultdict
from Code.data_acquisition.database.postgresql_setup import DatabaseManager

def load(data: List[Dict[str, Any]]) -> None:
    """Load transformed data into the database."""
    if not data:
        print("No data to load")
        return

    db_manager = DatabaseManager()
    
    # Group data by source using defaultdict for efficiency
    source_data = defaultdict(list)
    for item in data:
        source = item["source"]
        source_data[source].append(item)
    
    # Create tables and load data for each source
    for source, items in source_data.items():
        print(f"\nProcessing {source} data...")
        try:
            db_manager.create_table_for_source(source, items)
            db_manager.load_data_into_table(source, items)
            print(f"✅ Successfully processed {source} data")
        except Exception as e:
            print(f"❌ Error processing {source} data: {e}")
