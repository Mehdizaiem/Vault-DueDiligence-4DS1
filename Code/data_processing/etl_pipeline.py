import sys
import os
from datetime import datetime
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from extract import extract
from transform import transform
from load import load

def etl_pipeline():
    """Execute the complete ETL pipeline."""
    start_time = datetime.now()
    print(f"\n{'='*50}")
    print(f"Starting ETL Pipeline at {start_time}")
    print(f"{'='*50}\n")

    try:
        # Extract
        print("1. Extraction Phase")
        print("-"*20)
        extract_start_time = datetime.now()
        raw_data = extract()
        if not raw_data:
            print("❌ No data extracted. Stopping pipeline.")
            return
        extract_duration = datetime.now() - extract_start_time
        print(f"✅ Extraction complete - {len(raw_data)} records retrieved in {extract_duration}\n")

        # Transform
        print("2. Transformation Phase")
        print("-"*20)
        transform_start_time = datetime.now()
        transformed_data = transform(raw_data)
        if not transformed_data:
            print("❌ No data transformed. Stopping pipeline.")
            return
        transform_duration = datetime.now() - transform_start_time
        print(f"✅ Transformation complete - {len(transformed_data)} records processed in {transform_duration}\n")

        # Load
        print("3. Loading Phase")
        print("-"*20)
        load_start_time = datetime.now()
        load(transformed_data)
        load_duration = datetime.now() - load_start_time
        print(f"✅ Loading complete in {load_duration}\n")

        end_time = datetime.now()
        total_duration = end_time - start_time
        print(f"{'='*50}")
        print(f"ETL Pipeline completed successfully at {end_time}")
        print(f"Total duration: {total_duration}")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    etl_pipeline()
