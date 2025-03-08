# File path: load_time_series.py
#!/usr/bin/env python

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("time_series_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def check_weaviate():
    """Check if Weaviate is running"""
    logger.info("Checking Weaviate connection...")
    
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        client = get_weaviate_client()
        if client.is_live():
            logger.info("✅ Weaviate is running")
            client.close()
            return True
        else:
            logger.error("❌ Weaviate is not responding")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return False

def setup_crypto_time_series_schema():
    """Set up the CryptoTimeSeries schema if it doesn't exist"""
    logger.info("Setting up CryptoTimeSeries schema...")
    
    try:
        from Sample_Data.vector_store.schema_manager import create_crypto_time_series_schema
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        
        client = get_weaviate_client()
        try:
            create_crypto_time_series_schema(client)
            logger.info("✅ CryptoTimeSeries schema created or already exists")
            return True
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error setting up schema: {e}")
        return False

def list_available_symbols():
    """List all available cryptocurrency symbols in CSV files"""
    try:
        from Code.data_processing.csv_loader import CryptoCSVLoader
        
        loader = CryptoCSVLoader()
        symbols = loader.get_available_symbols()
        
        if symbols:
            print(f"\nFound {len(symbols)} symbols in CSV files:")
            
            # Format in columns
            columns = 4
            for i in range(0, len(symbols), columns):
                row = symbols[i:i+columns]
                print("  ".join(f"{sym:<10}" for sym in row))
        else:
            print("No symbols found in CSV files")
            
        return symbols
    except Exception as e:
        logger.error(f"Error listing symbols: {e}")
        return []

def load_single_symbol(symbol):
    """Load and store data for a single symbol"""
    logger.info(f"Loading data for symbol: {symbol}")
    
    try:
        from Code.data_processing.time_series_manager import TimeSeriesManager
        
        manager = TimeSeriesManager()
        try:
            start_time = time.time()
            success = manager.load_and_store_symbol(symbol)
            
            if success:
                elapsed = time.time() - start_time
                print(f"✅ Successfully loaded data for {symbol} in {elapsed:.2f} seconds")
                return True
            else:
                print(f"❌ Failed to load data for {symbol}")
                return False
        finally:
            manager.close()
    except Exception as e:
        logger.error(f"Error loading symbol {symbol}: {e}")
        return False

def load_all_symbols(force=False):
    """Load and store data for all available symbols"""
    logger.info("Loading data for all symbols")
    
    try:
        from Code.data_processing.time_series_manager import TimeSeriesManager
        
        manager = TimeSeriesManager()
        try:
            start_time = time.time()
            results = manager.load_and_store_all(force=force)
            
            elapsed = time.time() - start_time
            
            print(f"\nProcessed {results['total_symbols']} symbols in {elapsed:.2f} seconds:")
            print(f"  ✅ Success: {results['success_count']} symbols")
            print(f"  ❌ Failed: {results['failure_count']} symbols")
            print(f"  ⏭️ Skipped: {results['skipped_count']} symbols")
            print(f"  📊 Total data points: {results['total_data_points']}")
            
            return results['success_count'] > 0
        finally:
            manager.close()
    except Exception as e:
        logger.error(f"Error loading all symbols: {e}")
        return False

def validate_stored_data():
    """Validate that data was properly stored in Weaviate"""
    logger.info("Validating stored time series data...")
    
    try:
        from Sample_Data.vector_store.storage_manager import StorageManager
        from Code.data_processing.csv_loader import CryptoCSVLoader
        
        # Get available symbols
        loader = CryptoCSVLoader()
        symbols = loader.get_available_symbols()
        
        if not symbols:
            print("No symbols found in CSV files")
            return False
        
        # Get random symbol for validation
        import random
        sample_symbol = random.choice(symbols)
        
        # Create storage manager
        storage = StorageManager()
        try:
            # Retrieve data for symbol
            data = storage.retrieve_time_series(sample_symbol, limit=5)
            
            if data:
                print(f"\nValidation successful! Found data for {sample_symbol} in Weaviate:")
                for i, point in enumerate(data[:3]):  # Show first 3 points
                    print(f"\nData point {i+1}:")
                    print(f"  Date: {point.get('timestamp')}")
                    print(f"  Open: {point.get('open')}")
                    print(f"  High: {point.get('high')}")
                    print(f"  Low: {point.get('low')}")
                    print(f"  Close: {point.get('close')}")
                    print(f"  Volume: {point.get('volume')}")
                
                if len(data) > 3:
                    print(f"\n... and {len(data)-3} more data points")
                
                return True
            else:
                print(f"No data found for {sample_symbol} in Weaviate")
                return False
        finally:
            storage.close()
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Load cryptocurrency time series data into Weaviate")
    parser.add_argument("--list", action="store_true", help="List available symbols")
    parser.add_argument("--symbol", type=str, help="Load specific symbol")
    parser.add_argument("--all", action="store_true", help="Load all available symbols")
    parser.add_argument("--force", action="store_true", help="Force update even if data exists")
    parser.add_argument("--validate", action="store_true", help="Validate stored data")
    
    args = parser.parse_args()
    
    print("\n==== Cryptocurrency Time Series Loader ====\n")
    
    # Check Weaviate connection
    if not check_weaviate():
        return 1
    
    # Set up schema
    if not setup_crypto_time_series_schema():
        return 1
    
    # Execute requested operation
    if args.list:
        list_available_symbols()
        
    elif args.symbol:
        load_single_symbol(args.symbol)
        
    elif args.all:
        load_all_symbols(args.force)
        
    elif args.validate:
        validate_stored_data()
        
    else:
        # Interactive mode
        while True:
            print("\nOptions:")
            print("1. List available symbols")
            print("2. Load specific symbol")
            print("3. Load all symbols")
            print("4. Validate stored data")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                list_available_symbols()
            elif choice == "2":
                symbol = input("Enter symbol to load: ").strip()
                if symbol:
                    load_single_symbol(symbol)
            elif choice == "3":
                force = input("Force update? (y/n): ").lower() == 'y'
                load_all_symbols(force)
            elif choice == "4":
                validate_stored_data()
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid choice")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())