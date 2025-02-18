# config.py
from dotenv import load_dotenv
import os
from pathlib import Path

def load_config():
    try:
        # Get the absolute path to .env.local
        env_path = Path(__file__).parent / '.env.local'
        
        # Print debug information
        print(f"Looking for .env.local at: {env_path}")
        print(f"File exists: {env_path.exists()}")
        
        # Load environment variables
        load_result = load_dotenv(env_path)
        print(f"Load result: {load_result}")
        
        # Get environment variables with fallbacks
        config = {
            'API_KEY': os.getenv('COINGECKO_API_KEY'),
            'DB_CONNECTION': os.getenv('DB_CONNECTION_STRING'),
        }
        
        # Print current values (careful with sensitive info in production)
        print("Current config values:")
        print(f"API_KEY: {'Found' if config['API_KEY'] else 'Missing'}")
        print(f"DB_CONNECTION: {'Found' if config['DB_CONNECTION'] else 'Missing'}")
        
        # Validate required environment variables
        missing_vars = [key for key, value in config.items() if value is None]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        return config
        
    except Exception as e:
        print(f"Error in load_config: {str(e)}")
        raise