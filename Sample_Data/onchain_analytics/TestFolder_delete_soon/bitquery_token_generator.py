# bitquery_token_generator.py
import os
import requests
import json
import logging
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bitquery_token.log")
    ]
)
logger = logging.getLogger(__name__)

def load_env_variables():
    """Load environment variables from .env.local"""
    # Path to the root directory where .env.local is located
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    env_path = os.path.join(root_dir, '.env.local')
    
    if os.path.exists(env_path):
        logger.info(f"Loading .env.local from: {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logger.warning(f".env.local not found at {env_path}, trying default .env")
        load_dotenv()
    
    # Get the client ID and client secret
    client_id = os.getenv("BITQUERY_CLIENT_ID")
    client_secret = os.getenv("BITQUERY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        logger.warning("BITQUERY_CLIENT_ID or BITQUERY_CLIENT_SECRET not found in environment variables")
        logger.info("You will need to enter them manually if you want to generate a token programmatically")
    
    return client_id, client_secret

def generate_token_programmatically(client_id=None, client_secret=None):
    """Generate a token programmatically using client credentials"""
    if not client_id or not client_secret:
        logger.info("No client ID or client secret provided from environment variables")
        use_manual = input("Would you like to enter client ID and secret manually? (y/n): ")
        if use_manual.lower() == 'y':
            client_id = input("Enter your BitQuery Client ID: ")
            client_secret = input("Enter your BitQuery Client Secret: ")
        else:
            logger.info("Cannot generate token without client credentials")
            return None
    
    url = "https://oauth2.bitquery.io/oauth2/token"
    payload = f'grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}&scope=api'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        logger.info("Requesting access token from BitQuery OAuth server...")
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        
        if response.status_code == 200:
            resp_data = response.json()
            logger.info(f"Successfully retrieved access token")
            logger.info(f"Token expires in: {resp_data.get('expires_in')} seconds")
            logger.info(f"Token scope: {resp_data.get('scope')}")
            
            # Save token to a file
            access_token = resp_data.get('access_token')
            token_type = resp_data.get('token_type')
            
            token_info = {
                'access_token': access_token,
                'token_type': token_type,
                'generated_at': time.time(),
                'expires_in': resp_data.get('expires_in')
            }
            
            with open("bitquery_token.json", "w") as f:
                json.dump(token_info, f, indent=2)
            
            logger.info("Token saved to bitquery_token.json")
            
            # Update .env.local with the new token
            update_env = input("Would you like to update your .env.local with the new token? (y/n): ")
            if update_env.lower() == 'y':
                update_env_file(access_token)
            
            return access_token
        else:
            logger.error(f"Failed to retrieve token: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        logger.error(f"Error generating token: {str(e)}")
        return None

def update_env_file(token):
    """Update the .env.local file with the new token"""
    # Path to the root directory where .env.local is located
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    env_path = os.path.join(root_dir, '.env.local')
    
    if not os.path.exists(env_path):
        logger.warning(f".env.local not found at {env_path}")
        create_env = input("Would you like to create a new .env.local file? (y/n): ")
        if create_env.lower() != 'y':
            return False
    
    try:
        # Read current .env.local file
        env_content = ""
        token_line_exists = False
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
                
                for line in env_lines:
                    if line.startswith('BITQUERY_API_KEY='):
                        # Replace the token
                        env_content += f"BITQUERY_API_KEY={token}\n"
                        token_line_exists = True
                    else:
                        env_content += line
        
        # If BITQUERY_API_KEY line doesn't exist, add it
        if not token_line_exists:
            env_content += f"\nBITQUERY_API_KEY={token}\n"
        
        # Write back to .env.local
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        logger.info(f"Successfully updated {env_path} with new token")
        return True
    
    except Exception as e:
        logger.error(f"Error updating .env file: {str(e)}")
        return False

def test_token(token):
    """Test if the token works by making a simple API request"""
    if not token:
        logger.error("No token provided to test")
        return False
    
    url = "https://graphql.bitquery.io"
    
    # Simple query to test the token
    query = """
    {
      ethereum {
        blocks(options: {limit: 1}) {
          number
          timestamp {
            time
          }
        }
      }
    }
    """
    
    payload = {"query": query}
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': token
    }
    
    try:
        logger.info("Testing token with a simple query...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for GraphQL errors
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return False
            
            # Check if we got a valid response
            if 'data' in data and 'ethereum' in data['data'] and 'blocks' in data['data']['ethereum']:
                blocks = data['data']['ethereum']['blocks']
                if blocks and len(blocks) > 0:
                    block_number = blocks[0]['number']
                    timestamp = blocks[0]['timestamp']['time']
                    logger.info(f"Successfully queried Ethereum block {block_number} from {timestamp}")
                    return True
            
            logger.error("Received unexpected response structure")
            logger.info(f"Response: {data}")
            return False
        else:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing token: {str(e)}")
        return False

def use_existing_token():
    """Use the token from .env.local or from the saved token file"""
    # First try from .env.local
    load_env_variables()
    token = os.getenv("BITQUERY_API_KEY")
    
    if token:
        logger.info(f"Found token in environment variables: {token[:5]}...{token[-5:]}")
        valid = test_token(token)
        if valid:
            logger.info("Token is valid!")
            return token
        else:
            logger.warning("Token from environment variables is invalid")
    
    # Try from saved token file
    try:
        if os.path.exists("bitquery_token.json"):
            with open("bitquery_token.json", "r") as f:
                token_info = json.load(f)
            
            token = token_info.get('access_token')
            generated_at = token_info.get('generated_at', 0)
            expires_in = token_info.get('expires_in', 0)
            
            # Check if token is expired (with 10 minute buffer)
            current_time = time.time()
            expiration_time = generated_at + expires_in - 600  # 10 minute buffer
            
            if current_time > expiration_time:
                logger.warning("Saved token is expired or about to expire")
            else:
                logger.info(f"Found unexpired token in bitquery_token.json")
                valid = test_token(token)
                if valid:
                    logger.info("Token is valid!")
                    return token
                else:
                    logger.warning("Token from bitquery_token.json is invalid")
    
    except Exception as e:
        logger.error(f"Error reading saved token: {str(e)}")
    
    return None

def main():
    """Main function to manage BitQuery tokens"""
    logger.info("BitQuery Token Manager")
    
    # Try to use existing token
    token = use_existing_token()
    
    if token:
        logger.info("Using existing valid token")
    else:
        logger.info("No valid token found. Generating a new one...")
        client_id, client_secret = load_env_variables()
        token = generate_token_programmatically(client_id, client_secret)
        
        if token:
            test_token(token)
    
    if token:
        logger.info("\nToken to use in your API requests:")
        logger.info(f"X-API-KEY: {token}")
        logger.info("\nFor authorization header:")
        logger.info(f"Authorization: Bearer {token}")
        
        logger.info("\nTo use this token in your code:")
        logger.info('headers = {"Content-Type": "application/json", "X-API-KEY": "' + token + '"}')
        logger.info('# OR')
        logger.info('headers = {"Content-Type": "application/json", "Authorization": "Bearer ' + token + '"}')

if __name__ == "__main__":
    main()