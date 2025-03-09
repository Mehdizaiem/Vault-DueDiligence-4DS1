# File path: setup.py
#!/usr/bin/env python
import os
import sys
import logging
import argparse
import json
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "weaviate-client",
        "pandas",
        "numpy",
        "matplotlib",
        "requests",
        "sentence-transformers",
        "transformers",
        "torch",
        "scikit-learn",
        "textblob"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        
        install = input("Would you like to install missing packages? (y/n): ")
        if install.lower() == 'y':
            cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
            logger.info(f"Installing packages: {' '.join(missing_packages)}")
            subprocess.run(cmd)
            return True
        else:
            logger.error("Required packages missing. Please install them manually.")
            return False
    
    logger.info("All dependencies are satisfied")
    return True

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
        
        # Check if docker is running
        try:
            result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
            if "weaviate" in result.stdout:
                logger.info("Weaviate container is running, but connection failed")
            else:
                logger.warning("Weaviate container not found. Starting docker-compose...")
                
                # Try to start docker-compose
                docker_compose_path = os.path.join(project_root, "Sample_Data", "docker-compose.yml")
                if os.path.exists(docker_compose_path):
                    subprocess.run(["docker-compose", "-f", docker_compose_path, "up", "-d"])
                    logger.info("Docker-compose started. Waiting for Weaviate to initialize...")
                    
                    # Wait for initialization
                    import time
                    time.sleep(10)
                    
                    # Try connecting again
                    try:
                        client = get_weaviate_client()
                        if client.is_live():
                            logger.info("✅ Weaviate is now running")
                            client.close()
                            return True
                        else:
                            logger.error("❌ Weaviate still not responding after startup")
                            return False
                    except Exception as e2:
                        logger.error(f"Error connecting to Weaviate after startup: {e2}")
                        return False
                else:
                    logger.error(f"Docker-compose file not found at {docker_compose_path}")
        except Exception as docker_error:
            logger.error(f"Error checking Docker: {docker_error}")
            logger.error("Please make sure Docker and Weaviate are installed and running")
        
        return False