#!/usr/bin/env python
# Path: main.py
import os
import sys
import logging
import argparse
from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv # Import load_dotenv

# --- Path Setup ---
# Calculate project root (CryptoFundDueDiligence directory)
project_root = os.path.dirname(os.path.abspath(__file__))
# Calculate the parent directory (the one containing CryptoFundDueDiligence and Sample_Data)
parent_dir = os.path.dirname(project_root)

# Add parent directory to sys.path to allow finding Sample_Data
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir) # Insert at beginning

# Add project root itself if not already there (good practice)
if project_root not in sys.path:
     sys.path.append(project_root)

# Ensure subdirs within the project are accessible if needed
subdirs = ['processing', 'data', 'reporting', 'models', 'utils', 'analysis', 'llm']
for subdir in subdirs:
    subdir_path = os.path.join(project_root, subdir)
    if subdir_path not in sys.path:
        sys.path.append(subdir_path)
# --- End Path Setup ---


# --- Configure logging FIRST ---
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "crypto_due_diligence.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# --- Load Environment Variables ---
# Look for .env.local in the PARENT directory first, then project root
env_path_parent = os.path.join(parent_dir, '.env.local')
env_path_project = os.path.join(project_root, '.env.local')
env_path_to_use = None

if os.path.exists(env_path_parent):
    env_path_to_use = env_path_parent
    logger.info(f"Found .env.local in parent directory: {env_path_parent}")
elif os.path.exists(env_path_project):
    env_path_to_use = env_path_project
    logger.info(f"Found .env.local in project directory: {env_path_project}")

if env_path_to_use:
    logger.info(f"Loading environment variables from: {env_path_to_use}")
    load_dotenv(dotenv_path=env_path_to_use, override=True)
else:
    logger.warning(".env.local file not found in parent or project directory. Ensure GROQ_API_KEY and other necessary variables are set in environment or standard .env file.")

# Load standard .env (from project root, .env.local takes precedence)
dotenv_path_standard = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path_standard):
    logger.info("Loading environment variables from standard .env file in project root.")
    load_dotenv(dotenv_path=dotenv_path_standard)
# --- End Environment Variables ---


# --- Rest of the imports ---
# Now these should work because the parent dir is in sys.path
from processing.coordinator import DueDiligenceCoordinator
from reporting.report_generator import ReportGenerator
# Verify Sample_Data import is possible (optional check)
try:
    from Sample_Data.vector_store.storage_manager import StorageManager # Test import
    logger.debug("Successfully tested import from Sample_Data.")
except ModuleNotFoundError:
    logger.error("FATAL: Still cannot import from Sample_Data even after adding parent directory to sys.path. Check directory structure.")
    sys.exit(1)
except ImportError as ie:
     logger.error(f"ImportError when testing Sample_Data import: {ie}. Check dependencies within Sample_Data.")
     sys.exit(1)


# --- Class Definition (No changes needed inside the class itself) ---
class CryptoFundDueDiligence:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        # Coordinator now relies on Sample_Data being importable
        self.coordinator = DueDiligenceCoordinator()
        template_config_path = self.config.get("report_templates")
        absolute_template_path = None
        if template_config_path:
            if not os.path.isabs(template_config_path):
                absolute_template_path = os.path.join(project_root, template_config_path)
            else:
                absolute_template_path = template_config_path
            logger.info(f"Using report template path: {absolute_template_path}")
        else:
             logger.info("No report template specified in config, ReportGenerator will use its default.")

        self.report_generator = ReportGenerator(template_path=absolute_template_path)
        logger.info("CryptoFundDueDiligence system initialized.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        default_config = {
            "output_dir": "reports",
            "report_templates": None,
            "data_sources": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment", "MarketMetrics", "OnChainAnalytics"],
            "analysis_depth": "comprehensive",
            "validation_threshold": 0.7
        }
        config_to_load = os.path.join(project_root, config_path)

        if os.path.exists(config_to_load):
            try:
                with open(config_to_load, "r") as f:
                    config = json.load(f)
                for key, value in default_config.items():
                    config.setdefault(key, value)
                logger.info(f"Loaded configuration from {config_to_load}")
                return config
            except Exception as e:
                logger.error(f"Error loading config from {config_to_load}: {e}. Using default config.")
                return default_config
        else:
            logger.warning(f"Config file not found at {config_to_load}. Using default config.")
            return default_config

    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        logger.info(f"Analyzing document: {document_id}")
        try:
            analysis_results = self.coordinator.process_document(document_id)
            return analysis_results
        except Exception as e:
            logger.error(f"Error analyzing document {document_id}: {e}", exc_info=True)
            return {"error": f"Error analyzing document: {str(e)}"}

    def generate_report(self, document_id: str, output_format: str = "pptx") -> Dict[str, Any]:
        logger.info(f"Generating {output_format} report for document: {document_id}")

        if output_format != "pptx":
            logger.warning(f"Output format '{output_format}' not fully supported. Generating PPTX.")

        try:
            analysis_results = self.analyze_document(document_id)

            if "error" in analysis_results:
                logger.error(f"Analysis failed for {document_id}: {analysis_results['error']}")
                return {"error": f"Analysis failed: {analysis_results['error']}"}

            output_dir = os.path.join(project_root, self.config.get("output_dir", "reports"))
            os.makedirs(output_dir, exist_ok=True)

            # Get template path from config
            template_path = None
            if "report_templates" in self.config:
                template_dir = os.path.join(project_root, self.config.get("report_templates"))
                template_path = os.path.join(template_dir, "base_template.pptx")
                if not os.path.exists(template_path):
                    logger.warning(f"Template not found at {template_path}")
                    template_path = None

            fund_name = analysis_results.get("fund_info", {}).get("fund_name", "UnknownFund")
            safe_fund_name = "".join(c if c.isalnum() else "_" for c in fund_name)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_fund_name}_DueDiligence_{timestamp}.pptx"
            filepath = os.path.join(output_dir, filename)

            report_generator = ReportGenerator(template_path=template_path)
            report_path = report_generator.generate_report(analysis_results, output_path=filepath)

            logger.info(f"Report generation complete. Saved to: {report_path}")
            return {
                "success": True,
                "report_path": report_path,
            }
        except Exception as e:
            logger.error(f"Error generating report for document {document_id}: {e}", exc_info=True)
            return {"error": f"Error generating report: {str(e)}"}

# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Crypto Fund Due Diligence System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file relative to project root")
    parser.add_argument("--document", type=str, required=True, help="Document ID to analyze and report on")
    args = parser.parse_args()

    # Check for API key after loading attempts
    if not os.getenv("GROQ_API_KEY"):
        logger.error("FATAL: GROQ_API_KEY environment variable is not set. Please set it in .env.local, .env, or your system environment.")
        print("\nFATAL ERROR: GROQ_API_KEY is not set. Cannot proceed.")
        sys.exit(1)

    system = None
    try:
        logger.info("Initializing CryptoFundDueDiligence system...")
        system = CryptoFundDueDiligence(config_path=args.config)
        logger.info("System initialized. Generating report...")
        result = system.generate_report(args.document)

        if "error" in result:
            print(f"\nError: {result['error']}")
            logger.error(f"Report generation failed: {result['error']}")
        else:
            print(f"\nReport successfully generated: {result['report_path']}")
            logger.info(f"Report generation successful: {result['report_path']}")

    except Exception as e:
         logger.critical(f"An critical error occurred in main: {e}", exc_info=True)
         print(f"\nA critical error occurred: {e}")
    finally:
        if system:
            logger.info("Closing system...")
            system.close()
            logger.info("System closed.")

if __name__ == "__main__":
    logger.info("Crypto Due Diligence System Started")
    main()
    logger.info("Crypto Due Diligence System Finished")