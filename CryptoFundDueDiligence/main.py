#!/usr/bin/env python
"""
Crypto Fund Due Diligence System

This is the main entry point for the crypto fund due diligence system.
It orchestrates the entire pipeline from document analysis to report generation.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_due_diligence.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import system components
from processing.coordinator import DueDiligenceCoordinator
from data.retriever import DataRetriever
from Sample_Data.vector_store.storage_manager import StorageManager
from reporting.report_generator import ReportGenerator # Import ReportGenerator

class CryptoFundDueDiligence:
    """
    Main class for the Crypto Fund Due Diligence system.
    Provides a high-level interface for running the due diligence process.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the system with configuration.

        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.storage = StorageManager()
        self.retriever = DataRetriever(self.storage)
        self.coordinator = DueDiligenceCoordinator(storage_manager=self.storage, data_retriever=self.retriever)
        self.report_generator = ReportGenerator() # Initialize ReportGenerator

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "output_dir": "reports",
            "report_templates": "templates",
            "data_sources": ["CryptoDueDiligenceDocuments", "CryptoNewsSentiment", "MarketMetrics", "OnChainAnalytics"],
            "analysis_depth": "comprehensive",
            "validation_threshold": 0.7
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Update with any missing defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value

                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Save default config
            try:
                # Ensure the base directory exists if config_path includes directories
                config_dir = os.path.dirname(config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                elif not config_dir and config_path: # Handle case where config_path is just a filename
                     pass # No directory needed if it's in the current folder
                else:
                     # Fallback or handle cases where config_path might be empty or invalid
                     logger.warning(f"Invalid config path directory provided: '{config_dir}'")


                # Check again if the directory was created successfully or already existed
                if not config_dir or os.path.exists(config_dir):
                    with open(config_path, "w") as f:
                        json.dump(default_config, f, indent=2)
                        logger.info(f"Default config saved to {config_path}")
                else:
                     logger.error(f"Failed to create directory for config file: {config_dir}")


            except Exception as e:
                logger.error(f"Error saving default config to {config_path}: {e}")

            return default_config

    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        """
        Analyze a document to extract fund information.

        Args:
            document_id: ID of the document to analyze

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing document: {document_id}")

        try:
            # Initialize system components
            if not self.storage.connect():
                logger.error("Failed to connect to storage")
                return {"error": "Failed to connect to storage"}

            # Process the document
            analysis_results = self.coordinator.process_document(document_id)

            return analysis_results
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {"error": f"Error analyzing document: {str(e)}"}

    def generate_report(self, document_id: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Generate a due diligence report for a document.

        Args:
            document_id: ID of the document to analyze
            output_format: Format for the output report (markdown, json, html, pptx)

        Returns:
            Dictionary with report result
        """
        logger.info(f"Generating report for document: {document_id} in format {output_format}")

        try:
            # First analyze the document
            analysis_results = self.analyze_document(document_id)

            if "error" in analysis_results:
                logger.error(f"Analysis failed for {document_id}: {analysis_results['error']}")
                return analysis_results

            # Generate report based on analysis using ReportGenerator
            # Assuming ReportGenerator has a method like 'generate' or similar
            # Adjust method name and arguments as needed based on report_generator.py
            report_content_or_path = self.report_generator.generate_report(analysis_results)

            # Save report to output directory
            output_dir = self.config.get("output_dir", "reports")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Try to get a more specific fund name if available
            fund_name = analysis_results.get("fund_info", {}).get("fund_name", "UnknownFund")
            if not fund_name or fund_name == "UnknownFund":
                 # Fallback if fund_name is missing
                 fund_name = analysis_results.get("fund_info", {}).get("name", "UnknownFund")

            safe_fund_name = "".join(c if c.isalnum() else "_" for c in fund_name)
            if not safe_fund_name or safe_fund_name == "UnknownFund":
                safe_fund_name = f"Doc_{document_id[:8]}" # Use part of doc ID if name is truly unknown


            filename = f"{safe_fund_name}_DueDiligence_{timestamp}.{output_format}"
            filepath = os.path.join(output_dir, filename)

            # Handle saving differently based on format if needed
            # Assuming generate returns content for text formats and handles saving for pptx
            if output_format in ["markdown", "json", "html"]:
                with open(filepath, "w", encoding="utf-8") as f:
                    if output_format == "json":
                         # Assuming generate returns a dict/list for json
                        json.dump(report_content_or_path, f, indent=2)
                    else:
                        # Assuming generate returns a string for markdown/html
                        f.write(str(report_content_or_path))
                logger.info(f"Report content saved to: {filepath}")
            elif output_format == "pptx":
                 # Assuming the generate method for pptx saves the file directly and returns the path
                 filepath = report_content_or_path # The generator returned the path where it saved the file
                 logger.info(f"PowerPoint report generated and saved to: {filepath}")
            else:
                 logger.warning(f"Unsupported report format '{output_format}' for saving content directly.")
                 # Handle other formats or indicate they are saved internally by the generator
                 filepath = report_content_or_path if isinstance(report_content_or_path, str) else filepath


            return {
                "success": True,
                "report_path": filepath,
                "report": report_content_or_path if output_format != "pptx" else f"Report saved to {filepath}" # Return content or path info
            }
        except Exception as e:
            logger.exception(f"Error generating report for document {document_id}: {e}") # Use exception for full traceback
            return {"error": f"Error generating report: {str(e)}"}

    def close(self):
        """Close connections and clean up resources"""
        if self.storage:
            self.storage.close()
        # Close other resources if needed, e.g., report_generator if it holds connections
        # if hasattr(self, 'report_generator') and hasattr(self.report_generator, 'close'):
        #     self.report_generator.close()
        logger.info("CryptoFundDueDiligence system resources closed.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Crypto Fund Due Diligence System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--document", type=str, help="Document ID to analyze")
    parser.add_argument("--report", action="store_true", help="Generate a report")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "json", "html", "pptx"], help="Report format")

    args = parser.parse_args()

    # Ensure config directory exists before initializing the system if a non-default path is given
    config_dir = os.path.dirname(args.config)
    if config_dir and not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir, exist_ok=True)
            logger.info(f"Created directory for config file: {config_dir}")
        except Exception as e:
            logger.error(f"Could not create directory for config file '{args.config}': {e}. Using default.")
            args.config = "config.json" # Fallback to default if dir creation fails


    system = CryptoFundDueDiligence(config_path=args.config)

    try:
        if args.document:
            if args.report:
                result = system.generate_report(args.document, args.format)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Report generated: {result['report_path']}")
            else:
                result = system.analyze_document(args.document)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    # Ensure result is serializable before printing as JSON
                    try:
                        print(json.dumps(result, indent=2, default=str)) # Use default=str for non-serializable types like datetime
                    except TypeError as e:
                        logger.error(f"Could not serialize analysis result to JSON: {e}")
                        print("Analysis Result (non-serializable parts omitted):")
                        # Basic print as fallback
                        for key, value in result.items():
                             try:
                                 json.dumps({key: value}, default=str)
                                 print(f"  {key}: {value}")
                             except TypeError:
                                 print(f"  {key}: <Non-serializable data>")

        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        system.close()

if __name__ == "__main__":
    main()