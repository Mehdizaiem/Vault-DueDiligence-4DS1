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
        self.coordinator = DueDiligenceCoordinator(self.retriever)
    
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
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving default config: {e}")
                
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
            analysis_results = self.coordinator.analyze_document(document_id)
            
            return analysis_results
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {"error": f"Error analyzing document: {str(e)}"}
    
    def generate_report(self, document_id: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Generate a due diligence report for a document.
        
        Args:
            document_id: ID of the document to analyze
            output_format: Format for the output report (markdown, json, html)
            
        Returns:
            Dictionary with report result
        """
        logger.info(f"Generating report for document: {document_id}")
        
        try:
            # First analyze the document
            analysis_results = self.analyze_document(document_id)
            
            if "error" in analysis_results:
                return analysis_results
            
            # Generate report based on analysis
            report = self.coordinator.generate_report(analysis_results, output_format)
            
            # Save report to output directory
            output_dir = self.config.get("output_dir", "reports")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fund_name = analysis_results.get("fund_info", {}).get("name", "Unknown")
            safe_fund_name = "".join(c if c.isalnum() else "_" for c in fund_name)
            
            filename = f"{safe_fund_name}_DueDiligence_{timestamp}.{output_format}"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                if output_format == "json":
                    json.dump(report, f, indent=2)
                else:
                    f.write(report)
            
            logger.info(f"Report generated and saved to: {filepath}")
            
            return {
                "success": True,
                "report_path": filepath,
                "report": report
            }
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": f"Error generating report: {str(e)}"}
    
    def close(self):
        """Close connections and clean up resources"""
        if self.storage:
            self.storage.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Crypto Fund Due Diligence System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--document", type=str, help="Document ID to analyze")
    parser.add_argument("--report", action="store_true", help="Generate a report")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "json", "html"], help="Report format")
    
    args = parser.parse_args()
    
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
                    print(json.dumps(result, indent=2))
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        system.close()

if __name__ == "__main__":
    main()