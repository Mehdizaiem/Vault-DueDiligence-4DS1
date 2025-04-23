#!/usr/bin/env python
"""
Report Generator Runner

This script demonstrates how to use the report generation system to create
PowerPoint presentations from crypto fund due diligence analysis results.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import system components
from reporting.report_generator import ReportGenerator
from processing.coordinator import DueDiligenceCoordinator
from data.retriever import DataRetriever
from Sample_Data.vector_store.storage_manager import StorageManager

def generate_report_from_document(document_id: str, output_path: Optional[str] = None) -> str:
    """
    Generate a report from a document analysis.
    
    Args:
        document_id: ID of the document to analyze
        output_path: Path where to save the report (optional)
        
    Returns:
        Path to the generated report
    """
    logger.info(f"Generating report for document: {document_id}")
    
    try:
        # Initialize storage and data components
        storage = StorageManager()
        if not storage.connect():
            logger.error("Failed to connect to storage")
            return "Error: Failed to connect to storage"
        
        retriever = DataRetriever(storage)
        coordinator = DueDiligenceCoordinator(retriever)
        
        # Process the document
        analysis_results = coordinator.analyze_document(document_id)
        
        if "error" in analysis_results:
            logger.error(f"Analysis error: {analysis_results['error']}")
            return f"Error: {analysis_results['error']}"
        
        # Generate the report using the analysis results
        report_generator = ReportGenerator()
        report_path = report_generator.generate_report(analysis_results, output_path)
        
        logger.info(f"Report generated successfully: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"Error: {str(e)}"
    finally:
        # Clean up resources
        if storage:
            storage.close()

def generate_report_from_file(results_file: str, output_path: Optional[str] = None) -> str:
    """
    Generate a report from a saved analysis results JSON file.
    
    Args:
        results_file: Path to the analysis results JSON file
        output_path: Path where to save the report (optional)
        
    Returns:
        Path to the generated report
    """
    logger.info(f"Generating report from file: {results_file}")
    
    try:
        # Load analysis results from file
        with open(results_file, 'r') as f:
            analysis_results = json.load(f)
        
        # Generate the report using the loaded results
        report_generator = ReportGenerator()
        report_path = report_generator.generate_report(analysis_results, output_path)
        
        logger.info(f"Report generated successfully: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating report from file: {e}")
        return f"Error: {str(e)}"

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Crypto Fund Due Diligence Report Generator")
    parser.add_argument("--document", type=str, help="Document ID to analyze and generate report")
    parser.add_argument("--file", type=str, help="Load analysis results from JSON file")
    parser.add_argument("--output", type=str, help="Output path for the generated report")
    
    args = parser.parse_args()
    
    if args.document:
        result = generate_report_from_document(args.document, args.output)
        print(result)
    elif args.file:
        result = generate_report_from_file(args.file, args.output)
        print(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()