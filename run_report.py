#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the report generator and explicitly call all needed methods.
"""

import os
import sys
from reportgenertor import ReportGenerator

def main():
    print("Starting report generation process...")
    
    # Create report generator
    generator = ReportGenerator()
    print("ReportGenerator initialized")
    
    # Connect to database
    connected = generator.connect_to_weaviate()
    if not connected:
        print("Failed to connect to Weaviate database")
        return 1
    print("Connected to Weaviate database")
    
    # Generate the report content
    try:
        print("Generating report content...")
        generator.add_to_report(f"# Crypto Due Diligence Data Analysis Report\n")
        generator.add_to_report(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Generate each section explicitly
        print("Generating executive summary...")
        generator.generate_executive_summary()
        
        print("Analyzing market data...")
        generator.analyze_market_data()
        
        print("Analyzing time series...")
        generator.analyze_time_series()
        
        print("Analyzing news sentiment...")
        generator.analyze_news_sentiment()
        
        print("Analyzing documents...")
        generator.analyze_documents()
        
        print("Analyzing on-chain data...")
        generator.analyze_onchain_data()
        
        print("Generating cross-collection insights...")
        generator.generate_cross_collection_insights()
        
        print("Assessing data quality...")
        generator.assess_data_quality()
        
        print("Generating recommendations...")
        generator.generate_recommendations()
        
        # Save the report
        print("Saving report...")
        generator.save_report()
        
        print("Report generation completed successfully")
        return 0
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    finally:
        # Close the connection
        generator.close_connection()

if __name__ == "__main__":
    # Import datetime here to avoid any initialization issues
    from datetime import datetime
    
    # Run the report generation
    sys.exit(main())