#!/usr/bin/env python

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Code.report_generation.ppt_generator import CryptoPPTGenerator
from Code.report_generation.report_config import REPORT_TOPICS

def main():
    parser = argparse.ArgumentParser(description="Generate Crypto Due Diligence PowerPoint Report")
    parser.add_argument("--symbol", type=str, required=True, help="Cryptocurrency symbol (e.g., BTC, ETH)")
    parser.add_argument("--output", type=str, help="Output filename (optional)", default=None)
    parser.add_argument("--topics", nargs="+", help="Topics to include", default=REPORT_TOPICS)
    
    args = parser.parse_args()
    
    # Create generator and generate report
    generator = CryptoPPTGenerator()
    
    try:
        output_path = generator.generate_report(args.topics, args.symbol, args.output)
        if output_path:
            print(f"Report successfully generated: {output_path}")
            print(f"Report saved in: {os.path.dirname(output_path)}")
        else:
            print("Failed to generate report")
            return 1
    finally:
        generator.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())