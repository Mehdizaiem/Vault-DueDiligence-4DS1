#!/usr/bin/env python
"""
Script to check stored forecasts in Weaviate
"""

import argparse
import base64
import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.image import imread
from Sample_Data.vector_store.forecast_storage import retrieve_latest_forecast

def format_timestamp(timestamp_str):
    """Format ISO timestamp to a more readable format"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def display_image(base64_image):
    """Display a base64 encoded image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        # Convert to image
        image = imread(io.BytesIO(image_data))
        # Display image
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        return True
    except Exception as e:
        print(f"Error displaying image: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check stored forecasts in Weaviate")
    parser.add_argument("--symbol", type=str, default="ETH_USD", help="Cryptocurrency symbol to check")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of forecasts to retrieve")
    parser.add_argument("--show-images", action="store_true", help="Display forecast plots")
    
    args = parser.parse_args()
    
    # Retrieve the forecasts
    forecasts = retrieve_latest_forecast(args.symbol, args.limit)
    
    if not forecasts:
        print(f"No forecasts found for {args.symbol}")
        return
    
    print(f"\n===== STORED FORECASTS FOR {args.symbol} =====")
    print(f"Total forecasts found: {len(forecasts)}")
    print("=" * 50)
    
    for i, forecast in enumerate(forecasts, 1):
        print(f"\nForecast #{i}:")
        print(f"Timestamp: {format_timestamp(forecast.get('forecast_timestamp'))}")
        print(f"Model: {forecast.get('model_name')}")
        print(f"Days Ahead: {forecast.get('days_ahead')}")
        print(f"Current Price: ${forecast.get('current_price', 0):.2f}")
        print(f"Final Forecast: ${forecast.get('final_forecast', 0):.2f}")
        print(f"Change %: {forecast.get('change_pct', 0):.2f}%")
        print(f"Trend: {forecast.get('trend')}")
        print(f"Probability of Increase: {forecast.get('probability_increase', 0):.1f}%")
        print(f"Average Uncertainty: ±{forecast.get('average_uncertainty', 0):.1f}%")
        print(f"Plot Path: {forecast.get('plot_path')}")
        print("\nInsight:")
        print(forecast.get('insight', 'No insight available'))
        print("-" * 50)
        
        # Display image if requested and available
        if args.show_images:
            plot_image = forecast.get('plot_image')
            if plot_image:
                print("\nDisplaying forecast plot...")
                display_image(plot_image)
            else:
                print("\nNo image data available in the database")
                print("Note: You may need to run a new forecast with the updated storage code")

if __name__ == "__main__":
    main() 