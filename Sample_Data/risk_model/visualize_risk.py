# File: visualize_risk.py

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Define where to save the plots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Save plots in a 'risk_plots' subdirectory relative to the script
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "risk_plots")
# --- End Configuration ---


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed (adjust if necessary)
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    # We need Sort for fetching
    from weaviate.classes.query import Sort
except ImportError as e:
     # Try adjusting path if running from a different directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Go up one level less
    if project_root not in sys.path:
         sys.path.append(project_root)
    # Try importing again
    try:
        from Sample_Data.vector_store.weaviate_client import get_weaviate_client
        from weaviate.classes.query import Sort
    except ImportError:
         logger.error(f"Could not import get_weaviate_client or Sort. Ensure script is run from a location where Sample_Data is accessible or adjust sys.path: {e}")
         sys.exit(1)


COLLECTION_NAME = "RiskProfiles"

def fetch_risk_profiles(client: Any, limit: int = 1000) -> pd.DataFrame:
    """
    Fetches risk profile data from Weaviate and returns it as a Pandas DataFrame.
    """
    # ... (Keep fetch_risk_profiles function exactly as before) ...
    if not client or not client.is_live():
        logger.error("Cannot fetch risk profiles: Weaviate client not connected or not live.")
        return pd.DataFrame()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        logger.info(f"Fetching up to {limit} latest risk profiles from '{COLLECTION_NAME}'...")
        response = collection.query.fetch_objects(limit=limit, sort=Sort.by_property("analysis_timestamp", ascending=False))
        if not response.objects: logger.warning(f"No objects found in '{COLLECTION_NAME}'."); return pd.DataFrame()
        data = [];
        for obj in response.objects: profile = obj.properties; profile['uuid'] = str(obj.uuid); data.append(profile)
        df = pd.DataFrame(data); logger.info(f"Successfully fetched {len(df)} risk profiles.")
        if 'analysis_timestamp' in df.columns: df['analysis_timestamp_dt'] = pd.to_datetime(df['analysis_timestamp'], errors='coerce', utc=True); df.dropna(subset=['analysis_timestamp_dt'], inplace=True)
        if 'risk_score' in df.columns: df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce'); df['risk_score'] = df['risk_score'].replace(-1.0, np.nan)
        return df
    except Exception as e: logger.error(f"Error fetching risk profiles from {COLLECTION_NAME}: {e}", exc_info=True); return pd.DataFrame()


# --- Plotting Functions (Modified to Save) ---

def plot_risk_score_distribution(df: pd.DataFrame, output_dir: str, filename_prefix: str, latest_run_timestamp: Optional[datetime] = None):
    """Plots the distribution of valid risk scores and saves it to a file."""
    if df.empty or 'risk_score' not in df.columns:
        logger.warning("No data or 'risk_score' column available for score distribution plot.")
        return

    if latest_run_timestamp: df_plot = df[df['analysis_timestamp_dt'] == latest_run_timestamp].copy()
    else: df_plot = df.copy()
    valid_scores = df_plot['risk_score'].dropna()
    if valid_scores.empty: logger.warning("No valid risk scores found for distribution plot."); return

    plt.figure(figsize=(10, 6))
    sns.histplot(valid_scores, kde=True, bins=10)
    run_label = latest_run_timestamp.strftime("%Y-%m-%d_%H-%M") if latest_run_timestamp else "all_runs"
    plt.title(f'Distribution of Calculated Risk Scores ({run_label})')
    plt.xlabel('Risk Score (0-100)'); plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5); plt.tight_layout()

    # --- Save Figure ---
    save_path = os.path.join(output_dir, f"{filename_prefix}_score_distribution.png")
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150) # Save with tight bounding box and decent resolution
        logger.info(f"Saved score distribution plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save score distribution plot: {e}")
    finally:
        plt.close() # Close the figure to free memory
    # --- End Save Figure ---

def plot_risk_category_distribution(df: pd.DataFrame, output_dir: str, filename_prefix: str, latest_run_timestamp: Optional[datetime] = None):
    """Plots the count of assets in each risk category and saves it to a file."""
    if df.empty or 'risk_category' not in df.columns:
        logger.warning("No data or 'risk_category' column available for category plot."); return

    if latest_run_timestamp: df_plot = df[df['analysis_timestamp_dt'] == latest_run_timestamp].copy()
    else: df_plot = df.copy()
    if df_plot.empty: logger.warning("No data found for the selected run for category plot."); return

    category_order = ["Very Low", "Low", "Moderate", "High", "Very High", "Undetermined", "Error"]
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_plot, x='risk_category', order=category_order, palette='viridis')
    run_label = latest_run_timestamp.strftime("%Y-%m-%d_%H-%M") if latest_run_timestamp else "all_runs"
    plt.title(f'Asset Count by Risk Category ({run_label})')
    plt.xlabel('Risk Category'); plt.ylabel('Number of Assets')
    plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', alpha=0.5); plt.tight_layout()

    # --- Save Figure ---
    save_path = os.path.join(output_dir, f"{filename_prefix}_category_distribution.png")
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved category distribution plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save category distribution plot: {e}")
    finally:
        plt.close()
    # --- End Save Figure ---

def plot_scores_by_asset(df: pd.DataFrame, output_dir: str, filename_prefix: str, latest_run_timestamp: Optional[datetime] = None):
    """Plots the risk score for each asset from the latest run and saves it to a file."""
    if df.empty or 'symbol' not in df.columns or 'risk_score' not in df.columns:
        logger.warning("Missing columns for scores by asset plot."); return

    # Filter for the latest run
    if latest_run_timestamp: df_plot = df[df['analysis_timestamp_dt'] == latest_run_timestamp].copy()
    else:
        if 'analysis_timestamp_dt' in df.columns:
             latest_ts_inferred = df['analysis_timestamp_dt'].max()
             if pd.notna(latest_ts_inferred): df_plot = df[df['analysis_timestamp_dt'] == latest_ts_inferred].copy(); latest_run_timestamp = latest_ts_inferred; logger.info(f"Inferred latest run timestamp: {latest_run_timestamp}")
             else: logger.warning("Could not infer latest timestamp, plotting all."); df_plot = df.copy()
        else: logger.warning("Timestamp column missing, plotting all."); df_plot = df.copy()

    df_plot.dropna(subset=['risk_score'], inplace=True)
    if df_plot.empty: logger.warning("No assets with valid scores found for the latest run."); return
    df_plot.sort_values('risk_score', ascending=False, inplace=True)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_plot, x='symbol', y='risk_score', palette='coolwarm')
    run_label = latest_run_timestamp.strftime("%Y-%m-%d_%H-%M") if pd.notna(latest_run_timestamp) else "all_runs"
    plt.title(f'Risk Score by Asset ({run_label})')
    plt.xlabel('Asset Symbol'); plt.ylabel('Risk Score (0-100)')
    plt.xticks(rotation=75, ha='right'); plt.grid(axis='y', alpha=0.5); plt.ylim(0, 100); plt.tight_layout()

    # --- Save Figure ---
    save_path = os.path.join(output_dir, f"{filename_prefix}_scores_by_asset.png")
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved scores by asset plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save scores by asset plot: {e}")
    finally:
        plt.close()
    # --- End Save Figure ---

def plot_risk_factor_frequency(df: pd.DataFrame, output_dir: str, filename_prefix: str, latest_run_timestamp: Optional[datetime] = None):
    """Plots the frequency of identified risk factors and saves it to a file."""
    if df.empty or 'risk_factors' not in df.columns:
        logger.warning("No 'risk_factors' column available for factor frequency plot."); return

    if latest_run_timestamp: df_plot = df[df['analysis_timestamp_dt'] == latest_run_timestamp].copy()
    else: df_plot = df.copy()

    df_plot = df_plot[df_plot['risk_factors'].apply(lambda x: isinstance(x, list))]
    if df_plot.empty: logger.info("No rows with list-type risk factors found."); return

    try: all_factors = df_plot['risk_factors'].explode().dropna()
    except AttributeError: logger.error("Pandas version might be too old for Series.explode()."); return

    if all_factors.empty:
        logger.info("No risk factors were identified in the selected run.")
        return # Don't generate an empty plot

    factor_counts = all_factors.value_counts()
    plt.figure(figsize=(10, max(6, len(factor_counts) * 0.5)))
    sns.barplot(y=factor_counts.index, x=factor_counts.values, palette='mako', orient='h')
    run_label = latest_run_timestamp.strftime("%Y-%m-%d_%H-%M") if latest_run_timestamp else "all_runs"
    plt.title(f'Frequency of Identified Risk Factors ({run_label})')
    plt.xlabel('Number of Assets'); plt.ylabel('Risk Factor')
    plt.grid(axis='x', alpha=0.5); plt.tight_layout()

    # --- Save Figure ---
    save_path = os.path.join(output_dir, f"{filename_prefix}_factor_frequency.png")
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved risk factor frequency plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save risk factor frequency plot: {e}")
    finally:
        plt.close()
    # --- End Save Figure ---

# --- Main Execution ---
if __name__ == "__main__":
    client = None
    try:
        logger.info("Connecting to Weaviate...")
        client = get_weaviate_client()
        if not client or not client.is_live():
            raise ConnectionError("Failed to connect to Weaviate.")
        logger.info("Connected.")

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Plots will be saved to: {OUTPUT_DIR}")

        # Fetch all recent profiles
        profiles_df = fetch_risk_profiles(client, limit=500)

        if not profiles_df.empty:
             # --- Determine Latest Run ---
             latest_ts = None
             latest_ts_str = "all_runs" # Default filename prefix part
             if 'analysis_timestamp_dt' in profiles_df.columns:
                  latest_ts = profiles_df['analysis_timestamp_dt'].max()
                  if pd.isna(latest_ts): latest_ts = None
                  else:
                       logger.info(f"Visualizing data for the latest run timestamp: {latest_ts}")
                       latest_ts_str = latest_ts.strftime("%Y%m%d_%H%M%S") # Use timestamp in filename
             else: logger.warning("Cannot determine latest run without 'analysis_timestamp_dt' column.")

             # --- Generate and Save Plots ---
             # Pass output directory and filename prefix
             plot_risk_score_distribution(profiles_df, OUTPUT_DIR, latest_ts_str, latest_ts)
             plot_risk_category_distribution(profiles_df, OUTPUT_DIR, latest_ts_str, latest_ts)
             plot_scores_by_asset(profiles_df, OUTPUT_DIR, latest_ts_str, latest_ts)
             plot_risk_factor_frequency(profiles_df, OUTPUT_DIR, latest_ts_str, latest_ts)

             # --- Remove plt.show() ---
             # plt.show()
             logger.info(f"Finished generating plots. Check the directory: {OUTPUT_DIR}")

        else:
            logger.info("No risk profiles found in Weaviate to visualize.")

    except ConnectionError as ce:
        logger.error(f"Connection Error: {ce}")
    except Exception as e:
        logger.error(f"An error occurred during visualization: {e}", exc_info=True)
    finally:
        if client:
            client.close()
            logger.info("Weaviate connection closed.")