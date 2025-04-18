import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import sys
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sample_Data.vector_store.weaviate_client import get_weaviate_client

# Set aesthetic styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def generate_risk_visualizations():
    """Generate risk visualizations and dashboard PDF"""
    # Connect to Weaviate
    client = get_weaviate_client()
    collection = client.collections.get("RiskProfiles")

    # Fetch all risk profiles
    print("Fetching risk profiles...")
    objects = collection.query.fetch_objects(limit=1000)

    # Extract and structure data with all risk metrics
    data = []
    for obj in objects.objects:
        props = obj.properties
        data.append({
            "fund_id": props["fund_id"],
            "timestamp": props["timestamp"],
            "overall_score": props["overall_score"],
            "regulatory": props["regulatory"],
            "market": props["market"],
            "technical": props["technical"],
            "operational": props["operational"],
            "fraud": props["fraud"],
            "alerts_count": len(props.get("alerts", []))
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(by=["fund_id", "timestamp"], inplace=True)

    # Create directory for saving plots
    output_dir = Path("risk_visualizations")
    output_dir.mkdir(exist_ok=True)
    print(f"Visualizations will be saved to {output_dir.absolute()}")

    # Get average risk by fund for all dimensions
    risk_dims = ["regulatory", "market", "technical", "operational", "fraud", "overall_score"]
    risk_avg = df.groupby("fund_id")[risk_dims].mean().sort_values("overall_score", ascending=False)
    top_n = 15
    top_risk_funds = risk_avg.head(top_n)

    # ----- GENERATE INDIVIDUAL IMAGES -----
    
    # 1. Risk Dimension Heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(top_risk_funds, annot=True, cmap="YlOrRd", fmt=".2f", 
                linewidths=.5, cbar_kws={"label": "Risk Score"})
    plt.title(f"Risk Dimensions for Top {top_n} Highest-Risk Funds", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "risk_heatmap.png", dpi=300)
    plt.close()
    print("✅ Generated risk heatmap")

    # 2. Risk Distribution Histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, dim in enumerate(risk_dims):
        if i < len(axes):
            sns.histplot(df[dim], kde=True, ax=axes[i], bins=15, 
                        color=sns.color_palette("deep")[i])
            axes[i].set_title(f"{dim.replace('_', ' ').title()} Distribution", fontweight='bold')
            axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tight_layout()
    plt.savefig(output_dir / "risk_distributions.png", dpi=300)
    plt.close()
    print("✅ Generated risk distributions")

    # 3. Generate Risk Evolution Images for each fund in top 10
    top_10_funds = risk_avg.head(10).index.tolist()
    for fund_id in top_10_funds:
        fund_data = df[df["fund_id"] == fund_id]
        
        if not fund_data.empty:
            plt.figure(figsize=(16, 8))
            for dim in risk_dims:
                if dim != "overall_score":  # Skip overall score for clarity
                    plt.plot(fund_data["timestamp"], fund_data[dim], 
                            label=dim.replace("_", " ").title(), linewidth=2.5)
            
            plt.xlabel("Date/Time", fontsize=14)
            plt.ylabel("Risk Score", fontsize=14)
            plt.title(f"Risk Dimension Evolution for {fund_id}", fontsize=18, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f"risk_evolution_{fund_id}.png", dpi=300)
            plt.close()
    
    print(f"✅ Generated risk evolution images for top {len(top_10_funds)} funds")

    # ----- GENERATE PDF DASHBOARD -----
    
    pdf_path = output_dir / "risk_dashboard.pdf"
    with PdfPages(pdf_path) as pdf:
        # Create a one-page comprehensive dashboard
        fig = plt.figure(figsize=(16, 20))
        
        # 1. Title
        plt.subplot(5, 1, 1)
        plt.axis('off')
        plt.text(0.5, 0.7, "RISK ANALYSIS DASHBOARD", 
                ha='center', va='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                ha='center', va='center', fontsize=14)
        plt.text(0.5, 0.3, f"Model: XGBoost", 
                ha='center', va='center', fontsize=14)
        
        # 2. Top Risk Funds - Mini Heatmap (top 5 only for clarity)
        plt.subplot(5, 1, 2)
        mini_heatmap = risk_avg.head(5)
        sns.heatmap(mini_heatmap, annot=True, cmap="YlOrRd", fmt=".2f", 
                   linewidths=.5, cbar_kws={"label": "Risk Score"})
        plt.title("Top 5 Highest-Risk Funds", fontsize=14, fontweight='bold')
        
        # 3. Risk Evolution of Top Fund
        plt.subplot(5, 1, 3)
        highest_risk_fund = risk_avg.index[0]
        fund_data = df[df["fund_id"] == highest_risk_fund].sort_values('timestamp')
        
        for dim in risk_dims:
            if dim != "overall_score":  # Skip overall score for clarity
                plt.plot(fund_data["timestamp"], fund_data[dim], 
                        label=dim.replace("_", " ").title(), linewidth=2)
        
        plt.title(f"Risk Evolution: {highest_risk_fund}", fontsize=14, fontweight='bold')
        plt.xlabel("Date/Time", fontsize=10)
        plt.ylabel("Risk Score", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        
        # 4. Daily Average Risk Trends
        plt.subplot(5, 1, 4)
        df['date'] = df['timestamp'].dt.date
        daily_avg = df.groupby(['date'])[risk_dims].mean().reset_index()
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])
        
        for dim in risk_dims:
            if dim != "overall_score":  # Exclude overall for clarity
                plt.plot(daily_avg['date'], daily_avg[dim], 
                        label=dim.replace("_", " ").title(), linewidth=2)
        
        plt.title("Daily Average Risk Scores", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Avg Risk Score", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        
        # 5. Alert Summary - Table of highest risk funds and their alerts
        plt.subplot(5, 1, 5)
        plt.axis('off')
        
        # Get top 5 funds with most alerts
        alert_counts = df.groupby('fund_id')['alerts_count'].mean().sort_values(ascending=False).head(5)
        alert_data = [["Fund ID", "Alerts", "Overall Risk"]]
        
        for fund_id in alert_counts.index:
            fund_risk = risk_avg.loc[fund_id, 'overall_score']
            alert_data.append([fund_id, f"{alert_counts[fund_id]:.1f}", f"{fund_risk:.2f}"])
        
        table = plt.table(cellText=alert_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        plt.title("Top Funds by Alert Activity", fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"✅ PDF Dashboard saved to {pdf_path}")
    
    # Close connection
    client.close()
    
    return pdf_path, output_dir

if __name__ == "__main__":
    pdf_path, output_dir = generate_risk_visualizations()
    print(f"\nAll visualizations generated successfully:")
    print(f"- PDF dashboard: {pdf_path}")
    print(f"- Image directory: {output_dir}")