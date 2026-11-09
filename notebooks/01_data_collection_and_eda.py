"""
Notebook 01: Data Collection, Quality Checks & Exploratory Data Analysis (EDA).
Week 1 Deliverable: Generates exploratory visualizations and summary demand diagnostics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from src.config import INTERIM_DATA_DIR, FIGURES_DIR

def run_eda():
    print("Running Week 1: Exploratory Data Analysis...")
    data_path = INTERIM_DATA_DIR / "cleaned_demand_daily.csv"
    if not data_path.exists():
        from src.data_quality import run_data_quality_pipeline
        run_data_quality_pipeline()
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Product Summary Metrics Table
    summary = df.groupby(["product_id", "product_name", "category"]).agg(
        total_units=("units_sold", "sum"),
        mean_daily_sales=("units_sold", "mean"),
        std_daily_sales=("units_sold", "std"),
        zero_sales_pct=("units_sold", lambda s: (s == 0).mean() * 100),
        promo_rate_pct=("is_promoted", lambda s: s.mean() * 100),
        stockout_rate_pct=("stockout_flag", lambda s: s.mean() * 100)
    ).reset_index()
    
    print("\n--- SKU Demand & Operating Profile Summary ---")
    print(summary.to_string(index=False))
    
    # Matplotlib styling for professional appearance
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 2. Time Series Sales Trends & Seasonality Visual
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # High-volume seasonal beverage
    bev_ts = df[df["product_id"] == "SKU_BEV_01"].groupby("date")["units_sold"].sum()
    axes[0].plot(bev_ts.index, bev_ts.values, color="#1f77b4", alpha=0.4, label="Daily Total Sales")
    axes[0].plot(bev_ts.index, bev_ts.rolling(14, min_periods=1).mean(), color="#0d47a1", linewidth=2.0, label="14-Day Rolling Avg")
    axes[0].set_title("SKU_BEV_01: Sparkling Soda 2L (Summer Seasonality + Promo Spikes)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Total Units Sold")
    axes[0].legend(loc="upper left")
    
    # Winter apparel (strong winter seasonal)
    app_ts = df[df["product_id"] == "SKU_APP_05"].groupby("date")["units_sold"].sum()
    axes[1].plot(app_ts.index, app_ts.values, color="#e65100", alpha=0.4, label="Daily Total Sales")
    axes[1].plot(app_ts.index, app_ts.rolling(14, min_periods=1).mean(), color="#bf360c", linewidth=2.0, label="14-Day Rolling Avg")
    axes[1].set_title("SKU_APP_05: Thermal Fleece Jacket (Annual Winter Peak)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Total Units Sold")
    axes[1].legend(loc="upper left")
    
    # Daily fresh staple: Milk (high steady volume, weekly cycle)
    dairy_ts = df[df["product_id"] == "SKU_DAIRY_02"].groupby("date")["units_sold"].sum()
    axes[2].plot(dairy_ts.index, dairy_ts.values, color="#2e7d32", alpha=0.4, label="Daily Total Sales")
    axes[2].plot(dairy_ts.index, dairy_ts.rolling(14, min_periods=1).mean(), color="#1b5e20", linewidth=2.0, label="14-Day Rolling Avg")
    axes[2].set_title("SKU_DAIRY_02: Organic Whole Milk 1 Gallon (Fast Turning Daily Staple)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Total Units Sold")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left")
    
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    trend_path = FIGURES_DIR / "sales_trend_seasonality.png"
    plt.savefig(trend_path, dpi=200)
    plt.close()
    print(f"Saved figure: {trend_path}")
    
    # 3. Day of Week Seasonality Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_means = df.groupby(["day_of_week", "category"])["units_sold"].mean().unstack()
    dow_means.index = dow_labels
    dow_means.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")
    ax.set_title("Day-of-Week Seasonality by Merchandise Category", fontsize=12, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Units Sold Per Store-Day")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    dow_path = FIGURES_DIR / "dow_seasonality_patterns.png"
    plt.savefig(dow_path, dpi=200)
    plt.close()
    print(f"Saved figure: {dow_path}")
    
    print("EDA Visualizations successfully generated.")

if __name__ == "__main__":
    run_eda()
