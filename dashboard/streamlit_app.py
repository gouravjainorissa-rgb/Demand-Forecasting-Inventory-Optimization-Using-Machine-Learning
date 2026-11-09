"""
Streamlit Decision-Support Dashboard Prototype (Week 4).
Simplified interface for predicting future sales and managing product stock.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RAW_DATA_DIR = ROOT / "data" / "raw"

st.set_page_config(
    page_title="Predict Future Sales & Manage Product Stock",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Predict Future Sales & Manage Product Stock")
st.markdown("Smart assistant that predicts customer demand and tells store managers when and how much stock to order.")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Sales Overview",
        "2. Predict Product Sales",
        "3. Stock Alerts & Order Advice",
        "4. How Well the AI Model Works"
    ]
)

@st.cache_data
def load_data():
    forecast_df = pd.read_csv(PROCESSED_DATA_DIR / "forecast_output.csv")
    recom_df = pd.read_csv(PROCESSED_DATA_DIR / "inventory_replenishment_recommendations.csv")
    catalog_df = pd.read_csv(RAW_DATA_DIR / "product_master.csv")
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    return forecast_df, recom_df, catalog_df

forecast_df, recom_df, catalog_df = load_data()

if page == "1. Sales Overview":
    st.header("Sales Overview & Accuracy Summary")
    st.info("💡 **How it works:** This system predicts customer demand using previous sales, day-of-week trends, and holiday shopping patterns so store managers never run out of popular items.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Prediction Error (%)", "18.3%", "↓ 34% more accurate than guessing")
    col2.metric("Prediction Difference (Bias)", "-0.30 items", "Balanced (Not over or under-predicting)")
    col3.metric("Product Availability Target", "95.0%", "Target: In-stock 95% of the time")
    col4.metric("Tracked Products & Stores", f"{len(catalog_df)} Products", "3 Store Locations")
    
    st.subheader("Real Customer Sales vs AI Predicted Sales (Test Period)")
    agg_ts = forecast_df.groupby("date")[["units_sold", "forecast_demand"]].sum()
    agg_ts.columns = ["Real Sales (Units Sold)", "AI Predicted Sales (Units)"]
    st.line_chart(agg_ts)

elif page == "2. Predict Product Sales":
    st.header("Predict Future Sales for Any Product")
    col1, col2 = st.columns(2)
    selected_sku = col1.selectbox("Choose Product", sorted(catalog_df["product_id"].unique()))
    selected_store = col2.selectbox("Store Location", sorted(forecast_df["store_id"].unique()))
    
    sku_slice = forecast_df[(forecast_df["product_id"] == selected_sku) & (forecast_df["store_id"] == selected_store)].sort_values("date")
    chart_data = sku_slice.set_index("date")[["units_sold", "forecast_demand"]]
    chart_data.columns = ["Real Sales", "AI Predicted Sales"]
    
    st.line_chart(chart_data)

elif page == "3. Stock Alerts & Order Advice":
    st.header("Which Products May Run Out of Stock?")
    st.info("💡 **Guide:** If current stock on hand is below the 'When to Order More' level, an immediate order is recommended to prevent empty shelves.")
    
    display_df = recom_df[[
        "product_name", "store_id", "current_on_hand", 
        "on_order_units", "reorder_point", "days_of_supply", 
        "trigger_order", "suggested_order_qty", "stockout_risk_level"
    ]].copy()
    
    display_df.columns = [
        "Product Name", "Store", "Stock on Hand", 
        "Orders on the Way", "When to Order More", "Days of Stock Left", 
        "Order Needed?", "Recommended Order Quantity", "Risk of Running Out"
    ]
    st.dataframe(display_df, use_container_width=True)

elif page == "4. How Well the AI Model Works":
    st.header("How Well Does the AI Model Work Compared to Guessing?")
    perf_data = pd.DataFrame([
        {"Method / Model": "Simple Guess (Yesterday's Sales)", "Average Prediction Error (MAE)": 6.371, "Large Mistakes Penalty (RMSE)": 9.587, "Overall Prediction Error % (WAPE)": 30.80},
        {"Method / Model": "Weekly Guess (Same Day Last Week)", "Average Prediction Error (MAE)": 5.738, "Large Mistakes Penalty (RMSE)": 8.414, "Overall Prediction Error % (WAPE)": 27.74},
        {"Method / Model": "7-Day Moving Average", "Average Prediction Error (MAE)": 5.811, "Large Mistakes Penalty (RMSE)": 8.479, "Overall Prediction Error % (WAPE)": 28.09},
        {"Method / Model": "Basic Linear Math (Ridge)", "Average Prediction Error (MAE)": 4.203, "Large Mistakes Penalty (RMSE)": 5.897, "Overall Prediction Error % (WAPE)": 20.32},
        {"Method / Model": "Random Forest Tree Model", "Average Prediction Error (MAE)": 3.800, "Large Mistakes Penalty (RMSE)": 5.535, "Overall Prediction Error % (WAPE)": 18.37},
        {"Method / Model": "Our Champion AI Model (Tree Boosting)", "Average Prediction Error (MAE)": 3.794, "Large Mistakes Penalty (RMSE)": 5.416, "Overall Prediction Error % (WAPE)": 18.34}
    ])
    st.table(perf_data)
    st.caption("📘 Lower error numbers mean better accuracy. Our Champion AI model reduced overall prediction mistakes from 27.7% to 18.3%.")
