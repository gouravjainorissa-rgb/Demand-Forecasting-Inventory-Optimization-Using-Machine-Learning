"""
Streamlit Decision-Support Dashboard Prototype (Week 4).
Alternative visual interface providing Demand Overview, Product Forecasts,
Inventory Replenishment Risk Matrix, and Model Performance.
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
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Demand Forecasting & Inventory Optimization Dashboard")
st.markdown("Decision-support system translating machine learning demand forecasts into inventory recommendations.")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation Menu",
    ["1. Demand Overview", "2. Product/Store Forecasts", "3. Inventory Risk & Replenishment", "4. Model Performance"]
)

@st.cache_data
def load_data():
    forecast_df = pd.read_csv(PROCESSED_DATA_DIR / "forecast_output.csv")
    recom_df = pd.read_csv(PROCESSED_DATA_DIR / "inventory_replenishment_recommendations.csv")
    catalog_df = pd.read_csv(RAW_DATA_DIR / "product_master.csv")
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    return forecast_df, recom_df, catalog_df

forecast_df, recom_df, catalog_df = load_data()

if page == "1. Demand Overview":
    st.header("Executive Demand Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Champion Model WAPE", "18.3%", "-34% vs Baseline")
    col2.metric("Mean Forecast Bias", "-0.30 units", "Structurally Unbiased")
    col3.metric("Service Level", "95.0%", "Z = 1.645")
    col4.metric("Monitored SKUs", len(catalog_df), "3 Store Locations")
    
    st.subheader("Aggregated Observed vs Forecast Demand (Holdout Test Window)")
    agg_ts = forecast_df.groupby("date")[["units_sold", "forecast_demand"]].sum()
    st.line_chart(agg_ts)

elif page == "2. Product/Store Forecasts":
    st.header("Product Demand Forecast Explorer")
    col1, col2 = st.columns(2)
    selected_sku = col1.selectbox("Select Product SKU", sorted(catalog_df["product_id"].unique()))
    selected_store = col2.selectbox("Select Store Facility", sorted(forecast_df["store_id"].unique()))
    
    sku_slice = forecast_df[(forecast_df["product_id"] == selected_sku) & (forecast_df["store_id"] == selected_store)].sort_values("date")
    
    st.line_chart(sku_slice.set_index("date")[["units_sold", "forecast_demand"]])

elif page == "3. Inventory Risk & Replenishment":
    st.header("Inventory Risk & Replenishment Matrix")
    st.dataframe(
        recom_df[[
            "product_id", "product_name", "store_id", "current_on_hand", 
            "on_order_units", "reorder_point", "days_of_supply", 
            "trigger_order", "suggested_order_qty", "stockout_risk_level"
        ]],
        use_container_width=True
    )

elif page == "4. Model Performance":
    st.header("Candidate Model Evaluation Benchmarks")
    perf_data = pd.DataFrame([
        {"Model": "Baseline: Naive (Lag 1)", "MAE": 6.371, "RMSE": 9.587, "WAPE (%)": 30.80},
        {"Model": "Baseline: Seasonal-Naive (Lag 7)", "MAE": 5.738, "RMSE": 8.414, "WAPE (%)": 27.74},
        {"Model": "Baseline: Rolling 7-Day Mean", "MAE": 5.811, "RMSE": 8.479, "WAPE (%)": 28.09},
        {"Model": "ML: Ridge Regression", "MAE": 4.203, "RMSE": 5.897, "WAPE (%)": 20.32},
        {"Model": "ML: Random Forest", "MAE": 3.800, "RMSE": 5.535, "WAPE (%)": 18.37},
        {"Model": "ML: HistGradientBoosting (Champion)", "MAE": 3.794, "RMSE": 5.416, "WAPE (%)": 18.34}
    ])
    st.table(perf_data)
