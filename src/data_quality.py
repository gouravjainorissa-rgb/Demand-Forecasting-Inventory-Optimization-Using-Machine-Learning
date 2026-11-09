"""
Data Quality Validation and Cleaning Pipeline (Week 1).
Performs integrity checks on raw sales, inventory, product, and calendar data.
Handles duplicates, negative quantities, date gaps, grid completeness,
and documents stockout censoring (constrained sales vs true demand).
Outputs cleaned dataset to data/interim/cleaned_demand_daily.csv.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
    DATE_COL,
    PRODUCT_COL,
    STORE_COL,
    TARGET_COL
)

def run_data_quality_pipeline() -> tuple[pd.DataFrame, dict]:
    """
    Executes end-to-end data quality validation and produces cleaned interim data.
    """
    raw_sales_path = RAW_DATA_DIR / "sales_transactions.csv"
    raw_inv_path = RAW_DATA_DIR / "inventory_levels.csv"
    raw_prod_path = RAW_DATA_DIR / "product_master.csv"
    raw_cal_path = RAW_DATA_DIR / "calendar_events.csv"
    
    # Verify raw file presence
    for p in [raw_sales_path, raw_inv_path, raw_prod_path, raw_cal_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing raw data file: {p}")
            
    sales_df = pd.read_csv(raw_sales_path)
    inv_df = pd.read_csv(raw_inv_path)
    prod_df = pd.read_csv(raw_prod_path)
    cal_df = pd.read_csv(raw_cal_path)
    
    dq_report = {
        "timestamp": datetime.now().isoformat(),
        "initial_sales_rows": len(sales_df),
        "initial_inventory_rows": len(inv_df),
        "product_count": len(prod_df),
        "calendar_days": len(cal_df),
        "issues_detected": {},
        "cleaning_actions": {}
    }
    
    # 1. Deduplication check
    dups_count = sales_df.duplicated(subset=[DATE_COL, STORE_COL, PRODUCT_COL]).sum()
    dq_report["issues_detected"]["duplicate_sales_records"] = int(dups_count)
    if dups_count > 0:
        sales_df = sales_df.drop_duplicates(subset=[DATE_COL, STORE_COL, PRODUCT_COL], keep="first")
        dq_report["cleaning_actions"]["deduplication"] = f"Removed {dups_count} duplicate rows."
        
    # 2. Check and handle negative quantities & returns
    neg_units = (sales_df[TARGET_COL] < 0).sum()
    dq_report["issues_detected"]["negative_sales_records"] = int(neg_units)
    if neg_units > 0:
        # Move negative sales into returns_units where appropriate and floor units_sold to 0
        mask_neg = sales_df[TARGET_COL] < 0
        sales_df.loc[mask_neg, "returns_units"] = sales_df.loc[mask_neg, TARGET_COL].abs()
        sales_df.loc[mask_neg, TARGET_COL] = 0
        dq_report["cleaning_actions"]["negative_units_correction"] = (
            f"Normalized {neg_units} negative sales rows into returns_units and floored units_sold to 0."
        )

    # 3. Foreign key integrity validation
    valid_products = set(prod_df["product_id"])
    valid_stores = set(inv_df["store_id"])
    invalid_prod_count = (~sales_df[PRODUCT_COL].isin(valid_products)).sum()
    invalid_store_count = (~sales_df[STORE_COL].isin(valid_stores)).sum()
    dq_report["issues_detected"]["invalid_product_foreign_keys"] = int(invalid_prod_count)
    dq_report["issues_detected"]["invalid_store_foreign_keys"] = int(invalid_store_count)
    
    # 4. Complete Date-Product-Store Grid & Missing Periods
    # Ensure every active store-product combination has a continuous daily record
    sales_df[DATE_COL] = pd.to_datetime(sales_df[DATE_COL])
    cal_df[DATE_COL] = pd.to_datetime(cal_df[DATE_COL])
    inv_df[DATE_COL] = pd.to_datetime(inv_df[DATE_COL])
    
    min_date = sales_df[DATE_COL].min()
    max_date = sales_df[DATE_COL].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")
    
    all_stores = sales_df[STORE_COL].unique()
    all_products = sales_df[PRODUCT_COL].unique()
    
    full_grid = pd.MultiIndex.from_product(
        [full_dates, all_stores, all_products],
        names=[DATE_COL, STORE_COL, PRODUCT_COL]
    ).to_frame().reset_index(drop=True)
    
    grid_size = len(full_grid)
    actual_size = len(sales_df)
    missing_records = grid_size - actual_size
    dq_report["issues_detected"]["missing_grid_records"] = int(missing_records)
    
    # Merge sales into full grid
    merged = pd.merge(full_grid, sales_df, on=[DATE_COL, STORE_COL, PRODUCT_COL], how="left")
    
    # Impute missing values with documented rules:
    # If a date was missing in sales, fill units_sold with 0 (distinguishing true zero / non-activity)
    merged[TARGET_COL] = merged[TARGET_COL].fillna(0).astype(int)
    merged["returns_units"] = merged["returns_units"].fillna(0).astype(int)
    merged["is_promoted"] = merged["is_promoted"].fillna(0).astype(int)
    
    # Merge Product Attributes
    merged = pd.merge(merged, prod_df, on=PRODUCT_COL, how="left")
    
    # Fill prices from product master if missing
    merged["unit_price"] = merged["unit_price"].fillna(merged["retail_price"])
    merged["base_price"] = merged["base_price"].fillna(merged["retail_price"])
    merged["discount_amount"] = merged["discount_amount"].fillna(0.0)
    
    # Merge Calendar Features
    cal_cols = ["date", "day_of_week", "day_name", "is_weekend", "month", "quarter", "is_holiday", "holiday_name"]
    merged = pd.merge(merged, cal_df[cal_cols], on=DATE_COL, how="left")
    
    # Merge Daily Inventory State (for stockout analysis)
    inv_cols = ["date", "store_id", "product_id", "beginning_inventory", "ending_inventory", "stockout_flag", "on_order_units"]
    merged = pd.merge(merged, inv_df[inv_cols], on=[DATE_COL, STORE_COL, PRODUCT_COL], how="left")
    
    # Fill inventory defaults if missing
    merged["beginning_inventory"] = merged["beginning_inventory"].fillna(0).astype(int)
    merged["ending_inventory"] = merged["ending_inventory"].fillna(0).astype(int)
    merged["stockout_flag"] = merged["stockout_flag"].fillna(0).astype(int)
    merged["on_order_units"] = merged["on_order_units"].fillna(0).astype(int)
    
    # 5. Stockout Analysis & Demand Censoring Documentation
    stockout_episodes = merged[merged["stockout_flag"] == 1]
    dq_report["stockout_metrics"] = {
        "total_stockout_days": int(len(stockout_episodes)),
        "stockout_rate_pct": round(float(len(stockout_episodes) / len(merged) * 100), 2),
        "mean_sales_on_stockout_days": round(float(stockout_episodes[TARGET_COL].mean()), 2),
        "mean_sales_on_in_stock_days": round(float(merged[merged["stockout_flag"] == 0][TARGET_COL].mean()), 2),
        "censoring_note": (
            "During stockout periods (ending_inventory == 0), observed sales represent constrained sales, "
            "not unconstrained customer demand. Training models purely on observed sales without acknowledging "
            "stockouts introduces downward forecast bias. In Week 2 feature engineering, stockout flags are preserved."
        )
    }
    
    # Sort chronologically and by store/product
    merged = merged.sort_values(by=[PRODUCT_COL, STORE_COL, DATE_COL]).reset_index(drop=True)
    
    # Save cleaned interim dataset
    interim_path = INTERIM_DATA_DIR / "cleaned_demand_daily.csv"
    merged.to_csv(interim_path, index=False)
    dq_report["cleaned_interim_rows"] = len(merged)
    dq_report["saved_path"] = str(interim_path)
    
    print(f"Data Quality Pipeline Complete. Cleaned interim dataset saved to: {interim_path}")
    print(f"Total rows: {len(merged):,} across {len(all_products)} products, {len(all_stores)} stores, {len(full_dates)} days.")
    print(f"Stockout rate: {dq_report['stockout_metrics']['stockout_rate_pct']}%")
    
    return merged, dq_report

if __name__ == "__main__":
    run_data_quality_pipeline()
