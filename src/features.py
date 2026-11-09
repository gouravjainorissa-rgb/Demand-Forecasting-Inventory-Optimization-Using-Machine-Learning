"""
Feature Engineering and Chronological Dataset Splitting (Week 2).
Constructs leak-free lag, rolling, calendar, promotional, and inventory features.
Splits data strictly chronologically into Train, Validation, and Test sets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATE_COL,
    PRODUCT_COL,
    STORE_COL,
    TARGET_COL,
    TRAIN_DAYS,
    VAL_DAYS,
    TEST_DAYS
)

def create_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate forecasting features using only historical information available at forecast time.
    Strictly avoids future leakage.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(by=[PRODUCT_COL, STORE_COL, DATE_COL]).reset_index(drop=True)
    
    # 1. Group-level Lag Features (Target history)
    # lag_k is y_{t-k}
    group = df.groupby([PRODUCT_COL, STORE_COL])[TARGET_COL]
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = group.shift(lag)
        
    # 2. Rolling Window Statistics on Lagged Demand (strictly shifted by 1 to prevent leakage)
    # shift(1) ensures today's actual target y_t is NEVER in the rolling window
    shifted_target = group.shift(1)
    
    # 7-day rolling window
    df["rolling_mean_7"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(7, min_periods=7).mean()
    )
    df["rolling_std_7"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(7, min_periods=7).std()
    )
    df["rolling_min_7"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(7, min_periods=7).min()
    )
    df["rolling_max_7"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(7, min_periods=7).max()
    )

    # 14-day rolling window
    df["rolling_mean_14"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )
    df["rolling_std_14"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(14, min_periods=14).std()
    )

    # 28-day rolling window
    df["rolling_mean_28"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(28, min_periods=28).mean()
    )
    df["rolling_std_28"] = shifted_target.groupby([df[PRODUCT_COL], df[STORE_COL]]).transform(
        lambda s: s.rolling(28, min_periods=28).std()
    )
    
    # Demand Momentum / Trend Ratio
    df["momentum_7_28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-4)

    # 3. Calendar & Cyclical Features
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["quarter"] = df[DATE_COL].dt.quarter
    
    # Cyclical sin/cos encodings
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 4. Price & Promotional Features
    df["discount_ratio"] = df["discount_amount"] / (df["base_price"] + 1e-4)
    # Lagged promo flag (to see if promo just started or is ongoing)
    df["lag_1_promo"] = df.groupby([PRODUCT_COL, STORE_COL])["is_promoted"].shift(1).fillna(0).astype(int)
    
    # 5. Inventory & Stockout History Features
    df["lag_1_stockout"] = df.groupby([PRODUCT_COL, STORE_COL])["stockout_flag"].shift(1).fillna(0).astype(int)
    df["stockout_last_7"] = (
        df.groupby([PRODUCT_COL, STORE_COL])["stockout_flag"]
        .shift(1)
        .groupby([df[PRODUCT_COL], df[STORE_COL]])
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
        .fillna(0)
    )

    # 6. Static Product & Supplier Attributes
    # (cost_price, retail_price, lead_time_days, minimum_order_qty, pack_size are already present)
    
    # Drop warm-up rows containing NaN in lag_28 or rolling_mean_28
    initial_len = len(df)
    df = df.dropna(subset=["lag_28", "rolling_mean_28"]).reset_index(drop=True)
    dropped_warmup = initial_len - len(df)
    print(f"Dropped {dropped_warmup} warmup days (28 days per series) with incomplete historical lags.")
    
    return df

def split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into chronological Train, Validation, and Test sets.
    Preserves exact temporal order without random shuffling.
    """
    unique_dates = sorted(df[DATE_COL].unique())
    total_dates = len(unique_dates)
    
    # Split windows:
    # Test set: last 105 days
    # Val set: preceding 105 days
    # Train set: remaining earlier days
    test_cutoff_idx = total_dates - TEST_DAYS
    val_cutoff_idx = test_cutoff_idx - VAL_DAYS
    
    train_dates = unique_dates[:val_cutoff_idx]
    val_dates = unique_dates[val_cutoff_idx:test_cutoff_idx]
    test_dates = unique_dates[test_cutoff_idx:]
    
    train_df = df[df[DATE_COL].isin(train_dates)].copy().reset_index(drop=True)
    val_df = df[df[DATE_COL].isin(val_dates)].copy().reset_index(drop=True)
    test_df = df[df[DATE_COL].isin(test_dates)].copy().reset_index(drop=True)
    
    print("\n--- Chronological Split Plan ---")
    print(f"Train Window:      {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)} days, {len(train_df):,} rows)")
    print(f"Validation Window: {val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')} ({len(val_dates)} days, {len(val_df):,} rows)")
    print(f"Test Window:       {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} days, {len(test_df):,} rows)")
    
    return train_df, val_df, test_df

def process_and_save_features():
    """Execute feature pipeline and save processed sets."""
    interim_path = INTERIM_DATA_DIR / "cleaned_demand_daily.csv"
    if not interim_path.exists():
        from src.data_quality import run_data_quality_pipeline
        run_data_quality_pipeline()
        
    raw_interim = pd.read_csv(interim_path)
    feature_df = create_feature_pipeline(raw_interim)
    
    train_df, val_df, test_df = split_chronological(feature_df)
    
    # Persist processed sets
    feature_df.to_csv(PROCESSED_DATA_DIR / "forecasting_features.csv", index=False)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_set.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_set.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_set.csv", index=False)
    
    print(f"Saved processed features and train/val/test splits in: {PROCESSED_DATA_DIR}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    process_and_save_features()
