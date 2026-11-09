"""
Unit tests for Feature Engineering & Chronological Splitting.
Ensures zero data leakage, correct lag shifting, and chronological order.
"""
import pandas as pd
import numpy as np
import pytest
from src.features import create_feature_pipeline, split_chronological

def test_no_future_leakage_in_lags():
    # Construct a simple test dataframe
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "product_id": "SKU_TEST",
        "store_id": "STORE_TEST",
        "units_sold": np.arange(1, 41), # 1, 2, 3, ... 40
        "base_price": 10.0,
        "discount_amount": 0.0,
        "is_promoted": 0,
        "stockout_flag": 0
    })
    
    feats = create_feature_pipeline(df)
    
    # Check that lag_1 is strictly y_{t-1}
    # For day with index i (units_sold = x), lag_1 must equal x - 1
    for _, row in feats.iterrows():
        expected_lag_1 = row["units_sold"] - 1
        assert row["lag_1"] == expected_lag_1
        
        # lag_7 must be x - 7
        assert row["lag_7"] == row["units_sold"] - 7
        
        # rolling_mean_7 should equal mean of past 7 days: mean((x-7) ... (x-1))
        past_7_vals = np.arange(row["units_sold"] - 7, row["units_sold"])
        assert np.isclose(row["rolling_mean_7"], past_7_vals.mean())

def test_chronological_splits_have_no_overlap():
    dates = pd.date_range("2024-01-01", periods=250, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "product_id": "SKU_TEST",
        "store_id": "STORE_TEST",
        "units_sold": 10
    })
    
    train, val, test = split_chronological(df)
    
    train_max = train["date"].max()
    val_min = val["date"].min()
    val_max = val["date"].max()
    test_min = test["date"].min()
    
    # Strict temporal precedence
    assert train_max < val_min
    assert val_max < test_min
