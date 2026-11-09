"""
Unit tests for Data Quality Validation Pipeline.
"""
import pandas as pd
import numpy as np
import pytest
from src.data_quality import run_data_quality_pipeline
from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

def test_data_quality_pipeline_run():
    cleaned_df, dq_report = run_data_quality_pipeline()
    
    # 1. Row count integrity
    assert len(cleaned_df) == 21930
    
    # 2. No negative units sold in cleaned dataset
    assert (cleaned_df["units_sold"] < 0).sum() == 0
    
    # 3. Deduplication was executed
    assert cleaned_df.duplicated(subset=["date", "store_id", "product_id"]).sum() == 0
    
    # 4. Stockout flag is present and binary
    assert set(cleaned_df["stockout_flag"].unique()).issubset({0, 1})
    
    # 5. Report structure
    assert "issues_detected" in dq_report
    assert "stockout_metrics" in dq_report
    assert dq_report["stockout_metrics"]["total_stockout_days"] > 0
