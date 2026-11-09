"""
Baseline Forecasting Models and Evaluation Metrics (Week 2).
Provides Naive, Seasonal-Naive (weekly cycle), and Rolling Average benchmarks.
Implements MAE, RMSE, WAPE (zero-safe percentage error), MAPE, and Forecast Bias.
"""
import numpy as np
import pandas as pd

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Compute comprehensive demand forecasting KPIs.
    Handles zero values gracefully using WAPE as primary percentage error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip predictions to non-negative (demand cannot be negative)
    y_pred = np.maximum(0.0, y_pred)
    
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # 1. Mean Absolute Error (interpretable in units sold)
    mae = float(np.mean(abs_errors))
    
    # 2. Root Mean Squared Error (penalizes large swings)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # 3. Weighted Absolute Percentage Error (WAPE = sum(|y - y_hat|) / sum(y) * 100)
    # Industry standard for inventory/retail because it is mathematically defined when individual y_i = 0
    total_actual = float(np.sum(y_true))
    wape = float((np.sum(abs_errors) / total_actual * 100.0)) if total_actual > 0 else 0.0
    
    # 4. MAPE (Mean Absolute Percentage Error on non-zero actuals only)
    non_zero_mask = y_true > 0
    if np.any(non_zero_mask):
        mape = float(np.mean(abs_errors[non_zero_mask] / y_true[non_zero_mask]) * 100.0)
    else:
        mape = 0.0
        
    # 5. Forecast Bias (Mean Forecast Error: MFE = mean(y_pred - y_true))
    # Positive indicates systematic overforecasting; Negative indicates underforecasting
    bias = float(np.mean(errors))
    bias_pct = float((np.sum(errors) / total_actual * 100.0)) if total_actual > 0 else 0.0
    
    return {
        "model": model_name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "wape_pct": round(wape, 2),
        "mape_pct": round(mape, 2),
        "forecast_bias": round(bias, 3),
        "forecast_bias_pct": round(bias_pct, 2)
    }

class NaiveForecaster:
    """
    Naive baseline (persistence): Forecast equals the previous day's actual demand (lag 1).
    """
    def __init__(self):
        self.name = "Naive (Lag 1)"
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["lag_1"].values

class SeasonalNaiveForecaster:
    """
    Seasonal Naive baseline: Forecast equals the demand from 7 days ago (lag 7).
    Captures weekly shopping day-of-week seasonality.
    """
    def __init__(self):
        self.name = "Seasonal-Naive (Lag 7)"
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["lag_7"].values

class RollingAverageForecaster:
    """
    Rolling 7-Day Average baseline: Forecast equals the mean of the past 7 days.
    """
    def __init__(self):
        self.name = "Rolling 7-Day Mean"
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["rolling_mean_7"].values
