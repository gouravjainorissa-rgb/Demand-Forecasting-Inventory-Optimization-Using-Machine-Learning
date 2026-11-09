"""
Configuration module for Demand Forecasting & Inventory Optimization System.
Defines directory paths, hyperparameters, service-level constants, and feature definitions.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for complete reproducibility
RANDOM_SEED = 42

# Time Series & Forecasting Horizon Configuration
DATE_COL = "date"
PRODUCT_COL = "product_id"
STORE_COL = "store_id"
TARGET_COL = "units_sold"

DEFAULT_FORECAST_HORIZON_DAYS = 14
EVALUATION_HORIZONS = [7, 14, 28]

# Chronological Validation Split Config
# 730 days total (2 years):
# Train: 520 days (~17 months)
# Validation: 105 days (~3.5 months)
# Test: 105 days (~3.5 months)
TRAIN_DAYS = 520
VAL_DAYS = 105
TEST_DAYS = 105

# Feature Engineering Windows
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]

# Inventory Optimization & Replenishment Constants
# Standard normal Z-score values for Target Service Levels:
# 90% -> 1.282
# 95% -> 1.645
# 98% -> 2.054
# 99% -> 2.326
SERVICE_LEVEL_Z = {
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326
}
DEFAULT_SERVICE_LEVEL = 0.95

# Days of supply risk thresholds
STOCKOUT_RISK_HIGH_DAYS = 3.0    # Stock on hand covers < 3 days
STOCKOUT_RISK_MED_DAYS = 7.0     # Stock on hand covers < 7 days
