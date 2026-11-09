# Enterprise Demand Forecasting & Inventory Optimization System
**Data Science & Machine Learning Internship — Production Decision-Support Platform**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)]()
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7.svg)](https://demand-forecasting-inventory-tv55.onrender.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **Live Interactive Web Application**: [https://demand-forecasting-inventory-tv55.onrender.com/](https://demand-forecasting-inventory-tv55.onrender.com/)  
> 📖 **Interactive Swagger API Documentation**: [https://demand-forecasting-inventory-tv55.onrender.com/docs](https://demand-forecasting-inventory-tv55.onrender.com/docs)

---

## Executive Overview
Retail and supply-chain operations must constantly balance product availability against the holding cost of excess inventory. Inaccurate demand estimates cause stockouts, lost sales, emergency freight costs, and inventory write-downs.

This repository implements an end-to-end, leak-free machine learning workflow that:
1. **Forecasts Future Demand**: Predicts customer unit demand across multi-store retail networks using historical sales, calendar seasonality, and promotional features.
2. **Benchmarks Against Baselines**: Evaluates models against Naive and Seasonal-Naive baselines on strict chronological splits.
3. **Optimizes Replenishment Decisions**: Computes Lead-Time Demand, Safety Stock (90%–99% service levels), Reorder Points (ROP), and Order Quantities respecting supplier Minimum Order Quantities (MOQ) and Case-Pack sizes.
4. **Delivers Operational Visibility**: Serves a 4-page interactive decision-support dashboard and high-performance RESTful FastAPI service.

---

## 4-Week Development Lifecycle & Deliverables

| Lifecycle Phase | Focus Area | Technical Implementations | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Week 1** | Data Quality & Exploratory Analysis | Deduplication, negative sales normalization, complete Cartesian grid, stockout censoring analysis. | `src/data_quality.py`, `reports/DATA_DICTIONARY.md`, `reports/DATA_QUALITY_REPORT.md`, `notebooks/01_data_collection_and_eda.py` |
| **Week 2** | Feature Engineering & Baselines | Leak-free lags (1-28d), rolling means/std, cyclical calendar, chronological train/val/test splits, Naive & Seasonal-Naive benchmarks. | `src/features.py`, `src/baselines.py`, `reports/BASELINE_EVALUATION.md`, `notebooks/02_feature_engineering_and_baselines.py` |
| **Week 3** | ML Model Training & Replenishment Logic | Ridge Regression, Random Forest, HistGradientBoosting; Safety stock, ROP, MOQ, and pack size rounding. | `src/models.py`, `src/inventory.py`, `reports/MODEL_BENCHMARK_REPORT.md`, `notebooks/03_model_training_and_replenishment.py` |
| **Week 4** | Dashboard, REST API & Reporting | Interactive 4-page dashboard, FastAPI REST endpoints, Pydantic validation, 15 unit tests, final report & presentation. | `src/api.py`, `dashboard/streamlit_app.py`, `tests/`, `reports/FINAL_ANALYTICAL_REPORT.md`, `reports/PRESENTATION.md` |

---

## Repository Structure

```
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                              # Immutable source data
│   │   ├── sales_transactions.csv
│   │   ├── product_master.csv
│   │   ├── inventory_levels.csv
│   │   └── calendar_events.csv
│   ├── interim/                          # Standardized cleaned analytical data
│   │   └── cleaned_demand_daily.csv
│   └── processed/                        # Feature-engineered sets and outputs
│       ├── forecasting_features.csv
│       ├── train_set.csv
│       ├── val_set.csv
│       ├── test_set.csv
│       ├── forecast_output.csv
│       └── inventory_replenishment_recommendations.csv
├── notebooks/                            # Reproducible pipeline stages
│   ├── 01_data_collection_and_eda.py
│   ├── 02_feature_engineering_and_baselines.py
│   ├── 03_model_training_and_replenishment.py
│   └── 04_dashboard_and_api_demo.py
├── src/                                  # Core production source modules
│   ├── __init__.py
│   ├── config.py                         # System paths, constants & service levels
│   ├── data_generator.py                 # Realistic retail simulation with stockouts
│   ├── data_quality.py                   # Data cleaning, deduplication & grid completion
│   ├── features.py                       # Leak-free lag and rolling feature pipeline
│   ├── baselines.py                      # Naive and Seasonal-Naive models + metrics
│   ├── models.py                         # ML model training, tuning & evaluation
│   ├── inventory.py                      # Safety stock, ROP, MOQ & order optimization
│   └── api.py                            # FastAPI application & embedded dashboard
├── dashboard/
│   └── streamlit_app.py                  # Alternative Streamlit dashboard prototype
├── reports/
│   ├── figures/                          # Generated analytical visualizations
│   │   ├── sales_trend_seasonality.png
│   │   ├── dow_seasonality_patterns.png
│   │   ├── model_comparison_metrics.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_distribution.png
│   │   └── inventory_rop_simulation.png
│   ├── DATA_DICTIONARY.md                # Attribute definitions and physical units
│   ├── DATA_QUALITY_REPORT.md            # Data audit findings and censoring notes
│   ├── BASELINE_EVALUATION.md            # Week 2 benchmark results
│   ├── MODEL_BENCHMARK_REPORT.md         # Week 3 model comparison report
│   ├── FINAL_ANALYTICAL_REPORT.md        # Comprehensive technical report
│   └── PRESENTATION.md                   # Final presentation slides & speaker notes
└── tests/                                # Automated test suite (pytest)
    ├── __init__.py
    ├── test_data_quality.py
    ├── test_features.py
    ├── test_inventory.py
    └── test_api.py
```

---

## Out-of-Sample Model Benchmarks (105-Day Test Window)

All models were evaluated on the final 105-day out-of-sample holdout test period:

| Model Architecture | MAE (Units) | RMSE | WAPE (%) | MAPE (%) | Forecast Bias | Bias (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline: Naive (Lag 1)** | 6.371 | 9.587 | 30.80% | 48.42% | -0.041 | -0.20% |
| **Baseline: Seasonal-Naive (Lag 7)** | 5.738 | 8.414 | 27.74% | 45.54% | -0.187 | -0.90% |
| **Baseline: Rolling 7-Day Mean** | 5.811 | 8.479 | 28.09% | 39.99% | -0.126 | -0.61% |
| **ML: Ridge Regression** | 4.203 | 5.897 | 20.32% | 37.71% | -0.249 | -1.20% |
| **ML: Random Forest** | 3.800 | 5.535 | 18.37% | 32.37% | -0.220 | -1.06% |
| **ML: HistGradientBoosting (Champion)** | **3.794** | **5.416** | **18.34%** | **33.61%** | **-0.306** | **-1.48%** |

### Benchmark Highlights:
- **34% Error Reduction**: The champion Gradient Boosting model reduced WAPE from 27.74% (Seasonal-Naive) to **18.34%**.
- **Near-Zero Bias**: Forecast bias of -0.306 units per day guarantees inventory replenishment is not starved.
- **WAPE as Primary Metric**: Standard MAPE divides by actual demand $y_t$, causing division-by-zero or extreme outliers on zero-demand days. Weighted Absolute Percentage Error ($\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$) is mathematically stable and industry standard.

---

## Inventory Replenishment Formulation

The inventory optimization module (`src/inventory.py`) converts demand forecasts into procurement decisions:

1. **Lead Time Demand ($\mu_{LTD}$)**:
   $$\mu_{LTD} = \text{Lead Time} \times \bar{D}_{forecast}$$

2. **Safety Stock ($SS$)**:
   $$SS = \left\lceil Z \times \sqrt{L \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2} \right\rceil$$
   where $Z$ is the standard normal quantile for service levels:
   - 90% $\implies Z = 1.282$
   - 95% $\implies Z = 1.645$ (Default)
   - 98% $\implies Z = 2.054$
   - 99% $\implies Z = 2.326$

3. **Reorder Point ($ROP$)**:
   $$ROP = \lceil \mu_{LTD} + SS \rceil$$

4. **Recommended Order Quantity ($Q$)**:
   $$\text{Inventory Position} (IP) = \text{On-Hand} + \text{On-Order}$$
   $$\text{Trigger} = (IP \le ROP)$$
   $$\text{Deficit} = \max(0, \text{Target Base Stock} - IP)$$
   $$Q = \left\lceil \frac{\max(\text{MOQ}, \text{Deficit})}{\text{Pack Size}} \right\rceil \times \text{Pack Size}$$

---

## Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/demand-forecasting-inventory-optimization.git
cd "demand forecasting and inventory optimization using machine learning"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pipeline Stages
```bash
# Week 1: Generate raw data & run data quality validation
python -m src.data_generator
python -m src.data_quality
python -m notebooks.01_data_collection_and_eda

# Week 2: Build leak-free features, chronological splits & baselines
python -m notebooks.02_feature_engineering_and_baselines

# Week 3: Train ML models, select champion, and optimize replenishment
python -m notebooks.03_model_training_and_replenishment

# Week 4: Verify API endpoints and dashboard
python -m notebooks.04_dashboard_and_api_demo
```

### 3. Run Automated Tests
```bash
pytest -v tests/
```
Output:
```text
======================= 15 passed in 5.94s =======================
```

### 4. Launch the Interactive Dashboard & API
Start the FastAPI server:
```bash
uvicorn src.api:app --reload --port 8000
```
- Open **Interactive Dashboard**: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) (or [http://127.0.0.1:8000/](http://127.0.0.1:8000/))
- Open **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

*(Optional) Launch Streamlit UI:*
```bash
streamlit run dashboard/streamlit_app.py
```

---

## REST API Specification

### `POST /forecast`
Accepts a SKU, store location, and horizon (1–90 days).
```json
{
  "product_id": "SKU_BEV_01",
  "store_id": "STORE_EAST",
  "horizon_days": 14
}
```
**Response (200 OK)**:
```json
{
  "product_id": "SKU_BEV_01",
  "product_name": "Sparkling Citrus Soda 2L",
  "store_id": "STORE_EAST",
  "horizon_days": 14,
  "mean_daily_forecast": 50.46,
  "total_forecast_demand": 706.42,
  "forecast_std_error": 7.94,
  "daily_forecasts": [
    {
      "date": "2026-01-01",
      "day_name": "Thursday",
      "forecast_demand": 37.8,
      "lower_bound_95": 22.24,
      "upper_bound_95": 53.36
    }
  ]
}
```

### `POST /inventory/recommendation`
Calculates safety stock, ROP, order quantities, and stockout risk tiers.
```json
{
  "product_id": "SKU_APP_05",
  "store_id": "STORE_EAST",
  "current_on_hand": 8,
  "on_order_units": 0,
  "service_level": 0.95,
  "review_period_days": 7
}
```
**Response (200 OK)**:
```json
{
  "product_id": "SKU_APP_05",
  "product_name": "Thermal Fleece Winter Jacket",
  "store_id": "STORE_EAST",
  "service_level_pct": "95%",
  "daily_forecast_demand": 10.98,
  "lead_time_days": 21.0,
  "lead_time_demand": 230.6,
  "safety_stock": 70.0,
  "reorder_point": 300.0,
  "current_on_hand": 8,
  "on_order_units": 0,
  "inventory_position": 8,
  "trigger_order": true,
  "suggested_order_qty": 370,
  "suggested_packs": 74,
  "pack_size": 5,
  "days_of_supply": 0.7,
  "stockout_risk_level": "HIGH",
  "stockout_risk_description": "Critical risk! Days of supply (0.7d) is strictly less than supplier lead time (21.0d). Stockout imminent without expedited orders."
}
```

---

## Responsible AI & Decision-Support Guidelines
1. **Decision Support, Not Autonomous Execution**: Forecasts and purchase recommendations are designed to assist human procurement planners. Planners should interpret suggestions in the context of warehouse capacity, known supplier disruptions, and upcoming promotions.
2. **Stockout Censoring Consideration**: When evaluating periods with known stockouts, observed sales underestimate true customer demand. Planners should review unconstrained estimates.
3. **Data Availability Fallbacks**: If supplier lead-time variance or stock on hand is unrecorded, the system gracefully falls back to deterministic safety stock formulas with logged warnings.

---

## License
This project is licensed under the MIT License.
