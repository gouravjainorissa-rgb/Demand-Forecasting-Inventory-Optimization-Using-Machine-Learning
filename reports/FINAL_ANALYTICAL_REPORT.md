# Final Analytical Report: Demand Forecasting & Inventory Optimization System

**Project Title**: Enterprise Demand Forecasting & Inventory Optimization Using Machine Learning  
**Author**: Data Science & Machine Learning Engineering Internship  
**System Status**: Fully Implemented, Chronologically Validated, Production-Ready Prototype  
**Date**: November 2026  

---

## Executive Summary

Retailers and global supply-chain operators face a fundamental trade-off: **under-replenishment** triggers stockouts, lost revenue, and customer defection, whereas **over-replenishment** ties up working capital in carrying costs, obsolescence, and markdowns. 

This project delivers an end-to-end, leak-free predictive analytics and inventory optimization pipeline that:
1. Harmonizes point-of-sale transactions, product specifications, inventory status, and calendar events across multi-store retail environments (21,930 historical records over 731 days).
2. Establishes empirical benchmarks using Naive (Lag 1) and Seasonal-Naive (Lag 7) baselines.
3. Implements regularized linear, ensemble tree, and gradient-boosted models (**HistGradientBoostingRegressor** as Champion), reducing out-of-sample **Weighted Absolute Percentage Error (WAPE)** from **27.74% to 18.34%** (a 34% relative error reduction) with near-zero structural forecast bias (-0.30 units / -1.48%).
4. Converts demand forecasts into decision-support inventory replenishment recommendations using statistical lead-time demand, safety stock across target service levels (90% to 99%), and vendor constraints (MOQ and Case-Pack rounding).
5. Deploys an interactive four-page web dashboard and high-performance FastAPI service.

---

## 1. System Architecture & Development Lifecycle

The system was developed strictly according to the Four-Week Engineering Lifecycle:

```
+-----------------------------------------------------------------------------------+
|                            WEEK 1: DATA QUALITY & EDA                             |
|  data/raw/ -> Deduplication -> Returns Normalization -> Complete Grid -> Interim  |
+-----------------------------------------------------------------------------------+
                                          |
+-----------------------------------------------------------------------------------+
|                     WEEK 2: FEATURE ENGINEERING & BASELINES                       |
|  Lags (1-28d) -> Rolling Means/Std -> Calendar/Promo -> Chronological Partitions  |
+-----------------------------------------------------------------------------------+
                                          |
+-----------------------------------------------------------------------------------+
|                  WEEK 3: MODEL TRAINING & INVENTORY OPTIMIZATION                  |
|  Ridge / RF / Champion HistGB -> Service Levels -> Safety Stock -> ROP -> Orders  |
+-----------------------------------------------------------------------------------+
                                          |
+-----------------------------------------------------------------------------------+
|                        WEEK 4: REST API, DASHBOARD & TESTS                        |
|  FastAPI Endpoints -> Interactive Web UI -> 15 Automated Unit Tests -> Reports    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Data Engineering & Leakage Prevention

### 2.1 Preserving Raw Sources
All original datasets (`sales_transactions.csv`, `inventory_levels.csv`, `product_master.csv`, `calendar_events.csv`) are immutably stored under `data/raw/`.

### 2.2 Data Hygiene & Anomaly Correction
- **Deduplication**: 15 duplicate POS transmissions were isolated and purged.
- **Negative Sales Normalization**: Return events (negative sales quantities) were converted to dedicated `returns_units` and sales were bounded at 0.
- **True Zero Sales vs. Missing Records**: A complete Cartesian grid (731 days $\times$ 3 stores $\times$ 10 SKUs = 21,930 rows) was built. Legitimate zero-demand days were separated from communication dropouts.

### 2.3 Stockout Censoring
When store inventory hits zero (`ending_inventory == 0`), observed sales are bounded by available inventory, artificially masking true customer demand. In the audited dataset, a **4.94% stockout rate** was observed. To prevent downward bias, stockout indicators are explicitly tracked, and demand uncertainty is factored into safety buffers.

### 2.4 Leakage Prevention Protocol
All feature transformations (lagged targets $y_{t-1}, y_{t-7}, y_{t-28}$, rolling averages, and rolling volatility) use strictly shifted series ($t-1$). No future-window rolling or target shuffles were permitted.

---

## 3. Forecasting Evaluation & Baseline Comparison

### 3.1 Chronological Split Plan
Records were chronologically partitioned without lookahead bias:
- **Training Set**: 2024-01-29 to 2025-06-04 (493 days, 14,790 records)
- **Validation Set**: 2025-06-05 to 2025-09-17 (105 days, 3,150 records)
- **Holdout Test Set**: 2025-09-18 to 2025-12-31 (105 days, 3,150 records)

### 3.2 Evaluation Metrics
- **MAE (Mean Absolute Error)**: Interpretable average unit error.
- **RMSE (Root Mean Squared Error)**: Penalizes severe operational misses.
- **WAPE (Weighted Absolute Percentage Error)**: $\frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$ — Primary metric because retail sales contain zero-demand days where MAPE divides by zero.
- **Forecast Bias (Mean Forecast Error)**: $\frac{1}{N} \sum (\hat{y} - y)$ — Measures systematic over- or under-prediction.

### 3.3 Out-of-Sample Benchmark Results (Holdout Test Window)

| Model Architecture | MAE (Units) | RMSE | WAPE (%) | MAPE (%) | Forecast Bias | Bias (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline: Naive (Lag 1)** | 6.371 | 9.587 | 30.80% | 48.42% | -0.041 | -0.20% |
| **Baseline: Seasonal-Naive (Lag 7)** | 5.738 | 8.414 | 27.74% | 45.54% | -0.187 | -0.90% |
| **Baseline: Rolling 7-Day Mean** | 5.811 | 8.479 | 28.09% | 39.99% | -0.126 | -0.61% |
| **ML: Ridge Regression** | 4.203 | 5.897 | 20.32% | 37.71% | -0.249 | -1.20% |
| **ML: Random Forest** | 3.800 | 5.535 | 18.37% | 32.37% | -0.220 | -1.06% |
| **ML: HistGradientBoosting (Champion)** | **3.794** | **5.416** | **18.34%** | **33.61%** | **-0.306** | **-1.48%** |

### 3.4 Findings
1. **Seasonal-Naive beat Naive Persistence**: Capturing weekly consumer rhythm (Saturday spikes in grocery/bakery) reduced WAPE from 30.80% to 27.74%.
2. **Gradient Boosting achieved superior accuracy**: **HistGradientBoosting** achieved the lowest MAE (3.794 units) and RMSE (5.416), outperforming regularized linear regression by modeling promotional lifts, annual holidays, and SKU-level interactions.
3. **Unbiased Forecasts**: The champion model exhibits an average error of only -0.306 units per day, avoiding chronic under-ordering.

---

## 4. Inventory Replenishment & Decision-Support Logic

### 4.1 Statistical Formulation
1. **Lead Time Demand ($\mu_{LTD}$)**:
   $$\mu_{LTD} = L \times \hat{d}$$
   where $L$ is vendor lead time and $\hat{d}$ is the forward daily demand forecast.

2. **Safety Stock ($SS$)**:
   $$SS = \left\lceil Z \times \sqrt{L \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2} \right\rceil$$
   Accounts for both customer demand volatility ($\sigma_D$) and vendor lead-time jitter ($\sigma_L$). Configured for 90% ($Z=1.282$), 95% ($Z=1.645$), 98% ($Z=2.054$), or 99% ($Z=2.326$) service levels.

3. **Reorder Point ($ROP$)**:
   $$ROP = \lceil \mu_{LTD} + SS \rceil$$

4. **Recommended Order Quantity ($Q$)**:
   An order is triggered when $\text{Inventory Position} (IP = \text{On-Hand} + \text{On-Order}) \le ROP$:
   $$\text{Deficit} = \max(0, \text{Target Stock} - IP)$$
   $$\text{Raw Order} = \max(\text{MOQ}, \text{Deficit})$$
   $$Q = \left\lceil \frac{\text{Raw Order}}{\text{Pack Size}} \right\rceil \times \text{Pack Size}$$

### 4.2 Days of Supply & Risk Tiers
- **HIGH RISK**: Days of Supply $< \text{Lead Time}$ (imminent stockout before supplier delivery).
- **MEDIUM RISK**: Lead Time $\le$ Days of Supply $< (\text{Lead Time} + \text{Safety Buffer})$.
- **LOW RISK**: Days of Supply $\ge (\text{Lead Time} + \text{Safety Buffer})$.

---

## 5. Decision Support & Responsible AI Use

1. **Advisory Nature of Recommendations**: Model outputs are decision-support guidelines, not automated purchasing orders. Supply planners must review supplier lead-time changes, impending vendor price increases, and warehouse capacity constraints.
2. **Stockout Awareness**: If a product suffered extended stockouts, planners should cross-reference unconstrained demand estimates rather than relying purely on censored sales history.
3. **Graceful Degradation**: If supplier parameters (MOQ, lead-time variance) are unrecorded, the engine applies conservative heuristic fallbacks with documented log notices.

---

## 6. Deliverable Artifacts Summary

- **Repository**: Fully version-controlled Git repository with incremental commits per development week.
- **REST API**: Production FastAPI app with Swagger UI (`/docs`), Pydantic validation, and health checks.
- **Dashboard**: Built-in 4-page interactive UI (Chart.js) and Streamlit script.
- **Test Suite**: 15 unit and integration tests passing with 100% success rate in pytest.
- **Figures**: Publication-grade charts under `reports/figures/`.
