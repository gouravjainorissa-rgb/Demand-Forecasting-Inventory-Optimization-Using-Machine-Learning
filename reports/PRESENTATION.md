# Presentation: Demand Forecasting & Inventory Optimization System
**Data Science & Machine Learning Internship Project Showcase**

---

### Slide 1: Title & Executive Overview
**Headline**: Machine Learning Demand Forecasting & Automated Inventory Replenishment  
**Presenter**: ML & Supply Chain Analytics Engineering Team  
**Objective**: Build a leak-free, decision-support forecasting and inventory management system that balances on-shelf product availability with minimal working-capital carrying costs.

*Speaker Notes:*
> "Good morning, everyone. In modern retail supply chains, poor demand estimates create two expensive failure modes: stockouts that cost immediate sales and erode customer loyalty, and excess inventory that drives up warehouse carrying costs and leads to steep promotional markdowns. Today, we present an end-to-end predictive analytics and inventory decision-support platform built and evaluated over four development cycles."

---

### Slide 2: Business Problem & System Scope
**The Core Challenge**:
- Balancing inventory availability (service level) against inventory holding costs.
- Demand drivers: Weekly cycles, annual seasonality, promotional price drops, and vendor lead times.
- Key Risk: Stockout Censoring — when inventory is zero, observed sales understate customer demand.

**System Objectives**:
1. Chronologically forecast unit demand across multi-store retail networks.
2. Quantify forecast value against Naive and Seasonal baselines.
3. Translate demand forecasts into actionable purchase quantities, safety stocks, and reorder points.
4. Expose forecasts and risk indicators via an interactive dashboard and RESTful API.

*Speaker Notes:*
> "Our goal wasn't just to predict a number, but to bridge the gap between machine learning predictions and physical warehouse decisions. If a model predicts 50 units, what does that mean for an inventory manager who has an 8-day lead time, a minimum order quantity of 24 units, and a 95% on-shelf availability target? Our platform solves that complete workflow."

---

### Slide 3: Week 1 & 2 — Data Pipeline & Leakage Prevention
**Data Architecture**:
- Immutable raw data preserved in `data/raw/` (731 days, 3 stores, 10 diverse SKUs = 21,930 rows).
- Data Hygiene: Deduplicated 15 transmission errors, separated returns from net demand, reconstructed complete date-store-SKU Cartesian grids.
- Stockout Flagging: 4.94% historical stockout rate isolated to prevent downward model bias.

**Leakage Prevention**:
- Grouped historical lag features: $y_{t-1}, y_{t-2}, y_{t-7}, y_{t-14}, y_{t-28}$.
- Rolling statistics shifted strictly by 1 day: no future information allowed into past windows.
- Strict chronological split:
  - **Train**: 493 operating days (14,790 records)
  - **Validation**: 105 operating days (3,150 records)
  - **Holdout Test**: 105 operating days (3,150 records)

*Speaker Notes:*
> "A frequent trap in time-series machine learning is random k-fold cross-validation, which shuffles future data into past predictions. In our architecture, we strictly enforce chronological train, validation, and holdout test windows. Every single rolling feature and lag is calculated using past information only."

---

### Slide 4: Week 3 — Model Benchmarking & Empirical Results
**Evaluation Metrics**:
- MAE (Mean Absolute Error in units)
- RMSE (Penalizes larger operational swings)
- WAPE (Weighted Absolute Percentage Error — primary metric for zero-inclusive retail sales)
- Forecast Bias (Mean Forecast Error to detect over- or under-forecasting)

**Out-of-Sample Test Window Results (105 Days)**:
| Candidate Model | MAE | RMSE | WAPE (%) | Forecast Bias |
| :--- | :--- | :--- | :--- | :--- |
| Baseline: Naive (Lag 1) | 6.37 | 9.59 | 30.80% | -0.04 |
| Baseline: Seasonal-Naive (Lag 7) | 5.74 | 8.41 | 27.74% | -0.19 |
| Baseline: Rolling 7-Day Mean | 5.81 | 8.48 | 28.09% | -0.13 |
| ML: Ridge Regression | 4.20 | 5.90 | 20.32% | -0.25 |
| ML: Random Forest | 3.80 | 5.54 | 18.37% | -0.22 |
| **ML: HistGradientBoosting (Champion)** | **3.79** | **5.42** | **18.34%** | **-0.31** |

*Speaker Notes:*
> "Notice the steady progression: Seasonal-Naive outperformed simple Naive persistence by capturing weekly shopping cycles. But our Champion Gradient Boosting model achieved an outstanding 18.34% WAPE—a 34% relative error reduction over the seasonal baseline. Furthermore, its forecast bias is virtually zero at -0.31 units, ensuring warehouse replenishment is never systematically starved."

---

### Slide 5: Week 3 & 4 — Inventory Decision-Support Engine
**Formulation**:
- **Lead-Time Demand**: $\mu_{LTD} = \text{LeadTime} \times \bar{D}$
- **Safety Stock**: $SS = \left\lceil Z \times \sqrt{L \cdot \sigma_D^2 + D^2 \cdot \sigma_L^2} \right\rceil$
  - Multipliers: 90% ($Z=1.282$), 95% ($Z=1.645$), 98% ($Z=2.054$), 99% ($Z=2.326$)
- **Reorder Point (ROP)**: $ROP = \mu_{LTD} + SS$
- **Replenishment Order**:
  - Triggered when $\text{Inventory Position} (\text{On-Hand} + \text{On-Order}) \le ROP$.
  - Enforces Minimum Order Quantity (MOQ) and Case-Pack rounding.
- **Stockout Risk Tiers**:
  - HIGH ($DOS < \text{Lead Time}$), MEDIUM, LOW.

*Speaker Notes:*
> "Rather than hand-waving safety stock, our formulas explicitly capture both customer demand variance and supplier lead-time variance. If a supplier's delivery has jitter, safety stock automatically expands to prevent stockouts."

---

### Slide 6: Week 4 — Dashboard, REST API & Usability
**Interactive Dashboard (4 Functional Modules)**:
1. **Demand Overview**: Executive KPIs, WAPE benchmarks, and historical vs predicted actuals.
2. **Product Forecasts**: 7, 14, and 28-day forward forecasts with 95% confidence intervals.
3. **Inventory Risk & ROP**: Real-time stockout risk badges, Days of Supply, and recommended order quantities.
4. **Model Performance**: Comprehensive metric charts comparing candidate models and error distributions.

**FastAPI Service**:
- `/health`: System and champion model status.
- `/products` & `/stores`: Catalog inspection.
- `/forecast`: Validated forward prediction payload with confidence bounds.
- `/inventory/recommendation`: Instant safety stock, ROP, and replenishment calculation.

*Speaker Notes:*
> "The system includes both an interactive 4-page responsive dashboard and a lightweight FastAPI service. Planners can visually drill into individual SKUs or integrate the recommendations into automated ERP systems."

---

### Slide 7: Responsible AI, Limitations & Next Steps
**Human-in-the-Loop Interpretation**:
- Recommendations are **decision-support advisories**, not autonomous PO submissions.
- Planners must account for unmodelled externalities (supplier strikes, local weather disruptions, warehouse shelf limits).
- Missing vendor data triggers conservative safety buffers rather than silent failures.

**Future Enhancements**:
- Hierarchical reconciliation across stores and distribution centers.
- Dynamic price elasticity modeling for markdown optimization.
- Multi-echelon inventory routing between central warehouses and store fronts.

*Speaker Notes:*
> "To conclude: This system demonstrates how clean data engineering, leak-free feature design, and rigorous statistical inventory theory combine to create high-impact, production-ready supply chain analytics. Thank you, and I look forward to your questions."
