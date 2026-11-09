# Data Quality & Integrity Report (Week 1 Deliverable)

## Executive Summary
This Data Quality Report documents the validation, hygiene, and audit results for the historical retail sales, catalog, inventory, and calendar data spanning the 2-year operational window (2024-01-01 through 2025-12-31, 731 days).

Raw source datasets are stored immutably in `data/raw/` and are never modified in place. The transformation pipeline (`src/data_quality.py`) generates the standardized analytical base table in `data/interim/cleaned_demand_daily.csv`.

---

## 1. Summary of Data Auditing & Cleaning Actions

| Check Category | Detected Issue in Raw Source | Remediation Action Applied | Status |
| :--- | :--- | :--- | :--- |
| **Duplicate Transactions** | 15 duplicate transaction entries observed due to POS transmission retries. | Deduplicated by composite key `[date, store_id, product_id]`, retaining first valid instance. | Resolved |
| **Negative Sales Values** | 5 instances of negative `units_sold` recorded from counter returns. | Separated return volume into dedicated `returns_units` attribute; floored `units_sold` at zero. | Resolved |
| **Foreign Key Consistency** | Validated 100% of product IDs against `product_master` and stores against `inventory_levels`. | No orphaned records detected; all foreign keys match catalog primary keys. | Passed |
| **Grid Completeness & Missing Dates** | Store `STORE_WEST` exhibited 2 missing transmission dates (`2024-06-15`, `2025-03-20`). | Reconstructed complete Cartesian grid (731 days $\times$ 3 stores $\times$ 10 SKUs = 21,930 rows). Gaps filled with zero sales (`units_sold = 0`). | Resolved |
| **Unit Consistency** | All sales recorded in integer piece counts; prices in USD ($). | Confirmed uniform data types across tables. | Passed |

---

## 2. True Zero Sales vs. Missing Records
A key risk in retail time series modeling is treating missing records as implicit zeros or vice-versa:
1. **True Zero Sales**: Occurs when a store is operational, stock is available, but zero customer purchases occur (frequently seen in slow-moving items like `SKU_TOOL_10` or off-season `SKU_APP_05`). These are legitimate zero demand observations.
2. **Missing Records**: Occurs when data logging or network transmission drops.
- **Handling Protocol**: We explicitly construct a continuous date grid per store-product pair. Any non-reporting interval is verified against inventory logs; if the store was open and stock was present, sales are imputed as 0; if closed, calendar flags indicate holiday/closure status.

---

## 3. Stockout Censoring & Latent Demand Analysis

### The Censoring Problem in Retail Supply Chains
When physical on-hand inventory reaches zero (`ending_inventory == 0` or `beginning_inventory < customer demand`), observed sales are **artificially capped** at available stock.

$$\text{Observed Sales} = \min(\text{Latent Demand}, \text{Available Inventory})$$

Naively training predictive models on observed sales during stockout days causes severe **downward forecast bias**:
- The model learns that demand is zero (or very low) precisely when demand was high enough to deplete all stock!
- In subsequent periods, lower forecast demand prompts smaller replenishment orders, perpetuating a recurring stockout cycle.

### Empirical Findings in the Audited Dataset
- **Total Stockout Days**: 1,084 store-SKU days out of 21,930 total observations (**4.94% overall stockout rate**).
- **Average Observed Sales on Stockout Days**: 7.64 units/day (capped by remaining stock).
- **Average Observed Sales on Fully Stocked Days**: 18.23 units/day.
- **Preservation Strategy**: We explicitly preserve `stockout_flag` throughout the feature engineering and modeling layers. For baseline benchmarking and model evaluation, forecast error metrics are tracked both across all periods and partitioned by in-stock vs. out-of-stock periods.

---

## 4. Final Analytical Interim Dataset Metrics
- **File Location**: `data/interim/cleaned_demand_daily.csv`
- **Total Row Count**: 21,930 rows
- **Date Range**: 2024-01-01 to 2025-12-31 (731 days)
- **Active SKUs**: 10 products across 6 departments
- **Active Facilities**: 3 retail stores (`STORE_EAST`, `STORE_WEST`, `STORE_CENTRAL`)
- **Memory Footprint**: ~4.2 MB (minimal system load)
