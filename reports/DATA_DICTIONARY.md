# Data Dictionary: Enterprise Retail Demand & Inventory System

This data dictionary provides a formal specification of all data entities, attributes, data types, physical units, and business definitions utilized throughout the four-week predictive analytics workflow.

---

## 1. Sales Transactions Table (`data/raw/sales_transactions.csv`)
Records daily point-of-sale transactions and observed customer sales by store and product.

| Field Name | Data Type | Physical Unit | Description / Business Meaning |
| :--- | :--- | :--- | :--- |
| `date` | String (ISO-8601 `YYYY-MM-DD`) | Date | Date of the recorded transaction/sales aggregate. |
| `store_id` | String | Identifier | Unique code for the retail facility (`STORE_EAST`, `STORE_WEST`, `STORE_CENTRAL`). |
| `product_id` | String | Identifier | Stock Keeping Unit (SKU) identifier (e.g., `SKU_BEV_01`, `SKU_DAIRY_02`). |
| `units_sold` | Integer | Units (Count) | Number of units sold to customers on that date. Observed sales may be constrained by stockouts. |
| `unit_price` | Float | USD ($) | Net transacted price per unit after promotional discounts. |
| `base_price` | Float | USD ($) | Manufacturer suggested regular retail selling price. |
| `discount_amount` | Float | USD ($) | Promotional dollar markdown per unit (`base_price - unit_price`). |
| `is_promoted` | Integer (0 or 1) | Binary Indicator | `1` if an active marketing campaign or temporary price reduction was active; else `0`. |
| `returns_units` | Integer | Units (Count) | Quantity of items returned or refunded by customers on this date. |
| `unconstrained_demand_est` | Integer | Units (Count) | Ground-truth latent demand before physical inventory censoring (used in evaluation). |

---

## 2. Product Master Catalog (`data/raw/product_master.csv`)
Static dimensional attributes, supplier contract parameters, and inventory handling specifications.

| Field Name | Data Type | Physical Unit | Description / Business Meaning |
| :--- | :--- | :--- | :--- |
| `product_id` | String | Identifier | Primary Key. SKU identifier. |
| `product_name` | String | Text | Descriptive human-readable commercial name. |
| `category` | String | Categorical | Primary merchandise category (Beverages, Dairy, Electronics, Apparel, etc.). |
| `sub_category` | String | Categorical | Sub-merchandise classification. |
| `cost_price` | Float | USD ($) | Procurement unit acquisition cost paid to supplier. |
| `retail_price` | Float | USD ($) | Standard list retail price. |
| `supplier_id` | String | Identifier | Unique vendor code fulfilling replenishment orders. |
| `lead_time_days` | Integer | Days | Mean supplier order-to-delivery lead time. |
| `lead_time_std_days` | Float | Days | Standard deviation of supplier fulfillment lead time (lead-time variability). |
| `minimum_order_qty` | Integer | Units | Supplier-enforced Minimum Order Quantity (MOQ). |
| `pack_size` | Integer | Units | Case pack / shipping container size. Orders must be integer multiples of `pack_size`. |
| `shelf_life_days` | Integer | Days | Product expiration/shelf-life limit (relevant for perishables like dairy/bakery). |
| `storage_type` | Categorical | Categorical | Warehouse handling constraint (`Ambient`, `Cold`, `Bulk`). |

---

## 3. Daily Inventory Levels (`data/raw/inventory_levels.csv`)
Tracks daily warehouse/store stock balance, receipts, and supply-chain pipe.

| Field Name | Data Type | Physical Unit | Description / Business Meaning |
| :--- | :--- | :--- | :--- |
| `date` | String (`YYYY-MM-DD`) | Date | Date of the inventory snapshot. |
| `store_id` | String | Identifier | Store location identifier. |
| `product_id` | String | Identifier | Product SKU identifier. |
| `beginning_inventory` | Integer | Units | Inventory on hand at store opening, after processing morning replenishment receipts. |
| `replenishment_received`| Integer | Units | Quantity of units received from supplier/distribution center on that morning. |
| `ending_inventory` | Integer | Units | On-hand physical stock remaining at end of day (`beginning_inventory - units_sold`). |
| `on_order_units` | Integer | Units | Pipeline inventory: purchase orders dispatched but not yet received at store. |
| `stockout_flag` | Integer (0 or 1) | Binary Indicator | `1` if ending inventory reached 0 or demand exceeded stock; else `0`. |

---

## 4. Calendar Events (`data/raw/calendar_events.csv`)
Chronological features, day cycles, and promotional holiday milestones.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `date` | String (`YYYY-MM-DD`) | Calendar date. |
| `day_of_week` | Integer (0 to 6) | Day index where 0 = Monday, 6 = Sunday. |
| `day_name` | String | Full day name (e.g., 'Monday', 'Saturday'). |
| `is_weekend` | Integer (0 or 1) | `1` for Saturday/Sunday; `0` for weekdays. |
| `month` | Integer (1 to 12) | Calendar month index. |
| `quarter` | Integer (1 to 4) | Fiscal/calendar quarter index. |
| `year` | Integer | Calendar year. |
| `is_holiday` | Integer (0 or 1) | `1` for statutory or major retail holiday; `0` otherwise. |
| `holiday_name` | String | Name of the calendar event (e.g., Thanksgiving, New Year, Christmas, None). |

---

## 5. Cleaned Interim Demand Table (`data/interim/cleaned_demand_daily.csv`)
The harmonized analytical view output from Week 1 data quality pipeline:
- Deduplicated transactions.
- Zero-filled complete store $\times$ SKU $\times$ date Cartesian grid.
- Corrected returns and non-negative demand values.
- Integrated product master, inventory status, and calendar attributes.
