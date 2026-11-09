"""
Synthetic Retail Data Generator for Demand Forecasting & Inventory Optimization.
Generates realistic multi-store, multi-product historical sales, product catalog,
daily inventory tracking with stockout episodes, and calendar events.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import RAW_DATA_DIR, RANDOM_SEED

def generate_product_catalog() -> pd.DataFrame:
    """Generate product master catalog with physical and supplier constraints."""
    products = [
        {
            "product_id": "SKU_BEV_01",
            "product_name": "Sparkling Citrus Soda 2L",
            "category": "Beverages",
            "sub_category": "Carbonated Drinks",
            "cost_price": 1.20,
            "retail_price": 2.49,
            "supplier_id": "SUPP_BEV_CORP",
            "lead_time_days": 3,
            "lead_time_std_days": 0.5,
            "minimum_order_qty": 24,
            "pack_size": 12,
            "shelf_life_days": 180,
            "storage_type": "Ambient"
        },
        {
            "product_id": "SKU_DAIRY_02",
            "product_name": "Organic Whole Milk 1 Gallon",
            "category": "Dairy & Chilled",
            "sub_category": "Fresh Milk",
            "cost_price": 2.80,
            "retail_price": 4.59,
            "supplier_id": "SUPP_VALLEY_FARM",
            "lead_time_days": 2,
            "lead_time_std_days": 0.3,
            "minimum_order_qty": 10,
            "pack_size": 4,
            "shelf_life_days": 14,
            "storage_type": "Cold"
        },
        {
            "product_id": "SKU_ELEC_03",
            "product_name": "Wireless Noise-Cancelling Headphones",
            "category": "Electronics",
            "sub_category": "Audio",
            "cost_price": 55.00,
            "retail_price": 99.99,
            "supplier_id": "SUPP_TECH_IMPORT",
            "lead_time_days": 14,
            "lead_time_std_days": 2.0,
            "minimum_order_qty": 5,
            "pack_size": 1,
            "shelf_life_days": 720,
            "storage_type": "Ambient"
        },
        {
            "product_id": "SKU_GROC_04",
            "product_name": "Royal Himalayan Basmati Rice 5kg",
            "category": "Pantry Staples",
            "sub_category": "Grains & Rice",
            "cost_price": 8.50,
            "retail_price": 14.99,
            "supplier_id": "SUPP_GLOBAL_GRAIN",
            "lead_time_days": 7,
            "lead_time_std_days": 1.2,
            "minimum_order_qty": 20,
            "pack_size": 4,
            "shelf_life_days": 365,
            "storage_type": "Ambient"
        },
        {
            "product_id": "SKU_APP_05",
            "product_name": "Thermal Fleece Winter Jacket",
            "category": "Apparel",
            "sub_category": "Outerwear",
            "cost_price": 28.00,
            "retail_price": 59.99,
            "supplier_id": "SUPP_NORDIC_WEAR",
            "lead_time_days": 21,
            "lead_time_std_days": 3.5,
            "minimum_order_qty": 15,
            "pack_size": 5,
            "shelf_life_days": 999,
            "storage_type": "Ambient"
        },
        {
            "product_id": "SKU_SNK_06",
            "product_name": "Crispy Kettle Potato Chips 200g",
            "category": "Snacks",
            "sub_category": "Chips & Crisps",
            "cost_price": 1.10,
            "retail_price": 2.29,
            "supplier_id": "SUPP_SNACK_VENTURES",
            "lead_time_days": 4,
            "lead_time_std_days": 0.8,
            "minimum_order_qty": 36,
            "pack_size": 12,
            "shelf_life_days": 120,
            "storage_type": "Ambient"
        },
        {
            "product_id": "SKU_PC_07",
            "product_name": "Gentle Hydrating Shampoo 500ml",
            "category": "Personal Care",
            "sub_category": "Hair Care",
            "cost_price": 3.40,
            "retail_price": 6.99,
            "supplier_id": "SUPP_PURE_BEAUTY",
            "lead_time_days": 6,
            "lead_time_std_days": 1.0,
            "minimum_order_qty": 12,
            "pack_size": 6,
            "shelf_life_days": 540,
            "storage_type": "Ambient"
        },
        {
            "product_id": "SKU_BAK_08",
            "product_name": "Rustic Sourdough Artisan Loaf",
            "category": "Bakery",
            "sub_category": "Fresh Breads",
            "cost_price": 1.50,
            "retail_price": 3.99,
            "supplier_id": "SUPP_LOCAL_OVEN",
            "lead_time_days": 1,
            "lead_time_std_days": 0.2,
            "minimum_order_qty": 8,
            "pack_size": 2,
            "shelf_life_days": 4,
            "storage_type": "Ambient"
        },
        {
            "product_id": "SKU_HOME_09",
            "product_name": "Tri-Ply Stainless Steel Fry Pan 10-Inch",
            "category": "Home & Kitchen",
            "sub_category": "Cookware",
            "cost_price": 24.00,
            "retail_price": 49.99,
            "supplier_id": "SUPP_CHEF_GEAR",
            "lead_time_days": 10,
            "lead_time_std_days": 1.5,
            "minimum_order_qty": 6,
            "pack_size": 2,
            "shelf_life_days": 999,
            "storage_type": "Bulk"
        },
        {
            "product_id": "SKU_TOOL_10",
            "product_name": "18V Cordless Compact Drill Kit",
            "category": "Hardware & Tools",
            "sub_category": "Power Tools",
            "cost_price": 42.00,
            "retail_price": 79.99,
            "supplier_id": "SUPP_PRO_BUILD",
            "lead_time_days": 12,
            "lead_time_std_days": 2.0,
            "minimum_order_qty": 4,
            "pack_size": 1,
            "shelf_life_days": 999,
            "storage_type": "Bulk"
        }
    ]
    return pd.DataFrame(products)

def generate_calendar(start_date: str = "2024-01-01", days: int = 731) -> pd.DataFrame:
    """Generate calendar events including weekends, monthly cycles, and holidays."""
    date_range = pd.date_range(start=start_date, periods=days, freq="D")
    records = []
    
    holidays_dict = {
        "01-01": "New Year's Day",
        "05-27": "Memorial Day",
        "07-04": "Independence Day",
        "09-02": "Labor Day",
        "11-28": "Thanksgiving / Black Friday",
        "12-25": "Christmas Day"
    }

    for dt in date_range:
        md = dt.strftime("%m-%d")
        is_holiday = 1 if md in holidays_dict else 0
        holiday_name = holidays_dict.get(md, "None")
        
        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "day_of_week": dt.dayofweek, # 0 = Monday, 6 = Sunday
            "day_name": dt.day_name(),
            "is_weekend": 1 if dt.dayofweek in [5, 6] else 0,
            "month": dt.month,
            "quarter": dt.quarter,
            "year": dt.year,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name
        })
    return pd.DataFrame(records)

def generate_raw_sales_and_inventory(
    start_date: str = "2024-01-01",
    days: int = 731,
    seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic sales, inventory levels, catalog, and calendar.
    Incorporates trend, day-of-week seasonality, annual seasonality, promo spikes,
    inventory depletion, replenishment orders, and stockout constraints.
    """
    rng = np.random.default_rng(seed)
    products_df = generate_product_catalog()
    calendar_df = generate_calendar(start_date, days)
    
    stores = [
        {"store_id": "STORE_EAST", "traffic_factor": 1.25, "location": "Urban Metro"},
        {"store_id": "STORE_WEST", "traffic_factor": 0.95, "location": "Suburban Center"},
        {"store_id": "STORE_CENTRAL", "traffic_factor": 1.05, "location": "Regional Hub"}
    ]
    
    sales_records = []
    inventory_records = []
    
    # Base daily demand rates per SKU (mean units at base store factor 1.0)
    base_demands = {
        "SKU_BEV_01": 35.0,  # High volume beverage
        "SKU_DAIRY_02": 28.0, # High turnover daily staple
        "SKU_ELEC_03": 4.5,   # Lower volume high value
        "SKU_GROC_04": 18.0,  # Medium steady
        "SKU_APP_05": 6.0,    # Strong winter seasonal
        "SKU_SNK_06": 42.0,   # Very high promo sensitivity
        "SKU_PC_07": 9.0,     # Steady low variance
        "SKU_BAK_08": 22.0,   # Strong weekend peak
        "SKU_HOME_09": 3.0,   # Slow moving
        "SKU_TOOL_10": 2.5    # Low volume intermittent
    }

    # Store inventory state tracking
    # Dict key: (store_id, product_id) -> current on_hand
    inventory_state = {}
    # Pipeline orders tracking: list of dicts: {'arrive_day_idx': int, 'qty': int, 'key': tuple}
    pipeline_orders = []
    
    dates = calendar_df["date"].tolist()
    
    # Initialize initial inventory: roughly 15 days of base demand
    for store in stores:
        for p_id in base_demands:
            lead_time = int(products_df.loc[products_df["product_id"] == p_id, "lead_time_days"].values[0])
            init_stock = int(base_demands[p_id] * store["traffic_factor"] * (lead_time + 7))
            inventory_state[(store["store_id"], p_id)] = max(init_stock, 30)

    for day_idx, date_str in enumerate(dates):
        cal_row = calendar_df.iloc[day_idx]
        month = cal_row["month"]
        dow = cal_row["day_of_week"]
        is_wknd = cal_row["is_weekend"]
        is_holiday = cal_row["is_holiday"]
        
        # 1. Process arriving orders for today
        arrived = [o for o in pipeline_orders if o["arrive_day_idx"] == day_idx]
        pipeline_orders = [o for o in pipeline_orders if o["arrive_day_idx"] > day_idx]
        
        arrived_by_key = {}
        for arr in arrived:
            k = arr["key"]
            arrived_by_key[k] = arrived_by_key.get(k, 0) + arr["qty"]
            
        for store in stores:
            s_id = store["store_id"]
            s_factor = store["traffic_factor"]
            
            for _, p_row in products_df.iterrows():
                p_id = p_row["product_id"]
                key = (s_id, p_id)
                base_mu = base_demands[p_id] * s_factor
                
                # Start of day inventory after receiving replenishment
                received_qty = arrived_by_key.get(key, 0)
                beginning_inv = inventory_state[key] + received_qty
                
                # Demand Components:
                # 1. Long-term trend (+3% annual growth)
                trend = 1.0 + (day_idx / days) * 0.06
                
                # 2. Day-of-week seasonality
                dow_factors = {
                    0: 0.85, 1: 0.90, 2: 0.95, 3: 1.00, 4: 1.15, 5: 1.40, 6: 1.25
                }
                # Bakery and Snacks have higher weekend peaks
                if p_id in ["SKU_BAK_08", "SKU_SNK_06", "SKU_BEV_01"]:
                    dow_factors[5] = 1.65
                    dow_factors[6] = 1.45
                    
                dow_mult = dow_factors[dow]
                
                # 3. Annual Seasonality (Winter Apparel peaks Nov-Jan; Beverages peak June-August)
                annual_mult = 1.0
                if p_id == "SKU_APP_05": # Fleece jacket
                    annual_mult = 2.4 if month in [11, 12, 1, 2] else 0.4
                elif p_id in ["SKU_BEV_01", "SKU_SNK_06"]:
                    annual_mult = 1.35 if month in [6, 7, 8] else 0.90
                elif p_id in ["SKU_ELEC_03", "SKU_TOOL_10"]:
                    # Holiday shopping boost
                    annual_mult = 1.8 if month == 12 or (month == 11 and cal_row["quarter"] == 4 and cal_row["is_holiday"]) else 0.95
                
                # 4. Promotions (random periodic promotions ~10% probability, lasting 3-7 days)
                # Deterministic promo trigger based on hash
                promo_active = 1 if ((hash(f"{p_id}_{s_id}_{day_idx // 14}") % 10) == 0 and (day_idx % 14 < 4)) else 0
                promo_boost = 1.55 if promo_active else 1.0
                
                # 5. Holiday boost
                holiday_boost = 1.3 if (is_holiday and p_id in ["SKU_BEV_01", "SKU_SNK_06", "SKU_GROC_04", "SKU_ELEC_03"]) else 1.0
                
                # Unconstrained latent customer demand
                lambda_demand = base_mu * trend * dow_mult * annual_mult * promo_boost * holiday_boost
                
                # Random draw from Poisson / Negative Binomial
                raw_demand = rng.poisson(max(0.1, lambda_demand))
                
                # 6. Physical inventory constraint: Actual sales CANNOT exceed beginning inventory
                stockout_occurred = 0
                if beginning_inv <= 0:
                    units_sold = 0
                    stockout_occurred = 1
                    ending_inv = 0
                elif raw_demand >= beginning_inv:
                    units_sold = beginning_inv
                    stockout_occurred = 1
                    ending_inv = 0
                else:
                    units_sold = raw_demand
                    stockout_occurred = 0
                    ending_inv = beginning_inv - units_sold
                    
                # Update current state
                inventory_state[key] = ending_inv
                
                # Pricing
                retail_p = p_row["retail_price"]
                discount = round(retail_p * 0.18, 2) if promo_active else 0.0
                actual_price = round(retail_p - discount, 2)
                
                # Returns (rare, ~1-3% of units sold with 1-2 day lag or immediate refund)
                returns = int(rng.binomial(n=max(0, units_sold), p=0.015)) if units_sold > 0 else 0
                
                # Replenishment Order Logic for Data Generation:
                # When ending inventory falls below (Lead time demand * 1.5), place order
                lead_time_val = p_row["lead_time_days"]
                rop_approx = int(base_mu * (lead_time_val + 2))
                
                on_order_units = sum(o["qty"] for o in pipeline_orders if o["key"] == key)
                total_pos = ending_inv + on_order_units
                
                if total_pos <= rop_approx:
                    # Place order respecting MOQ and Pack Size
                    pack = p_row["pack_size"]
                    moq = p_row["minimum_order_qty"]
                    target_stock = rop_approx * 2
                    deficit = max(moq, target_stock - total_pos)
                    # Round up to multiple of pack size
                    order_qty = int(np.ceil(deficit / pack) * pack)
                    
                    # Actual lead time has minor variance
                    lead_jitter = max(1, int(round(lead_time_val + rng.normal(0, p_row["lead_time_std_days"]))))
                    pipeline_orders.append({
                        "arrive_day_idx": day_idx + lead_jitter,
                        "qty": order_qty,
                        "key": key
                    })
                
                # Record daily sales transaction row
                sales_records.append({
                    "date": date_str,
                    "store_id": s_id,
                    "product_id": p_id,
                    "units_sold": units_sold,
                    "unit_price": actual_price,
                    "base_price": retail_p,
                    "discount_amount": discount,
                    "is_promoted": promo_active,
                    "returns_units": returns,
                    "unconstrained_demand_est": raw_demand # Useful benchmark in validation
                })
                
                # Record daily inventory level row
                inventory_records.append({
                    "date": date_str,
                    "store_id": s_id,
                    "product_id": p_id,
                    "beginning_inventory": beginning_inv,
                    "replenishment_received": received_qty,
                    "ending_inventory": ending_inv,
                    "on_order_units": on_order_units,
                    "stockout_flag": stockout_occurred
                })
                
    sales_df = pd.DataFrame(sales_records)
    inventory_df = pd.DataFrame(inventory_records)
    
    # Introduce explicit realistic data quality anomalies into sales_df to test Week 1 DQ pipeline:
    # 1. Duplicate records (e.g. 15 accidental re-transmissions)
    dup_samples = sales_df.sample(n=15, random_state=seed)
    sales_df = pd.concat([sales_df, dup_samples], ignore_index=True)
    
    # 2. Standalone negative quantity return row (e.g. 5 raw return lines with units_sold < 0)
    return_rows = sales_df.sample(n=5, random_state=seed + 1).copy()
    return_rows["units_sold"] = -1 * rng.integers(1, 4, size=5)
    return_rows["returns_units"] = 0
    sales_df = pd.concat([sales_df, return_rows], ignore_index=True)
    
    # 3. Simulate a missing date record (e.g., store communication dropout for 2 isolated dates)
    drop_indices = sales_df[(sales_df["store_id"] == "STORE_WEST") & (sales_df["date"].isin(["2024-06-15", "2025-03-20"]))].index
    sales_df = sales_df.drop(index=drop_indices).reset_index(drop=True)
    
    return sales_df, inventory_df, products_df, calendar_df

def save_raw_datasets():
    """Execute generation and persist raw files immutably to data/raw/."""
    print("Generating raw synthetic enterprise retail datasets...")
    sales_df, inventory_df, products_df, calendar_df = generate_raw_sales_and_inventory()
    
    sales_path = RAW_DATA_DIR / "sales_transactions.csv"
    inv_path = RAW_DATA_DIR / "inventory_levels.csv"
    prod_path = RAW_DATA_DIR / "product_master.csv"
    cal_path = RAW_DATA_DIR / "calendar_events.csv"
    
    sales_df.to_csv(sales_path, index=False)
    inventory_df.to_csv(inv_path, index=False)
    products_df.to_csv(prod_path, index=False)
    calendar_df.to_csv(cal_path, index=False)
    
    print(f"Saved: {sales_path} ({len(sales_df):,} rows)")
    print(f"Saved: {inv_path} ({len(inventory_df):,} rows)")
    print(f"Saved: {prod_path} ({len(products_df):,} rows)")
    print(f"Saved: {cal_path} ({len(calendar_df):,} rows)")
    return sales_path, inv_path, prod_path, cal_path

if __name__ == "__main__":
    save_raw_datasets()
