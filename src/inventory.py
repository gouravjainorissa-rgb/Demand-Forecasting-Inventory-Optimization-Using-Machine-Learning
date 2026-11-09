"""
Inventory Optimization and Replenishment Decision Support Module (Week 3).
Translates machine learning demand forecasts into actionable inventory replenishment decisions:
- Lead Time Demand (LTD)
- Safety Stock (SS) across parameterized service levels (90%, 95%, 98%, 99%)
- Reorder Point (ROP)
- Recommended Order Quantity (accounting for On-Hand, On-Order, MOQ, and Pack Size)
- Stockout Risk Indicator & Days of Supply (DOS)
"""
import numpy as np
import pandas as pd
from typing import Optional, Union
from src.config import SERVICE_LEVEL_Z, DEFAULT_SERVICE_LEVEL

def compute_safety_stock(
    lead_time_days: float,
    lead_time_std_days: float,
    daily_demand_mean: float,
    daily_demand_std: float,
    service_level: float = DEFAULT_SERVICE_LEVEL
) -> float:
    """
    Calculate Safety Stock using standard supply chain inventory theory.
    Incorporates both demand variability and supplier lead-time variability:
    SS = Z * sqrt( L * sigma_D^2 + D^2 * sigma_L^2 )
    """
    z = SERVICE_LEVEL_Z.get(service_level, 1.645)
    
    # Supply chain variance over lead time:
    # Var(LTD) = L * Var(D) + D^2 * Var(L)
    var_ltd = (lead_time_days * (daily_demand_std ** 2)) + ((daily_demand_mean ** 2) * (lead_time_std_days ** 2))
    sigma_ltd = np.sqrt(max(0.0, var_ltd))
    
    safety_stock = z * sigma_ltd
    return float(np.ceil(safety_stock))

def compute_reorder_point(
    lead_time_days: float,
    daily_forecast_demand: float,
    safety_stock: float
) -> float:
    """
    Calculate Reorder Point (ROP):
    ROP = Lead Time Demand + Safety Stock
    where Lead Time Demand = lead_time_days * daily_forecast_demand
    """
    ltd = lead_time_days * daily_forecast_demand
    rop = ltd + safety_stock
    return float(np.ceil(rop))

def compute_replenishment_order(
    current_on_hand: int,
    on_order_units: int,
    reorder_point: float,
    daily_forecast_demand: float,
    review_period_days: int = 7,
    minimum_order_qty: int = 1,
    pack_size: int = 1
) -> dict:
    """
    Calculate recommended replenishment order quantity using a periodic base-stock policy.
    Enforces Minimum Order Quantity (MOQ) and integer case-pack rounding.
    """
    inventory_position = current_on_hand + on_order_units
    
    # Target maximum inventory level S = ROP + Cycle Stock (Review Period * Daily Demand)
    cycle_stock = review_period_days * daily_forecast_demand
    target_base_stock = reorder_point + cycle_stock
    
    # Check if reorder trigger is met
    trigger_order = inventory_position <= reorder_point
    
    if trigger_order:
        raw_deficit = max(0.0, target_base_stock - inventory_position)
        # Apply Minimum Order Quantity (MOQ)
        moq_adjusted = max(float(minimum_order_qty), raw_deficit)
        # Round up to nearest integer multiple of pack_size
        if pack_size > 1:
            suggested_packs = int(np.ceil(moq_adjusted / pack_size))
            recommended_qty = suggested_packs * pack_size
        else:
            suggested_packs = int(np.ceil(moq_adjusted))
            recommended_qty = suggested_packs
    else:
        suggested_packs = 0
        recommended_qty = 0
        
    return {
        "inventory_position": int(inventory_position),
        "trigger_order": bool(trigger_order),
        "target_base_stock": float(round(target_base_stock, 1)),
        "suggested_order_qty": int(recommended_qty),
        "suggested_packs": int(suggested_packs)
    }

def assess_stockout_risk(
    current_on_hand: int,
    daily_forecast_demand: float,
    lead_time_days: float,
    safety_stock: float
) -> dict:
    """
    Calculate Days of Supply (DOS) and categorize operational stockout risk.
    """
    safe_demand = max(0.05, daily_forecast_demand)
    days_of_supply = round(current_on_hand / safe_demand, 1)
    
    safety_stock_days = safety_stock / safe_demand
    
    if days_of_supply < lead_time_days:
        risk_level = "HIGH"
        risk_description = f"Critical risk! Days of supply ({days_of_supply}d) is strictly less than supplier lead time ({lead_time_days}d). Stockout imminent without expedited orders."
    elif days_of_supply < (lead_time_days + safety_stock_days):
        risk_level = "MEDIUM"
        risk_description = f"Moderate risk. Days of supply ({days_of_supply}d) encroaches on safety buffer ({safety_stock_days:.1f}d). Monitor pipeline orders."
    else:
        risk_level = "LOW"
        risk_description = f"Healthy inventory buffer. Days of supply ({days_of_supply}d) comfortably exceeds lead time ({lead_time_days}d)."
        
    return {
        "days_of_supply": days_of_supply,
        "risk_level": risk_level,
        "risk_description": risk_description
    }

def generate_inventory_recommendation(
    product_id: str,
    product_name: str,
    store_id: str,
    current_on_hand: int,
    on_order_units: int,
    daily_forecast_demand: float,
    daily_demand_std: float,
    lead_time_days: float,
    lead_time_std_days: float = 0.5,
    minimum_order_qty: int = 10,
    pack_size: int = 4,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    review_period_days: int = 7
) -> dict:
    """
    Unified end-to-end inventory replenishment recommendation for a specific product and location.
    Handles data missingness with clear defensive fallbacks and documented assumptions.
    """
    # Defensive validations and defaults
    lead_time_days = max(1.0, float(lead_time_days if lead_time_days is not None else 7.0))
    lead_time_std_days = max(0.0, float(lead_time_std_days if lead_time_std_days is not None else 0.5))
    minimum_order_qty = max(1, int(minimum_order_qty if minimum_order_qty is not None else 1))
    pack_size = max(1, int(pack_size if pack_size is not None else 1))
    daily_forecast_demand = max(0.0, float(daily_forecast_demand))
    daily_demand_std = max(0.1, float(daily_demand_std if daily_demand_std is not None else daily_forecast_demand * 0.3))
    
    # Calculations
    ltd = round(lead_time_days * daily_forecast_demand, 1)
    safety_stock = compute_safety_stock(
        lead_time_days,
        lead_time_std_days,
        daily_forecast_demand,
        daily_demand_std,
        service_level
    )
    rop = compute_reorder_point(lead_time_days, daily_forecast_demand, safety_stock)
    
    replenishment = compute_replenishment_order(
        current_on_hand=current_on_hand,
        on_order_units=on_order_units,
        reorder_point=rop,
        daily_forecast_demand=daily_forecast_demand,
        review_period_days=review_period_days,
        minimum_order_qty=minimum_order_qty,
        pack_size=pack_size
    )
    
    risk = assess_stockout_risk(
        current_on_hand=current_on_hand,
        daily_forecast_demand=daily_forecast_demand,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock
    )
    
    return {
        "product_id": product_id,
        "product_name": product_name,
        "store_id": store_id,
        "service_level": service_level,
        "service_level_pct": f"{int(service_level * 100)}%",
        "daily_forecast_demand": round(daily_forecast_demand, 2),
        "lead_time_days": lead_time_days,
        "lead_time_demand": ltd,
        "safety_stock": safety_stock,
        "reorder_point": rop,
        "current_on_hand": int(current_on_hand),
        "on_order_units": int(on_order_units),
        "inventory_position": replenishment["inventory_position"],
        "trigger_order": replenishment["trigger_order"],
        "suggested_order_qty": replenishment["suggested_order_qty"],
        "suggested_packs": replenishment["suggested_packs"],
        "minimum_order_qty": minimum_order_qty,
        "pack_size": pack_size,
        "days_of_supply": risk["days_of_supply"],
        "stockout_risk_level": risk["risk_level"],
        "stockout_risk_description": risk["risk_description"]
    }
