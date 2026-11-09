"""
Unit tests for Inventory Optimization & Replenishment Logic.
Tests Safety Stock, Reorder Point, MOQ, Case-Pack rounding, and Stockout Risk levels.
"""
import pytest
from src.inventory import (
    compute_safety_stock,
    compute_reorder_point,
    compute_replenishment_order,
    assess_stockout_risk,
    generate_inventory_recommendation
)

def test_compute_safety_stock():
    # Deterministic lead time (std=0), L=4, sigma_D=5, 95% service level (Z=1.645)
    # SS = 1.645 * sqrt(4 * 25) = 1.645 * 10 = 16.45 -> ceil = 17
    ss = compute_safety_stock(
        lead_time_days=4.0,
        lead_time_std_days=0.0,
        daily_demand_mean=20.0,
        daily_demand_std=5.0,
        service_level=0.95
    )
    assert ss == 17.0

def test_compute_safety_stock_with_lead_time_variance():
    # L=4, std_L=1.0, mean_D=20, std_D=5, Z=1.645
    # Var = 4*25 + 400*1 = 100 + 400 = 500
    # sqrt(500) = 22.36 -> SS = 1.645 * 22.36 = 36.78 -> ceil = 37
    ss = compute_safety_stock(
        lead_time_days=4.0,
        lead_time_std_days=1.0,
        daily_demand_mean=20.0,
        daily_demand_std=5.0,
        service_level=0.95
    )
    assert ss == 37.0

def test_compute_reorder_point():
    # LTD = 5 * 10 = 50, SS = 15 -> ROP = 65
    rop = compute_reorder_point(lead_time_days=5.0, daily_forecast_demand=10.0, safety_stock=15.0)
    assert rop == 65.0

def test_replenishment_order_trigger_and_pack_rounding():
    # Current on-hand = 20, on-order = 10 -> IP = 30
    # ROP = 50 -> Trigger = True
    # Target base stock = ROP + (7 * 10) = 50 + 70 = 120
    # Deficit = 120 - 30 = 90
    # Pack size = 12, MOQ = 24
    # ceil(90 / 12) = 8 packs * 12 = 96 units
    rep = compute_replenishment_order(
        current_on_hand=20,
        on_order_units=10,
        reorder_point=50.0,
        daily_forecast_demand=10.0,
        review_period_days=7,
        minimum_order_qty=24,
        pack_size=12
    )
    assert rep["trigger_order"] is True
    assert rep["inventory_position"] == 30
    assert rep["suggested_order_qty"] == 96
    assert rep["suggested_packs"] == 8

def test_replenishment_no_trigger_when_stock_sufficient():
    # Current on-hand = 100, on-order = 0 -> IP = 100
    # ROP = 50 -> Trigger = False
    rep = compute_replenishment_order(
        current_on_hand=100,
        on_order_units=0,
        reorder_point=50.0,
        daily_forecast_demand=10.0,
        review_period_days=7
    )
    assert rep["trigger_order"] is False
    assert rep["suggested_order_qty"] == 0

def test_assess_stockout_risk_tiers():
    # Case 1: High risk (DOS < Lead time)
    risk_high = assess_stockout_risk(current_on_hand=10, daily_forecast_demand=10.0, lead_time_days=5.0, safety_stock=15.0)
    assert risk_high["risk_level"] == "HIGH"
    assert risk_high["days_of_supply"] == 1.0

    # Case 2: Low risk (DOS >= Lead time + SS days)
    risk_low = assess_stockout_risk(current_on_hand=100, daily_forecast_demand=10.0, lead_time_days=5.0, safety_stock=15.0)
    assert risk_low["risk_level"] == "LOW"
    assert risk_low["days_of_supply"] == 10.0
