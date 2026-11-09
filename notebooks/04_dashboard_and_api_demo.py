"""
Notebook 04: Dashboard & API Verification Demo (Week 4).
Demonstrates end-to-end programmatic interaction with the FastAPI service:
1. System health check
2. Product catalog inquiry
3. Forward demand forecast generation
4. Dynamic replenishment calculation
5. HTML dashboard generation
"""
from src.api import (
    health_check,
    list_products,
    list_stores,
    generate_forecast,
    get_inventory_recommendation,
    serve_dashboard,
    ForecastRequest,
    ReplenishmentRequest
)

def run_api_demo():
    print("Executing Week 4: API & Dashboard Verification Demo...\n")
    
    # 1. Health check
    health = health_check()
    print("=== [GET /health] ===")
    print(health)
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
    
    # 2. Product Catalog
    products = list_products()
    stores = list_stores()
    print(f"\n=== [GET /products] ===")
    print(f"Total Monitored SKUs: {len(products)} | Stores: {stores}")
    print(f"Sample SKU: {products[0]['product_id']} - {products[0]['product_name']} (Lead Time: {products[0]['lead_time_days']} days)")
    
    # 3. Forward Demand Forecast Endpoint
    req_fc = ForecastRequest(
        product_id="SKU_BEV_01",
        store_id="STORE_EAST",
        horizon_days=14
    )
    fc_res = generate_forecast(req_fc)
    print(f"\n=== [POST /forecast] ===")
    print(f"Product: {fc_res.product_name} ({fc_res.product_id}) at {fc_res.store_id}")
    print(f"14-Day Horizon Total Demand: {fc_res.total_forecast_demand} units")
    print(f"Mean Daily Demand: {fc_res.mean_daily_forecast} units/day (Residual RMSE: {fc_res.forecast_std_error})")
    print("Sample daily points (first 3):")
    for d in fc_res.daily_forecasts[:3]:
        print(f"  {d.date} ({d.day_name}): {d.forecast_demand} units [95% CI: {d.lower_bound_95} - {d.upper_bound_95}]")
    assert fc_res.horizon_days == 14
    assert len(fc_res.daily_forecasts) == 14
    
    # 4. Inventory Replenishment Recommendation Endpoint
    # Scenario A: Critical low stock triggering reorder
    req_rep_low = ReplenishmentRequest(
        product_id="SKU_APP_05",
        store_id="STORE_EAST",
        current_on_hand=8,
        on_order_units=0,
        service_level=0.95,
        review_period_days=7
    )
    rep_low = get_inventory_recommendation(req_rep_low)
    print(f"\n=== [POST /inventory/recommendation - Low Stock Scenario] ===")
    print(f"SKU: {rep_low['product_name']} | On-Hand: {rep_low['current_on_hand']} | ROP: {rep_low['reorder_point']}")
    print(f"Order Triggered: {rep_low['trigger_order']} | Recommended Order: {rep_low['suggested_order_qty']} units ({rep_low['suggested_packs']} packs)")
    print(f"Days of Supply: {rep_low['days_of_supply']} days | Risk Tier: {rep_low['stockout_risk_level']}")
    print(f"Detail: {rep_low['stockout_risk_description']}")
    assert rep_low["trigger_order"] is True
    assert rep_low["stockout_risk_level"] == "HIGH"
    
    # Scenario B: Healthy stock level
    req_rep_healthy = ReplenishmentRequest(
        product_id="SKU_BEV_01",
        store_id="STORE_EAST",
        current_on_hand=350,
        on_order_units=0,
        service_level=0.95,
        review_period_days=7
    )
    rep_healthy = get_inventory_recommendation(req_rep_healthy)
    print(f"\n=== [POST /inventory/recommendation - Healthy Stock Scenario] ===")
    print(f"SKU: {rep_healthy['product_name']} | On-Hand: {rep_healthy['current_on_hand']} | ROP: {rep_healthy['reorder_point']}")
    print(f"Order Triggered: {rep_healthy['trigger_order']} | Risk Tier: {rep_healthy['stockout_risk_level']}")
    assert rep_healthy["trigger_order"] is False
    assert rep_healthy["stockout_risk_level"] == "LOW"

    # 5. Interactive Dashboard Route Test
    dash_html = serve_dashboard()
    print(f"\n=== [GET /dashboard] ===")
    print(f"Dashboard HTML rendered successfully! Status: {dash_html.status_code} | Size: {len(dash_html.body)} bytes")
    assert dash_html.status_code == 200
    
    print("\n[SUCCESS] All API endpoints, Pydantic schemas, and decision-support logic verified successfully!")

if __name__ == "__main__":
    run_api_demo()
