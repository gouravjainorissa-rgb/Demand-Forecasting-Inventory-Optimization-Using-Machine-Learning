"""
Unit and integration tests for FastAPI application endpoints and schemas.
"""
import pytest
from pydantic import ValidationError
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

def test_api_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["model_loaded"] is True
    assert res["total_skus"] == 10
    assert res["total_stores"] == 3

def test_list_products_and_stores():
    products = list_products()
    assert len(products) == 10
    assert "product_id" in products[0]
    assert "lead_time_days" in products[0]
    
    stores = list_stores()
    assert set(stores) == {"STORE_EAST", "STORE_WEST", "STORE_CENTRAL"}

def test_forecast_endpoint_valid_request():
    req = ForecastRequest(
        product_id="SKU_BEV_01",
        store_id="STORE_EAST",
        horizon_days=7
    )
    res = generate_forecast(req)
    assert res.product_id == "SKU_BEV_01"
    assert res.horizon_days == 7
    assert len(res.daily_forecasts) == 7
    assert res.mean_daily_forecast > 0
    assert res.total_forecast_demand > 0

def test_forecast_request_validation():
    # Horizon out of bounds (< 1 or > 90) should raise ValidationError
    with pytest.raises(ValidationError):
        ForecastRequest(product_id="SKU_BEV_01", store_id="STORE_EAST", horizon_days=0)
        
    with pytest.raises(ValidationError):
        ForecastRequest(product_id="SKU_BEV_01", store_id="STORE_EAST", horizon_days=100)

def test_inventory_recommendation_endpoint():
    req = ReplenishmentRequest(
        product_id="SKU_BEV_01",
        store_id="STORE_EAST",
        current_on_hand=50,
        on_order_units=0,
        service_level=0.95,
        review_period_days=7
    )
    res = get_inventory_recommendation(req)
    assert res["product_id"] == "SKU_BEV_01"
    assert "reorder_point" in res
    assert "safety_stock" in res
    assert "suggested_order_qty" in res
    assert res["days_of_supply"] > 0
    assert res["stockout_risk_level"] in ["HIGH", "MEDIUM", "LOW"]

def test_dashboard_html_serving():
    res = serve_dashboard()
    assert res.status_code == 200
    assert b"Demand Forecasting &amp; Inventory Optimization" in res.body or b"Demand" in res.body
