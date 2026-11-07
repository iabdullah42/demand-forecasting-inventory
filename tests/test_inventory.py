import numpy as np
import pytest

from src.inventory import safety_stock, simulate, z_score


def test_z_score_known_values():
    assert z_score(0.50) == pytest.approx(0.0, abs=1e-9)
    assert z_score(0.95) == pytest.approx(1.645, abs=0.001)


def test_safety_stock_grows_with_square_root_of_lead_time():
    # 4x the lead time should mean 2x the safety stock, not 4x.
    assert safety_stock(10, 0.95, 20) == pytest.approx(2 * safety_stock(10, 0.95, 5))


def test_perfect_forecast_never_stocks_out():
    demand = np.full(60, 100.0)
    result = simulate(demand, demand, sigma_error=0.0)
    assert result["fill_rate_%"] == 100
    assert result["stockout_days"] == 0


def test_higher_service_level_means_more_stock_and_fewer_stockouts():
    rng = np.random.default_rng(0)
    demand = rng.normal(100, 25, 120).round().clip(0)
    forecast = np.full(120, 100.0)
    low = simulate(demand, forecast, sigma_error=25, service_level=0.80)
    high = simulate(demand, forecast, sigma_error=25, service_level=0.99)
    assert high["avg_on_hand"] > low["avg_on_hand"]
    assert high["stockout_days"] <= low["stockout_days"]
