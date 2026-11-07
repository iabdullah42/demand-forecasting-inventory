"""Turn daily forecasts into inventory decisions and simulate the results.

Policy: daily review, order-up-to level.
  Each day, after demand ships, the planner checks the inventory position
  (on hand + already ordered). If it is below the target level, they order
  the difference. Orders arrive after a fixed lead time.

  An order placed today must cover demand until the NEXT order arrives,
  which is `lead_time` days of exposure (the "protection interval").

  target level = forecast demand over the protection interval + safety stock
  safety stock = z * sigma_error * sqrt(lead time)
"""

from statistics import NormalDist

import numpy as np


def z_score(service_level: float) -> float:
    """Number of standard deviations of protection for a target service level."""
    return NormalDist().inv_cdf(service_level)


def safety_stock(sigma_error: float, service_level: float, lead_time: int) -> float:
    return z_score(service_level) * sigma_error * np.sqrt(lead_time)


def simulate(
    actual: np.ndarray,
    forecast: np.ndarray,
    sigma_error: float,
    service_level: float = 0.95,
    lead_time: int = 5,
) -> dict:
    """Simulate one SKU day by day. Unmet demand is lost (not backordered)."""
    n = len(actual)
    ss = safety_stock(sigma_error, service_level, lead_time)
    cover = lead_time

    def target(day: int) -> float:
        window = forecast[day : day + cover]
        # Near the end of the horizon, extend the last known forecast.
        window = np.concatenate([window, np.repeat(forecast[-1], cover - len(window))])
        return window.sum() + ss

    on_hand = target(0)  # start fully stocked: lead-time demand plus safety stock
    arrivals = np.zeros(n + lead_time + 1)
    on_order = 0.0
    shipped = stockout_days = 0
    on_hand_history = []

    for day in range(n):
        on_hand += arrivals[day]
        on_order -= arrivals[day]

        sold = min(on_hand, actual[day])
        shipped += sold
        stockout_days += int(sold < actual[day])
        on_hand -= sold
        on_hand_history.append(on_hand)

        order = max(0.0, target(day + 1) - (on_hand + on_order))
        arrivals[day + lead_time] += order
        on_order += order

    return {
        "fill_rate_%": 100 * shipped / actual.sum(),
        "stockout_days": stockout_days,
        "avg_on_hand": float(np.mean(on_hand_history)),
        "safety_stock": ss,
    }
