"""Generate a realistic daily sales history for a consumer-goods distribution center.

Why synthetic data? Because we control the "true" demand pattern, we can later
check whether our forecasts and inventory policy actually work.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2023-01-01", "2025-12-31"

# Actual US holiday dates (Memorial Day, July 4, Labor Day, Thanksgiving, Christmas).
HOLIDAYS = [
    "2023-05-29",
    "2023-07-04",
    "2023-09-04",
    "2023-11-23",
    "2023-12-25",
    "2024-05-27",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
    "2025-05-26",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
]

# Each product: (sku, category, base daily units, yearly trend, seasonality strength, price)
PRODUCTS = [
    ("SKU-001", "Beverages", 120, 0.08, 0.30, 2.49),
    ("SKU-002", "Beverages", 85, 0.05, 0.35, 3.99),
    ("SKU-003", "Beverages", 40, -0.04, 0.25, 5.49),
    ("SKU-004", "Snacks", 95, 0.10, 0.10, 3.29),
    ("SKU-005", "Snacks", 60, 0.02, 0.12, 4.79),
    ("SKU-006", "Snacks", 25, 0.15, 0.08, 6.99),
    ("SKU-007", "Personal Care", 30, 0.03, 0.05, 7.99),
    ("SKU-008", "Personal Care", 18, -0.02, 0.05, 12.49),
    ("SKU-009", "Household", 45, 0.01, 0.15, 9.99),
    ("SKU-010", "Household", 12, 0.06, 0.20, 15.99),
]


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    t_years = (dates - dates[0]).days.to_numpy() / 365.25
    day_of_year = dates.day_of_year.to_numpy()
    weekday = dates.day_of_week.to_numpy()  # Monday = 0

    # Shipments out of a DC are heavier early in the week, light on weekends.
    weekday_effect = np.array([1.15, 1.10, 1.05, 1.00, 0.95, 0.85, 0.90])[weekday]

    # Demand spikes in the days before major US holidays, as stores restock.
    holiday_boost = np.ones(len(dates))
    for holiday in HOLIDAYS:
        peak = pd.Timestamp(holiday) - pd.Timedelta(days=3)
        days_away = np.abs((dates - peak).days.to_numpy())
        holiday_boost += 0.35 * np.exp(-days_away / 3)

    rows = []
    for sku, category, base, trend, season, price in PRODUCTS:
        # Summer peak for beverages, winter peak for everything else.
        peak_day = 196 if category == "Beverages" else 15
        yearly = 1 + season * np.cos(2 * np.pi * (day_of_year - peak_day) / 365.25)
        growth = (1 + trend) ** t_years

        # About 1 in 20 days has a promotion that lifts sales 30-60%.
        promo = rng.random(len(dates)) < 0.05
        promo_lift = np.where(promo, rng.uniform(1.3, 1.6, len(dates)), 1.0)

        mean = base * growth * yearly * weekday_effect * holiday_boost * promo_lift
        # Negative binomial noise: real demand is more variable than Poisson.
        dispersion = 20
        units = rng.negative_binomial(dispersion, dispersion / (dispersion + mean))

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku": sku,
                    "category": category,
                    "units_sold": units,
                    "on_promo": promo.astype(int),
                    "unit_price": price,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "sales.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")
    print(df.head())
    print("\nAverage daily units by SKU:")
    print(df.groupby("sku")["units_sold"].mean().round(1))
