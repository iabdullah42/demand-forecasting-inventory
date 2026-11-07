"""Demand Forecasting & Inventory Planner - interactive dashboard.

Run from the project folder:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inventory import simulate

ROOT = Path(__file__).resolve().parent
CAL_END = pd.Timestamp(
    "2025-11-01"
)  # October = measure forecast error; Nov-Dec = simulate
MODEL_NAMES = {
    "ensemble": "Ensemble",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "naive": "Naive baseline",
}

st.set_page_config(
    page_title="Demand & Inventory Planner", page_icon="📦", layout="wide"
)


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data" / "processed" / "forecasts.csv", parse_dates=["date"]
    )


def run_policy(
    g: pd.DataFrame, model: str, service_level: float, lead_time: int
) -> dict:
    """Measure forecast error in October, then simulate Nov-Dec with that policy."""
    calib, sim = g[g["date"] < CAL_END], g[g["date"] >= CAL_END]
    sigma = (calib["units_sold"] - calib[model]).std()
    result = simulate(
        sim["units_sold"].to_numpy(float),
        sim[model].to_numpy(float),
        sigma,
        service_level,
        lead_time,
    )
    result["inventory_value"] = result["avg_on_hand"] * g["unit_price"].iloc[0]
    return result


def wape(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.abs(actual - forecast).sum() / actual.sum() * 100)


fc = load_forecasts()

# ---------- Sidebar: the planner's choices ----------
st.sidebar.header("Planner settings")
sku = st.sidebar.selectbox("Product (SKU)", sorted(fc["sku"].unique()))
model = st.sidebar.selectbox(
    "Forecast model", list(MODEL_NAMES), format_func=MODEL_NAMES.get
)
service_level = st.sidebar.slider(
    "Target service level", 0.80, 0.995, 0.95, 0.005, format="%.3f"
)
lead_time = st.sidebar.slider("Supplier lead time (days)", 1, 14, 5)

g = fc[fc["sku"] == sku]
result = run_policy(g, model, service_level, lead_time)
baseline = run_policy(g, "naive", service_level, lead_time)

# ---------- Header and KPIs ----------
st.title("📦 Demand Forecasting & Inventory Planner")
st.caption(
    f"{sku} · {g['category'].iloc[0]} · ${g['unit_price'].iloc[0]:.2f}/unit · "
    "forecast Oct-Dec 2025, inventory simulated Nov-Dec 2025"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Forecast error (WAPE)",
    f"{wape(g['units_sold'], g[model]):.1f}%",
    f"{wape(g['units_sold'], g[model]) - wape(g['units_sold'], g['naive']):+.1f} pts vs naive",
    delta_color="inverse",
)
k2.metric(
    "Fill rate",
    f"{result['fill_rate_%']:.1f}%",
    f"{result['fill_rate_%'] - baseline['fill_rate_%']:+.1f} pts vs naive",
)
k3.metric(
    "Stockout days",
    f"{result['stockout_days']} of 61",
    f"{result['stockout_days'] - baseline['stockout_days']:+d} vs naive",
    delta_color="inverse",
)
k4.metric("Safety stock", f"{result['safety_stock']:.0f} units")
k5.metric(
    "Avg inventory value",
    f"${result['inventory_value']:,.0f}",
    f"{result['inventory_value'] - baseline['inventory_value']:+,.0f} vs naive",
    delta_color="off",
)


# ---------- Chart 1: forecast vs actual ----------
st.subheader("Forecast vs actual demand")
fig = go.Figure()
fig.add_scatter(
    x=g["date"], y=g["units_sold"], name="Actual", line=dict(color="lightgray")
)
if model != "naive":
    fig.add_scatter(
        x=g["date"], y=g["naive"], name="Naive baseline", line=dict(dash="dash")
    )
fig.add_scatter(x=g["date"], y=g[model], name=MODEL_NAMES[model], line=dict(width=3))
promo_days = g.loc[g["on_promo"] == 1, "date"]
fig.add_scatter(
    x=promo_days,
    y=g.loc[g["on_promo"] == 1, "units_sold"],
    mode="markers",
    name="Promotion day",
    marker=dict(symbol="star", size=10),
)
fig.add_vline(x=CAL_END, line_dash="dot", line_color="gray")
fig.add_annotation(
    x=CAL_END,
    y=1,
    yref="paper",
    text="simulation starts",
    showarrow=False,
    xanchor="left",
)
fig.update_layout(
    height=380, hovermode="x unified", yaxis_title="Units per day", margin=dict(t=20)
)
st.plotly_chart(fig)

# ---------- Chart 2: the inventory trade-off curve ----------
st.subheader("Inventory vs service trade-off")
levels = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995]
curve = go.Figure()
for m in dict.fromkeys(["naive", model]):
    pts = pd.DataFrame([run_policy(g, m, sl, lead_time) | {"sl": sl} for sl in levels])
    curve.add_scatter(
        x=pts["inventory_value"],
        y=pts["fill_rate_%"],
        mode="lines+markers",
        name=MODEL_NAMES[m],
        text=[f"{sl:.1%} target" for sl in levels],
        hovertemplate="%{text}<br>$%{x:,.0f} inventory<br>%{y:.1f}% fill rate<extra></extra>",
    )
curve.add_scatter(
    x=[result["inventory_value"]],
    y=[result["fill_rate_%"]],
    mode="markers",
    name="Your setting",
    marker=dict(size=16, symbol="diamond", color="black"),
)
curve.update_layout(
    height=380,
    xaxis_title="Average inventory value ($)",
    yaxis_title="Fill rate (%)",
    margin=dict(t=20),
)
st.plotly_chart(curve)
st.caption(
    "Up and to the left is better: more demand shipped for less money on the shelf."
)

# ---------- Method ----------
with st.expander("How this works"):
    st.markdown(f"""
- **Forecasts** for Oct-Dec 2025 come from models trained only on data through Sep 30, 2025.
- **Policy:** daily review, order-up-to. Each day the planner orders enough to bring stock on hand
  plus on order up to *forecast demand over the lead time + safety stock*. Orders arrive after
  **{lead_time} days**; unmet demand is lost.
- **Safety stock** = z × σ × √lead time, where σ is the forecast's daily error measured in **October**.
  The policy is then judged on **November-December**, a period it never saw.
- **Data** is synthetic (generated with a fixed seed), so no real company data is used.
""")
