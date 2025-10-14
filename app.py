import io
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="CPF War Room – AI Super Egg Forecast",
    page_icon="🥚",
    layout="wide"
)

# CPF Theme Colors
CPF_GREEN = "#76d9c1"
CPF_YELLOW = "#f8cf40"
CPF_GRAY = "#cfd3da"
DARK_BG = "#101827"

# ----------------------------
# Helper Functions
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv(file, date_col=None):
    df = pd.read_csv(file)
    # Detect date column
    guess_cols = [date_col] if date_col else [c for c in df.columns if c.lower() in ["ds", "date"]]
    for c in guess_cols:
        if c and c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df.rename(columns={c: "ds"}, inplace=True)
            break
    # Normalize y and yhat
    for yname in ["y", "PriceMarket", "Price", "Actual"]:
        if yname in df.columns:
            df.rename(columns={yname: "y"}, inplace=True)
    for yhat in ["yhat", "Predicted", "PriceMarket_predicted"]:
        if yhat in df.columns:
            df.rename(columns={yhat: "yhat"}, inplace=True)
    return df

def compute_metrics(actual_df, pred_df):
    df = pd.merge(
        actual_df[["ds", "y"]].dropna(),
        pred_df[["ds", "yhat"]],
        on="ds",
        how="inner"
    )
    if df.empty:
        return None
    mae = np.mean(np.abs(df["y"] - df["yhat"]))
    rmse = np.sqrt(np.mean((df["y"] - df["yhat"]) ** 2))
    mape = np.mean(np.abs((df["y"] - df["yhat"]) / df["y"])) * 100
    return round(mae, 2), round(rmse, 2), round(mape, 2)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/3a/Charoen_Pokphand_Foods_logo.png", width=160)
    st.markdown("### ⚙️ Upload AI Model Outputs")
    fc_file = st.file_uploader("📈 Forecast file (from Colab)", type=["csv"])
    ac_file = st.file_uploader("📊 Actual file (optional)", type=["csv"])
    show_ci = st.toggle("Show Confidence Interval", True)
    smooth = st.slider("Smoothing Window", 1, 12, 1)
    st.caption("💡 Tip: Upload forecast.csv + actual.csv to compare AI prediction accuracy.")

# ----------------------------
# Header
# ----------------------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown(
        f"""
        <h2 style='color:{CPF_GREEN}; font-weight:700;'>
            🥚 CPF War Room | AI Super Egg Price Forecast
        </h2>
        <p style='color:{CPF_GRAY};'>
            Data-driven insight dashboard powered by Google Colab model outputs.
        </p>
        """, unsafe_allow_html=True
    )
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/3a/Charoen_Pokphand_Foods_logo.png", width=120)

st.markdown("---")

# ----------------------------
# Data Load
# ----------------------------
if fc_file:
    forecast_df = read_csv(fc_file)
    st.success(f"✅ Forecast data loaded: {len(forecast_df)} rows")
else:
    st.warning("กรุณาอัปโหลดไฟล์ Forecast ก่อนเริ่ม")
    st.stop()

actual_df = read_csv(ac_file) if ac_file else None

# ----------------------------
# Dynamic Date Range
# ----------------------------
dmin, dmax = forecast_df["ds"].min(), forecast_df["ds"].max()
start, end = st.slider(
    "📆 Select Date Range",
    min_value=dmin.to_pydatetime(),
    max_value=dmax.to_pydatetime(),
    value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
    format="YYYY-MM-DD"
)

# Filter by selected range
mask = (forecast_df["ds"] >= start) & (forecast_df["ds"] <= end)
forecast_df = forecast_df.loc[mask]
if actual_df is not None:
    actual_df = actual_df.loc[(actual_df["ds"] >= start) & (actual_df["ds"] <= end)]

# ----------------------------
# Compute Metrics
# ----------------------------
if actual_df is not None:
    mae, rmse, mape = compute_metrics(actual_df, forecast_df)
else:
    mae, rmse, mape = (None, None, None)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("📉 MAE", f"{mae or '-'}")
kpi2.metric("📈 RMSE", f"{rmse or '-'}")
kpi3.metric("🎯 MAPE (%)", f"{mape or '-'}")

# ----------------------------
# Visualization
# ----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_df["ds"], y=forecast_df["yhat"],
    mode="lines", name="Predicted Price",
    line=dict(color=CPF_YELLOW, width=3)
))

if actual_df is not None and "y" in actual_df.columns:
    fig.add_trace(go.Scatter(
        x=actual_df["ds"], y=actual_df["y"],
        mode="lines+markers", name="Actual Price",
        line=dict(color=CPF_GREEN, width=2)
    ))

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
    title="Price Forecast Visualization",
    xaxis_title="Date",
    yaxis_title="Price (THB)",
    font=dict(color=CPF_GRAY),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left")
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Data Table
# ----------------------------
with st.expander("🔍 Show Data Preview"):
    st.dataframe(forecast_df.head(20))

st.markdown(f"<p style='text-align:center; color:{CPF_GRAY};'>© 2025 CPF Digital Transformation | War Room AI Super Egg</p>", unsafe_allow_html=True)
