import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="CPF War Room – AI Super Egg Forecast Plus",
    page_icon="🥚",
    layout="wide"
)

# Theme Colors
CPF_GREEN = "#76d9c1"
CPF_YELLOW = "#f8cf40"
CPF_GRAY = "#cfd3da"
DARK_BG = "#101827"

# ----------------------------
# Helper Functions
# ----------------------------
def read_csv_auto(file_path_or_buffer):
    """อ่าน CSV แล้ว normalize คอลัมน์ชื่อ Date/y/yhat"""
    df = pd.read_csv(file_path_or_buffer)
    df.rename(columns=lambda c: c.strip(), inplace=True)
    if "Date" in df.columns:
        df.rename(columns={"Date": "ds"}, inplace=True)
    if "PriceMarket" in df.columns:
        df.rename(columns={"PriceMarket": "yhat"}, inplace=True)
    if "PriceMarket_predicted" in df.columns:
        df.rename(columns={"PriceMarket_predicted": "yhat"}, inplace=True)
    if "Price" in df.columns:
        df.rename(columns={"Price": "y"}, inplace=True)
    return df

def compute_metrics(actual_df, forecast_df):
    """คำนวณ MAE, RMSE, MAPE"""
    df = pd.merge(
        actual_df[["ds", "y"]],
        forecast_df[["ds", "yhat"]],
        on="ds",
        how="inner"
    )
    if df.empty:
        return None, None, None
    mae = np.mean(np.abs(df["y"] - df["yhat"]))
    rmse = np.sqrt(np.mean((df["y"] - df["yhat"]) ** 2))
    mape = np.mean(np.abs((df["y"] - df["yhat"]) / df["y"])) * 100
    return round(mae, 2), round(rmse, 2), round(mape, 2)

# ----------------------------
# Sidebar (Auto-load + Upload + Compare)
# ----------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/3a/Charoen_Pokphand_Foods_logo.png", width=160)
    st.markdown("### ⚙️ Upload or Auto-Load AI Model Outputs")

    forecast_df, actual_df, forecast_v2 = None, None, None

    # Auto-load forecast.csv
    if os.path.exists("forecast.csv"):
        st.success("📈 Auto-loaded: forecast.csv")
        forecast_df = read_csv_auto("forecast.csv")
    else:
        fc_file = st.file_uploader("📈 Upload Forecast (Model V1)", type=["csv"], key="v1")
        if fc_file:
            forecast_df = read_csv_auto(fc_file)

    # Auto-load actual.csv
    if os.path.exists("actual.csv"):
        st.success("📊 Auto-loaded: actual.csv")
        actual_df = read_csv_auto("actual.csv")
    else:
        ac_file = st.file_uploader("📊 Upload Actual Data (Optional)", type=["csv"])
        if ac_file:
            actual_df = read_csv_auto(ac_file)

    st.divider()
    st.markdown("### 📊 Compare Model V2 (Optional)")
    fc_file_v2 = st.file_uploader("Upload Forecast V2 (optional)", type=["csv"], key="v2")
    if fc_file_v2:
        forecast_v2 = read_csv_auto(fc_file_v2)

    show_ci = st.toggle("Show Confidence Interval", True)
    smooth = st.slider("Smoothing Window", 1, 12, 1)
    st.caption("💡 ถ้ามีไฟล์ forecast.csv / actual.csv ใน repo แอปจะโหลดให้อัตโนมัติ")

# ----------------------------
# Header
# ----------------------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown(
        f"""
        <h2 style='color:{CPF_GREEN}; font-weight:700;'>
            🥚 CPF War Room | AI Super Egg Forecast (Plus Edition)
        </h2>
        <p style='color:{CPF_GRAY};'>
            Interactive Dashboard with Auto-Load, Compare Models, and CSV Export.
        </p>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/3a/Charoen_Pokphand_Foods_logo.png", width=120)

st.markdown("---")

# ----------------------------
# Data Validation
# ----------------------------
if forecast_df is None:
    st.warning("⚠️ กรุณาอัปโหลดหรือใส่ไฟล์ forecast.csv ก่อนเริ่มต้น")
    st.stop()

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

mask = (forecast_df["ds"] >= start) & (forecast_df["ds"] <= end)
forecast_df = forecast_df.loc[mask]
if actual_df is not None:
    actual_df = actual_df.loc[(actual_df["ds"] >= start) & (actual_df["ds"] <= end)]

# ----------------------------
# Metrics
# ----------------------------
if actual_df is not None:
    mae, rmse, mape = compute_metrics(actual_df, forecast_df)
else:
    mae, rmse, mape = (None, None, None)

k1, k2, k3 = st.columns(3)
k1.metric("📉 MAE", f"{mae or '-'}")
k2.metric("📈 RMSE", f"{rmse or '-'}")
k3.metric("🎯 MAPE (%)", f"{mape or '-'}")

# ----------------------------
# Visualization
# ----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_df["ds"], y=forecast_df["yhat"],
    mode="lines", name="Forecast Model V1",
    line=dict(color=CPF_YELLOW, width=3)
))

if actual_df is not None:
    fig.add_trace(go.Scatter(
        x=actual_df["ds"], y=actual_df["y"],
        mode="lines+markers", name="Actual",
        line=dict(color=CPF_GREEN, width=2, dash="dot")
    ))

if forecast_v2 is not None:
    forecast_v2 = forecast_v2[(forecast_v2["ds"] >= start) & (forecast_v2["ds"] <= end)]
    fig.add_trace(go.Scatter(
        x=forecast_v2["ds"], y=forecast_v2["yhat"],
        mode="lines", name="Forecast Model V2",
        line=dict(color="#ff6b6b", width=2, dash="dash")
    ))

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
    xaxis_title="Date",
    yaxis_title="Price (THB)",
    font=dict(color=CPF_GRAY),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left")
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Download filtered data
# ----------------------------
st.markdown("### 🔽 Download Filtered Data")
if forecast_df is not None:
    combined = forecast_df.copy()
    if actual_df is not None:
        combined = pd.merge(
            forecast_df[["ds", "yhat"]],
            actual_df[["ds", "y"]],
            on="ds",
            how="outer"
        ).sort_values("ds")
    csv_data = combined.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download current view (CSV)",
        data=csv_data,
        file_name=f"filtered_egg_forecast_{start.date()}_{end.date()}.csv",
        mime="text/csv"
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown(f"<p style='text-align:center; color:{CPF_GRAY};'>© 2025 CPF Digital Transformation | War Room AI Super Egg – Plus Edition</p>", unsafe_allow_html=True)
