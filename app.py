import io
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="AI Super Egg – Price Forecast",
    page_icon="🥚",
    layout="wide"
)

PRIMARY_COLOR = "#0ea5e9"

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv(file, date_col=None):
    df = pd.read_csv(file)
    # date normalization
    guess_cols = [date_col] if date_col else [c for c in df.columns if c.lower() in ["ds", "date"]]
    for c in guess_cols:
        if c and c in df.columns:
            # try several common formats before letting pandas infer
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    df[c] = pd.to_datetime(df[c], format=fmt)
                    break
                except Exception:
                    pass
            if not np.issubdtype(df[c].dtype, np.datetime64):
                df[c] = pd.to_datetime(df[c], errors="coerce")
            df.rename(columns={c: "ds"}, inplace=True)
            break
    # standardize target name if present
    for yname in ["y", "Price", "PriceMarket", "yMarket", "price", "pricemarket"]:
        if yname in df.columns:
            df.rename(columns={yname: "y"}, inplace=True)
            break
    # Prophet convention: yhat bands
    ren = {}
    for k in df.columns:
        lk = k.lower()
        if lk in ["yhat_lower", "yhat upper", "yhat_upper", "lower", "lo"]:
            ren[k] = "yhat_lower"
        if lk in ["yhat_upper", "upper", "hi"]:
            ren[k] = "yhat_upper"
    if ren:
        df.rename(columns=ren, inplace=True)
    return df

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.warning(f"ไฟล์ **{name}** ขาดคอลัมน์ที่จำเป็น: {missing}")
        return False
    return True

def infer_bounds(dates: pd.Series):
    dmin = pd.to_datetime(dates.min())
    dmax = pd.to_datetime(dates.max())
    return dmin, dmax

def compute_metrics(actual_df: pd.DataFrame, pred_df: pd.DataFrame):
    # merge on ds for alignment
    df = pd.merge(
        actual_df[["ds", "y"]].dropna(),
        pred_df[["ds", "yhat"]].rename(columns={"yhat": "pred"}),
        on="ds", how="inner"
    ).sort_values("ds")
    if df.empty:
        return None, None
    mae = np.mean(np.abs(df["y"] - df["pred"]))
    rmse = np.sqrt(np.mean((df["y"] - df["pred"])**2))
    # avoid division by zero in MAPE
    denom = np.where(df["y"] == 0, np.nan, np.abs(df["y"]))
    mape = np.nanmean(np.abs((df["y"] - df["pred"]) / denom)) * 100
    return df, {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}

def style_kpi(val, fmt="{:,.2f}"):
    try:
        return fmt.format(val) if pd.notnull(val) else "—"
    except Exception:
        return "—"

# ----------------------------
# UI – Sidebar (Inputs)
# ----------------------------
with st.sidebar:
    st.header("🔧 Data Inputs")
    st.caption("อัปโหลด output จาก Google Colab (เช่น **forecast.csv** และ **history/test.csv**)")
    fc_file = st.file_uploader("Forecast file (ต้องมีคอลัมน์: ds, yhat[, yhat_lower, yhat_upper])", type=["csv"], key="fc")
    ac_file = st.file_uploader("Actual/History file (มี ds, y)", type=["csv"], key="ac")

    st.divider()

    st.header("🎛️ Display Options")
    show_ci = st.toggle("แสดงช่วงความเชื่อมั่น (yhat_lower–yhat_upper)", value=True)
    show_points = st.toggle("แสดงจุดข้อมูล (markers)", value=False)
    smooth = st.slider("Smoothing (moving average windows)", 1, 12, 1, help="เฉลี่ยเคลื่อนที่ (สัปดาห์) สำหรับเส้นคาดการณ์และ actual")
    st.caption("💡 ถ้าไม่มี yhat_lower/yhat_upper จะไม่วาด CI ให้")

# ----------------------------
# Main
# ----------------------------
st.title("🥚 AI Super Egg – Price Forecast")
st.write("แอปสำหรับสำรวจผลพยากรณ์ราคาไข่ไก่จากโมเดล (Prophet) ที่เทรนใน Google Colab และปรับช่วงข้อมูลแบบ **Dynamic Range** ได้ทันที")

tab_plot, tab_table, tab_about = st.tabs(["📈 Visualization", "📊 Tables", "ℹ️ About"])

with tab_about:
    st.subheader("วิธีใช้งานโดยย่อ")
    st.markdown("""
1) อัปโหลด **forecast.csv** จาก Colab (คอลัมน์ต้องมีอย่างน้อย `ds, yhat` และถ้ามี `yhat_lower, yhat_upper` จะวาดช่วงความเชื่อมั่นให้)  
2) (แนะนำ) อัปโหลด **actual/history.csv** สำหรับ `y` เพื่อคำนวณ KPI (MAE / RMSE / MAPE) และเปรียบเทียบจริง–พยากรณ์  
3) เลือกช่วงวันที่ด้วยสไลเดอร์ เพื่อโฟกัสเฉพาะช่วงที่ต้องการ  
4) ปรับตัวเลือกการแสดงผล (CI / markers / smoothing)
    """)
    st.caption("หมายเหตุ: สคีมาข้อมูลยืดหยุ่น – แอปจะพยายามแม็พ `Date`→`ds` และ `Price/PriceMarket`→`y` ให้อัตโนมัติ")

# ----------------------------
# Load Data
# ----------------------------
forecast_df = None
actual_df = None
if fc_file:
    forecast_df = read_csv(fc_file)
    # require ds & yhat at minimum
    if not require_cols(forecast_df, ["ds", "yhat"], "Forecast"):
        forecast_df = None

if ac_file:
    actual_df = read_csv(ac_file)
    # require ds & y at minimum
    if not require_cols(actual_df, ["ds", "y"], "Actual/History"):
        actual_df = None

if (forecast_df is None) and (actual_df is None):
    st.info("อัปโหลดไฟล์ด้านซ้ายเพื่อเริ่มต้น")
    st.stop()

# ----------------------------
# Date Range (Dynamic)
# ----------------------------
all_dates = []
if forecast_df is not None:
    all_dates.append(forecast_df["ds"])
if actual_df is not None:
    all_dates.append(actual_df["ds"])
date_min, date_max = infer_bounds(pd.concat(all_dates)) if all_dates else (None, None)

with st.container():
    st.subheader("ช่วงวันที่ (Dynamic Range)")
    if date_min and date_max:
        start, end = st.slider(
            "เลือกช่วงเวลา",
            min_value=pd.to_datetime(date_min).to_pydatetime(),
            max_value=pd.to_datetime(date_max).to_pydatetime(),
            value=(pd.to_datetime(max(date_min, date_max - pd.Timedelta(days=180))).to_pydatetime(),
                   pd.to_datetime(date_max).to_pydatetime()),
            format="YYYY-MM-DD",
            key="date_range",
        )
    else:
        st.warning("ไม่พบช่วงวันที่ที่ใช้ได้")
        st.stop()

# filter by range
def slice_by_range(df):
    if df is None:
        return None
    return df[(df["ds"] >= pd.to_datetime(start)) & (df["ds"] <= pd.to_datetime(end))].copy()

fc_view = slice_by_range(forecast_df)
ac_view = slice_by_range(actual_df)

# moving average smoothing (optional)
def apply_smooth(df, col):
    if df is None or col not in df.columns or smooth <= 1:
        return df
    df = df.sort_values("ds").copy()
    df[col] = df[col].rolling(window=smooth, min_periods=1).mean()
    return df

fc_view = apply_smooth(fc_view, "yhat")
fc_view = apply_smooth(fc_view, "yhat_lower")
fc_view = apply_smooth(fc_view, "yhat_upper")
ac_view = apply_smooth(ac_view, "y")

# ----------------------------
# Metrics
# ----------------------------
eval_df, metrics = (None, None)
if (ac_view is not None) and (fc_view is not None):
    eval_df, metrics = compute_metrics(ac_view, fc_view)

kpi_cols = st.columns(3)
with kpi_cols[0]:
    st.metric("MAE", style_kpi(metrics["MAE"]) if metrics else "—")
with kpi_cols[1]:
    st.metric("RMSE", style_kpi(metrics["RMSE"]) if metrics else "—")
with kpi_cols[2]:
    st.metric("MAPE (%)", style_kpi(metrics["MAPE(%)"], "{:,.2f}%") if metrics else "—")

# ----------------------------
# Plot
# ----------------------------
with tab_plot:
    fig = go.Figure()

    if (fc_view is not None) and ("yhat" in fc_view.columns):
        fig.add_trace(go.Scatter(
            x=fc_view["ds"], y=fc_view["yhat"],
            mode="lines+markers" if show_points else "lines",
            name="Forecast (yhat)",
            line=dict(width=2)
        ))
        if show_ci and ("yhat_lower" in fc_view.columns) and ("yhat_upper" in fc_view.columns):
            fig.add_traces([
                go.Scatter(
                    x=pd.concat([fc_view["ds"], fc_view["ds"][::-1]]),
                    y=pd.concat([fc_view["yhat_upper"], fc_view["yhat_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(14,165,233,0.15)",  # based on PRIMARY_COLOR
                    line=dict(width=0),
                    hoverinfo="skip",
                    name="Confidence Interval"
                )
            ])

    if (ac_view is not None) and ("y" in ac_view.columns):
        fig.add_trace(go.Scatter(
            x=ac_view["ds"], y=ac_view["y"],
            mode="lines+markers" if show_points else "lines",
            name="Actual (y)",
            line=dict(width=2, dash="dot")
        ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        title="Actual vs Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Tables
# ----------------------------
with tab_table:
    left, right = st.columns(2)
    with left:
        st.subheader("Forecast (filtered)")
        if fc_view is not None and not fc_view.empty:
            st.dataframe(fc_view.sort_values("ds"), use_container_width=True)
        else:
            st.caption("—")
    with right:
        st.subheader("Actual/History (filtered)")
        if ac_view is not None and not ac_view.empty:
            st.dataframe(ac_view.sort_values("ds"), use_container_width=True)
        else:
            st.caption("—")

# ----------------------------
# Footer
# ----------------------------
st.caption("Tips: ถ้าไฟล์จาก Colab มีชื่อคอลัมน์แปลก (เช่น Layingheny/Feedy) ไม่กระทบ เพราะแอปนี้ใช้เฉพาะ ds, y, yhat[, yhat_lower, yhat_upper]")
