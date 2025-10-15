
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from layer_aibp_super_egg import train_df

st.set_page_config(page_title="Layer-X Super Egg – War Room", page_icon="🥚", layout="wide")

try:
    logo = Image.open("LOGO-CPF.jpg")
    st.sidebar.image(logo, width=150)
except Exception:
    st.sidebar.write("CPF")
st.sidebar.markdown("### Layer-X Super Egg Dashboard")
st.sidebar.write("Executive War Room View + Analytics")

st.markdown("""
<style>
.kpi-card {border-radius: 16px; padding: 18px 18px 12px; margin: 6px; box-shadow: 0 4px 16px rgba(0,0,0,0.06);}
.kpi-title {font-size: 0.85rem; color:#2f3b3f; margin-bottom: 6px;}
.kpi-value {font-size: 1.6rem; font-weight: 700; margin-bottom: 4px;}
.kpi-foot {font-size: 0.8rem; color:#6b6f72;}
.bg-mint {background:#e8fbf6;}
.bg-gold {background:#fff6d6;}
.bg-silver {background:#eef2f5;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color:#76d9c1;'>🥚 Layer-X Super Egg – War Room </h2>", unsafe_allow_html=True)

# RANGE FILTER
df = train_df.copy()
if "ds" in df.columns:
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
if "ds" in df.columns and df["ds"].notna().any():
    min_date = pd.to_datetime(df["ds"].min()).to_pydatetime().date()
    max_date = pd.to_datetime(df["ds"].max()).to_pydatetime().date()
    date_range = st.sidebar.slider("เลือกช่วงวันที่", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    df = df[(df["ds"] >= pd.Timestamp(date_range[0])) & (df["ds"] <= pd.Timestamp(date_range[1]))].copy()
else:
    st.sidebar.info("ไม่พบคอลัมน์วันที่ (ds) หรือข้อมูลไม่ถูกต้อง")

# KPI
has_price = "PriceMarket" in df.columns
kpi_avg = float(df["PriceMarket"].mean()) if has_price else None
kpi_max = float(df["PriceMarket"].max()) if has_price else None
kpi_min = float(df["PriceMarket"].min()) if has_price else None
kpi_vol = float((df["PriceMarket"].std() / df["PriceMarket"].mean()) * 100) if has_price and df["PriceMarket"].mean() else None

r=None
if has_price and "FeedPrice" in df.columns:
    corr=df[["FeedPrice","PriceMarket"]].dropna()
    if len(corr)>=2: r=float(corr["FeedPrice"].corr(corr["PriceMarket"]))

tabs=st.tabs(["🏛️ War Room Summary","📈 52-Week Forecast","📊 Actual vs Predicted","📅 Yearly Trend","🔗 Correlation (Feed vs Market)","🌍 Heatmap by Province","🎞️ Animated Line (Weekly)"])

with tabs[0]:
    st.subheader("Executive KPI Overview")
    c1,c2,c3,c4,c5=st.columns(5)
    if kpi_avg is not None:
        with c1: st.markdown(f"<div class='kpi-card bg-mint'><div class='kpi-title'>Avg Price</div><div class='kpi-value'>{kpi_avg:.2f} THB</div><div class='kpi-foot'>ช่วงที่เลือก</div></div>",unsafe_allow_html=True)
    else:
        with c1: st.info("ไม่มี PriceMarket")
    if kpi_max is not None:
        with c2: st.markdown(f"<div class='kpi-card bg-gold'><div class='kpi-title'>Max Price</div><div class='kpi-value'>{kpi_max:.2f} THB</div><div class='kpi-foot'>ค่าสูงสุด</div></div>",unsafe_allow_html=True)
    if kpi_min is not None:
        with c3: st.markdown(f"<div class='kpi-card bg-silver'><div class='kpi-title'>Min Price</div><div class='kpi-value'>{kpi_min:.2f} THB</div><div class='kpi-foot'>ค่าต่ำสุด</div></div>",unsafe_allow_html=True)
    if kpi_vol is not None:
        with c4: st.markdown(f"<div class='kpi-card bg-mint'><div class='kpi-title'>Volatility (CV)</div><div class='kpi-value'>{kpi_vol:.2f}%</div><div class='kpi-foot'>Std/Mean × 100</div></div>",unsafe_allow_html=True)
    if r is not None:
        with c5: st.markdown(f"<div class='kpi-card bg-gold'><div class='kpi-title'>Correlation (Feed ↔ Market)</div><div class='kpi-value'>{r:.3f}</div><div class='kpi-foot'>Pearson r</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    insights=[]
    if kpi_vol is not None: insights.append("ราคามีความผันผวนสูง (>10% CV) — ควรเฝ้าระวัง") if kpi_vol>10 else insights.append("ราคามีความผันผวนต่ำ — แนวโน้มคงที่")
    if r is not None:
        if r>0.5: insights.append("FeedPrice สัมพันธ์เชิงบวกกับราคาตลาด")
        elif r<-0.3: insights.append("FeedPrice สัมพันธ์เชิงลบกับราคาตลาด")
        else: insights.append("ความสัมพันธ์ Feed ↔ Market ไม่ชัดเจน")
    if insights:
        st.markdown("#### Dynamic Insights")
        for tip in insights: st.markdown(f"- {tip}")

st.markdown("<p style='text-align:center; color:#a0a0a0;'>© 2025 CPF Digital PMO | Layer-X Initiative</p>", unsafe_allow_html=True)
