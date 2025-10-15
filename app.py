
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from layer_aibp_super_egg import train_df

# ================== CONFIG ==================
st.set_page_config(page_title="Layer-X Super Egg – War Room", page_icon="🥚", layout="wide")

# Sidebar Branding
try:
    logo = Image.open("LOGO-CPF.jpg")
    st.sidebar.image(logo, width=150)
except Exception:
    st.sidebar.write("CPF")
st.sidebar.markdown("### Layer-X Super Egg Dashboard (v3)")
st.sidebar.write("Executive War Room View + Analytics")

st.markdown("""
<style>
.kpi-card {border-radius: 16px; padding: 18px 18px 12px; margin: 6px;
           box-shadow: 0 4px 16px rgba(0,0,0,0.06);}
.kpi-title {font-size: 0.85rem; color:#2f3b3f; margin-bottom: 6px;}
.kpi-value {font-size: 1.6rem; font-weight: 700; margin-bottom: 4px;}
.kpi-foot {font-size: 0.8rem; color:#6b6f72;}
.bg-mint {background:#e8fbf6;}    /* #76d9c1 tint */
.bg-gold {background:#fff6d6;}    /* #f8cf40 tint */
.bg-silver {background:#eef2f5;}  /* #cfd3da tint */
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color:#76d9c1;'>🥚 Layer-X Super Egg – War Room v3</h2>", unsafe_allow_html=True)

# ================== RANGE FILTER ==================
df = train_df.copy()

# ensure datetime dtype for ds column
if "ds" in df.columns:
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

if "ds" in df.columns and df["ds"].notna().any():
    # get min/max date and convert to pure date type
    min_date = pd.to_datetime(df["ds"].min()).to_pydatetime().date()
    max_date = pd.to_datetime(df["ds"].max()).to_pydatetime().date()

    # create a safe slider
    date_range = st.sidebar.slider(
        "เลือกช่วงวันที่",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # convert slider output (date) back to Timestamp for filtering
    df = df[
        (df["ds"] >= pd.Timestamp(date_range[0])) &
        (df["ds"] <= pd.Timestamp(date_range[1]))
    ].copy()
else:
    st.sidebar.info("ไม่พบคอลัมน์วันที่ (ds) หรือข้อมูลไม่ถูกต้อง")


# ================== KPI COMPUTE ==================
has_price = "PriceMarket" in df.columns
kpi_avg = float(df["PriceMarket"].mean()) if has_price else None
kpi_max = float(df["PriceMarket"].max()) if has_price else None
kpi_min = float(df["PriceMarket"].min()) if has_price else None
kpi_vol = float((df["PriceMarket"].std() / df["PriceMarket"].mean()) * 100) if has_price and df["PriceMarket"].mean() else None

# Correlation
r = None
if has_price and "FeedPrice" in df.columns:
    corr = df[["FeedPrice", "PriceMarket"]].dropna()
    if len(corr) >= 2:
        r = float(corr["FeedPrice"].corr(corr["PriceMarket"]))

# ================== TABS ==================
tabs = st.tabs([
    "🏛️ War Room Summary",
    "📈 52-Week Forecast",
    "📊 Actual vs Predicted",
    "📅 Yearly Trend",
    "🔗 Correlation (Feed vs Market)",
    "🌍 Heatmap by Province",
    "🎞️ Animated Line (Weekly)"
])

# ---- (0) War Room Summary ----
with tabs[0]:
    st.subheader("Executive KPI Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='kpi-card bg-mint'>
          <div class='kpi-title'>Avg Price</div>
          <div class='kpi-value'>{kpi_avg:.2f} THB</div>
          <div class='kpi-foot'>ช่วงที่เลือก</div>
        </div>""", unsafe_allow_html=True) if kpi_avg is not None else st.info("ไม่มี PriceMarket")

    with c2:
        st.markdown(f"""
        <div class='kpi-card bg-gold'>
          <div class='kpi-title'>Max Price</div>
          <div class='kpi-value'>{kpi_max:.2f} THB</div>
          <div class='kpi-foot'>ค่าสูงสุด</div>
        </div>""", unsafe_allow_html=True) if kpi_max is not None else st.empty()

    with c3:
        st.markdown(f"""
        <div class='kpi-card bg-silver'>
          <div class='kpi-title'>Min Price</div>
          <div class='kpi-value'>{kpi_min:.2f} THB</div>
          <div class='kpi-foot'>ค่าต่ำสุด</div>
        </div>""", unsafe_allow_html=True) if kpi_min is not None else st.empty()

    with c4:
        st.markdown(f"""
        <div class='kpi-card bg-mint'>
          <div class='kpi-title'>Volatility (CV)</div>
          <div class='kpi-value'>{kpi_vol:.2f}%</div>
          <div class='kpi-foot'>Std/Mean × 100</div>
        </div>""", unsafe_allow_html=True) if kpi_vol is not None else st.empty()

    with c5:
        if r is not None:
            st.markdown(f"""
            <div class='kpi-card bg-gold'>
              <div class='kpi-title'>Correlation (Feed ↔ Market)</div>
              <div class='kpi-value'>{r:.3f}</div>
              <div class='kpi-foot'>Pearson r</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("ต้องการคอลัมน์ 'FeedPrice' และ 'PriceMarket' เพื่อคำนวณ Correlation")

    st.markdown("---")

    # Quick Insights
    insights = []
    if kpi_vol is not None:
        if kpi_vol > 10:
            insights.append("ราคามีความผันผวนสูง (>10% CV) — ควรเฝ้าระวังการตั้งราคาและสต็อก")
        else:
            insights.append("ราคามีความผันผวนต่ำ — แนวโน้มคงที่")
    if r is not None:
        if r > 0.5:
            insights.append("FeedPrice สัมพันธ์เชิงบวกกับราคาตลาดค่อนข้างมาก")
        elif r < -0.3:
            insights.append("FeedPrice สัมพันธ์เชิงลบกับราคาตลาด")
        else:
            insights.append("ความสัมพันธ์ Feed ↔ Market ไม่ชัดเจน")

    if insights:
        st.markdown("#### Dynamic Insights")
        for tip in insights:
            st.markdown(f"- {tip}")

# ---- (1) 52-Week Forecast ----
with tabs[1]:
    st.subheader("52 Week Egg Price Forecast")
    if "yhat" in df.columns and "ds" in df.columns:
        fig1 = px.line(df, x="ds", y="yhat", title="52-Week Forecast", color_discrete_sequence=["#76d9c1"])
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ 'yhat' หรือ 'ds' ในข้อมูล")

# ---- (2) Actual vs Predicted ----
with tabs[2]:
    st.subheader("Actual vs Predicted Egg Price")
    y_cols = [c for c in ["PriceMarket", "yhat"] if c in df.columns]
    if "ds" in df.columns and y_cols:
        fig2 = px.line(df, x="ds", y=y_cols, title="Actual vs Predicted",
                       color_discrete_sequence=["#f8cf40", "#76d9c1"])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ต้องมี 'ds' และอย่างน้อยหนึ่งใน ['PriceMarket','yhat']")

# ---- (3) Yearly Trend ----
with tabs[3]:
    st.subheader("Yearly Trend Component")
    if "trend" in df.columns and "ds" in df.columns:
        fig3 = px.line(df, x="ds", y="trend", title="Yearly Trend", color_discrete_sequence=["#cfd3da"])
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ 'trend'")

# ---- (4) Correlation (Feed vs Market) ----
with tabs[4]:
    st.subheader("FeedPrice vs PriceMarket (Correlation)")
    if "FeedPrice" in df.columns and "PriceMarket" in df.columns:
        corr = df[["FeedPrice", "PriceMarket"]].dropna()
        if len(corr) >= 2:
            r = corr["FeedPrice"].corr(corr["PriceMarket"])
            st.write(f"**Correlation Coefficient (Pearson r)** = {r:.3f}")
            fig_corr = px.scatter(corr, x="FeedPrice", y="PriceMarket", trendline="ols",
                                  title="FeedPrice vs PriceMarket Correlation",
                                  color_discrete_sequence=["#76d9c1"])
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("ข้อมูลไม่เพียงพอสำหรับคำนวณค่าสหสัมพันธ์")
    else:
        st.info("ต้องการคอลัมน์ 'FeedPrice' และ 'PriceMarket'")

# ---- (5) Heatmap by Province ----
with tabs[5]:
    st.subheader("Egg Price Heatmap by Province")
    if "Province" in df.columns and "PriceMarket" in df.columns:
        if "Month" not in df.columns and "ds" in df.columns:
            df["Month"] = df["ds"].dt.month
        if "Month" in df.columns:
            pivot = df.pivot_table(index="Province", columns="Month", values="PriceMarket", aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            fig_heat = px.imshow(pivot.values,
                                 x=[f"Month {m}" for m in pivot.columns],
                                 y=pivot.index,
                                 labels={"x": "Month", "y": "Province", "color": "Avg Price"},
                                 title="Average Egg Price by Province (Monthly)",
                                 color_continuous_scale="YlGnBu")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("ไม่พบคอลัมน์ 'Month'")
    else:
        st.info("ต้องการคอลัมน์ 'Province' และ 'PriceMarket'")

# ---- (6) Animated Line (Weekly) ----
with tabs[6]:
    st.subheader("Animated Weekly Price")
    if "Week" in df.columns and "PriceMarket" in df.columns and "ds" in df.columns:
        fig_anim = px.line(df, x="ds", y="PriceMarket", animation_frame="Week",
                           title="Animated Weekly Egg Price",
                           color_discrete_sequence=["#f8cf40"])
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info("ต้องการคอลัมน์ 'Week', 'ds' และ 'PriceMarket'")

# ---- Export CSV ----
st.markdown("---")
csv = df.to_csv(index=False)
st.download_button("🔽 Download Filtered Data", csv, "warroom_filtered.csv", "text/csv")

st.markdown("<p style='text-align:center; color:#a0a0a0;'>© 2025 CPF Digital PMO | Layer-X Initiative</p>", unsafe_allow_html=True)
