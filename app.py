
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from datetime import datetime
from PIL import Image
from io import BytesIO

# PDF building
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

from layer_aibp_super_egg import train_df

st.set_page_config(page_title="Layer-X Super Egg – War Room", page_icon="🥚", layout="wide")

# Sidebar Branding
try:
    logo = Image.open("LOGO-CPF.jpg")
    st.sidebar.image(logo, width=150)
except Exception:
    logo = None
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

st.markdown("<h2 style='text-align:center; color:#76d9c1;'>🥚 Layer-X Super Egg – War Room</h2>", unsafe_allow_html=True)

# ================== RANGE FILTER ==================
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

# ================== KPI COMPUTE ==================
has_price = "PriceMarket" in df.columns
kpi_avg = float(df["PriceMarket"].mean()) if has_price else None
kpi_max = float(df["PriceMarket"].max()) if has_price else None
kpi_min = float(df["PriceMarket"].min()) if has_price else None
kpi_vol = float((df["PriceMarket"].std() / df["PriceMarket"].mean()) * 100) if has_price and df["PriceMarket"].mean() else None

r=None
if has_price and "FeedPrice" in df.columns:
    corr=df[["FeedPrice","PriceMarket"]].dropna()
    if len(corr)>=2: r=float(corr["FeedPrice"].corr(corr["PriceMarket"]))

# ================== TABS ==================
tabs=st.tabs(["🏛️ War Room Summary","📈 52-Week Forecast","📊 Actual vs Predicted","📅 Yearly Trend","🔗 Correlation (Feed vs Market)","🌍 Heatmap by Province"])

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
    if kpi_vol is not None:
        insights.append("ราคามีความผันผวนสูง (>10% CV) — ควรเฝ้าระวัง") if kpi_vol>10 else insights.append("ราคามีความผันผวนต่ำ — แนวโน้มคงที่")
    if r is not None:
        if r>0.5: insights.append("FeedPrice สัมพันธ์เชิงบวกกับราคาตลาด")
        elif r<-0.3: insights.append("FeedPrice สัมพันธ์เชิงลบกับราคาตลาด")
        else: insights.append("ความสัมพันธ์ Feed ↔ Market ไม่ชัดเจน")
    if insights:
        st.markdown("#### Dynamic Insights")
        for tip in insights: st.markdown(f"- {tip}")

# ========== BUILD FIGURES (for UI & PDF) ==========
def build_figures(dataframe):
    figs = {}

    # 1) Forecast
    if "yhat" in dataframe.columns and "ds" in dataframe.columns:
        figs["forecast"] = px.line(dataframe, x="ds", y="yhat", title="52-Week Forecast", color_discrete_sequence=["#76d9c1"])

    # 2) Actual vs Predicted
    y_cols = [c for c in ["PriceMarket", "yhat"] if c in dataframe.columns]
    if "ds" in dataframe.columns and y_cols:
        figs["actual_vs_pred"] = px.line(dataframe, x="ds", y=y_cols, title="Actual vs Predicted",
                                         color_discrete_sequence=["#f8cf40", "#76d9c1"])

    # 3) Yearly Trend
    if "trend" in dataframe.columns and "ds" in dataframe.columns:
        figs["trend"] = px.line(dataframe, x="ds", y="trend", title="Yearly Trend", color_discrete_sequence=["#cfd3da"])

    # 4) Correlation
    if "FeedPrice" in dataframe.columns and "PriceMarket" in dataframe.columns:
        corr_df = dataframe[["FeedPrice","PriceMarket"]].dropna()
        if len(corr_df) >= 2:
            figs["correlation"] = px.scatter(corr_df, x="FeedPrice", y="PriceMarket", trendline="ols",
                                             title="FeedPrice vs PriceMarket Correlation",
                                             color_discrete_sequence=["#76d9c1"])

    # 5) Heatmap
    if "Province" in dataframe.columns and "PriceMarket" in dataframe.columns:
        df_hm = dataframe.copy()
        if "Month" not in df_hm.columns and "ds" in df_hm.columns:
            df_hm["Month"] = df_hm["ds"].dt.month
        if "Month" in df_hm.columns:
            pivot = df_hm.pivot_table(index="Province", columns="Month", values="PriceMarket", aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            figs["heatmap"] = px.imshow(pivot.values,
                                        x=[f"Month {m}" for m in pivot.columns],
                                        y=pivot.index,
                                        labels={"x": "Month", "y": "Province", "color": "Avg Price"},
                                        title="Average Egg Price by Province (Monthly)",
                                        color_continuous_scale="YlGnBu")
    return figs

figs = build_figures(df)

# Show other tabs with figures
with tabs[1]:
    if "forecast" in figs: st.plotly_chart(figs["forecast"], use_container_width=True)
    else: st.info("ไม่พบคอลัมน์ 'yhat' หรือ 'ds' ในข้อมูล")

with tabs[2]:
    if "actual_vs_pred" in figs: st.plotly_chart(figs["actual_vs_pred"], use_container_width=True)
    else: st.info("ต้องมี 'ds' และอย่างน้อยหนึ่งใน ['PriceMarket','yhat']")

with tabs[3]:
    if "trend" in figs: st.plotly_chart(figs["trend"], use_container_width=True)
    else: st.info("ไม่พบคอลัมน์ 'trend'")

with tabs[4]:
    if "correlation" in figs: st.plotly_chart(figs["correlation"], use_container_width=True)
    else: st.info("ต้องการคอลัมน์ 'FeedPrice' และ 'PriceMarket'")

with tabs[5]:
    if "heatmap" in figs: st.plotly_chart(figs["heatmap"], use_container_width=True)
    else: st.info("ต้องการคอลัมน์ 'Province' และ 'PriceMarket'")

# ========== EXPORT PDF HANDLER ==========
def fig_to_png_bytes(fig, width=1200, height=700, scale=1.0):
    # Requires kaleido
    return pio.to_image(fig, format="png", width=width, height=height, scale=scale)

def build_pdf(dataframe, figs_dict, logo_img):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # ลงทะเบียนฟอนต์ไทย
    pdfmetrics.registerFont(TTFont("Sarabun-Regular.ttf", "Sarabun-Bold.ttf"))

    # Prepare PDF in memory
    buf = BytesIO()
    c = Canvas(buf, pagesize=landscape(A4))
    width, height = landscape(A4)

    # Common header/footer drawer
    def header_footer(page_title, page_num, total_pages):
        # Header background (subtle)
        c.setFillColorRGB(0.91, 0.97, 0.96)  # mint tint
        c.rect(0, height-60, width, 60, fill=1, stroke=0)
        # Logo
        if logo_img is not None:
            c.drawImage(ImageReader(logo_img), 30, height-50, width=80, height=40, preserveAspectRatio=True, mask='auto')
        # Title
        c.setFillColor(colors.HexColor("#2f3b3f"))
        c.setFont("Sarabun", 16)
        c.drawString(120, height-40, "Layer-X Super Egg War Room Summary")
        # Page number
        c.setFont("Sarabun", 10)
        c.setFillColor(colors.HexColor("#6b6f72"))
        c.drawRightString(width-30, 20, f"Page {page_num} of {total_pages}")
        # Footer line
        c.setStrokeColor(colors.HexColor("#eef2f5"))
        c.setLineWidth(1)
        c.line(20, 50, width-20, 50)
        # Footer text
        c.setFont("Sarabun", 9)
        c.setFillColor(colors.HexColor("#6b6f72"))
        c.drawString(30, 35, "© CPF Digital PMO | Layer-X Initiative")

    # Pre-calc pages = 1 (KPI) + number of figs
    total_pages = 1 + len(figs_dict)

    # ---- Page 1: KPI + Insights
    header_footer("KPI Summary", 1, total_pages)
    y = height - 90

    # KPI cards as colored boxes
    def kpi_card(x, y, w, h, title, value, color_hex):
        c.setFillColor(colors.HexColor(color_hex))
        c.roundRect(x, y-h, w, h, 10, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#2f3b3f"))
        c.setFont("Sarabun", 10); c.drawString(x+12, y-24, title)
        c.setFont("Sarabun-Bold", 18); c.drawString(x+12, y-48, value)

    card_w, card_h = 210, 70
    gap = 15
    x0 = 30
    if dataframe is not None:
        avg_txt = f"{np.nanmean(dataframe['PriceMarket']):.2f} THB" if "PriceMarket" in dataframe.columns else "N/A"
        max_txt = f"{np.nanmax(dataframe['PriceMarket']):.2f} THB" if "PriceMarket" in dataframe.columns else "N/A"
        min_txt = f"{np.nanmin(dataframe['PriceMarket']):.2f} THB" if "PriceMarket" in dataframe.columns else "N/A"
        vol_txt = f"{(np.nanstd(dataframe['PriceMarket'])/np.nanmean(dataframe['PriceMarket'])*100):.2f}%" if "PriceMarket" in dataframe.columns and np.nanmean(dataframe['PriceMarket']) else "N/A"
        corr_txt = "N/A"
        if "FeedPrice" in dataframe.columns and "PriceMarket" in dataframe.columns:
            cdf = dataframe[["FeedPrice","PriceMarket"]].dropna()
            if len(cdf)>=2:
                corr_txt = f"{cdf['FeedPrice'].corr(cdf['PriceMarket']):.3f}"

        kpi_card(x0, y, card_w, card_h, "Avg Price", avg_txt, "#e8fbf6")
        kpi_card(x0+card_w+gap, y, card_w, card_h, "Max Price", max_txt, "#fff6d6")
        kpi_card(x0+2*(card_w+gap), y, card_w, card_h, "Min Price", min_txt, "#eef2f5")
        kpi_card(x0+3*(card_w+gap), y, card_w, card_h, "Volatility (CV)", vol_txt, "#e8fbf6")
        kpi_card(x0+4*(card_w+gap), y, card_w, card_h, "Correlation (r)", corr_txt, "#fff6d6")

    # Insights
    c.setFont("Sarabun-Bold", 12); c.setFillColor(colors.HexColor("#2f3b3f"))
    c.drawString(30, y-90, "Dynamic Insights")
    c.setFont("Sarabun", 10); c.setFillColor(colors.HexColor("#2f3b3f"))
    insights = []
    if "PriceMarket" in dataframe.columns and np.nanmean(dataframe["PriceMarket"]) and (np.nanstd(dataframe["PriceMarket"])/np.nanmean(dataframe["PriceMarket"]))*100 > 10:
        insights.append("ราคามีความผันผวนสูง (>10% CV) — ควรเฝ้าระวัง")
    else:
        insights.append("ราคามีความผันผวนต่ำ — แนวโน้มคงที่")
    if "FeedPrice" in dataframe.columns and "PriceMarket" in dataframe.columns:
        cdf = dataframe[["FeedPrice","PriceMarket"]].dropna()
        if len(cdf)>=2:
            r = cdf["FeedPrice"].corr(cdf["PriceMarket"])
            if r>0.5: insights.append("FeedPrice สัมพันธ์เชิงบวกกับราคาตลาด")
            elif r<-0.3: insights.append("FeedPrice สัมพันธ์เชิงลบกับราคาตลาด")
            else: insights.append("ความสัมพันธ์ Feed ↔ Market ไม่ชัดเจน")
    y_text = y-110
    for tip in insights:
        c.drawString(40, y_text, f"• {tip}")
        y_text -= 14

    c.showPage()

    # ---- Subsequent pages: figures
    page_titles = {
        "forecast": "52-Week Forecast",
        "actual_vs_pred": "Actual vs Predicted",
        "trend": "Yearly Trend",
        "correlation": "Feed vs Market Correlation",
        "heatmap": "Price Heatmap by Province",
    }

    # Precompute total pages with available figs
    keys = [k for k in ["forecast","actual_vs_pred","trend","correlation","heatmap"] if k in figs_dict]
    total_pages = 1 + len(keys)

    page_num = 2
    for key in keys:
        header_footer(page_titles.get(key, key), page_num, total_pages)
        fig = figs_dict[key]
        # Export fig to PNG bytes
        img_bytes = fig_to_png_bytes(fig, width=1400, height=800, scale=1.0)
        img = ImageReader(BytesIO(img_bytes))
        # Draw image to fit margins
        margin = 30
        avail_w = width - 2*margin
        avail_h = height - 2*margin - 60  # header space
        c.drawImage(img, margin, 60, width=avail_w, height=avail_h, preserveAspectRatio=True, anchor='c')
        c.showPage()
        page_num += 1

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

st.markdown("---")
if st.button("📄 Export War Room Report (PDF)"):
    try:
        pdf_bytes = build_pdf(df, figs, logo)
        today = datetime.now().strftime("%Y-%m-%d")
        st.download_button("Download PDF", data=pdf_bytes, file_name=f"CPF_Egg_WarRoom_{today}.pdf", mime="application/pdf")
        st.success("สร้างไฟล์ PDF สำเร็จ")
    except Exception as e:
        st.error(f"ไม่สามารถสร้าง PDF ได้: {e} (ตรวจสอบว่าได้ติดตั้ง kaleido และ reportlab เรียบร้อยใน requirements.txt)")

st.markdown("<p style='text-align:center; color:#a0a0a0;'>© 2025 CPF Digital PMO | Layer-X Initiative</p>", unsafe_allow_html=True)
