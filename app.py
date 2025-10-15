import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from PIL import Image
from layer_aibp_super_egg import train_df, test_df

# ================== CONFIG ==================
st.set_page_config(page_title="Layer-X Super Egg Forecast", page_icon="🥚", layout="wide")

logo = Image.open("LOGO-CPF.jpg")
st.sidebar.image(logo, width=150)
st.sidebar.markdown("### Layer-X Super Egg Dashboard")
st.sidebar.write("AI Forecasting powered by CPF Digital PMO")

st.markdown("<h2 style='text-align:center; color:#76d9c1;'>🥚 Layer-X Super Egg Forecast Dashboard</h2>", unsafe_allow_html=True)

# ================== MODEL ==================
model = Prophet()
model.add_regressor('Layingheny')
model.add_regressor('Feedy')
model.add_regressor('Stock')

train_df = train_df.rename(columns={'yMarket': 'y'})
model.fit(train_df)

future = model.make_future_dataframe(periods=52, freq='W')
future['Layingheny'] = train_df['Layingheny'].mean()
future['Feedy'] = train_df['Feedy'].mean()
future['Stock'] = train_df['Stock'].mean()
forecast = model.predict(future)

# ================== SIDEBAR ==================
min_date, max_date = forecast['ds'].min(), forecast['ds'].max()
date_range = st.sidebar.slider(
    "เลือกช่วงวันที่สำหรับดูผลการพยากรณ์",
    min_value=pd.to_datetime(min_date),
    max_value=pd.to_datetime(max_date),
    value=(pd.to_datetime(min_date), pd.to_datetime(max_date))
)
filtered_forecast = forecast[(forecast['ds'] >= date_range[0]) & (forecast['ds'] <= date_range[1])]

# ================== VISUALIZATION ==================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 52-Week Forecast", "📊 Actual vs Predicted", "📅 Yearly Trend",
    "🔗 Correlation", "🌍 Heatmap by Province", "🎞️ Weekly Animation"
])

# ---- (1) Forecast ----
with tab1:
    st.subheader("52 Week Egg Price Forecast")
    fig1 = px.line(filtered_forecast, x='ds', y='yhat', title='52-Week Egg Price Forecast', color_discrete_sequence=["#76d9c1"])
    st.plotly_chart(fig1, use_container_width=True)

# ---- (2) Actual vs Predicted ----
with tab2:
    st.subheader("Actual vs Predicted Egg Price")
    test_df = test_df.rename(columns={'PriceMarket': 'y'})
    test_future = test_df[['ds', 'LayinghenPrice', 'FeedPrice', 'Stock']].rename(columns={'LayinghenPrice': 'Layingheny', 'FeedPrice': 'Feedy'})
    test_pred = model.predict(test_future)
    compare_df = pd.DataFrame({'Date': test_df['ds'], 'Actual': test_df['y'], 'Predicted': test_pred['yhat']})
    fig2 = px.line(compare_df, x='Date', y=['Actual', 'Predicted'], title='Actual vs Predicted Egg Price',
                   color_discrete_sequence=["#f8cf40", "#76d9c1"])
    st.plotly_chart(fig2, use_container_width=True)

# ---- (3) Yearly Trend ----
with tab3:
    st.subheader("Yearly Trend Component")
    fig3 = px.line(forecast, x='ds', y='trend', title='Yearly Trend', color_discrete_sequence=["#cfd3da"])
    st.plotly_chart(fig3, use_container_width=True)

# ---- (4) Correlation ----
with tab4:
    st.subheader("FeedPrice vs MarketPrice Correlation")
    corr_value = train_df['Feedy'].corr(train_df['y'])
    st.write(f"**Correlation Coefficient (r)** = {corr_value:.3f}")
    fig_corr = px.scatter(train_df, x='Feedy', y='y', trendline='ols',
                          title='FeedPrice vs MarketPrice Correlation',
                          color_discrete_sequence=["#76d9c1"])
    st.plotly_chart(fig_corr, use_container_width=True)

# ---- (5) Heatmap ----
with tab5:
    st.subheader("Egg Price Heatmap by Province (Mock Data)")
    provinces = ['Bangkok', 'Chonburi', 'Khon Kaen', 'Chiang Mai', 'Rayong', 'Nakhon Pathom']
    price_matrix = np.random.uniform(3.0, 4.2, size=(len(provinces), 12))
    fig_heat = px.imshow(price_matrix,
                         x=[f'Month {i+1}' for i in range(12)],
                         y=provinces,
                         color_continuous_scale='YlGnBu',
                         labels={'color': 'Avg Price (THB)'},
                         title='Average Egg Price by Province')
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("※ ข้อมูลจำลองเพื่อแสดงตัวอย่าง Heatmap Layout")

# ---- (6) Animation ----
with tab6:
    st.subheader("Weekly Egg Price Movement (Animated)")
    if 'Week' in train_df.columns:
        fig_anim = px.line(train_df, x='ds', y='y', animation_frame='Week',
                           title='Animated Weekly Egg Price Movement',
                           color_discrete_sequence=["#f8cf40"])
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info("ไม่มีคอลัมน์ 'Week' ใน dataset — โปรดตรวจสอบไฟล์ train_df.")

# ---- Export ----
csv = filtered_forecast.to_csv(index=False)
st.download_button("🔽 Download Filtered Data as CSV", csv, "layer_super_egg_forecast.csv", "text/csv")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#a0a0a0;'>© 2025 CPF Digital PMO | Layer-X Initiative</p>", unsafe_allow_html=True)
