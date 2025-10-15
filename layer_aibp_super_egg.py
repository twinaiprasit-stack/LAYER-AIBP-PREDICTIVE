# layer_aibp_super_egg.py
import pandas as pd
from prophet import Prophet

# ===========================================================
# 1. LOAD REAL DATA (ใช้ไฟล์ในโฟลเดอร์เดียวกัน)
# ===========================================================
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# ===========================================================
# 2. PREPARE DATA FOR PROPHET
# ===========================================================
# ปรับชื่อคอลัมน์ให้ Prophet เข้าใจ
rename_map = {
    "Date": "ds",
    "PriceMarket": "y",
    "LayinghenPrice": "Layingheny",
    "FeedPrice": "Feedy"
}

train_df = train_df.rename(columns=rename_map)
test_df = test_df.rename(columns=rename_map)

# แปลงวันที่ให้เป็น datetime
train_df["ds"] = pd.to_datetime(train_df["ds"], errors="coerce")
test_df["ds"] = pd.to_datetime(test_df["ds"], errors="coerce")

# กรองแถวที่ไม่มี ds หรือ y
train_df = train_df.dropna(subset=["ds", "y"])

# ===========================================================
# 3. TRAIN MODEL
# ===========================================================
model = Prophet()
for reg in ["Layingheny", "Feedy", "Stock"]:
    if reg in train_df.columns:
        model.add_regressor(reg)

model.fit(train_df)

# ===========================================================
# 4. CREATE FUTURE DATAFRAME (52 สัปดาห์ข้างหน้า)
# ===========================================================
future = model.make_future_dataframe(periods=52, freq="W")

# ใส่ค่าเฉลี่ยของ regressor สำหรับอนาคต
for reg in ["Layingheny", "Feedy", "Stock"]:
    if reg in train_df.columns:
        future[reg] = train_df[reg].mean()

# ===========================================================
# 5. PREDICT
# ===========================================================
forecast = model.predict(future)

# เตรียม test_future สำหรับเปรียบเทียบจริง
test_future = test_df[["ds", "Layingheny", "Feedy", "Stock"]].copy()
test_forecast = model.predict(test_future)

# ===========================================================
# 6. EXPORT RESULTS (ให้ Streamlit นำไปใช้)
# ===========================================================
# train_df, test_df คือข้อมูลจริง
# forecast คือผลพยากรณ์อนาคต
# test_forecast คือผลเทียบกับข้อมูลทดสอบจริง
print("✅ Model trained successfully with real data.")
