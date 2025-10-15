import pandas as pd

# โหลดผลลัพธ์ที่ได้จาก Colab (เช่นไฟล์นี้)
train_df = pd.read_csv("train_data.csv")  # หรือชื่อไฟล์ forecast ที่คุณใช้
test_df = train_df.copy()

# ปรับให้ชื่อคอลัมน์สอดคล้องกับโครงสร้างที่ app.py ใช้
if "yhat" in train_df.columns:
    train_df = train_df.rename(columns={"yhat": "PriceMarket_predicted"})
if "ds" in train_df.columns:
    train_df["ds"] = pd.to_datetime(train_df["ds"], errors="coerce")

# mock ค่า target จริง (เพราะ forecast ไม่มี “y” จริง)
train_df["PriceMarket"] = train_df["PriceMarket_predicted"] * 0.98

# ส่งออก DataFrame ให้ Streamlit ใช้งาน
print("✅ Forecast data loaded successfully for visualization.")
