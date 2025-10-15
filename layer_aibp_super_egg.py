import pandas as pd
from prophet import Prophet

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# ---- ตรวจสอบคอลัมน์ก่อน rename ----
print("Columns found in train_df:", train_df.columns)

# ---- Rename ให้ตรงชื่อ ----
rename_map = {
    "Date": "ds",
    "PriceMarket": "y",
    "LayinghenPrice": "Layingheny",
    "FeedPrice": "Feedy"
}
train_df = train_df.rename(columns=rename_map)
test_df = test_df.rename(columns=rename_map)

# ---- แปลงวันที่ และกรองค่า NA หลัง rename ----
train_df["ds"] = pd.to_datetime(train_df["ds"], errors="coerce")
test_df["ds"] = pd.to_datetime(test_df["ds"], errors="coerce")
train_df = train_df.dropna(subset=["ds", "y"])

# ---- Train model ----
model = Prophet()
for reg in ["Layingheny", "Feedy", "Stock"]:
    if reg in train_df.columns:
        model.add_regressor(reg)
model.fit(train_df)

print("✅ Model fitted successfully with columns:", train_df.columns)
