
import pandas as pd
import numpy as np

# ================== LOAD FORECAST RESULTS ==================
train_df = pd.read_csv("train_data.csv")
test_df = train_df.copy()

# Convert ds (date) safely
possible_date_cols = ["ds", "Date", "date"]
for c in possible_date_cols:
    if c in train_df.columns:
        train_df.rename(columns={c: "ds"}, inplace=True)
        break
if "ds" in train_df.columns:
    train_df["ds"] = pd.to_datetime(train_df["ds"], errors="coerce")

# Auto-map potential Colab columns
col_map = {
    "PriceMarket_predicted": "yhat",
    "Forecast_mean_orig": "yhat",
    "Forecast_sum_orig": "yhat",
    "PriceMarket_orig": "PriceMarket",
    "Actual": "PriceMarket",
    "FeedPrice_mean_orig": "FeedPrice",
    "Trend_mean_orig": "trend",
    "Trend_sum_orig": "trend",
}
for old, new in col_map.items():
    if old in train_df.columns and new not in train_df.columns:
        train_df.rename(columns={old: new}, inplace=True)

# Ensure numeric
for col in ["yhat", "trend", "FeedPrice", "LayinghenPrice", "Stock", "PriceMarket", "y"]:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

# Fill mock data if missing
if "PriceMarket" not in train_df.columns or train_df["PriceMarket"].isna().all():
    if "y" in train_df.columns and not train_df["y"].isna().all():
        train_df["PriceMarket"] = train_df["y"]
    elif "yhat" in train_df.columns:
        train_df["PriceMarket"] = train_df["yhat"] * 0.98

if "Province" not in train_df.columns:
    provinces = ["Bangkok", "Chonburi", "Rayong", "Khon Kaen", "Chiang Mai", "Nakhon Pathom"]
    train_df["Province"] = np.random.choice(provinces, size=len(train_df))

if "Week" not in train_df.columns:
    train_df["Week"] = np.arange(1, len(train_df) + 1)

if "Month" not in train_df.columns and "ds" in train_df.columns:
    train_df["Month"] = train_df["ds"].dt.month

print("✅ Auto-detected columns (Export Mode):", list(train_df.columns))
