
import pandas as pd
import numpy as np

train_df = pd.read_csv("train_data.csv")
test_df = train_df.copy()

if "ds" in train_df.columns:
    train_df["ds"] = pd.to_datetime(train_df["ds"], errors="coerce")

for col in ["yhat","trend","FeedPrice","LayinghenPrice","Stock","PriceMarket","y"]:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

if "PriceMarket" not in train_df.columns or train_df["PriceMarket"].isna().all():
    if "y" in train_df.columns and not train_df["y"].isna().all():
        train_df["PriceMarket"] = train_df["y"]
    elif "yhat" in train_df.columns:
        train_df["PriceMarket"] = train_df["yhat"] * 0.98

if "Province" not in train_df.columns:
    provinces = ["Bangkok","Chonburi","Rayong","Khon Kaen","Chiang Mai","Nakhon Pathom"]
    train_df["Province"] = np.random.choice(provinces, size=len(train_df))

if "Week" not in train_df.columns:
    train_df["Week"] = np.arange(1, len(train_df)+1)

if "Month" not in train_df.columns and "ds" in train_df.columns:
    train_df["Month"] = train_df["ds"].dt.month

print("✅ Forecast data loaded (v3.1). Columns:", list(train_df.columns))
