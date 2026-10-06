import pandas as pd
import os

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/ev_charging_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values(
    ["timestamp", "station_id"]
).reset_index(drop=True)

print("=" * 60)
print("PREPARING ML DATASET")
print("=" * 60)

print(f"\nOriginal shape: {df.shape}")

# --------------------------------------------------
# FEATURES AVAILABLE BEFORE THE NEXT HOUR
# --------------------------------------------------

features = [
    "timestamp",
    "station_id",
    "location",
    "charger_count",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "temperature",
    "humidity",
    "rainfall",
    "previous_hour_demand",
    "previous_day_demand",
    "previous_week_demand",
    "rolling_24h_mean",
    "rolling_7d_mean",
    "energy_consumed_kwh"
]

df_model = df[features].copy()

# --------------------------------------------------
# CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")
print(df_model.isnull().sum().sum())

# --------------------------------------------------
# CHECK DUPLICATES
# --------------------------------------------------

print("\nDuplicate rows:")
print(df_model.duplicated().sum())

# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

output_path = "data/ev_charging_ml.csv"

df_model.to_csv(
    output_path,
    index=False
)

# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

print("\nML Dataset Shape:")
print(df_model.shape)

print("\nML Features:")
for column in df_model.columns:
    print("-", column)

print("\nTarget:")
print("energy_consumed_kwh")

print("\nTarget statistics:")
print(
    df_model["energy_consumed_kwh"].describe()
)

print(f"\nSaved to: {output_path}")

print("\nML dataset preparation completed!")