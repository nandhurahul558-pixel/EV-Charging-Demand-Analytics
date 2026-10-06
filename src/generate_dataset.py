import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

START_DATE = "2024-01-01"
END_DATE = "2026-09-30 23:00:00"

stations = {
    "ST001": {"location": "Kochi", "chargers": 8},
    "ST002": {"location": "Trivandrum", "chargers": 12},
    "ST003": {"location": "Kottayam", "chargers": 10},
    "ST004": {"location": "Kozhikode", "chargers": 15},
    "ST005": {"location": "Thrissur", "chargers": 6},
    "ST006": {"location": "Kannur", "chargers": 10},
    "ST007": {"location": "Alappuzha", "chargers": 14},
    "ST008": {"location": "Kollam", "chargers": 8},
}

# ---------------------------------------------------------
# CREATE CONTINUOUS HOURLY TIMESTAMPS
# ---------------------------------------------------------

timestamps = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="h"
)

# Create every timestamp for every station
records = []

for station_id, info in stations.items():

    for timestamp in timestamps:

        records.append({
            "timestamp": timestamp,
            "station_id": station_id,
            "location": info["location"],
            "charger_count": info["chargers"]
        })

df = pd.DataFrame(records)

print(f"Generated {len(df):,} hourly station records")

# ---------------------------------------------------------
# CALENDAR FEATURES
# ---------------------------------------------------------

df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# ---------------------------------------------------------
# HOLIDAYS
# ---------------------------------------------------------

holiday_dates = pd.to_datetime([
    "2024-01-01",
    "2024-01-26",
    "2024-05-01",
    "2024-08-15",
    "2024-10-02",
    "2024-12-25",

    "2025-01-01",
    "2025-01-26",
    "2025-05-01",
    "2025-08-15",
    "2025-10-02",
    "2025-12-25",

    "2026-01-01",
    "2026-01-26",
    "2026-05-01",
    "2026-08-15",
    "2026-10-02"
]).date

df["is_holiday"] = (
    df["date"].isin(holiday_dates)
).astype(int)

# ---------------------------------------------------------
# WEATHER
# ---------------------------------------------------------

monthly_temperature = {
    1: 27,
    2: 28,
    3: 30,
    4: 31,
    5: 30,
    6: 28,
    7: 27,
    8: 27,
    9: 28,
    10: 28,
    11: 28,
    12: 27
}

df["temperature"] = df["month"].map(
    monthly_temperature
)

# Daily temperature variation
df["temperature"] += (
    3 * np.sin(
        (df["hour"] - 6) * np.pi / 12
    )
)

df["temperature"] += np.random.normal(
    0,
    1.5,
    len(df)
)

df["temperature"] = df["temperature"].round(1)

# Humidity
df["humidity"] = np.random.normal(
    75,
    10,
    len(df)
)

df["humidity"] = np.clip(
    df["humidity"],
    40,
    100
).round(1)

# Rainfall
monsoon = df["month"].isin(
    [6, 7, 8, 9]
)

df["rainfall"] = np.where(
    monsoon,
    np.random.exponential(2.5, len(df)),
    np.random.exponential(0.5, len(df))
)

# Most hours have no rainfall
no_rain = np.random.random(len(df)) < 0.65

df.loc[no_rain, "rainfall"] = 0

df["rainfall"] = df["rainfall"].round(2)

# ---------------------------------------------------------
# DEMAND PATTERN
# ---------------------------------------------------------

def hour_multiplier(hour):

    if hour in [0, 1, 2, 3, 4, 5]:
        return 0.35

    elif hour in [6, 7, 8]:
        return 0.90

    elif hour in [9, 10, 11, 12, 13, 14, 15]:
        return 0.75

    elif hour in [16, 17]:
        return 1.15

    elif hour in [18, 19, 20, 21]:
        return 1.70

    elif hour in [22, 23]:
        return 1.10

    return 1.0


df["hour_multiplier"] = df["hour"].apply(
    hour_multiplier
)

# ---------------------------------------------------------
# STATION FACTOR
# ---------------------------------------------------------

station_factor = (
    df["charger_count"] / 10
)

# ---------------------------------------------------------
# WEEKEND EFFECT
# ---------------------------------------------------------

weekend_factor = np.where(
    df["is_weekend"] == 1,
    1.15,
    1.0
)

# ---------------------------------------------------------
# HOLIDAY EFFECT
# ---------------------------------------------------------

holiday_factor = np.where(
    df["is_holiday"] == 1,
    1.10,
    1.0
)

# ---------------------------------------------------------
# WEATHER EFFECT
# ---------------------------------------------------------

rain_factor = np.where(
    df["rainfall"] > 10,
    0.75,
    1.0
)

temperature_factor = np.where(
    df["temperature"] > 32,
    1.05,
    1.0
)

# ---------------------------------------------------------
# CHARGING SESSIONS
# ---------------------------------------------------------

base_sessions = np.random.poisson(
    7,
    len(df)
)

sessions = (
    base_sessions
    * station_factor
    * df["hour_multiplier"]
    * weekend_factor
    * holiday_factor
    * rain_factor
)

sessions += np.random.normal(
    0,
    1.5,
    len(df)
)

sessions = np.maximum(
    sessions,
    0
)

# Limit according to charger capacity
max_sessions = df["charger_count"] * 4

df["num_charging_sessions"] = np.minimum(
    sessions,
    max_sessions
).round().astype(int)

# ---------------------------------------------------------
# SESSION DURATION
# ---------------------------------------------------------

df["avg_session_duration"] = np.random.normal(
    42,
    10,
    len(df)
)

df["avg_session_duration"] += np.where(
    df["hour"].isin([18, 19, 20, 21]),
    5,
    0
)

df["avg_session_duration"] = np.clip(
    df["avg_session_duration"],
    15,
    90
).round(1)

# ---------------------------------------------------------
# ENERGY PER SESSION
# ---------------------------------------------------------

df["avg_energy_per_session"] = np.random.normal(
    12,
    3,
    len(df)
)

df["avg_energy_per_session"] = np.clip(
    df["avg_energy_per_session"],
    5,
    25
).round(2)

# ---------------------------------------------------------
# CHARGER UTILIZATION
# ---------------------------------------------------------

df["charger_utilization"] = (
    df["num_charging_sessions"]
    / (df["charger_count"] * 4)
)

df["charger_utilization"] = np.clip(
    df["charger_utilization"],
    0,
    1
).round(3)

# ---------------------------------------------------------
# ENERGY CONSUMPTION
# ---------------------------------------------------------

energy = (
    df["num_charging_sessions"]
    * df["avg_energy_per_session"]
)

energy *= temperature_factor

energy += np.random.normal(
    0,
    5,
    len(df)
)

df["energy_consumed_kwh"] = np.maximum(
    energy,
    0
).round(2)

# ---------------------------------------------------------
# SORT DATA
# ---------------------------------------------------------

df = df.sort_values(
    ["station_id", "timestamp"]
).reset_index(drop=True)

# ---------------------------------------------------------
# CREATE REAL HISTORICAL LAG FEATURES
# ---------------------------------------------------------

df["previous_hour_demand"] = (
    df.groupby("station_id")[
        "energy_consumed_kwh"
    ].shift(1)
)

df["previous_day_demand"] = (
    df.groupby("station_id")[
        "energy_consumed_kwh"
    ].shift(24)
)

df["previous_week_demand"] = (
    df.groupby("station_id")[
        "energy_consumed_kwh"
    ].shift(168)
)

# ---------------------------------------------------------
# ROLLING FEATURES
# ---------------------------------------------------------

df["rolling_24h_mean"] = (
    df.groupby("station_id")[
        "energy_consumed_kwh"
    ]
    .transform(
        lambda x: x.shift(1).rolling(24).mean()
    )
)

df["rolling_7d_mean"] = (
    df.groupby("station_id")[
        "energy_consumed_kwh"
    ]
    .transform(
        lambda x: x.shift(1).rolling(168).mean()
    )
)

# ---------------------------------------------------------
# REMOVE ROWS WITHOUT HISTORICAL DATA
# ---------------------------------------------------------

df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# REMOVE HELPER COLUMN
# ---------------------------------------------------------

df.drop(
    columns=["hour_multiplier"],
    inplace=True
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

os.makedirs(
    "data",
    exist_ok=True
)

output_path = "data/ev_charging_data.csv"

df.to_csv(
    output_path,
    index=False
)

# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EV CHARGING DATASET CREATED")
print("=" * 60)

print(f"\nRows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nTarget statistics:")
print(
    df["energy_consumed_kwh"].describe()
)

print("\nStation distribution:")
print(
    df["station_id"].value_counts()
)

print(f"\nSaved to: {output_path}")

print("\nDataset generation completed successfully!")