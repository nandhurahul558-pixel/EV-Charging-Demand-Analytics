import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/ev_charging_data.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("=" * 60)
print("EV CHARGING DEMAND - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. Basic information
# --------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

# --------------------------------------------------
# 2. Target distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["energy_consumed_kwh"],
    bins=50,
    kde=True
)

plt.title("Distribution of EV Charging Energy Demand")
plt.xlabel("Energy Consumed (kWh)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. Demand by hour
# --------------------------------------------------

hourly_demand = (
    df.groupby("hour")["energy_consumed_kwh"]
    .mean()
)

plt.figure(figsize=(10, 6))

hourly_demand.plot(
    marker="o"
)

plt.title("Average EV Charging Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Demand (kWh)")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4. Demand by day of week
# --------------------------------------------------

daily_demand = (
    df.groupby("day_of_week")["energy_consumed_kwh"]
    .mean()
)

plt.figure(figsize=(10, 6))

daily_demand.plot(
    kind="bar"
)

plt.title("Average Charging Demand by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Energy Demand (kWh)")

plt.xticks(
    range(7),
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ],
    rotation=45
)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Weekday vs weekend
# --------------------------------------------------

weekend_demand = (
    df.groupby("is_weekend")["energy_consumed_kwh"]
    .mean()
)

print("\nWeekday vs Weekend Demand:")
print(weekend_demand)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=["Weekday", "Weekend"],
    y=weekend_demand.values
)

plt.title("Weekday vs Weekend Charging Demand")
plt.ylabel("Average Energy Demand (kWh)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Demand by station
# --------------------------------------------------

station_demand = (
    df.groupby("station_id")["energy_consumed_kwh"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

station_demand.plot(
    kind="bar"
)

plt.title("Average Charging Demand by Station")
plt.xlabel("Station")
plt.ylabel("Average Energy Demand (kWh)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. Temperature vs demand
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="temperature",
    y="energy_consumed_kwh",
    alpha=0.3
)

plt.title("Temperature vs Charging Demand")
plt.xlabel("Temperature (°C)")
plt.ylabel("Energy Demand (kWh)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Rainfall vs demand
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="rainfall",
    y="energy_consumed_kwh",
    alpha=0.3
)

plt.title("Rainfall vs Charging Demand")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Energy Demand (kWh)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 9. Charger utilization vs demand
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="charger_utilization",
    y="energy_consumed_kwh",
    alpha=0.3
)

plt.title("Charger Utilization vs Energy Demand")
plt.xlabel("Charger Utilization")
plt.ylabel("Energy Demand (kWh)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. Correlation matrix
# --------------------------------------------------

numeric_columns = df.select_dtypes(
    include="number"
).columns

correlation = df[numeric_columns].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 11. Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)