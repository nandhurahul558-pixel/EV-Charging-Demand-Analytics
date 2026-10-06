import pandas as pd
import os

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/ev_charging_ml.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

print("=" * 60)
print("TIME-BASED DATA SPLIT")
print("=" * 60)

# --------------------------------------------------
# DISPLAY DATE RANGE
# --------------------------------------------------

print("\nFull Dataset:")
print(f"Start: {df['timestamp'].min()}")
print(f"End  : {df['timestamp'].max()}")

# --------------------------------------------------
# DEFINE SPLIT DATES
# --------------------------------------------------

train_end = "2025-12-31 23:00:00"
validation_end = "2026-06-30 23:00:00"

# Training data
train = df[
    df["timestamp"] <= train_end
].copy()

# Validation data
validation = df[
    (df["timestamp"] > train_end)
    & (df["timestamp"] <= validation_end)
].copy()

# Test data
test = df[
    df["timestamp"] > validation_end
].copy()

# --------------------------------------------------
# DISPLAY SPLIT INFORMATION
# --------------------------------------------------

print("\nTRAINING SET")
print("-" * 30)
print(f"Rows : {len(train):,}")
print(f"Start: {train['timestamp'].min()}")
print(f"End  : {train['timestamp'].max()}")

print("\nVALIDATION SET")
print("-" * 30)
print(f"Rows : {len(validation):,}")
print(f"Start: {validation['timestamp'].min()}")
print(f"End  : {validation['timestamp'].max()}")

print("\nTEST SET")
print("-" * 30)
print(f"Rows : {len(test):,}")
print(f"Start: {test['timestamp'].min()}")
print(f"End  : {test['timestamp'].max()}")

# --------------------------------------------------
# VERIFY NO OVERLAP
# --------------------------------------------------

print("\nChecking for overlap...")

train_max = train["timestamp"].max()
validation_min = validation["timestamp"].min()

validation_max = validation["timestamp"].max()
test_min = test["timestamp"].min()

print(
    "Train → Validation:",
    train_max < validation_min
)

print(
    "Validation → Test:",
    validation_max < test_min
)

# --------------------------------------------------
# SAVE SPLITS
# --------------------------------------------------

os.makedirs("data/splits", exist_ok=True)

train.to_csv(
    "data/splits/train.csv",
    index=False
)

validation.to_csv(
    "data/splits/validation.csv",
    index=False
)

test.to_csv(
    "data/splits/test.csv",
    index=False
)

print("\nFiles saved:")
print("data/splits/train.csv")
print("data/splits/validation.csv")
print("data/splits/test.csv")

print("\n" + "=" * 60)
print("TIME-BASED SPLIT COMPLETED")
print("=" * 60)