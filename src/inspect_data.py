import pandas as pd

# Load dataset
df = pd.read_csv("data/ev_charging_data.csv")

print("=" * 60)
print("EV CHARGING DATASET INSPECTION")
print("=" * 60)

# 1. Shape
print("\n1. DATASET SHAPE")
print("-" * 30)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# 2. Columns
print("\n2. COLUMNS")
print("-" * 30)

for column in df.columns:
    print(column)

# 3. First 5 rows
print("\n3. FIRST 5 ROWS")
print("-" * 30)
print(df.head())

# 4. Data types
print("\n4. DATA TYPES")
print("-" * 30)
print(df.dtypes)

# 5. Missing values
print("\n5. MISSING VALUES")
print("-" * 30)
print(df.isnull().sum())

# 6. Duplicate rows
print("\n6. DUPLICATE ROWS")
print("-" * 30)
print(df.duplicated().sum())

# 7. Statistical summary
print("\n7. STATISTICAL SUMMARY")
print("-" * 30)
print(df.describe())

# 8. Categorical information
print("\n8. STATIONS")
print("-" * 30)
print(df["station_id"].value_counts())

# 9. Target statistics
print("\n9. ENERGY CONSUMPTION")
print("-" * 30)
print(df["energy_consumed_kwh"].describe())

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)