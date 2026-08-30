import os
import numpy as np
import pandas as pd


INPUT_PATH = "data/raw/karachi_daily_aqi_weather.csv"

OUTPUT_PATH = "data/processed/karachi_aqi_features.csv"



print("\nLoading raw data...")

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"Original shape: {df.shape}")

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print("\nCreating time features...")

df["day"] = df["date"].dt.day

df["month"] = df["date"].dt.month

df["day_of_week"] = df["date"].dt.dayofweek

df["day_of_year"] = df["date"].dt.dayofyear

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_of_year_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365
)

df["day_of_year_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365
)

print("Creating lag features...")

lag_columns = [
    "AQI",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity"
]

lags = [
    1,
    2,
    3,
    7,
    14,
    21,
    30
]

for column in lag_columns:

    for lag in lags:

        df[f"{column}_lag_{lag}"] = (
            df[column].shift(lag)
        )



print("Creating rolling features...")

rolling_columns = [
    "AQI",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3"
]

windows = [
    3,
    7,
    14,
    30
]

for column in rolling_columns:

    for window in windows:

        # Mean
        df[f"{column}_rolling_mean_{window}"] = (
            df[column]
            .rolling(window)
            .mean()
        )

        # Standard deviation
        df[f"{column}_rolling_std_{window}"] = (
            df[column]
            .rolling(window)
            .std()
        )

        # Minimum
        df[f"{column}_rolling_min_{window}"] = (
            df[column]
            .rolling(window)
            .min()
        )

        # Maximum
        df[f"{column}_rolling_max_{window}"] = (
            df[column]
            .rolling(window)
            .max()
        )



print("Creating change features...")

pollutants = [
    "AQI",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3"
]

for column in pollutants:

    df[f"{column}_change_1"] = (
        df[column].diff(1)
    )

    df[f"{column}_change_3"] = (
        df[column].diff(3)
    )

    df[f"{column}_change_7"] = (
        df[column].diff(7)
    )

    df[f"{column}_pct_change_1"] = (
        df[column].pct_change(1) * 100
    )

    df[f"{column}_pct_change_7"] = (
        df[column].pct_change(7) * 100
    )

df["AQI_momentum_3"] = (
    df["AQI"] -
    df["AQI"].shift(3)
)

df["AQI_momentum_7"] = (
    df["AQI"] -
    df["AQI"].shift(7)
)

df["AQI_momentum_14"] = (
    df["AQI"] -
    df["AQI"].shift(14)
)

df["PM25_PM10_ratio"] = (
    df["PM2.5"] /
    (df["PM10"] + 1e-6)
)

df["NO2_O3_ratio"] = (
    df["NO2"] /
    (df["O3"] + 1e-6)
)

df["CO_NO2_ratio"] = (
    df["CO"] /
    (df["NO2"] + 1e-6)
)


df["Temp_Humidity_interaction"] = (
    df["Temperature"] *
    df["Humidity"]
)

df["PM25_Humidity_interaction"] = (
    df["PM2.5"] *
    df["Humidity"]
)

df["PM10_Humidity_interaction"] = (
    df["PM10"] *
    df["Humidity"]
)



print("Creating 3-day forecast targets...")

df["AQI_t+1"] = (
    df["AQI"].shift(-1)
)

df["AQI_t+2"] = (
    df["AQI"].shift(-2)
)

df["AQI_t+3"] = (
    df["AQI"].shift(-3)
)



if "Next_Day_AQI" in df.columns:

    df = df.drop(
        columns=["Next_Day_AQI"]
    )



print("Cleaning infinite values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_columns = [
    column
    for column in df.columns
    if column not in [
        "date",
        "AQI_t+1",
        "AQI_t+2",
        "AQI_t+3"
    ]
]

before = len(df)

df = df.dropna(
    subset=feature_columns
).reset_index(drop=True)

after = len(df)

print(
    f"Removed {before - after} rows "
    f"because of missing input features."
)

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)



print("\n========================================")
print("FEATURE ENGINEERING COMPLETE")
print("========================================")

print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nTarget columns:")

print([
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
])

print("\nSaved to:")

print(OUTPUT_PATH)

print("\nFinal date range:")

print(
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)



print("\n========================================")
print("LATEST DATA CHECK")
print("========================================")

print(
    df[
        [
            "date",
            "AQI",
            "AQI_t+1",
            "AQI_t+2",
            "AQI_t+3"
        ]
    ].tail(5).to_string(index=False)
)