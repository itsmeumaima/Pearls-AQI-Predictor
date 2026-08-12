import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/karachi_aqi_features.csv"

MODEL_PATH = "models/random_forest_model.pkl"

OUTPUT_DIR = "results"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "latest_aqi_prediction.csv"
)


# ============================================================
# TARGET COLUMNS
# ============================================================

TARGETS = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]


# ============================================================
# COLUMNS NOT USED AS FEATURES
# ============================================================

DROP_COLUMNS = [
    "date",
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("\n========================================")
print("LOADING TRAINED MODEL")
print("========================================")


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Please run model_training.py first."
    )


model = joblib.load(MODEL_PATH)

print(
    "Random Forest model loaded successfully."
)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("\n========================================")
print("LOADING PROCESSED DATA")
print("========================================")


if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Processed data not found: {DATA_PATH}\n"
        "Please run preprocess_daily_data.py first."
    )


df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


# ============================================================
# 3. HANDLE INF VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# IMPORTANT:
#
# DO NOT DO:
#
# df = df.dropna()
#
# because the latest rows have NaN future targets.
#
# We only need the latest row with a valid AQI.
# ============================================================


# ============================================================
# 4. FIND LATEST OBSERVED DATA
# ============================================================

print("\n========================================")
print("LATEST AVAILABLE DATA")
print("========================================")


# Only require the CURRENT AQI to exist.
# Future AQI targets are allowed to be NaN.

available_rows = df[
    df["AQI"].notna()
].copy()


if available_rows.empty:

    raise ValueError(
        "No valid AQI observations found."
    )


latest_row = (
    available_rows
    .sort_values("date")
    .iloc[-1]
)


latest_date = latest_row["date"]

latest_aqi = latest_row["AQI"]


print(
    f"Date : {latest_date.date()}"
)

print(
    f"AQI  : {latest_aqi:.2f}"
)


# ============================================================
# 5. PREPARE FEATURES
# ============================================================

print("\nPreparing features...")


# Remove date and future targets
X_all = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)


# Get the same row as latest_row
latest_index = latest_row.name

X_latest = X_all.loc[
    [latest_index]
].copy()


# ============================================================
# 6. CHECK MODEL FEATURES
# ============================================================

print("\nChecking feature compatibility...")


# Random Forest remembers feature names if trained
# using a pandas DataFrame.

if hasattr(
    model,
    "feature_names_in_"
):

    expected_features = list(
        model.feature_names_in_
    )

    print(
        f"Expected features: "
        f"{len(expected_features)}"
    )

    print(
        f"Available features: "
        f"{X_latest.shape[1]}"
    )


    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X_latest.columns
    ]


    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:

            print(
                f" - {feature}"
            )

        raise ValueError(
            "Prediction data is missing "
            "features required by the model."
        )


    # --------------------------------------------------------
    # Keep EXACT SAME feature order
    # --------------------------------------------------------

    X_latest = X_latest[
        expected_features
    ]


else:

    # Fallback if feature names were not stored

    expected_features = (
        model.n_features_in_
    )

    actual_features = (
        X_latest.shape[1]
    )

    print(
        f"Expected features: "
        f"{expected_features}"
    )

    print(
        f"Available features: "
        f"{actual_features}"
    )

    if (
        expected_features
        != actual_features
    ):

        raise ValueError(
            "Feature count mismatch."
        )


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

if X_latest.isna().any().any():

    nan_features = (
        X_latest.columns[
            X_latest.isna().any()
        ]
        .tolist()
    )

    print(
        "\nERROR: Missing values found "
        "in latest row:"
    )

    for feature in nan_features:

        print(
            f" - {feature}"
        )

    raise ValueError(
        "Latest row contains missing "
        "feature values."
    )


# ============================================================
# 8. GENERATE PREDICTION
# ============================================================

print("\n========================================")
print("GENERATING AQI FORECAST")
print("========================================")


prediction = model.predict(
    X_latest
)


prediction = prediction[0]


# ============================================================
# 9. GET THREE PREDICTIONS
# ============================================================

aqi_day_1 = float(
    prediction[0]
)

aqi_day_2 = float(
    prediction[1]
)

aqi_day_3 = float(
    prediction[2]
)


# Prevent negative AQI
aqi_day_1 = max(
    0,
    aqi_day_1
)

aqi_day_2 = max(
    0,
    aqi_day_2
)

aqi_day_3 = max(
    0,
    aqi_day_3
)


# ============================================================
# 10. CREATE FUTURE DATES
# ============================================================

future_dates = [

    latest_date
    + pd.Timedelta(days=1),

    latest_date
    + pd.Timedelta(days=2),

    latest_date
    + pd.Timedelta(days=3)

]


# ============================================================
# 11. CREATE RESULTS
# ============================================================

results = pd.DataFrame({

    "forecast_date":
        future_dates,

    "horizon": [

        "AQI_t+1",

        "AQI_t+2",

        "AQI_t+3"

    ],

    "predicted_AQI": [

        aqi_day_1,

        aqi_day_2,

        aqi_day_3

    ]

})


# ============================================================
# 12. ROUND
# ============================================================

results["predicted_AQI"] = (

    results["predicted_AQI"]

    .round(2)

)


# ============================================================
# 13. AQI CATEGORY
# ============================================================

def aqi_category(aqi):

    if aqi <= 50:

        return "Good"

    elif aqi <= 100:

        return "Moderate"

    elif aqi <= 150:

        return (
            "Unhealthy for Sensitive Groups"
        )

    elif aqi <= 200:

        return "Unhealthy"

    elif aqi <= 300:

        return "Very Unhealthy"

    else:

        return "Hazardous"


results["category"] = (

    results["predicted_AQI"]

    .apply(aqi_category)

)


# ============================================================
# 14. FORMAT DATE
# ============================================================

results["forecast_date"] = (

    results["forecast_date"]

    .dt.strftime("%Y-%m-%d")

)


# ============================================================
# 15. SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 16. DISPLAY FORECAST
# ============================================================

print("\n========================================")
print("3-DAY AQI FORECAST")
print("========================================")


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("PREDICTION COMPLETE")
print("========================================")


print(
    f"Latest observed date : "
    f"{latest_date.date()}"
)

print(
    f"Latest observed AQI  : "
    f"{latest_aqi:.2f}"
)


print()


for _, row in results.iterrows():

    print(
        f"{row['forecast_date']} "
        f"→ AQI {row['predicted_AQI']:.2f} "
        f"({row['category']})"
    )


print()

print(
    "Prediction saved to:"
)

print(
    OUTPUT_PATH
)