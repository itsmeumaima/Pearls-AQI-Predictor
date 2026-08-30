import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DATA_PATH = "data/processed/karachi_aqi_features.csv"

MODEL_PATH = "models/xgboost_model.pkl"

OUTPUT_DIR = "results"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "latest_aqi_prediction.csv"
)

TARGETS = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]

DROP_COLUMNS = [
    "date",
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]

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
    "XGBoost model loaded successfully."
)


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


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


print("\n========================================")
print("LATEST AVAILABLE DATA")
print("========================================")



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



print("\nChecking feature compatibility...")



if hasattr(model, "estimators_"):

    base_model = model.estimators_[0]

    if hasattr(
        base_model,
        "feature_names_in_"
    ):

        expected_features = list(
            base_model.feature_names_in_
        )

    else:

        expected_features = list(
            X_latest.columns
        )

elif hasattr(
    model,
    "feature_names_in_"
):

    expected_features = list(
        model.feature_names_in_
    )


else:

    expected_features = list(
        X_latest.columns
    )


print(
    f"Expected features: "
    f"{len(expected_features)}"
)

print(
    f"Available features: "
    f"{X_latest.shape[1]}"
)


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


X_latest = X_latest[
    expected_features
]

if X_latest.shape[1] != len(
    expected_features
):

    raise ValueError(
        "Feature count mismatch."
    )

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



print("\n========================================")
print("GENERATING AQI FORECAST")
print("========================================")


prediction = model.predict(
    X_latest
)


prediction = prediction[0]

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


future_dates = [

    latest_date
    + pd.Timedelta(days=1),

    latest_date
    + pd.Timedelta(days=2),

    latest_date
    + pd.Timedelta(days=3)

]


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

results["predicted_AQI"] = (

    results["predicted_AQI"]

    .round(2)

)

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
results["forecast_date"] = (

    results["forecast_date"]

    .dt.strftime("%Y-%m-%d")

)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


results.to_csv(
    OUTPUT_PATH,
    index=False
)
print("\n========================================")
print("3-DAY AQI FORECAST")
print("========================================")


print(
    results.to_string(
        index=False
    )
)


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