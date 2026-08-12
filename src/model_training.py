import os
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/karachi_aqi_features.csv"

MODEL_DIR = "models"
RESULT_DIR = "results"
PLOT_DIR = "results/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42


TARGETS = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING PROCESSED DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Dataset shape : {df.shape}")
print(
    f"Date range    : "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


# ============================================================
# 2. CHECK TARGETS
# ============================================================

print("\nChecking target columns...")

for target in TARGETS:

    if target not in df.columns:
        raise ValueError(
            f"Missing target column: {target}"
        )

print("All target columns found.")


# ============================================================
# 3. CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna().reset_index(drop=True)

print(f"Clean dataset shape: {df.shape}")


# ============================================================
# 4. CREATE X AND y
# ============================================================

# Date is not directly given to the model.
#
# Time information is already represented by:
#
# month
# day
# day_of_week
# day_of_year
# sin/cos features
# etc.

DROP_COLUMNS = [
    "date",
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]


X = df.drop(
    columns=DROP_COLUMNS
)

y = df[TARGETS]


print("\n" + "=" * 60)
print("FEATURE / TARGET INFORMATION")
print("=" * 60)

print(f"Number of features : {X.shape[1]}")
print(f"Number of samples  : {X.shape[0]}")

print("\nFeatures:")
print(X.columns.tolist())

print("\nTargets:")
print(TARGETS)


# ============================================================
# 5. REMOVE NON-NUMERIC FEATURES
# ============================================================

non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric:

    print("\nRemoving non-numeric columns:")

    for col in non_numeric:
        print(f" - {col}")

    X = X.drop(
        columns=non_numeric
    )


# ============================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 60)

split_index = int(
    len(df) * (1 - TEST_SIZE)
)


X_train = X.iloc[
    :split_index
].copy()

X_test = X.iloc[
    split_index:
].copy()


y_train = y.iloc[
    :split_index
].copy()

y_test = y.iloc[
    split_index:
].copy()


print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)

print(
    f"Training period  : "
    f"{df.iloc[:split_index]['date'].min().date()} "
    f"→ "
    f"{df.iloc[:split_index]['date'].max().date()}"
)

print(
    f"Testing period   : "
    f"{df.iloc[split_index:]['date'].min().date()} "
    f"→ "
    f"{df.iloc[split_index:]['date'].max().date()}"
)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

results = []

horizon_results = []


def evaluate_model(
    name,
    y_true,
    y_pred
):

    # Overall metrics

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
        multioutput="uniform_average"
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"MAE :  {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²  :  {r2:.4f}")

    # --------------------------------------------------------
    # Individual horizon metrics
    # --------------------------------------------------------

    for i, target in enumerate(TARGETS):

        horizon_mae = mean_absolute_error(
            y_true[:, i],
            y_pred[:, i]
        )

        horizon_rmse = np.sqrt(
            mean_squared_error(
                y_true[:, i],
                y_pred[:, i]
            )
        )

        horizon_r2 = r2_score(
            y_true[:, i],
            y_pred[:, i]
        )

        horizon_results.append({
            "Model": name,
            "Horizon": target,
            "MAE": horizon_mae,
            "RMSE": horizon_rmse,
            "R2": horizon_r2
        })

        print(
            f"{target:<12} "
            f"MAE={horizon_mae:.4f}  "
            f"RMSE={horizon_rmse:.4f}  "
            f"R²={horizon_r2:.4f}"
        )

    return mae, rmse, r2


# ============================================================
# 8. BASELINE
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1 — BASELINE")
print("=" * 60)

# Predict every future day using today's AQI.

baseline_pred = np.column_stack([
    X_test["AQI"].values,
    X_test["AQI"].values,
    X_test["AQI"].values
])

evaluate_model(
    "Baseline",
    y_test.values,
    baseline_pred
)


# ============================================================
# 9. STANDARDIZATION FOR RIDGE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()

# IMPORTANT:
# Fit ONLY on training data.

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "feature_scaler.pkl"
    )
)

print(
    "Scaler saved to "
    "models/feature_scaler.pkl"
)


# ============================================================
# 10. RIDGE REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2 — RIDGE REGRESSION")
print("=" * 60)

ridge = Ridge(
    alpha=1.0
)

ridge.fit(
    X_train_scaled,
    y_train
)

ridge_pred = ridge.predict(
    X_test_scaled
)

evaluate_model(
    "Ridge Regression",
    y_test.values,
    ridge_pred
)

joblib.dump(
    ridge,
    os.path.join(
        MODEL_DIR,
        "ridge_model.pkl"
    )
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("MODEL 3 — RANDOM FOREST")
print("=" * 60)

random_forest = RandomForestRegressor(
    n_estimators=500,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features="sqrt",

    bootstrap=True,

    random_state=RANDOM_STATE,

    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)

rf_pred = random_forest.predict(
    X_test
)

evaluate_model(
    "Random Forest",
    y_test.values,
    rf_pred
)

joblib.dump(
    random_forest,
    os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    )
)

print(
    "\nRandom Forest saved."
)


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    feature_importance.head(20).to_string(
        index=False
    )
)

feature_importance.to_csv(
    os.path.join(
        RESULT_DIR,
        "feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 13. XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("MODEL 4 — XGBOOST")
print("=" * 60)

xgb_base = XGBRegressor(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=5,

    min_child_weight=3,

    subsample=0.85,

    colsample_bytree=0.85,

    reg_alpha=0.05,

    reg_lambda=1.0,

    objective="reg:squarederror",

    random_state=RANDOM_STATE,

    n_jobs=-1
)

xgb_model = MultiOutputRegressor(
    xgb_base
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
)

evaluate_model(
    "XGBoost",
    y_test.values,
    xgb_pred
)

joblib.dump(
    xgb_model,
    os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    )
)


# ============================================================
# 14. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

horizon_df = pd.DataFrame(
    horizon_results
)


# ============================================================
# 15. SORT MODELS
# ============================================================

results_df = (
    results_df
    .sort_values(
        by="RMSE",
        ascending=True
    )
    .reset_index(drop=True)
)


# ============================================================
# 16. PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("HORIZON-WISE RESULTS")
print("=" * 70)

print(
    horizon_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

horizon_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "horizon_results.csv"
    ),
    index=False
)


# ============================================================
# 18. SELECT BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

best_model_name = best_model["Model"]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model : {best_model_name}"
)

print(
    f"MAE   : {best_model['MAE']:.4f}"
)

print(
    f"RMSE  : {best_model['RMSE']:.4f}"
)

print(
    f"R²    : {best_model['R2']:.4f}"
)


# ============================================================
# 19. SAVE BEST MODEL INFORMATION
# ============================================================

best_model_info = {
    "model": best_model_name,
    "MAE": float(best_model["MAE"]),
    "RMSE": float(best_model["RMSE"]),
    "R2": float(best_model["R2"])
}

with open(
    os.path.join(
        RESULT_DIR,
        "best_model.json"
    ),
    "w"
) as f:

    json.dump(
        best_model_info,
        f,
        indent=4
    )


# ============================================================
# 20. ACTUAL VS PREDICTED PLOTS
# ============================================================

print("\nCreating prediction plots...")


plot_predictions = {
    "Baseline": baseline_pred,
    "Ridge Regression": ridge_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred
}


test_dates = df.iloc[
    split_index:
]["date"].values


for model_name, predictions in plot_predictions.items():

    for i, target in enumerate(TARGETS):

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            test_dates,
            y_test.iloc[:, i].values,
            label="Actual"
        )

        plt.plot(
            test_dates,
            predictions[:, i],
            label="Predicted"
        )

        plt.title(
            f"{model_name} - {target}"
        )

        plt.xlabel(
            "Date"
        )

        plt.ylabel(
            "AQI"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        plot_path = os.path.join(
            PLOT_DIR,
            f"{safe_name}_{target}.png"
        )

        plt.savefig(
            plot_path,
            dpi=150
        )

        plt.close()


# ============================================================
# 21. FEATURE IMPORTANCE PLOT
# ============================================================

top_features = (
    feature_importance
    .head(20)
    .sort_values(
        "Importance"
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest - Top 20 Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "random_forest_feature_importance.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "date": test_dates,

    "Actual_AQI_t+1":
        y_test["AQI_t+1"].values,

    "Predicted_AQI_t+1":
        rf_pred[:, 0],

    "Actual_AQI_t+2":
        y_test["AQI_t+2"].values,

    "Predicted_AQI_t+2":
        rf_pred[:, 1],

    "Actual_AQI_t+3":
        y_test["AQI_t+3"].values,

    "Predicted_AQI_t+3":
        rf_pred[:, 2]
})

prediction_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "random_forest_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nSaved models:")

print(
    "✓ models/feature_scaler.pkl"
)

print(
    "✓ models/ridge_model.pkl"
)

print(
    "✓ models/random_forest_model.pkl"
)

print(
    "✓ models/xgboost_model.pkl"
)

print("\nSaved results:")

print(
    "✓ results/model_comparison.csv"
)

print(
    "✓ results/horizon_results.csv"
)

print(
    "✓ results/feature_importance.csv"
)

print(
    "✓ results/random_forest_test_predictions.csv"
)

print(
    "✓ results/best_model.json"
)

print("\nSaved plots:")

print(
    f"✓ {PLOT_DIR}/"
)

print("\nBest model:")

print(
    f"✓ {best_model_name}"
)

print(
    f"✓ R² = {best_model['R2']:.4f}"
)

print(
    f"✓ MAE = {best_model['MAE']:.4f}"
)

print(
    f"✓ RMSE = {best_model['RMSE']:.4f}"
)

print("\nNext step:")
print(
    "Use the saved Random Forest model "
    "for the 3-day prediction pipeline."
)