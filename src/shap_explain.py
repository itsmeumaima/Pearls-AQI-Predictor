import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/karachi_aqi_features.csv"
MODEL_PATH = "models/xgboost_model.pkl"

OUTPUT_DIR = "outputs/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# VISUAL THEME
# ============================================================
# Same palette as the Streamlit dashboard, so the static SHAP
# exports feel like part of one product rather than a bolted-on
# library default.

INK_NAVY = "#14213D"
SOOT_GREY = "#5B6472"
HAZE_AMBER = "#E8A33D"
SKY_TEAL = "#2E7D8C"
FOG_BG = "#F3EFE7"
BORDER = "#E4DFD3"

plt.rcParams.update({
    "figure.facecolor": FOG_BG,
    "axes.facecolor": "white",
    "axes.edgecolor": BORDER,
    "axes.labelcolor": INK_NAVY,
    "axes.titlecolor": INK_NAVY,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.grid": True,
    "grid.color": BORDER,
    "grid.linewidth": 0.6,
    "xtick.color": SOOT_GREY,
    "ytick.color": SOOT_GREY,
    "text.color": INK_NAVY,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
})

def style_bar_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)


print("\nLoading processed data...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)



target_columns = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

X = df.drop(columns=["date"] + target_columns)
print("Feature matrix shape:", X.shape)



print("\nLoading XGBoost model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")
print("Model type:", type(model))

print("\nChecking feature compatibility...")
print("Model features:", model.n_features_in_)
print("Input features:", X.shape[1])

if model.n_features_in_ != X.shape[1]:
    raise ValueError(
        f"Feature mismatch! "
        f"Model expects {model.n_features_in_} features, "
        f"but dataset contains {X.shape[1]} features."
    )

print("Feature count matches!")


print("\nExtracting individual XGBoost models...")
estimators = model.estimators_
print("Number of target models:", len(estimators))

target_names = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

all_importance = []
shap_values_by_target = {}

for i, estimator in enumerate(estimators):

    target_name = target_names[i]

    print("\n========================================")
    print(f"SHAP ANALYSIS: {target_name}")
    print("========================================")

    print("Creating TreeExplainer...")
    explainer = shap.TreeExplainer(estimator)

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X)
    shap_values_by_target[target_name] = shap_values
    print("SHAP values calculated!")


    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    importance["target"] = target_name
    all_importance.append(importance)

    print("Creating SHAP summary bar plot...")

    top15 = importance.head(15).sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = np.linspace(0, 1, len(top15))
    bar_colors = [
        tuple(
            np.array([46, 125, 140]) / 255 * (1 - c) + np.array([232, 163, 61]) / 255 * c
        )
        for c in colors
    ]
    ax.barh(top15["feature"], top15["mean_abs_shap"], color=bar_colors, height=0.65)
    ax.set_title(f"Top Features Influencing {target_name}", loc="left", pad=14)
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    style_bar_axes(ax)
    plt.tight_layout()

    bar_path = os.path.join(OUTPUT_DIR, f"shap_bar_{target_name}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight", facecolor=FOG_BG)
    plt.close()
    print("Saved:", bar_path)

    print("Creating SHAP beeswarm plot...")
    plt.figure(figsize=(9, 6.5))
    try:
        shap.summary_plot(
            shap_values,
            X,
            max_display=15,
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        fig.set_facecolor(FOG_BG)
        for ax in fig.axes:
            ax.set_facecolor("white")
        plt.title(f"Feature Impact Distribution — {target_name}", loc="left")
        plt.tight_layout()
        beeswarm_path = os.path.join(OUTPUT_DIR, f"shap_summary_{target_name}.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight", facecolor=FOG_BG)
        print("Saved:", beeswarm_path)
    finally:
        plt.close()

print("\nCombining feature importance results...")
importance_all = pd.concat(all_importance, ignore_index=True)


overall_importance = (
    importance_all
    .groupby("feature")["mean_abs_shap"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
overall_importance.columns = ["feature", "mean_abs_shap"]


overall_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
overall_importance.to_csv(overall_path, index=False)
print("\nOverall feature importance saved to:", overall_path)



print("\nCreating cross-target comparison plot...")

top_features = overall_importance.head(10)["feature"].tolist()
pivot = (
    importance_all[importance_all["feature"].isin(top_features)]
    .pivot(index="feature", columns="target", values="mean_abs_shap")
    .reindex(top_features)
    .reindex(columns=target_names)
)

fig, ax = plt.subplots(figsize=(10, 6.5))
bar_width = 0.25
y_pos = np.arange(len(pivot))
target_colors = {target_names[0]: SKY_TEAL, target_names[1]: HAZE_AMBER, target_names[2]: INK_NAVY}

for i, target in enumerate(target_names):
    ax.barh(
        y_pos + (i - 1) * bar_width,
        pivot[target],
        height=bar_width,
        label=target,
        color=target_colors[target],
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(pivot.index)
ax.invert_yaxis()
ax.set_title("Top Drivers Across the 3-Day Forecast Horizon", loc="left", pad=14)
ax.set_xlabel("Mean |SHAP value|")
ax.legend(frameon=False, loc="lower right")
style_bar_axes(ax)
plt.tight_layout()

comparison_path = os.path.join(OUTPUT_DIR, "feature_importance_comparison.png")
plt.savefig(comparison_path, dpi=300, bbox_inches="tight", facecolor=FOG_BG)
plt.close()
print("Saved:", comparison_path)

print("\n========================================")
print("TOP AQI INFLUENCING FEATURES")
print("========================================")
print(overall_importance.head(15).to_string(index=False))


print("\n========================================")
print("SHAP ANALYSIS COMPLETE")
print("========================================")
print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print(" - shap_bar_<target>.png            (styled importance bar)")
print(" - shap_summary_<target>.png        (beeswarm distribution)")
print(" - feature_importance_comparison.png (cross-horizon comparison)")
print(" - feature_importance.csv           (overall ranking, used by dashboard)")