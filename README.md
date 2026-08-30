# Pearls AQI Predictor

An end-to-end **Air Quality Index (AQI) forecasting system for Karachi** that predicts AQI for the next 3 days using automated data collection, feature engineering, **Hopsworks Feature Store**, XGBoost, GitHub Actions, and a Streamlit dashboard.

---
Project Link: https://pearls-aqi-predictor1.streamlit.app/


## Project Overview

**Pearls AQI Predictor** is an automated machine learning system designed to forecast short-term air quality conditions in Karachi.

The system collects AQI, pollutant, and weather data, performs feature engineering, stores the engineered features in **Hopsworks Feature Store**, trains an XGBoost forecasting model, stores the trained model in a model registry, generates a 3-day AQI forecast, and displays the results through an interactive Streamlit dashboard.


# Key Features

## 1. Automated Data Collection

The system collects environmental data from external APIs such as:

* OpenWeather
* Open-Meteo

The collected data includes:

* AQI
* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃
* Temperature
* Humidity
* Precipitation

---

## 2. Feature Engineering

Raw environmental data is transformed into machine-learning features.

### Time Features

* Day
* Month
* Day of week
* Day of year
* Weekend indicator
* Cyclical month features
* Cyclical day-of-week features
* Cyclical day-of-year features

### Lag Features

Historical observations are used to capture temporal dependencies:

```text
1-day lag
2-day lag
3-day lag
7-day lag
14-day lag
21-day lag
30-day lag
```

### Rolling Features

Rolling statistics are calculated using:

```text
3-day window
7-day window
14-day window
30-day window
```

For AQI and pollutants:

* Rolling mean
* Rolling standard deviation
* Rolling minimum
* Rolling maximum

### Trend Features

The system calculates:

* AQI change
* Pollutant change
* Percentage change
* AQI momentum

### Ratio Features

Examples include:

```text
PM2.5 / PM10
NO₂ / O₃
CO / NO₂
```

### Weather Interaction Features

Examples include:

```text
Temperature × Humidity
PM2.5 × Humidity
PM10 × Humidity
```

---

# Hopsworks Feature Store

**Hopsworks** is used as the central Feature Store for the AQI prediction system.

Instead of relying only on local CSV files, the engineered features can be stored and retrieved through Hopsworks.

---

# Machine Learning

The primary forecasting model is **XGBoost**.

The model performs multi-output forecasting for:

```text
AQI_t+1
AQI_t+2
AQI_t+3
```

For example, if the latest observed date is:

```text
30 August 2026
```

the model predicts:

```text
31 August 2026
1 September 2026
2 September 2026
```

---

# Model Training

The training pipeline retrieves historical features from Hopsworks:

The model can be evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

Other models can also be experimented with:

* Random Forest
* Ridge Regression
* TensorFlow models
* PyTorch models

---

# Model Registry

The trained model can be stored and versioned using the Hopsworks Model Registry.

---

# 🔮 Prediction Pipeline

The prediction pipeline retrieves the latest feature data and registered model.

---

#  Automated CI/CD Pipeline

GitHub Actions is used to automate the AQI machine learning pipeline.

The project contains two primary workflows.

---

## Feature Pipeline

The feature pipeline runs automatically every hour.

---

#  Technology Stack

| Technology     | Purpose                        |
| -------------- | ------------------------------ |
| Python         | Core development               |
| Pandas         | Data processing                |
| NumPy          | Numerical computation          |
| Scikit-learn   | ML utilities and evaluation    |
| XGBoost        | AQI forecasting                |
| TensorFlow     | Deep learning experimentation  |
| SHAP           | Model explainability           |
| Hopsworks      | Feature Store & Model Registry |
| GitHub Actions | CI/CD automation               |
| Streamlit      | Web dashboard                  |
| Flask          | API layer                      |data                       |
| OpenWeather    | Weather/environmental data     |
| Open-Meteo     | Weather data                   |
| Git            | Version control                |

---
