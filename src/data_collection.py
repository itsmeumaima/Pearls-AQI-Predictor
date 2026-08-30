import requests
import pandas as pd
from datetime import date
from tqdm import tqdm
import os
import time


LAT = 24.8607
LON = 67.0011
TIMEZONE = "Asia/Karachi"

def fetch_day_data(day):

    try:

        air_url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={LAT}"
            f"&longitude={LON}"
            f"&start_date={day}"
            f"&end_date={day}"
            "&hourly=us_aqi,pm2_5,pm10,"
            "nitrogen_dioxide,sulphur_dioxide,"
            "carbon_monoxide,ozone"
            f"&timezone={TIMEZONE}"
        )

        air_response = requests.get(air_url, timeout=30)

        if air_response.status_code != 200:
            print(f"Air API failed for {day}")
            return None

        air_json = air_response.json()

        if "hourly" not in air_json:
            print(f"No air-quality data for {day}")
            return None

        air_data = air_json["hourly"]

        df_air = pd.DataFrame(air_data)

        weather_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}"
            f"&longitude={LON}"
            f"&start_date={day}"
            f"&end_date={day}"
            "&hourly=temperature_2m,"
            "relative_humidity_2m,precipitation"
            f"&timezone={TIMEZONE}"
        )

        weather_response = requests.get(
            weather_url,
            timeout=30
        )

        if weather_response.status_code != 200:
            print(f"Weather API failed for {day}")
            return None

        weather_json = weather_response.json()

        if "hourly" not in weather_json:
            print(f"No weather data for {day}")
            return None

        weather_data = weather_json["hourly"]

        df_weather = pd.DataFrame(weather_data)

        df = pd.merge(
            df_air,
            df_weather,
            on="time",
            how="inner"
        )

        df["time"] = pd.to_datetime(df["time"])

        return df

    except Exception as e:

        print(f"Error on {day}: {e}")

        return None

def process_daily(day, df):

    df = df.set_index("time")

    # Calculate daily average
    daily = df.resample("D").mean(numeric_only=True)

    daily["date"] = day

    # Rename columns
    daily = daily.rename(
        columns={
            "us_aqi": "AQI",
            "pm2_5": "PM2.5",
            "pm10": "PM10",
            "nitrogen_dioxide": "NO2",
            "sulphur_dioxide": "SO2",
            "carbon_monoxide": "CO",
            "ozone": "O3",
            "temperature_2m": "Temperature",
            "relative_humidity_2m": "Humidity",
            "precipitation": "Precipitation"
        }
    )

    columns = [
        "date",
        "AQI",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "Temperature",
        "Humidity",
        "Precipitation"
    ]

    return daily[columns]

def main():

    # Start date
    start = date(2023, 1, 1)

    # Today's date
    today = date.today()

    all_days = []

    dates = pd.date_range(
        start=start,
        end=today,
        freq="D"
    )

    print(f"Fetching {len(dates)} days of data...")

    for single_date in tqdm(
        dates,
        desc="Downloading"
    ):

        day = single_date.date().isoformat()

        raw_data = fetch_day_data(day)

        if raw_data is None or raw_data.empty:
            continue

        daily_data = process_daily(
            day,
            raw_data
        )

        if daily_data is not None:
            all_days.append(daily_data)

        # Small delay to avoid hammering API
        time.sleep(0.1)


    if not all_days:

        print("No data was collected.")

        return

    df = pd.concat(
        all_days,
        ignore_index=True
    )

    # Sort by date
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df["Next_Day_AQI"] = df["AQI"].shift(-1)


    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    output_path = (
        "data/raw/"
        "karachi_daily_aqi_weather.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print("====================================")
    print("DATA COLLECTION COMPLETE")
    print("====================================")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_path}")
    print()
    print(df.head())
    print()
    print(df.tail())


if __name__ == "__main__":
    main()