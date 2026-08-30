import requests
import pandas as pd
import os
import time
from datetime import date, timedelta
from tqdm import tqdm

LAT = 24.8607
LON = 67.0011
TIMEZONE = "Asia/Karachi"

RAW_PATH = "data/raw/karachi_daily_aqi_weather.csv"

REQUEST_DELAY = 0.2


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

        air_response = requests.get(
            air_url,
            timeout=30
        )

        if air_response.status_code != 200:

            print(
                f"Air API failed for {day}: "
                f"{air_response.status_code}"
            )

            return None

        air_json = air_response.json()

        if "hourly" not in air_json:

            print(
                f"No air-quality data for {day}"
            )

            return None

        df_air = pd.DataFrame(
            air_json["hourly"]
        )


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

            print(
                f"Weather API failed for {day}: "
                f"{weather_response.status_code}"
            )

            return None

        weather_json = weather_response.json()

        if "hourly" not in weather_json:

            print(
                f"No weather data for {day}"
            )

            return None

        df_weather = pd.DataFrame(
            weather_json["hourly"]
        )

        df = pd.merge(
            df_air,
            df_weather,
            on="time",
            how="inner"
        )

        if df.empty:

            print(
                f"No merged data available for {day}"
            )

            return None

        df["time"] = pd.to_datetime(
            df["time"]
        )

        return df


    except requests.exceptions.RequestException as e:

        print(
            f"Request error for {day}: {e}"
        )

        return None


    except Exception as e:

        print(
            f"Error processing {day}: {e}"
        )

        return None


def process_daily(day, df):

    try:

        df = df.set_index("time")

        # Daily average
        daily = df.resample("D").mean(
            numeric_only=True
        )

        daily["date"] = pd.to_datetime(day)


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

        # Make sure all expected columns exist
        for column in columns:

            if column not in daily.columns:

                daily[column] = None

        daily = daily[columns]

        return daily.reset_index(
            drop=True
        )


    except Exception as e:

        print(
            f"Daily processing error for {day}: {e}"
        )

        return None

def load_existing_data():

    if not os.path.exists(RAW_PATH):

        print(
            "\nExisting raw dataset not found."
        )

        print(
            "Please run data_collection.py first."
        )

        return None

    print(
        "\nLoading existing dataset..."
    )

    df = pd.read_csv(
        RAW_PATH
    )

    if "date" not in df.columns:

        raise ValueError(
            "Raw dataset does not contain 'date' column."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df



def update_dataset():

    print("\n========================================")
    print("KARACHI AQI DAILY DATA UPDATE")
    print("========================================")


    df_existing = load_existing_data()

    if df_existing is None:

        return


    latest_date = (
        df_existing["date"]
        .max()
        .date()
    )

    today = date.today()

    print(
        f"\nLatest stored date : {latest_date}"
    )

    print(
        f"Today's date       : {today}"
    )



    if latest_date >= today:

        print(
            "\nDataset is already up to date."
        )

        print(
            f"Latest available date: {latest_date}"
        )

        return


    start_date = latest_date + timedelta(
        days=1
    )

    end_date = today

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    print(
        f"\nNew dates to download: {len(dates)}"
    )

    print(
        f"Update period: "
        f"{start_date} → {end_date}"
    )


    new_days = []

    for single_date in tqdm(
        dates,
        desc="Downloading new data"
    ):

        day = single_date.date().isoformat()

        raw_data = fetch_day_data(
            day
        )

        if (
            raw_data is None
            or raw_data.empty
        ):

            continue


        daily_data = process_daily(
            day,
            raw_data
        )

        if (
            daily_data is not None
            and not daily_data.empty
        ):

            new_days.append(
                daily_data
            )


        time.sleep(
            REQUEST_DELAY
        )

    if not new_days:

        print(
            "\nNo new data was collected."
        )

        return



    df_new = pd.concat(
        new_days,
        ignore_index=True
    )

    df_new["date"] = pd.to_datetime(
        df_new["date"]
    )



    df_updated = pd.concat(
        [
            df_existing,
            df_new
        ],
        ignore_index=True
    )


    df_updated = (
        df_updated
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    df_updated["Next_Day_AQI"] = (
        df_updated["AQI"].shift(-1)
    )

    os.makedirs(
        os.path.dirname(RAW_PATH),
        exist_ok=True
    )

    df_updated.to_csv(
        RAW_PATH,
        index=False
    )

    print(
        "\n========================================"
    )

    print(
        "DATA UPDATE COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Old rows       : {len(df_existing)}"
    )

    print(
        f"New rows       : {len(df_new)}"
    )

    print(
        f"Updated rows   : {len(df_updated)}"
    )

    print(
        f"Latest date    : "
        f"{df_updated['date'].max().date()}"
    )

    print(
        f"Saved to       : {RAW_PATH}"
    )

    print(
        "\nLatest records:"
    )

    print(
        df_updated.tail().to_string(
            index=False
        )
    )

if __name__ == "__main__":

    update_dataset()